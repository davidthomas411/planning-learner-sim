import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.delivery3d import delivery_mode_3d
from dosim_sim.objective import PlanningPriorities
from dosim_sim.prostate_protocol import (
    PRESCRIPTION_GY,
    evaluate_prostate_60gy20fx,
    protocol_summary_rows,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d


STATUS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prostate DVH calibration status</title>
<style>
body{font-family:Arial,sans-serif;max-width:760px;margin:48px auto;padding:0 20px;color:#202124;background:#fff}
h1{font-size:24px;font-weight:500}.track{height:28px;background:#e8eaed;border-radius:4px;overflow:hidden}
.bar{height:100%;width:0;background:#1a73e8;transition:width .35s ease}.line{display:flex;justify-content:space-between;margin:10px 0}
.detail{color:#5f6368}.complete{background:#188038}.failed{background:#d93025}@media(prefers-color-scheme:dark){body{color:#e8eaed;background:#202124}.track{background:#3c4043}.detail{color:#bdc1c6}}
</style>
</head>
<body>
<h1>Prostate DVH calibration</h1>
<div class="line"><strong id="phase">Starting</strong><span id="percent">0.0%</span></div>
<div class="track" role="progressbar" aria-label="Calibration progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="bar" id="bar"></div></div>
<div class="line detail"><span id="count">0 / 0 plans</span><span id="eta">Estimating remaining time</span></div>
<p class="detail" id="case">Waiting for first plan.</p>
<p class="detail" id="updated"></p>
<script>
async function refresh(){try{const response=await fetch('progress.json?'+Date.now(),{cache:'no-store'});const p=await response.json();
const value=Math.max(0,Math.min(100,p.percent_complete||0));document.getElementById('bar').style.width=value+'%';
document.querySelector('.track').setAttribute('aria-valuenow',value.toFixed(1));document.getElementById('percent').textContent=value.toFixed(1)+'%';
document.getElementById('phase').textContent=p.status==='complete'?'Complete':p.status==='failed'?'Failed':'Running';
document.getElementById('count').textContent=p.completed+' / '+p.total+' plans';
document.getElementById('eta').textContent=p.status==='complete'?'Finished in '+format(p.elapsed_seconds):p.estimated_seconds_remaining==null?'Estimating remaining time':format(p.estimated_seconds_remaining)+' remaining';
document.getElementById('case').textContent=p.last_case?'Last completed: '+p.last_case+', DVH weight '+p.last_weight:'Waiting for first plan.';
document.getElementById('updated').textContent='Local update: '+new Date().toLocaleTimeString();
document.getElementById('bar').className='bar '+(p.status==='complete'?'complete':p.status==='failed'?'failed':'');}catch(error){document.getElementById('updated').textContent='Waiting for progress file...';}}
function format(seconds){seconds=Math.max(0,Math.round(seconds||0));const minutes=Math.floor(seconds/60);const remainder=seconds%60;return minutes?minutes+'m '+remainder+'s':remainder+'s';}
refresh();setInterval(refresh,2000);
</script>
</body>
</html>
"""


def write_progress(
    output_dir: Path,
    completed: int,
    total: int,
    started: float,
    status: str = "running",
    last_case: str | None = None,
    last_weight: float | None = None,
) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
    payload = {
        "status": status,
        "completed": completed,
        "total": total,
        "percent_complete": 100.0 * completed / max(total, 1),
        "elapsed_seconds": elapsed,
        "estimated_seconds_remaining": (total - completed) / rate if rate > 0 else None,
        "last_case": last_case,
        "last_weight": last_weight,
    }
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "progress.json")


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cumulative_dvh(values_gy: np.ndarray, bins_gy: np.ndarray) -> np.ndarray:
    return np.asarray([100.0 * np.mean(values_gy >= level) for level in bins_gy])


def save_pass_summary(rows: list[dict], weights: tuple[float, ...], path: Path) -> None:
    labels = [f"DVH weight {value:g}" for value in weights]
    per_protocol = [100.0 * np.mean([row["per_protocol"] for row in rows if row["clinical_dvh_weight"] == value]) for value in weights]
    variation = [100.0 * np.mean([row["variation_acceptable"] for row in rows if row["clinical_dvh_weight"] == value]) for value in weights]
    x = np.arange(len(weights))
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.bar(x - 0.18, per_protocol, 0.36, label="Per protocol")
    axis.bar(x + 0.18, variation, 0.36, label="Variation acceptable")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Plans meeting all represented goals (%)")
    axis.set_title("60 Gy in 20 fractions: represented target and OAR goals")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_representative_dvhs(case, plans: dict[float, object], path: Path) -> None:
    bins = np.linspace(0.0, 72.0, 217)
    colors = {"PTV": "#d62728", "bladder": "#1f77b4", "rectum": "#2ca02c", "femoral_heads": "#9467bd"}
    masks = {"PTV": case.target, **dict(zip(case.structure_names, case.oars, strict=True))}
    figure, axes = plt.subplots(1, len(plans), figsize=(4.8 * len(plans), 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (weight, plan) in zip(axes, plans.items(), strict=True):
        dose_gy = plan.dose.detach().float().cpu().numpy() * PRESCRIPTION_GY
        for name, mask in masks.items():
            axis.plot(bins, cumulative_dvh(dose_gy[mask], bins), color=colors[name], linewidth=1.8, label=name)
        evaluation = evaluate_prostate_60gy20fx(case, dose_gy / PRESCRIPTION_GY)
        status = "per protocol" if evaluation.per_protocol else "variation" if evaluation.variation_acceptable else "outside goals"
        axis.axvline(PRESCRIPTION_GY, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"DVH weight {weight:g}\n{status}")
        axis.set_xlabel("Dose (Gy)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Volume receiving at least dose (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    figure.suptitle(f"Representative seven-field plan: {case.case_id}, 60 Gy in 20 fractions", y=1.07)
    figure.subplots_adjust(left=0.06, right=0.995, bottom=0.13, top=0.78, wspace=0.06)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_constraint_review(rows: list[dict], selected_weight: float, path: Path) -> None:
    selected = [row for row in rows if row["clinical_dvh_weight"] == selected_weight]
    metrics = [key for key in selected[0] if key.startswith("goal_") and key.endswith("_ratio")]
    values = np.asarray([[row[key] for key in metrics] for row in selected], dtype=float)
    labels = [key.removeprefix("goal_").removesuffix("_ratio").replace("_", " ") for key in metrics]
    figure, axis = plt.subplots(figsize=(max(9, 0.75 * len(labels)), 5.2), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=max(1.5, float(values.max())), cmap="RdYlGn_r")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(selected)), [row["case_id"] for row in selected])
    axis.set_title(f"Constraint ratio at clinical DVH weight {selected_weight:g}; values at or below 1 pass")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(column_index, row_index, f"{values[row_index, column_index]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="Observed value / per-protocol limit")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate protocol-inspired prostate DVH objectives")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases-per-stratum", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--weights", nargs="+", type=float, default=(0.0, 0.5, 1.0, 2.0))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_clinical_dvh_pilot"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "status.html").write_text(STATUS_PAGE, encoding="utf-8")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = [row for row in load_records(args.dataset_dir / "trajectory_view.jsonl") if row["split"] == "validation"]
    selected = [
        row
        for difficulty in ("easy", "moderate", "hard")
        for row in [item for item in records if item["difficulty"] == difficulty][: args.cases_per_stratum]
    ]
    mode = delivery_mode_3d("static_7")
    rows: list[dict] = []
    representative_case = None
    representative_plans: dict[float, object] = {}
    total = len(selected) * len(args.weights)
    completed = 0
    started = time.perf_counter()
    write_progress(args.output_dir, completed, total, started)
    for record in selected:
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty=record["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, mode.angles_degrees, args.fluence_size, device=device, dtype=torch.float32)
        for weight in args.weights:
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                mode.active_beams,
                PlanningPriorities.for_case(case),
                iterations=args.iterations,
                normal_tissue_weight=50.0,
                normal_tissue_threshold=0.5,
                integral_dose_weight=2.0,
                clinical_dvh_weight=weight,
            )
            dose = plan.dose.detach().float().cpu().numpy()
            evaluation = evaluate_prostate_60gy20fx(case, dose)
            protocol_rows = protocol_summary_rows(evaluation)
            row = {
                "case_id": case.case_id,
                "seed": case.seed,
                "difficulty": case.difficulty,
                "clinical_dvh_weight": weight,
                "per_protocol": evaluation.per_protocol,
                "variation_acceptable": evaluation.variation_acceptable,
                "target_d98_gy": evaluation.target_d98_gy,
                "target_d99_gy": evaluation.target_d99_gy,
                "target_d02_gy": evaluation.target_d02_gy,
                "paddick_ci_95": plan.metrics.paddick_ci_95,
                "r50": plan.metrics.r50,
            }
            for item in protocol_rows:
                if item["structure"] == "PTV":
                    continue
                key = f"{item['structure']}_{item['metric']}".lower().replace(".", "p")
                limit = float(str(item["per_protocol_goal"]).removeprefix("<="))
                row[f"goal_{key}_observed"] = item["observed"]
                row[f"goal_{key}_ratio"] = float(item["observed"]) / limit
            rows.append(row)
            if case.difficulty == "moderate" and representative_case is None:
                representative_plans[weight] = plan
            completed += 1
            write_progress(
                args.output_dir,
                completed,
                total,
                started,
                last_case=case.case_id,
                last_weight=weight,
            )
            print(f"[{completed:02d}/{total}] {case.case_id} weight={weight:g}", flush=True)
        if case.difficulty == "moderate" and representative_case is None:
            representative_case = case

    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    weights = tuple(float(value) for value in args.weights)
    save_pass_summary(rows, weights, args.output_dir / "01_protocol_pass_summary.png")
    if representative_case is not None:
        save_representative_dvhs(representative_case, representative_plans, args.output_dir / "02_representative_protocol_dvhs.png")
    selected_weight = weights[int(np.argmax([
        np.mean([row["variation_acceptable"] for row in rows if row["clinical_dvh_weight"] == value])
        for value in weights
    ]))]
    save_constraint_review(rows, selected_weight, args.output_dir / "03_constraint_review.png")
    summary = {
        "status": "protocol-inspired DVH objective calibration; not clinical dose validation",
        "prescription_gy": PRESCRIPTION_GY,
        "fractions": 20,
        "dose_per_fraction_gy": 3.0,
        "cases": len(selected),
        "weights": {
            str(value): {
                "per_protocol_rate": float(np.mean([row["per_protocol"] for row in rows if row["clinical_dvh_weight"] == value])),
                "variation_acceptable_rate": float(np.mean([row["variation_acceptable"] for row in rows if row["clinical_dvh_weight"] == value])),
                "median_target_d98_gy": float(np.median([row["target_d98_gy"] for row in rows if row["clinical_dvh_weight"] == value])),
                "median_paddick_ci_95": float(np.median([row["paddick_ci_95"] for row in rows if row["clinical_dvh_weight"] == value])),
                "median_r50": float(np.median([row["r50"] for row in rows if row["clinical_dvh_weight"] == value])),
            }
            for value in weights
        },
        "selected_weight_for_review": selected_weight,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_progress(
        args.output_dir,
        completed,
        total,
        started,
        status="complete",
        last_case=rows[-1]["case_id"],
        last_weight=float(rows[-1]["clinical_dvh_weight"]),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
