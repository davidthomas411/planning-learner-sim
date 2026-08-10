import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dosim_sim.planning3d import HighLevelSearchConfig3D, run_reference_optimizer_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
from dosim_sim.volume3d import generate_prostate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def noncoverage_rules_pass(metrics, case, config) -> bool:
    return (
        metrics.target_d02 <= config.d02_max
        and all(value <= limit for value, limit in zip(metrics.oar_mean, case.oar_limits, strict=True))
        and metrics.paddick_ci_95 >= config.paddick_ci_95_min
        and metrics.r50 <= config.r50_max
        and metrics.field_count >= config.minimum_field_count
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a separate reduced-coverage prostate goal tier")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/prostate300_local/merged"))
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--standard-d95", type=float, default=0.94)
    parser.add_argument("--compromise-d95", type=float, default=0.90)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prostate_coverage_tier_pilot"))
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        row
        for row in load_records(args.dataset_dir / "trajectory_view.jsonl")
        if row.get("split") == "validation" and row["difficulty"] == "hard"
    ][: args.cases]
    config = HighLevelSearchConfig3D(
        d95_min=args.standard_d95,
        d02_max=1.22,
        normal_tissue_weight=50.0,
        normal_tissue_threshold=0.5,
        integral_dose_weight=2.0,
        paddick_ci_95_min=0.40,
        r50_max=15.0,
        minimum_field_count=7,
        priority_ceiling=25.0,
    )
    device = torch.device(args.device)
    rows = []
    for index, record in enumerate(records, start=1):
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty="hard")
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float32)
        plan = run_reference_optimizer_3d(case, engine, iterations=args.iterations, config=config)
        other_rules = noncoverage_rules_pass(plan.metrics, case, config)
        tier = (
            "standard"
            if other_rules and plan.metrics.target_d95 >= args.standard_d95
            else "compromise"
            if other_rules and plan.metrics.target_d95 >= args.compromise_d95
            else "unreached"
        )
        rows.append({
            "case_id": case.case_id,
            "tier": tier,
            "target_d95": plan.metrics.target_d95,
            "target_d02": plan.metrics.target_d02,
            "paddick_ci_95": plan.metrics.paddick_ci_95,
            "r50": plan.metrics.r50,
            "maximum_oar_ratio": max(value / limit for value, limit in zip(plan.metrics.oar_mean, case.oar_limits, strict=True)),
            "noncoverage_rules_pass": other_rules,
        })
        print(f"[{index}/{len(records)}] {case.case_id}: {tier}", flush=True)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {tier: sum(row["tier"] == tier for row in rows) for tier in ("standard", "compromise", "unreached")}
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar(list(counts), list(counts.values()), color=("#4E79A7", "#F28E2B", "#9C9C9C"))
    axis.bar_label(bars)
    axis.set_ylabel("Hard cases")
    axis.set_title("Coverage tiers after a standard-goal reference attempt")
    figure.savefig(args.output_dir / "01_coverage_tiers.png", dpi=180)
    plt.close(figure)
    summary = {
        "cases": len(rows),
        "standard_d95": args.standard_d95,
        "compromise_d95": args.compromise_d95,
        "counts": counts,
        "interpretation": "The compromise tier is separate from the primary standard-coverage comparison.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
