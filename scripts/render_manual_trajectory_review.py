import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.planning3d import HighLevelSearchConfig3D, run_high_level_search_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D
from dosim_sim.volume3d import generate_prostate_case_3d


ANGLES = tuple(float(value) for value in range(0, 360, 30))
STRUCTURES = (
    ("PTV", "#d62728"),
    ("bladder", "#1f77b4"),
    ("rectum", "#2ca02c"),
    ("femoral heads", "#9467bd"),
)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conformity(case, dose: np.ndarray) -> tuple[float, float]:
    target_volume = float(case.target.sum())
    covered = float(((dose >= 0.95) & case.target).sum())
    prescription_volume = float(((dose >= 0.95) & case.body).sum())
    half_volume = float(((dose >= 0.50) & case.body).sum())
    return (
        covered**2 / max(target_volume * prescription_volume, 1.0),
        half_volume / target_volume,
    )


def dvh(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = dose[mask]
    return np.asarray([100.0 * np.mean(values >= level) for level in bins])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render initial and final manual prostate plans")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--deep-iterations", type=int, default=120)
    parser.add_argument("--initial-field-count", type=int, default=7)
    parser.add_argument("--normal-tissue-weight", type=float, default=50.0)
    parser.add_argument("--normal-tissue-threshold", type=float, default=0.5)
    parser.add_argument("--integral-dose-weight", type=float, default=2.0)
    parser.add_argument("--d95-min", type=float, default=0.94)
    parser.add_argument("--d02-max", type=float, default=1.22)
    parser.add_argument("--paddick-ci-95-min", type=float, default=0.40)
    parser.add_argument("--r50-max", type=float, default=15.0)
    parser.add_argument("--minimum-field-count", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--deep-max-steps", type=int, default=10)
    parser.add_argument("--deep-beam-width", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    attempt_by_case = {
        row["case_id"]: row for row in load_records(args.dataset_dir / "attempt_manifest.jsonl")
    }
    selected = [next(row for row in records if row["difficulty"] == level) for level in ("easy", "moderate", "hard")]
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    review = []
    for row_index, record in enumerate(selected):
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty=record["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float32)
        search_tier = attempt_by_case[case.case_id]["search_tier"]
        config = HighLevelSearchConfig3D(
            max_steps=args.deep_max_steps if search_tier == "deep" else args.max_steps,
            beam_width=args.deep_beam_width if search_tier == "deep" else args.beam_width,
            add_candidates=3,
            remove_candidates=2,
            optimizer_iterations=args.deep_iterations if search_tier == "deep" else args.iterations,
            priority_ceiling=25.0 if search_tier == "deep" else 6.0,
            initial_field_count=args.initial_field_count,
            normal_tissue_weight=args.normal_tissue_weight,
            normal_tissue_threshold=args.normal_tissue_threshold,
            integral_dose_weight=args.integral_dose_weight,
            d95_min=args.d95_min,
            d02_max=args.d02_max,
            paddick_ci_95_min=args.paddick_ci_95_min,
            r50_max=args.r50_max,
            minimum_field_count=args.minimum_field_count,
        )
        trajectory = run_high_level_search_3d(case, engine, config)
        initial = trajectory.steps[0].plan
        final = trajectory.final.plan
        initial_dose = initial.dose.detach().float().cpu().numpy()
        final_dose = final.dose.detach().float().cpu().numpy()
        maximum = max(float(initial_dose.max()), float(final_dose.max()), 1.2)
        axial = int(np.argmax(case.target.sum(axis=(0, 1))))
        for column, (label, dose) in enumerate((("Initial", initial_dose), ("Final", final_dose))):
            axis = axes[row_index, column]
            image = axis.imshow(dose[:, :, axial].T, origin="lower", cmap="turbo", vmin=0.0, vmax=maximum)
            axis.contour(case.target[:, :, axial].T, levels=[0.5], colors="white", linewidths=1.4)
            for color, mask in zip(("cyan", "lime", "magenta"), case.oars, strict=True):
                axis.contour(mask[:, :, axial].T, levels=[0.5], colors=color, linewidths=0.8)
            ci, r50 = conformity(case, dose)
            axis.set_title(
                f"{record['difficulty'].title()} | {label} | {len((initial if label == 'Initial' else final).active_beams)} fields\n"
                f"CI95 {ci:.2f}; R50 {r50:.1f}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        bins = np.linspace(0.0, maximum, 161)
        masks = (case.target, *case.oars)
        axis = axes[row_index, 2]
        for (name, color), mask in zip(STRUCTURES, masks, strict=True):
            axis.plot(bins, dvh(initial_dose, mask, bins), color=color, linestyle="--", alpha=0.65)
            axis.plot(bins, dvh(final_dose, mask, bins), color=color, linewidth=1.8, label=name)
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_xlim(0.0, maximum)
        axis.set_ylim(0.0, 101.0)
        axis.set_xlabel("Relative dose (dashed initial; solid final)")
        axis.set_ylabel("Volume receiving at least dose (%)")
        axis.grid(alpha=0.2)
        actions = [step.action.description for step in trajectory.steps[1:] if step.action]
        action_text = "Manual actions: " + ("; ".join(actions) if actions else "none")
        axis.set_title(textwrap.fill(action_text, width=58), fontsize=8)
        initial_ci, initial_r50 = conformity(case, initial_dose)
        final_ci, final_r50 = conformity(case, final_dose)
        review.append({
            "case_id": case.case_id,
            "difficulty": case.difficulty,
            "actions": " | ".join(actions),
            "initial_beam_angles": "|".join(str(beam * 30) for beam in initial.active_beams),
            "final_beam_angles": "|".join(str(beam * 30) for beam in final.active_beams),
            "initial_d95": initial.metrics.target_d95,
            "final_d95": final.metrics.target_d95,
            "initial_d02": initial.metrics.target_d02,
            "final_d02": final.metrics.target_d02,
            "initial_maximum_oar_ratio": max(
                value / limit for value, limit in zip(initial.metrics.oar_mean, case.oar_limits, strict=True)
            ),
            "final_maximum_oar_ratio": max(
                value / limit for value, limit in zip(final.metrics.oar_mean, case.oar_limits, strict=True)
            ),
            "initial_ci95": initial_ci,
            "final_ci95": final_ci,
            "initial_r50": initial_r50,
            "final_r50": final_r50,
            "initial_violation": trajectory.steps[0].violation_score,
            "final_violation": trajectory.final.violation_score,
        })
    handles, labels = axes[0, 2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=4, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
