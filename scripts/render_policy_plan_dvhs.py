import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.dataset3d import state_features_3d
from dosim_sim.planning3d import HighLevelSearchConfig3D, PlanningStep3D, clinical_violation_score_3d, is_acceptable_3d
from dosim_sim.policy3d import action_settings_3d, initial_policy_step_3d, legal_action_mask_3d
from dosim_sim.representation3d import state_volume_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_prostate_case_3d
from train_3d_learner_pilot import load_records
from train_3d_volume_policy_pilot import ACTION_NAMES, ANGLES, MatchedVolumePolicyNet


def rollout(model, case, engine, config, volume_size: int) -> list[PlanningStep3D]:
    current = initial_policy_step_3d(case, engine, config)
    steps = [current]
    for step_index in range(1, config.max_steps + 1):
        if is_acceptable_3d(current.plan.metrics, case, config):
            break
        scalar = torch.tensor(
            state_features_3d(case, current, config.max_steps), dtype=torch.float32, device=engine.device
        ).unsqueeze(0)
        volume = state_volume_3d(case, current, config, volume_size).unsqueeze(0)
        with torch.no_grad():
            _, logits = model(scalar, volume)
        legal = torch.tensor(legal_action_mask_3d(case, current, config), device=engine.device)
        action_index = int(logits[0].masked_fill(~legal, -torch.inf).argmax().item())
        action, beams, priorities = action_settings_3d(action_index, current, config)
        if action is None:
            break
        plan = optimize_fluence_3d_torch(
            case, engine, beams, priorities, config.optimizer_iterations, initial_fluence=current.plan.fluence
        )
        current = PlanningStep3D(
            step_index, action, plan, clinical_violation_score_3d(plan.metrics, case, config)
        )
        steps.append(current)
    return steps


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = dose[mask]
    return np.asarray([100.0 * np.mean(values >= level) for level in bins], dtype=np.float64)


def conformity_metrics(case, dose: np.ndarray) -> dict[str, float]:
    target_volume = float(case.target.sum())
    prescription_isodose = (dose >= 0.95) & case.body
    half_prescription_isodose = (dose >= 0.50) & case.body
    covered_target = float((prescription_isodose & case.target).sum())
    prescription_volume = float(prescription_isodose.sum())
    paddick = covered_target**2 / max(target_volume * prescription_volume, 1.0)
    return {
        "paddick_ci_95": paddick,
        "rtog_ci_95": prescription_volume / target_volume,
        "r50": float(half_prescription_isodose.sum()) / target_volume,
        "body_mean_dose": float(dose[case.body].mean()),
        "target_v95_percent": 100.0 * covered_target / target_volume,
    }


def load_model(checkpoint_path: Path, scalar_count: int, device: torch.device) -> MatchedVolumePolicyNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    setting_count = int(checkpoint["model_state_dict"]["endpoint_head.weight"].shape[0])
    model = MatchedVolumePolicyNet(scalar_count, setting_count).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paired prostate policy DVHs")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fluence-size", type=int, default=12)
    parser.add_argument("--volume-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    config = HighLevelSearchConfig3D(max_steps=10, optimizer_iterations=args.iterations, priority_ceiling=25.0)
    records = [row for row in load_records(args.dataset_dir / "trajectory_view.jsonl") if row["split"] == "validation"]
    selected = {difficulty: next(row for row in records if row["difficulty"] == difficulty) for difficulty in ("easy", "moderate", "hard")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for difficulty, record in selected.items():
        case = generate_prostate_case_3d(int(record["seed"]), args.grid_size, difficulty=difficulty)
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float32)
        initial = initial_policy_step_3d(case, engine, config)
        scalar_count = len(state_features_3d(case, initial, config.max_steps))
        models = {
            condition: load_model(args.run_dir / f"model_{condition}_seed{args.seed}.pt", scalar_count, device)
            for condition in ("endpoint", "trajectory")
        }
        steps = {condition: rollout(model, case, engine, config, args.volume_size) for condition, model in models.items()}
        plans = {"initial": initial.plan, **{condition: value[-1].plan for condition, value in steps.items()}}
        doses = {name: plan.dose.detach().float().cpu().numpy() for name, plan in plans.items()}
        maximum = max(float(dose.max()) for dose in doses.values())
        bins = np.linspace(0.0, max(1.5, np.ceil(maximum * 20) / 20), 181)
        masks = {
            "PTV": case.target,
            "bladder": case.oars[0],
            "rectum": case.oars[1],
            "femoral heads": case.oars[2],
            "normal body": case.body & ~case.target,
        }
        colors = {
            "PTV": "#d62728",
            "bladder": "#1f77b4",
            "rectum": "#2ca02c",
            "femoral heads": "#9467bd",
            "normal body": "#6b6b6b",
        }
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
        for axis, (plan_name, dose) in zip(axes, doses.items(), strict=True):
            for structure, mask in masks.items():
                axis.plot(bins, cumulative_dvh(dose, mask, bins), label=structure, color=colors[structure], linewidth=2)
            axis.axvline(1.0, color="black", linestyle="--", linewidth=1, label="prescription" if plan_name == "initial" else None)
            axis.set_xlim(0.0, bins[-1])
            axis.set_ylim(0.0, 100.0)
            axis.set_xlabel("Relative dose")
            axis.set_title(plan_name.replace("_", " ").title())
            axis.grid(alpha=0.25)
            metrics = conformity_metrics(case, dose)
            metric_rows.append({
                "case_id": case.case_id,
                "difficulty": difficulty,
                "plan": plan_name,
                "target_d95": plans[plan_name].metrics.target_d95,
                "target_d02": plans[plan_name].metrics.target_d02,
                "oar_0_mean": plans[plan_name].metrics.oar_mean[0],
                "oar_1_mean": plans[plan_name].metrics.oar_mean[1],
                "oar_2_mean": plans[plan_name].metrics.oar_mean[2],
                **metrics,
            })
            axis.text(
                0.98,
                0.97,
                f"Paddick CI95 {metrics['paddick_ci_95']:.2f}\nR50 {metrics['r50']:.1f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=10,
            )
        axes[0].set_ylabel("Volume receiving at least dose (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=6, frameon=False)
        figure.suptitle(f"Paired prostate DVHs: {difficulty} case {case.case_id}", y=0.995)
        figure.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.82, wspace=0.05)
        figure.savefig(args.output_dir / f"dvh_{difficulty}_{case.case_id}.png", dpi=180)
        plt.close(figure)

        for plan_name, dose in doses.items():
            np.savez_compressed(
                args.output_dir / f"dvh_{difficulty}_{case.case_id}_{plan_name}.npz",
                dose_bins=bins,
                **{structure.replace(" ", "_"): cumulative_dvh(dose, mask, bins) for structure, mask in masks.items()},
            )

    with (args.output_dir / "conformity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"run_dir": str(args.run_dir), "seed": args.seed, "cases": list(selected)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
