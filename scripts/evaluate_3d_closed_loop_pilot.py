import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dosim_sim.objective import PlanningPriorities
from dosim_sim.optimizer3d import objective_value_3d
from dosim_sim.planning3d import HighLevelSearchConfig3D, clinical_violation_score_3d, is_acceptable_3d
from dosim_sim.policy3d import initial_policy_step_3d, rollout_policy_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d
from train_3d_learner_pilot import fit, load_records, tensors


ANGLES = tuple(float(value) for value in range(0, 360, 30))


def endpoint_settings(
    prediction: torch.Tensor,
    case,
    config: HighLevelSearchConfig3D,
) -> tuple[tuple[int, ...], PlanningPriorities]:
    values = torch.sigmoid(prediction).detach().float().cpu().numpy()
    beam_scores = values[:12]
    beams = [beam for beam in case.available_beams if beam_scores[beam] >= 0.5]
    if len(beams) < 3:
        beams = sorted(case.available_beams, key=lambda beam: (-beam_scores[beam], beam))[:3]
    priority_values = np.clip(values[12:] * 25.0, config.priority_floor, config.priority_ceiling)
    priorities = PlanningPriorities(
        target=float(priority_values[0]),
        hotspot=float(priority_values[1]),
        oars=tuple(float(value) for value in priority_values[2 : 2 + len(case.oars)]),
    )
    return tuple(sorted(beams)), priorities


def save_plot(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    conditions = ("endpoint_direct", "trajectory_policy")
    colors = {"endpoint_direct": "#4E79A7", "trajectory_policy": "#E15759"}
    x = np.arange(2)
    for seed in sorted({int(row["training_seed"]) for row in rows}):
        subset = [row for row in rows if int(row["training_seed"]) == seed]
        rates = [np.mean([bool(row["acceptable"]) for row in subset if row["condition"] == condition]) for condition in conditions]
        axes[0].plot(x, rates, marker="o", color="#777777", alpha=0.65)
    axes[0].set_xticks(x, ["endpoint\ndirect", "trajectory\npolicy"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Held-out acceptable-plan rate")
    axes[0].set_title("Paired seed-level acceptability")
    for index, condition in enumerate(conditions):
        values = [float(row["violation_score"]) for row in rows if row["condition"] == condition]
        axes[1].scatter(np.full(len(values), index), values, color=colors[condition], alpha=0.55, s=24)
    axes[1].set_xticks(x, ["endpoint\ndirect", "trajectory\npolicy"])
    axes[1].set_ylabel("Clinical violation score")
    axes[1].set_title("Forty paired case-seed evaluations")
    action_rows = [row for row in rows if row["condition"] == "trajectory_policy"]
    axes[2].hist([int(row["high_level_actions"]) for row in action_rows], bins=np.arange(-0.5, 11.5), color=colors["trajectory_policy"])
    axes[2].set_xlabel("Executed high-level actions")
    axes[2].set_ylabel("Case-seed evaluations")
    axes[2].set_title("Closed-loop trajectory length")
    fig.suptitle("Closed-loop 3D learner development pilot", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct endpoint and trajectory policy models in the 3D simulator")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/3d_dataset_pilot_revised"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--action-weight", type=float, default=0.02)
    parser.add_argument("--action-balancing", choices=["none", "sqrt_inverse"], default="sqrt_inverse")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_closed_loop_pilot"))
    args = parser.parse_args()
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")[:32]
    permutation = np.random.default_rng(20260809).permutation(len(records))
    train_records = [records[index] for index in permutation[:24]]
    test_records = [records[index] for index in permutation[24:32]]
    device = torch.device(args.device)
    train_tensors = tensors(train_records, device)
    cfg = HighLevelSearchConfig3D(max_steps=args.max_steps, optimizer_iterations=args.iterations, priority_ceiling=25.0)
    rows: list[dict] = []
    for training_seed in range(args.seeds):
        models = {}
        for condition in ("endpoint", "trajectory"):
            model, _ = fit(
                condition,
                training_seed,
                *train_tensors,
                epochs=args.epochs,
                learning_rate=0.003,
                action_weight=args.action_weight,
                action_balancing=args.action_balancing,
            )
            model.eval()
            models[condition] = model
        for record in test_records:
            case = generate_case_3d(int(record["seed"]), args.grid_size, difficulty=record["difficulty"])
            engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=torch.float16)
            initial = initial_policy_step_3d(case, engine, cfg)
            features = torch.tensor(record["initial_features"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                endpoint_prediction, _ = models["endpoint"](features)
            beams, priorities = endpoint_settings(endpoint_prediction[0], case, cfg)
            endpoint_plan = optimize_fluence_3d_torch(
                case,
                engine,
                beams,
                priorities,
                iterations=args.iterations * args.max_steps,
                initial_fluence=initial.plan.fluence,
            )
            rows.append({
                "training_seed": training_seed,
                "case_id": case.case_id,
                "case_seed": case.seed,
                "difficulty": case.difficulty,
                "condition": "endpoint_direct",
                "acceptable": is_acceptable_3d(endpoint_plan.metrics, case, cfg),
                "violation_score": clinical_violation_score_3d(endpoint_plan.metrics, case, cfg),
                "canonical_objective": objective_value_3d(case, endpoint_plan.dose.detach().float().cpu().numpy()),
                "high_level_actions": 1,
                "stopping_reason": "direct_terminal_settings",
            })

            def logits_function(array: np.ndarray) -> np.ndarray:
                tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, logits = models["trajectory"](tensor)
                return logits[0].detach().float().cpu().numpy()

            trajectory = rollout_policy_3d(case, engine, logits_function, cfg)
            final = trajectory.final.plan
            rows.append({
                "training_seed": training_seed,
                "case_id": case.case_id,
                "case_seed": case.seed,
                "difficulty": case.difficulty,
                "condition": "trajectory_policy",
                "acceptable": is_acceptable_3d(final.metrics, case, cfg),
                "violation_score": trajectory.final.violation_score,
                "canonical_objective": objective_value_3d(case, final.dose.detach().float().cpu().numpy()),
                "high_level_actions": len(trajectory.steps) - 1,
                "stopping_reason": trajectory.stopping_reason,
            })
        print(f"completed training seed {training_seed + 1}/{args.seeds}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "01_closed_loop_pilot.png")
    summary = {
        "status": "contextual development comparison; endpoint arm is a direct regressor, not the primary iterative endpoint-only policy",
        "heldout_cases": len(test_records),
        "training_seeds": args.seeds,
        "endpoint_direct_acceptable_rate": float(np.mean([bool(row["acceptable"]) for row in rows if row["condition"] == "endpoint_direct"])),
        "trajectory_policy_acceptable_rate": float(np.mean([bool(row["acceptable"]) for row in rows if row["condition"] == "trajectory_policy"])),
        "endpoint_direct_mean_violation": float(np.mean([float(row["violation_score"]) for row in rows if row["condition"] == "endpoint_direct"])),
        "trajectory_policy_mean_violation": float(np.mean([float(row["violation_score"]) for row in rows if row["condition"] == "trajectory_policy"])),
        "trajectory_policy_mean_actions": float(np.mean([int(row["high_level_actions"]) for row in rows if row["condition"] == "trajectory_policy"])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
