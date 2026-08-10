import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

from dosim_sim.dataset3d import ACTION_NAMES, state_features_3d
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
)
from dosim_sim.policy3d import (
    action_settings_3d,
    initial_policy_step_3d,
    legal_action_mask_3d,
    rollout_policy_3d,
)
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d
from train_3d_learner_pilot import MatchedPilotNet, endpoint_loss, load_records, tensors


ANGLES = tuple(float(value) for value in range(0, 360, 30))


@dataclass
class Episode:
    log_probability: torch.Tensor
    entropy: torch.Tensor
    reward: float
    acceptable: bool
    violation: float
    actions: int


def sample_episode(
    model: MatchedPilotNet,
    case,
    engine: TorchImplicitDoseEngine3D,
    initial: PlanningStep3D,
    config: HighLevelSearchConfig3D,
) -> Episode:
    current = initial
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    actions = 0
    for step_index in range(1, config.max_steps + 1):
        if is_acceptable_3d(current.plan.metrics, case, config):
            break
        features = torch.tensor(
            state_features_3d(case, current, config.max_steps),
            dtype=torch.float32,
            device=engine.device,
        ).unsqueeze(0)
        _, logits = model(features)
        legal = torch.tensor(
            legal_action_mask_3d(case, current, config), device=engine.device, dtype=torch.bool
        )
        masked_logits = logits[0].masked_fill(~legal, -torch.inf)
        distribution = Categorical(logits=masked_logits)
        action_index = distribution.sample()
        log_probabilities.append(distribution.log_prob(action_index))
        entropies.append(distribution.entropy())
        action, beams, priorities = action_settings_3d(int(action_index.item()), current, config)
        if action is None:
            break
        plan = optimize_fluence_3d_torch(
            case,
            engine,
            beams,
            priorities,
            config.optimizer_iterations,
            initial_fluence=current.plan.fluence,
        )
        actions += 1
        current = PlanningStep3D(
            step_index,
            action,
            plan,
            clinical_violation_score_3d(plan.metrics, case, config),
        )
    acceptable = is_acceptable_3d(current.plan.metrics, case, config)
    violation = current.violation_score
    reward = float(acceptable) - min(violation, 2.0) - 0.01 * actions
    zero = torch.zeros((), device=engine.device)
    return Episode(
        log_probability=torch.stack(log_probabilities).sum() if log_probabilities else zero,
        entropy=torch.stack(entropies).mean() if entropies else zero,
        reward=reward,
        acceptable=acceptable,
        violation=violation,
        actions=actions,
    )


def train_condition(
    condition: str,
    seed: int,
    records: list[dict],
    cases: list,
    engines: list[TorchImplicitDoseEngine3D],
    initials: list[PlanningStep3D],
    config: HighLevelSearchConfig3D,
    pretrain_updates: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    policy_weight: float,
    action_weight: float,
    entropy_weight: float,
) -> tuple[MatchedPilotNet, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = engines[0].device
    endpoint_x, endpoint_y, action_x, action_y = tensors(records, device)
    model = MatchedPilotNet(endpoint_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(pretrain_updates):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x)
        loss = endpoint_loss(endpoint_prediction, endpoint_y)
        if condition == "trajectory":
            _, action_logits = model(action_x)
            loss = loss + action_weight * torch.nn.functional.cross_entropy(action_logits, action_y)
        loss.backward()
        optimizer.step()
    schedule = np.random.default_rng(seed + 314159).integers(
        0, len(records), size=(updates, batch_size)
    )
    history: list[dict] = []
    for update_index, indices in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x)
        terminal_loss = endpoint_loss(endpoint_prediction, endpoint_y)
        episodes = [
            sample_episode(model, cases[int(index)], engines[int(index)], initials[int(index)], config)
            for index in indices
        ]
        rewards = torch.tensor([episode.reward for episode in episodes], device=device)
        baseline = rewards.mean()
        policy_loss = torch.stack(
            [-(rewards[index] - baseline).detach() * episode.log_probability for index, episode in enumerate(episodes)]
        ).mean()
        entropy = torch.stack([episode.entropy for episode in episodes]).mean()
        action_loss = torch.zeros((), device=device)
        if condition == "trajectory":
            _, action_logits = model(action_x)
            action_loss = torch.nn.functional.cross_entropy(action_logits, action_y)
        loss = terminal_loss + policy_weight * policy_loss + action_weight * action_loss - entropy_weight * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            _, diagnostic_action_logits = model(action_x)
            diagnostic_action_cross_entropy = torch.nn.functional.cross_entropy(
                diagnostic_action_logits, action_y
            )
            diagnostic_action_accuracy = (diagnostic_action_logits.argmax(dim=1) == action_y).float().mean()
        history.append({
            "condition": condition,
            "seed": seed,
            "update": update_index,
            "mean_reward": float(rewards.mean().item()),
            "acceptable_rate": float(np.mean([episode.acceptable for episode in episodes])),
            "mean_violation": float(np.mean([episode.violation for episode in episodes])),
            "mean_actions": float(np.mean([episode.actions for episode in episodes])),
            "terminal_loss": float(terminal_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "action_loss": float(action_loss.detach().item()),
            "diagnostic_action_cross_entropy": float(diagnostic_action_cross_entropy.item()),
            "diagnostic_action_accuracy": float(diagnostic_action_accuracy.item()),
        })
        if update_index % max(updates // 5, 1) == 0:
            print(
                f"seed={seed} condition={condition} update={update_index}/{updates} "
                f"reward={history[-1]['mean_reward']:.3f} acceptable={history[-1]['acceptable_rate']:.2f}",
                flush=True,
            )
    return model, history


def evaluate_model(
    model: MatchedPilotNet,
    condition: str,
    training_seed: int,
    records: list[dict],
    config: HighLevelSearchConfig3D,
    grid_size: int,
    fluence_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    model.eval()
    rows = []
    for record in records:
        case = generate_case_3d(int(record["seed"]), grid_size, difficulty=record["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, ANGLES, fluence_size, device=device, dtype=dtype)

        def logits_function(array: np.ndarray) -> np.ndarray:
            features = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, logits = model(features)
            return logits[0].detach().float().cpu().numpy()

        trajectory = rollout_policy_3d(case, engine, logits_function, config)
        rows.append({
            "training_seed": training_seed,
            "condition": condition,
            "case_id": case.case_id,
            "case_seed": case.seed,
            "difficulty": case.difficulty,
            "acceptable": is_acceptable_3d(trajectory.final.plan.metrics, case, config),
            "violation_score": trajectory.final.violation_score,
            "high_level_actions": len(trajectory.steps) - 1,
            "stopping_reason": trajectory.stopping_reason,
        })
    return rows


def save_plot(history: list[dict], results: list[dict], path: Path) -> None:
    colors = {"endpoint": "#4E79A7", "trajectory": "#E15759"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for condition in ("endpoint", "trajectory"):
        for seed in sorted({int(row["seed"]) for row in history}):
            subset = [row for row in history if row["condition"] == condition and int(row["seed"]) == seed]
            axes[0].plot([row["update"] for row in subset], [row["mean_reward"] for row in subset], color=colors[condition], alpha=0.35)
    axes[0].set_xlabel("Matched optimizer updates")
    axes[0].set_ylabel("Training rollout reward")
    axes[0].set_title("Terminal-outcome policy training")
    x = np.arange(2)
    for seed in sorted({int(row["training_seed"]) for row in results}):
        subset = [row for row in results if int(row["training_seed"]) == seed]
        rates = [np.mean([bool(row["acceptable"]) for row in subset if row["condition"] == condition]) for condition in ("endpoint", "trajectory")]
        axes[1].plot(x, rates, marker="o", color="#777777", alpha=0.7)
    axes[1].set_xticks(x, ["endpoint only", "trajectory supervised"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Held-out acceptable-plan rate")
    axes[1].set_title("Identical iterative rollout")
    for index, condition in enumerate(("endpoint", "trajectory")):
        values = [float(row["violation_score"]) for row in results if row["condition"] == condition]
        axes[2].scatter(np.full(len(values), index), values, color=colors[condition], alpha=0.6)
    axes[2].set_xticks(x, ["endpoint only", "trajectory supervised"])
    axes[2].set_ylabel("Held-out violation score")
    axes[2].set_title("Paired case-seed outcomes")
    fig.suptitle("Matched iterative-policy development pilot", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched iterative endpoint-only and trajectory-supervised policies")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/3d_dataset_pilot_revised"))
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--heldout-cases", type=int, default=8)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--pretrain-updates", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--policy-weight", type=float, default=0.2)
    parser.add_argument("--action-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_iterative_policy_pilot"))
    args = parser.parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    if records and all(record.get("split") in {"train", "validation"} for record in records):
        train_records = [record for record in records if record["split"] == "train"][: args.train_cases]
        test_records = [record for record in records if record["split"] == "validation"][: args.heldout_cases]
    else:
        required = args.train_cases + args.heldout_cases
        if len(records) < required:
            raise ValueError(f"dataset has {len(records)} records; {required} required")
        permutation = np.random.default_rng(20260809).permutation(len(records))
        train_records = [records[index] for index in permutation[: args.train_cases]]
        test_records = [records[index] for index in permutation[args.train_cases : required]]
    if len(train_records) != args.train_cases or len(test_records) != args.heldout_cases:
        raise ValueError(
            f"requested {args.train_cases} train and {args.heldout_cases} validation records; "
            f"found {len(train_records)} and {len(test_records)}"
        )
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.cuda.set_device(device)
    config = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        optimizer_iterations=args.iterations,
        priority_ceiling=25.0,
    )
    train_cases = [generate_case_3d(int(record["seed"]), args.grid_size, difficulty=record["difficulty"]) for record in train_records]
    train_engines = [TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=dtype) for case in train_cases]
    train_initials = [initial_policy_step_3d(case, engine, config) for case, engine in zip(train_cases, train_engines, strict=True)]
    histories: list[dict] = []
    results: list[dict] = []
    parameter_counts = set()
    seed_values = range(args.seed_start, args.seed_start + args.seeds)
    for seed in seed_values:
        for condition in ("endpoint", "trajectory"):
            model, history = train_condition(
                condition,
                seed,
                train_records,
                train_cases,
                train_engines,
                train_initials,
                config,
                args.pretrain_updates,
                args.updates,
                args.batch_size,
                args.learning_rate,
                args.policy_weight,
                args.action_weight,
                args.entropy_weight,
            )
            parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
            histories.extend(history)
            results.extend(evaluate_model(model, condition, seed, test_records, config, args.grid_size, args.fluence_size, device, dtype))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
        writer.writeheader(); writer.writerows(histories)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    save_plot(histories, results, args.output_dir / "01_iterative_policy_pilot.png")
    case_metrics_hash = hashlib.sha256((args.output_dir / "case_metrics.csv").read_bytes()).hexdigest()
    training_history_hash = hashlib.sha256((args.output_dir / "training_history.csv").read_bytes()).hexdigest()
    endpoint_rows = [row for row in results if row["condition"] == "endpoint"]
    trajectory_rows = [row for row in results if row["condition"] == "trajectory"]
    final_history = [row for row in histories if int(row["update"]) == args.updates]
    summary = {
        "status": "matched iterative-policy development pilot; not a primary result",
        "same_architecture_and_parameter_count": len(parameter_counts) == 1,
        "parameter_count": next(iter(parameter_counts)),
        "same_policy_updates": True,
        "same_total_optimizer_updates": True,
        "same_rollout_limit_and_action_mask": True,
        "training_cases": len(train_records),
        "heldout_cases": len(test_records),
        "training_seeds": args.seeds,
        "training_seed_start": args.seed_start,
        "updates": args.updates,
        "pretrain_updates": args.pretrain_updates,
        "batch_size": args.batch_size,
        "deterministic_algorithms": args.deterministic,
        "geometry_dtype": args.dtype,
        "case_metrics_sha256": case_metrics_hash,
        "training_history_sha256": training_history_hash,
        "endpoint_acceptable_rate": float(np.mean([bool(row["acceptable"]) for row in endpoint_rows])),
        "trajectory_acceptable_rate": float(np.mean([bool(row["acceptable"]) for row in trajectory_rows])),
        "endpoint_mean_violation": float(np.mean([float(row["violation_score"]) for row in endpoint_rows])),
        "trajectory_mean_violation": float(np.mean([float(row["violation_score"]) for row in trajectory_rows])),
        "endpoint_mean_actions": float(np.mean([int(row["high_level_actions"]) for row in endpoint_rows])),
        "trajectory_mean_actions": float(np.mean([int(row["high_level_actions"]) for row in trajectory_rows])),
        "endpoint_final_training_action_accuracy": float(np.mean([
            row["diagnostic_action_accuracy"] for row in final_history if row["condition"] == "endpoint"
        ])),
        "trajectory_final_training_action_accuracy": float(np.mean([
            row["diagnostic_action_accuracy"] for row in final_history if row["condition"] == "trajectory"
        ])),
        "endpoint_final_training_action_cross_entropy": float(np.mean([
            row["diagnostic_action_cross_entropy"] for row in final_history if row["condition"] == "endpoint"
        ])),
        "trajectory_final_training_action_cross_entropy": float(np.mean([
            row["diagnostic_action_cross_entropy"] for row in final_history if row["condition"] == "trajectory"
        ])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
