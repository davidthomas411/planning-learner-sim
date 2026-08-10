import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import Categorical

from dosim_sim.dataset3d import ACTION_NAMES, state_features_3d
from dosim_sim.planning3d import (
    HighLevelSearchConfig3D,
    PlanningStep3D,
    clinical_violation_score_3d,
    is_acceptable_3d,
)
from dosim_sim.policy3d import action_settings_3d, initial_policy_step_3d, legal_action_mask_3d
from dosim_sim.representation3d import VOLUME_CHANNEL_NAMES, state_volume_3d
from dosim_sim.torch_dose3d import TorchImplicitDoseEngine3D, optimize_fluence_3d_torch
from dosim_sim.volume3d import generate_case_3d, generate_prostate_case_3d
from train_3d_iterative_policy_pilot import save_plot
from train_3d_learner_pilot import endpoint_loss, load_records


ANGLES = tuple(float(value) for value in range(0, 360, 30))


class MatchedVolumePolicyNet(nn.Module):
    """Shared 3D image and scalar encoder for both supervision conditions."""

    def __init__(self, scalar_count: int) -> None:
        super().__init__()
        self.volume_encoder = nn.Sequential(
            nn.Conv3d(len(VOLUME_CHANNEL_NAMES), 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_count, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(48 + 64, 128), nn.ReLU())
        self.endpoint_head = nn.Linear(128, 17)
        self.action_head = nn.Linear(128, len(ACTION_NAMES))

    def forward(self, scalar: torch.Tensor, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_volume = self.volume_encoder(volume)
        encoded_scalar = self.scalar_encoder(scalar)
        fused = self.fusion(torch.cat([encoded_volume, encoded_scalar], dim=1))
        return self.endpoint_head(fused), self.action_head(fused)


@dataclass
class Episode:
    log_probability: torch.Tensor
    entropy: torch.Tensor
    reward: float
    acceptable: bool
    violation: float
    actions: int


def save_channel_montage(
    volume: torch.Tensor,
    output_path: Path,
    channel_names: tuple[str, ...] = VOLUME_CHANNEL_NAMES,
) -> None:
    """Save the central axial slice of each model input channel."""

    data = volume.detach().float().cpu().numpy()
    axial_index = int(np.argmax(data[1].sum(axis=(0, 1))))
    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index >= len(channel_names):
            axis.axis("off")
            continue
        image = axis.imshow(data[index, :, :, axial_index], origin="lower", cmap="viridis")
        axis.set_title(channel_names[index].replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("3D policy input: central axial slice")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def select_records(records: list[dict], train_cases: int, heldout_cases: int) -> tuple[list[dict], list[dict]]:
    if not records or not all(record.get("split") in {"train", "validation"} for record in records):
        raise ValueError("the 3D volume pilot requires stored train and validation partitions")
    train = [record for record in records if record["split"] == "train"][:train_cases]
    validation = [record for record in records if record["split"] == "validation"][:heldout_cases]
    if len(train) != train_cases or len(validation) != heldout_cases:
        raise ValueError(
            f"requested {train_cases} train and {heldout_cases} validation records; "
            f"found {len(train)} and {len(validation)}"
        )
    return train, validation


def replay_training_tensors(
    records: list[dict],
    cases: list,
    engines: list[TorchImplicitDoseEngine3D],
    attempt_by_case: dict[str, dict],
    rollout_config: HighLevelSearchConfig3D,
    volume_size: int,
    shallow_iterations: int,
    deep_iterations: int,
) -> tuple[torch.Tensor, ...]:
    endpoint_scalars = []
    endpoint_volumes = []
    endpoint_targets = []
    action_scalars = []
    action_volumes = []
    action_targets = []
    replay_differences = []
    for record, case, engine in zip(records, cases, engines, strict=True):
        tier = attempt_by_case[record["case_id"]]["search_tier"]
        iterations = deep_iterations if tier == "deep" else shallow_iterations
        replay_config = HighLevelSearchConfig3D(
            max_steps=10,
            optimizer_iterations=iterations,
            priority_ceiling=25.0 if tier == "deep" else 6.0,
        )
        current = initial_policy_step_3d(case, engine, replay_config)
        endpoint_scalars.append(state_features_3d(case, current, rollout_config.max_steps))
        endpoint_volumes.append(state_volume_3d(case, current, rollout_config, volume_size))
        endpoint_targets.append(record["final_settings"])
        for transition in record["trajectory"]:
            computed = state_features_3d(case, current, rollout_config.max_steps)
            stored = np.asarray(transition["state"], dtype=np.float32)
            replay_differences.append(float(np.mean(np.abs(computed - stored))))
            action_scalars.append(computed)
            action_volumes.append(state_volume_3d(case, current, rollout_config, volume_size))
            action_index = int(transition["action_index"])
            action_targets.append(action_index)
            if ACTION_NAMES[action_index] == "stop":
                break
            legal = legal_action_mask_3d(case, current, replay_config)
            if not legal[action_index]:
                raise ValueError(f"stored action {ACTION_NAMES[action_index]} is illegal during replay for {case.case_id}")
            action, beams, priorities = action_settings_3d(action_index, current, replay_config)
            if action is None:
                raise ValueError("non-stop demonstration action translated to stop")
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                beams,
                priorities,
                replay_config.optimizer_iterations,
                initial_fluence=current.plan.fluence,
            )
            current = PlanningStep3D(
                current.step + 1,
                action,
                plan,
                clinical_violation_score_3d(plan.metrics, case, replay_config),
            )
    device = engines[0].device
    return (
        torch.tensor(np.asarray(endpoint_scalars), dtype=torch.float32, device=device),
        torch.stack(endpoint_volumes),
        torch.tensor(np.asarray(endpoint_targets), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(action_scalars), dtype=torch.float32, device=device),
        torch.stack(action_volumes),
        torch.tensor(action_targets, dtype=torch.long, device=device),
        torch.tensor(replay_differences, dtype=torch.float32, device=device),
    )


def sample_episode(
    model: MatchedVolumePolicyNet,
    case,
    engine: TorchImplicitDoseEngine3D,
    initial: PlanningStep3D,
    config: HighLevelSearchConfig3D,
    volume_size: int,
) -> Episode:
    current = initial
    log_probabilities = []
    entropies = []
    actions = 0
    for step_index in range(1, config.max_steps + 1):
        if is_acceptable_3d(current.plan.metrics, case, config):
            break
        scalar = torch.tensor(
            state_features_3d(case, current, config.max_steps), dtype=torch.float32, device=engine.device
        ).unsqueeze(0)
        volume = state_volume_3d(case, current, config, volume_size).unsqueeze(0)
        _, logits = model(scalar, volume)
        legal = torch.tensor(legal_action_mask_3d(case, current, config), device=engine.device)
        distribution = Categorical(logits=logits[0].masked_fill(~legal, -torch.inf))
        action_index = distribution.sample()
        log_probabilities.append(distribution.log_prob(action_index))
        entropies.append(distribution.entropy())
        action, beams, priorities = action_settings_3d(int(action_index.item()), current, config)
        if action is None:
            break
        plan = optimize_fluence_3d_torch(
            case, engine, beams, priorities, config.optimizer_iterations, initial_fluence=current.plan.fluence
        )
        actions += 1
        current = PlanningStep3D(
            step_index, action, plan, clinical_violation_score_3d(plan.metrics, case, config)
        )
    acceptable = is_acceptable_3d(current.plan.metrics, case, config)
    violation = current.violation_score
    reward = float(acceptable) - min(violation, 2.0) - 0.01 * actions
    zero = torch.zeros((), device=engine.device)
    return Episode(
        torch.stack(log_probabilities).sum() if log_probabilities else zero,
        torch.stack(entropies).mean() if entropies else zero,
        reward,
        acceptable,
        violation,
        actions,
    )


def train_condition(
    condition: str,
    seed: int,
    tensors: tuple[torch.Tensor, ...],
    cases: list,
    engines: list[TorchImplicitDoseEngine3D],
    initials: list[PlanningStep3D],
    config: HighLevelSearchConfig3D,
    volume_size: int,
    pretrain_updates: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    policy_weight: float,
    action_weight: float,
    entropy_weight: float,
) -> tuple[MatchedVolumePolicyNet, list[dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    endpoint_x, endpoint_volume, endpoint_y, action_x, action_volume, action_y, _ = tensors
    model = MatchedVolumePolicyNet(endpoint_x.shape[1]).to(endpoint_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(pretrain_updates):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x, endpoint_volume)
        loss = endpoint_loss(endpoint_prediction, endpoint_y)
        if condition == "trajectory":
            _, action_logits = model(action_x, action_volume)
            loss = loss + action_weight * nn.functional.cross_entropy(action_logits, action_y)
        loss.backward()
        optimizer.step()
    schedule = np.random.default_rng(seed + 314159).integers(0, len(cases), size=(updates, batch_size))
    history = []
    for update_index, indices in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x, endpoint_volume)
        terminal_loss = endpoint_loss(endpoint_prediction, endpoint_y)
        episodes = [
            sample_episode(model, cases[int(index)], engines[int(index)], initials[int(index)], config, volume_size)
            for index in indices
        ]
        rewards = torch.tensor([episode.reward for episode in episodes], device=endpoint_x.device)
        baseline = rewards.mean()
        policy_loss = torch.stack([
            -(rewards[index] - baseline).detach() * episode.log_probability
            for index, episode in enumerate(episodes)
        ]).mean()
        entropy = torch.stack([episode.entropy for episode in episodes]).mean()
        action_loss = torch.zeros((), device=endpoint_x.device)
        if condition == "trajectory":
            _, action_logits = model(action_x, action_volume)
            action_loss = nn.functional.cross_entropy(action_logits, action_y)
        loss = terminal_loss + policy_weight * policy_loss + action_weight * action_loss - entropy_weight * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
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
        })
    with torch.no_grad():
        _, diagnostic_logits = model(action_x, action_volume)
        diagnostics = {
            "action_accuracy": float((diagnostic_logits.argmax(1) == action_y).float().mean().item()),
            "action_cross_entropy": float(nn.functional.cross_entropy(diagnostic_logits, action_y).item()),
        }
    return model, history, diagnostics


def evaluate_model(
    model: MatchedVolumePolicyNet,
    condition: str,
    seed: int,
    records: list[dict],
    config: HighLevelSearchConfig3D,
    grid_size: int,
    fluence_size: int,
    volume_size: int,
    device: torch.device,
    dtype: torch.dtype,
    anatomy: str,
) -> list[dict]:
    model.eval()
    rows = []
    for record in records:
        generator = generate_prostate_case_3d if anatomy == "prostate" else generate_case_3d
        case = generator(int(record["seed"]), grid_size, difficulty=record["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, ANGLES, fluence_size, device=device, dtype=dtype)
        current = initial_policy_step_3d(case, engine, config)
        steps = [current]
        for step_index in range(1, config.max_steps + 1):
            if is_acceptable_3d(current.plan.metrics, case, config):
                break
            scalar = torch.tensor(
                state_features_3d(case, current, config.max_steps), dtype=torch.float32, device=device
            ).unsqueeze(0)
            volume = state_volume_3d(case, current, config, volume_size).unsqueeze(0)
            with torch.no_grad():
                _, logits = model(scalar, volume)
            legal = torch.tensor(legal_action_mask_3d(case, current, config), device=device)
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
        rows.append({
            "training_seed": seed,
            "condition": condition,
            "case_id": case.case_id,
            "case_seed": case.seed,
            "difficulty": case.difficulty,
            "acceptable": is_acceptable_3d(current.plan.metrics, case, config),
            "violation_score": current.violation_score,
            "high_level_actions": len(steps) - 1,
            "stopping_reason": "acceptable" if is_acceptable_3d(current.plan.metrics, case, config) else "policy_step_limit",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched 3D image-plus-scalar planning policies")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/pilot300_local_v2/merged"))
    parser.add_argument("--train-cases", type=int, default=24)
    parser.add_argument("--heldout-cases", type=int, default=8)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--pretrain-updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--policy-weight", type=float, default=0.2)
    parser.add_argument("--action-weight", type=float, default=0.20)
    parser.add_argument("--entropy-weight", type=float, default=0.002)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--anatomy", choices=("generic", "prostate"), default="generic")
    parser.add_argument("--volume-size", type=int, default=16)
    parser.add_argument("--fluence-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--shallow-replay-iterations", type=int, default=20)
    parser.add_argument("--deep-replay-iterations", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_volume_policy_pilot"))
    args = parser.parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    train_records, validation_records = select_records(records, args.train_cases, args.heldout_cases)
    attempt_rows = load_records(args.dataset_dir / "attempt_manifest.jsonl")
    attempt_by_case = {row["case_id"]: row for row in attempt_rows}
    config = HighLevelSearchConfig3D(
        max_steps=args.max_steps, optimizer_iterations=args.iterations, priority_ceiling=25.0
    )
    generator = generate_prostate_case_3d if args.anatomy == "prostate" else generate_case_3d
    cases = [generator(int(row["seed"]), args.grid_size, difficulty=row["difficulty"]) for row in train_records]
    engines = [TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=dtype) for case in cases]
    initials = [initial_policy_step_3d(case, engine, config) for case, engine in zip(cases, engines, strict=True)]
    training_tensors = replay_training_tensors(
        train_records,
        cases,
        engines,
        attempt_by_case,
        config,
        args.volume_size,
        args.shallow_replay_iterations,
        args.deep_replay_iterations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_names = VOLUME_CHANNEL_NAMES
    if args.anatomy == "prostate":
        channel_names = (
            "body", "prostate_ptv", "bladder", "rectum", "femoral_heads", "dose",
            "target_underdose", "target_hotspot", "bladder_excess", "rectum_excess", "femoral_head_excess",
        )
    save_channel_montage(
        training_tensors[1][0], args.output_dir / "00_model_input_channels.png", channel_names
    )
    histories = []
    results = []
    diagnostics = []
    parameter_counts = set()
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for condition in ("endpoint", "trajectory"):
            model, history, diagnostic = train_condition(
                condition,
                seed,
                training_tensors,
                cases,
                engines,
                initials,
                config,
                args.volume_size,
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
            diagnostics.append({"condition": condition, "seed": seed, **diagnostic})
            results.extend(evaluate_model(
                model,
                condition,
                seed,
                validation_records,
                config,
                args.grid_size,
                args.fluence_size,
                args.volume_size,
                device,
                dtype,
                args.anatomy,
            ))
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(histories)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    with (args.output_dir / "training_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(diagnostics)
    save_plot(histories, results, args.output_dir / "01_volume_policy_pilot.png")
    endpoint_rows = [row for row in results if row["condition"] == "endpoint"]
    trajectory_rows = [row for row in results if row["condition"] == "trajectory"]
    summary = {
        "status": "3D image-plus-scalar policy development pilot; not a primary result",
        "anatomy": args.anatomy,
        "information_channels": list(VOLUME_CHANNEL_NAMES),
        "volume_size": args.volume_size,
        "same_architecture_and_parameter_count": len(parameter_counts) == 1,
        "parameter_count": next(iter(parameter_counts)),
        "endpoint_action_labels_used_for_training": False,
        "trajectory_action_labels_used_for_training": True,
        "training_cases": len(train_records),
        "heldout_cases": len(validation_records),
        "training_seeds": args.seeds,
        "seed_start": args.seed_start,
        "replay_scalar_mean_absolute_difference": float(training_tensors[-1].mean().item()),
        "endpoint_acceptable_rate": float(np.mean([row["acceptable"] for row in endpoint_rows])),
        "trajectory_acceptable_rate": float(np.mean([row["acceptable"] for row in trajectory_rows])),
        "endpoint_mean_violation": float(np.mean([row["violation_score"] for row in endpoint_rows])),
        "trajectory_mean_violation": float(np.mean([row["violation_score"] for row in trajectory_rows])),
        "endpoint_training_action_accuracy_diagnostic_only": float(np.mean([
            row["action_accuracy"] for row in diagnostics if row["condition"] == "endpoint"
        ])),
        "trajectory_training_action_accuracy": float(np.mean([
            row["action_accuracy"] for row in diagnostics if row["condition"] == "trajectory"
        ])),
        "case_metrics_sha256": hashlib.sha256((args.output_dir / "case_metrics.csv").read_bytes()).hexdigest(),
        "training_history_sha256": hashlib.sha256((args.output_dir / "training_history.csv").read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
