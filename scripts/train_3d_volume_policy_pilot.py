import argparse
import csv
import hashlib
import json
import os
import random
import time
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
    optimizer_objective_kwargs_3d,
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

    def __init__(self, scalar_count: int, setting_count: int = 18) -> None:
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
        self.endpoint_head = nn.Linear(128, setting_count)
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


@dataclass
class ProgressReporter:
    output_dir: Path
    total_units: int
    completed_units: int = 0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def report(self, phase: str, detail: str, advance: int = 0) -> None:
        self.completed_units = min(self.total_units, self.completed_units + advance)
        elapsed = time.monotonic() - self.started_at
        fraction = self.completed_units / max(self.total_units, 1)
        remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        width = 30
        filled = min(width, int(round(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        payload = {
            "phase": phase,
            "detail": detail,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "percent_complete": 100.0 * fraction,
            "progress_basis": "completed instrumented work units; units have unequal wall time",
            "elapsed_seconds": elapsed,
            "rough_estimated_remaining_seconds": remaining,
        }
        temporary = self.output_dir / "progress.json.tmp"
        progress_path = self.output_dir / "progress.json"
        progress_text = json.dumps(payload, indent=2) + "\n"
        status_saved = False
        for _ in range(20):
            try:
                temporary.write_text(progress_text, encoding="utf-8")
                temporary.replace(progress_path)
                status_saved = True
                break
            except PermissionError:
                time.sleep(0.025)
        if not status_saved:
            # Status reporting must never stop a scientific run. The append-only
            # log and terminal line remain available if a Windows reader holds
            # the JSON file longer than the retry window.
            try:
                progress_path.write_text(progress_text, encoding="utf-8")
            except PermissionError:
                pass
        eta = "unknown" if remaining is None else f"{remaining / 60:.1f} min"
        line = (
            f"[{bar}] {100 * fraction:5.1f}% work units | {phase} | {detail} | "
            f"elapsed {elapsed / 60:.1f} min | rough ETA {eta}"
        )
        with (self.output_dir / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def save_channel_montage(
    volume: torch.Tensor,
    output_path: Path,
    channel_names: tuple[str, ...] = VOLUME_CHANNEL_NAMES,
) -> None:
    """Save the central axial slice of each model input channel."""

    data = volume.detach().float().cpu().numpy()
    axial_index = int(np.argmax(data[1].sum(axis=(0, 1))))
    columns = 4
    rows = int(np.ceil(len(channel_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), constrained_layout=True)
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


def save_review_plan(
    case,
    steps: list[PlanningStep3D],
    condition: str,
    config: HighLevelSearchConfig3D,
    output_base: Path,
) -> None:
    """Save a human-readable plan audit with dose images and all manual actions."""

    axial_index = int(np.argmax(case.target.sum(axis=(0, 1))))
    initial_dose = steps[0].plan.dose.detach().float().cpu().numpy()
    final_dose = steps[-1].plan.dose.detach().float().cpu().numpy()
    maximum = max(float(initial_dose.max()), float(final_dose.max()), 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for axis, dose, title in zip(
        axes[:2], (initial_dose, final_dose), ("Initial optimized plan", "Final learned-policy plan"), strict=True
    ):
        image = axis.imshow(dose[:, :, axial_index].T, origin="lower", cmap="turbo", vmin=0.0, vmax=maximum)
        axis.contour(case.target[:, :, axial_index].T, levels=[0.5], colors=["white"], linewidths=1.5)
        if case.clinical_target is not None:
            axis.contour(
                case.clinical_target[:, :, axial_index].T,
                levels=[0.5],
                colors=["#ff66cc"],
                linewidths=1.0,
            )
        for color, oar in zip(("cyan", "lime", "magenta"), case.oars, strict=True):
            axis.contour(oar[:, :, axial_index].T, levels=[0.5], colors=[color], linewidths=1.0)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Relative dose")

    action_lines = []
    for step in steps[1:]:
        action_lines.append(f"{step.step}. {step.action.description if step.action else 'Stop'}")
    if not action_lines:
        action_lines = ["No change: initial plan met all rules"]
    initial_metrics = steps[0].plan.metrics
    final_metrics = steps[-1].plan.metrics
    names = getattr(case, "structure_names", ("OAR 0", "OAR 1", "OAR 2"))
    metric_lines = [
        f"Case: {case.case_id} ({case.difficulty})",
        f"Condition: {condition}",
        f"Acceptable: {is_acceptable_3d(final_metrics, case, config)}",
        f"Violation: {steps[0].violation_score:.3f} -> {steps[-1].violation_score:.3f}",
        f"Fields: {initial_metrics.field_count} -> {final_metrics.field_count} (min {config.minimum_field_count})",
        f"PTV D95: {initial_metrics.target_d95:.3f} -> {final_metrics.target_d95:.3f} (min {config.d95_min:.2f})",
        f"PTV D02: {initial_metrics.target_d02:.3f} -> {final_metrics.target_d02:.3f} (max {config.d02_max:.2f})",
        f"Paddick CI95: {initial_metrics.paddick_ci_95:.3f} -> {final_metrics.paddick_ci_95:.3f} (min {config.paddick_ci_95_min:.2f})",
        f"R50: {initial_metrics.r50:.2f} -> {final_metrics.r50:.2f} (max {config.r50_max:.1f})",
    ]
    metric_lines.extend(
        f"{name} mean: {initial_metrics.oar_mean[index]:.3f} -> {final_metrics.oar_mean[index]:.3f} "
        f"(max {case.oar_limits[index]:.2f})"
        for index, name in enumerate(names)
    )
    metric_lines.extend(["", "Manual-level actions:", *action_lines])
    axes[2].axis("off")
    axes[2].text(0.0, 1.0, "\n".join(metric_lines), va="top", ha="left", fontsize=10)
    figure.suptitle("Review plan: anatomy contours, dose, metrics, and actions")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=180)
    plt.close(figure)

    audit = {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "condition": condition,
        "axial_slice": axial_index,
        "acceptable": is_acceptable_3d(final_metrics, case, config),
        "steps": [
            {
                "step": step.step,
                "action": None if step.action is None else step.action.description,
                "active_beam_indices": list(step.plan.active_beams),
                "active_beam_angles_degrees": [ANGLES[index] for index in step.plan.active_beams],
                "priorities": {
                    "target": step.plan.priorities.target,
                    "hotspot": step.plan.priorities.hotspot,
                    "oars": list(step.plan.priorities.oars),
                    "normal_tissue": step.plan.priorities.normal_tissue,
                },
                "metrics": {
                    "target_d95": step.plan.metrics.target_d95,
                    "target_d02": step.plan.metrics.target_d02,
                    "oar_mean": list(step.plan.metrics.oar_mean),
                    "target_v95": step.plan.metrics.target_v95,
                    "paddick_ci_95": step.plan.metrics.paddick_ci_95,
                    "r50": step.plan.metrics.r50,
                    "body_mean_dose": step.plan.metrics.body_mean_dose,
                    "field_count": step.plan.metrics.field_count,
                    "violation_score": step.violation_score,
                },
            }
            for step in steps
        ],
    }
    output_base.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


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
    progress: ProgressReporter | None = None,
) -> tuple[torch.Tensor, ...]:
    endpoint_scalars = []
    endpoint_volumes = []
    endpoint_targets = []
    action_scalars = []
    action_volumes = []
    action_targets = []
    replay_differences = []
    for record_index, (record, case, engine) in enumerate(zip(records, cases, engines, strict=True), start=1):
        tier = attempt_by_case[record["case_id"]]["search_tier"]
        iterations = deep_iterations if tier == "deep" else shallow_iterations
        replay_config = HighLevelSearchConfig3D(
            max_steps=10,
            optimizer_iterations=iterations,
            priority_ceiling=25.0 if tier == "deep" else 6.0,
            initial_field_count=rollout_config.initial_field_count,
            normal_tissue_weight=rollout_config.normal_tissue_weight,
            normal_tissue_threshold=rollout_config.normal_tissue_threshold,
            integral_dose_weight=rollout_config.integral_dose_weight,
            clinical_dvh_weight=rollout_config.clinical_dvh_weight,
            target_hotspot_threshold=rollout_config.target_hotspot_threshold,
            target_hotspot_weight=rollout_config.target_hotspot_weight,
            high_dose_normal_tissue_weight=rollout_config.high_dose_normal_tissue_weight,
            high_dose_normal_tissue_threshold=rollout_config.high_dose_normal_tissue_threshold,
            target_normalization_d98=rollout_config.target_normalization_d98,
            target_normalization_d50=rollout_config.target_normalization_d50,
            target_normalization_interval=rollout_config.target_normalization_interval,
            prostate_protocol_tier=rollout_config.prostate_protocol_tier,
            d95_min=rollout_config.d95_min,
            d98_min=rollout_config.d98_min,
            d50_min=rollout_config.d50_min,
            d50_max=rollout_config.d50_max,
            d02_max=rollout_config.d02_max,
            paddick_ci_95_min=rollout_config.paddick_ci_95_min,
            covering_isodose_ratio_95_max=rollout_config.covering_isodose_ratio_95_max,
            r50_max=rollout_config.r50_max,
            minimum_field_count=rollout_config.minimum_field_count,
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
                **optimizer_objective_kwargs_3d(replay_config),
            )
            current = PlanningStep3D(
                current.step + 1,
                action,
                plan,
                clinical_violation_score_3d(plan.metrics, case, replay_config),
            )
        if progress is not None:
            progress.report("replay demonstrations", f"case {record_index}/{len(records)}", advance=1)
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
            case,
            engine,
            beams,
            priorities,
            config.optimizer_iterations,
            initial_fluence=current.plan.fluence,
            **optimizer_objective_kwargs_3d(config),
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
    progress: ProgressReporter | None = None,
) -> tuple[MatchedVolumePolicyNet, list[dict], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    endpoint_x, endpoint_volume, endpoint_y, action_x, action_volume, action_y, _ = tensors
    model = MatchedVolumePolicyNet(endpoint_x.shape[1], endpoint_y.shape[1]).to(endpoint_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for pretrain_index in range(1, pretrain_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x, endpoint_volume)
        loss = endpoint_loss(endpoint_prediction, endpoint_y)
        if condition == "trajectory":
            _, action_logits = model(action_x, action_volume)
            loss = loss + action_weight * nn.functional.cross_entropy(action_logits, action_y)
        loss.backward()
        optimizer.step()
        if progress is not None:
            progress.report(
                f"{condition} pretraining",
                f"update {pretrain_index}/{pretrain_updates}; loss {float(loss.detach().item()):.4f}",
                advance=1,
            )
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
        if progress is not None:
            progress.report(
                f"{condition} closed-loop training",
                f"update {update_index}/{updates}; acceptable {history[-1]['acceptable_rate']:.2f}; "
                f"violation {history[-1]['mean_violation']:.3f}",
                advance=1,
            )
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
    progress: ProgressReporter | None = None,
    review_dir: Path | None = None,
) -> list[dict]:
    model.eval()
    rows = []
    reviewed_difficulties: set[str] = set()
    for record_index, record in enumerate(records, start=1):
        generator = generate_prostate_case_3d if anatomy == "prostate" else generate_case_3d
        case = generator(int(record["seed"]), grid_size, difficulty=record["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, ANGLES, fluence_size, device=device, dtype=dtype)
        current = initial_policy_step_3d(case, engine, config)
        steps = [current]
        selected_actions = []
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
            selected_actions.append(ACTION_NAMES[action_index])
            action, beams, priorities = action_settings_3d(action_index, current, config)
            if action is None:
                break
            plan = optimize_fluence_3d_torch(
                case,
                engine,
                beams,
                priorities,
                config.optimizer_iterations,
                initial_fluence=current.plan.fluence,
                **optimizer_objective_kwargs_3d(config),
            )
            current = PlanningStep3D(
                step_index, action, plan, clinical_violation_score_3d(plan.metrics, case, config)
            )
            steps.append(current)
        final_metrics = current.plan.metrics
        row = {
            "training_seed": seed,
            "condition": condition,
            "case_id": case.case_id,
            "case_seed": case.seed,
            "difficulty": case.difficulty,
            "acceptable": is_acceptable_3d(current.plan.metrics, case, config),
            "violation_score": current.violation_score,
            "high_level_actions": len(steps) - 1,
            "action_sequence": "|".join(selected_actions),
            "stopping_reason": "acceptable" if is_acceptable_3d(current.plan.metrics, case, config) else "policy_step_limit",
            "target_d95_initial": steps[0].plan.metrics.target_d95,
            "target_d95_final": final_metrics.target_d95,
            "target_d02_initial": steps[0].plan.metrics.target_d02,
            "target_d02_final": final_metrics.target_d02,
            "paddick_ci_95_initial": steps[0].plan.metrics.paddick_ci_95,
            "paddick_ci_95_final": final_metrics.paddick_ci_95,
            "r50_initial": steps[0].plan.metrics.r50,
            "r50_final": final_metrics.r50,
            "field_count_initial": steps[0].plan.metrics.field_count,
            "field_count_final": final_metrics.field_count,
            "oar_0_mean_initial": steps[0].plan.metrics.oar_mean[0],
            "oar_0_mean_final": final_metrics.oar_mean[0],
            "oar_1_mean_initial": steps[0].plan.metrics.oar_mean[1],
            "oar_1_mean_final": final_metrics.oar_mean[1],
            "oar_2_mean_initial": steps[0].plan.metrics.oar_mean[2],
            "oar_2_mean_final": final_metrics.oar_mean[2],
            "final_beam_angles_degrees": "|".join(str(int(ANGLES[index])) for index in current.plan.active_beams),
            "final_target_priority": current.plan.priorities.target,
            "final_hotspot_priority": current.plan.priorities.hotspot,
            "final_oar_priorities": "|".join(str(value) for value in current.plan.priorities.oars),
        }
        rows.append(row)
        if review_dir is not None and case.difficulty not in reviewed_difficulties:
            save_review_plan(
                case,
                steps,
                condition,
                config,
                review_dir / f"{condition}_{case.difficulty}_{case.case_id}",
            )
            reviewed_difficulties.add(case.difficulty)
        if progress is not None:
            progress.report(
                f"{condition} validation",
                f"case {record_index}/{len(records)}; {case.difficulty}; "
                f"acceptable {row['acceptable']}; violation {row['violation_score']:.3f}",
                advance=1,
            )
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
    parser.add_argument("--initial-field-count", type=int, default=4)
    parser.add_argument("--normal-tissue-weight", type=float, default=0.0)
    parser.add_argument("--normal-tissue-threshold", type=float, default=0.5)
    parser.add_argument("--integral-dose-weight", type=float, default=0.0)
    parser.add_argument("--clinical-dvh-weight", type=float, default=0.0)
    parser.add_argument("--target-hotspot-threshold", type=float, default=1.10)
    parser.add_argument("--target-hotspot-weight", type=float, default=5.0)
    parser.add_argument("--high-dose-normal-tissue-weight", type=float, default=0.0)
    parser.add_argument("--high-dose-normal-tissue-threshold", type=float, default=0.95)
    parser.add_argument("--target-normalization-d98", type=float)
    parser.add_argument("--target-normalization-d50", type=float)
    parser.add_argument("--target-normalization-interval", type=int, default=0)
    parser.add_argument(
        "--prostate-protocol-tier",
        choices=(
            "off",
            "per_protocol",
            "variation_acceptable",
            "oar_per_protocol",
            "oar_variation_acceptable",
        ),
        default="off",
    )
    parser.add_argument("--d95-min", type=float, default=0.85)
    parser.add_argument("--d98-min", type=float, default=0.0)
    parser.add_argument("--d50-min", type=float, default=0.0)
    parser.add_argument("--d50-max", type=float, default=float("inf"))
    parser.add_argument("--d02-max", type=float, default=1.25)
    parser.add_argument("--paddick-ci-95-min", type=float, default=0.0)
    parser.add_argument("--covering-isodose-ratio-95-max", type=float, default=float("inf"))
    parser.add_argument("--r50-max", type=float, default=float("inf"))
    parser.add_argument("--minimum-field-count", type=int, default=0)
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
    torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, args.dtype)
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    train_records, validation_records = select_records(records, args.train_cases, args.heldout_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress.log").write_text("", encoding="utf-8")
    total_units = (
        2 * len(train_records)
        + args.seeds * 2 * (args.pretrain_updates + args.updates + len(validation_records))
    )
    progress = ProgressReporter(args.output_dir, total_units)
    progress.report("setup", f"device {device}; {len(train_records)} training and {len(validation_records)} validation cases")
    attempt_rows = load_records(args.dataset_dir / "attempt_manifest.jsonl")
    attempt_by_case = {row["case_id"]: row for row in attempt_rows}
    config = HighLevelSearchConfig3D(
        max_steps=args.max_steps,
        optimizer_iterations=args.iterations,
        priority_ceiling=25.0,
        initial_field_count=args.initial_field_count,
        normal_tissue_weight=args.normal_tissue_weight,
        normal_tissue_threshold=args.normal_tissue_threshold,
        integral_dose_weight=args.integral_dose_weight,
        clinical_dvh_weight=args.clinical_dvh_weight,
        target_hotspot_threshold=args.target_hotspot_threshold,
        target_hotspot_weight=args.target_hotspot_weight,
        high_dose_normal_tissue_weight=args.high_dose_normal_tissue_weight,
        high_dose_normal_tissue_threshold=args.high_dose_normal_tissue_threshold,
        target_normalization_d98=args.target_normalization_d98,
        target_normalization_d50=args.target_normalization_d50,
        target_normalization_interval=args.target_normalization_interval,
        prostate_protocol_tier=args.prostate_protocol_tier,
        d95_min=args.d95_min,
        d98_min=args.d98_min,
        d50_min=args.d50_min,
        d50_max=args.d50_max,
        d02_max=args.d02_max,
        paddick_ci_95_min=args.paddick_ci_95_min,
        covering_isodose_ratio_95_max=args.covering_isodose_ratio_95_max,
        r50_max=args.r50_max,
        minimum_field_count=args.minimum_field_count,
    )
    generator = generate_prostate_case_3d if args.anatomy == "prostate" else generate_case_3d
    cases = []
    engines = []
    initials = []
    for case_index, row in enumerate(train_records, start=1):
        case = generator(int(row["seed"]), args.grid_size, difficulty=row["difficulty"])
        engine = TorchImplicitDoseEngine3D(case, ANGLES, args.fluence_size, device=device, dtype=dtype)
        initial = initial_policy_step_3d(case, engine, config)
        cases.append(case)
        engines.append(engine)
        initials.append(initial)
        progress.report("create initial plans", f"case {case_index}/{len(train_records)}", advance=1)
    training_tensors = replay_training_tensors(
        train_records,
        cases,
        engines,
        attempt_by_case,
        config,
        args.volume_size,
        args.shallow_replay_iterations,
        args.deep_replay_iterations,
        progress,
    )
    channel_names = VOLUME_CHANNEL_NAMES
    if args.anatomy == "prostate":
        channel_names = (
            "body", "prostate_ptv", "bladder", "rectum", "femoral_heads", "dose",
            "target_underdose", "target_hotspot", "normal_tissue_high_dose",
            "bladder_excess", "rectum_excess", "femoral_head_excess",
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
                progress,
            )
            parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
            torch.save(
                {"condition": condition, "seed": seed, "model_state_dict": model.state_dict()},
                args.output_dir / f"model_{condition}_seed{seed}.pt",
            )
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
                progress,
                args.output_dir / "review_plans" / f"seed_{seed}",
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
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress.report("complete", "all matched conditions and validation plans saved")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
