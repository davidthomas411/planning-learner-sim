import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dosim_sim.dataset3d import ACTION_NAMES


class MatchedPilotNet(nn.Module):
    def __init__(self, feature_count: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.endpoint_head = nn.Linear(hidden, 17)
        self.action_head = nn.Linear(hidden, len(ACTION_NAMES))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.trunk(features)
        return self.endpoint_head(encoded), self.action_head(encoded)


@dataclass
class FitResult:
    condition: str
    seed: int
    endpoint_losses: list[float]
    action_losses: list[float]
    train_setting_mae: float
    train_beam_accuracy: float
    train_priority_mae: float
    train_action_accuracy: float | None
    test_setting_mae: float | None


def endpoint_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    beam = nn.functional.binary_cross_entropy_with_logits(prediction[:, :12], target[:, :12])
    priorities = nn.functional.mse_loss(torch.sigmoid(prediction[:, 12:]), target[:, 12:])
    return beam + priorities


def setting_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    beams = (torch.sigmoid(prediction[:, :12]) >= 0.5).float()
    beam_accuracy = float((beams == target[:, :12]).float().mean().item())
    priorities = torch.sigmoid(prediction[:, 12:])
    priority_mae = float(torch.mean(torch.abs(priorities - target[:, 12:])).item())
    combined = torch.cat([beams, priorities], dim=1)
    setting_mae = float(torch.mean(torch.abs(combined - target)).item())
    return setting_mae, beam_accuracy, priority_mae


def fit(
    condition: str,
    seed: int,
    endpoint_x: torch.Tensor,
    endpoint_y: torch.Tensor,
    action_x: torch.Tensor,
    action_y: torch.Tensor,
    epochs: int,
    learning_rate: float,
    action_weight: float,
    test_x: torch.Tensor | None = None,
    test_y: torch.Tensor | None = None,
) -> tuple[MatchedPilotNet, FitResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MatchedPilotNet(endpoint_x.shape[1]).to(endpoint_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    endpoint_losses: list[float] = []
    action_losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        endpoint_prediction, _ = model(endpoint_x)
        final_loss = endpoint_loss(endpoint_prediction, endpoint_y)
        if condition == "trajectory":
            _, action_prediction = model(action_x)
            intermediate_loss = nn.functional.cross_entropy(action_prediction, action_y)
            loss = final_loss + action_weight * intermediate_loss
        else:
            intermediate_loss = torch.zeros((), device=endpoint_x.device)
            loss = final_loss
        loss.backward()
        optimizer.step()
        endpoint_losses.append(float(final_loss.detach().item()))
        action_losses.append(float(intermediate_loss.detach().item()))
    with torch.no_grad():
        endpoint_prediction, _ = model(endpoint_x)
        setting_mae, beam_accuracy, priority_mae = setting_metrics(endpoint_prediction, endpoint_y)
        action_accuracy = None
        if condition == "trajectory":
            _, action_prediction = model(action_x)
            action_accuracy = float((action_prediction.argmax(1) == action_y).float().mean().item())
        test_setting_mae = None
        if test_x is not None and test_y is not None:
            test_prediction, _ = model(test_x)
            test_setting_mae = setting_metrics(test_prediction, test_y)[0]
    return model, FitResult(
        condition=condition,
        seed=seed,
        endpoint_losses=endpoint_losses,
        action_losses=action_losses,
        train_setting_mae=setting_mae,
        train_beam_accuracy=beam_accuracy,
        train_priority_mae=priority_mae,
        train_action_accuracy=action_accuracy,
        test_setting_mae=test_setting_mae,
    )


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tensors(records: list[dict], device: torch.device) -> tuple[torch.Tensor, ...]:
    endpoint_x = torch.tensor([row["initial_features"] for row in records], dtype=torch.float32, device=device)
    endpoint_y = torch.tensor([row["final_settings"] for row in records], dtype=torch.float32, device=device)
    transitions = [transition for row in records for transition in row["trajectory"]]
    action_x = torch.tensor([row["state"] for row in transitions], dtype=torch.float32, device=device)
    action_y = torch.tensor([row["action_index"] for row in transitions], dtype=torch.long, device=device)
    return endpoint_x, endpoint_y, action_x, action_y


def save_plot(overfit: list[FitResult], heldout: list[FitResult], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for result in overfit:
        axes[0].plot(result.endpoint_losses, color="#4E79A7" if result.condition == "endpoint" else "#E15759", alpha=0.55)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Matched optimizer updates")
    axes[0].set_ylabel("Terminal-setting loss")
    axes[0].set_title("32-case memorization check")
    conditions = ("endpoint", "trajectory")
    x = np.arange(2)
    means = [np.mean([row.train_setting_mae for row in overfit if row.condition == value]) for value in conditions]
    axes[1].bar(x, means, color=["#4E79A7", "#E15759"])
    axes[1].set_xticks(x, ["endpoint only", "trajectory supervised"])
    axes[1].set_ylabel("Training final-setting MAE")
    axes[1].set_title("Identical terminal targets")
    for index, condition in enumerate(conditions):
        values = [row.test_setting_mae for row in heldout if row.condition == condition]
        axes[2].scatter(np.full(len(values), index), values, s=60, color=["#4E79A7", "#E15759"][index])
        axes[2].plot([index - 0.12, index + 0.12], [np.mean(values), np.mean(values)], color="black")
    axes[2].set_xticks(x, ["endpoint only", "trajectory supervised"])
    axes[2].set_ylabel("Held-out final-setting MAE")
    axes[2].set_title("Eight-case development split")
    fig.suptitle(
        "Matched-network learner pilot\nEndpoint targets are identical; only one condition receives intermediate action labels",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched endpoint/trajectory learner pipeline pilot")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/3d_dataset_pilot"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--action-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3d_learner_pilot"))
    args = parser.parse_args()
    records = load_records(args.dataset_dir / "trajectory_view.jsonl")
    if len(records) < 32:
        raise ValueError("At least 32 retained records are required")
    records = records[:32]
    device = torch.device(args.device)
    permutation = np.random.default_rng(20260809).permutation(len(records))
    train_records = [records[index] for index in permutation[:24]]
    test_records = [records[index] for index in permutation[24:32]]
    full_tensors = tensors(records, device)
    train_tensors = tensors(train_records, device)
    test_endpoint_x, test_endpoint_y, _, _ = tensors(test_records, device)
    overfit: list[FitResult] = []
    heldout: list[FitResult] = []
    parameter_counts = set()
    for seed in range(args.seeds):
        for condition in ("endpoint", "trajectory"):
            model, result = fit(
                condition,
                seed,
                *full_tensors,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                action_weight=args.action_weight,
            )
            parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
            overfit.append(result)
            _, heldout_result = fit(
                condition,
                seed,
                *train_tensors,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                action_weight=args.action_weight,
                test_x=test_endpoint_x,
                test_y=test_endpoint_y,
            )
            heldout.append(heldout_result)
            print(f"seed={seed} condition={condition} overfit_mae={result.train_setting_mae:.6f} heldout_mae={heldout_result.test_setting_mae:.6f}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plot(overfit, heldout, args.output_dir / "01_learner_pilot.png")
    result_rows = []
    for phase, rows in (("overfit32", overfit), ("train24_test8", heldout)):
        for row in rows:
            result_rows.append({
                "phase": phase,
                "condition": row.condition,
                "seed": row.seed,
                "train_setting_mae": row.train_setting_mae,
                "train_beam_accuracy": row.train_beam_accuracy,
                "train_priority_mae": row.train_priority_mae,
                "train_action_accuracy": row.train_action_accuracy,
                "test_setting_mae": row.test_setting_mae,
            })
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    endpoint_test = np.array([row.test_setting_mae for row in heldout if row.condition == "endpoint"])
    trajectory_test = np.array([row.test_setting_mae for row in heldout if row.condition == "trajectory"])
    summary = {
        "status": "pipeline and memorization pilot; not the primary iterative-policy comparison",
        "same_architecture_and_parameter_count": len(parameter_counts) == 1,
        "parameter_count": next(iter(parameter_counts)),
        "same_optimizer_updates": True,
        "cases": 32,
        "training_cases_for_heldout_check": 24,
        "heldout_cases": 8,
        "seeds": args.seeds,
        "epochs_per_fit": args.epochs,
        "endpoint_overfit_setting_mae_mean": float(np.mean([row.train_setting_mae for row in overfit if row.condition == "endpoint"])),
        "trajectory_overfit_setting_mae_mean": float(np.mean([row.train_setting_mae for row in overfit if row.condition == "trajectory"])),
        "trajectory_action_accuracy_mean": float(np.mean([row.train_action_accuracy for row in overfit if row.condition == "trajectory"])),
        "endpoint_heldout_setting_mae_mean": float(endpoint_test.mean()),
        "trajectory_heldout_setting_mae_mean": float(trajectory_test.mean()),
        "paired_heldout_difference_trajectory_minus_endpoint": (trajectory_test - endpoint_test).tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
