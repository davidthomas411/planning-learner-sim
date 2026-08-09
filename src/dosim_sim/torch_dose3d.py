"""Optional batched PyTorch backend for the implicit 3D dose operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import PlanningPriorities
from .optimizer3d import PlanMetrics3D, evaluate_plan_3d
from .volume3d import SyntheticCase3D

try:
    import torch
except ModuleNotFoundError:  # The CPU reference installation does not require PyTorch.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the GPU backend. Install the 'gpu' optional dependency."
        )


@dataclass(frozen=True)
class TorchOptimizedPlan3D:
    active_beams: tuple[int, ...]
    priorities: PlanningPriorities
    fluence: "torch.Tensor"
    dose: "torch.Tensor"
    metrics: PlanMetrics3D
    iterations: int


class TorchImplicitDoseEngine3D:
    """Batched, differentiable counterpart to ``ImplicitDoseEngine3D``.

    The leading dimensions of ``fluence`` are treated as a batch, so one engine
    can evaluate many candidate fluence states for the same anatomy at once.
    Cases remain independent and can be distributed across separate GPUs.
    """

    def __init__(
        self,
        case: SyntheticCase3D,
        beam_angles_degrees: tuple[float, ...],
        fluence_size: int = 8,
        attenuation: float = 0.32,
        device: str | "torch.device" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> None:
        _require_torch()
        if fluence_size < 2 or fluence_size > 255:
            raise ValueError("fluence_size must be between 2 and 255")
        self.case = case
        self.beam_angles_degrees = tuple(float(value) for value in beam_angles_degrees)
        self.fluence_size = int(fluence_size)
        self.shape = case.body.shape
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32

        body_flat_np = np.flatnonzero(case.body.ravel()).astype(np.int64)
        i, j, k = np.unravel_index(body_flat_np, self.shape)
        x = case.axis[i]
        y = case.axis[j]
        z = case.axis[k]
        maps: list[tuple[np.ndarray, ...]] = []
        for angle_degrees in self.beam_angles_degrees:
            angle = np.deg2rad(angle_degrees)
            lateral = -np.sin(angle) * x + np.cos(angle) * y
            depth = np.cos(angle) * x + np.sin(angle) * y
            u = np.clip((lateral + 1.0) * 0.5 * (fluence_size - 1), 0.0, fluence_size - 1.0)
            v = np.clip((z + 1.0) * 0.5 * (fluence_size - 1), 0.0, fluence_size - 1.0)
            u0 = np.minimum(np.floor(u).astype(np.int16), fluence_size - 2)
            v0 = np.minimum(np.floor(v).astype(np.int16), fluence_size - 2)
            maps.append(
                (
                    u0,
                    v0,
                    (u - u0).astype(np.float32),
                    (v - v0).astype(np.float32),
                    np.exp(-attenuation * (depth + 1.0)).astype(np.float32),
                )
            )

        self.body_flat = torch.as_tensor(body_flat_np, device=self.device)
        self.u0 = torch.as_tensor(np.stack([item[0] for item in maps]), device=self.device)
        self.v0 = torch.as_tensor(np.stack([item[1] for item in maps]), device=self.device)
        self.du = torch.as_tensor(
            np.stack([item[2] for item in maps]), device=self.device, dtype=self.dtype
        )
        self.dv = torch.as_tensor(
            np.stack([item[3] for item in maps]), device=self.device, dtype=self.dtype
        )
        self.attenuation = torch.as_tensor(
            np.stack([item[4] for item in maps]), device=self.device, dtype=self.dtype
        )

    @property
    def n_beams(self) -> int:
        return len(self.beam_angles_degrees)

    @property
    def cache_bytes(self) -> int:
        tensors = (self.body_flat, self.u0, self.v0, self.du, self.dv, self.attenuation)
        return int(sum(value.numel() * value.element_size() for value in tensors))

    def _corner_indices(self) -> tuple["torch.Tensor", ...]:
        u0 = self.u0.to(torch.int64)
        v0 = self.v0.to(torch.int64)
        size = self.fluence_size
        return (
            u0 * size + v0,
            (u0 + 1) * size + v0,
            u0 * size + v0 + 1,
            (u0 + 1) * size + v0 + 1,
        )

    def forward(self, fluence: "torch.Tensor") -> "torch.Tensor":
        expected_tail = (self.n_beams, self.fluence_size, self.fluence_size)
        if tuple(fluence.shape[-3:]) != expected_tail:
            raise ValueError(f"fluence final dimensions must be {expected_tail}")
        if fluence.device != self.device:
            raise ValueError(f"fluence must be on {self.device}, got {fluence.device}")
        batch_shape = tuple(fluence.shape[:-3])
        batch_count = int(np.prod(batch_shape)) if batch_shape else 1
        flat_fluence = fluence.reshape(batch_count, self.n_beams, -1)
        indices = self._corner_indices()
        expanded = tuple(index.unsqueeze(0).expand(batch_count, -1, -1) for index in indices)
        weights = (
            (1.0 - self.du) * (1.0 - self.dv),
            self.du * (1.0 - self.dv),
            (1.0 - self.du) * self.dv,
            self.du * self.dv,
        )
        sampled = torch.zeros(
            (batch_count, self.n_beams, self.body_flat.numel()),
            device=self.device,
            dtype=fluence.dtype,
        )
        for index, weight in zip(expanded, weights, strict=True):
            sampled = sampled + torch.gather(flat_fluence, 2, index) * weight.unsqueeze(0)
        body_dose = torch.sum(sampled * self.attenuation.unsqueeze(0), dim=1)
        flat_dose = torch.zeros(
            (batch_count, int(np.prod(self.shape))), device=self.device, dtype=fluence.dtype
        )
        flat_dose = flat_dose.scatter(
            1, self.body_flat.unsqueeze(0).expand(batch_count, -1), body_dose
        )
        return flat_dose.reshape(*batch_shape, *self.shape)

    def adjoint(self, voxel_values: "torch.Tensor") -> "torch.Tensor":
        """Apply the exact transpose; useful for parity and solver diagnostics."""

        expected_tail = self.shape
        if tuple(voxel_values.shape[-3:]) != expected_tail:
            raise ValueError(f"voxel_values final dimensions must be {expected_tail}")
        batch_shape = tuple(voxel_values.shape[:-3])
        batch_count = int(np.prod(batch_shape)) if batch_shape else 1
        body_values = voxel_values.reshape(batch_count, -1).index_select(1, self.body_flat)
        weighted_values = body_values.unsqueeze(1) * self.attenuation.unsqueeze(0)
        indices = self._corner_indices()
        weights = (
            (1.0 - self.du) * (1.0 - self.dv),
            self.du * (1.0 - self.dv),
            (1.0 - self.du) * self.dv,
            self.du * self.dv,
        )
        result = torch.zeros(
            (batch_count, self.n_beams, self.fluence_size**2),
            device=self.device,
            dtype=voxel_values.dtype,
        )
        for index, weight in zip(indices, weights, strict=True):
            result.scatter_add_(
                2,
                index.unsqueeze(0).expand(batch_count, -1, -1),
                weighted_values * weight.unsqueeze(0),
            )
        return result.reshape(*batch_shape, self.n_beams, self.fluence_size, self.fluence_size)


def _torch_loss(
    case: SyntheticCase3D,
    dose: "torch.Tensor",
    priorities: PlanningPriorities,
) -> "torch.Tensor":
    device = dose.device
    target = torch.as_tensor(case.target, device=device)
    target_values = dose[target]
    target_under = torch.relu(1.0 - target_values)
    loss = priorities.target * 20.0 * torch.mean(target_under.square())
    loss = loss + priorities.hotspot * 5.0 * torch.mean(torch.relu(target_values - 1.10).square())
    for mask, limit, priority in zip(case.oars, case.oar_limits, priorities.oars, strict=True):
        values = dose[torch.as_tensor(mask, device=device)]
        loss = loss + priority * 5.0 * torch.mean(torch.relu(values - limit).square())
    return loss


def optimize_fluence_3d_torch(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    active_beams: tuple[int, ...],
    priorities: PlanningPriorities,
    iterations: int = 60,
    learning_rate: float = 0.08,
    initial_fluence: "torch.Tensor | None" = None,
) -> TorchOptimizedPlan3D:
    """GPU-capable inner optimizer for fixed human-selected settings."""

    _require_torch()
    active = torch.zeros(engine.n_beams, device=engine.device, dtype=torch.bool)
    active[list(active_beams)] = True
    if not bool(torch.any(active)):
        raise ValueError("At least one beam must be active")
    shape = (engine.n_beams, engine.fluence_size, engine.fluence_size)
    if initial_fluence is None:
        fluence = torch.full(
            shape,
            0.20 / len(active_beams),
            device=engine.device,
            dtype=engine.dtype,
        )
    else:
        fluence = initial_fluence.detach().to(device=engine.device, dtype=engine.dtype).clone()
    fluence[~active] = 0.0
    fluence.requires_grad_(True)
    optimizer = torch.optim.Adam([fluence], lr=learning_rate)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        dose = engine.forward(fluence)
        loss = _torch_loss(case, dose, priorities) + 0.0005 * torch.mean(fluence.square())
        loss.backward()
        if fluence.grad is not None:
            fluence.grad[~active] = 0.0
        optimizer.step()
        with torch.no_grad():
            fluence.clamp_(min=0.0)
            fluence[~active] = 0.0
    with torch.no_grad():
        dose = engine.forward(fluence)
        final_loss = float(_torch_loss(case, dose, priorities).item())
        dose_numpy = dose.detach().float().cpu().numpy()
        metrics = evaluate_plan_3d(case, dose_numpy, final_loss)
    return TorchOptimizedPlan3D(
        active_beams=tuple(sorted(active_beams)),
        priorities=priorities,
        fluence=fluence.detach(),
        dose=dose.detach(),
        metrics=metrics,
        iterations=iterations,
    )
