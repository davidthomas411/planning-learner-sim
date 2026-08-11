"""Optional batched PyTorch backend for the implicit 3D dose operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import PlanningPriorities
from .prostate_protocol import PROSTATE_60GY_20FX_OAR_GOALS, prostate_evaluation_masks
from .optimizer3d import (
    OptimizationTarget3D,
    PlanMetrics3D,
    evaluate_plan_3d,
    full_optimization_target_3d,
)
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
    optimization_target: OptimizationTarget3D | None = None


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
        self.target_flat = torch.as_tensor(
            np.flatnonzero(case.target.ravel()).astype(np.int64), device=self.device
        )
        self.clinical_target_flat = torch.as_tensor(
            np.flatnonzero(
                (
                    np.zeros_like(case.target, dtype=bool)
                    if case.clinical_target is None
                    else case.clinical_target
                ).ravel()
            ).astype(np.int64),
            device=self.device,
        )
        self.normal_tissue_flat = torch.as_tensor(
            np.flatnonzero((case.body & ~case.target).ravel()).astype(np.int64), device=self.device
        )
        self.oar_flat = tuple(
            torch.as_tensor(np.flatnonzero(mask.ravel()).astype(np.int64), device=self.device)
            for mask in case.oars
        )
        evaluation_masks = (
            prostate_evaluation_masks(case)
            if case.anatomy in {"prostate", "tcia_prostate"}
            else {}
        )
        self.protocol_goal_flat = tuple(
            torch.as_tensor(
                np.flatnonzero(evaluation_masks[goal.structure].ravel()).astype(np.int64),
                device=self.device,
            )
            for goal in PROSTATE_60GY_20FX_OAR_GOALS
        ) if evaluation_masks else ()
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
        tensors = (
            self.body_flat,
            self.target_flat,
            self.clinical_target_flat,
            self.normal_tissue_flat,
            *self.oar_flat,
            *self.protocol_goal_flat,
            self.u0,
            self.v0,
            self.du,
            self.dv,
            self.attenuation,
        )
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
    engine: TorchImplicitDoseEngine3D,
    dose: "torch.Tensor",
    priorities: PlanningPriorities,
    target_hotspot_threshold: float = 1.10,
    target_hotspot_weight: float = 5.0,
    normal_tissue_weight: float = 0.0,
    normal_tissue_threshold: float = 0.5,
    integral_dose_weight: float = 0.0,
    high_dose_normal_tissue_weight: float = 0.0,
    high_dose_normal_tissue_threshold: float = 0.95,
    clinical_dvh_weight: float = 0.0,
    optimization_target_flat: "torch.Tensor | None" = None,
    relaxed_overlap_flat: "torch.Tensor | None" = None,
    relaxed_overlap_minimum: float = 1.0,
) -> "torch.Tensor":
    case = engine.case
    flat_dose = dose.reshape(-1)
    coverage_flat = (
        engine.target_flat if optimization_target_flat is None else optimization_target_flat
    )
    target_values = flat_dose.index_select(0, coverage_flat)
    target_under = torch.relu(1.0 - target_values)
    loss = priorities.target * 20.0 * torch.mean(target_under.square())
    tail_count = max(1, int(np.ceil(0.10 * target_values.numel())))
    tail_under = torch.topk(target_under, k=tail_count, largest=True).values
    loss = loss + priorities.target * 30.0 * torch.mean(tail_under.square())

    if relaxed_overlap_flat is not None and relaxed_overlap_flat.numel() > 0:
        relaxed_values = flat_dose.index_select(0, relaxed_overlap_flat)
        relaxed_under = torch.relu(relaxed_overlap_minimum - relaxed_values)
        loss = loss + priorities.target * 20.0 * torch.mean(relaxed_under.square())
        relaxed_tail_count = max(1, int(np.ceil(0.02 * relaxed_values.numel())))
        relaxed_tail = torch.topk(
            relaxed_under,
            k=relaxed_tail_count,
            largest=True,
        ).values
        loss = loss + priorities.target * 30.0 * torch.mean(relaxed_tail.square())

    original_target_values = flat_dose.index_select(0, engine.target_flat)
    if case.anatomy in {"prostate", "tcia_prostate"}:
        if case.voxel_volume_cc is None:
            raise ValueError("physical voxel volume is required for PTV D1cc")
        # Permit less than 1 cc above the limit. Penalize every other PTV
        # voxel above the limit and scale the loss by the physical 1-cc voxel
        # count so that a small hot region is not diluted by the full PTV.
        hot_count = min(
            original_target_values.numel(),
            max(1, int(np.ceil(1.0 / case.voxel_volume_cc))),
        )
        allowed_hot = min(original_target_values.numel() - 1, hot_count - 1)
        hot_values = torch.topk(
            original_target_values,
            k=original_target_values.numel() - allowed_hot,
            largest=False,
        ).values
        hot_loss = torch.sum(
            torch.relu(hot_values - target_hotspot_threshold).square()
        ) / hot_count
    else:
        hot_count = max(1, int(np.ceil(0.02 * original_target_values.numel())))
        hot_values = torch.topk(original_target_values, k=hot_count, largest=True).values
        hot_loss = torch.mean(torch.relu(hot_values - target_hotspot_threshold).square())
    loss = loss + priorities.hotspot * target_hotspot_weight * hot_loss
    for indices, limit, priority in zip(
        engine.oar_flat, case.oar_limits, priorities.oars, strict=True
    ):
        values = flat_dose.index_select(0, indices)
        loss = loss + priority * 5.0 * torch.mean(torch.relu(values - limit).square())
    if clinical_dvh_weight > 0.0 and case.anatomy in {"prostate", "tcia_prostate"}:
        if case.voxel_volume_cc is None or engine.clinical_target_flat.numel() == 0:
            raise ValueError("clinical target and physical voxel volume are required")

        # Prostate V60 Gy >= 99%.
        prostate_values = flat_dose.index_select(0, engine.clinical_target_flat)
        prostate_cold = int(np.floor(0.01 * prostate_values.numel()))
        prostate_count = max(1, prostate_values.numel() - prostate_cold)
        prostate_constrained = torch.topk(
            prostate_values,
            k=prostate_count,
            largest=True,
        ).values
        loss = loss + (
            clinical_dvh_weight
            * priorities.target
            * 20.0
            * torch.mean(torch.relu(1.0 - prostate_constrained).square())
        )

        # A manually created PTV-minus-OAR target activates the trial-informed
        # acceptable variation of PTV V57 Gy >=95%. The unmodified plan keeps
        # the standard PTV V57 Gy >=99% objective.
        ptv_cold_fraction = (
            0.05
            if relaxed_overlap_flat is not None and relaxed_overlap_flat.numel() > 0
            else 0.01
        )
        ptv_cold = int(np.floor(ptv_cold_fraction * original_target_values.numel()))
        ptv_count = max(1, original_target_values.numel() - ptv_cold)
        ptv_constrained = torch.topk(
            original_target_values,
            k=ptv_count,
            largest=True,
        ).values
        ptv_coefficient = 100.0 if ptv_cold_fraction == 0.05 else 20.0
        loss = loss + (
            clinical_dvh_weight
            * priorities.target
            * ptv_coefficient
            * torch.mean(torch.relu(0.95 - ptv_constrained).square())
        )

        name_to_index = {name: index for index, name in enumerate(case.structure_names)}
        for goal, goal_indices in zip(
            PROSTATE_60GY_20FX_OAR_GOALS,
            engine.protocol_goal_flat,
            strict=True,
        ):
            structure_index = name_to_index[goal.planning_structure_name]
            values = flat_dose.index_select(0, goal_indices)
            allowed = int(np.floor(goal.per_protocol_volume_percent * values.numel() / 100.0))
            constrained_count = max(1, values.numel() - allowed)
            constrained = torch.topk(values, k=constrained_count, largest=False).values
            loss = loss + (
                clinical_dvh_weight
                * priorities.oars[structure_index]
                * 2.0
                * torch.mean(torch.relu(constrained - goal.relative_dose).square())
            )
    if (
        normal_tissue_weight > 0.0
        or integral_dose_weight > 0.0
        or high_dose_normal_tissue_weight > 0.0
    ):
        normal_values = flat_dose.index_select(0, engine.normal_tissue_flat)
        if normal_tissue_weight > 0.0:
            loss = loss + priorities.normal_tissue * normal_tissue_weight * torch.mean(
                torch.relu(normal_values - normal_tissue_threshold).square()
            )
        if integral_dose_weight > 0.0:
            loss = loss + priorities.normal_tissue * integral_dose_weight * torch.mean(
                normal_values.square()
            )
        if high_dose_normal_tissue_weight > 0.0:
            high_excess = torch.relu(normal_values - high_dose_normal_tissue_threshold)
            loss = loss + (
                priorities.normal_tissue
                * high_dose_normal_tissue_weight
                * torch.sum(high_excess.square())
                / original_target_values.numel()
            )
    return loss


def optimize_fluence_3d_torch(
    case: SyntheticCase3D,
    engine: TorchImplicitDoseEngine3D,
    active_beams: tuple[int, ...],
    priorities: PlanningPriorities,
    iterations: int = 60,
    learning_rate: float = 0.08,
    initial_fluence: "torch.Tensor | None" = None,
    target_hotspot_threshold: float = 1.10,
    target_hotspot_weight: float = 5.0,
    normal_tissue_weight: float = 0.0,
    normal_tissue_threshold: float = 0.5,
    integral_dose_weight: float = 0.0,
    high_dose_normal_tissue_weight: float = 0.0,
    high_dose_normal_tissue_threshold: float = 0.95,
    clinical_dvh_weight: float = 0.0,
    target_normalization_d98: float | None = None,
    target_normalization_d50: float | None = None,
    clinical_target_normalization_d99: float | None = None,
    target_normalization_interval: int = 0,
    optimization_target: OptimizationTarget3D | None = None,
) -> TorchOptimizedPlan3D:
    """GPU-capable inner optimizer for fixed human-selected settings."""

    _require_torch()
    normalizations = (
        target_normalization_d98,
        target_normalization_d50,
        clinical_target_normalization_d99,
    )
    if sum(value is not None for value in normalizations) > 1:
        raise ValueError("Only one target normalization may be active")
    if target_normalization_d98 is not None and target_normalization_d98 <= 0.0:
        raise ValueError("target_normalization_d98 must be positive")
    if target_normalization_d50 is not None and target_normalization_d50 <= 0.0:
        raise ValueError("target_normalization_d50 must be positive")
    if (
        clinical_target_normalization_d99 is not None
        and clinical_target_normalization_d99 <= 0.0
    ):
        raise ValueError("clinical_target_normalization_d99 must be positive")
    if clinical_target_normalization_d99 is not None and engine.clinical_target_flat.numel() == 0:
        raise ValueError("clinical target is required for clinical-target normalization")
    if target_normalization_interval < 0:
        raise ValueError("target_normalization_interval must be nonnegative")
    target_definition = optimization_target or full_optimization_target_3d(case)
    target_definition.validate(case)
    optimization_target_flat = torch.as_tensor(
        np.flatnonzero(target_definition.coverage_mask.ravel()).astype(np.int64),
        device=engine.device,
    )
    relaxed_overlap_flat = torch.as_tensor(
        np.flatnonzero(target_definition.relaxed_overlap_mask.ravel()).astype(np.int64),
        device=engine.device,
    )
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
            dtype=torch.float32,
        )
    else:
        # Keep master fluence and Adam moments in FP32 even when the cached
        # geometry uses FP16/BF16. Pure FP16 Adam can underflow its epsilon.
        fluence = initial_fluence.detach().to(device=engine.device, dtype=torch.float32).clone()
    fluence[~active] = 0.0

    def project_target_normalization() -> None:
        if all(value is None for value in normalizations):
            return
        projected_dose = engine.forward(fluence)
        if target_normalization_d98 is not None:
            target_values = projected_dose.reshape(-1).index_select(0, engine.target_flat)
            current = torch.quantile(target_values, 0.02)
            requested = target_normalization_d98
        elif target_normalization_d50 is not None:
            target_values = projected_dose.reshape(-1).index_select(0, engine.target_flat)
            current = torch.quantile(target_values, 0.50)
            requested = target_normalization_d50
        else:
            clinical_values = projected_dose.reshape(-1).index_select(
                0,
                engine.clinical_target_flat,
            )
            current = torch.quantile(clinical_values, 0.01, interpolation="lower")
            requested = clinical_target_normalization_d99
        fluence.mul_(requested / torch.clamp(current, min=1e-8))

    periodic_normalization = (
        target_normalization_interval > 0
        and any(value is not None for value in normalizations)
    )
    if periodic_normalization:
        with torch.no_grad():
            project_target_normalization()
    fluence.requires_grad_(True)
    completed = 0
    while completed < iterations:
        stage_iterations = min(
            target_normalization_interval if periodic_normalization else iterations,
            iterations - completed,
        )
        optimizer = torch.optim.Adam([fluence], lr=learning_rate, eps=1e-7)
        for stage_step in range(stage_iterations):
            optimizer.zero_grad(set_to_none=True)
            dose = engine.forward(fluence)
            loss = _torch_loss(
                engine,
                dose,
                priorities,
                target_hotspot_threshold,
                target_hotspot_weight,
                normal_tissue_weight,
                normal_tissue_threshold,
                integral_dose_weight,
                high_dose_normal_tissue_weight,
                high_dose_normal_tissue_threshold,
                clinical_dvh_weight,
                optimization_target_flat,
                relaxed_overlap_flat,
                target_definition.relaxed_overlap_minimum,
            ) + 0.0005 * torch.sum(fluence.square())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite 3D optimization loss at iteration {completed + stage_step + 1}"
                )
            loss.backward()
            if fluence.grad is not None:
                fluence.grad[~active] = 0.0
            optimizer.step()
            with torch.no_grad():
                fluence.clamp_(min=0.0)
                fluence[~active] = 0.0
        completed += stage_iterations
        with torch.no_grad():
            if periodic_normalization:
                project_target_normalization()
    with torch.no_grad():
        dose = engine.forward(fluence)
        if target_normalization_d98 is not None:
            target_values = dose.reshape(-1).index_select(0, engine.target_flat)
            current_d98 = torch.quantile(target_values, 0.02)
            scale = target_normalization_d98 / torch.clamp(current_d98, min=1e-8)
            fluence.mul_(scale)
            dose = dose * scale
        elif target_normalization_d50 is not None:
            target_values = dose.reshape(-1).index_select(0, engine.target_flat)
            current_d50 = torch.quantile(target_values, 0.50)
            scale = target_normalization_d50 / torch.clamp(current_d50, min=1e-8)
            fluence.mul_(scale)
            dose = dose * scale
        elif clinical_target_normalization_d99 is not None:
            clinical_values = dose.reshape(-1).index_select(
                0,
                engine.clinical_target_flat,
            )
            current_d99 = torch.quantile(
                clinical_values,
                0.01,
                interpolation="lower",
            )
            scale = clinical_target_normalization_d99 / torch.clamp(
                current_d99,
                min=1e-8,
            )
            fluence.mul_(scale)
            dose = dose * scale
        final_loss = float(
            _torch_loss(
                engine,
                dose,
                priorities,
                target_hotspot_threshold,
                target_hotspot_weight,
                normal_tissue_weight,
                normal_tissue_threshold,
                integral_dose_weight,
                high_dose_normal_tissue_weight,
                high_dose_normal_tissue_threshold,
                clinical_dvh_weight,
                optimization_target_flat,
                relaxed_overlap_flat,
                target_definition.relaxed_overlap_minimum,
            ).item()
        )
        dose_numpy = dose.detach().float().cpu().numpy()
        metrics = evaluate_plan_3d(
            case,
            dose_numpy,
            final_loss,
            field_count=len(active_beams),
            optimization_target=target_definition,
        )
    return TorchOptimizedPlan3D(
        active_beams=tuple(sorted(active_beams)),
        priorities=priorities,
        fluence=fluence.detach(),
        dose=dose.detach(),
        metrics=metrics,
        iterations=completed,
        optimization_target=target_definition,
    )
