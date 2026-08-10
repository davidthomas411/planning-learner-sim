"""Clinically analogous image channels for learned 3D planning policies."""

from __future__ import annotations

import numpy as np

from .planning3d import HighLevelSearchConfig3D, PlanningStep3D
from .volume3d import SyntheticCase3D

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - the GPU dependency is optional
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


VOLUME_CHANNEL_NAMES = (
    "body",
    "target",
    "oar_0",
    "oar_1",
    "oar_2",
    "dose",
    "target_underdose",
    "target_hotspot",
    "oar_0_excess",
    "oar_1_excess",
    "oar_2_excess",
)


def state_volume_3d(
    case: SyntheticCase3D,
    step: PlanningStep3D,
    config: HighLevelSearchConfig3D,
    output_size: int | None = 16,
) -> "torch.Tensor":
    """Return contours, current dose, and visible dose-rule violations.

    The channels contain only information available at the current planning
    state. They do not contain a future action or a terminal plan.
    """

    if torch is None or functional is None:
        raise RuntimeError("PyTorch is required for the 3D image representation")
    device = step.plan.dose.device
    dose = step.plan.dose.detach().to(device=device, dtype=torch.float32)
    if tuple(dose.shape) != tuple(case.body.shape):
        raise ValueError(f"dose shape {tuple(dose.shape)} does not match case shape {case.body.shape}")

    def mask_tensor(mask: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(mask, device=device, dtype=torch.float32)

    body = mask_tensor(case.body)
    target = mask_tensor(case.target)
    oars = [mask_tensor(mask) for mask in case.oars]
    while len(oars) < 3:
        oars.append(torch.zeros_like(body))
    dose_channel = torch.clamp(dose / 1.5, 0.0, 1.5)
    target_underdose = target * torch.clamp((config.d95_min - dose) / config.d95_min, 0.0, 1.0)
    target_hotspot = target * torch.clamp((dose - config.d02_max) / config.d02_max, 0.0, 1.0)
    oar_excess = []
    for index, oar in enumerate(oars):
        if index < len(case.oar_limits):
            oar_excess.append(oar * torch.clamp(dose / case.oar_limits[index] - 1.0, 0.0, 1.0))
        else:
            oar_excess.append(torch.zeros_like(body))
    volume = torch.stack(
        [body, target, *oars, dose_channel, target_underdose, target_hotspot, *oar_excess]
    )
    if output_size is not None and tuple(volume.shape[1:]) != (output_size,) * 3:
        volume = functional.interpolate(
            volume.unsqueeze(0),
            size=(output_size, output_size, output_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    return volume.contiguous()
