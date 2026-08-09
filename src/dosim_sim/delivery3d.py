from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryMode3D:
    """Fixed angular sampling used by the synthetic dose optimizer."""

    name: str
    angles_degrees: tuple[float, ...]
    arc_like: bool = False

    @property
    def active_beams(self) -> tuple[int, ...]:
        return tuple(range(len(self.angles_degrees)))


def delivery_mode_3d(name: str) -> DeliveryMode3D:
    modes = {
        "static_4": DeliveryMode3D("static_4", (0.0, 90.0, 180.0, 270.0)),
        "static_12": DeliveryMode3D(
            "static_12", tuple(float(value) for value in range(0, 360, 30))
        ),
        "arc_like_180": DeliveryMode3D(
            "arc_like_180", tuple(float(value) for value in range(90, 271, 10)), True
        ),
        "arc_like_360": DeliveryMode3D(
            "arc_like_360", tuple(float(value) for value in range(0, 360, 10)), True
        ),
    }
    try:
        return modes[name]
    except KeyError as error:
        raise ValueError(f"Unknown delivery mode {name!r}; choose from {tuple(modes)}") from error


def standard_delivery_modes_3d() -> tuple[DeliveryMode3D, ...]:
    return tuple(
        delivery_mode_3d(name)
        for name in ("static_4", "static_12", "arc_like_180", "arc_like_360")
    )
