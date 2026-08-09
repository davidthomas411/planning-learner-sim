from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .volume3d import SyntheticCase3D


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class _BeamMap:
    u0: NDArray[np.uint8]
    v0: NDArray[np.uint8]
    du: FloatArray
    dv: FloatArray
    attenuation: FloatArray


class ImplicitDoseEngine3D:
    """Linear 3D dose operator without a voxel-by-beamlet influence matrix.

    A beam's 2D fluence map is bilinearly sampled at every body voxel after
    rotating coordinates into beam's-eye view. Depth attenuation is then
    applied. ``adjoint`` is the exact transpose used by the inner optimizer.
    """

    def __init__(
        self,
        case: SyntheticCase3D,
        beam_angles_degrees: tuple[float, ...],
        fluence_size: int = 8,
        attenuation: float = 0.32,
    ) -> None:
        if fluence_size < 2 or fluence_size > 255:
            raise ValueError("fluence_size must be between 2 and 255")
        self.case = case
        self.beam_angles_degrees = tuple(float(value) for value in beam_angles_degrees)
        self.fluence_size = int(fluence_size)
        self.shape = case.body.shape
        self._body_flat = np.flatnonzero(case.body.ravel())
        i, j, k = np.unravel_index(self._body_flat, self.shape)
        x = case.axis[i]
        y = case.axis[j]
        z = case.axis[k]
        self._maps: list[_BeamMap] = []

        for angle_degrees in self.beam_angles_degrees:
            angle = np.deg2rad(angle_degrees)
            lateral = -np.sin(angle) * x + np.cos(angle) * y
            depth = np.cos(angle) * x + np.sin(angle) * y
            u = np.clip((lateral + 1.0) * 0.5 * (fluence_size - 1), 0.0, fluence_size - 1.0)
            v = np.clip((z + 1.0) * 0.5 * (fluence_size - 1), 0.0, fluence_size - 1.0)
            u0 = np.minimum(np.floor(u).astype(np.uint8), fluence_size - 2)
            v0 = np.minimum(np.floor(v).astype(np.uint8), fluence_size - 2)
            self._maps.append(
                _BeamMap(
                    u0=u0,
                    v0=v0,
                    du=(u - u0).astype(np.float32),
                    dv=(v - v0).astype(np.float32),
                    attenuation=np.exp(-attenuation * (depth + 1.0)).astype(np.float32),
                )
            )

    @property
    def n_beams(self) -> int:
        return len(self._maps)

    @property
    def cache_bytes(self) -> int:
        return int(
            self._body_flat.nbytes
            + sum(
                beam.u0.nbytes
                + beam.v0.nbytes
                + beam.du.nbytes
                + beam.dv.nbytes
                + beam.attenuation.nbytes
                for beam in self._maps
            )
        )

    def _sample(self, plane: FloatArray, mapping: _BeamMap) -> FloatArray:
        u0, v0, du, dv = mapping.u0, mapping.v0, mapping.du, mapping.dv
        return mapping.attenuation * (
            plane[u0, v0] * (1.0 - du) * (1.0 - dv)
            + plane[u0 + 1, v0] * du * (1.0 - dv)
            + plane[u0, v0 + 1] * (1.0 - du) * dv
            + plane[u0 + 1, v0 + 1] * du * dv
        )

    def forward(self, fluence: FloatArray) -> FloatArray:
        fluence = np.asarray(fluence, dtype=np.float32)
        expected = (self.n_beams, self.fluence_size, self.fluence_size)
        if fluence.shape != expected:
            raise ValueError(f"fluence shape must be {expected}, got {fluence.shape}")
        body_dose = np.zeros(self._body_flat.size, dtype=np.float32)
        for plane, mapping in zip(fluence, self._maps, strict=True):
            body_dose += self._sample(plane, mapping)
        dose = np.zeros(int(np.prod(self.shape)), dtype=np.float32)
        dose[self._body_flat] = body_dose
        return dose.reshape(self.shape)

    def adjoint(self, voxel_values: FloatArray) -> FloatArray:
        values = np.asarray(voxel_values, dtype=np.float32)
        if values.shape != self.shape:
            raise ValueError(f"voxel_values shape must be {self.shape}, got {values.shape}")
        body_values = values.ravel()[self._body_flat]
        result = np.zeros((self.n_beams, self.fluence_size, self.fluence_size), dtype=np.float32)
        size = self.fluence_size
        for beam_index, mapping in enumerate(self._maps):
            u0 = mapping.u0.astype(np.int32)
            v0 = mapping.v0.astype(np.int32)
            weighted = body_values * mapping.attenuation
            corners = (
                (u0, v0, (1.0 - mapping.du) * (1.0 - mapping.dv)),
                (u0 + 1, v0, mapping.du * (1.0 - mapping.dv)),
                (u0, v0 + 1, (1.0 - mapping.du) * mapping.dv),
                (u0 + 1, v0 + 1, mapping.du * mapping.dv),
            )
            flat = np.zeros(size * size, dtype=np.float32)
            for u, v, interpolation_weight in corners:
                flat += np.bincount(
                    u * size + v,
                    weights=weighted * interpolation_weight,
                    minlength=size * size,
                ).astype(np.float32)
            result[beam_index] = flat.reshape(size, size)
        return result
