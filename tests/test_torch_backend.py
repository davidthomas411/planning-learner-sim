import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dosim_sim.dose3d import ImplicitDoseEngine3D
from dosim_sim.objective import PlanningPriorities
from dosim_sim.torch_dose3d import (
    TorchImplicitDoseEngine3D,
    optimize_fluence_3d_torch,
)
from dosim_sim.volume3d import generate_case_3d


def test_torch_forward_and_adjoint_match_numpy_reference() -> None:
    case = generate_case_3d(11, grid_size=24)
    angles = (0.0, 90.0)
    numpy_engine = ImplicitDoseEngine3D(case, angles, fluence_size=4)
    torch_engine = TorchImplicitDoseEngine3D(case, angles, fluence_size=4)
    rng = np.random.default_rng(12)
    fluence = rng.random((2, 4, 4), dtype=np.float32)
    voxel_probe = rng.random(case.body.shape, dtype=np.float32)

    torch_dose = torch_engine.forward(torch.from_numpy(fluence)).numpy()
    torch_adjoint = torch_engine.adjoint(torch.from_numpy(voxel_probe)).numpy()
    assert np.allclose(torch_dose, numpy_engine.forward(fluence), rtol=2e-5, atol=2e-5)
    assert np.allclose(torch_adjoint, numpy_engine.adjoint(voxel_probe), rtol=2e-5, atol=2e-4)


def test_torch_forward_batches_candidate_fluence_states() -> None:
    case = generate_case_3d(13, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 120.0, 240.0), fluence_size=4)
    fluence = torch.rand(3, 3, 4, 4)
    batched = engine.forward(fluence)
    individual = torch.stack([engine.forward(item) for item in fluence])
    assert torch.allclose(batched, individual)


def test_torch_optimizer_keeps_inactive_beams_zero() -> None:
    case = generate_case_3d(14, grid_size=24)
    engine = TorchImplicitDoseEngine3D(case, (0.0, 90.0, 180.0, 270.0), fluence_size=4)
    plan = optimize_fluence_3d_torch(
        case,
        engine,
        (0, 2),
        PlanningPriorities.for_case(case),
        iterations=3,
    )
    assert torch.any(plan.fluence[[0, 2]] > 0)
    assert torch.all(plan.fluence[[1, 3]] == 0)
