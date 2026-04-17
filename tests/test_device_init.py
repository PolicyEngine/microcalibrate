"""Regression tests for the duplicated device-init block (finding #6).

Two problems in the old code:

1. Lines 100-110 re-ran the device resolution already done at lines
   68-75, and the duplicate block's "else" branch was unreachable
   because the default was "cpu". Users who passed ``device="cuda"``
   therefore got ``torch.manual_seed`` but never ``torch.cuda.manual_seed``.

2. ``if self.device == "cuda"`` compared a ``torch.device`` to a
   string, which is technically true in modern torch for CUDA devices
   but fragile and confusing, and it was inside the unreachable
   branch anyway.
"""

import numpy as np
import pandas as pd
import torch

from microcalibrate.calibration import Calibration


def _make_calibrator(**kwargs) -> Calibration:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=20),
            "y": rng.normal(size=20),
        }
    )
    estimate_matrix = pd.DataFrame(
        {
            "sum_x": data["x"],
            "sum_y": data["y"],
        }
    )
    weights = np.ones(len(data))
    targets = np.array([data["x"].sum(), data["y"].sum()])
    return Calibration(
        estimate_matrix=estimate_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.1,
        epochs=5,
        learning_rate=0.01,
        dropout_rate=0,
        **kwargs,
    )


def test_device_is_torch_device_not_string() -> None:
    """``self.device`` must be a torch.device object after init."""
    calibrator = _make_calibrator(device="cpu")
    assert isinstance(calibrator.device, torch.device)
    assert calibrator.device.type == "cpu"


def test_cpu_path_seeds_torch() -> None:
    """Passing ``device='cpu'`` with a seed must still seed torch.

    The previous block lived inside ``if device is not None`` and only
    called ``torch.manual_seed`` there, but it had a redundant second
    device resolution. The simplified code seeds torch uniformly on
    every path.
    """
    calibrator = _make_calibrator(device="cpu", seed=123)
    # Generate a random number after construction; redo the construction
    # and verify we get the same draw (torch was seeded).
    draw_a = torch.rand(3)

    calibrator_2 = _make_calibrator(device="cpu", seed=123)
    draw_b = torch.rand(3)

    assert torch.equal(draw_a, draw_b)


def test_default_device_resolves_without_crashing() -> None:
    """With device=None the fallback chain runs exactly once."""
    calibrator = _make_calibrator(device=None)
    # Should resolve to some valid torch.device.
    assert isinstance(calibrator.device, torch.device)
    assert calibrator.device.type in {"cuda", "mps", "cpu"}
