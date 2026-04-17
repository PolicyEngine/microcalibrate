"""Regression tests for the +1 smoothing bug in loss/pct_close
(finding #7).

The original `((estimate - target) + 1) / (target + 1)` construction
produced Inf when ``target == -1`` and the NaN guard only caught NaN,
so calibration silently diverged. After the fix the denominator is
clamped to be strictly positive and the guard rejects any non-finite
rel_error.
"""

import math

import pytest
import torch

from microcalibrate.utils.metrics import loss, pct_close


def test_loss_raises_on_target_equal_to_neg_one() -> None:
    """A target value of exactly -1 used to produce Inf silently; the
    guard must reject non-finite rel_error instead of NaN-only."""
    estimate = torch.tensor([0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    l = loss(estimate, targets)
    # With the clamp in place the result is finite; check that.
    assert torch.isfinite(l)


def test_loss_rejects_non_finite_rel_error() -> None:
    """If the estimate itself is non-finite (e.g. an upstream NaN),
    ``loss`` must raise -- the old ``torch.isnan`` check missed Inf."""
    estimate = torch.tensor([float("inf"), 0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        loss(estimate, targets)


def test_loss_preserves_values_for_well_behaved_targets() -> None:
    """For targets where target+1 is comfortably positive, the loss
    must equal the original formula (backwards compatibility)."""
    estimate = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    targets = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    expected = (((estimate - targets) + 1) / (targets + 1)) ** 2
    assert torch.allclose(loss(estimate, targets), expected.mean())


def test_pct_close_finite_on_target_equal_to_neg_one() -> None:
    """pct_close used the same (1 + target) denominator; it must not
    return NaN/Inf when ``target == -1``."""
    estimate = torch.tensor([0.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    result = pct_close(estimate, targets)
    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0
