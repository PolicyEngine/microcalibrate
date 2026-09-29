"""Regression tests for the log-space dropout bug (finding #2).

The previous implementation set masked entries in log space to ``0``,
which corresponds to ``exp(0) = 1`` in linear space -- the opposite of
dropping them. It then divided masked log-weights by their sum, which
is not a meaningful normalisation on logs and could cross zero,
producing ``inf`` / ``NaN`` on realistic weight scales.
"""

import logging

import numpy as np
import pytest
import torch

from microcalibrate.reweight import dropout_weights, reweight


def test_dropout_p_zero_is_identity() -> None:
    """p=0 must return the input tensor unchanged (no dropout)."""
    weights = torch.log(torch.tensor([10.0, 100.0, 1000.0]))
    result = dropout_weights(weights, 0.0)
    assert torch.equal(result, weights)


def test_dropout_p_one_zeroes_all_linear_weights() -> None:
    """p=1 must zero every linear-space weight."""
    weights = torch.log(torch.tensor([10.0, 100.0, 1000.0]))
    result = dropout_weights(weights, 1.0)
    linear = torch.exp(result)
    assert torch.all(linear == 0)


def test_dropout_preserves_sum_in_expectation_on_realistic_scale() -> None:
    """On realistic (non-unit) weights dropout must not produce NaN/Inf
    and the expected linear-space sum must be preserved.

    Regression: on realistic survey weights (hundreds to thousands),
    ``log(w)`` is ~6-8 and the previous normalisation step could cross
    zero, yielding ``inf`` or ``NaN``. With inverted dropout, the
    expected linear-space sum of the output equals the linear-space
    sum of the input.
    """
    rng = np.random.default_rng(7)
    linear_weights = rng.uniform(100.0, 5000.0, size=500)
    log_weights = torch.tensor(np.log(linear_weights), dtype=torch.float32)

    torch.manual_seed(0)
    n_trials = 200
    totals = []
    for _ in range(n_trials):
        out = dropout_weights(log_weights, 0.3)
        linear_out = torch.exp(out)
        assert torch.isfinite(linear_out).all()
        totals.append(linear_out.sum().item())

    expected_sum = linear_weights.sum()
    observed_mean = float(np.mean(totals))
    # Monte Carlo tolerance: standard error scales ~sqrt(p*(1-p)/n).
    assert abs(observed_mean - expected_sum) / expected_sum < 0.02, (
        f"Inverted dropout should preserve the linear-space sum in "
        f"expectation; got mean {observed_mean:.1f} vs expected "
        f"{expected_sum:.1f}."
    )


def test_dropout_drops_approximately_p_fraction() -> None:
    """At p=0.5, roughly half the linear-space outputs must be zero."""
    weights = torch.log(torch.ones(10_000) * 42.0)
    torch.manual_seed(1)
    result = dropout_weights(weights, 0.5)
    fraction_zero = (torch.exp(result) == 0).float().mean().item()
    assert 0.45 < fraction_zero < 0.55


def test_dropout_rejects_out_of_range() -> None:
    """Out-of-range dropout probabilities must raise explicitly."""
    weights = torch.log(torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        dropout_weights(weights, 1.5)
    with pytest.raises(ValueError):
        dropout_weights(weights, -0.1)


def test_reweight_runs_with_realistic_scale_dropout() -> None:
    """End-to-end: training with dropout_rate > 0 on realistic-scale
    weights must not inject NaN/Inf into the loss.
    """

    def estimate_function(w: torch.Tensor) -> torch.Tensor:
        return w.sum().unsqueeze(0)

    rng = np.random.default_rng(0)
    original_weights = rng.uniform(100.0, 5000.0, size=200)
    targets = np.array([original_weights.sum() * 1.05])
    logger = logging.getLogger("test_dropout_regression")

    torch.manual_seed(0)
    final_weights, _sparse, _df = reweight(
        original_weights=original_weights,
        estimate_function=estimate_function,
        targets_array=targets,
        target_names=np.array(["total"]),
        l0_lambda=0.0,
        init_mean=0.999,
        temperature=0.5,
        regularize_with_l0=False,
        sparse_learning_rate=0.2,
        dropout_rate=0.3,
        epochs=5,
        noise_level=0.0,
        learning_rate=1e-3,
        logger=logger,
    )
    assert np.all(np.isfinite(final_weights))
