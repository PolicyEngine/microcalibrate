"""Gradient-accumulation batch mode must produce the same weights as full-batch.

The chi-squared loss is separable across record batches *given* the
pre-computed per-target coefficient c_j = 2*(S_j - t_j) / (t_j + 1)^2,
because the estimate S_j is a sum over records. Implementing this as
two passes (accumulate S under no_grad, then per-batch backward) keeps
peak memory O(B * k) instead of O(N * k) — critical at v7 scale where
N ≈ 1.5M and k ≈ 500.

These tests verify that:

1. `batch_size=None` (default) matches the existing full-batch behavior bit
   for bit.
2. `batch_size < N` produces final weights within tight numerical tolerance
   of the full-batch run (relative error < 1e-4).
3. `batch_size` that does not evenly divide N still processes every row.
4. `batch_size >= N` degenerates to full-batch and matches exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from microcalibrate import Calibration


def _problem(n_rows: int = 1_000, n_cols: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(2.0, 10.0, size=n_rows).astype(np.float64)
    matrix = rng.uniform(0.0, 1.0, size=(n_rows, n_cols)).astype(np.float32)
    # Targets that require weight adjustments of ~5-20%.
    targets = matrix.sum(axis=0) * rng.uniform(0.85, 1.15, size=n_cols)
    target_names = np.array([f"c{i}" for i in range(n_cols)])
    df = pd.DataFrame(matrix, columns=target_names)
    return weights, targets, target_names, df


def _calibrate(
    batch_size,
    epochs=50,
    seed=42,
    dropout_rate=0.0,
    normalization_factor=None,
    excluded_targets=None,
    regularize_with_l0=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights, targets, target_names, df = _problem()
    calibration = Calibration(
        weights=weights.copy(),
        targets=targets,
        target_names=target_names,
        estimate_matrix=df,
        epochs=epochs,
        learning_rate=0.05,
        noise_level=0.0,
        dropout_rate=dropout_rate,
        normalization_factor=normalization_factor,
        excluded_targets=excluded_targets,
        regularize_with_l0=regularize_with_l0,
        seed=seed,
        batch_size=batch_size,
    )
    calibration.calibrate()
    return calibration.weights


class TestBatchEquivalence:
    def test_full_batch_default_matches_none(self) -> None:
        """batch_size=None is the existing full-batch path; reproducible."""
        w1 = _calibrate(batch_size=None)
        w2 = _calibrate(batch_size=None)
        np.testing.assert_allclose(w1, w2, rtol=1e-6, atol=0.0)

    def test_batched_matches_full_batch(self) -> None:
        """Smaller batch size must produce the same final weights."""
        full = _calibrate(batch_size=None)
        batched = _calibrate(batch_size=100)
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        # 1e-4 is tight enough to catch implementation bugs but loose
        # enough for fp32 rounding in matmul over N/B partial sums.
        assert rel_err.max() < 1e-4, (
            f"batched vs full max rel error = {rel_err.max():.6e}; "
            f"full[:5]={full[:5]}, batched[:5]={batched[:5]}"
        )

    def test_batch_size_not_evenly_dividing(self) -> None:
        """batch_size=333 with N=1000 must still cover every row exactly once per epoch."""
        full = _calibrate(batch_size=None)
        batched = _calibrate(batch_size=333)
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        assert rel_err.max() < 1e-4, rel_err.max()

    def test_batch_size_at_or_above_n(self) -> None:
        """batch_size >= n_rows is a no-op; must match full-batch exactly."""
        full = _calibrate(batch_size=None)
        batched = _calibrate(batch_size=10_000)  # > n_rows=1000
        np.testing.assert_allclose(batched, full, rtol=1e-6, atol=0.0)

    def test_batch_size_one(self) -> None:
        """Extreme case: one record per batch — N backward calls per epoch."""
        full = _calibrate(batch_size=None, epochs=5)
        batched = _calibrate(batch_size=1, epochs=5)
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        assert rel_err.max() < 1e-4, rel_err.max()


class TestBatchInteractionsWithOtherFeatures:
    """Equivalence must hold when combined with dropout and normalization."""

    def test_equivalence_with_nonzero_dropout(self) -> None:
        """Single per-epoch dropout mask is shared across batches; matches full."""
        full = _calibrate(batch_size=None, dropout_rate=0.3)
        batched = _calibrate(batch_size=250, dropout_rate=0.3)
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        assert rel_err.max() < 1e-4, rel_err.max()

    def test_equivalence_with_normalization_factor(self) -> None:
        """Per-target normalization_factor multiplies the coefficient identically."""
        n_cols = 5
        normalization_factor = torch.tensor(
            [0.5, 1.0, 2.0, 1.5, 0.8], dtype=torch.float32
        )
        full = _calibrate(
            batch_size=None, normalization_factor=normalization_factor
        )
        batched = _calibrate(
            batch_size=100, normalization_factor=normalization_factor
        )
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        assert rel_err.max() < 1e-4, rel_err.max()

    def test_equivalence_with_excluded_targets(self) -> None:
        """exclude_targets() filters `estimate_matrix` via torch indexing; batched still agrees."""
        full = _calibrate(batch_size=None, excluded_targets=["c4"])
        batched = _calibrate(batch_size=100, excluded_targets=["c4"])
        rel_err = np.abs(batched - full) / np.maximum(np.abs(full), 1e-6)
        assert rel_err.max() < 1e-4, rel_err.max()


class TestBatchGuardrails:
    """Configurations that aren't supported must fail loudly, not silently no-op."""

    def test_batch_size_with_l0_raises(self) -> None:
        """L0 sparse loop is not batched; the combination must raise."""
        import pytest

        with pytest.raises(ValueError, match="regularize_with_l0"):
            _calibrate(batch_size=100, regularize_with_l0=True)
