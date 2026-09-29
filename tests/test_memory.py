"""Memory-footprint regression tests for the Calibration class.

At the 1.5M-row scale used by microplex-us's v7 pipeline, holding both
the user-provided pandas DataFrame *and* an independent torch.float32
copy of the same matrix roughly doubles peak RSS during calibrate().

This test pins the fix: after Calibration builds the torch tensor,
self.original_estimate_matrix must be released so its storage is
garbage-collectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcalibrate import Calibration


def _small_problem(n_rows: int = 200, n_cols: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(1.0, 5.0, size=n_rows)
    matrix = rng.uniform(0.0, 1.0, size=(n_rows, n_cols)).astype(np.float64)
    targets = matrix.sum(axis=0) * 1.1
    target_names = np.array([f"c{i}" for i in range(n_cols)])
    estimate_matrix = pd.DataFrame(matrix, columns=target_names)
    return weights, targets, target_names, estimate_matrix


class TestOriginalEstimateMatrixReleased:
    """After __init__, the user-provided DataFrame must be releasable."""

    def test_original_estimate_matrix_released_after_init(self) -> None:
        weights, targets, target_names, estimate_matrix = _small_problem()
        calibration = Calibration(
            weights=weights,
            targets=targets,
            target_names=target_names,
            estimate_matrix=estimate_matrix,
            epochs=4,
            noise_level=0.0,
        )
        assert calibration.original_estimate_matrix is None, (
            "Calibration retained original_estimate_matrix; at v7 scale "
            "(1.5M rows x 500 cols float64) this is a 6 GB leak."
        )
        # The authoritative matrix is the cached float32 torch tensor;
        # downstream code (hyperparameter tuning, evaluation) reads this.
        assert calibration.estimate_matrix_tensor is not None
        assert calibration.estimate_matrix_tensor.dtype.is_floating_point
        assert calibration.estimate_matrix_tensor.shape == (
            len(weights),
            len(target_names),
        )

    def test_calibrate_still_works_after_release(self) -> None:
        """Convergence behavior must be preserved after the matrix is freed."""
        weights, targets, target_names, estimate_matrix = _small_problem()
        calibration = Calibration(
            weights=weights,
            targets=targets,
            target_names=target_names,
            estimate_matrix=estimate_matrix,
            epochs=200,
            learning_rate=0.05,
            noise_level=0.0,
        )
        performance = calibration.calibrate()
        # Loss is strictly decreasing on a well-posed small problem.
        losses = performance["loss"].tolist() if "loss" in performance else []
        assert len(losses) >= 2, performance
        assert (
            losses[-1] < losses[0]
        ), f"Calibration did not improve loss: {losses[:5]}...{losses[-5:]}"
