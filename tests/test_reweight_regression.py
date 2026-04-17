"""Regression tests for bugs in src/microcalibrate/reweight.py.

Covers:
- Off-by-one epoch guard (finding #1): the returned final weights must
  be consistent with a training loop that steps every epoch, and the
  final logged estimate must correspond to the final tracked epoch.
- L0 branch divide-by-zero when ``start_loss == 0`` (finding #5).
- ``np.log(original_weights)`` on non-positive initial weights in the
  L0 branch (finding #8).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from microcalibrate.calibration import Calibration
from microcalibrate.reweight import reweight


def _make_dataset(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "income": rng.normal(40000, 50000, size=n),
        }
    )
    weights = np.ones(len(data))
    targets_matrix = pd.DataFrame(
        {
            "income_aged_20_30": (
                (data["age"] >= 20) & (data["age"] <= 30)
            ).astype(float)
            * data["income"],
            "income_aged_40_50": (
                (data["age"] >= 40) & (data["age"] <= 50)
            ).astype(float)
            * data["income"],
        }
    )
    targets = np.array(
        [
            (targets_matrix["income_aged_20_30"] * weights).sum() * 1.1,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 1.1,
        ]
    )
    return targets_matrix, weights, targets


def test_final_epoch_matches_tracker() -> None:
    """Finding #1: the final tracker row must correspond to the final epoch.

    The previous implementation silently skipped the penultimate epoch's
    step via ``if i != max_epochs - 1`` with ``max_epochs = epochs - 1``.
    After the fix, every epoch steps AND the tracker always contains a
    row for the last epoch, so ``epochs_list[-1] == epochs - 1``.
    """
    targets_matrix, weights, targets = _make_dataset()

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.0,
        epochs=25,
        learning_rate=0.05,
        dropout_rate=0,
        seed=0,
    )
    performance_df = calibrator.calibrate()

    # The tracker must include the final epoch so the last logged
    # estimate/loss correspond to the returned state.
    last_tracked_epoch = performance_df["epoch"].max()
    assert last_tracked_epoch == calibrator.epochs - 1, (
        f"Tracker must include final epoch; got {last_tracked_epoch} "
        f"for epochs={calibrator.epochs}."
    )


def test_all_epochs_step() -> None:
    """Finding #1: every epoch must contribute a gradient step.

    We compare the final weights after N and N-1 epochs with otherwise
    identical inputs and zero noise (so initialisation is deterministic
    regardless of RNG seeding). Under the previous bug, the penultimate
    epoch was a silent no-op which could make runs converge to the same
    point; after the fix every epoch moves the weights.
    """

    def estimate_function(weights: torch.Tensor) -> torch.Tensor:
        return weights

    original_weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    targets = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    logger = logging.getLogger("test_reweight_regression")

    def _run(epochs: int) -> np.ndarray:
        torch.manual_seed(0)
        final_weights, _sparse, _df = reweight(
            original_weights=original_weights,
            estimate_function=estimate_function,
            targets_array=targets,
            target_names=np.array(["a", "b", "c", "d"]),
            l0_lambda=0.0,
            init_mean=0.999,
            temperature=0.5,
            regularize_with_l0=False,
            sparse_learning_rate=0.2,
            dropout_rate=0.0,
            epochs=epochs,
            noise_level=0.0,  # deterministic init
            learning_rate=0.1,
            logger=logger,
        )
        return final_weights

    w_n_minus_1 = _run(10)
    w_n = _run(11)
    assert not np.allclose(w_n_minus_1, w_n), (
        "Weights after N epochs must differ from weights after N-1 epochs "
        "(i.e. the final epoch must step)."
    )


def test_l0_start_loss_zero_does_not_crash() -> None:
    """Finding #5: the sparse loop must not crash if ``start_loss == 0``.

    We drive the initial loss to (numerically) zero by using already-
    calibrated weights with ``noise_level=0`` and an ``l0_lambda`` of
    zero. The tqdm postfix previously divided by zero; after the fix
    the postfix is written without raising.
    """

    def estimate_function(weights: torch.Tensor) -> torch.Tensor:
        # Each weight is its own estimate; weights == targets so loss=0.
        return weights

    original_weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    targets = original_weights.copy()
    logger = logging.getLogger("test_reweight_regression")

    # Minimal L0 run; a ZeroDivisionError would propagate out.
    final_weights, final_sparse, _df = reweight(
        original_weights=original_weights,
        estimate_function=estimate_function,
        targets_array=targets,
        target_names=np.array(["a", "b", "c", "d"]),
        l0_lambda=0.0,  # drive start_loss to ~0
        init_mean=0.999,
        temperature=0.5,
        regularize_with_l0=True,
        sparse_learning_rate=0.01,
        dropout_rate=0.0,
        epochs=3,
        noise_level=0.0,
        learning_rate=1e-3,
        logger=logger,
    )
    assert final_sparse is not None
    assert final_weights.shape == original_weights.shape


def test_l0_log_guard_on_zero_initial_weight() -> None:
    """Finding #8: zero initial weights must not produce NaNs in L0 path.

    Without the clamp in ``np.log(original_weights)`` the L0 branch
    produces ``log(0) = -inf`` which poisons gradients immediately.
    """

    def estimate_function(weights: torch.Tensor) -> torch.Tensor:
        return weights

    original_weights = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float64)
    targets = np.array([0.5, 0.5, 1.0, 2.0], dtype=np.float64)
    logger = logging.getLogger("test_reweight_regression")

    final_weights, final_sparse, _df = reweight(
        original_weights=original_weights,
        estimate_function=estimate_function,
        targets_array=targets,
        target_names=np.array(["a", "b", "c", "d"]),
        l0_lambda=1e-6,
        init_mean=0.999,
        temperature=0.5,
        regularize_with_l0=True,
        sparse_learning_rate=0.01,
        dropout_rate=0.0,
        epochs=3,
        noise_level=0.0,
        learning_rate=1e-3,
        logger=logger,
    )
    assert final_sparse is not None
    assert np.isfinite(final_sparse).all(), (
        "Sparse weights contained non-finite values; the np.log guard "
        "on zero initial weights is not working."
    )
    assert np.isfinite(final_weights).all()
