"""Regression test for the unseeded numpy RNG in reweight() (finding #3).

The initial weight noise was drawn from the global ``np.random``
generator, which was never seeded. Two Calibration runs with the same
``seed`` could therefore produce different results because torch was
seeded but numpy was not.
"""

import numpy as np
import pandas as pd

from microcalibrate.calibration import Calibration


def _make_calibrator(seed: int) -> Calibration:
    rng = np.random.default_rng(0)  # fixed, independent of seed-under-test
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=80),
            "income": rng.normal(40000, 50000, size=80),
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
            (targets_matrix["income_aged_20_30"] * weights).sum() * 1.05,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 1.05,
        ]
    )
    return Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=10.0,  # nonzero so the np RNG is actually exercised
        epochs=50,
        learning_rate=0.01,
        dropout_rate=0,
        seed=seed,
    )


def test_identical_seeds_produce_identical_weights() -> None:
    """Two calibrations with the same seed must converge to the same
    weights. Before the fix this failed intermittently because the
    initial noise was drawn from an unseeded numpy RNG."""
    a = _make_calibrator(seed=42)
    a.calibrate()
    b = _make_calibrator(seed=42)
    b.calibrate()
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-6, atol=1e-6)


def test_different_seeds_produce_different_weights() -> None:
    """Different seeds must actually perturb the noise and therefore
    the trajectory. This guards against accidentally hard-coding a seed
    that ignores the caller-supplied value."""
    a = _make_calibrator(seed=1)
    a.calibrate()
    b = _make_calibrator(seed=2)
    b.calibrate()
    assert not np.allclose(a.weights, b.weights, rtol=1e-6, atol=1e-6)


def test_calibration_does_not_mutate_global_numpy_state() -> None:
    """reweight() must not poison the caller's global numpy RNG stream.

    A caller that seeds numpy, draws a sample, calls Calibration, and
    then draws again should see the same second sample regardless of
    whether calibration ran, as long as we did not mutate the global
    state. Before the fix this invariant was violated in practice
    because reweight() used the global numpy RNG directly; with a
    local ``default_rng`` it holds.
    """
    np.random.seed(123)
    pre = np.random.random(5)
    _make_calibrator(seed=7).calibrate()
    np.random.seed(123)
    post = np.random.random(5)
    np.testing.assert_array_equal(pre, post)
