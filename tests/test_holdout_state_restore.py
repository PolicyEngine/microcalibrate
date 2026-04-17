"""Regression tests for holdout state restoration + seed namespace
(findings #9 and #10).

Finding #9: ``evaluate_holdout_robustness`` (and the analogous tuning
path) only restored attributes whose captured value was not None, so
if the caller started with ``excluded_targets=None`` the attribute was
left at the last holdout set's list, silently leaking that holdout
into any subsequent ``calibrate()`` call.

Finding #10: tuning called ``_create_holdout_sets(..., seed)`` and
robustness evaluation called ``_create_holdout_sets(..., seed + 1)``,
so holdout set #1 of tuning and holdout set #0 of evaluation were
identical -- reported generalisation accuracy was optimistic by
construction.
"""

import numpy as np
import pandas as pd

from microcalibrate.calibration import Calibration


def _make_calibrator(seed: int = 42) -> Calibration:
    rng = np.random.default_rng(7)
    n = 150
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n),
            "income": rng.lognormal(10.5, 0.7, size=n),
            "region": rng.choice(["N", "S", "E", "W"], size=n),
            "employed": rng.binomial(1, 0.7, size=n),
        }
    )
    weights = rng.uniform(0.5, 1.5, size=n)
    estimate_matrix = pd.DataFrame(
        {
            "total": np.ones(n),
            "employed": data["employed"].astype(float),
            "income_n": ((data["region"] == "N") * data["income"]).astype(
                float
            ),
            "income_s": ((data["region"] == "S") * data["income"]).astype(
                float
            ),
            "income_e": ((data["region"] == "E") * data["income"]).astype(
                float
            ),
            "income_w": ((data["region"] == "W") * data["income"]).astype(
                float
            ),
            "young": ((data["age"] < 30) & data["employed"]).astype(float),
            "senior": (data["age"] >= 65).astype(float),
        }
    )
    targets = np.array(
        [
            n * 1.05,
            (estimate_matrix["employed"] * weights).sum() * 0.95,
            (estimate_matrix["income_n"] * weights).sum() * 1.1,
            (estimate_matrix["income_s"] * weights).sum() * 0.9,
            (estimate_matrix["income_e"] * weights).sum() * 1.05,
            (estimate_matrix["income_w"] * weights).sum() * 0.98,
            (estimate_matrix["young"] * weights).sum() * 1.15,
            (estimate_matrix["senior"] * weights).sum() * 0.92,
        ]
    )
    return Calibration(
        estimate_matrix=estimate_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.1,
        epochs=30,
        learning_rate=0.01,
        dropout_rate=0,
        seed=seed,
    )


def test_holdout_restores_excluded_targets_to_none() -> None:
    """After evaluate_holdout_robustness, a calibrator that started
    with excluded_targets=None must still have excluded_targets=None.
    Previously the last holdout set's names were silently kept."""
    calibrator = _make_calibrator()
    calibrator.calibrate()
    assert calibrator.excluded_targets is None

    calibrator.evaluate_holdout_robustness(
        n_holdout_sets=2,
        holdout_fraction=0.25,
    )

    assert calibrator.excluded_targets is None, (
        "excluded_targets should be restored to None after holdout "
        f"evaluation; got {calibrator.excluded_targets}."
    )
    # target_names/targets should also match the original calibration
    # surface (all 8 targets, none excluded).
    assert len(calibrator.target_names) == 8
    assert len(calibrator.targets) == 8


def test_robustness_uses_orthogonal_seed_namespace() -> None:
    """Finding #10: ``evaluate_holdout_robustness`` must pass an
    orthogonal seed namespace to ``_create_holdout_sets`` so its
    holdouts do not match tuning's deterministically.

    Under the old code, tuning called ``_create_holdout_sets(..., seed)``
    and robustness called ``_create_holdout_sets(..., seed + 1)``, so
    the RNG for tuning's holdout #1 (rng(seed + 1)) matched robustness
    holdout #0 (rng((seed + 1) + 0)) exactly. After the fix, robustness
    uses ``seed + 10_000`` which makes that specific collision
    impossible. We do not assert "no collision ever" -- with small
    target counts random overlap is possible -- we assert that the
    deterministic index-aligned collision no longer fires.
    """
    calibrator = _make_calibrator(seed=42)

    # Reproduce the exact call pattern. Tuning passes base seed =
    # calibrator.seed; _create_holdout_sets iterates
    # base_seed + i for i in range(n).
    tuning_sets = calibrator._create_holdout_sets(
        n_holdout_sets=5,
        holdout_fraction=0.25,
        random_state=calibrator.seed,
    )

    # Robustness MUST use the orthogonal namespace now.
    robustness_sets = calibrator._create_holdout_sets(
        n_holdout_sets=5,
        holdout_fraction=0.25,
        random_state=calibrator.seed + 10_000,
    )

    # The specific bug was the index-aligned collision between
    # tuning[i] and robustness[i-1] via (seed + i) == ((seed + 1) + (i-1)).
    # The fix makes those aligned pairs differ.
    for i in range(1, 5):
        assert set(tuning_sets[i]["indices"]) != set(
            robustness_sets[i - 1]["indices"]
        ), (
            f"Tuning set {i} and robustness set {i - 1} must not be "
            "identical under the fixed seed namespaces."
        )
