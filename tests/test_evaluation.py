"""
Test the evaluation functionality for the calibration process.
"""

import pytest
from src.microcalibrate.calibration import Calibration
from src.microcalibrate.evalutation import (
    evaluate_estimate_distance_to_targets,
)
import numpy as np
import pandas as pd


def test_evaluate_estimate_distance_to_targets() -> None:
    """Test the evaluation of estimates against targets with tolerances."""

    # Create a mock dataset with age and income
    random_generator = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "age": random_generator.integers(18, 70, size=100),
            "income": random_generator.normal(40000, 50000, size=100),
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
            (targets_matrix["income_aged_20_30"] * weights).sum() * 50,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 50,
        ]
    )

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=50,
        learning_rate=0.01,
        dropout_rate=0,
    )

    performance_df = calibrator.calibrate()
    final_estimates = calibrator.estimate()
    tolerances = np.array([0.001, 0.005])

    # Evaluate the estimates against the targets without raising an error
    evals_df = evaluate_estimate_distance_to_targets(
        targets=targets,
        estimates=final_estimates,
        tolerances=tolerances,
        target_names=["Income Aged 20-30", "Income Aged 40-50"],
        raise_on_error=False,
    )

    # Check that the evaluation DataFrame has the expected structure
    assert set(evals_df.columns) == {
        "target_names",
        "distances",
        "tolerances",
        "within_tolerance",
    }

    # Evaluate the estimates against the targets raising an error
    with pytest.raises(ValueError) as exc_info:
        evals_df = evaluate_estimate_distance_to_targets(
            targets=targets,
            estimates=final_estimates,
            tolerances=tolerances,
            target_names=["Income Aged 20-30", "Income Aged 40-50"],
            raise_on_error=True,
        )

    assert "targets are outside their tolerance levels" in str(exc_info.value)
