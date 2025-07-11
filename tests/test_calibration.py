"""
Test the calibration process.
"""

from src.microcalibrate.calibration import Calibration
import logging
import numpy as np
import pandas as pd


def test_calibration_basic() -> None:
    """Test the calibration process with a basic setup where the weights are already correctly calibrated to fit the targets."""

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
            (targets_matrix["income_aged_20_30"] * weights).sum() * 1,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 1,
        ]
    )

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=528,
        learning_rate=0.01,
        dropout_rate=0,
    )

    # Call calibrate method on our data and targets of interest
    performance_df = calibrator.calibrate()

    final_estimates = calibrator.estimate()

    # Check that the calibration process has improved the weights
    np.testing.assert_allclose(
        final_estimates,
        targets,
        rtol=0.01,  # relative tolerance
        err_msg="Calibrated totals do not match target values",
    )


def test_calibration_harder_targets() -> None:
    """Test the calibration process with targets that are 15% higher than the sum of the orginal weights."""

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
            (targets_matrix["income_aged_20_30"] * weights).sum() * 1.15,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 1.15,
        ]
    )

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=528,
        learning_rate=0.01,
        dropout_rate=0,
        csv_path="calibration_log.csv",
    )

    # Call calibrate method on our data and targets of interest
    performance_df = calibrator.calibrate()

    final_estimates = calibrator.estimate()

    # Check that the calibration process has improved the weights
    np.testing.assert_allclose(
        final_estimates,
        targets,
        rtol=0.01,  # relative tolerance
        err_msg="Calibrated totals do not match target values",
    )
