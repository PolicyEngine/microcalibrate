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


def test_calibration_warnings_system(caplog) -> None:
    """Test the calibration process raises the expected warnings in response to certain inputs."""

    # Create a sample dataset for testing
    random_generator = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "age": np.append(random_generator.integers(18, 70, size=120), 71),
            "income": random_generator.normal(40000, 10000, size=121),
        }
    )

    weights = np.ones(len(data))

    # Calculate target values:
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
            "income_aged_71": (data["age"] == 71).astype(float)
            * data["income"],
            "income_aged_72": (data["age"] == 72).astype(float)
            * data["income"],
        }
    )

    # Add specific characteristics to the targets to trigger warnings
    targets = np.array(
        [
            (targets_matrix["income_aged_20_30"] * weights * 1000).sum(),
            (targets_matrix["income_aged_40_50"] * weights * 1.15).sum(),
            (targets_matrix["income_aged_71"] * weights * 1.15).sum(),
            (targets_matrix["income_aged_71"] * weights * -1.15).sum(),
        ]
    )

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=128,
        learning_rate=0.01,
        dropout_rate=0,
    )

    with caplog.at_level(logging.WARNING, logger="microcalibrate.calibration"):
        performance_df = calibrator.calibrate()

    log_text = "\n".join(record.getMessage() for record in caplog.records)

    # Expected fragments
    assert (
        "Target income_aged_20_30" in log_text
        and "orders of magnitude" in log_text
    ), "Magnitude-mismatch warning not emitted."

    assert (
        "Target income_aged_71 is supported by only" in log_text
    ), "Low-support warning not emitted."

    assert (
        "Column income_aged_72 has a zero estimate sum" in log_text
    ), "Zero estimate sum warning not emitted."

    assert (
        "Some targets are negative" in log_text
    ), "Negative target warning not emitted."


def test_calibration_excluded_targets() -> None:
    """Test the calibration process works correctly with excluded targets."""

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
                (data["age"] >= 20) & (data["age"] < 30)
            ).astype(float)
            * data["income"],
            "income_aged_30_40": (
                (data["age"] >= 30) & (data["age"] < 40)
            ).astype(float)
            * data["income"],
            "income_aged_40_50": (
                (data["age"] >= 40) & (data["age"] < 50)
            ).astype(float)
            * data["income"],
            "income_aged_50_60": (
                (data["age"] >= 50) & (data["age"] < 60)
            ).astype(float)
            * data["income"],
            "income_aged_60_70": (
                (data["age"] >= 60) & (data["age"] <= 70)
            ).astype(float)
            * data["income"],
        }
    )
    targets = np.array(
        [
            (targets_matrix["income_aged_20_30"] * weights).sum() * 1.2,
            (targets_matrix["income_aged_30_40"] * weights).sum() * 1.2,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 1.2,
            (targets_matrix["income_aged_50_60"] * weights).sum() * 1.2,
            (targets_matrix["income_aged_60_70"] * weights).sum() * 1.2,
        ]
    )

    excluded_targets = ["income_aged_20_30"]

    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=528,
        learning_rate=0.01,
        dropout_rate=0,
        excluded_targets=excluded_targets,
    )

    first_performance_df = calibrator.calibrate()
    first_calibration_estimates = calibrator.estimate()

    assert len(first_calibration_estimates) == len(
        np.array(calibrator.targets)
    ), "Excluded target income_aged_20_30 should not be calibrated."

    # iteratively exclude new targets and calibrate
    new_target_to_exclude = ["income_aged_30_40"]
    calibrator.exclude_targets(new_target_to_exclude)
    second_performance_df = calibrator.calibrate()
    second_calibration_estimates = calibrator.estimate()

    assert (
        new_target_to_exclude[0] not in calibrator.target_names
    ), f"Target {new_target_to_exclude[0]} should be excluded from calibration."
