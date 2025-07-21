"""
Test the different calibration diagnostics and user warnings.
"""

from src.microcalibrate.calibration import Calibration
import logging
import numpy as np
import pandas as pd


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


def test_calibration_analytical_solution(caplog) -> None:
    """Test the calibration process produces the expected analytical target evaluation reporting."""

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
            (targets_matrix["income_aged_20_30"] * weights).sum() * 2,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 2,
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

    analytical_assessment = calibrator.assess_analytical_solution()

    assert set(analytical_assessment["target_added"]) == set(
        list(targets_matrix.columns)
    ), "Not all targets were added to the assessment."

    assert (
        analytical_assessment["loss"].all() != 0
    ), "Loss values should not be zero."
