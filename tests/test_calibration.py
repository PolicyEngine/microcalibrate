"""
Test the calibration process.
"""

import numpy as np
import pandas as pd

from src.microcalibrate.calibration import Calibration
from src.microcalibrate.utils import simulate_contradictory_data


def test_calibration_basic() -> None:
    """Test the calibration process with a basic setup where the weights are already correctly calibrated to fit the targets."""

    # Create a mock dataset with age and income
    random_generator = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "age": random_generator.integers(18, 70, size=100),
            "income": random_generator.normal(40000, 10000, size=100),
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
        loss_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=528,
        learning_rate=0.01,
        dropout_rate=0,
        subsample_every=0,
    )

    # Call calibrate method on our data and targets of interest
    calibrator.calibrate()

    final_estimates = (
        targets_matrix.mul(calibrator.weights, axis=0).sum().values
    )

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
            "income": random_generator.normal(40000, 10000, size=100),
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
        loss_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=528,
        learning_rate=0.01,
        dropout_rate=0,
        subsample_every=0,
    )

    # Call calibrate method on our data and targets of interest
    calibrator.calibrate()

    final_estimates = (
        targets_matrix.mul(calibrator.weights, axis=0).sum().values
    )

    # Check that the calibration process has improved the weights
    np.testing.assert_allclose(
        final_estimates,
        targets,
        rtol=0.01,  # relative tolerance
        err_msg="Calibrated totals do not match target values",
    )


def test_calibration_responds_to_contradiction() -> None:
    """Test that calibration performance degrades predicably with increased contradiction"""

    results = []
    for contradiction_factor in [0.1, 0.2]:

        sample_df, totals = simulate_contradictory_data(
            T=6000, k=3, c=contradiction_factor, n=30
        )

        metrics_matrix = pd.DataFrame(
            {
                "y_all": sample_df["y_ij"],
                "y_stratum_1": (
                    (sample_df["stratum_id"] == 1).astype(float)
                    * sample_df["y_ij"]
                ),
                "y_stratum_2": (
                    (sample_df["stratum_id"] == 2).astype(float)
                    * sample_df["y_ij"]
                ),
                "y_stratum_3": (
                    (sample_df["stratum_id"] == 3).astype(float)
                    * sample_df["y_ij"]
                ),
            }
        )

        calibrator = Calibration(
            loss_matrix=metrics_matrix,
            weights=sample_df["w_ij"],
            targets=np.hstack(
                [totals["T_official"], totals["S_star_official"]]
            ),
        )

        calibrator.calibrate()

        new_weights = calibrator.weights
        new_estimates = np.matmul(metrics_matrix.T, new_weights)

        overall_loss = (totals["T_official"] - new_estimates["y_all"]) ** 2
        strata_loss = (
            (totals["S_star_official"][0] - new_estimates["y_stratum_1"]) ** 2
            + (totals["S_star_official"][1] - new_estimates["y_stratum_2"])
            ** 2
            + (totals["S_star_official"][2] - new_estimates["y_stratum_3"])
            ** 2
        )

        results.append([contradiction_factor, overall_loss, strata_loss])

    results_df = pd.DataFrame(
        results,
        columns=["contradiction_factor", "overall_loss", "strata_loss"],
    )

    assert results_df[
        "overall_loss"
    ].is_monotonic_increasing, "Overall loss is not increasing"
    assert results_df[
        "strata_loss"
    ].is_monotonic_increasing, "Strata loss is not increasing"
