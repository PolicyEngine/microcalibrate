"""
Test the calibration process with L0 regularization.
"""

from microcalibrate.calibration import Calibration
from microcalibrate.utils.l0 import evaluate_sparse_weights
import logging
import numpy as np
import pandas as pd


def test_calibration_with_l0_regularization() -> None:
    # Create a sample dataset for testing
    random_generator = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "age": np.append(random_generator.integers(18, 70, size=500), 71),
            "income": random_generator.normal(40000, 10000, size=501),
        }
    )

    weights = np.ones(len(data))

    # Calculate target values:
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
            (targets_matrix["income_aged_30_40"] * weights).sum() * 1.3,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 0.9,
            (targets_matrix["income_aged_50_60"] * weights).sum() * 1.5,
            (targets_matrix["income_aged_60_70"] * weights).sum() * 1.2,
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
        regularize_with_l0=True,
        csv_path="tests/calibration_log.csv",
    )

    performance_df = calibrator.calibrate()
    weights = calibrator.weights
    sparse_weights = calibrator.sparse_weights

    percentage_within_10 = evaluate_sparse_weights(
        optimised_weights=sparse_weights,
        estimate_matrix=targets_matrix,
        targets_array=targets,
    )

    sparse_calibration_log = pd.read_csv(
        str(calibrator.csv_path).replace(".csv", "_sparse.csv")
    )

    # Get the final epoch average relative absolute error from the dense calibration log
    final_epoch = performance_df["epoch"].max()
    final_epoch_data = performance_df[performance_df["epoch"] == final_epoch]
    avg_error_dense_final_epoch = final_epoch_data["rel_abs_error"].mean()

    # Get final epoch data from sparse calibration log
    sparse_final_epoch = sparse_calibration_log["epoch"].max()
    sparse_final_epoch_data = sparse_calibration_log[
        sparse_calibration_log["epoch"] == sparse_final_epoch
    ]
    avg_error_sparse_final_epoch = sparse_final_epoch_data[
        "rel_abs_error"
    ].mean()

    assert (
        avg_error_sparse_final_epoch < 0.05
    ), "Final average relative absolute error is more than 5%."

    percentage_below_threshold = (
        (sparse_weights < 0.5).sum() / len(sparse_weights) * 100
    )
    assert (
        percentage_below_threshold > 10
    ), f"Only {percentage_below_threshold:.1f}% of sparse weights are below 0.5 (expected > 10%)"
