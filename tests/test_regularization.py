"""
Test the calibration process with L0 regularization.
"""

from microcalibrate.calibration import Calibration
from microcalibrate.evaluation import evaluate_sparse_weights
import logging
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data():
    """Create sample dataset and targets for L0 hyperparameter tuning tests."""
    random_generator = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "age": np.append(random_generator.integers(18, 70, size=500), 71),
            "income": random_generator.normal(40000, 10000, size=501),
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
            (targets_matrix["income_aged_30_40"] * weights).sum() * 1.3,
            (targets_matrix["income_aged_40_50"] * weights).sum() * 0.9,
            (targets_matrix["income_aged_50_60"] * weights).sum() * 1.5,
            (targets_matrix["income_aged_60_70"] * weights).sum() * 1.2,
        ]
    )

    return {
        "targets_matrix": targets_matrix,
        "weights": weights,
        "targets": targets,
    }


def test_calibration_with_l0_regularization(test_data) -> None:
    "Test calibration with L0 regularization."
    targets_matrix = test_data["targets_matrix"]
    weights = test_data["weights"]
    targets = test_data["targets"]

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


def test_l0_hyperparameter_tuning_with_holdouts(test_data) -> None:
    """Test L0 hyperparameter tuning with holdout validation."""
    targets_matrix = test_data["targets_matrix"]
    weights = test_data["weights"]
    targets = test_data["targets"]

    # Create calibrator instance
    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=200,
        learning_rate=0.01,
        dropout_rate=0,
        regularize_with_l0=False,
    )

    # Test hyperparameter tuning
    best_params = calibrator.tune_l0_hyperparameters(
        n_trials=20,  # Fewer trials for testing
        epochs_per_trial=50,  # Shorter epochs for quick testing
        objectives_balance={
            "loss": 1.0,
            "accuracy": 100.0,  # Prioritize hitting targets
            "sparsity": 30.0,
        },
        n_jobs=1,
    )

    # Verify that best_params contains expected keys
    assert "l0_lambda" in best_params, "Missing l0_lambda in best parameters"
    assert "init_mean" in best_params, "Missing init_mean in best parameters"
    assert (
        "temperature" in best_params
    ), "Missing temperature in best parameters"
    assert (
        "mean_val_loss" in best_params
    ), "Missing mean_val_loss in best parameters"
    assert (
        "mean_val_accuracy" in best_params
    ), "Missing mean_val_accuracy in best parameters"
    assert "sparsity" in best_params, "Missing sparsity in best parameters"
    assert (
        "holdout_objectives" in best_params
    ), "Missing holdout_objectives in best parameters"

    # Verify parameter ranges
    assert (
        1e-6 <= best_params["l0_lambda"] <= 1e-4
    ), f"l0_lambda {best_params['l0_lambda']} out of range"
    assert (
        0.5 <= best_params["init_mean"] <= 0.999
    ), f"init_mean {best_params['init_mean']} out of range"
    assert (
        0.5 <= best_params["temperature"] <= 2.0
    ), f"temperature {best_params['temperature']} out of range"

    # Verify metrics are reasonable
    assert (
        0 <= best_params["mean_val_accuracy"] <= 1
    ), "mean_val_accuracy should be between 0 and 1"
    assert (
        0 <= best_params["sparsity"] <= 1
    ), "sparsity should be between 0 and 1"
    assert (
        best_params["mean_val_loss"] >= 0
    ), "mean_val_loss should be non-negative"

    best_params["evaluation_history"].to_csv(
        "tests/l0_hyperparameter_tuning_history_with_holdouts.csv", index=False
    )

    # Now run calibration with the best parameters
    calibrator.l0_lambda = best_params["l0_lambda"]
    calibrator.init_mean = best_params["init_mean"]
    calibrator.temperature = best_params["temperature"]
    calibrator.regularize_with_l0 = True

    # Run the full calibration
    performance_df = calibrator.calibrate()
    sparse_weights = calibrator.sparse_weights

    assert (
        sparse_weights is not None
    ), "Sparse weights should be generated with L0 regularization"

    # Evaluate the final calibration
    percentage_within_10 = evaluate_sparse_weights(
        optimised_weights=sparse_weights,
        estimate_matrix=targets_matrix,
        targets_array=targets,
    )

    # The tuned parameters should give reasonable results
    assert (
        percentage_within_10 > 50
    ), f"Only {percentage_within_10:.1f}% of targets within 10% (expected > 50%)"

    # Check that we achieved some sparsity
    actual_sparsity = np.mean(sparse_weights == 0)
    assert (
        actual_sparsity > 0.1
    ), f"Sparsity {actual_sparsity:.1%} is too low (expected > 10%)"


def test_l0_hyperparameter_tuning_without_holdouts(test_data) -> None:
    """Test L0 hyperparameter tuning without holdout validation (simpler case)."""
    targets_matrix = test_data["targets_matrix"]
    weights = test_data["weights"]
    targets = test_data["targets"]

    # Create calibrator instance
    calibrator = Calibration(
        estimate_matrix=targets_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.05,
        epochs=200,
        learning_rate=0.01,
        dropout_rate=0,
        regularize_with_l0=False,
    )

    # Test hyperparameter tuning WITHOUT holdouts
    best_params = calibrator.tune_l0_hyperparameters(
        n_trials=10,
        epochs_per_trial=30,
        n_holdout_sets=1,  # Single holdout set
        holdout_fraction=0,  # No holdouts - use all data for both training and validation
        objectives_balance={
            "loss": 1.0,
            "accuracy": 100.0,
            "sparsity": 30.0,
        },
        n_jobs=1,
    )

    # Verify that best_params contains expected keys
    assert "l0_lambda" in best_params, "Missing l0_lambda in best parameters"
    assert "init_mean" in best_params, "Missing init_mean in best parameters"
    assert (
        "temperature" in best_params
    ), "Missing temperature in best parameters"
    assert (
        "mean_val_loss" in best_params
    ), "Missing mean_val_loss in best parameters"
    assert (
        "mean_val_accuracy" in best_params
    ), "Missing mean_val_accuracy in best parameters"
    assert "sparsity" in best_params, "Missing sparsity in best parameters"

    # Verify parameter ranges
    assert (
        1e-6 <= best_params["l0_lambda"] <= 1e-4
    ), f"l0_lambda {best_params['l0_lambda']} out of range"
    assert (
        0.5 <= best_params["init_mean"] <= 0.999
    ), f"init_mean {best_params['init_mean']} out of range"
    assert (
        0.5 <= best_params["temperature"] <= 2.0
    ), f"temperature {best_params['temperature']} out of range"

    # Verify metrics are reasonable
    assert (
        0 <= best_params["mean_val_accuracy"] <= 1
    ), "mean_val_accuracy should be between 0 and 1"
    assert (
        0 <= best_params["sparsity"] <= 1
    ), "sparsity should be between 0 and 1"
    assert (
        best_params["mean_val_loss"] >= 0
    ), "mean_val_loss should be non-negative"

    # When there are no holdouts, n_holdout_sets should be 1 and aggregation should work
    assert best_params["n_holdout_sets"] == 1, "Should have 1 holdout set"
    assert (
        "holdout_objectives" in best_params
    ), "Should still have holdout_objectives"
    assert (
        len(best_params["holdout_objectives"]) == 1
    ), "Should have exactly 1 objective"

    best_params["evaluation_history"].to_csv(
        "tests/l0_hyperparameter_tuning_history_without_holdouts.csv",
        index=False,
    )

    # Run calibration with the best parameters
    calibrator.l0_lambda = best_params["l0_lambda"]
    calibrator.init_mean = best_params["init_mean"]
    calibrator.temperature = best_params["temperature"]
    calibrator.regularize_with_l0 = True

    # Run the full calibration
    performance_df = calibrator.calibrate()
    sparse_weights = calibrator.sparse_weights

    assert (
        sparse_weights is not None
    ), "Sparse weights should be generated with L0 regularization"

    # Evaluate the final calibration
    percentage_within_10 = evaluate_sparse_weights(
        optimised_weights=sparse_weights,
        estimate_matrix=targets_matrix,
        targets_array=targets,
    )

    # The tuned parameters should give reasonable results
    assert (
        percentage_within_10 > 50
    ), f"Only {percentage_within_10:.1f}% of targets within 10% (expected > 50%)"

    # Check that we achieved some sparsity
    actual_sparsity = np.mean(sparse_weights == 0)
    assert (
        actual_sparsity > 0.05
    ), f"Sparsity {actual_sparsity:.1%} is too low (expected > 5%)"
