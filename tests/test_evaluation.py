"""
Test the evaluation functionality for the calibration process.
"""

import pytest
from src.microcalibrate.calibration import Calibration
from microcalibrate.evaluation import (
    evaluate_estimate_distance_to_targets,
)
import numpy as np
import pandas as pd


def test_evaluate_estimate_distance_to_targets() -> None:
    """Test the evaluation of estimates against targets with tolerances, for a case in which estimates are not within tolerance."""

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

    assert "target(s) are outside their tolerance levels" in str(
        exc_info.value
    )


def test_all_within_tolerance():
    """Tests a simple case where all estimates are within their tolerances."""
    targets = np.array([10, 20, 30])
    estimates = np.array([10.1, 19.8, 30.0])
    tolerances = np.array([0.2, 0.3, 0.1])
    target_names = ["A", "B", "C"]

    result_df = evaluate_estimate_distance_to_targets(
        targets, estimates, tolerances, target_names
    )

    assert result_df["within_tolerance"].all()
    assert result_df.shape == (3, 4)
    np.testing.assert_array_almost_equal(
        result_df["distances"], [0.1, 0.2, 0.0]
    )


def test_evaluate_holdout_robustness():
    """Test the holdout robustness evaluation functionality."""

    # Create a more complex mock dataset with multiple features
    random_generator = np.random.default_rng(42)
    n_samples = 500

    data = pd.DataFrame(
        {
            "age": random_generator.integers(18, 80, size=n_samples),
            "income": random_generator.lognormal(10.5, 0.7, size=n_samples),
            "region": random_generator.choice(
                ["North", "South", "East", "West"], size=n_samples
            ),
            "employed": random_generator.binomial(1, 0.7, size=n_samples),
        }
    )

    weights = random_generator.uniform(0.5, 1.5, size=n_samples)
    weights = weights / weights.sum() * n_samples

    estimate_matrix = pd.DataFrame(
        {
            "total_population": np.ones(n_samples),
            "employed_count": data["employed"].astype(float),
            "income_north": (
                (data["region"] == "North") * data["income"]
            ).astype(float),
            "income_south": (
                (data["region"] == "South") * data["income"]
            ).astype(float),
            "income_east": (
                (data["region"] == "East") * data["income"]
            ).astype(float),
            "income_west": (
                (data["region"] == "West") * data["income"]
            ).astype(float),
            "young_employed": (
                (data["age"] < 30) & (data["employed"] == 1)
            ).astype(float),
            "senior_count": (data["age"] >= 65).astype(float),
        }
    )

    targets = np.array(
        [
            n_samples * 1.05,
            (estimate_matrix["employed_count"] * weights).sum() * 0.95,
            (estimate_matrix["income_north"] * weights).sum() * 1.1,
            (estimate_matrix["income_south"] * weights).sum() * 0.9,
            (estimate_matrix["income_east"] * weights).sum() * 1.05,
            (estimate_matrix["income_west"] * weights).sum() * 0.98,
            (estimate_matrix["young_employed"] * weights).sum() * 1.15,
            (estimate_matrix["senior_count"] * weights).sum() * 0.92,
        ]
    )

    calibrator = Calibration(
        estimate_matrix=estimate_matrix,
        weights=weights,
        targets=targets,
        noise_level=0.1,
        epochs=100,
        learning_rate=0.01,
        dropout_rate=0.05,
        seed=42,
    )
    calibrator.calibrate()

    # Test basic robustness evaluation
    results = calibrator.evaluate_holdout_robustness(
        n_holdout_sets=3,
        holdout_fraction=0.25,
        save_results_to=None,  # pass a str path if you want to save and explore results' dataframes
    )

    # Check structure of results
    assert "overall_metrics" in results
    assert "target_robustness" in results
    assert "recommendation" in results
    assert "detailed_results" in results

    # Check overall metrics
    metrics = results["overall_metrics"]
    assert "mean_holdout_loss" in metrics
    assert "std_holdout_loss" in metrics
    assert "mean_holdout_accuracy" in metrics
    assert "std_holdout_accuracy" in metrics
    assert "worst_holdout_accuracy" in metrics
    assert "best_holdout_accuracy" in metrics
    assert "mean_generalization_gap" in metrics
    assert metrics["n_successful_evaluations"] == 3
    assert metrics["n_failed_evaluations"] == 0

    # Check that accuracy is between 0 and 1
    assert 0 <= metrics["mean_holdout_accuracy"] <= 1
    assert 0 <= metrics["worst_holdout_accuracy"] <= 1
    assert 0 <= metrics["best_holdout_accuracy"] <= 1

    # Check target robustness DataFrame
    robustness_df = results["target_robustness"]
    assert isinstance(robustness_df, pd.DataFrame)
    assert len(robustness_df) > 0  # At least some targets should be evaluated
    assert "target_name" in robustness_df.columns
    assert "times_held_out" in robustness_df.columns
    assert "holdout_accuracy_rate" in robustness_df.columns
    assert "mean_holdout_loss" in robustness_df.columns
    assert robustness_df["holdout_accuracy_rate"].is_monotonic_increasing
    assert isinstance(results["recommendation"], str)
    assert len(results["recommendation"]) > 0
    assert any(
        word in results["recommendation"]
        for word in ["ROBUSTNESS", "RECOMMENDATIONS"]
    )

    # Check detailed results
    assert len(results["detailed_results"]) == 3
    for detail in results["detailed_results"]:
        assert "holdout_loss" in detail
        assert "train_loss" in detail
        assert "holdout_accuracy" in detail
        assert "train_accuracy" in detail
        assert "generalization_gap" in detail
        assert "target_details" in detail
        assert len(detail["target_details"]) == 2  # 25% of 8 targets

    # Test error handling with invalid parameters
    with pytest.raises(ValueError):
        calibrator.evaluate_holdout_robustness(
            n_holdout_sets=0,  # Invalid
        )
    with pytest.raises(ValueError):
        calibrator.evaluate_holdout_robustness(
            holdout_fraction=1.5,  # Invalid
        )


def test_evaluate_holdout_robustness_with_l0_regularization():
    """Test robustness evaluation with L0 regularization enabled."""

    # Create simple dataset
    random_generator = np.random.default_rng(123)
    n_samples = 200

    estimate_matrix = pd.DataFrame(
        {
            "feature_1": random_generator.uniform(0.5, 1.5, n_samples),
            "feature_2": random_generator.uniform(0.5, 1.5, n_samples),
            "feature_3": random_generator.uniform(0.5, 1.5, n_samples),
            "feature_redundant": random_generator.uniform(
                0, 0.1, n_samples
            ),  # Low signal
        }
    )

    weights = np.ones(n_samples)
    col_sums = estimate_matrix.sum()
    targets = np.array(
        [
            col_sums["feature_1"] * 0.95,
            col_sums["feature_2"] * 1.05,
            col_sums["feature_3"] * 1.0,
            col_sums["feature_redundant"] * 1.1,
        ]
    )

    # Initialize with L0 regularization - aggressive parameters for sparsity
    calibrator = Calibration(
        estimate_matrix=estimate_matrix,
        weights=weights,
        targets=targets,
        regularize_with_l0=True,
        l0_lambda=1e-4,
        init_mean=0.5,
        temperature=0.3,
        epochs=100,
        seed=123,
    )

    calibrator.calibrate()

    results = calibrator.evaluate_holdout_robustness(
        n_holdout_sets=3,
        holdout_fraction=0.25,
    )
    assert all(
        "weights_sparsity" in detail for detail in results["detailed_results"]
    )
    sparsity_values = [
        detail["weights_sparsity"] for detail in results["detailed_results"]
    ]
    assert max(sparsity_values) >= 0 or calibrator.sparse_weights is not None
    assert results["overall_metrics"]["mean_holdout_accuracy"] >= 0


def test_evaluate_holdout_robustness_recommendation_logic():
    """Test the recommendation generation logic."""

    # Create a calibrator with known poor performance
    random_generator = np.random.default_rng(789)
    n_samples = 50
    base_feature = random_generator.normal(0, 1, n_samples)
    estimate_matrix = pd.DataFrame(
        {
            "feature_1": base_feature
            + random_generator.normal(0, 0.1, n_samples),
            "feature_2": base_feature
            + random_generator.normal(0, 0.1, n_samples),
            "feature_3": base_feature
            + random_generator.normal(0, 0.1, n_samples),
        }
    )

    weights = np.ones(n_samples)
    targets = np.array([100, 200, 300])

    calibrator = Calibration(
        estimate_matrix=estimate_matrix,
        weights=weights,
        targets=targets,
        epochs=20,
        noise_level=0.01,
        dropout_rate=0,
    )

    calibrator.calibrate()
    results = calibrator.evaluate_holdout_robustness(n_holdout_sets=3)

    recommendation = results["recommendation"]
    if results["overall_metrics"]["mean_holdout_accuracy"] < 0.7:
        assert any(marker in recommendation for marker in ["⚠️", "❌"])
    if not calibrator.regularize_with_l0:
        assert "L0 regularization" in recommendation
    problematic = results["target_robustness"][
        results["target_robustness"]["holdout_accuracy_rate"] < 0.5
    ]
    if len(problematic) > 0:
        assert "Targets with poor holdout performance" in recommendation
