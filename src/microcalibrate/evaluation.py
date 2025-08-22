import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from microcalibrate.utils.metrics import loss, pct_close

logger = logging.getLogger(__name__)


def evaluate_estimate_distance_to_targets(
    targets: np.ndarray,
    estimates: np.ndarray,
    tolerances: np.ndarray,
    target_names: Optional[List[str]] = None,
    raise_on_error: Optional[bool] = False,
):
    """
    Evaluate the distance between estimates and targets against tolerances.

    Args:
        targets (np.ndarray): The ground truth target values.
        estimates (np.ndarray): The estimated values to compare against the targets.
        tolerances (np.ndarray): The acceptable tolerance levels for each target.
        target_names (Optional[List[str]]): The names of the targets for reporting.
        raise_on_error (Optional[bool]): If True, raises an error if any estimate is outside its tolerance. Default is False.

    Returns:
        evals (pd.DataFrame): A DataFrame containing the evaluation results, including:
            - target_names: Names of the targets (if provided).
            - distances: The absolute differences between estimates and targets.
            - tolerances: The tolerance levels for each target.
            - within_tolerance: Boolean array indicating if each estimate is within its tolerance.
    """
    if targets.shape != estimates.shape or targets.shape != tolerances.shape:
        raise ValueError(
            "Targets, estimates, and tolerances must have the same shape."
        )

    distances = np.abs(estimates - targets)
    within_tolerance = distances <= tolerances

    evals = {
        "target_names": (
            target_names
            if target_names is not None
            else list(np.nan for _ in targets)
        ),
        "distances": distances,
        "tolerances": tolerances,
        "within_tolerance": within_tolerance,
    }

    num_outside_tolerance = (~within_tolerance).sum()
    if raise_on_error and num_outside_tolerance > 0:
        raise ValueError(
            f"{num_outside_tolerance} target(s) are outside their tolerance levels."
        )

    return pd.DataFrame(evals)


def evaluate_sparse_weights(
    optimised_weights: Union[torch.Tensor, np.ndarray],
    estimate_matrix: Union[torch.Tensor, np.ndarray],
    targets_array: Union[torch.Tensor, np.ndarray],
    label: Optional[str] = "L0 Sparse Weights",
) -> float:
    """
    Evaluate the performance of sparse weights against targets.

    Args:
        optimised_weights (torch.Tensor or np.ndarray): The optimised weights.
        estimate_matrix (torch.Tensor or pd.DataFrame): The estimate matrix.
        targets_array (torch.Tensor or np.ndarray): The target values.
        label (str): A label for logging purposes.

    Returns:
        float: The percentage of estimates within 10% of the targets.
    """
    # Convert all inputs to NumPy arrays right at the start
    optimised_weights_np = (
        optimised_weights.numpy()
        if hasattr(optimised_weights, "numpy")
        else np.asarray(optimised_weights)
    )
    estimate_matrix_np = (
        estimate_matrix.numpy()
        if hasattr(estimate_matrix, "numpy")
        else np.asarray(estimate_matrix)
    )
    targets_array_np = (
        targets_array.numpy()
        if hasattr(targets_array, "numpy")
        else np.asarray(targets_array)
    )

    logging.info(f"\n\n---{label}: reweighting quick diagnostics----\n")
    logging.info(
        f"{np.sum(optimised_weights_np == 0)} are zero, "
        f"{np.sum(optimised_weights_np != 0)} weights are nonzero"
    )

    # All subsequent calculations use the guaranteed NumPy versions
    estimate = optimised_weights_np @ estimate_matrix_np

    rel_error = (
        ((estimate - targets_array_np) + 1) / (targets_array_np + 1)
    ) ** 2
    within_10_percent_mask = np.abs(estimate - targets_array_np) <= (
        0.10 * np.abs(targets_array_np)
    )
    percent_within_10 = np.mean(within_10_percent_mask) * 100
    logging.info(
        f"rel_error: min: {np.min(rel_error):.2f}\n"
        f"max: {np.max(rel_error):.2f}\n"
        f"mean: {np.mean(rel_error):.2f}\n"
        f"median: {np.median(rel_error):.2f}\n"
        f"Within 10% of target: {percent_within_10:.2f}%"
    )
    logging.info("Relative error over 100% for:")
    for i in np.where(rel_error > 1)[0]:
        # Keep this check, as Tensors won't have a .columns attribute
        if hasattr(estimate_matrix, "columns"):
            logging.info(f"target_name: {estimate_matrix.columns[i]}")
        else:
            logging.info(f"target_index: {i}")

        logging.info(f"target_value: {targets_array_np[i]}")
        logging.info(f"estimate_value: {estimate[i]}")
        logging.info(f"has rel_error: {rel_error[i]:.2f}\n")
    logging.info("---End of reweighting quick diagnostics------")
    return percent_within_10


def _evaluate_single_holdout_robustness(
    calibration,
    holdout_set: Dict[str, Any],
    holdout_idx: int,
    n_holdout_sets: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single holdout set for robustness analysis.

    Args:
        calibration: Calibration instance
        holdout_set: Dictionary with holdout information
        holdout_idx: Index of current holdout set
        n_holdout_sets: Total number of holdout sets

    Returns:
        Dictionary with evaluation results or None if failed
    """
    try:
        logger.info(
            f"Evaluating holdout set {holdout_idx + 1}/{n_holdout_sets}"
        )

        # Run calibration on training targets
        start_time = pd.Timestamp.now()
        calibration.calibrate()
        calibration_time = (pd.Timestamp.now() - start_time).total_seconds()

        # Get final weights (sparse if using L0, otherwise regular)
        final_weights = (
            calibration.sparse_weights
            if calibration.sparse_weights is not None
            else calibration.weights
        )

        # Evaluate on all targets
        weights_tensor = torch.tensor(
            final_weights, dtype=torch.float32, device=calibration.device
        )

        # Get estimates for all targets using original estimate function/matrix
        if calibration.original_estimate_matrix is not None:
            original_matrix_tensor = torch.tensor(
                calibration.original_estimate_matrix.values,
                dtype=torch.float32,
                device=calibration.device,
            )
            all_estimates = (
                (weights_tensor @ original_matrix_tensor).cpu().numpy()
            )
        else:
            all_estimates = (
                calibration.original_estimate_function(weights_tensor)
                .cpu()
                .numpy()
            )

        # Calculate metrics for holdout vs training sets
        holdout_indices = holdout_set["indices"]
        train_indices = [
            i
            for i in range(len(calibration.original_target_names))
            if i not in holdout_indices
        ]

        holdout_estimates = all_estimates[holdout_indices]
        holdout_targets = calibration.original_targets[holdout_indices]
        holdout_names = holdout_set["names"]

        train_estimates = all_estimates[train_indices]
        train_targets = calibration.original_targets[train_indices]

        # Calculate losses and accuracies
        holdout_loss = loss(
            torch.tensor(
                holdout_estimates,
                dtype=torch.float32,
                device=calibration.device,
            ),
            torch.tensor(
                holdout_targets, dtype=torch.float32, device=calibration.device
            ),
            None,
        ).item()

        holdout_accuracy = pct_close(
            torch.tensor(
                holdout_estimates,
                dtype=torch.float32,
                device=calibration.device,
            ),
            torch.tensor(
                holdout_targets, dtype=torch.float32, device=calibration.device
            ),
        )

        train_loss = loss(
            torch.tensor(
                train_estimates, dtype=torch.float32, device=calibration.device
            ),
            torch.tensor(
                train_targets, dtype=torch.float32, device=calibration.device
            ),
            None,
        ).item()

        train_accuracy = pct_close(
            torch.tensor(
                train_estimates, dtype=torch.float32, device=calibration.device
            ),
            torch.tensor(
                train_targets, dtype=torch.float32, device=calibration.device
            ),
        )

        # Calculate per-target metrics for holdout targets
        target_details = []
        for idx, name in enumerate(holdout_names):
            rel_error = (
                holdout_estimates[idx] - holdout_targets[idx]
            ) / holdout_targets[idx]
            target_details.append(
                {
                    "target_name": name,
                    "target_value": holdout_targets[idx],
                    "estimate": holdout_estimates[idx],
                    "relative_error": rel_error,
                    "within_10pct": abs(rel_error) <= 0.1,
                }
            )

        generalization_gap = holdout_loss - train_loss
        accuracy_gap = train_accuracy - holdout_accuracy

        result = {
            "holdout_set_idx": holdout_idx,
            "n_holdout_targets": len(holdout_indices),
            "n_train_targets": len(train_indices),
            "holdout_loss": holdout_loss,
            "train_loss": train_loss,
            "generalization_gap": generalization_gap,
            "holdout_accuracy": holdout_accuracy,
            "train_accuracy": train_accuracy,
            "accuracy_gap": accuracy_gap,
            "calibration_time_seconds": calibration_time,
            "holdout_target_names": holdout_names,
            "target_details": target_details,
            "weights_sparsity": (
                np.mean(final_weights == 0)
                if calibration.sparse_weights is not None
                else 0
            ),
        }

        return result

    except Exception as e:
        logger.error(f"Error in holdout set {holdout_idx}: {str(e)}")
        return None


def _save_holdout_results(
    save_path: str,
    overall_metrics: Dict[str, float],
    target_robustness_df: pd.DataFrame,
    detailed_results: List[Dict[str, Any]],
) -> None:
    """Save detailed holdout results to CSV files.

    Args:
        save_path: Path to save results
        overall_metrics: Overall metrics dictionary
        target_robustness_df: Target robustness dataframe
        detailed_results: List of detailed results
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame([overall_metrics])
    overall_path = save_path.with_name(f"{save_path.stem}_overall.csv")
    overall_df.to_csv(overall_path, index=False)

    robustness_path = save_path.with_name(
        f"{save_path.stem}_target_robustness.csv"
    )
    target_robustness_df.to_csv(robustness_path, index=False)

    detailed_data = []
    for result in detailed_results:
        for target_detail in result["target_details"]:
            detailed_data.append(
                {
                    "holdout_set_idx": result["holdout_set_idx"],
                    "target_name": target_detail["target_name"],
                    "target_value": target_detail["target_value"],
                    "estimate": target_detail["estimate"],
                    "relative_error": target_detail["relative_error"],
                    "within_10pct": target_detail["within_10pct"],
                    "holdout_loss": result["holdout_loss"],
                    "train_loss": result["train_loss"],
                    "generalization_gap": result["generalization_gap"],
                }
            )

    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = save_path.with_name(f"{save_path.stem}_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)


def _generate_robustness_recommendation(
    overall_metrics: Dict[str, float],
    target_robustness_df: pd.DataFrame,
    regularize_with_l0: bool,
) -> str:
    """Generate interpretation and recommendations based on robustness evaluation.

    Args:
        overall_metrics: Overall metrics dictionary
        target_robustness_df: Target robustness dataframe
        regularize_with_l0: Whether L0 regularization is enabled

    Returns:
        Recommendation string
    """
    mean_acc = overall_metrics["mean_holdout_accuracy"]
    std_acc = overall_metrics["std_holdout_accuracy"]
    worst_acc = overall_metrics["worst_holdout_accuracy"]
    gen_gap = overall_metrics["mean_generalization_gap"]
    problematic_targets = target_robustness_df[
        target_robustness_df["holdout_accuracy_rate"] < 0.5
    ]["target_name"].tolist()

    rec_parts = []

    # Overall assessment
    if mean_acc >= 0.9 and std_acc <= 0.05:
        rec_parts.append(
            "✅ EXCELLENT ROBUSTNESS: The calibration generalizes very well."
        )
    elif mean_acc >= 0.8 and std_acc <= 0.1:
        rec_parts.append(
            "👍 GOOD ROBUSTNESS: The calibration shows good generalization."
        )
    elif mean_acc >= 0.7:
        rec_parts.append(
            "⚠️ MODERATE ROBUSTNESS: The calibration has decent but improvable generalization."
        )
    else:
        rec_parts.append(
            "❌ POOR ROBUSTNESS: The calibration shows weak generalization."
        )

    rec_parts.append(
        f"\nOn average, {mean_acc:.1%} of held-out targets are within 10% of their true values."
    )

    # Stability assessment
    if std_acc > 0.15:
        rec_parts.append(
            f"\n ⚠️ High variability (std={std_acc:.1%}) suggests instability across different target combinations."
        )

    # Worst-case analysis
    if worst_acc < 0.5:
        rec_parts.append(
            f"\n ⚠️ Worst-case scenario: Only {worst_acc:.1%} accuracy in some holdout sets."
        )

    # Problematic targets
    if problematic_targets:
        rec_parts.append(
            f"\n\n📊 Targets with poor holdout performance (<50% accuracy):"
        )
        for target in problematic_targets[:5]:
            target_data = target_robustness_df[
                target_robustness_df["target_name"] == target
            ].iloc[0]
            rec_parts.append(
                f"\n  - {target}: {target_data['holdout_accuracy_rate']:.1%} accuracy"
            )

    rec_parts.append("\n\n💡 RECOMMENDATIONS:")

    if mean_acc < 0.8 or std_acc > 0.1:
        if regularize_with_l0:
            rec_parts.append(
                "\n  1. Consider tuning L0 regularization parameters with tune_hyperparameters()"
            )
        else:
            rec_parts.append(
                "\n  1. Consider enabling L0 regularization for better generalization"
            )

        rec_parts.append(
            "\n  2. Increase the noise_level parameter to improve robustness"
        )
        rec_parts.append(
            "\n  3. Try increasing dropout_rate to reduce overfitting"
        )

    if problematic_targets:
        rec_parts.append(
            f"\n  4. Investigate why these targets are hard to predict: {', '.join(problematic_targets[:3])}"
        )
        rec_parts.append(
            "\n  5. Consider if these targets have sufficient support in the microdata"
        )

    if gen_gap > 0.01:
        rec_parts.append(
            f"\n  6. Generalization gap of {gen_gap:.4f} suggests some overfitting - consider regularization"
        )

    return "".join(rec_parts)


def evaluate_holdout_robustness(
    calibration,
    n_holdout_sets: Optional[int] = 5,
    holdout_fraction: Optional[float] = 0.2,
    save_results_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate calibration robustness using holdout validation.

    This function assesses how well the calibration generalizes by:
    1. Repeatedly holding out random subsets of targets
    2. Calibrating on the remaining targets
    3. Evaluating performance on held-out targets

    Args:
        calibration: Calibration instance to evaluate
        n_holdout_sets: Number of different holdout sets to evaluate.
            More sets provide better estimates but increase computation time.
        holdout_fraction: Fraction of targets to hold out in each set.
        save_results_to: Path to save detailed results as CSV. If None, no saving.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - overall_metrics: Summary statistics across all holdouts
            - target_robustness: DataFrame showing each target's performance when held out
            - recommendation: String with interpretation and recommendations
            - detailed_results: (if requested) List of detailed results per holdout
    """
    logger.info(
        f"Starting holdout robustness evaluation with {n_holdout_sets} sets, "
        f"holding out {holdout_fraction:.1%} of targets each time."
    )

    logger.warning(
        "Data leakage warning: Targets often share overlapping information "
        "(e.g., geographic breakdowns like 'snap in CA' and 'snap in US'). "
        "Holdout validation may not provide complete isolation between training and validation sets. "
        "The robustness metrics should be interpreted with this limitation in mind - "
        "they may overestimate the model's true generalization performance."
    )

    # Store original state
    original_state = {
        "weights": calibration.weights.copy(),
        "excluded_targets": (
            calibration.excluded_targets.copy()
            if calibration.excluded_targets
            else None
        ),
        "targets": calibration.targets.copy(),
        "target_names": (
            calibration.target_names.copy()
            if calibration.target_names is not None
            else None
        ),
        "sparse_weights": (
            calibration.sparse_weights.copy()
            if calibration.sparse_weights is not None
            else None
        ),
    }

    # Create holdout sets
    holdout_sets = calibration._create_holdout_sets(
        n_holdout_sets, holdout_fraction, calibration.seed + 1
    )

    # Collect results
    all_results = []
    target_performance = {
        name: {"held_out_losses": [], "held_out_accuracies": []}
        for name in calibration.original_target_names
    }

    try:
        for i in range(n_holdout_sets):
            holdout_set = holdout_sets[i]

            # Reset to original state
            calibration.weights = original_state["weights"].copy()
            calibration.excluded_targets = holdout_set["names"]
            calibration.exclude_targets()

            result = _evaluate_single_holdout_robustness(
                calibration, holdout_set, i, n_holdout_sets
            )

            if result is not None:
                all_results.append(result)

                # Update target performance tracking
                for detail in result["target_details"]:
                    name = detail["target_name"]
                    target_performance[name]["held_out_losses"].append(
                        (detail["estimate"] - detail["target_value"]) ** 2
                    )
                    target_performance[name]["held_out_accuracies"].append(
                        detail["within_10pct"]
                    )
    finally:
        # Restore original state
        for key, value in original_state.items():
            if value is not None:
                setattr(
                    calibration,
                    key,
                    value.copy() if hasattr(value, "copy") else value,
                )
        if calibration.excluded_targets:
            calibration.exclude_targets()

    if not all_results:
        raise ValueError("No successful holdout evaluations completed")

    # Calculate overall metrics
    holdout_losses = [r["holdout_loss"] for r in all_results]
    holdout_accuracies = [r["holdout_accuracy"] for r in all_results]
    train_losses = [r["train_loss"] for r in all_results]
    train_accuracies = [r["train_accuracy"] for r in all_results]
    generalization_gaps = [r["generalization_gap"] for r in all_results]

    overall_metrics = {
        "mean_holdout_loss": np.mean(holdout_losses),
        "std_holdout_loss": np.std(holdout_losses),
        "mean_holdout_accuracy": np.mean(holdout_accuracies),
        "std_holdout_accuracy": np.std(holdout_accuracies),
        "worst_holdout_accuracy": np.min(holdout_accuracies),
        "best_holdout_accuracy": np.max(holdout_accuracies),
        "mean_train_loss": np.mean(train_losses),
        "mean_train_accuracy": np.mean(train_accuracies),
        "mean_generalization_gap": np.mean(generalization_gaps),
        "std_generalization_gap": np.std(generalization_gaps),
        "n_successful_evaluations": len(all_results),
        "n_failed_evaluations": n_holdout_sets - len(all_results),
    }

    target_robustness_data = []
    for target_name in calibration.original_target_names:
        perf = target_performance[target_name]
        if perf[
            "held_out_losses"
        ]:  # Only include if target was held out at least once
            target_robustness_data.append(
                {
                    "target_name": target_name,
                    "times_held_out": len(perf["held_out_losses"]),
                    "mean_holdout_loss": np.mean(perf["held_out_losses"]),
                    "std_holdout_loss": np.std(perf["held_out_losses"]),
                    "holdout_accuracy_rate": np.mean(
                        perf["held_out_accuracies"]
                    ),
                }
            )

    target_robustness_df = pd.DataFrame(target_robustness_data)
    target_robustness_df = target_robustness_df.sort_values(
        "holdout_accuracy_rate", ascending=True
    )

    # Generate recommendations
    recommendation = _generate_robustness_recommendation(
        overall_metrics, target_robustness_df, calibration.regularize_with_l0
    )

    # Save results if requested
    if save_results_to:
        _save_holdout_results(
            save_results_to, overall_metrics, target_robustness_df, all_results
        )

    results = {
        "overall_metrics": overall_metrics,
        "target_robustness": target_robustness_df,
        "recommendation": recommendation,
        "detailed_results": all_results,
    }

    logger.info(
        f"\nHoldout evaluation completed:"
        f"\n  Mean holdout accuracy: {overall_metrics['mean_holdout_accuracy']:.2%} "
        f"(±{overall_metrics['std_holdout_accuracy']:.2%})"
        f"\n  Worst-case accuracy: {overall_metrics['worst_holdout_accuracy']:.2%}"
        f"\n  Generalization gap: {overall_metrics['mean_generalization_gap']:.6f}"
        f"\n  Least robust targets: {', '.join(target_robustness_df.head(5)['target_name'].tolist())}"
    )

    return results
