"""Hyperparameter tuning functionality for calibration."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch

from microcalibrate.utils.metrics import loss, pct_close

logger = logging.getLogger(__name__)


def _evaluate_single_holdout(
    calibration,
    holdout_set: Dict[str, Any],
    hyperparameters: Dict[str, float],
    epochs_per_trial: int,
    objectives_balance: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate hyperparameters on a single holdout set.

    Args:
        calibration: Calibration instance
        holdout_set: Dictionary with 'names' and 'indices' of holdout targets
        hyperparameters: Dictionary with l0_lambda, init_mean, temperature
        epochs_per_trial: Number of epochs to run
        objectives_balance: Weights for different objectives

    Returns:
        Dictionary with evaluation metrics and holdout target names
    """
    # Store original parameters
    original_params = {
        "l0_lambda": calibration.l0_lambda,
        "init_mean": calibration.init_mean,
        "temperature": calibration.temperature,
        "regularize_with_l0": calibration.regularize_with_l0,
        "epochs": calibration.epochs,
    }

    try:
        # Update parameters for this evaluation
        calibration.l0_lambda = hyperparameters["l0_lambda"]
        calibration.init_mean = hyperparameters["init_mean"]
        calibration.temperature = hyperparameters["temperature"]
        calibration.regularize_with_l0 = True
        calibration.epochs = epochs_per_trial

        # Set up calibration with this holdout set
        calibration.excluded_targets = holdout_set["names"]
        calibration.exclude_targets()

        # Run calibration
        calibration.calibrate()
        sparse_weights = calibration.sparse_weights

        # Get estimates for all targets
        weights_tensor = torch.tensor(
            sparse_weights, dtype=torch.float32, device=calibration.device
        )

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

        # Split into train/validation
        n_targets = len(calibration.original_target_names)
        val_indices = holdout_set["indices"]
        train_indices = [i for i in range(n_targets) if i not in val_indices]

        val_estimates = all_estimates[val_indices]
        val_targets = calibration.original_targets[val_indices]
        train_estimates = all_estimates[train_indices]
        train_targets = calibration.original_targets[train_indices]

        # Calculate metrics
        val_loss = loss(
            torch.tensor(
                val_estimates, dtype=torch.float32, device=calibration.device
            ),
            torch.tensor(
                val_targets, dtype=torch.float32, device=calibration.device
            ),
            None,
        ).item()

        val_accuracy = pct_close(
            torch.tensor(
                val_estimates, dtype=torch.float32, device=calibration.device
            ),
            torch.tensor(
                val_targets, dtype=torch.float32, device=calibration.device
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

        sparsity = np.mean(sparse_weights == 0)

        # Calculate objective
        objective = (
            val_loss * objectives_balance["loss"]
            + (1 - val_accuracy) * objectives_balance["accuracy"]
            + (1 - sparsity) * objectives_balance["sparsity"]
        )

        return {
            "objective": objective,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "sparsity": sparsity,
            "n_nonzero_weights": int(np.sum(sparse_weights != 0)),
            "holdout_targets": holdout_set["names"],
            "hyperparameters": hyperparameters.copy(),
        }

    finally:
        # Restore original parameters
        for key, value in original_params.items():
            setattr(calibration, key, value)


def _create_objective_function(
    calibration,
    holdout_sets: List[Dict[str, Any]],
    epochs_per_trial: int,
    objectives_balance: Dict[str, float],
    aggregation: str,
    all_evaluations: List,
    original_state: Dict,
):
    """Create the objective function for Optuna optimization.

    Args:
        calibration: Calibration instance
        holdout_sets: List of holdout sets
        epochs_per_trial: Number of epochs per trial
        objectives_balance: Weights for different objectives
        aggregation: How to aggregate results across holdouts
        all_evaluations: List to collect evaluation records
        original_state: Original calibration state to restore

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Suggest hyperparameters
            hyperparameters = {
                "l0_lambda": trial.suggest_float(
                    "l0_lambda", 1e-6, 1e-4, log=True
                ),
                "init_mean": trial.suggest_float("init_mean", 0.5, 0.999),
                "temperature": trial.suggest_float("temperature", 0.5, 2.0),
            }

            # Evaluate on all holdout sets
            holdout_results = []
            for holdout_idx, holdout_set in enumerate(holdout_sets):
                result = _evaluate_single_holdout(
                    calibration=calibration,
                    holdout_set=holdout_set,
                    hyperparameters=hyperparameters,
                    epochs_per_trial=epochs_per_trial,
                    objectives_balance=objectives_balance,
                )
                # Add trial and holdout identifiers for tracking
                evaluation_record = result.copy()
                evaluation_record["trial_number"] = trial.number
                evaluation_record["holdout_set_idx"] = holdout_idx
                all_evaluations.append(evaluation_record)
                holdout_results.append(result)

            # Aggregate objectives
            objectives = [r["objective"] for r in holdout_results]

            if aggregation == "mean":
                final_objective = np.mean(objectives)
            elif aggregation == "median":
                final_objective = np.median(objectives)
            elif aggregation == "worst":
                final_objective = np.max(objectives)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            # Store detailed metrics
            trial.set_user_attr(
                "holdout_objectives", [r["objective"] for r in holdout_results]
            )
            trial.set_user_attr(
                "mean_val_loss",
                np.mean([r["val_loss"] for r in holdout_results]),
            )
            trial.set_user_attr(
                "std_val_loss",
                np.std([r["val_loss"] for r in holdout_results]),
            )
            trial.set_user_attr(
                "mean_val_accuracy",
                np.mean([r["val_accuracy"] for r in holdout_results]),
            )
            trial.set_user_attr(
                "std_val_accuracy",
                np.std([r["val_accuracy"] for r in holdout_results]),
            )
            trial.set_user_attr(
                "mean_train_loss",
                np.mean([r["train_loss"] for r in holdout_results]),
            )
            trial.set_user_attr(
                "mean_train_accuracy",
                np.mean([r["train_accuracy"] for r in holdout_results]),
            )

            # Use the last holdout's sparsity metrics
            last_result = holdout_results[-1]
            trial.set_user_attr("sparsity", last_result["sparsity"])
            trial.set_user_attr(
                "n_nonzero_weights", last_result.get("n_nonzero_weights", 0)
            )

            # Log progress
            if trial.number % 5 == 0:
                objectives = [r["objective"] for r in holdout_results]
                val_accuracies = [r["val_accuracy"] for r in holdout_results]
                logger.info(
                    f"Trial {trial.number}:\n"
                    f"  Objectives by holdout: {[f'{obj:.4f}' for obj in objectives]}\n"
                    f"  {aggregation.capitalize()} objective: {final_objective:.4f}\n"
                    f"  Mean val accuracy: {np.mean(val_accuracies):.2%} (±{np.std(val_accuracies):.2%})\n"
                    f"  Sparsity: {last_result['sparsity']:.2%}"
                )

            return final_objective

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return 1e10

        finally:
            # Restore original state
            calibration.excluded_targets = original_state["excluded_targets"]
            calibration.targets = original_state["targets"]
            calibration.target_names = original_state["target_names"]
            calibration.exclude_targets()

    return objective


def tune_l0_hyperparameters(
    calibration,
    n_trials: Optional[int] = 30,
    objectives_balance: Optional[Dict[str, float]] = None,
    epochs_per_trial: Optional[int] = None,
    n_holdout_sets: Optional[int] = 3,
    holdout_fraction: Optional[float] = 0.2,
    aggregation: Optional[str] = "mean",
    timeout: Optional[float] = None,
    n_jobs: Optional[int] = 1,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: Optional[bool] = False,
    direction: Optional[str] = "minimize",
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    pruner: Optional["optuna.pruners.BasePruner"] = None,
) -> Dict[str, Any]:
    """
    Tune hyperparameters for L0 regularization using Optuna.

    This method optimizes l0_lambda, init_mean, and temperature to achieve:
    1. Low calibration loss
    2. High percentage of targets within 10% of their true values
    3. Sparse weights (fewer non-zero weights)

    Args:
        calibration: Calibration instance to tune
        n_trials: Number of optimization trials to run.
        objectives_balance: Dictionary to balance the importance of loss, accuracy, and sparsity
            in the objective function. Default prioritizes being within 10% of targets.
        epochs_per_trial: Number of epochs per trial. If None, uses calibration.epochs // 4.
        n_holdout_sets: Number of different holdout sets to create and evaluate on
        holdout_fraction: Fraction of targets in each holdout set
        aggregation: How to combine scores across holdouts ("mean", "median", "worst")
        timeout: Stop study after this many seconds. None means no timeout.
        n_jobs: Number of parallel jobs. -1 means using all processors.
        study_name: Name of the study for storage.
        storage: Database URL for distributed optimization.
        load_if_exists: Whether to load existing study.
        direction: Optimization direction ('minimize' or 'maximize').
        sampler: Optuna sampler for hyperparameter suggestions.
        pruner: Optuna pruner for early stopping of trials.

    Returns:
        Dictionary containing the best hyperparameters found.
    """
    # Suppress Optuna's logs during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if objectives_balance is None:
        objectives_balance = {"loss": 1.0, "accuracy": 100.0, "sparsity": 10.0}

    if epochs_per_trial is None:
        epochs_per_trial = max(calibration.epochs // 4, 100)

    holdout_sets = calibration._create_holdout_sets(
        n_holdout_sets, holdout_fraction, calibration.seed
    )

    logger.info(
        f"Multi-holdout hyperparameter tuning:\n"
        f"  - {n_holdout_sets} holdout sets\n"
        f"  - {len(holdout_sets[0]['indices'])} targets per holdout ({holdout_fraction:.1%})\n"
        f"  - Aggregation: {aggregation}\n"
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
        "excluded_targets": calibration.excluded_targets,
        "targets": calibration.targets.copy(),
        "target_names": (
            calibration.target_names.copy()
            if calibration.target_names is not None
            else None
        ),
    }

    # Initialize list to collect all holdout evaluations
    all_evaluations = []

    # Create objective function
    objective = _create_objective_function(
        calibration=calibration,
        holdout_sets=holdout_sets,
        epochs_per_trial=epochs_per_trial,
        objectives_balance=objectives_balance,
        aggregation=aggregation,
        all_evaluations=all_evaluations,
        original_state=original_state,
    )

    # Create or load study
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=calibration.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    best_params["mean_val_loss"] = best_trial.user_attrs.get("mean_val_loss")
    best_params["std_val_loss"] = best_trial.user_attrs.get("std_val_loss")
    best_params["mean_val_accuracy"] = best_trial.user_attrs.get(
        "mean_val_accuracy"
    )
    best_params["std_val_accuracy"] = best_trial.user_attrs.get(
        "std_val_accuracy"
    )
    best_params["holdout_objectives"] = best_trial.user_attrs.get(
        "holdout_objectives"
    )
    best_params["sparsity"] = best_trial.user_attrs.get("sparsity")
    best_params["n_holdout_sets"] = n_holdout_sets
    best_params["aggregation"] = aggregation

    # Create evaluation tracking dataframe
    evaluation_df = pd.DataFrame(all_evaluations)

    # Convert holdout_targets list to string for easier viewing
    if "holdout_targets" in evaluation_df.columns:
        evaluation_df["holdout_targets"] = evaluation_df[
            "holdout_targets"
        ].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    best_params["evaluation_history"] = evaluation_df

    logger.info(
        f"\nMulti-holdout tuning completed!"
        f"\nBest parameters:"
        f"\n  - l0_lambda: {best_params['l0_lambda']:.2e}"
        f"\n  - init_mean: {best_params['init_mean']:.4f}"
        f"\n  - temperature: {best_params['temperature']:.4f}"
        f"\nPerformance across {n_holdout_sets} holdouts:"
        f"\n  - Mean val loss: {best_params['mean_val_loss']:.6f} (±{best_params['std_val_loss']:.6f})"
        f"\n  - Mean val accuracy: {best_params['mean_val_accuracy']:.2%} (±{best_params['std_val_accuracy']:.2%})"
        f"\n  - Individual objectives: {[f'{obj:.4f}' for obj in best_params['holdout_objectives']]}"
        f"\n  - Sparsity: {best_params['sparsity']:.2%}"
        f"\n\nEvaluation history saved with {len(evaluation_df)} records across {n_trials} trials."
    )

    return best_params
