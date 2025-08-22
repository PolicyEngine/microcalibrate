import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from torch import Tensor


class Calibration:
    def __init__(
        self,
        weights: np.ndarray,
        targets: np.ndarray,
        target_names: Optional[np.ndarray] = None,
        estimate_matrix: Optional[pd.DataFrame] = None,
        estimate_function: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        epochs: Optional[int] = 32,
        noise_level: Optional[float] = 10.0,
        learning_rate: Optional[float] = 1e-3,
        dropout_rate: Optional[float] = 0,  # default to no dropout for now
        normalization_factor: Optional[torch.Tensor] = None,
        excluded_targets: Optional[List[str]] = None,
        csv_path: Optional[str] = None,
        device: str = "cpu",  # fix to cpu for now to avoid user device-specific issues
        l0_lambda: float = 5e-6,  # best between 1e-6 and 1e-5
        init_mean: float = 0.999,  # initial proportion with non-zero weights, set near 0
        sparse_learning_rate: float = 0.2,
        temperature: float = 0.5,  # usual values .5 to 3
        sparse_learning_rate: Optional[float] = 0.2,
        regularize_with_l0: Optional[bool] = False,
        seed: Optional[int] = 42,
    ):
        """Initialize the Calibration class.

        Args:
            weights (np.ndarray): Array of original weights.
            targets (np.ndarray): Array of target values.
            target_names (Optional[np.ndarray]): Optional names of the targets for logging. Defaults to None. You MUST pass these names if you are not passing in an estimate matrix, and just passing in an estimate function.
            estimate_matrix (pd.DataFrame): DataFrame containing the estimate matrix.
            estimate_function (Optional[Callable[[torch.Tensor], torch.Tensor]]): Function to estimate targets from weights. Defaults to None, in which case it will use the estimate_matrix.
            epochs (int): Optional number of epochs for calibration. Defaults to 32.
            noise_level (float): Optional level of noise to add to weights. Defaults to 10.0.
            learning_rate (float): Optional learning rate for the optimizer. Defaults to 1e-3.
            dropout_rate (float): Optional probability of dropping weights during training. Defaults to 0.1.
            normalization_factor (Optional[torch.Tensor]): Optional normalization factor for the loss (handles multi-level geographical calibration). Defaults to None.
            excluded_targets (Optional[List]): Optional List of targets to exclude from calibration. Defaults to None.
            csv_path (str): Optional path to save performance logs as CSV. Defaults to None.
            device (str): Optional device to run the calibration on. Defaults to None, which will use CUDA if available, otherwise MPS, otherwise CPU.
            l0_lambda (float): Regularization parameter for L0 regularization. Defaults to 5e-6.
            init_mean (float): Initial mean for L0 regularization, representing the initial proportion of non-zero weights. Defaults to 0.999.
            temperature (float): Temperature parameter for L0 regularization, controlling the sparsity of the model. Defaults to 0.5.
            sparse_learning_rate (float): Learning rate for the regularizing optimizer. Defaults to 0.2.
            regularize_with_l0 (Optional[bool]): Whether to apply L0 regularization. Defaults to False.
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )

        self.logger = logging.getLogger(__name__)

        self.original_estimate_matrix = estimate_matrix
        self.original_targets = targets
        self.original_target_names = target_names
        self.original_estimate_function = estimate_function
        self.weights = weights
        self.excluded_targets = excluded_targets
        self.epochs = epochs
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.normalization_factor = normalization_factor
        self.csv_path = csv_path
        self.performance_df = None
        self.sparse_weights = None
        self.l0_lambda = l0_lambda
        self.init_mean = init_mean
        self.temperature = temperature
        self.sparse_learning_rate = sparse_learning_rate
        self.regularize_with_l0 = regularize_with_l0
        self.seed = seed

        if device is not None:
            self.device = torch.device(device)
            torch.manual_seed(self.seed)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )
            if self.device == "cuda":
                torch.cuda.manual_seed(self.seed)

        self.estimate_matrix = None
        self.targets = None
        self.target_names = None
        self.estimate_function = None
        self.excluded_target_data = {}

        # Set target names from estimate_matrix if not provided
        if target_names is None and self.original_estimate_matrix is not None:
            self.original_target_names = (
                self.original_estimate_matrix.columns.to_numpy()
            )

        if self.excluded_targets is not None:
            self.exclude_targets()
        else:
            self.targets = self.original_targets
            self.target_names = self.original_target_names
            if self.original_estimate_matrix is not None:
                self.estimate_matrix = torch.tensor(
                    self.original_estimate_matrix.values,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                self.estimate_matrix = None

        if self.original_estimate_function is None:
            if self.estimate_matrix is not None:
                self.estimate_function = (
                    lambda weights: weights @ self.estimate_matrix
                )
            else:
                raise ValueError(
                    "Either estimate_function or estimate_matrix must be provided"
                )
        elif self.excluded_targets:
            self.logger.warning(
                "You are passing an estimate function with excluded targets. "
                "Make sure the function handles excluded targets correctly, as reweight() will handle the filtering."
            )

    def calibrate(self) -> None:
        """Calibrate the weights based on the estimate function and targets."""

        self._assess_targets(
            estimate_function=self.estimate_function,
            estimate_matrix=self.estimate_matrix,
            weights=self.weights,
            targets=self.targets,
            target_names=self.target_names,
        )

        from .reweight import reweight

        new_weights, sparse_weights, self.performance_df = reweight(
            original_weights=self.weights,
            estimate_function=self.estimate_function,
            targets_array=self.targets,
            target_names=self.target_names,
            epochs=self.epochs,
            noise_level=self.noise_level,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            normalization_factor=self.normalization_factor,
            excluded_targets=self.excluded_targets,
            excluded_target_data=self.excluded_target_data,
            csv_path=self.csv_path,
            device=self.device,
            l0_lambda=self.l0_lambda,
            init_mean=self.init_mean,
            temperature=self.temperature,
            sparse_learning_rate=self.sparse_learning_rate,
            regularize_with_l0=self.regularize_with_l0,
            logger=self.logger,
        )

        self.weights = new_weights
        self.sparse_weights = sparse_weights

        return self.performance_df

    def exclude_targets(
        self, excluded_targets: Optional[List[str]] = None
    ) -> None:
        """Exclude specified targets from calibration.

        Args:
            excluded_targets (Optional[List[str]]): List of target names to exclude from calibration. If None, the original excluded_targets passed to the calibration constructor will be excluded.
        """
        if excluded_targets is not None:
            self.excluded_targets = excluded_targets
        excluded_indices = []
        self.excluded_target_data = {}
        if self.excluded_targets and self.original_target_names is not None:
            # Find indices of excluded targets
            for i, name in enumerate(self.original_target_names):
                if name in self.excluded_targets:
                    excluded_indices.append(i)
                    self.excluded_target_data[name] = {
                        "target": self.original_targets[i],
                        "index": i,
                    }

            # Remove excluded targets from calibration
            calibration_mask = ~np.isin(
                np.arange(len(self.original_target_names)), excluded_indices
            )
            targets_array = self.original_targets[calibration_mask]
            target_names = (
                self.original_target_names[calibration_mask]
                if self.original_target_names is not None
                else None
            )

            self.logger.info(
                f"Excluded {len(excluded_indices)} targets from calibration: {self.excluded_targets}"
            )
            self.logger.info(f"Calibrating {len(targets_array)} targets")
        else:
            targets_array = self.original_targets
            target_names = self.original_target_names

        # Get initial estimates for excluded targets if needed
        if self.excluded_targets:
            initial_weights_tensor = torch.tensor(
                self.weights, dtype=torch.float32, device=self.device
            )
            if self.original_estimate_function is not None:
                initial_estimates_all = (
                    self.original_estimate_function(initial_weights_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                )
            elif self.original_estimate_matrix is not None:
                # Get initial estimates using the original full matrix
                original_estimate_matrix_tensor = torch.tensor(
                    self.original_estimate_matrix.values,
                    dtype=torch.float32,
                    device=self.device,
                )
                initial_estimates_all = (
                    (initial_weights_tensor @ original_estimate_matrix_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Filter estimate matrix for calibration
                filtered_estimate_matrix = self.original_estimate_matrix.iloc[
                    :, calibration_mask
                ]
                self.estimate_matrix = torch.tensor(
                    filtered_estimate_matrix.values,
                    dtype=torch.float32,
                    device=self.device,
                )

                self.estimate_function = (
                    lambda weights: weights @ self.estimate_matrix
                )
            else:
                raise ValueError(
                    "Either estimate_function or estimate_matrix must be provided"
                )

            # Store initial estimates for excluded targets
            for name in self.excluded_targets:
                if name in self.excluded_target_data:
                    self.excluded_target_data[name]["initial_estimate"] = (
                        initial_estimates_all[
                            self.excluded_target_data[name]["index"]
                        ]
                    )

        else:
            if self.original_estimate_matrix is not None:
                self.estimate_matrix = torch.tensor(
                    self.original_estimate_matrix.values,
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.original_estimate_function is None:
                    self.estimate_function = (
                        lambda weights: weights @ self.estimate_matrix
                    )
            else:
                self.estimate_matrix = None

        # Set up final attributes
        self.targets = targets_array
        self.target_names = target_names

    def estimate(self, weights: Optional[np.ndarray] = None) -> pd.Series:
        if weights is None:
            weights = self.weights
        return pd.Series(
            index=self.target_names,
            data=self.estimate_function(
                torch.tensor(weights, dtype=torch.float32, device=self.device)
            )
            .cpu()
            .detach()
            .numpy(),
        )

    def _assess_targets(
        self,
        estimate_function: Callable[[torch.Tensor], torch.Tensor],
        estimate_matrix: Optional[pd.DataFrame],
        weights: np.ndarray,
        targets: np.ndarray,
        target_names: Optional[np.ndarray] = None,
    ) -> None:
        """Assess the targets to ensure they do not violate basic requirements like compatibility, correct order of magnitude, etc.

        Args:
            estimate_function (Callable[[torch.Tensor], torch.Tensor]): Function to estimate the targets from weights.
            estimate_matrix (Optional[pd.DataFrame]): DataFrame containing the estimate matrix. Defaults to None.
            weights (np.ndarray): Array of original weights.
            targets (np.ndarray): Array of target values.
            target_names (np.ndarray): Optional names of the targets for logging. Defaults to None.

        Raises:
            ValueError: If the targets do not match the expected format or values.
            ValueError: If the targets are not compatible with each other.
        """
        self.logger.info("Performing basic target assessment...")

        if targets.ndim != 1:
            raise ValueError("Targets must be a 1D NumPy array.")

        if np.any(np.isnan(targets)):
            raise ValueError("Targets contain NaN values.")

        if np.any(targets < 0):
            self.logger.warning(
                "Some targets are negative. This may not make sense for totals."
            )

        if estimate_matrix is None and self.excluded_targets is not None:
            logger.warning(
                "You are excluding targets but not passing an estimate matrix. Make sure the estimate function handles excluded targets correctly, otherwise you may face operand errors."
            )

        # Estimate order of magnitude from column sums and warn if they are off by an order of magnitude from targets
        one_weights = weights * 0 + 1
        estimates = (
            estimate_function(
                torch.tensor(
                    one_weights, dtype=torch.float32, device=self.device
                )
            )
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )

        # Use a small epsilon to avoid division by zero
        eps = 1e-4
        adjusted_estimates = np.where(estimates == 0, eps, estimates)
        ratios = targets / adjusted_estimates

        for i, (target_val, estimate_val, ratio) in enumerate(
            zip(targets, estimates, ratios)
        ):
            target_name = (
                target_names[i] if target_names is not None else f"target_{i}"
            )

            if estimate_val == 0:
                self.logger.warning(
                    f"Column {target_name} has a zero estimate sum; using ε={eps} for comparison."
                )

            order_diff = np.log10(abs(ratio)) if ratio != 0 else np.inf
            if order_diff > 1:
                self.logger.warning(
                    f"Target {target_name} ({target_val:.2e}) differs from initial estimate ({estimate_val:.2e}) "
                    f"by {order_diff:.2f} orders of magnitude."
                )
            if estimate_matrix is not None:
                # Check if estimate_matrix is a tensor or DataFrame
                if hasattr(estimate_matrix, "iloc"):
                    contributing_mask = estimate_matrix.iloc[:, i] != 0
                    contribution_ratio = (
                        contributing_mask.sum() / estimate_matrix.shape[0]
                    )
                else:
                    contributing_mask = estimate_matrix[:, i] != 0
                    contribution_ratio = (
                        contributing_mask.sum().item()
                        / estimate_matrix.shape[0]
                    )
                if contribution_ratio < 0.01:
                    self.logger.warning(
                        f"Target {target_name} is supported by only {contribution_ratio:.2%} "
                        f"of records in the loss matrix. This may make calibration unstable or ineffective."
                    )

    def assess_analytical_solution(
        self, use_sparse: Optional[bool] = False
    ) -> None:
        """Assess analytically which targets complicate achieving calibration accuracy as an optimization problem.

        Uses the Moore-Penrose inverse for least squares solution to relax the assumption that weights need be positive and measure by how much loss increases when trying to solve for a set of equations (the more targets, the larger the number of equations, the harder the optimization problem).

        Args:
            use_sparse (bool): Whether to use sparse matrix methods for the analytical solution. Defaults to False.
        """
        if self.estimate_matrix is None:
            raise ValueError(
                "Estimate matrix is not provided. Cannot assess analytical solution from the estimate function alone."
            )

        def _get_linear_loss(metrics_matrix, target_vector, sparse=False):
            """Gets the mean squared error loss of X.T @ w wrt y for least squares solution"""
            X = metrics_matrix
            y = target_vector
            normalization_factor = (
                self.normalization_factor
                if self.normalization_factor is not None
                else 1
            )
            if not sparse:
                X_inv_mp = np.linalg.pinv(X)  # Moore-Penrose inverse
                w_mp = X_inv_mp.T @ y
                y_hat = X.T @ w_mp

            else:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import lsqr

                X_sparse = csr_matrix(X)
                result = lsqr(
                    X_sparse.T, y
                )  # iterative method for sparse matrices
                w_sparse = result[0]
                y_hat = X_sparse.T @ w_sparse

            return np.mean(((y - y_hat) ** 2) * normalization_factor)

        X = self.original_estimate_matrix.values
        y = self.targets

        results = []
        slices = []
        idx_dict = {
            self.original_estimate_matrix.columns.to_list()[i]: i
            for i in range(len(self.original_estimate_matrix.columns))
        }

        self.logger.info(
            "Assessing analytical solution to the optimization problem for each target... \n"
            "This evaluates how much each target complicates achieving calibration accuracy. The loss reported is the mean squared error of the least squares solution."
        )

        for target_name, index_list in idx_dict.items():
            slices.append(index_list)
            loss = _get_linear_loss(X[:, slices], y[slices], use_sparse)
            delta = loss - results[-1]["loss"] if results else None

            results.append(
                {
                    "target_added": target_name,
                    "loss": loss,
                    "delta_loss": delta,
                }
            )

        return pd.DataFrame(results)

    def summary(
        self,
    ) -> pd.DataFrame:
        """Generate a summary of the calibration process."""
        if self.performance_df is None:
            return "No calibration has been performed yet, make sure to run .calibrate() before requesting a summary."

        last_epoch = self.performance_df["epoch"].max()
        final_rows = self.performance_df[
            self.performance_df["epoch"] == last_epoch
        ]

        df = final_rows[["target_name", "target", "estimate"]].copy()
        df.rename(
            columns={
                "target_name": "Metric",
                "target": "Official target",
                "estimate": "Final estimate",
            },
            inplace=True,
        )
        df["Relative error"] = (
            df["Final estimate"] - df["Official target"]
        ) / df["Official target"]
        df = df.reset_index(drop=True)
        return df

    def tune_hyperparameters(
        self,
        n_trials: Optional[int] = 30,
        objectives_balance: Optional[Dict[str, float]] = {
            "loss": 1.0,
            "accuracy": 100.0,
            "sparsity": 10.0,
        },
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
            n_trials: Number of optimization trials to run.
            objectives_balance: Dictionary to balance the importance of loss, accuracy, and sparsity in the objective function. Default prioritizes being within 10% of targets.
            epochs_per_trial: Number of epochs per trial. If None, uses self.epochs // 4.
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

        if epochs_per_trial is None:
            epochs_per_trial = max(self.epochs // 4, 100)

        holdout_sets = self._create_holdout_sets(
            n_holdout_sets, holdout_fraction, self.seed
        )

        logger.info(
            f"Multi-holdout hyperparameter tuning:\n"
            f"  - {n_holdout_sets} holdout sets\n"
            f"  - {len(holdout_sets[0]['indices'])} targets per holdout ({holdout_fraction:.1%})\n"
            f"  - Aggregation: {aggregation}\n"
        )

        # Store original state
        original_state = {
            "excluded_targets": self.excluded_targets,
            "targets": self.targets.copy(),
            "target_names": (
                self.target_names.copy()
                if self.target_names is not None
                else None
            ),
        }

        # Initialize list to collect all holdout evaluations
        all_evaluations = []

        def evaluate_single_holdout(
            holdout_set: Dict[str, Any],
            hyperparameters: Dict[str, float],
            epochs_per_trial: int,
            objectives_balance: Dict[str, float],
        ) -> Dict[str, Any]:
            """Evaluate hyperparameters on a single holdout set.

            Args:
                holdout_set: Dictionary with 'names' and 'indices' of holdout targets
                hyperparameters: Dictionary with l0_lambda, init_mean, temperature
                epochs_per_trial: Number of epochs to run
                objectives_balance: Weights for different objectives

            Returns:
                Dictionary with evaluation metrics and holdout target names
            """
            # Store original parameters
            original_params = {
                "l0_lambda": self.l0_lambda,
                "init_mean": self.init_mean,
                "temperature": self.temperature,
                "regularize_with_l0": self.regularize_with_l0,
                "epochs": self.epochs,
            }

            try:
                # Update parameters for this evaluation
                self.l0_lambda = hyperparameters["l0_lambda"]
                self.init_mean = hyperparameters["init_mean"]
                self.temperature = hyperparameters["temperature"]
                self.regularize_with_l0 = True
                self.epochs = epochs_per_trial

                # Set up calibration with this holdout set
                self.excluded_targets = holdout_set["names"]
                self.exclude_targets()

                # Run calibration
                self.calibrate()
                sparse_weights = self.sparse_weights

                # Get estimates for all targets
                weights_tensor = torch.tensor(
                    sparse_weights, dtype=torch.float32, device=self.device
                )

                if self.original_estimate_matrix is not None:
                    original_matrix_tensor = torch.tensor(
                        self.original_estimate_matrix.values,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    all_estimates = (
                        (weights_tensor @ original_matrix_tensor).cpu().numpy()
                    )
                else:
                    all_estimates = (
                        self.original_estimate_function(weights_tensor)
                        .cpu()
                        .numpy()
                    )

                # Split into train/validation
                n_targets = len(self.original_target_names)
                val_indices = holdout_set["indices"]
                train_indices = [
                    i for i in range(n_targets) if i not in val_indices
                ]

                val_estimates = all_estimates[val_indices]
                val_targets = self.original_targets[val_indices]
                train_estimates = all_estimates[train_indices]
                train_targets = self.original_targets[train_indices]

                # Calculate metrics
                from .utils.metrics import loss, pct_close

                val_loss = loss(
                    torch.tensor(
                        val_estimates, dtype=torch.float32, device=self.device
                    ),
                    torch.tensor(
                        val_targets, dtype=torch.float32, device=self.device
                    ),
                    None,
                ).item()

                val_accuracy = pct_close(
                    torch.tensor(
                        val_estimates, dtype=torch.float32, device=self.device
                    ),
                    torch.tensor(
                        val_targets, dtype=torch.float32, device=self.device
                    ),
                )

                train_loss = loss(
                    torch.tensor(
                        train_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        train_targets, dtype=torch.float32, device=self.device
                    ),
                    None,
                ).item()

                train_accuracy = pct_close(
                    torch.tensor(
                        train_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        train_targets, dtype=torch.float32, device=self.device
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
                    setattr(self, key, value)

        def objective(
            trial: optuna.Trial,
            objectives_balance: Dict[str, float] = objectives_balance,
        ) -> float:
            """Objective function for Optuna optimization."""
            try:
                # Suggest hyperparameters
                hyperparameters = {
                    "l0_lambda": trial.suggest_float(
                        "l0_lambda", 1e-6, 1e-4, log=True
                    ),
                    "init_mean": trial.suggest_float("init_mean", 0.5, 0.999),
                    "temperature": trial.suggest_float(
                        "temperature", 0.5, 2.0
                    ),
                }

                # Evaluate on all holdout sets
                holdout_results = []
                for holdout_idx, holdout_set in enumerate(holdout_sets):
                    result = evaluate_single_holdout(
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
                    raise ValueError(
                        f"Unknown aggregation method: {aggregation}"
                    )

                # Store detailed metrics
                trial.set_user_attr(
                    "holdout_objectives",
                    [r["objective"] for r in holdout_results],
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
                    "n_nonzero_weights",
                    last_result.get("n_nonzero_weights", 0),
                )

                # Log progress
                if trial.number % 5 == 0:
                    objectives = [r["objective"] for r in holdout_results]
                    val_accuracies = [
                        r["val_accuracy"] for r in holdout_results
                    ]
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
                self.excluded_targets = original_state["excluded_targets"]
                self.targets = original_state["targets"]
                self.target_names = original_state["target_names"]
                self.exclude_targets()

        # Create or load study
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.seed)

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
        best_params["mean_val_loss"] = best_trial.user_attrs.get(
            "mean_val_loss"
        )
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

    def _create_holdout_sets(
        self,
        n_holdout_sets: int,
        holdout_fraction: float,
        random_state: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Create multiple holdout sets for cross-validation.

        Args:
            n_holdout_sets: Number of holdout sets to create
            holdout_fraction: Fraction of targets in each holdout set
            random_state: Base random seed for reproducibility
            exclude_excluded: Whether to exclude already excluded targets from the holdout sets

        Returns:
            List of dictionaries containing holdout names and indices
        """
        n_targets = len(self.target_names)
        n_holdout_targets = max(1, int(n_targets * holdout_fraction))

        holdout_sets = []
        for i in range(n_holdout_sets):
            # Each holdout set gets a different random selection
            set_rng = np.random.default_rng((random_state or self.seed) + i)
            holdout_indices = set_rng.choice(
                n_targets, size=n_holdout_targets, replace=False
            )
            holdout_names = [self.target_names[idx] for idx in holdout_indices]
            holdout_sets.append(
                {"names": holdout_names, "indices": holdout_indices}
            )

        return holdout_sets

    def evaluate_holdout_robustness(
        self,
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
            n_holdout_sets (int): Number of different holdout sets to evaluate.
            More sets provide better estimates but increase computation time.
            holdout_fraction (float): Fraction of targets to hold out in each set.
            save_results_to (str): Path to save detailed results as CSV. If None, no saving.

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

        # Store original state
        original_state = {
            "weights": self.weights.copy(),
            "excluded_targets": (
                self.excluded_targets.copy() if self.excluded_targets else None
            ),
            "targets": self.targets.copy(),
            "target_names": (
                self.target_names.copy()
                if self.target_names is not None
                else None
            ),
            "sparse_weights": (
                self.sparse_weights.copy()
                if self.sparse_weights is not None
                else None
            ),
        }

        # Create holdout sets
        holdout_sets = self._create_holdout_sets(
            n_holdout_sets, holdout_fraction, self.seed + 1
        )

        # Collect results
        all_results = []
        target_performance = {
            name: {"held_out_losses": [], "held_out_accuracies": []}
            for name in self.original_target_names
        }

        def evaluate_single_holdout_robustness(
            holdout_idx: int,
        ) -> Dict[str, Any]:
            """Evaluate a single holdout set."""
            try:
                holdout_set = holdout_sets[holdout_idx]
                logger.info(
                    f"Evaluating holdout set {holdout_idx + 1}/{n_holdout_sets}"
                )

                # Reset to original state
                self.weights = original_state["weights"].copy()
                self.excluded_targets = holdout_set["names"]
                self.exclude_targets()

                # Run calibration on training targets
                start_time = pd.Timestamp.now()
                self.calibrate()
                calibration_time = (
                    pd.Timestamp.now() - start_time
                ).total_seconds()

                # Get final weights (sparse if using L0, otherwise regular)
                final_weights = (
                    self.sparse_weights
                    if self.sparse_weights is not None
                    else self.weights
                )

                # Evaluate on all targets
                weights_tensor = torch.tensor(
                    final_weights, dtype=torch.float32, device=self.device
                )

                # Get estimates for all targets using original estimate function/matrix
                if self.original_estimate_matrix is not None:
                    original_matrix_tensor = torch.tensor(
                        self.original_estimate_matrix.values,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    all_estimates = (
                        (weights_tensor @ original_matrix_tensor).cpu().numpy()
                    )
                else:
                    all_estimates = (
                        self.original_estimate_function(weights_tensor)
                        .cpu()
                        .numpy()
                    )

                # Calculate metrics for holdout vs training sets
                holdout_indices = holdout_set["indices"]
                train_indices = [
                    i
                    for i in range(len(self.original_target_names))
                    if i not in holdout_indices
                ]

                holdout_estimates = all_estimates[holdout_indices]
                holdout_targets = self.original_targets[holdout_indices]
                holdout_names = holdout_set["names"]

                train_estimates = all_estimates[train_indices]
                train_targets = self.original_targets[train_indices]

                # Calculate losses and accuracies
                from .utils.metrics import loss, pct_close

                holdout_loss = loss(
                    torch.tensor(
                        holdout_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        holdout_targets,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    None,
                ).item()

                holdout_accuracy = pct_close(
                    torch.tensor(
                        holdout_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        holdout_targets,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )

                train_loss = loss(
                    torch.tensor(
                        train_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        train_targets, dtype=torch.float32, device=self.device
                    ),
                    None,
                ).item()

                train_accuracy = pct_close(
                    torch.tensor(
                        train_estimates,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.tensor(
                        train_targets, dtype=torch.float32, device=self.device
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

                    target_performance[name]["held_out_losses"].append(
                        (holdout_estimates[idx] - holdout_targets[idx]) ** 2
                    )
                    target_performance[name]["held_out_accuracies"].append(
                        abs(rel_error) <= 0.1
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
                        if self.sparse_weights is not None
                        else 0
                    ),
                }

                return result

            except Exception as e:
                logger.error(f"Error in holdout set {holdout_idx}: {str(e)}")
                return None
            finally:
                # Restore original state
                for key, value in original_state.items():
                    if value is not None:
                        setattr(
                            self,
                            key,
                            value.copy() if hasattr(value, "copy") else value,
                        )
                if self.excluded_targets:
                    self.exclude_targets()

        for i in range(n_holdout_sets):
            result = evaluate_single_holdout_robustness(i)
            if result is not None:
                all_results.append(result)

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
        for target_name in self.original_target_names:
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
        recommendation = self._generate_robustness_recommendation(
            overall_metrics, target_robustness_df
        )

        # Save results if requested
        def save_holdout_results(
            save_path: str,
            overall_metrics: Dict[str, float],
            target_robustness_df: pd.DataFrame,
            detailed_results: List[Dict[str, Any]],
        ) -> None:
            """Save detailed holdout results to CSV files."""
            from pathlib import Path

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
            detailed_path = save_path.with_name(
                f"{save_path.stem}_detailed.csv"
            )
            detailed_df.to_csv(detailed_path, index=False)

        if save_results_to:
            save_holdout_results(
                save_results_to,
                overall_metrics,
                target_robustness_df,
                all_results,
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

    def _generate_robustness_recommendation(
        self,
        overall_metrics: Dict[str, float],
        target_robustness_df: pd.DataFrame,
    ) -> str:
        """Generate interpretation and recommendations based on robustness evaluation."""

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
            if self.regularize_with_l0:
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
