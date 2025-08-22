import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from torch import Tensor

from microcalibrate.evaluation import (
    evaluate_holdout_robustness as _evaluate_holdout_robustness,
)
from microcalibrate.hyperparameter_tuning import (
    tune_l0_hyperparameters as _tune_l0_hyperparameters,
)
from microcalibrate.reweight import reweight


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
        l0_lambda: Optional[float] = 5e-6,  # best between 1e-6 and 1e-5
        init_mean: Optional[
            float
        ] = 0.999,  # initial proportion with non-zero weights, set near 0
        temperature: Optional[float] = 0.5,  # usual values .5 to 3
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
            self.logger.warning(
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

    def tune_l0_hyperparameters(
        self,
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
        return _tune_l0_hyperparameters(
            calibration=self,
            n_trials=n_trials,
            objectives_balance=objectives_balance,
            epochs_per_trial=epochs_per_trial,
            n_holdout_sets=n_holdout_sets,
            holdout_fraction=holdout_fraction,
            aggregation=aggregation,
            timeout=timeout,
            n_jobs=n_jobs,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

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
        return _evaluate_holdout_robustness(
            calibration=self,
            n_holdout_sets=n_holdout_sets,
            holdout_fraction=holdout_fraction,
            save_results_to=save_results_to,
        )
