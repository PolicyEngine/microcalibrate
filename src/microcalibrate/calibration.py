import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class Calibration:
    def __init__(
        self,
        weights: np.ndarray,
        targets: np.ndarray,
        target_names: Optional[np.ndarray] = None,
        estimate_matrix: Optional[pd.DataFrame] = None,
        estimate_function: Optional[Callable[[Tensor], Tensor]] = None,
        epochs: Optional[int] = 32,
        noise_level: Optional[float] = 10.0,
        learning_rate: Optional[float] = 1e-3,
        dropout_rate: Optional[float] = 0.1,
        normalization_factor: Optional[torch.Tensor] = None,
        excluded_targets: Optional[List[str]] = None,
        csv_path: Optional[str] = None,
        device: str = None,
    ):
        """Initialize the Calibration class.

        Args:
            weights (np.ndarray): Array of original weights.
            targets (np.ndarray): Array of target values.
            target_names (Optional[np.ndarray]): Optional names of the targets for logging. Defaults to None. You MUST pass these names if you are not passing in an estimate matrix, and just passing in an estimate function.
            estimate_matrix (pd.DataFrame): DataFrame containing the estimate matrix.
            estimate_function (Optional[Callable[[Tensor], Tensor]]): Function to estimate targets from weights. Defaults to None, in which case it will use the estimate_matrix.
            epochs (int): Optional number of epochs for calibration. Defaults to 32.
            noise_level (float): Optional level of noise to add to weights. Defaults to 10.0.
            learning_rate (float): Optional learning rate for the optimizer. Defaults to 1e-3.
            dropout_rate (float): Optional probability of dropping weights during training. Defaults to 0.1.
            normalization_factor (Optional[torch.Tensor]): Optional normalization factor for the loss (handles multi-level geographical calibration). Defaults to None.
            excluded_targets (Optional[List]): Optional List of targets to exclude from calibration. Defaults to None.
            csv_path (str): Optional path to save performance logs as CSV. Defaults to None.
            device (str): Optional device to run the calibration on. Defaults to None, which will use CUDA if available, otherwise MPS, otherwise CPU.
        """
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )
        self.original_estimate_matrix = estimate_matrix
        self.original_targets = targets
        self.original_target_names = target_names
        self.weights = weights
        self.excluded_targets = excluded_targets
        self.estimate_function = estimate_function
        self.epochs = epochs
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.normalization_factor = normalization_factor
        self.csv_path = csv_path
        self.performance_df = None

        self.estimate_matrix = None
        self.targets = None
        self.target_names = None
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

        if self.estimate_function is None:
            if self.estimate_matrix is not None:
                self.estimate_function = (
                    lambda weights: weights @ self.estimate_matrix
                )
            else:
                raise ValueError(
                    "Either estimate_function or estimate_matrix must be provided"
                )
        elif self.excluded_targets:
            logger.warning(
                "You are passing an estimate function with excluded targets. "
                "Make sure the function handles excluded targets correctly, as reweight() will handle the filtering."
            )

    def calibrate(self) -> None:
        """Calibrate the weights based on the estimate function and targets."""

        self._assess_targets(
            estimate_function=self.estimate_function,
            estimate_matrix=getattr(
                self, "original_estimate_matrix", self.estimate_matrix
            ),
            weights=self.weights,
            targets=self.targets,
            target_names=self.target_names,
        )

        from .reweight import reweight

        new_weights, self.performance_df = reweight(
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
        )

        self.weights = new_weights

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

            logger.info(
                f"Excluded {len(excluded_indices)} targets from calibration: {self.excluded_targets}"
            )
            logger.info(f"Calibrating {len(targets_array)} targets")
        else:
            targets_array = self.original_targets
            target_names = self.original_target_names

        # Get initial estimates for excluded targets if needed
        if self.excluded_targets:
            initial_weights_tensor = torch.tensor(
                self.weights, dtype=torch.float32, device=self.device
            )
            if self.estimate_function is not None:
                initial_estimates_all = (
                    self.estimate_function(initial_weights_tensor)
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
            else:
                self.estimate_matrix = None

        # Set up final attributes
        self.targets = targets_array
        self.target_names = target_names

    def estimate(self) -> pd.Series:
        return pd.Series(
            index=self.target_names,
            data=self.estimate_function(
                torch.tensor(
                    self.weights, dtype=torch.float32, device=self.device
                )
            )
            .cpu()
            .detach()
            .numpy(),
        )

    def _assess_targets(
        self,
        estimate_function: Callable[[Tensor], Tensor],
        estimate_matrix: Optional[pd.DataFrame],
        weights: np.ndarray,
        targets: np.ndarray,
        target_names: Optional[np.ndarray] = None,
    ) -> None:
        """Assess the targets to ensure they do not violate basic requirements like compatibility, correct order of magnitude, etc.

        Args:
            estimate_function (Callable[[Tensor], Tensor]): Function to estimate the targets from weights.
            estimate_matrix (Optional[pd.DataFrame]): DataFrame containing the estimate matrix. Defaults to None.
            weights (np.ndarray): Array of original weights.
            targets (np.ndarray): Array of target values.
            target_names (np.ndarray): Optional names of the targets for logging. Defaults to None.

        Raises:
            ValueError: If the targets do not match the expected format or values.
            ValueError: If the targets are not compatible with each other.
        """
        logger.info("Performing basic target assessment...")

        if targets.ndim != 1:
            raise ValueError("Targets must be a 1D NumPy array.")

        if np.any(np.isnan(targets)):
            raise ValueError("Targets contain NaN values.")

        if np.any(targets < 0):
            logger.warning(
                "Some targets are negative. This may not make sense for totals."
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
                logger.warning(
                    f"Column {target_name} has a zero estimate sum; using ε={eps} for comparison."
                )

            order_diff = np.log10(abs(ratio)) if ratio != 0 else np.inf
            if order_diff > 1:
                logger.warning(
                    f"Target {target_name} ({target_val:.2e}) differs from initial estimate ({estimate_val:.2e}) "
                    f"by {order_diff:.2f} orders of magnitude."
                )
            if estimate_matrix is not None:
                # Check if estimate_matrix is a tensor or DataFrame
                if hasattr(estimate_matrix, "iloc"):
                    # It's a DataFrame
                    contributing_mask = estimate_matrix.iloc[:, i] != 0
                    contribution_ratio = (
                        contributing_mask.sum() / estimate_matrix.shape[0]
                    )
                else:
                    # It's a tensor
                    contributing_mask = estimate_matrix[:, i] != 0
                    contribution_ratio = (
                        contributing_mask.sum().item()
                        / estimate_matrix.shape[0]
                    )
                if contribution_ratio < 0.01:
                    logger.warning(
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

        if logger.level != logging.INFO:
            previous_level = logger.level
            logger.setLevel(logging.INFO)

        def _get_linear_loss(metrics_matrix, target_vector, sparse=False):
            """Gets the mean squared error loss of X.T @ w wrt y for least squares solution"""
            X = metrics_matrix
            y = target_vector
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

            return round(np.mean((y - y_hat) ** 2), 3)  # mostly for display

        X = self.estimate_matrix_tensor.cpu().numpy()
        y = self.targets

        slices = []
        iterative_losses = []
        idx_dict = {
            self.estimate_matrix.columns.to_list()[i]: i
            for i in range(len(self.estimate_matrix.columns))
        }

        logger.info(
            "Assessing analytical solution to the optimization problem for each target... \n"
            "This evaluates how much each target complicates achieving calibration accuracy. The loss reported is the mean squared error of the least squares solution."
        )

        for target_name, index_list in idx_dict.items():
            slices.append(index_list)
            loss = _get_linear_loss(X[:, slices], y[slices], use_sparse)
            iterative_losses.append(loss)
            logger.info(
                f"Adding: {target_name} to the above, Loss: {loss}, Change in loss: {iterative_losses[-1] - iterative_losses[-2] if len(iterative_losses) > 1 else 'N/A'}"
            )

        logger.setLevel(previous_level)

    def summary(
        self,
    ) -> str:
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
