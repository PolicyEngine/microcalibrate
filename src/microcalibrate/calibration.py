import logging
from typing import Callable, Optional

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
            csv_path (str): Optional path to save performance logs as CSV. Defaults to None.
        """

        self.estimate_function = estimate_function
        self.target_names = target_names

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )

        if self.estimate_function is None:
            self.estimate_function = (
                lambda weights: weights @ self.estimate_matrix_tensor
            )
        if estimate_matrix is not None:
            self.estimate_matrix = estimate_matrix
            self.estimate_matrix_tensor = torch.tensor(
                estimate_matrix.values, dtype=torch.float32, device=self.device
            )
            self.target_names = estimate_matrix.columns.to_numpy()
        else:
            self.estimate_matrix = None
        self.weights = weights
        self.targets = targets
        self.epochs = epochs
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.csv_path = csv_path
        self.performance_df = None

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

        new_weights, self.performance_df = reweight(
            original_weights=self.weights,
            estimate_function=self.estimate_function,
            targets_array=self.targets,
            target_names=self.target_names,
            epochs=self.epochs,
            noise_level=self.noise_level,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            csv_path=self.csv_path,
            device=self.device,
        )

        self.weights = new_weights

        return self.performance_df

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
            if estimate_val == 0:
                logger.warning(
                    f"Column {target_names[i]} has a zero estimate sum; using ε={eps} for comparison."
                )

            order_diff = np.log10(abs(ratio)) if ratio != 0 else np.inf
            if order_diff > 1:
                logger.warning(
                    f"Target {target_names[i]} ({target_val:.2e}) differs from initial estimate ({estimate_val:.2e}) "
                    f"by {order_diff:.2f} orders of magnitude."
                )
            if estimate_matrix is not None:
                contributing_mask = estimate_matrix.iloc[:, i] != 0
                contribution_ratio = (
                    contributing_mask.sum() / estimate_matrix.shape[0]
                )
                if contribution_ratio < 0.01:
                    logger.warning(
                        f"Target {target_names[i]} is supported by only {contribution_ratio:.2%} "
                        f"of records in the loss matrix. This may make calibration unstable or ineffective."
                    )

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
