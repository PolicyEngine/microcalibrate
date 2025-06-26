import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from .utils.log_performance import log_performance_over_epochs
from .utils.metrics import loss, pct_close

logger = logging.getLogger(__name__)


def reweight(
    original_weights: np.ndarray,
    estimate_function: Callable[[Tensor], Tensor],
    targets_array: np.ndarray,
    target_names: np.ndarray,
    dropout_rate: Optional[float] = 0.05,
    epochs: Optional[int] = 2_000,
    noise_level: Optional[float] = 10.0,
    learning_rate: Optional[float] = 1e-3,
    csv_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reweight the original weights based on the loss matrix and targets.

    Args:
        original_weights (np.ndarray): Original weights to be reweighted.
        estimate_function (Callable[[Tensor], Tensor]): Function to estimate targets from weights.
        targets_array (np.ndarray): Array of target values.
        target_names (np.ndarray): Names of the targets.
        dropout_rate (float): Optional probability of dropping weights during training.
        epochs (int): Optional number of epochs for training.
        noise_level (float): Optional level of noise to add to the original weights.
        learning_rate (float): Optional learning rate for the optimizer.

    Returns:
        np.ndarray: Reweighted weights.
        performance_df (pd.DataFrame): DataFrame containing the performance metrics over epochs.
    """
    if csv_path is not None and not csv_path.endswith(".csv"):
        raise ValueError("csv_path must be a string ending with .csv")

    logger.info(
        f"Starting calibration process for targets {target_names}: {targets_array}"
    )
    logger.info(
        f"Original weights - mean: {original_weights.mean():.4f}, "
        f"std: {original_weights.std():.4f}"
    )

    targets = torch.tensor(targets_array, dtype=torch.float32, device=device)
    random_noise = np.random.random(original_weights.shape) * noise_level
    weights = torch.tensor(
        np.log(original_weights + random_noise),
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    logger.info(
        f"Initial weights after noise - mean: {torch.exp(weights).mean().item():.4f}, "
        f"std: {torch.exp(weights).std():.4f}"
    )

    def dropout_weights(weights: torch.Tensor, p: float) -> torch.Tensor:
        """Apply dropout to the weights.

        Args:
            weights (torch.Tensor): Current weights in log space.
            p (float): Probability of dropping weights.

        Returns:
            torch.Tensor: Weights after applying dropout.
        """
        if p == 0:
            return weights
        total_weight = weights.sum()
        mask = torch.rand_like(weights) < p
        masked_weights = weights.clone()
        masked_weights[mask] = 0
        masked_weights = masked_weights / masked_weights.sum() * total_weight
        return masked_weights

    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    iterator = tqdm(range(epochs), desc="Reweighting progress", unit="epoch")
    tracking_n = max(1, epochs // 10) if epochs > 10 else 1
    progress_update_interval = 10

    loss_over_epochs = []
    estimates_over_epochs = []
    pct_close_over_epochs = []
    max_epochs = epochs - 1 if epochs > 0 else 0
    epochs = []

    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        estimate = estimate_function(torch.exp(weights_))
        l = loss(estimate, targets)
        close = pct_close(estimate, targets)

        if i % progress_update_interval == 0:
            iterator.set_postfix(
                {
                    "loss": l.item(),
                    "weights_mean": torch.exp(weights).mean().item(),
                    "weights_std": torch.exp(weights).std().item(),
                    "weights_min": torch.exp(weights).min().item(),
                }
            )

        if i % tracking_n == 0:
            epochs.append(i)
            loss_over_epochs.append(l.item())
            pct_close_over_epochs.append(close)
            estimates_over_epochs.append(estimate.detach().cpu().numpy())

            logger.info(f"Within 10% from targets: {close:.2%} \n")

            if len(loss_over_epochs) > 1:
                loss_change = loss_over_epochs[-2] - l.item()
                logger.info(
                    f"Epoch {i:4d}: Loss = {l.item():.6f}, "
                    f"Change = {loss_change:.6f} "
                    f"({'improving' if loss_change > 0 else 'worsening'})"
                )

        if i != max_epochs - 1:
            l.backward()
            optimizer.step()

    tracker_dict = {
        "epochs": epochs,
        "loss": loss_over_epochs,
        "estimates": estimates_over_epochs,
    }

    performance_df = log_performance_over_epochs(
        tracker_dict, targets, target_names
    )

    if csv_path:
        # Create directory if it doesn't exist
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(csv_path, index=True)

    logger.info(f"Reweighting completed. Final sample size: {len(weights)}")

    final_weights = torch.exp(weights_).detach().cpu().numpy()

    return (
        final_weights,
        performance_df,
    )
