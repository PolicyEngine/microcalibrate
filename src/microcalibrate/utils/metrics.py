""" " Metrics for evaluating performance in microcalibration."""

from typing import Optional

import torch


def loss(
    estimate: torch.Tensor,
    targets_array: torch.Tensor,
    normalization_factor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calculate the loss based on the current weights and targets.

    Args:
        estimate (torch.Tensor): Current estimates in log space.
        targets_array (torch.Tensor): Array of target values.
        normalization_factor (Optional[torch.Tensor]): Optional normalization factor for the loss (handles multi-level geographical calibration).

    Returns:
        torch.Tensor: Mean squared relative error between estimated and target values.
    """
    rel_error = (((estimate - targets_array) + 1) / (targets_array + 1)) ** 2
    if normalization_factor is not None:
        rel_error *= normalization_factor
    if torch.isnan(rel_error).any():
        raise ValueError("Relative error contains NaNs")
    return rel_error.mean()


def pct_close(
    estimate: torch.Tensor,
    targets_array: torch.Tensor,
    t: Optional[float] = 0.1,
) -> float:
    """Calculate the percentage of estimates close to targets.

    Args:
        estimate (torch.Tensor): Current estimates in log space.
        targets_array (torch.Tensor): Array of target values.
        t (float): Optional threshold for closeness.

    Returns:
        float: Percentage of estimates within the threshold.
    """
    abs_error = torch.abs((estimate - targets_array) / (1 + targets_array))
    return ((abs_error < t).sum() / abs_error.numel()).item()
