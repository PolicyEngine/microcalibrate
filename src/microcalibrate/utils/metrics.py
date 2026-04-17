""" " Metrics for evaluating performance in microcalibration."""

from typing import Optional

import torch


def _safe_denominator(
    targets_array: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Return a strictly-positive denominator for the (est-t)/(t+1) style
    relative-error metrics.

    The original formulation used ``targets_array + 1`` which crosses
    zero at ``target == -1`` (producing Inf) and is nearly zero for any
    target in a small neighbourhood. Clamping by a small ``eps``
    preserves the exact value for all well-behaved targets
    (``target + 1 >= eps``) while guaranteeing finite output when a
    target happens to equal ``-1``.
    """
    return torch.clamp(targets_array + 1, min=eps)


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
    denominator = _safe_denominator(targets_array)
    rel_error = (((estimate - targets_array) + 1) / denominator) ** 2
    if normalization_factor is not None:
        rel_error *= normalization_factor
    if not torch.isfinite(rel_error).all():
        raise ValueError(
            "Relative error contains non-finite values (NaN or Inf). "
            "Check the inputs: targets near -1, zero initial weights, "
            "or Inf/NaN from the estimate function can cause this."
        )
    return rel_error.mean()


def pct_close(
    estimate: torch.Tensor,
    targets: torch.Tensor,
    t: Optional[float] = 0.1,
) -> float:
    """Calculate the percentage of estimates close to targets.

    Args:
        estimate (torch.Tensor): Current estimates in log space.
        targets (torch.Tensor): Array of target values.
        t (float): Optional threshold for closeness.

    Returns:
        float: Percentage of estimates within the threshold.
    """
    denominator = _safe_denominator(targets)
    abs_error = torch.abs((estimate - targets) / denominator)
    return ((abs_error < t).sum() / abs_error.numel()).item()
