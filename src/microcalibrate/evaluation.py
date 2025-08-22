import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

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
