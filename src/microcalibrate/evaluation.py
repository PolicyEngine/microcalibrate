import logging
from typing import List, Optional

import numpy as np
import pandas as pd

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
