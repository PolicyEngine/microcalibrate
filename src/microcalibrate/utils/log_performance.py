from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def log_performance_over_epochs(
    tracked: Dict[str, List[Any]],
    targets: torch.Tensor,
    target_names: List[str],
) -> pd.DataFrame:
    """
    Calculate the errors and performance metrics for the model for all the logged epochs.

    Args:
        tracked (Dict[str, List[Any]]): Dictionary containing lists of tracked metrics.
        targets (torch.Tensor): Array of target values.
        targets_names (List[str]): Array of target names.

    Returns:
        performance_df: DataFrame containing the calculated errors and performance metrics.
    """
    targets = targets.detach().cpu().numpy().astype(float)  # (k,)
    k = len(targets)

    rows = []
    for epoch_i, epoch in enumerate(tracked["epochs"]):
        base = {
            "epoch": epoch,
            "loss": tracked["loss"][epoch_i],
        }

        # each estimate vector has shape (k,)
        estimates_vector = np.asarray(
            tracked["estimates"][epoch_i], dtype=float
        )

        for t_idx in range(k):
            target_val = targets[t_idx]
            est_val = estimates_vector[t_idx]
            err = est_val - target_val

            rows.append(
                {
                    **base,
                    "target_name": (
                        target_names[t_idx]
                        if target_names is not None
                        else None
                    ),
                    "target": target_val,
                    "estimate": est_val,
                    "error": err,
                    "abs_error": abs(err),
                    "rel_abs_error": (
                        abs(err) / abs(target_val)
                        if target_val != 0
                        else np.nan
                    ),
                }
            )

    df = pd.DataFrame(rows)

    if target_names is None:
        df = df.drop(columns="target_name")

    return df
