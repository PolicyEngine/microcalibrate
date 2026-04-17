import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from l0 import HardConcrete
from torch import Tensor
from tqdm import tqdm

from .utils.log_performance import log_performance_over_epochs
from .utils.metrics import loss, pct_close


def dropout_weights(weights: torch.Tensor, p: float) -> torch.Tensor:
    """Apply inverted dropout to weights held in log space.

    ``weights`` represents log(w); downstream code computes
    ``torch.exp(weights_)`` to recover linear-space weights. Dropping
    an entry therefore means sending its log to ``-inf`` so that
    ``exp`` returns 0. Surviving entries are scaled by ``1/(1-p)`` in
    linear space (equivalently, ``-log(1-p)`` added in log space) so
    the expected linear-space sum is preserved, matching standard
    inverted dropout semantics.

    Args:
        weights (torch.Tensor): Current weights in log space.
        p (float): Probability of dropping each weight, in [0, 1].

    Returns:
        torch.Tensor: Weights in log space after applying dropout.
    """
    if p == 0:
        return weights
    if p < 0 or p > 1:
        raise ValueError(f"dropout_rate must be in [0, 1]; got {p}.")
    if p == 1:
        # Everything is dropped: zero all linear-space weights. The
        # result has no gradient path back to ``weights`` because every
        # entry is a constant -inf; callers must not rely on training
        # under full dropout.
        return torch.full_like(weights, float("-inf"))
    # ``survive_mask`` is True where an entry SURVIVES.
    survive_mask = torch.rand_like(weights) >= p
    neg_inf = torch.full_like(weights, float("-inf"))
    scale = -float(np.log1p(-p))  # == log(1/(1-p))
    scaled = weights + scale
    return torch.where(survive_mask, scaled, neg_inf)


def reweight(
    original_weights: np.ndarray,
    estimate_function: Callable[[Tensor], Tensor],
    targets_array: np.ndarray,
    target_names: np.ndarray,
    l0_lambda: float,
    init_mean: float,
    temperature: float,
    regularize_with_l0: bool,
    sparse_learning_rate: Optional[float] = 0.2,
    dropout_rate: Optional[float] = 0.05,
    epochs: Optional[int] = 2_000,
    noise_level: Optional[float] = 10.0,
    learning_rate: Optional[float] = 1e-3,
    normalization_factor: Optional[torch.Tensor] = None,
    excluded_targets: Optional[List] = None,
    excluded_target_data: Optional[dict] = None,
    csv_path: Optional[str] = None,
    device: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, Union[np.ndarray, None], pd.DataFrame]:
    """Reweight the original weights based on the loss matrix and targets.

    Args:
        original_weights (np.ndarray): Original weights to be reweighted.
        estimate_function (Callable[[Tensor], Tensor]): Function to estimate targets from weights.
        targets_array (np.ndarray): Array of target values.
        target_names (np.ndarray): Names of the targets.
        l0_lambda (float): Regularization parameter for L0 regularization.
        init_mean (float): Initial mean for L0 regularization, representing the initial proportion of non-zero weights.
        temperature (float): Temperature parameter for L0 regularization, controlling the sparsity of the model.
        sparse_learning_rate (float): Learning rate for the regularizing optimizer.
        regularize_with_l0 (bool): Whether to apply L0 regularization.
        dropout_rate (float): Optional probability of dropping weights during training.
        epochs (int): Optional number of epochs for training.
        noise_level (float): Optional level of noise to add to the original weights.
        learning_rate (float): Optional learning rate for the optimizer.
        normalization_factor (Optional[torch.Tensor]): Optional normalization factor for the loss (handles multi-level geographical calibration).
        excluded_targets (Optional[List]): Optional List of targets to exclude from calibration.
        excluded_target_data (Optional[dict]): Optional dictionary containing excluded target data with initial estimates and targets.
        csv_path (Optional[str]): Optional path to save the performance metrics as a CSV file.
        device (Optional[str]): Device to run the calibration on (e.g., 'cpu' or 'cuda'). If None, uses the default device.
        logger (Optional[logging.Logger]): Logger for logging progress and metrics.

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

    targets = torch.tensor(
        targets_array,
        dtype=torch.float32,
        device=device,
    )

    random_noise = np.random.random(original_weights.shape) * noise_level
    # Guard against non-positive values (e.g. zero initial weights with
    # noise_level=0) which would produce -inf in log space and NaN
    # gradients downstream.
    initial_weights = np.maximum(
        np.asarray(original_weights, dtype=np.float64) + random_noise,
        1e-12,
    )
    weights = torch.tensor(
        np.log(initial_weights),
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    logger.info(
        f"Initial weights after noise - mean: {torch.exp(weights).mean().item():.4f}, "
        f"std: {torch.exp(weights).std():.4f}"
    )

    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    iterator = tqdm(range(epochs), desc="Reweighting progress", unit="epoch")
    tracking_n = max(1, epochs // 10) if epochs > 10 else 1
    progress_update_interval = 10

    loss_over_epochs = []
    estimates_over_epochs = []
    pct_close_over_epochs = []
    epochs_list = []

    for i in iterator:
        optimizer.zero_grad()
        weights_ = dropout_weights(weights, dropout_rate)
        estimate = estimate_function(torch.exp(weights_))
        l = loss(estimate, targets, normalization_factor)
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

        # Log a tracking row every `tracking_n` epochs and always on the
        # final epoch so the tracker ends with the state that corresponds
        # to the returned weights (post last step = start of next epoch).
        is_final_epoch = i == epochs - 1
        if i % tracking_n == 0 or is_final_epoch:
            epochs_list.append(i)
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

        # Step every epoch. The returned final_weights reflect the state
        # after the last step; the final logged row above reflects the
        # pre-step state of the same (last) epoch.
        l.backward()
        optimizer.step()

    tracker_dict = {
        "epochs": epochs_list,
        "loss": loss_over_epochs,
        "estimates": estimates_over_epochs,
    }

    performance_df = log_performance_over_epochs(
        tracker_dict,
        targets,
        target_names,
        excluded_targets,
        excluded_target_data,
    )

    if csv_path:
        # Create directory if it doesn't exist
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        performance_df.to_csv(csv_path, index=True)

    logger.info(
        f"Dense reweighting completed. Final sample size: {len(weights)}"
    )

    final_weights = torch.exp(weights_).detach().cpu().numpy()

    if regularize_with_l0:
        logger.info("Applying L0 regularization to the weights.")

        # Sparse, regularized weights depending on temperature, init_mean, l0_lambda -----
        # Guard against zero/negative initial weights which would produce
        # -inf or NaN after np.log and poison gradients.
        safe_original_weights = np.maximum(
            np.asarray(original_weights, dtype=np.float64), 1e-12
        )
        weights = torch.tensor(
            np.log(safe_original_weights),
            requires_grad=True,
            dtype=torch.float32,
            device=device,
        )
        gates = HardConcrete(
            len(original_weights),
            init_mean=init_mean,
            temperature=temperature,
        ).to(device)
        # NOTE: Results are pretty sensitve to learning rates
        # optimizer breaks down somewhere near .005, does better at above .1
        optimizer = torch.optim.Adam(
            [weights] + list(gates.parameters()), lr=sparse_learning_rate
        )
        start_loss = None

        loss_over_epochs_sparse = []
        estimates_over_epochs_sparse = []
        pct_close_over_epochs_sparse = []
        epochs_sparse = []

        iterator = tqdm(
            range(epochs * 2), desc="Sparse reweighting progress", unit="epoch"
        )  # lower learning rate, harder optimization

        for i in iterator:
            optimizer.zero_grad()
            weights_ = dropout_weights(weights, dropout_rate)
            masked = torch.exp(weights_) * gates()
            estimate = estimate_function(masked)
            l_main = loss(estimate, targets, normalization_factor)
            l = l_main + l0_lambda * gates.get_penalty()
            close = pct_close(estimate, targets)
            # The sparse loop runs 2x as many epochs as the dense loop,
            # so log twice as often (half the dense tracking stride).
            # Without explicit parentheses the original expression
            # `i % tracking_n / 2 == 0` parses as
            # `(i % tracking_n) / 2 == 0`, which is equivalent to
            # `i % tracking_n == 0` and silently loses the x2 density.
            sparse_tracking_n = max(1, tracking_n // 2)
            if i % sparse_tracking_n == 0:
                epochs_sparse.append(i)
                loss_over_epochs_sparse.append(l.item())
                pct_close_over_epochs_sparse.append(close)
                estimates_over_epochs_sparse.append(
                    estimate.detach().cpu().numpy()
                )

                logger.info(
                    f"Within 10% from targets in sparse calibration: {close:.2%} \n"
                )

                if len(loss_over_epochs_sparse) > 1:
                    loss_change = loss_over_epochs_sparse[-2] - l.item()
                    logger.info(
                        f"Epoch {i:4d}: Loss = {l.item():.6f}, "
                        f"Change = {loss_change:.6f} "
                        f"({'improving' if loss_change > 0 else 'worsening'})"
                    )
            if start_loss is None:
                start_loss = l.item()
            # Guard against a zero starting loss (trivial/pre-calibrated
            # data, or L0 warmup pushing the penalty term near zero) to
            # avoid ZeroDivisionError / inf in the tqdm postfix.
            if abs(start_loss) < 1e-12:
                loss_rel_change = 0.0
            else:
                loss_rel_change = (l.item() - start_loss) / start_loss
            l.backward()
            iterator.set_postfix(
                {"loss": l.item(), "loss_rel_change": loss_rel_change}
            )
            optimizer.step()

        gates.eval()
        final_weights_sparse = (
            (torch.exp(weights) * gates()).detach().cpu().numpy()
        )

        tracker_dict_sparse = {
            "epochs": epochs_sparse,
            "loss": loss_over_epochs_sparse,
            "estimates": estimates_over_epochs_sparse,
        }

        sparse_performance_df = log_performance_over_epochs(
            tracker_dict_sparse,
            targets,
            target_names,
            excluded_targets,
            excluded_target_data,
        )

        if csv_path:
            # Create directory if it doesn't exist
            csv_path = Path(str(csv_path).replace(".csv", "_sparse.csv"))
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sparse_performance_df.to_csv(csv_path, index=True)
    else:
        final_weights_sparse = None

    return (
        final_weights,
        final_weights_sparse,
        performance_df,
    )
