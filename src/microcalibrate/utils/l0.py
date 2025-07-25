import logging
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class HardConcrete(nn.Module):
    """HardConcrete distribution for L0 regularization."""

    def __init__(
        self,
        input_dim,
        output_dim=None,
        temperature=0.5,
        stretch=0.1,
        init_mean=0.5,
    ):
        super().__init__()
        if output_dim is None:
            self.gate_size = (input_dim,)
        else:
            self.gate_size = (input_dim, output_dim)
        self.qz_logits = nn.Parameter(torch.zeros(self.gate_size))
        self.temperature = temperature
        self.stretch = stretch
        self.gamma = -0.1
        self.zeta = 1.1
        self.init_mean = init_mean
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_mean is not None:
            init_val = math.log(self.init_mean / (1 - self.init_mean))
            self.qz_logits.data.fill_(init_val)

    def forward(
        self, input_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        if self.training:
            gates = self._sample_gates()
        else:
            gates = self._deterministic_gates()
        if input_shape is not None and len(input_shape) > len(gates.shape):
            gates = gates.unsqueeze(-1).unsqueeze(-1)
        return gates

    def _sample_gates(self) -> torch.Tensor:
        u = torch.zeros_like(self.qz_logits).uniform_(1e-8, 1.0 - 1e-8)
        s = torch.log(u) - torch.log(1 - u) + self.qz_logits
        s = torch.sigmoid(s / self.temperature)
        s = s * (self.zeta - self.gamma) + self.gamma
        gates = torch.clamp(s, 0, 1)
        return gates

    def _deterministic_gates(self) -> torch.Tensor:
        probs = torch.sigmoid(self.qz_logits)
        gates = probs * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(gates, 0, 1)

    def get_penalty(self) -> torch.Tensor:
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        prob_active = torch.sigmoid(logits_shifted)
        return prob_active.sum()

    def get_active_prob(self) -> torch.Tensor:
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        return torch.sigmoid(logits_shifted)


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
