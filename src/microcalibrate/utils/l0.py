import math

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

    def reset_parameters(self):
        if self.init_mean is not None:
            init_val = math.log(self.init_mean / (1 - self.init_mean))
            self.qz_logits.data.fill_(init_val)

    def forward(self, input_shape=None):
        if self.training:
            gates = self._sample_gates()
        else:
            gates = self._deterministic_gates()
        if input_shape is not None and len(input_shape) > len(gates.shape):
            gates = gates.unsqueeze(-1).unsqueeze(-1)
        return gates

    def _sample_gates(self):
        u = torch.zeros_like(self.qz_logits).uniform_(1e-8, 1.0 - 1e-8)
        s = torch.log(u) - torch.log(1 - u) + self.qz_logits
        s = torch.sigmoid(s / self.temperature)
        s = s * (self.zeta - self.gamma) + self.gamma
        gates = torch.clamp(s, 0, 1)
        return gates

    def _deterministic_gates(self):
        probs = torch.sigmoid(self.qz_logits)
        gates = probs * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(gates, 0, 1)

    def get_penalty(self):
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        prob_active = torch.sigmoid(logits_shifted)
        return prob_active.sum()

    def get_active_prob(self):
        logits_shifted = self.qz_logits - self.temperature * math.log(
            -self.gamma / self.zeta
        )
        return torch.sigmoid(logits_shifted)
