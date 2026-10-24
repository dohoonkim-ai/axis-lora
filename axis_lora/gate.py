"""Per-query axis gate.

The gate maps the frozen-base query embedding to routing weights over the four
axes. The default activation is sparsemax (Martins & Astudillo, ICML 2016),
which projects onto the simplex and returns exact zeros for inactive axes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax along `dim`: non-negative, sums to 1, exact zeros below the threshold."""
    sorted_logits, _ = logits.sort(dim=dim, descending=True)

    # Support: largest k with 1 + k * z_(k) > sum_{j<=k} z_(j)
    cumsum = sorted_logits.cumsum(dim=dim)
    n_dim = sorted_logits.size(dim)
    rho = torch.arange(1, n_dim + 1, device=logits.device, dtype=logits.dtype)
    rho_shape = [1] * sorted_logits.dim()
    rho_shape[dim] = -1
    rho = rho.view(*rho_shape)
    support_mask = (1 + rho * sorted_logits) > cumsum

    k = support_mask.to(logits.dtype).sum(dim=dim, keepdim=True)
    k_int = k.long().clamp(min=1)

    # Threshold tau and projection max(z - tau, 0)
    cumsum_at_k = cumsum.gather(dim, k_int - 1)
    tau = (cumsum_at_k - 1) / k
    return (logits - tau).clamp(min=0)


class SparsemaxGate(nn.Module):
    """MLP (hidden -> 64 -> n_axes) followed by a temperature-scaled activation.

    mode:
        "sparsemax"  paper default
        "softmax"    gate variant used with axis supervision (appendix)
        "uniform"    ablation: constant 1/n_axes, the gate is not used
    The temperature `tau` is annealed by the training loop (2.0 -> 0.5).
    """

    def __init__(self, hidden_size: int, n_axes: int = 4,
                 mlp_hidden: int = 64, init_tau: float = 2.0,
                 mode: str = "sparsemax"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, n_axes),
        )
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        assert mode in ("sparsemax", "softmax", "uniform"), mode
        self.mode = mode
        self.n_axes = int(n_axes)

    def forward(self, q_base: torch.Tensor) -> torch.Tensor:
        """q_base: (B, hidden_size) -> routing weights (B, n_axes)."""
        logits = self.mlp(q_base)
        if self.mode == "uniform":
            return torch.full_like(logits, 1.0 / self.n_axes)
        if self.mode == "softmax":
            return torch.softmax(logits / self.tau.clamp(min=1e-3), dim=-1)
        return sparsemax(logits / self.tau.clamp(min=1e-3), dim=-1)
