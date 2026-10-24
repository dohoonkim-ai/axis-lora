"""Training losses: composite InfoNCE over axis counterfactuals + KD anchor."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

AXES = ("TEMPORAL", "SPATIAL", "ENTITY", "EVENT")


def score_matrix(q_emb: Dict[str, torch.Tensor],
                 m_emb: Dict[str, torch.Tensor],
                 gate_module,
                 alpha: float,
                 w_override: torch.Tensor | None = None) -> torch.Tensor:
    """(Bq, Bm) composite scores for all query x memory pairs.

    w_override: optional (Bq, 4) routing weights used in place of the gate
    output (fixed or label-based routing in the gate-analysis appendix).
    """
    base_sim = q_emb["base"] @ m_emb["base"].T
    w = w_override if w_override is not None else gate_module(q_emb["base"])
    axis_sim = torch.zeros_like(base_sim)
    for ai, ax in enumerate(AXES):
        axis_sim = axis_sim + w[:, ai:ai + 1] * (q_emb[ax] @ m_emb[ax].T)
    return base_sim + alpha * axis_sim


def composite_infonce_loss(q_emb, mem_emb, pos_indices, gate_module, alpha,
                           tau: float = 0.05, w_override=None) -> torch.Tensor:
    """InfoNCE over the batch memory pool.

    The pool holds every query's positive plus all sampled axis counterfactuals,
    so each query is contrasted against its own counterfactuals (hard negatives)
    and the other in-batch memories.
    """
    logits = score_matrix(q_emb, mem_emb, gate_module, alpha, w_override=w_override) / tau
    return F.cross_entropy(logits, pos_indices)


def kd_anchor_loss(anchor_emb: Dict[str, torch.Tensor], axes: List[str] = list(AXES)) -> torch.Tensor:
    """Keeps each axis branch close to the frozen base on general-domain text.

    Mean over axes of the squared L2 distance between phi_A(x) and z_b(x).
    """
    base = anchor_emb["base"]
    loss = base.new_zeros(())
    for ax in axes:
        loss = loss + ((anchor_emb[ax] - base) ** 2).sum(dim=-1).mean()
    return loss / max(len(axes), 1)


def total_loss(q_emb, mem_emb, pos_indices, anchor_emb, gate_module, alpha,
               tau: float = 0.05, lambda_kd: float = 1.0,
               w_override=None) -> tuple[torch.Tensor, dict]:
    L_contrastive = composite_infonce_loss(q_emb, mem_emb, pos_indices, gate_module,
                                           alpha, tau, w_override=w_override)
    parts = {"contrastive": float(L_contrastive.detach())}
    L = L_contrastive
    if anchor_emb is not None and lambda_kd > 0:
        L_kd = kd_anchor_loss(anchor_emb)
        L = L + lambda_kd * L_kd
        parts["kd"] = float(L_kd.detach())
    parts["total"] = float(L.detach())
    return L, parts
