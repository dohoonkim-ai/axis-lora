"""Axis-LoRA: a frozen embedding model with four axis-conditional LoRA branches
and a per-query gate.

    score(q, m) = <z_b(q), z_b(m)> + alpha * sum_A w_A(q) <phi_A(q), phi_A(m)>

z_b is the frozen-base embedding, phi_A the embedding with only the axis-A LoRA
branch active, and w_A the gate's routing weight for axis A. All embeddings are
L2-normalized. Setting alpha = 0 recovers the frozen base model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from .gate import SparsemaxGate

AXES = ("TEMPORAL", "SPATIAL", "ENTITY", "EVENT")


def last_layers(n_layers: int, last_pct: float = 0.2) -> List[int]:
    """Indices of the last `last_pct` fraction of transformer layers."""
    return list(range(math.ceil((1 - last_pct) * n_layers), n_layers))


def make_target_modules(layers: Sequence[int], base_model: str,
                        targets: Tuple[str, ...] = None) -> List[str]:
    """Explicit per-layer LoRA target modules (query and value projections).

    Qwen-style decoders: layers.{i}.self_attn.{q_proj,v_proj}
    BERT-style encoders (MiniLM, bge, e5, ...): encoder.layer.{i}.attention.self.{query,value}
    """
    name = base_model.lower()
    is_bert = ("minilm" in name or "bert" in name or "bge" in name
               or "arctic-embed" in name or "e5-" in name)
    if is_bert:
        targets = targets or ("query", "value")
        return [f"encoder.layer.{i}.attention.self.{m}" for i in layers for m in targets]
    targets = targets or ("q_proj", "v_proj")
    return [f"layers.{i}.self_attn.{m}" for i in layers for m in targets]


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last non-padded token per row (right padding)."""
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


@dataclass
class AxisLoRAConfig:
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    pooling: str = "last"              # "last" (Qwen3-Embedding) or "mean" (MiniLM, bge)
    rank: int = 16
    lora_alpha_mul: int = 2            # LoRA alpha = rank * lora_alpha_mul
    lora_dropout: float = 0.05
    last_pct: float = 0.2              # LoRA on the last 20% of attention layers
    gate_mlp_hidden: int = 64
    gate_init_tau: float = 2.0
    gate_mode: str = "sparsemax"
    bf16: bool = True
    trust_remote_code: bool = False
    padding_side: str = "right"

    @classmethod
    def from_train_args(cls, saved: dict) -> "AxisLoRAConfig":
        """Build the architecture config from a training run's config.json.

        Also accepts configs written by the original research code, which used
        separate TEMPORAL/other ranks and a TEMPORAL-specific layer range; the
        released runs set these so that every axis uses the same rank and the
        same last-20% layers, which is the only setting supported here.
        """
        saved = dict(saved)
        if "rank_temporal" in saved:
            rt, ro = saved.pop("rank_temporal"), saved.pop("rank_other", None)
            if ro is not None and ro != rt:
                raise ValueError(f"unsupported legacy config: rank_temporal={rt} != rank_other={ro}")
            saved.setdefault("rank", rt)
        last_pct = saved.get("last_pct", cls.last_pct)
        tf, tm = saved.pop("temp_first_pct", 0.0), saved.pop("temp_mid_start_pct", None)
        if tf != 0.0 or (tm is not None and not math.isclose(tm, 1 - last_pct)):
            raise ValueError("unsupported legacy config: TEMPORAL layer range differs from last_pct")
        if saved.get("no_layer_prune"):
            raise ValueError("unsupported legacy config: no_layer_prune=True")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in saved.items() if k in names})


class AxisLoRAModel(nn.Module):
    """Frozen base + 4 axis-conditional LoRA adapters (PEFT multi-adapter) + gate.

    encode_all(...) returns {"base", "TEMPORAL", "SPATIAL", "ENTITY", "EVENT"},
    each (B, d) and L2-normalized, from 5 forward passes (base + one per axis).
    """

    def __init__(self, cfg: AxisLoRAConfig):
        super().__init__()
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        base = AutoModel.from_pretrained(cfg.base_model, dtype=dtype,
                                         trust_remote_code=cfg.trust_remote_code)
        for p in base.parameters():
            p.requires_grad_(False)

        self.n_layers = base.config.num_hidden_layers
        self.hidden_size = base.config.hidden_size
        self.lora_layers = last_layers(self.n_layers, cfg.last_pct)

        def lora_cfg():
            return LoraConfig(
                r=cfg.rank,
                lora_alpha=cfg.rank * cfg.lora_alpha_mul,
                lora_dropout=cfg.lora_dropout,
                target_modules=make_target_modules(self.lora_layers, cfg.base_model),
                bias="none",
            )

        self.peft_model = get_peft_model(base, lora_cfg(), adapter_name=AXES[0])
        for ax in AXES[1:]:
            self.peft_model.add_adapter(ax, lora_cfg())

        # PEFT only activates the current adapter; all four are trained jointly.
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

        self.gate = SparsemaxGate(
            hidden_size=self.hidden_size,
            n_axes=len(AXES),
            mlp_hidden=cfg.gate_mlp_hidden,
            init_tau=cfg.gate_init_tau,
            mode=cfg.gate_mode,
        ).to(dtype=torch.float32)          # the gate stays in fp32

        self._tokenizer = None

    def _pool(self, hidden, attention_mask):
        if self.cfg.pooling == "last":
            return last_token_pool(hidden, attention_mask)
        return mean_pool(hidden, attention_mask)

    def encode_base(self, input_ids, attention_mask, **kwargs):
        """Frozen-base encoding (all adapters disabled)."""
        with self.peft_model.disable_adapter():
            out = self.peft_model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=False)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_axis(self, input_ids, attention_mask, axis: str, **kwargs):
        """Encoding with only the axis-`axis` LoRA branch active."""
        assert axis in AXES, axis
        self.peft_model.set_adapter(axis)
        # set_adapter() disables grad on the inactive adapters; re-enable all of
        # them so that every branch keeps training.
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        out = self.peft_model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=False)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_all(self, input_ids, attention_mask, **kwargs) -> Dict[str, torch.Tensor]:
        result = {"base": self.encode_base(input_ids, attention_mask)}
        for ax in AXES:
            result[ax] = self.encode_axis(input_ids, attention_mask, ax)
        return {k: F.normalize(v.float(), dim=-1) for k, v in result.items()}

    def score(self, q_emb: Dict[str, torch.Tensor], m_emb: Dict[str, torch.Tensor],
              alpha: float) -> torch.Tensor:
        """Paired score for (B, d) query/memory embedding dicts -> (B,)."""
        base_sim = (q_emb["base"] * m_emb["base"]).sum(-1)
        w = self.gate(q_emb["base"])
        axis_sim = torch.zeros_like(base_sim)
        for ai, ax in enumerate(AXES):
            axis_sim = axis_sim + w[:, ai] * (q_emb[ax] * m_emb[ax]).sum(-1)
        return base_sim + alpha * axis_sim

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.base_model, trust_remote_code=self.cfg.trust_remote_code,
                padding_side=self.cfg.padding_side)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def trainable_param_summary(self):
        per_axis = {ax: 0 for ax in AXES}
        gate = 0
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                for ax in AXES:
                    if f".{ax}." in n:
                        per_axis[ax] += p.numel()
                        break
            elif "gate" in n:
                gate += p.numel()
        return {"per_axis": per_axis, "gate": gate, "total": sum(per_axis.values()) + gate}
