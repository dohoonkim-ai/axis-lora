"""Shared-trunk ablation: one shared LoRA + four axis projection heads.

Instead of four axis-conditional LoRA branches, a single LoRA (same layers)
adapts the backbone and four post-pooling MLP heads produce the per-axis
embeddings. Used for the shared-trunk rows of the ablation tables.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from .gate import SparsemaxGate
from .model import AXES, AxisLoRAConfig, last_layers, last_token_pool, make_target_modules, mean_pool


class SharedTrunkModel(nn.Module):

    def __init__(self, cfg: AxisLoRAConfig):
        super().__init__()
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        base = AutoModel.from_pretrained(cfg.base_model, dtype=dtype)
        for p in base.parameters():
            p.requires_grad_(False)

        self.n_layers = base.config.num_hidden_layers
        self.hidden_size = base.config.hidden_size
        self.lora_layers = last_layers(self.n_layers, cfg.last_pct)
        self.peft_model = get_peft_model(base, LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank * cfg.lora_alpha_mul,
            lora_dropout=cfg.lora_dropout,
            target_modules=make_target_modules(self.lora_layers, cfg.base_model),
            bias="none",
        ))
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)

        self.projection_heads = nn.ModuleDict({
            ax: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            ).to(dtype=torch.float32)
            for ax in AXES
        })
        self.gate = SparsemaxGate(
            hidden_size=self.hidden_size,
            n_axes=len(AXES),
            mlp_hidden=cfg.gate_mlp_hidden,
            init_tau=cfg.gate_init_tau,
        ).to(dtype=torch.float32)
        self._tokenizer = None

    def _pool(self, hidden, attention_mask):
        if self.cfg.pooling == "last":
            return last_token_pool(hidden, attention_mask)
        return mean_pool(hidden, attention_mask)

    def encode_all(self, input_ids, attention_mask, **kwargs) -> Dict[str, torch.Tensor]:
        """One forward with the shared LoRA on, then the four heads."""
        out = self.peft_model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=False)
        pooled = self._pool(out.last_hidden_state, attention_mask).float()
        result = {"base": F.normalize(pooled, dim=-1)}
        for ax in AXES:
            result[ax] = F.normalize(self.projection_heads[ax](pooled), dim=-1)
        return result

    def encode_for_kd(self, input_ids, attention_mask, **kwargs):
        """(pooled with LoRA, pooled frozen), for the backbone KD anchor."""
        out_lora = self.peft_model(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=False)
        pooled_lora = self._pool(out_lora.last_hidden_state, attention_mask)
        with self.peft_model.disable_adapter():
            out_no = self.peft_model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=False)
        pooled_no = self._pool(out_no.last_hidden_state, attention_mask)
        return (F.normalize(pooled_lora.float(), dim=-1),
                F.normalize(pooled_no.float(), dim=-1))

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def trainable_param_summary(self):
        n_lora = sum(p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad)
        n_heads = sum(p.numel() for n, p in self.named_parameters()
                      if "projection_heads" in n and p.requires_grad)
        n_gate = sum(p.numel() for n, p in self.named_parameters() if "gate" in n and p.requires_grad)
        return {"lora_shared": n_lora, "heads": n_heads, "gate": n_gate,
                "total": n_lora + n_heads + n_gate}
