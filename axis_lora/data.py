"""AxisCF training data and the general-domain KD anchor pool."""
from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

AXES = ("TEMPORAL", "SPATIAL", "ENTITY", "EVENT")


def open_text(path: str | Path):
    """Open a .jsonl or .jsonl.gz file for reading."""
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


@dataclass
class AxisCFRow:
    triplet_id: str
    query: str
    positive: str
    cfs_per_axis: Dict[str, List[str]]   # axis -> counterfactual texts
    axis_combination: List[str]          # axes whose slots appear in the memory


def load_axiscf(path: str | Path) -> List[AxisCFRow]:
    rows = []
    with open_text(path) as f:
        for line in f:
            r = json.loads(line)
            cfs = {}
            for ax in AXES:
                lst = (r.get("counterfactual_negatives") or {}).get(ax) or []
                cfs[ax] = [c["content"] for c in lst if isinstance(c, dict) and c.get("content")]
            rows.append(AxisCFRow(
                triplet_id=r["triplet_id"],
                query=r.get("query") or r["positive_memory"]["content"],
                positive=r["positive_memory"]["content"],
                cfs_per_axis=cfs,
                axis_combination=r.get("axis_combination") or [],
            ))
    return rows


class AxisCFDataset(Dataset):
    """One item = query, positive memory, and counterfactuals of up to
    `max_active_axes` of its axes (`cf_per_axis` each)."""

    def __init__(self, rows: List[AxisCFRow], cf_per_axis: int = 2, max_active_axes: int = 2):
        self.rows = rows
        self.cf_per_axis = cf_per_axis
        self.max_active_axes = max_active_axes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        active_axes = [a for a in r.axis_combination if a in AXES]
        random.shuffle(active_axes)
        active_axes = active_axes[: self.max_active_axes]
        cfs = []
        for ax in active_axes:
            ax_cfs = r.cfs_per_axis.get(ax, [])
            random.shuffle(ax_cfs)
            for cf in ax_cfs[: self.cf_per_axis]:
                cfs.append((ax, cf))
        return {
            "query": r.query,
            "positive": r.positive,
            "cfs": cfs,
            "tid": r.triplet_id,
            "axis_combination": active_axes,
        }


def load_kd_pool(path: str | Path) -> List[str]:
    """General-domain passages for the KD anchor (one {"text": ...} per line)."""
    texts = []
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text") if isinstance(r, dict) else r
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    return texts


class JointCollator:
    """Tokenizes a batch of AxisCF items and appends KD-anchor passages.

    Returns a dict with
        tok_q        queries (instruction prefix prepended if given)
        tok_m        memory pool: the B positives first (pos_indices = arange(B)),
                     then all sampled counterfactuals
        axis_targets (B, 4) uniform distribution over each item's active axes
        tok_kd       `kd_per_batch` passages sampled from the KD pool
    """

    def __init__(self, tokenizer, max_len: int = 384, kd_per_batch: int = 0,
                 kd_pool: Optional[List[str]] = None, rng_seed: int = 0,
                 instruction_prefix: Optional[str] = None):
        self.tok = tokenizer
        self.max_len = max_len
        self.kd_per_batch = kd_per_batch
        self.kd_pool = kd_pool or []
        # NOTE: this RNG is copied into every DataLoader worker (and every DDP
        # process) with the same state, and workers are re-created each epoch,
        # so a run cycles through a fixed set of KD passages. Kept as-is to
        # reproduce the paper.
        self.rng = random.Random(rng_seed)
        self.instruction_prefix = instruction_prefix

    def __call__(self, batch):
        B = len(batch)
        instr = self.instruction_prefix or ""
        q_texts = [instr + b["query"] for b in batch]
        m_texts = [b["positive"] for b in batch]
        for b in batch:
            for _, cf in b["cfs"]:
                m_texts.append(cf)

        tok_q = self.tok(q_texts, padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt")
        tok_m = self.tok(m_texts, padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt")

        axis_targets = torch.zeros(B, len(AXES), dtype=torch.float32)
        for i, b in enumerate(batch):
            axes = [a for a in (b.get("axis_combination") or []) if a in AXES]
            for ax in axes:
                axis_targets[i, AXES.index(ax)] = 1.0 / len(axes)

        out = {
            "tok_q": dict(tok_q),
            "tok_m": dict(tok_m),
            "pos_indices": torch.arange(B, dtype=torch.long),
            "axis_targets": axis_targets,
        }
        if self.kd_per_batch > 0 and self.kd_pool:
            kd_texts = self.rng.sample(self.kd_pool, min(self.kd_per_batch, len(self.kd_pool)))
            out["tok_kd"] = dict(self.tok(kd_texts, padding=True, truncation=True,
                                          max_length=self.max_len, return_tensors="pt"))
        return out
