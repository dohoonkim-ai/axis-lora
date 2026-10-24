"""Saving and loading Axis-LoRA checkpoints.

A checkpoint directory contains
    peft/{TEMPORAL,SPATIAL,ENTITY,EVENT}/adapter_model.safetensors   (Axis-LoRA)
    peft/adapter_model.safetensors                                    (shared trunk)
    gate.pt
    projection_heads.pt                                               (shared trunk)
and the training config.json either in the directory itself or in its parent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .model import AXES, AxisLoRAConfig, AxisLoRAModel
from .shared_trunk import SharedTrunkModel

log = logging.getLogger(__name__)


def _load_state_file(d: Path):
    files = list(d.glob("adapter_model.*"))
    if not files:
        return None
    if files[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(files[0])
    return torch.load(files[0], map_location="cpu")


def find_train_config(ckpt: Path) -> dict:
    for p in (ckpt / "config.json", ckpt.parent / "config.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no config.json in {ckpt} or its parent")


def save_checkpoint(model, out: Path, train_config: dict):
    out.mkdir(parents=True, exist_ok=True)
    model.peft_model.save_pretrained(str(out / "peft"))
    torch.save(model.gate.state_dict(), out / "gate.pt")
    if isinstance(model, SharedTrunkModel):
        torch.save(model.projection_heads.state_dict(), out / "projection_heads.pt")
    (out / "config.json").write_text(json.dumps(train_config, indent=2))


def load_checkpoint(ckpt: str | Path, **overrides):
    """Rebuild the model from a checkpoint directory and load its weights."""
    ckpt = Path(ckpt)
    saved = find_train_config(ckpt)
    cfg = AxisLoRAConfig.from_train_args({**saved, **overrides})
    variant = saved.get("variant", "axis_lora")

    if variant in ("shared_trunk", "form_b"):
        model = SharedTrunkModel(cfg)
        sd = _load_state_file(ckpt / "peft")
        target = model.peft_model.state_dict()
        n = 0
        for k, v in (sd or {}).items():
            tk = (k.replace(".lora_A.weight", ".lora_A.default.weight")
                   .replace(".lora_B.weight", ".lora_B.default.weight"))
            key = tk if tk in target else (k if k in target else None)
            if key is not None:
                target[key].copy_(v.to(target[key].dtype))
                n += 1
        log.info(f"shared LoRA: loaded {n} tensors")
        heads = ckpt / "projection_heads.pt"
        if heads.exists():
            model.projection_heads.load_state_dict(torch.load(heads, map_location="cpu"))
    else:
        model = AxisLoRAModel(cfg)
        # Collect all four adapters into one state dict and load it once
        # (per-adapter in-place copies are unreliable with PEFT multi-adapter models).
        combined = {}
        for ax in AXES:
            sd = _load_state_file(ckpt / "peft" / ax)
            if sd is None:
                raise FileNotFoundError(f"missing adapter weights for {ax} in {ckpt / 'peft'}")
            for k, v in sd.items():
                if ".lora_A.weight" in k:
                    combined[k.replace(".lora_A.weight", f".lora_A.{ax}.weight")] = v
                elif ".lora_B.weight" in k:
                    combined[k.replace(".lora_B.weight", f".lora_B.{ax}.weight")] = v
        _, unexpected = model.peft_model.load_state_dict(combined, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected adapter keys: {unexpected[:5]}")
        log.info(f"loaded {len(combined)} adapter tensors from {ckpt / 'peft'}")

    model.gate.load_state_dict(torch.load(ckpt / "gate.pt", map_location="cpu"))
    return model
