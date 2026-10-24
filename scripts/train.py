#!/usr/bin/env python3
"""Train Axis-LoRA on AxisCF (composite InfoNCE + KD anchor).

Paper settings for each encoder are in configs/*.sh. Example (4 GPUs):

    accelerate launch --num_processes 4 scripts/train.py \
        --base-model Qwen/Qwen3-Embedding-0.6B --pooling last --max-len 1536 \
        --batch-size 2 --grad-accum 2 --epochs 5 --cf-per-axis 1 --kd-per-batch 2 \
        --instruction-prefix $'Instruct: Given a personal-memory question, retrieve relevant memory entries.\nQuery: ' \
        --out-dir runs/qwen3-0.6b/seed42 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axis_lora.checkpoint import save_checkpoint
from axis_lora.data import AxisCFDataset, JointCollator, load_axiscf, load_kd_pool
from axis_lora.losses import AXES, total_loss
from axis_lora.model import AxisLoRAConfig, AxisLoRAModel
from axis_lora.shared_trunk import SharedTrunkModel

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def build_args():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train-file", default=str(ROOT / "data/axiscf/train.jsonl.gz"))
    ap.add_argument("--kd-pool", default=str(ROOT / "data/kd_anchor/msmarco_50k.jsonl"),
                    help="General-domain passages for the KD anchor (see scripts/prepare_kd_pool.py).")
    ap.add_argument("--kd-per-batch", type=int, default=2, help="KD passages per batch (per device).")
    ap.add_argument("--cf-per-axis", type=int, default=2)
    ap.add_argument("--max-active-axes", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    # Model
    ap.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--padding-side", default="right", choices=["left", "right"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--last-pct", type=float, default=0.2)
    ap.add_argument("--variant", default="axis_lora", choices=["axis_lora", "shared_trunk"],
                    help="axis_lora: four axis-conditional LoRA branches (paper). "
                         "shared_trunk: one shared LoRA + four projection heads (ablation).")
    ap.add_argument("--instruction-prefix", default=None,
                    help="Query instruction prefix (none for MiniLM).")
    # Gate
    ap.add_argument("--gate-mode", default="sparsemax", choices=["sparsemax", "softmax", "uniform"])
    ap.add_argument("--gate-init-tau", type=float, default=2.0)
    ap.add_argument("--gate-tau-final", type=float, default=0.5)
    ap.add_argument("--lambda-gate-axis", type=float, default=0.0,
                    help="Weight of the gate axis-supervision loss (0 = off; appendix gate variants).")
    ap.add_argument("--gate-axis-loss", default="ce", choices=["ce", "mse"])
    ap.add_argument("--route-by-label", action="store_true",
                    help="Hard routing by the per-sample axis labels during training (appendix).")
    ap.add_argument("--route-fixed-axis", default=None, choices=list(AXES),
                    help="Single-adapter control: route every sample to one branch (appendix).")
    # Optimization
    ap.add_argument("--batch-size", type=int, default=8, help="per device")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-gate", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.25, help="mixing weight of the gated axis term")
    ap.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature")
    ap.add_argument("--lambda-kd", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    # Output
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0, help="intermediate checkpoints (0 = off)")
    ap.add_argument("--max-steps", type=int, default=None)
    return ap.parse_args()


def setup_optimizer(model, lr, lr_gate, weight_decay):
    """Two parameter groups: LoRA (and heads) at `lr`, gate at `lr_gate`."""
    lora_params, gate_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (gate_params if ("gate" in n and "lora_" not in n) else lora_params).append(p)
    return torch.optim.AdamW([
        {"params": lora_params, "lr": lr},
        {"params": gate_params, "lr": lr_gate},
    ], weight_decay=weight_decay)


def annealed_tau(step, total_steps, tau_init, tau_final):
    if total_steps <= 0:
        return tau_final
    return tau_init + (tau_final - tau_init) * min(1.0, step / total_steps)


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from accelerate import Accelerator, DistributedDataParallelKwargs
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "no",
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    is_main = accelerator.is_main_process
    if is_main:
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        log.info(f"output: {out_dir}  world_size: {accelerator.num_processes}")

    # ----- Data -----
    rows = load_axiscf(args.train_file)
    # The KD batch is encoded whenever a pool is given, also with --lambda-kd 0
    # (as in the paper's no-KD ablation); pass --kd-pool "" to skip it entirely.
    kd_pool = load_kd_pool(args.kd_pool) if args.kd_pool else None
    if is_main:
        log.info(f"AxisCF rows: {len(rows)}  KD pool: {len(kd_pool) if kd_pool else 0}")

    # ----- Model -----
    cfg = AxisLoRAConfig(
        base_model=args.base_model, pooling=args.pooling, rank=args.rank,
        lora_dropout=args.lora_dropout, last_pct=args.last_pct,
        gate_init_tau=args.gate_init_tau, gate_mode=args.gate_mode, bf16=args.bf16,
        trust_remote_code=args.trust_remote_code, padding_side=args.padding_side,
    )
    model = SharedTrunkModel(cfg) if args.variant == "shared_trunk" else AxisLoRAModel(cfg)
    if is_main:
        log.info(f"LoRA layers: {model.lora_layers}  trainable: {model.trainable_param_summary()}")

    train_ds = AxisCFDataset(rows, cf_per_axis=args.cf_per_axis, max_active_axes=args.max_active_axes)
    collator = JointCollator(model.tokenizer, max_len=args.max_len,
                             kd_per_batch=args.kd_per_batch if kd_pool else 0,
                             kd_pool=kd_pool, rng_seed=args.seed,
                             instruction_prefix=args.instruction_prefix)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, drop_last=True)

    optimizer = setup_optimizer(model, args.lr, args.lr_gate, args.weight_decay)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    unwrapped = accelerator.unwrap_model(model)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(s):   # linear warmup, then linear decay to 0
        if s < warmup_steps:
            return s / warmup_steps
        return max(0.0, 1.0 - (s - warmup_steps) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Train -----
    step, history, t_start = 0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        for it, batch in enumerate(train_loader):
            tau_now = annealed_tau(step, total_steps, args.gate_init_tau, args.gate_tau_final)
            with torch.no_grad():
                unwrapped.gate.tau.fill_(tau_now)

            dev = accelerator.device
            tok_q = {k: v.to(dev) for k, v in batch["tok_q"].items()}
            tok_m = {k: v.to(dev) for k, v in batch["tok_m"].items()}
            pos_idx = batch["pos_indices"].to(dev)
            q_emb = unwrapped.encode_all(**tok_q)
            m_emb = unwrapped.encode_all(**tok_m)

            anchor_emb, kd_trunk_loss = None, None
            if kd_pool and "tok_kd" in batch:
                tok_kd = {k: v.to(dev) for k, v in batch["tok_kd"].items()}
                if args.variant == "shared_trunk":
                    p_lora, p_frozen = unwrapped.encode_for_kd(**tok_kd)
                    kd_trunk_loss = ((p_lora - p_frozen) ** 2).sum(-1).mean()
                else:
                    anchor_emb = unwrapped.encode_all(**tok_kd)

            w_override = None
            if args.route_fixed_axis is not None:
                w_override = q_emb["base"].new_zeros(q_emb["base"].shape[0], len(AXES))
                w_override[:, AXES.index(args.route_fixed_axis)] = 1.0
            elif args.route_by_label:
                w_override = batch["axis_targets"].to(dev).float()
                blank = w_override.sum(dim=-1) <= 0
                if blank.any():
                    w_override[blank] = 0.25

            L, parts = total_loss(q_emb, m_emb, pos_idx, anchor_emb, unwrapped.gate,
                                  args.alpha, args.tau, args.lambda_kd, w_override=w_override)

            if args.lambda_gate_axis > 0:
                axis_targets = batch["axis_targets"].to(dev)
                active = axis_targets.sum(dim=-1) > 0
                if active.any():
                    gate_w = unwrapped.gate(q_emb["base"].float())
                    if args.gate_axis_loss == "ce":
                        L_gate = -(axis_targets[active] *
                                   torch.log(gate_w[active].clamp_min(1e-8))).sum(dim=-1).mean()
                    else:
                        L_gate = F.mse_loss(gate_w[active], axis_targets[active])
                    L = L + args.lambda_gate_axis * L_gate
                    parts["gate_axis"] = float(L_gate.detach())
                    parts["total"] = float(L.detach())
            if kd_trunk_loss is not None:
                L = L + args.lambda_kd * kd_trunk_loss
                parts["kd"] = float(kd_trunk_loss.detach())
                parts["total"] = float(L.detach())

            accelerator.backward(L / args.grad_accum)
            if (it + 1) % args.grad_accum == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if is_main and step % args.log_every == 0:
                    log.info(f"[{100.0 * step / total_steps:5.1f}%] ep{epoch} step{step}/{total_steps}  "
                             f"L={parts['total']:.4f}  L_c={parts.get('contrastive', 0):.4f}  "
                             f"L_kd={parts.get('kd', 0):.4f}  tau_g={tau_now:.2f}  "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}")
                    history.append({"step": step, "epoch": epoch, **parts,
                                    "tau_gate": tau_now, "lr": optimizer.param_groups[0]["lr"]})
                if is_main and args.save_every and step % args.save_every == 0:
                    save_checkpoint(unwrapped, out_dir / f"ckpt-{step}", vars(args))
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if is_main:
        save_checkpoint(unwrapped, out_dir / "final", vars(args))
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        log.info(f"final -> {out_dir / 'final'}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
