#!/usr/bin/env python3
"""Evaluate an Axis-LoRA checkpoint on LongMemEval, ATM-Bench (Normal/Hard), and LoCoMo.

For every query it scores the memory pool twice:
    base  s(q, m) = <z_b(q), z_b(m)>                                   (frozen model)
    ours  s(q, m) = <z_b(q), z_b(m)> + alpha * sum_A w_A(q) <phi_A(q), phi_A(m)>
and writes one per-query file per benchmark ({bench}.json) with first-gold rank,
MRR, R@{1,5,10,100} (fraction of gold memories in the top k), per-gold ranks,
top-10 ids and gate weights, plus summary.json. These files are what
scripts/aggregate.py turns into the main table and confidence intervals.

Benchmark files are produced by scripts/prepare_benchmarks.py. Example:

    python scripts/evaluate.py --ckpt runs/qwen3-0.6b/seed42/final \
        --instruction-prefix $'Instruct: Given a personal-memory question, retrieve relevant memory entries.\nQuery: ' \
        --batch-encode 16 --batch-encode-m 4 --max-len-q 2048 --max-len-m 4096 \
        --out-dir runs/qwen3-0.6b/seed42/eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from axis_lora.checkpoint import load_checkpoint
from axis_lora.data import open_text
from axis_lora.model import AXES

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

BENCHES = ("LME", "ATM_normal", "ATM_hard", "LoCoMo")
KS = (1, 5, 10, 100)


# ---------------- Benchmarks ----------------

def read_jsonl(path: Path):
    with open_text(path) as f:
        return [json.loads(line) for line in f]


def resolve(data_dir: Path, name: str) -> Path:
    for p in (data_dir / name, data_dir / (name + ".gz")):
        if p.exists():
            return p
    raise FileNotFoundError(f"{data_dir / name} not found; run scripts/prepare_benchmarks.py first")


def load_bench(name: str, data_dir: Path, locomo_scoped_ids: bool = False):
    """-> (queries [(qid, text)], pool [(id, text)], qrels {qid: set(ids)})"""
    if name == "LME":
        qs = read_jsonl(resolve(data_dir, "lme_queries.jsonl"))
        pool = [(str(r["session_id"]), r["text"]) for r in read_jsonl(resolve(data_dir, "lme_sessions.jsonl"))]
        return ([(r["qid"], r["query_text"]) for r in qs], pool,
                {r["qid"]: set(r["gold_session_ids"]) for r in qs})
    if name == "LoCoMo":
        # The paper identifies turns by the benchmark's dia_id (e.g. "D1:3"), which
        # repeats across the 10 conversations, so a same-id turn of another
        # conversation also counts as a hit. --locomo-scoped-ids uses conv_id::dia_id.
        scope = (lambda r, d: f"{r['conv_id']}::{d}") if locomo_scoped_ids else (lambda r, d: str(d))
        qs = read_jsonl(resolve(data_dir, "locomo_queries.jsonl"))
        pool = [(scope(r, r["dia_id"]), r["text"]) for r in read_jsonl(resolve(data_dir, "locomo_turns.jsonl"))]
        qid = lambda r: f"{r['conv_id']}_{r['qa_idx']}"
        return ([(qid(r), r["query_text"]) for r in qs], pool,
                {qid(r): {scope(r, g) for g in r["gold_dia_ids"]} for r in qs})
    split = {"ATM_normal": "normal", "ATM_hard": "hard"}[name]
    qs = [r for r in read_jsonl(resolve(data_dir, "atm_queries.jsonl")) if r["split"] == split]
    pool = [(str(r["evidence_id"]), r["text"]) for r in read_jsonl(resolve(data_dir, "atm_memory.jsonl"))]
    return ([(r["qid"], r["query_text"]) for r in qs], pool,
            {r["qid"]: set(r["gold_evidence_ids"]) for r in qs})


# ---------------- Encoding and scoring ----------------

@torch.no_grad()
def encode_texts_all_axes(model, texts: List[str], batch_size: int = 32, max_len: int = 512,
                          device: str = "cuda") -> Dict[str, np.ndarray]:
    """-> {"base": (N, d), "TEMPORAL": (N, d), ...}, L2-normalized."""
    tok = model.tokenizer
    out = {k: np.empty((len(texts), model.hidden_size), dtype=np.float32) for k in ["base", *AXES]}
    t0 = time.time()
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        tk = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        emb = model.encode_all(input_ids=tk["input_ids"], attention_mask=tk["attention_mask"])
        for k in out:
            out[k][s:s + len(batch)] = emb[k].cpu().numpy()
        if (s // batch_size) % 20 == 0:
            log.info(f"  encoded {s + len(batch)}/{len(texts)}  ({time.time() - t0:.0f}s)")
    return out


def compute_score_matrix(q_emb, m_emb, gate_w: np.ndarray, alpha: float, chunk: int = 128,
                         device: str = "cuda") -> np.ndarray:
    """Axis-LoRA scores (n_q, n_m), computed on the GPU in query chunks."""
    n_q, n_m = q_emb["base"].shape[0], m_emb["base"].shape[0]
    scores = np.empty((n_q, n_m), dtype=np.float32)
    q_base = torch.from_numpy(q_emb["base"]).to(device)
    m_base = torch.from_numpy(m_emb["base"]).to(device)
    q_axis = {ax: torch.from_numpy(q_emb[ax]).to(device) for ax in AXES}
    m_axis = {ax: torch.from_numpy(m_emb[ax]).to(device) for ax in AXES}
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        base_sim = q_base[s:e] @ m_base.T
        axis_sim = torch.zeros_like(base_sim)
        w = torch.from_numpy(gate_w[s:e]).to(device)
        for ai, ax in enumerate(AXES):
            axis_sim = axis_sim + w[:, ai:ai + 1] * (q_axis[ax][s:e] @ m_axis[ax].T)
        scores[s:e] = (base_sim + alpha * axis_sim).cpu().numpy()
    return scores


def query_record(scores_row: np.ndarray, gold: set, ids: List[str]) -> dict:
    order = np.argsort(-scores_row)
    ranked = [ids[i] for i in order]
    id_rank = {mid: k + 1 for k, mid in enumerate(ranked)}
    first = next((k + 1 for k, mid in enumerate(ranked) if mid in gold), len(ids) + 1)
    rec = {"rank": first, "rr": 1.0 / first,
           "gold_ranks": {g: id_rank.get(g, len(ids) + 1) for g in sorted(gold)},
           "top10": ranked[:10]}
    for k in KS:
        rec[f"R@{k}"] = len(gold & set(ranked[:k])) / len(gold)
    return rec


def aggregate(recs: List[dict]) -> dict:
    out = {f"R@{k}": float(np.mean([r[f"R@{k}"] for r in recs])) for k in KS}
    out["rr"] = float(np.mean([r["rr"] for r in recs]))
    out["median_rank"] = float(np.median([r["rank"] for r in recs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (peft/, gate.pt, config.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data/benchmarks"))
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--alpha-sweep", default=None,
                    help="extra comma-separated alphas, summary only (e.g. 0.1,0.5,1.0)")
    ap.add_argument("--instruction-prefix", default=None, help="query prefix (match training)")
    ap.add_argument("--gate-mode-override", default=None, choices=["sparsemax", "softmax", "uniform"],
                    help="inference-time gate mode (e.g. uniform routing on a sparsemax-trained gate)")
    ap.add_argument("--batch-encode", type=int, default=32)
    ap.add_argument("--batch-encode-m", type=int, default=None)
    ap.add_argument("--max-len-q", type=int, default=256)
    ap.add_argument("--max-len-m", type=int, default=4096)
    ap.add_argument("--max-len-m-locomo", type=int, default=256,
                    help="memory max length for LoCoMo (single dialogue turns)")
    ap.add_argument("--locomo-scoped-ids", action="store_true",
                    help="identify LoCoMo turns by conv_id::dia_id (the paper uses bare dia_ids)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt).to(args.device).eval()
    gate = model.gate
    if args.gate_mode_override:
        log.info(f"gate mode at inference: {gate.mode} -> {args.gate_mode_override}")
        gate.mode = args.gate_mode_override
    sweep = [float(a) for a in args.alpha_sweep.split(",")] if args.alpha_sweep else []
    prefix = args.instruction_prefix or ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in [b.strip() for b in args.benches.split(",") if b.strip()]:
        queries, pool, qrels = load_bench(name, Path(args.data_dir), args.locomo_scoped_ids)
        log.info(f"== {name}: {len(queries)} queries x {len(pool)} memories ==")
        q_ids, q_raw = [q[0] for q in queries], [q[1] for q in queries]
        m_ids, m_texts = [m[0] for m in pool], [m[1] for m in pool]
        q_emb = encode_texts_all_axes(model, [prefix + t for t in q_raw], args.batch_encode,
                                      args.max_len_q, args.device)
        max_len_m = args.max_len_m_locomo if name == "LoCoMo" else args.max_len_m
        m_emb = encode_texts_all_axes(model, m_texts, args.batch_encode_m or args.batch_encode,
                                      max_len_m, args.device)
        with torch.no_grad():
            gate_w = gate(torch.from_numpy(q_emb["base"]).to(args.device)).cpu().numpy()

        base_scores = q_emb["base"] @ m_emb["base"].T     # frozen base (fp32, CPU)
        ours_scores = compute_score_matrix(q_emb, m_emb, gate_w, args.alpha, device=args.device)

        records = []
        for i, qid in enumerate(q_ids):
            gold = qrels.get(qid, set())
            if not gold:
                continue
            records.append({
                "qid": qid, "query_text": q_raw[i], "n_gold": len(gold),
                "gate_weights": {ax: float(gate_w[i, ai]) for ai, ax in enumerate(AXES)},
                "base": query_record(base_scores[i], gold, m_ids),
                "ours": query_record(ours_scores[i], gold, m_ids),
            })
        agg = {side: aggregate([r[side] for r in records]) for side in ("base", "ours")}
        for a in sweep:
            sc = compute_score_matrix(q_emb, m_emb, gate_w, a, device=args.device)
            agg[f"alpha={a}"] = aggregate([query_record(sc[i], qrels[qid], m_ids)
                                           for i, qid in enumerate(q_ids) if qrels.get(qid)])
        log.info(f"  base R@10={agg['base']['R@10']:.4f}  ours R@10={agg['ours']['R@10']:.4f}")
        summary[name] = agg
        (out_dir / f"{name}.json").write_text(json.dumps({
            "meta": {"ckpt": str(args.ckpt), "bench": name, "alpha": args.alpha,
                     "n_queries": len(records), "aggregate": agg},
            "queries": records,
        }, indent=1, ensure_ascii=False))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
