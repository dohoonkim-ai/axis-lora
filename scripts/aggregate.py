#!/usr/bin/env python3
"""Main-table R@10 (4-seed mean +- std) and query-level bootstrap CIs.

Reads the per-query files written by scripts/evaluate.py. Each encoder is given
as NAME=PATTERN, where PATTERN contains {seed} and {bench}:

    python scripts/aggregate.py \
        --encoder minilm=runs/minilm/seed{seed}/eval/{bench}.json \
        --encoder bge=runs/bge/seed{seed}/eval/{bench}.json \
        --encoder qwen3=runs/qwen3-0.6b/seed{seed}/eval/{bench}.json

Outputs
  * per benchmark: Base R@10, Axis-LoRA R@10 (mean +- population std over seeds), delta (pp)
  * 4-benchmark mean gain per encoder with a 95% CI: resample queries within each
    benchmark (per-query deltas averaged over seeds), average the 4 benchmark means
  * ATM-Hard gain pooled over all given encoders (per-query deltas averaged over
    encoders and seeds), with a 95% CI over its 31 queries
Bootstrap: 20k resamples, numpy default_rng(0), encoders processed in the given order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BENCHES = ["LME", "ATM_normal", "ATM_hard", "LoCoMo"]


def load(pattern: str, bench: str, seed: str) -> dict:
    return json.loads(Path(pattern.format(bench=bench, seed=seed)).read_text())


def per_query_delta(pattern: str, bench: str, seeds) -> np.ndarray:
    """Per-query (ours - base) R@10, averaged over seeds, ordered by qid."""
    per_seed = [{r["qid"]: r for r in load(pattern, bench, s)["queries"]} for s in seeds]
    qids = sorted(per_seed[0])
    return np.array([np.mean([d[q]["ours"]["R@10"] - d[q]["base"]["R@10"] for d in per_seed])
                     for q in qids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", required=True, help="NAME=PATTERN")
    ap.add_argument("--seeds", default="42,1,7,13")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON output")
    args = ap.parse_args()
    encoders = dict(e.split("=", 1) for e in args.encoder)
    seeds = [s.strip() for s in args.seeds.split(",")]
    res = {"table": {}, "ci": {}}

    # ---- Main table ----
    for enc, pat in encoders.items():
        print(f"\n== {enc} (R@10; Axis-LoRA mean +- std over {len(seeds)} seeds) ==")
        print(f"{'bench':<11}{'Base':>8}{'Axis-LoRA':>18}{'delta(pp)':>11}")
        rows = {}
        for b in BENCHES:
            aggs = [load(pat, b, s)["meta"]["aggregate"] for s in seeds]
            base = float(np.mean([a["base"]["R@10"] for a in aggs]))
            ours = np.array([a["ours"]["R@10"] for a in aggs])
            rows[b] = {"base": base, "ours_mean": float(ours.mean()), "ours_std": float(ours.std()),
                       "ours_per_seed": ours.tolist()}
            print(f"{b:<11}{base:>8.4f}{ours.mean():>11.4f} +-{ours.std():.3f}"
                  f"{100 * (ours.mean() - base):>+11.2f}")
        base_avg = np.mean([rows[b]["base"] for b in BENCHES])
        ours_avg_seed = np.mean([rows[b]["ours_per_seed"] for b in BENCHES], axis=0)
        rows["Avg"] = {"base": float(base_avg), "ours_mean": float(ours_avg_seed.mean()),
                       "ours_std": float(ours_avg_seed.std())}
        print(f"{'Avg':<11}{base_avg:>8.4f}{ours_avg_seed.mean():>11.4f} +-{ours_avg_seed.std():.3f}"
              f"{100 * (ours_avg_seed.mean() - base_avg):>+11.2f}")
        res["table"][enc] = rows

    # ---- Bootstrap CIs ----
    rng = np.random.default_rng(args.rng_seed)
    print(f"\n== 4-benchmark mean gain, 95% CI ({args.n_boot} resamples) ==")
    for enc, pat in encoders.items():
        deltas = {b: per_query_delta(pat, b, seeds) for b in BENCHES}
        point = float(np.mean([d.mean() for d in deltas.values()]))
        boot = np.empty(args.n_boot)
        for i in range(args.n_boot):
            boot[i] = np.mean([deltas[b][rng.integers(0, len(deltas[b]), len(deltas[b]))].mean()
                               for b in BENCHES])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res["ci"][f"4bench_{enc}"] = {"delta_pp": 100 * point, "ci95_pp": [100 * lo, 100 * hi]}
        print(f"{enc:<10} {100 * point:+.2f} pp  [{100 * lo:+.2f}, {100 * hi:+.2f}]")

    d_hard = np.mean([per_query_delta(pat, "ATM_hard", seeds) for pat in encoders.values()], axis=0)
    boot = d_hard[rng.integers(0, len(d_hard), size=(args.n_boot, len(d_hard)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res["ci"]["atm_hard_pooled"] = {"delta_pp": 100 * float(d_hard.mean()), "ci95_pp": [100 * lo, 100 * hi]}
    print(f"\nATM-Hard pooled over {len(encoders)} encoders (n={len(d_hard)}): "
          f"{100 * d_hard.mean():+.2f} pp  [{100 * lo:+.2f}, {100 * hi:+.2f}]")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
