#!/usr/bin/env python3
"""Train/validation split of AxisCF, and the single-axis training subsets.

Split: stratified 90/10 over (memory format x axis combination), random.Random(42);
reproduces data/axiscf/{train,val}.jsonl.gz (4,196 / 468) from the full corpus.

    python generation/split_axiscf.py split --src triplets.jsonl --out data/axiscf

Single-axis subsets (appendix baselines): training triplets that have at least one
counterfactual for the axis, keeping only that axis's counterfactuals.

    python generation/split_axiscf.py single-axis --train data/axiscf/train.jsonl.gz --out data/axiscf
    # -> train_{TEMPORAL,SPATIAL,ENTITY,EVENT}_only.jsonl (3,220 / 2,560 / 2,540 / 3,041 rows)
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axis_lora.data import AXES, open_text


def write_jsonl(path: Path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def split(args):
    with open_text(args.src) as f:
        rows = [json.loads(line) for line in f]
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r.get("slice", "unknown"), tuple(sorted(r.get("axis_combination") or [])))].append(i)
    rng = random.Random(args.seed)
    val_idx, train_idx = set(), set()
    for idxs in strata.values():
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * args.val_frac)) if len(idxs) >= 2 else 0
        val_idx.update(idxs[:n_val])
        train_idx.update(idxs[n_val:])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"train.jsonl{args.ext}", [rows[i] for i in sorted(train_idx)])
    write_jsonl(out / f"val.jsonl{args.ext}", [rows[i] for i in sorted(val_idx)])
    print(f"{len(strata)} strata; val per format: {dict(Counter(rows[i]['slice'] for i in val_idx))}")


def single_axis(args):
    with open_text(args.train) as f:
        rows = [json.loads(line) for line in f]
    out = Path(args.out)
    for ax in AXES:
        sub = []
        for r in rows:
            cfs = (r.get("counterfactual_negatives") or {}).get(ax)
            if ax in (r.get("axis_combination") or []) and cfs:
                r = dict(r)
                r["axis_combination"] = [ax]
                r["counterfactual_negatives"] = {ax: cfs}
                sub.append(r)
        write_jsonl(out / f"train_{ax}_only.jsonl{args.ext}", sub)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("split")
    a.add_argument("--src", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--val-frac", type=float, default=0.10)
    a.add_argument("--ext", default="", help='"" for .jsonl, ".gz" for .jsonl.gz')
    b = sub.add_parser("single-axis")
    b.add_argument("--train", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--ext", default="")
    args = ap.parse_args()
    split(args) if args.cmd == "split" else single_axis(args)


if __name__ == "__main__":
    main()
