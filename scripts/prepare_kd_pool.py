#!/usr/bin/env python3
"""Build the KD-anchor pool: 50,000 MS MARCO passages from the BEIR corpus.

Streams BeIR/msmarco (8.8M passages) and reservoir-samples with random.Random(42),
writing one {"text": "<title>. <body>"} per line. With the pinned dataset revision
this reproduces the pool used in the paper (md5 c4373a032598ea8c7563129fd005369b).

    python scripts/prepare_kd_pool.py            # -> data/kd_anchor/msmarco_50k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", default="BeIR/msmarco")
    ap.add_argument("--revision", default="a918e0d11a77ed33f42f29d98340b655593b96ad")
    ap.add_argument("--out", default=str(ROOT / "data/kd_anchor/msmarco_50k.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.source, "corpus", split="corpus", streaming=True, revision=args.revision)

    rng = random.Random(args.seed)
    reservoir, n_seen = [], 0
    for r in ds:
        title, body = (r.get("title") or "").strip(), (r.get("text") or "").strip()
        full = f"{title}. {body}" if title and body else (body or title)
        if not full:
            continue
        if len(reservoir) < args.n:
            reservoir.append(full)
        else:
            j = rng.randint(0, n_seen)
            if j < args.n:
                reservoir[j] = full
        n_seen += 1
        if n_seen % 500_000 == 0:
            print(f"  seen {n_seen} passages")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in reservoir:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"sampled {len(reservoir)} of {n_seen} passages -> {out}")


if __name__ == "__main__":
    main()
