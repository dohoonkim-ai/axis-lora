#!/usr/bin/env python3
"""Convert the original benchmark files into the retrieval format used by evaluate.py.

Inputs (download separately, see README):
    --lme     longmemeval_oracle.json        (LongMemEval)
    --locomo  locomo10.json                  (LoCoMo)
    --atm     atm-bench.json, atm-bench-hard.json  (ATM-Bench QA; directory)

Outputs in --out (default data/benchmarks/):
    lme_sessions.jsonl     {session_id, text}
    lme_queries.jsonl      {qid, qtype, query_text, gold_session_ids}
    locomo_turns.jsonl     {conv_id, dia_id, speaker, session_idx, session_dt, text}
    locomo_queries.jsonl   {conv_id, qa_idx, category, query_text, answer, gold_dia_ids}
    atm_queries.jsonl      {qid, query_text, qtype, split, gold_evidence_ids}

The ATM-Bench memory pool (atm_memory.jsonl.gz) is shipped with this repository,
since it is built from image/video captions that are costly to regenerate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def prepare_lme(src: Path, out: Path):
    """Sessions of all questions' haystacks, deduplicated, as one flat pool."""
    raw = json.loads(src.read_text())
    sessions, queries = {}, []
    for q in raw:
        for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid not in sessions:
                sessions[sid] = "\n".join(f"[{t.get('role', 'user').upper()}] {(t.get('content') or '').strip()}"
                                          for t in turns)
        qdate = q.get("question_date", "")
        queries.append({
            "qid": q["question_id"],
            "qtype": q["question_type"],
            "query_text": f"[{qdate}] {q['question']}" if qdate else q["question"],
            "gold_session_ids": list(q["answer_session_ids"]),
        })
    write_jsonl(out / "lme_sessions.jsonl", [{"session_id": s, "text": t} for s, t in sessions.items()])
    write_jsonl(out / "lme_queries.jsonl", queries)


def locomo_turns(conv: dict):
    idxs = sorted(int(k.split("_")[1]) for k in conv
                  if k.startswith("session_") and not k.endswith("_date_time") and k.split("_")[1].isdigit())
    for idx in idxs:
        dt = conv.get(f"session_{idx}_date_time", "")
        turns = conv.get(f"session_{idx}") or []
        if not isinstance(turns, list):
            continue
        for t in turns:
            if t.get("dia_id"):
                yield t["dia_id"], t.get("speaker", ""), (t.get("text") or "").strip(), idx, dt


def prepare_locomo(src: Path, out: Path):
    """Turn-level pool over all conversations; queries keep only resolvable evidence."""
    convs = json.loads(src.read_text())
    turns, queries, skipped = [], [], 0
    for ci, conv in enumerate(convs):
        cid = conv.get("sample_id") or f"conv_{ci}"
        ct = list(locomo_turns(conv["conversation"]))
        valid = {d for d, *_ in ct}
        for dia, spk, txt, sidx, dt in ct:
            turns.append({"conv_id": cid, "dia_id": dia, "speaker": spk, "session_idx": sidx,
                          "session_dt": dt, "text": f"[{dt}] {spk}: {txt}" if dt else f"{spk}: {txt}"})
        for qa_idx, qa in enumerate(conv.get("qa", [])):
            gold = [e for e in qa.get("evidence", []) if e in valid]
            if not gold:
                skipped += 1
                continue
            answer = qa.get("answer")
            queries.append({"conv_id": cid, "qa_idx": qa_idx, "category": qa.get("category"),
                            "query_text": qa.get("question") or "",
                            "answer": "" if answer is None else str(answer),
                            "gold_dia_ids": gold})
    write_jsonl(out / "locomo_turns.jsonl", turns)
    write_jsonl(out / "locomo_queries.jsonl", queries)
    print(f"  (skipped {skipped} LoCoMo questions without resolvable evidence)")


def prepare_atm(src_dir: Path, out: Path):
    rows = []
    for split, fname in (("normal", "atm-bench.json"), ("hard", "atm-bench-hard.json")):
        qs = json.loads((src_dir / fname).read_text())
        if isinstance(qs, dict):
            qs = qs["qas"]
        for i, q in enumerate(qs):
            rows.append({"qid": f"{split}_{i}", "query_text": q["question"], "qtype": q.get("qtype", "?"),
                         "split": split, "gold_evidence_ids": q.get("evidence_ids", [])})
    write_jsonl(out / "atm_queries.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lme", type=Path, help="path to longmemeval_oracle.json")
    ap.add_argument("--locomo", type=Path, help="path to locomo10.json")
    ap.add_argument("--atm", type=Path, help="directory with atm-bench.json and atm-bench-hard.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/benchmarks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.lme:
        prepare_lme(args.lme, args.out)
    if args.locomo:
        prepare_locomo(args.locomo, args.out)
    if args.atm:
        prepare_atm(args.atm, args.out)


if __name__ == "__main__":
    main()
