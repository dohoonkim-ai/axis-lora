#!/usr/bin/env python3
"""Generate the AxisCF corpus with an LLM behind an OpenAI-compatible chat endpoint.

The paper used Qwen3.5-27B served with vLLM (non-thinking mode, temperature 0.7):

    vllm serve Qwen/Qwen3.5-27B --port 8000
    python generation/generate_axiscf.py --endpoint http://localhost:8000/v1/chat/completions \
        --n 1250 --k 3 --out data/axiscf_raw

Step 1 writes positives.jsonl: --n memories for each of four formats
    short_turn           one sentence, one primary axis (TEMPORAL 60% / EVENT 40%)
    multi_turn_dialogue  8-10 turns covering 2-3 axes
    multi_field_passage  email / image log / video transcript covering all 4 axes
    hard_composition     long passage, dialogue or email thread spanning 3-4 axes
Step 2 asks for --k counterfactuals per axis for every memory (prompts/cf_generation.txt)
and writes triplets.jsonl. Failed generations are dropped (the released corpus has
4,664 triplets from 4 x 1,250 requests). The query field is the first 200
characters of the memory. LLM sampling is not deterministic, so a new run gives a
different corpus; the corpus used in the paper is in data/axiscf/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Optional

import httpx

PROMPTS = Path(__file__).parent / "prompts"
AXES = ("TEMPORAL", "SPATIAL", "ENTITY", "EVENT")
SYSTEM_PROMPT = ("You are a corpus generation assistant. Output only the requested JSON. "
                 "No reasoning, no explanation, no markdown fences.")

SPEAKERS = ["Sarah", "Marcus", "Mom", "Dad", "Lisa", "Daniel", "Julia", "Andrew",
            "Mei", "Hyun", "Jiwon", "Aunt Kyungmi", "Tom", "Olivia",
            "Carlos", "Priya", "Ahmed", "Yuki", "Kevin", "Emma"]
FORMATS_C = ["email", "image_log", "video_transcript"]


def random_date():
    y = random.choice([2022, 2023, 2024])
    return (f"{y:04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} "
            f"{random.randint(7, 22):02d}:{random.choice([0, 15, 30, 45]):02d}")


def extract_json(text: str) -> Optional[dict]:
    """Parse the reply, falling back to the largest balanced {...} block."""
    try:
        return json.loads(text)
    except Exception:
        pass
    depth, start, best = 0, -1, None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                cand = text[start:i + 1]
                try:
                    parsed = json.loads(cand)
                    if best is None or len(cand) > len(best[0]):
                        best = (cand, parsed)
                except Exception:
                    pass
                start = -1
    return best[1] if best else None


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


class ChatClient:
    def __init__(self, endpoint: str, model: str, concurrency: int = 64,
                 temperature: float = 0.7, api_key: Optional[str] = None):
        self.endpoint, self.model, self.temperature = endpoint, model, temperature
        self.concurrency, self.api_key = concurrency, api_key
        self.sem = asyncio.Semaphore(concurrency)
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self.client = httpx.AsyncClient(
            timeout=600.0, headers=headers,
            limits=httpx.Limits(max_connections=self.concurrency + 16,
                                max_keepalive_connections=self.concurrency))
        return self

    async def __aexit__(self, *_):
        await self.client.aclose()

    async def chat(self, user_prompt: str, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.client.post(self.endpoint, json=payload, follow_redirects=True)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    if (content or "").strip():
                        return content
                except Exception:
                    pass
                await asyncio.sleep(2 * (attempt + 1))
            return ""


# ---------------- Positive memories ----------------

async def gen_short_turn(client, n):
    template = load_prompt("short_turn")
    axes = ["TEMPORAL"] * int(n * 0.6) + ["EVENT"] * (n - int(n * 0.6))
    random.shuffle(axes)

    async def one(idx, axis):
        d = extract_json(await client.chat(template.replace("{primary_axis}", axis), max_tokens=400))
        if not d or "content" not in d:
            return None
        ax = d.get("primary_axis", axis)
        return {"triplet_id": f"v6_a_{idx:04d}", "slice": "short_turn", "primary_axis": ax,
                "axis_combination": [ax], "positive_memory": {"content": d["content"]}}
    return [r for r in await asyncio.gather(*[one(i, a) for i, a in enumerate(axes)]) if r]


async def gen_dialogue(client, n):
    template = load_prompt("multi_turn_dialogue")

    async def one(idx):
        axis_combo = random.sample(list(AXES), random.randint(2, 3))
        date = random_date().split()[0]
        speakers = random.sample(SPEAKERS, random.choice([2, 3]))
        prompt = (template.replace("{axis_combination}", json.dumps(axis_combo))
                  .replace("{session_date}", date).replace("{speakers}", json.dumps(speakers)))
        d = extract_json(await client.chat(prompt, max_tokens=1200))
        if not d or "turns" not in d:
            return None
        rendered = "\n".join(f"{t.get('role')}: {t.get('content', '')}" for t in d["turns"])
        return {"triplet_id": f"v6_b_{idx:04d}", "slice": "multi_turn_dialogue",
                "axis_combination": d.get("axis_combination", axis_combo),
                "session_date": d.get("session_date", date), "speakers": d.get("speakers", speakers),
                "positive_memory": {"content": rendered, "turns": d["turns"]}}
    return [r for r in await asyncio.gather(*[one(i) for i in range(n)]) if r]


async def gen_passage(client, n):
    template = load_prompt("multi_field_passage")

    async def one(idx):
        fmt = FORMATS_C[idx % 3]
        d = extract_json(await client.chat(template.replace("{format}", fmt), max_tokens=8000))
        if not d or "content" not in d:
            return None
        return {"triplet_id": f"v6_c_{idx:04d}", "slice": "multi_field_passage",
                "format": d.get("format", fmt), "axis_combination": d.get("axis_combination", list(AXES)),
                "fields": d.get("fields", {}), "positive_memory": {"content": d["content"]}}
    return [r for r in await asyncio.gather(*[one(i) for i in range(n)]) if r]


async def gen_hard(client, n):
    template = load_prompt("hard_composition")

    async def one(idx):
        axis_combo = list(AXES)
        query_axes = random.sample(axis_combo, random.choice([3, 4]))
        fmt = random.choice(["passage", "dialogue", "email_thread"])
        prompt = (template.replace("{axis_combination}", json.dumps(axis_combo))
                  .replace("{query_axes}", json.dumps(query_axes)).replace("{format}", fmt))
        d = extract_json(await client.chat(prompt, max_tokens=1800))
        if not d or "content" not in d:
            return None
        return {"triplet_id": f"v6_d_{idx:04d}", "slice": "hard_composition",
                "axis_combination": d.get("axis_combination", axis_combo),
                "query_axes": d.get("query_axes", query_axes), "format": d.get("format", fmt),
                "positive_memory": {"content": d["content"]}}
    return [r for r in await asyncio.gather(*[one(i) for i in range(n)]) if r]


# ---------------- Counterfactuals ----------------

async def gen_cfs(client, trip: dict, k: int = 3) -> dict:
    """K counterfactuals for each of the four axes (one request per axis)."""
    template = load_prompt("cf_generation")
    pos = trip["positive_memory"]["content"]

    async def one_axis(axis):
        prompt = template.replace("{positive_text}", pos).replace("{axis}", axis).replace("{K}", str(k))
        max_tokens = min(4000, max(800, int(len(pos.split()) * 1.5 * (k + 1))))
        d = extract_json(await client.chat(prompt, max_tokens=max_tokens))
        if not d or "cf_texts" not in d:
            return axis, []
        return axis, [{"content": t} for t in d["cf_texts"][:k]]
    return dict(await asyncio.gather(*[one_axis(a) for a in AXES]))


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else None
    async with ChatClient(args.endpoint, args.model, args.concurrency, api_key=api_key) as client:
        t0 = time.time()
        parts = await asyncio.gather(gen_short_turn(client, args.n), gen_dialogue(client, args.n),
                                     gen_passage(client, args.n), gen_hard(client, args.n))
        trips = [t for p in parts for t in p]
        print(f"positives: {[len(p) for p in parts]} ({time.time() - t0:.0f}s)")
        with (out / "positives.jsonl").open("w") as f:
            for t in trips:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        t0 = time.time()
        cfs = await asyncio.gather(*[gen_cfs(client, t, k=args.k) for t in trips])
        for trip, c in zip(trips, cfs):
            trip["counterfactual_negatives"] = c
            trip["query"] = trip["positive_memory"]["content"][:200]
            trip["sampling_weight"] = 1.0
        with (out / "triplets.jsonl").open("w") as f:
            for t in trips:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"counterfactuals done ({time.time() - t0:.0f}s): "
              f"{ {a: sum(len(t['counterfactual_negatives'][a]) for t in trips) for a in AXES} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--n", type=int, default=1250, help="memories per format")
    ap.add_argument("--k", type=int, default=3, help="counterfactuals per axis")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
