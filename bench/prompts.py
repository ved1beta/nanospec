"""MT-Bench turn-1 questions (80), chat-templated for the target model."""

from __future__ import annotations

import json
from pathlib import Path

MT_BENCH = Path(__file__).with_name("mt_bench.jsonl")


def questions() -> list[str]:
    return [json.loads(l)["turns"][0] for l in MT_BENCH.read_text().splitlines() if l.strip()]


def chat_prompts(tokenizer) -> list[str]:
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in questions()
    ]
