"""Verifiable-reward tasks for the GRPO harness: prompts (token ids + metadata) and a
reward over the decoded completion.

gsm8k: openai/gsm8k (main), chat-templated, reward 1 if the last number in the completion
       equals the answer after '####'.
toy:   the 32 test prompts; reward = min(1, count of the word "the" / 4). Trains in a
       few steps on the 50M local model, so the whole loop runs on a laptop.
"""

from __future__ import annotations

import random
import re

from tests.prompts import G1_PROMPTS

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> str | None:
    m = _NUM.findall(text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


class Task:
    name = ""

    def __init__(self, tok, max_prompt: int = 512) -> None:
        self.tok, self.max_prompt = tok, max_prompt
        self.train: list[tuple[str, str]] = []  # (question, answer)
        self.test: list[tuple[str, str]] = []

    def encode(self, question: str) -> list[int]:
        raise NotImplementedError

    def reward(self, completion: str, answer: str) -> float:
        raise NotImplementedError

    def sample(self, n: int, rng: random.Random) -> list[tuple[list[int], str]]:
        return [(self.encode(q), a) for q, a in rng.sample(self.train, n)]


class GSM8K(Task):
    name = "gsm8k"
    SYSTEM = "Solve the problem step by step. Finish with the final numeric answer on its own line as '#### <number>'."

    def __init__(self, tok, max_prompt: int = 512, n_test: int = 200) -> None:
        super().__init__(tok, max_prompt)
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        split = lambda rows: [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
        self.train, self.test = split(ds["train"]), split(ds["test"])[:n_test]

    def encode(self, question: str) -> list[int]:
        msgs = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": question}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False).input_ids[-self.max_prompt :]  # template already has <|begin_of_text|>

    def reward(self, completion: str, answer: str) -> float:
        got = _last_number(completion)
        try:
            return float(got is not None and abs(float(got) - float(answer)) < 1e-6)
        except ValueError:
            return 0.0


class Toy(Task):
    name = "toy"

    def __init__(self, tok, max_prompt: int = 512) -> None:
        super().__init__(tok, max_prompt)
        self.train = [(p, "the") for p in G1_PROMPTS]
        self.test = self.train[:8]

    def encode(self, question: str) -> list[int]:
        return self.tok(question).input_ids[-self.max_prompt :]

    def reward(self, completion: str, answer: str) -> float:
        return min(1.0, len(re.findall(r"\bthe\b", completion.lower())) / 4)


TASKS = {"gsm8k": GSM8K, "toy": Toy}
