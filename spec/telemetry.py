"""Acceptance telemetry: one JSON line per verify step per request.

{"step", "req", "pos" (output tokens before this step), "depth", "topk", "accepted",
 "nodes": [{"tok", "parent", "p", "q", "u", "d"}...]   # p: target prob at the parent row,
 "final": "bonus" | "resample", "ms": {phase: ms}}     # q: draft log-prob, d: accept|reject|-
"""

from __future__ import annotations

import json


class Telemetry:
    def __init__(self, path: str) -> None:
        self.f = open(path, "a", buffering=1 << 20)
        self.step = 0

    def write(self, rows: list[dict], ms: dict[str, float] | None) -> None:
        for r in rows:
            self.f.write(json.dumps({"step": self.step, **r, "ms": ms}) + "\n")
        self.step += 1

    def close(self) -> None:
        self.f.close()
