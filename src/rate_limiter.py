"""Per-tool rate limiting and per-claim call budgets.

Enforced at the MCP protocol boundary — not as soft instructions inside
a prompt the agent can ignore.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ToolBudget:
    """Per-invocation budget for a single tool."""
    max_calls: int
    calls: int = 0

    def consume(self) -> bool:
        if self.calls >= self.max_calls:
            return False
        self.calls += 1
        return True

    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls)


@dataclass
class ClaimBudget:
    """Per-claim budget — tracks tool usage across all tools for one claim."""
    claim_id: str
    searches_used: int = 0
    fetches_used: int = 0
    max_searches: int = 8
    max_fetches: int = 12

    def can_search(self) -> bool:
        return self.searches_used < self.max_searches

    def can_fetch(self) -> bool:
        return self.fetches_used < self.max_fetches

    def record_search(self):
        self.searches_used += 1

    def record_fetch(self):
        self.fetches_used += 1

    def exhausted(self) -> bool:
        return self.searches_used >= self.max_searches and self.fetches_used >= self.max_fetches

    def status(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "searches": f"{self.searches_used}/{self.max_searches}",
            "fetches": f"{self.fetches_used}/{self.max_fetches}",
            "exhausted": self.exhausted(),
        }


class RateLimiter:
    """Sliding-window rate limiter per tool.

    Defaults: 60 requests per 60 seconds globally. Per-tool overrides possible.
    """

    def __init__(self, max_requests: int = 60, window_sec: float = 60.0):
        self._max = max_requests
        self._window = window_sec
        self._global: list[float] = []           # timestamps of global requests
        self._per_tool: dict[str, list[float]] = defaultdict(list)
        self._claim_budgets: dict[str, ClaimBudget] = {}

    def _prune(self, timestamps: list[float]) -> list[float]:
        cutoff = time.monotonic() - self._window
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        return timestamps

    def allow(self, tool_name: str) -> bool:
        self._global = self._prune(self._global)
        if len(self._global) >= self._max:
            return False
        per_tool = self._prune(self._per_tool[tool_name])
        if len(per_tool) >= self._max // 2:   # 30/min per tool
            return False
        now = time.monotonic()
        self._global.append(now)
        per_tool.append(now)
        self._per_tool[tool_name] = per_tool
        return True

    def get_claim_budget(self, claim_id: str) -> ClaimBudget:
        if claim_id not in self._claim_budgets:
            self._claim_budgets[claim_id] = ClaimBudget(claim_id=claim_id)
        return self._claim_budgets[claim_id]

    def claim_can_search(self, claim_id: str) -> bool:
        return self.get_claim_budget(claim_id).can_search()

    def claim_can_fetch(self, claim_id: str) -> bool:
        return self.get_claim_budget(claim_id).can_fetch()

    def claim_record_search(self, claim_id: str):
        self.get_claim_budget(claim_id).record_search()

    def claim_record_fetch(self, claim_id: str):
        self.get_claim_budget(claim_id).record_fetch()

    def claim_status(self, claim_id: str) -> dict:
        return self.get_claim_budget(claim_id).status()

    def remaining_global(self) -> int:
        self._global = self._prune(self._global)
        return max(0, self._max - len(self._global))


# Singleton for the MCP server
_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
