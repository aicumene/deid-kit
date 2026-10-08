# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""What a task cost at Anthropic's API prices — with the prompt cache, and as it would without it.

The proxy reads the usage of every answer (:meth:`RequestUsage.from_anthropic`): the model, the input
the cache did not cover, the cache writes by lifetime and the cache reads, the output. A write costs
1.25x the input price for 5 minutes and 2x for 1 hour; a read costs 0.1x on most models — 0.05x on
Claude Opus 5.5, 0.025x on Claude Fable 5.1 and Mythos 5.1. Without the cache every input token would
cost the input price, and the output the same. A cold session pays the write premium and gains from
the next request that reads what it wrote.

Prices: Anthropic's first-party list prices, $ per million tokens, as of 25 September 2026. A model
not in the table is counted in tokens only. On a personal sign-in nothing is billed per token: the
figures say what the same work costs on the organization's key.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: model id prefix → (input, output, cache read), $ per million tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-mythos-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.00),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
WRITE_5M = 1.25
WRITE_1H = 2.0


def price(model: str) -> tuple[float, float, float] | None:
    """The prices of `model`: its id, or the longest known id it begins with (dated or variant ids)."""
    known = [p for p in PRICES if model == p or model.startswith(p + "-") or model.startswith(p + "[")]
    return PRICES[max(known, key=len)] if known else None


@dataclass
class RequestUsage:
    """One answer of the model, as the provider counted it."""

    model: str = ""
    input: int = 0
    write_5m: int = 0
    write_1h: int = 0
    read: int = 0
    output: int = 0

    @classmethod
    def from_anthropic(cls, usage: dict, model: str) -> RequestUsage:
        """From the Messages API's `usage`. The split of the cache writes by lifetime comes in
        `cache_creation`; without it the writes count as the default 5-minute ones."""
        def n(value) -> int:
            return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

        split = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else None
        written = n(usage.get("cache_creation_input_tokens"))
        long = n(split.get("ephemeral_1h_input_tokens")) if split else 0
        short = n(split.get("ephemeral_5m_input_tokens")) if split else written
        if split and short + long < written:                     # a split that does not add up: the rest is 5m
            short = written - long
        return cls(model=model or "", input=n(usage.get("input_tokens")), write_5m=short, write_1h=long,
                   read=n(usage.get("cache_read_input_tokens")), output=n(usage.get("output_tokens")))


def task_cost(requests: list[RequestUsage]) -> dict:
    """The task's tokens and, when every model is priced, its cost with the cache and without it:
    `usd`, `usd_without_cache`, `saved_usd` (negative when the cache was only written)."""
    totals = {f.name: sum(getattr(r, f.name) for r in requests) for f in fields(RequestUsage) if f.name != "model"}
    out: dict = {"requests": len(requests), "models": sorted({r.model for r in requests if r.model}),
                 "input": totals["input"], "cache_write_5m": totals["write_5m"],
                 "cache_write_1h": totals["write_1h"], "cache_read": totals["read"], "output": totals["output"],
                 "usd": None, "usd_without_cache": None, "saved_usd": None}
    priced = [(r, price(r.model)) for r in requests]
    if not requests or any(p is None for _, p in priced):
        return out
    with_cache = without = 0.0
    for r, (pin, pout, pread) in priced:
        with_cache += (r.input * pin + r.write_5m * pin * WRITE_5M + r.write_1h * pin * WRITE_1H
                       + r.read * pread + r.output * pout) / 1e6
        without += ((r.input + r.write_5m + r.write_1h + r.read) * pin + r.output * pout) / 1e6
    out.update(usd=round(with_cache, 4), usd_without_cache=round(without, 4), saved_usd=round(without - with_cache, 4))
    return out


__all__ = ["PRICES", "RequestUsage", "WRITE_1H", "WRITE_5M", "price", "task_cost"]
