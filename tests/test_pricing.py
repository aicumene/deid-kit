# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""What a task cost, with the prompt cache and without it — on the numbers of real tasks."""

import pytest

from deidkit.agents.pricing import RequestUsage, price, task_cost


def test_a_cold_session_pays_the_write_and_gains_nothing_yet():
    # MEASURED 07.10.2026: a fresh Claude Code session on Claude Opus 5.5, one short question —
    # Claude Code writes its tools and instructions to the 1-hour cache, at twice the input price.
    first = RequestUsage.from_anthropic({"input_tokens": 2, "cache_creation_input_tokens": 30294,
                                         "cache_read_input_tokens": 0, "output_tokens": 194,
                                         "cache_creation": {"ephemeral_5m_input_tokens": 0,
                                                            "ephemeral_1h_input_tokens": 30294}},
                                        "claude-opus-5-5")
    assert (first.write_1h, first.write_5m) == (30294, 0)
    cost = task_cost([first])
    assert cost["usd"] == pytest.approx(0.2462, abs=1e-4)
    assert cost["usd_without_cache"] == pytest.approx(0.1251, abs=1e-4)
    assert cost["saved_usd"] < 0


def test_a_warm_session_reads_the_cache_at_a_twentieth_of_the_price():
    # The owner's first task in the app, 07.10.2026 (its writes counted as 1-hour ones).
    warm = RequestUsage(model="claude-opus-5-5", input=16, write_1h=71016, read=195857, output=2083)
    cost = task_cost([warm])
    assert cost["usd"] == pytest.approx(0.6490, abs=1e-4)
    assert cost["usd_without_cache"] == pytest.approx(1.1092, abs=1e-4)
    assert cost["saved_usd"] == pytest.approx(0.4602, abs=1e-4)
    assert cost["cache_read"] == 195857 and cost["requests"] == 1


def test_each_model_reads_at_its_own_price_and_writes_at_the_lifetimes():
    million = 1_000_000
    assert task_cost([RequestUsage(model="claude-fable-5-1", read=million)])["usd"] == pytest.approx(0.25)
    assert task_cost([RequestUsage(model="claude-sonnet-5-5", read=million)])["usd"] == pytest.approx(0.20)
    assert task_cost([RequestUsage(model="claude-haiku-4-5", write_5m=million)])["usd"] == pytest.approx(1.25)
    assert task_cost([RequestUsage(model="claude-haiku-4-5", write_1h=million)])["usd"] == pytest.approx(2.0)


def test_a_model_not_in_the_table_is_counted_in_tokens_only():
    cost = task_cost([RequestUsage(model="claude-opus-5-5", input=10), RequestUsage(model="local-qwen", input=5)])
    assert cost["input"] == 15 and cost["usd"] is None and cost["models"] == ["claude-opus-5-5", "local-qwen"]


def test_ids_find_their_prices_and_not_a_shorter_neighbour():
    assert price("claude-opus-5-5") == (4.0, 20.0, 0.20)
    assert price("claude-opus-5-20260101") == (5.0, 25.0, 0.50)     # not Claude Opus 5.5
    assert price("claude-haiku-4-5-20251001") == (1.0, 5.0, 0.10)
    assert price("claude-opus-5-5[1m]") == (4.0, 20.0, 0.20)
    assert price("gpt-5") is None


def test_writes_without_a_split_count_as_the_default_lifetime():
    u = RequestUsage.from_anthropic({"input_tokens": 3, "cache_creation_input_tokens": 100,
                                     "cache_read_input_tokens": 40, "output_tokens": 7}, "claude-sonnet-5-5")
    assert (u.input, u.write_5m, u.write_1h, u.read, u.output) == (3, 100, 0, 40, 7)
