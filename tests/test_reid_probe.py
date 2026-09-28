# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The re-identification probe: rank and margin are counted per SCOPE, not per chunk; the
source chunk is excluded from the ranking; the judge accepts only a reference from the card
index and runs on the trusted plane."""

from __future__ import annotations

import json
import math
import uuid

import pytest

from deidkit import reid
from deidkit.classification import Sensitivity
from deidkit.model import QUERY


class _Index:
    """A passage index that answers with fixed rows and records what it was asked."""

    def __init__(self, rows, own=None):
        self.rows, self.own, self.calls = rows, own, []

    async def nearest(self, vector, *, k, exclude_chunk_id=None):
        self.calls.append(("nearest", k, exclude_chunk_id))
        return list(self.rows)

    async def nearest_in_scope(self, vector, scope_id, *, exclude_chunk_id=None):
        self.calls.append(("in_scope", scope_id, exclude_chunk_id))
        return self.own


class _Router:
    def __init__(self, text="{}"):
        self.text, self.requests, self.embeds = text, [], []

    async def embed(self, texts, *, sensitivity, kind):
        self.embeds.append((list(texts), sensitivity, kind))
        return [[0.0] * 4 for _ in texts]

    async def generate(self, request):
        self.requests.append(request)

        class R:  # noqa: D401
            text = self.text
        return R()


async def test_rank_and_margin_are_per_scope_and_source_chunk_is_excluded():
    src, other = uuid.uuid4(), uuid.uuid4()
    rows = [(src, 0.10), (src, 0.12), (other, 0.30), (src, 0.31), (other, 0.35)]
    idx, router, chunk = _Index(rows), _Router(), uuid.uuid4()
    p = await reid.probe_retrieval(idx, router, "text", source_scope_id=src, exclude_chunk_id=chunk)
    assert p.source_rank == 1 and p.margin == pytest.approx(0.20) and p.identifies
    assert p.top == [(str(src), 0.10), (str(other), 0.30)]
    assert idx.calls == [("nearest", 20, chunk)]                # own chunk out of the ranking
    assert router.embeds == [(["text"], Sensitivity.CONFIDENTIAL, QUERY)]


async def test_source_absent_from_top_k_gets_its_own_distance_and_no_rank():
    src, a, b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    idx = _Index([(a, 0.20), (b, 0.25)], own=0.40)
    p = await reid.probe_retrieval(idx, _Router(), "text", source_scope_id=src)
    assert p.source_rank is None and p.margin == pytest.approx(-0.20) and not p.identifies
    assert idx.calls[-1] == ("in_scope", src, None)


async def test_same_topic_scope_within_threshold_is_not_identification():
    src, twin = uuid.uuid4(), uuid.uuid4()
    p = await reid.probe_retrieval(_Index([(src, 0.30), (twin, 0.32)]), _Router(), "t",
                                   source_scope_id=src, threshold=0.05)
    assert p.source_rank == 1 and not p.identifies              # margin 0.02 < 0.05: a twin


async def test_the_in_memory_index_ranks_by_cosine_distance():
    idx = reid.InMemoryPassageIndex()
    idx.add("a1", "A", [1.0, 0.0])
    idx.add("a2", "A", [0.0, 1.0])
    idx.add("b1", "B", [0.8, 0.6])
    idx.add("c1", "C", None)                                    # no vector: never returned
    idx.add("z1", "Z", [0.0, 0.0])                              # zero vector: NaN, sorted last
    got = await idx.nearest([1.0, 0.0], k=10)
    assert [s for s, _ in got] == ["A", "B", "A", "Z"]
    assert got[0][1] == pytest.approx(0.0) and got[1][1] == pytest.approx(0.2)
    assert math.isnan(got[-1][1])
    assert [s for s, _ in await idx.nearest([1.0, 0.0], k=10, exclude_chunk_id="a1")][0] == "B"
    assert await idx.nearest_in_scope([1.0, 0.0], "A", exclude_chunk_id="a1") == pytest.approx(1.0)
    assert await idx.nearest_in_scope([1.0, 0.0], "C") is None


async def test_the_probe_over_the_in_memory_index():
    idx = reid.InMemoryPassageIndex()
    idx.add("s1", "SRC", [1.0, 0.0, 0.0])
    idx.add("s2", "SRC", [0.9, 0.1, 0.0])
    idx.add("o1", "OTHER", [0.0, 1.0, 0.0])

    class Router(_Router):
        async def embed(self, texts, *, sensitivity, kind):
            return [[1.0, 0.0, 0.0]]

    p = await reid.probe_retrieval(idx, Router(), "x", source_scope_id="SRC", exclude_chunk_id="s1")
    assert p.source_rank == 1 and p.identifies
    assert p.margin == pytest.approx(1.0 - reid.cosine_distance([1, 0, 0], [0.9, 0.1, 0]))


async def test_judge_accepts_only_a_listed_reference_and_runs_on_the_trusted_plane():
    cands = [reid.Candidate("TEST-SCOPE-A", "Weber / Lindqvist"),
             reid.Candidate("TEST-SCOPE-B", "Harbor Homes / Kessler")]
    r = _Router(json.dumps({"reference": "TEST-SCOPE-B", "clue": "arrears of 735 EUR"}))
    v = await reid.probe_judge(r, "excerpt", cands, model="judge-model:latest")
    assert v.reference == "TEST-SCOPE-B" and v.clue == "arrears of 735 EUR"
    req = r.requests[0]
    assert req.model == "judge-model:latest" and req.sensitivity.name == "CONFIDENTIAL"
    assert req.think is False and req.temperature == 0.0 and req.max_tokens == 200
    assert req.response_format["properties"]["reference"]["enum"] == ["TEST-SCOPE-A", "TEST-SCOPE-B",
                                                                      "UNKNOWN"]
    assert "TEST-SCOPE-A: Weber / Lindqvist" in req.messages[-1].content
    # a reference not on the card → UNKNOWN, never someone else's line
    r2 = _Router(json.dumps({"reference": "TEST-SCOPE-Z", "clue": "x"}))
    assert (await reid.probe_judge(r2, "excerpt", cands)).reference == "UNKNOWN"
    # an answer that is not JSON at all → UNKNOWN
    assert (await reid.probe_judge(_Router("no idea"), "excerpt", cands)).reference == "UNKNOWN"
