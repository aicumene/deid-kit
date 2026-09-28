# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The Messages API transforms: what leaves in tokens, what comes back in names."""

import json

import pytest

from deidkit.patterns import RegexDetector
from deidkit.proxy.anthropic import Refused, StreamRestorer, prepare_request
from deidkit.proxy.engine import ScopeEngine
from deidkit.proxy.server import MemoryRestore
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

REAL = ("Brenner", "Harrowgate", "ada.brenner@", "7946 0958", "Lena", "Voss")
ATTRIBUTION = {"type": "text", "text": "x-anthropic-billing-header: cc_version=9.9.9; cch=0f3a;"}


def engine() -> ScopeEngine:
    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    seeds.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    return ScopeEngine(InMemoryTokenStore(), "case-1", seeds=seeds, detector=RegexDetector())


def request() -> dict:
    return {
        "model": "m", "max_tokens": 100, "stream": True,
        "system": [ATTRIBUTION, {"type": "text", "text": "You help Ada Brenner of Harrowgate Freight Ltd."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text":
                "Summarise the letter from Ada Brenner (ada.brenner@harrowgate.example)."}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "The letter is from PERSON_1.", "signature": "c2ln"},
                {"type": "tool_use", "id": "t1", "name": "Read",
                 "input": {"file_path": "/work/Brenner/letter.md"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "Dear Ms Brenner, Harrowgate Freight Ltd confirms. "
                                         "Call +44 20 7946 0958."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "iVBORw0KGgo="}}]}]},
        ],
    }


def leaks(obj) -> list[str]:
    blob = json.dumps(obj, ensure_ascii=False)
    return [r for r in REAL if r in blob]


async def test_everything_from_the_users_side_leaves_in_tokens():
    prep = await prepare_request(request(), engine(), restore_get=MemoryRestore().restore_get)
    assert leaks(prep.body) == []
    assert prep.body["system"][0] == ATTRIBUTION                    # the attribution block as sent
    assert prep.body["messages"][1]["content"][0] == request()["messages"][1]["content"][0]
    assert prep.withheld == ["image"]
    note = prep.body["messages"][2]["content"][0]["content"][1]
    assert note["type"] == "text" and "withheld" in note["text"]
    assert any("Brenner" in v for v in prep.mapping.values())


async def test_the_request_is_the_same_bytes_on_every_turn():
    eng = engine()
    first = await prepare_request(request(), eng, restore_get=MemoryRestore().restore_get)
    again = await prepare_request(request(), eng, restore_get=MemoryRestore().restore_get)
    assert json.dumps(first.body) == json.dumps(again.body)


async def test_a_name_learned_late_in_a_request_is_hidden_everywhere_in_it():
    # The e-mail in the second piece enrols "Lena Voss" as a person; the first piece, tokenised
    # before that, must be tokenised again.
    eng = engine()
    reds = await eng.tokenize_all(["Meeting with Lena Voss on Monday.",
                                   "Her mailbox: lena.voss@northfield.example"])
    assert all("Lena" not in r.text and "Voss" not in r.text for r in reds)


async def test_an_unknown_block_type_refuses_the_request():
    body = request()
    body["messages"][0]["content"].append({"type": "hologram", "data": "?"})
    with pytest.raises(Refused):
        await prepare_request(body, engine(), restore_get=MemoryRestore().restore_get)


async def test_images_can_refuse_the_request_instead():
    with pytest.raises(Refused):
        await prepare_request(request(), engine(), restore_get=MemoryRestore().restore_get,
                              binary="refuse")


def _split(text: str, at: int) -> list[str]:
    return [text[:at], text[at:]]


async def _answer(eng, restore, model_text: str, tool_input: dict, *, lower: bool = False):
    """Stream a scripted answer through the restorer; the model writes the person's token."""
    prep = await prepare_request(request(), eng, restore_get=restore.restore_get)
    token = next(t for t, v in prep.mapping.items() if "Brenner" in v and t.startswith("PERSON"))
    written = token.lower() if lower else token
    text = model_text.format(t=written)
    raw_input = json.dumps({k: v.format(t=written) for k, v in tool_input.items()})
    events = [
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        *[("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": part}})
          for part in _split(text, text.index(written[:4]) + 3)],        # a token split in two
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block":
            {"type": "tool_use", "id": "t2", "name": "Write", "input": {}}}),
        *[("content_block_delta", {"type": "content_block_delta", "index": 1,
                                   "delta": {"type": "input_json_delta", "partial_json": part}})
          for part in _split(raw_input, 17)],
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("content_block_start", {"type": "content_block_start", "index": 2, "content_block":
            {"type": "tool_use", "id": "t3", "name": "WebFetch", "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 2, "delta": {
            "type": "input_json_delta", "partial_json": json.dumps({"url": f"https://x.example/?q={token}"})}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
    ]
    restorer = StreamRestorer(eng, prep.mapping, restore.restore_put)
    out = []
    for name, data in events:
        out += await restorer.event(name, data)
    return prep, token, text, raw_input, out


def _collect(out, index, key):
    return "".join(d["delta"].get(key, "") for n, d in out
                   if n == "content_block_delta" and d["index"] == index)


async def test_the_answer_comes_back_in_names_and_the_web_keeps_tokens():
    eng, restore = engine(), MemoryRestore()
    prep, token, text, raw_input, out = await _answer(
        eng, restore, "The letter from {t} is short.",
        {"file_path": "/work/summary.md", "content": "{t} confirmed."})
    real = prep.mapping[token]
    assert _collect(out, 0, "text") == f"The letter from {real} is short."
    assert json.loads(_collect(out, 1, "partial_json")) == {
        "file_path": "/work/summary.md", "content": f"{real} confirmed."}
    assert token in _collect(out, 2, "partial_json")                # WebFetch keeps the token


async def test_the_models_own_words_go_back_byte_for_byte():
    # The model writes the token in lower case; the agent is shown the name; the next request
    # must carry the model's own spelling again, so its history is exactly what it produced.
    eng, restore = engine(), MemoryRestore()
    prep, token, text, raw_input, out = await _answer(
        eng, restore, "the letter from {t} is short.", {"content": "{t} confirmed."}, lower=True)
    shown_text = _collect(out, 0, "text")
    shown_input = json.loads(_collect(out, 1, "partial_json"))
    assert prep.mapping[token] in shown_text and token.lower() not in shown_text
    body = request()
    body["messages"].append({"role": "assistant", "content": [
        {"type": "text", "text": shown_text},
        {"type": "tool_use", "id": "t2", "name": "Write", "input": shown_input}]})
    nxt = await prepare_request(body, eng, restore_get=restore.restore_get)
    history = nxt.body["messages"][-1]["content"]
    assert history[0]["text"] == text
    assert history[1]["input"] == json.loads(raw_input)
    assert nxt.restored == 2
    assert leaks(nxt.body) == []
