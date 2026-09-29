# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The proxy end to end, against a scripted upstream that plays the model."""

import json
import re

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web                                     # noqa: E402
from aiohttp.test_utils import TestClient, TestServer       # noqa: E402

from deidkit.patterns import RegexDetector                  # noqa: E402
from deidkit.proxy.server import Proxy, ProxyConfig         # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity    # noqa: E402
from deidkit.sqlite_store import SQLiteTokenStore           # noqa: E402

TOKEN = re.compile(r"PERSON_\d+")


def sse(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


class Model:
    """Plays the provider: records what it received, answers with the person's token."""

    def __init__(self):
        self.received: list[dict] = []
        self.headers: list[dict] = []

    async def messages(self, request):
        body = await request.json()
        self.received.append(body)
        self.headers.append(dict(request.headers))
        token = TOKEN.search(json.dumps(body)).group(0)
        text = f"I read the letter from {token}. Writing the summary."
        tool = json.dumps({"file_path": "/work/summary.md", "content": f"{token} confirmed the terms."})
        resp = web.StreamResponse(headers={"content-type": "text/event-stream",
                                           "request-id": "req_1", "x-should-retry": "false"})
        await resp.prepare(request)
        await resp.write(sse("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [],
            "model": body["model"], "usage": {"input_tokens": 1, "output_tokens": 0}}}))
        await resp.write(sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                     "content_block": {"type": "text", "text": ""}}))
        cut = text.index(token) + 4
        for part in (text[:cut], text[cut:]):
            await resp.write(sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                         "delta": {"type": "text_delta", "text": part}}))
            await resp.write(sse("ping", {"type": "ping"}))
        await resp.write(sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
        await resp.write(sse("content_block_start", {"type": "content_block_start", "index": 1,
                                                     "content_block": {"type": "tool_use", "id": "tu_1",
                                                                       "name": "Write", "input": {}}}))
        for part in (tool[:20], tool[20:]):
            await resp.write(sse("content_block_delta", {"type": "content_block_delta", "index": 1,
                                                         "delta": {"type": "input_json_delta",
                                                                   "partial_json": part}}))
        await resp.write(sse("content_block_stop", {"type": "content_block_stop", "index": 1}))
        await resp.write(sse("message_delta", {"type": "message_delta",
                                               "delta": {"stop_reason": "tool_use"},
                                               "usage": {"output_tokens": 20}}))
        await resp.write(sse("message_stop", {"type": "message_stop"}))
        await resp.write_eof()
        return resp

    async def count(self, request):
        self.received.append(await request.json())
        self.headers.append(dict(request.headers))
        return web.json_response({"input_tokens": 42})


async def read_stream(resp):
    text, tool, names = "", "", []
    async for raw in resp.content:
        line = raw.decode().strip()
        if line.startswith("event:"):
            names.append(line[6:].strip())
        if not line.startswith("data:"):
            continue
        data = json.loads(line[5:])
        if data.get("type") == "content_block_delta":
            d = data["delta"]
            text += d.get("text", "")
            tool += d.get("partial_json", "")
    return text, tool, names


@pytest.fixture
async def rig(tmp_path):
    model = Model()
    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/messages", model.messages)
    upstream_app.router.add_post("/v1/messages/count_tokens", model.count)
    upstream = TestServer(upstream_app)
    await upstream.start_server()

    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    store = SQLiteTokenStore(tmp_path / "vault.sqlite")
    record = tmp_path / "record.jsonl"
    cfg = ProxyConfig(default_scope="case-1", upstream=str(upstream.make_url("")).rstrip("/"),
                      record=record, audit=tmp_path / "audit.jsonl", detector=RegexDetector(),
                      seeds=seeds)
    client = TestClient(TestServer(Proxy(store, cfg).app()))
    await client.start_server()
    yield model, client, record, tmp_path
    await client.close()
    await upstream.close()


def first_request():
    return {"model": "claude-test", "max_tokens": 200, "stream": True,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=9; cch=1;"},
                       {"type": "text", "text": "Working directory: /work/Brenner"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text":
                          "Summarise the letter Ada Brenner sent (ada.brenner@harrowgate.example)."}]}]}


async def test_a_turn_through_the_proxy(rig):
    model, client, record, tmp = rig
    resp = await client.post("/v1/messages", json=first_request(),
                             headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01",
                                      "anthropic-beta": "a,b"})
    assert resp.status == 200
    assert resp.headers["request-id"] == "req_1" and resp.headers["x-should-retry"] == "false"
    text, tool, names = await read_stream(resp)

    sent = json.dumps(model.received[0])
    assert "Brenner" not in sent and "ada.brenner@" not in sent        # nothing real crossed
    assert model.headers[0]["x-api-key"] == "test-key"                   # credentials pass through
    assert model.headers[0]["anthropic-beta"] == "a,b"
    assert "x-deid-scope" not in {k.lower() for k in model.headers[0]}
    assert text == "I read the letter from Ada Brenner. Writing the summary."
    assert json.loads(tool) == {"file_path": "/work/summary.md", "content": "Ada Brenner confirmed the terms."}
    assert names.count("ping") == 2                                      # keep-alives forwarded

    # The record holds what crossed, in tokens; the audit holds hashes, never content.
    crossed = record.read_text()
    assert "Brenner" not in crossed and "PERSON_" in crossed
    audit = (tmp / "audit.jsonl").read_text()
    assert "Brenner" not in audit and "sha256" in audit

    # Next turn: the agent sends back what it was shown; the model gets its own words again.
    body = first_request()
    body["messages"] += [
        {"role": "assistant", "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": "tu_1", "name": "Write", "input": json.loads(tool)}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1",
                                      "content": "File written: /work/summary.md"}]}]
    resp = await client.post("/v1/messages", json=body, headers={"x-api-key": "test-key"})
    await read_stream(resp)
    second = model.received[1]
    assert second["messages"][:1] == model.received[0]["messages"]       # history unchanged
    token = TOKEN.search(json.dumps(model.received[0])).group(0)
    assert second["messages"][1]["content"][0]["text"] == \
        f"I read the letter from {token}. Writing the summary."
    assert second["messages"][1]["content"][1]["input"]["content"] == f"{token} confirmed the terms."
    assert "Brenner" not in json.dumps(second)


async def test_count_tokens_is_tokenised_too(rig):
    model, client, record, tmp = rig
    body = first_request()
    body.pop("stream")
    resp = await client.post("/v1/messages/count_tokens", json=body)
    assert resp.status == 200 and (await resp.json()) == {"input_tokens": 42}
    assert "Brenner" not in json.dumps(model.received[0])


async def test_what_the_proxy_cannot_read_is_not_sent(rig):
    model, client, record, tmp = rig
    body = first_request()
    body["messages"][0]["content"].append({"type": "hologram"})
    resp = await client.post("/v1/messages", json=body)
    assert resp.status == 400 and "hologram" in (await resp.json())["error"]["message"]
    resp = await client.post("/v1/files", data=b"x")
    assert resp.status == 403
    assert model.received == []


async def test_a_held_key_is_lent_only_to_the_agent(tmp_path):
    model = Model()
    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/messages/count_tokens", model.count)

    async def hello(request):
        model.headers.append(dict(request.headers))
        return web.Response()
    upstream_app.router.add_route("HEAD", "/api/hello", hello)
    upstream = TestServer(upstream_app)
    await upstream.start_server()
    cfg = ProxyConfig(default_scope="case-1", upstream=str(upstream.make_url("")).rstrip("/"),
                      upstream_key="sk-org-real", agent_key="stand-in-1")
    client = TestClient(TestServer(Proxy(SQLiteTokenStore(tmp_path / "v.sqlite"), cfg).app()))
    await client.start_server()
    try:
        body = first_request()
        body.pop("stream")
        for key in ("", "sk-org-real", "stand-in-2"):
            r = await client.post("/v1/messages/count_tokens", json=body, headers={"x-api-key": key})
            assert r.status == 401
        assert model.received == []
        r = await client.post("/v1/messages/count_tokens", json=body,
                              headers={"x-api-key": "stand-in-1", "authorization": "Bearer personal"})
        assert r.status == 200
        sent = {k.lower(): v for k, v in model.headers[-1].items()}
        assert sent["x-api-key"] == "sk-org-real" and "authorization" not in sent
        r = await client.post("/v1/responses", json={"input": "x"}, headers={"x-api-key": "stand-in-1"})
        assert r.status == 403
        # a connectivity check without the stand-in goes out, but with no key at all
        r = await client.head("/api/hello", headers={"authorization": "Bearer personal"})
        assert r.status == 200
        sent = {k.lower() for k in model.headers[-1]}
        assert "x-api-key" not in sent and "authorization" not in sent
    finally:
        await client.close()
        await upstream.close()


async def test_the_agent_opens_a_file_named_after_a_party(tmp_path):
    """MEASURED 29.09.2026: a file whose name held the client's short name crossed with a token
    in it, and the call to read it came back with the canonical name in its place."""
    received = []

    async def model(request):
        body = await request.json()
        received.append(body)
        folder = re.search(r"/work/([A-Z]+_\d+)", body["system"][0]["text"]).group(1)
        glued, spaced = body["messages"][2]["content"][0]["content"].splitlines()
        calls = [("Read", {"file_path": f"/work/{folder}/letters/{glued}"}),
                 ("Bash", {"command": f'cat "letters/{spaced}" | head -5'})]
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(sse("message_start", {"type": "message_start", "message": {
            "id": "msg_2", "type": "message", "role": "assistant", "content": [],
            "model": body["model"], "usage": {"input_tokens": 1, "output_tokens": 0}}}))
        for i, (name, args) in enumerate(calls):
            raw = json.dumps(args)
            await resp.write(sse("content_block_start", {"type": "content_block_start", "index": i,
                                                         "content_block": {"type": "tool_use", "id": f"tu_{i + 1}",
                                                                           "name": name, "input": {}}}))
            for part in (raw[:15], raw[15:]):
                await resp.write(sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                             "delta": {"type": "input_json_delta",
                                                                       "partial_json": part}}))
            await resp.write(sse("content_block_stop", {"type": "content_block_stop", "index": i}))
        await resp.write(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                                               "usage": {"output_tokens": 9}}))
        await resp.write(sse("message_stop", {"type": "message_stop"}))
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/messages", model)
    upstream = TestServer(app)
    await upstream.start_server()
    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    seeds.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    cfg = ProxyConfig(default_scope="case-1", upstream=str(upstream.make_url("")).rstrip("/"),
                      detector=RegexDetector(), seeds=seeds)
    client = TestClient(TestServer(Proxy(SQLiteTokenStore(tmp_path / "v.sqlite"), cfg).app()))
    await client.start_server()
    body = {"model": "claude-test", "max_tokens": 100, "stream": True,
            "system": [{"type": "text", "text": "Primary working directory: /work/Brenner\n"
                                                "Matter: HARROWGATE FREIGHT LTD, claim by Ada Brenner"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Open the reply to Harrowgate."}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_0", "name": "Bash",
                                                   "input": {"command": "ls letters"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_0", "content":
                                              "03-Reply-to-Harrowgate-Freight.md\n04 Brenner, Ada - note.md"}]}]}
    try:
        resp = await client.post("/v1/messages", json=body)
        assert resp.status == 200
        inputs: dict[int, str] = {}
        async for raw in resp.content:
            line = raw.decode().strip()
            if line.startswith("data:"):
                d = json.loads(line[5:])
                if d.get("type") == "content_block_delta" and d["delta"].get("type") == "input_json_delta":
                    inputs[d["index"]] = inputs.get(d["index"], "") + d["delta"]["partial_json"]
        assert json.loads(inputs[0]) == {"file_path": "/work/Brenner/letters/03-Reply-to-Harrowgate-Freight.md"}
        assert json.loads(inputs[1]) == {"command": 'cat "letters/04 Brenner, Ada - note.md" | head -5'}
        assert "Harrowgate" not in json.dumps(received[0]) and "Brenner" not in json.dumps(received[0])
    finally:
        await client.close()
        await upstream.close()
