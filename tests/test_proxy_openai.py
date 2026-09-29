# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The Responses API path (Codex): what leaves in tokens, what comes back in names."""

import json

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web                                      # noqa: E402
from aiohttp.test_utils import TestClient, TestServer        # noqa: E402

from deidkit.patterns import RegexDetector                   # noqa: E402
from deidkit.proxy.anthropic import Refused                  # noqa: E402
from deidkit.proxy.engine import ScopeEngine                 # noqa: E402
from deidkit.proxy.openai import ResponsesRestorer, prepare_responses  # noqa: E402
from deidkit.proxy.server import MemoryRestore, Proxy, ProxyConfig    # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity     # noqa: E402
from deidkit.store import InMemoryTokenStore                 # noqa: E402

REAL = ("Brenner", "Harrowgate", "ada.brenner@", "7946 0958")
TURN_META = json.dumps({"session_id": "s1", "workspaces": {"/work/Brenner": {"remote": "git@host.example:harrowgate/brenner.git"}}})


def seeds() -> InMemorySeedSource:
    src = InMemorySeedSource()
    src.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    src.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    return src


def engine() -> ScopeEngine:
    return ScopeEngine(InMemoryTokenStore(), "case-1", seeds=seeds(), detector=RegexDetector())


def request() -> dict:
    return {
        "model": "gpt-test", "stream": True, "store": False,
        "include": ["reasoning.encrypted_content"], "prompt_cache_key": "k1",
        "client_metadata": {"session_id": "s1", "x-codex-turn-metadata": TURN_META},
        "input": [
            {"type": "additional_tools", "role": "developer", "id": "i0",
             "tools": [{"type": "namespace", "name": "functions"}]},
            {"type": "message", "role": "developer", "id": "i1", "content": [
                {"type": "input_text", "text": "<environment_context><cwd>/work/Brenner</cwd></environment_context>"}]},
            {"type": "message", "role": "user", "id": "i2", "content": [
                {"type": "input_text", "text": "Summarise the letter Ada Brenner sent (ada.brenner@harrowgate.example)."}]},
            {"type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": "Read PERSON_1."}],
             "encrypted_content": "gAAAAB..."},
            {"type": "function_call", "id": "f1", "call_id": "c1", "name": "exec_command",
             "arguments": json.dumps({"cmd": "cat /work/Brenner/letter.md"})},
            {"type": "function_call_output", "call_id": "c1",
             "output": "Dear Ms Brenner, Harrowgate Freight Ltd confirms. Call +44 20 7946 0958."},
        ],
    }


HEADERS = {"Authorization": "Bearer t", "chatgpt-account-id": "acc", "x-codex-turn-metadata": TURN_META}


def leaks(*objs) -> list[str]:
    blob = json.dumps(objs, ensure_ascii=False)
    return [r for r in REAL if r in blob]


async def test_everything_from_the_users_side_leaves_in_tokens():
    prep = await prepare_responses(request(), engine(), headers=HEADERS,
                                   restore_get=MemoryRestore().restore_get)
    assert leaks(prep.body, prep.headers) == []
    assert prep.body["input"][0] == request()["input"][0]            # the tool list as sent
    assert prep.body["input"][3] == request()["input"][3]            # reasoning untouched
    assert prep.headers["Authorization"] == "Bearer t"               # credentials untouched
    assert json.loads(prep.body["input"][4]["arguments"])["cmd"].startswith("cat /work/PERSON_")


async def test_images_are_withheld_and_unknown_items_refused():
    body = request()
    body["input"][2]["content"].append({"type": "input_image", "image_url": "data:image/png;base64,AA"})
    prep = await prepare_responses(body, engine(), headers={}, restore_get=MemoryRestore().restore_get)
    assert prep.withheld == ["input_image"]
    body["input"].append({"type": "teleport_call"})
    with pytest.raises(Refused):
        await prepare_responses(body, engine(), headers={}, restore_get=MemoryRestore().restore_get)


def ev(t, **kw):
    return (t, {"type": t, "sequence_number": 0, **kw})


async def _stream(eng, restore, token):
    text = f"The letter from {token} is short."
    args = json.dumps({"cmd": f"grep -n {token} /work/notes.md"})
    patch = f"*** Begin Patch\n*** Add File: summary.md\n+{token} confirmed.\n*** End Patch"
    cut = text.index(token) + 3
    events = [
        ev("response.created", response={"id": "resp_1"}),
        ev("response.output_item.added", output_index=0, item={"type": "message", "id": "m1", "role": "assistant", "content": []}),
        ev("response.output_text.delta", item_id="m1", output_index=0, content_index=0, delta=text[:cut]),
        ev("response.output_text.delta", item_id="m1", output_index=0, content_index=0, delta=text[cut:]),
        ev("response.output_text.done", item_id="m1", output_index=0, content_index=0, text=text),
        ev("response.output_item.done", output_index=0, item={"type": "message", "id": "m1", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}),
        ev("response.output_item.added", output_index=1, item={"type": "function_call", "id": "f2", "call_id": "c2",
            "name": "exec_command", "arguments": ""}),
        ev("response.function_call_arguments.delta", item_id="f2", output_index=1, delta=args),
        ev("response.output_item.done", output_index=1, item={"type": "function_call", "id": "f2", "call_id": "c2",
            "name": "exec_command", "arguments": args}),
        ev("response.output_item.added", output_index=2, item={"type": "custom_tool_call", "id": "p1", "call_id": "c3",
            "name": "apply_patch", "input": ""}),
        ev("response.custom_tool_call_input.delta", item_id="p1", output_index=2, delta=patch),
        ev("response.custom_tool_call_input.done", item_id="p1", output_index=2, input=patch),
        ev("response.output_item.done", output_index=2, item={"type": "custom_tool_call", "id": "p1", "call_id": "c3",
            "name": "apply_patch", "input": patch}),
        ev("response.output_item.done", output_index=3, item={"type": "function_call", "id": "f3", "call_id": "c4",
            "name": "mcp__search__query", "arguments": json.dumps({"q": token})}),
        ev("response.completed", response={"id": "resp_1", "output": [
            {"type": "message", "id": "m1", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}),
    ]
    prep = await prepare_responses(request(), eng, headers={}, restore_get=restore.restore_get)
    restorer = ResponsesRestorer(eng, prep.mapping, restore.restore_put)
    out = []
    for name, data in events:
        out += await restorer.event(name, data)
    return prep, text, args, patch, out


async def test_the_answer_comes_back_in_names_and_mcp_keeps_tokens():
    eng, restore = engine(), MemoryRestore()
    prep0 = await prepare_responses(request(), eng, headers={}, restore_get=restore.restore_get)
    token = next(t for t, v in prep0.mapping.items() if "Brenner" in v and t.startswith("PERSON"))
    real = prep0.mapping[token]
    prep, text, args, patch, out = await _stream(eng, restore, token)
    shown = "".join(d["delta"] for n, d in out if d["type"] == "response.output_text.delta")
    assert shown == f"The letter from {real} is short."
    done = {d["item"]["id"]: d["item"] for n, d in out if d["type"] == "response.output_item.done"}
    assert done["m1"]["content"][0]["text"] == shown
    assert json.loads(done["f2"]["arguments"]) == {"cmd": f"grep -n {real} /work/notes.md"}
    assert f"+{real} confirmed." in done["p1"]["input"]
    assert token in done["f3"]["arguments"]                          # MCP keeps the token
    completed = next(d for n, d in out if d["type"] == "response.completed")
    assert real in completed["response"]["output"][0]["content"][0]["text"]
    assert [d["sequence_number"] for n, d in out] == list(range(len(out)))


async def test_the_models_own_words_go_back_byte_for_byte():
    eng, restore = engine(), MemoryRestore()
    prep0 = await prepare_responses(request(), eng, headers={}, restore_get=restore.restore_get)
    token = next(t for t, v in prep0.mapping.items() if "Brenner" in v and t.startswith("PERSON"))
    prep, text, args, patch, out = await _stream(eng, restore, token)
    done = {d["item"]["id"]: d["item"] for n, d in out if d["type"] == "response.output_item.done"}
    body = request()
    body["input"] += [done["m1"], done["f2"],
                      {"type": "function_call_output", "call_id": "c2", "output": "3: Ada Brenner"},
                      done["p1"],
                      {"type": "custom_tool_call_output", "call_id": "c3", "output": "Done"}]
    nxt = await prepare_responses(body, eng, headers={}, restore_get=restore.restore_get)
    items = {i.get("id"): i for i in nxt.body["input"] if i.get("id")}
    assert items["m1"]["content"][0]["text"] == text
    assert items["f2"]["arguments"] == args
    assert items["p1"]["input"] == patch
    assert nxt.restored == 3
    assert leaks(nxt.body) == []


class Backend:
    def __init__(self):
        self.received, self.headers = [], []

    async def responses(self, request):
        body = await request.json()
        self.received.append(body)
        self.headers.append(dict(request.headers))
        token = next(t for t in json.dumps(body).replace('"', " ").split() if t.startswith("PERSON_"))
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        text = f"From {token}."
        for i, (t, extra) in enumerate([
                ("response.created", {"response": {"id": "r"}}),
                ("response.output_item.added", {"output_index": 0, "item": {"type": "message", "id": "m", "content": []}}),
                ("response.output_text.delta", {"item_id": "m", "output_index": 0, "content_index": 0, "delta": text}),
                ("response.output_item.done", {"output_index": 0, "item": {"type": "message", "id": "m",
                    "role": "assistant", "content": [{"type": "output_text", "text": text}]}}),
                ("response.completed", {"response": {"id": "r", "output": []}})]):
            await resp.write(f"event: {t}\ndata: {json.dumps({'type': t, 'sequence_number': i, **extra})}\n\n".encode())
        await resp.write_eof()
        return resp

    async def models(self, request):
        return web.json_response({"models": []})


async def test_codex_through_the_proxy(tmp_path):
    backend = Backend()
    app = web.Application()
    app.router.add_post("/codex/responses", backend.responses)
    app.router.add_get("/codex/models", backend.models)
    upstream = TestServer(app)
    await upstream.start_server()
    cfg = ProxyConfig(default_scope="case-1", upstream_chatgpt=str(upstream.make_url("/codex")),
                      detector=RegexDetector(), seeds=seeds(), record=tmp_path / "record.jsonl")
    client = TestClient(TestServer(Proxy(InMemoryTokenStore(), cfg).app()))
    await client.start_server()
    try:
        r = await client.get("/v1/models?client_version=1", headers={"chatgpt-account-id": "acc"})
        assert r.status == 200 and await r.json() == {"models": []}
        r = await client.get("/v1/responses", headers={"Upgrade": "websocket", "Connection": "Upgrade"})
        assert r.status == 426
        r = await client.post("/v1/responses", json=request(), headers=HEADERS)
        assert r.status == 200
        shown = ""
        async for raw in r.content:
            line = raw.decode().strip()
            if line.startswith("data:"):
                d = json.loads(line[5:])
                if d["type"] == "response.output_text.delta":
                    shown += d["delta"]
        assert shown == "From Ada Brenner."
        assert leaks(backend.received[0]) == []
        assert "Brenner" not in backend.headers[0]["x-codex-turn-metadata"]
        assert backend.headers[0]["Authorization"] == "Bearer t"
        assert "Brenner" not in (tmp_path / "record.jsonl").read_text()
    finally:
        await client.close()
        await upstream.close()


async def test_names_inside_json_tool_output_are_hidden():
    # Codex's code mode returns a command's output as JSON inside a text part: line breaks are
    # "\n" escapes, and a name right after one used to cross in clear.
    body = request()
    output = json.dumps({"chunk_id": "c", "output": "Dear Ms Brenner,\n\nHarrowgate Freight Ltd asks.\nAda Brenner"})
    body["input"].append({"type": "custom_tool_call_output", "call_id": "c9", "output": [
        {"type": "input_text", "text": "Script completed\nOutput:\n"}, {"type": "input_text", "text": output}]})
    prep = await prepare_responses(body, engine(), headers={}, restore_get=MemoryRestore().restore_get)
    decoded = json.loads(prep.body["input"][-1]["output"][1]["text"])["output"]
    assert "Harrowgate" not in decoded and "Brenner" not in decoded and "Ada" not in decoded


async def test_a_code_mode_script_gets_names_only_when_it_stays_on_the_machine():
    from deidkit.proxy.openai import restore_item
    eng = engine()
    prep = await prepare_responses(request(), eng, headers={}, restore_get=MemoryRestore().restore_get)
    token = next(t for t, v in prep.mapping.items() if "Brenner" in v and t.startswith("PERSON"))
    mapping = {**prep.mapping, token: 'Ada "Brenner"'}              # a value with quotes
    local = {"type": "custom_tool_call", "name": "exec", "input":
             f'text(await tools.apply_patch("*** Begin Patch\\n+{token} confirmed.\\n*** End Patch"));'}
    out = await restore_item(local, eng, mapping, None)
    assert '+Ada \\"Brenner\\" confirmed.' in out["input"]         # escaped for the literal
    remote = {"type": "custom_tool_call", "name": "exec",
              "input": f'text(await tools.mcp__crm__lookup({{q:"{token}"}}));'}
    assert (await restore_item(remote, eng, mapping, None))["input"] == remote["input"]
