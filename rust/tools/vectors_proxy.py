# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what deidkit.proxy's request preparation, answer restoration and spellings give, for the Rust
ones to give the same.

    python tools/vectors_proxy.py OUT_DIR   → OUT_DIR/proxy.json   (DEIDKIT_NATIVE=0)
"""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["DEIDKIT_NATIVE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deidkit import patterns  # noqa: E402
from deidkit.proxy import anthropic as A, openai as O, spelling  # noqa: E402
from deidkit.proxy.engine import ScopeEngine  # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity  # noqa: E402
from deidkit.store import InMemoryTokenStore  # noqa: E402

SALT = bytes(range(32))


class MemoryRestore:
    """deidkit.proxy.server.MemoryRestore (that module needs aiohttp)."""

    def __init__(self):
        self._d = {}

    def restore_get(self, key):
        return self._d.get(key)

    def restore_put(self, key, value):
        self._d[key] = value
ENTITIES = [{"type": "individual", "name": "Ada Brenner"}, {"type": "company", "name": "Harrowgate Freight Ltd"},
            {"type": "individual", "name": "Tobias Wren"}]

LETTER = "Dear Ada Brenner,\nHarrowgate Freight Ltd writes about the lease at Musterweg 12, 20095 Hamburg."
FILES = "letters/04 Brenner, Ada - note.md\n03-Reply-to-Harrowgate Freight Ltd-Freight.md\n/work/Harrowgate Freight Ltd/a.md"

MESSAGES = [
    {"model": "claude", "stream": True,
     "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                {"type": "text", "text": "You help Ada Brenner. Primary working directory: /work/matter"}],
     "messages": [{"role": "user", "content": LETTER}]},
    {"model": "claude", "stream": True, "system": "Plain system for Tobias Wren.",
     "messages": [{"role": "user", "content": [{"type": "text", "text": LETTER},
                                               {"type": "image", "source": {"type": "base64", "data": "AAAA"}, "cache_control": {"type": "ephemeral"}},
                                               {"type": "document", "source": {"type": "text", "data": "Tobias Wren's contract"}, "title": "Ada Brenner file"}]},
                  {"role": "assistant", "content": [{"type": "thinking", "thinking": "Ada Brenner?", "signature": "s"},
                                                    {"type": "text", "text": "I will read the file of Ada Brenner."},
                                                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/work/" + FILES.split("\n")[1]}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": FILES},
                                               {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "Ada Brenner"}, {"type": "image", "source": {}}]}]}]},
    {"model": "claude", "messages": [{"role": "user", "content": [{"type": "mystery"}]}]},
    {"model": "claude", "messages": [{"role": "user", "content": 5}]},
]


def sse_of(blocks):
    """A Messages API stream: text in pieces that split tokens, a tool call's arguments in pieces."""
    ev = [("message_start", {"type": "message_start", "message": {"id": "m", "model": "claude", "usage": {"input_tokens": 5}}})]
    for i, (kind, value, name) in enumerate(blocks):
        if kind == "text":
            ev.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": value[:3]}}))
            rest = value[3:]
            for j in range(0, len(rest), 5):
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": rest[j:j + 5]}}))
        else:
            ev.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": f"u{i}", "name": name, "input": {}}}))
            raw = json.dumps(value)
            for j in range(0, len(raw), 7):
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": raw[j:j + 7]}}))
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    ev.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}}))
    ev.append(("message_stop", {"type": "message_stop"}))
    return ev


RESPONSES = [
    {"model": "gpt", "instructions": "Help Ada Brenner.", "client_metadata": {"repo": "/work/Harrowgate Freight Ltd"},
     "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": LETTER}, {"type": "input_image", "image_url": "x"}]},
               {"role": "developer", "content": "Tobias Wren rules"},
               {"type": "reasoning", "encrypted_content": "zz"},
               {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": json.dumps({"cmd": "cat '03-Reply-to-Harrowgate Freight Ltd-Freight.md'"})},
               {"type": "function_call_output", "call_id": "c1", "output": FILES},
               {"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "*** Update File: letters/04 Brenner, Ada - note.md\n+Ada Brenner"},
               {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "done for Ada Brenner"}]},
               {"type": "local_shell_call", "action": {"command": ["ls", "/work/Harrowgate Freight Ltd"]}},
               {"type": "function_call", "call_id": "c3", "name": "x", "arguments": "not json Ada Brenner"}],
     "tools": [{"type": "namespace", "name": "collaboration"}, {"type": "function", "name": "f"}]},
    {"model": "gpt", "input": [{"type": "weird"}]},
]


def responses_sse(items):
    ev = [("response.created", {"type": "response.created", "sequence_number": 0, "response": {"id": "r"}})]
    for i, it in enumerate(items):
        ev.append(("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 0, "output_index": i,
                                                  "item": {**it, "content": []} if it["type"] == "message" else it}))
        if it["type"] == "message":
            text = it["content"][0]["text"]
            for j in range(0, len(text), 4):
                ev.append(("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 0, "item_id": it["id"],
                                                          "output_index": i, "content_index": 0, "delta": text[j:j + 4]}))
            ev.append(("response.output_text.done", {"type": "response.output_text.done", "sequence_number": 0, "item_id": it["id"],
                                                     "output_index": i, "content_index": 0, "text": text}))
            ev.append(("response.content_part.done", {"type": "response.content_part.done", "sequence_number": 0, "item_id": it["id"],
                                                      "part": {"type": "output_text", "text": text}}))
        elif it["type"] == "custom_tool_call":
            for j in range(0, len(it["input"]), 6):
                ev.append(("response.custom_tool_call_input.delta", {"type": "response.custom_tool_call_input.delta", "sequence_number": 0,
                                                                     "item_id": it["id"], "output_index": i, "delta": it["input"][j:j + 6]}))
        ev.append(("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 0, "output_index": i, "item": it}))
    ev.append(("response.completed", {"type": "response.completed", "sequence_number": 0, "response": {"id": "r", "output": items, "usage": {"input_tokens": 3}}}))
    return ev


def main(out):
    store = InMemoryTokenStore()
    seeds = InMemorySeedSource()
    for e in ENTITIES:
        seeds.add_entity("s", SeedEntity(**e))
    store.add_salt("s", SALT)
    engine = ScopeEngine(store, "s", seeds=seeds, detector=patterns.RegexDetector())
    restore = MemoryRestore()
    v = {"salt": list(SALT), "entities": ENTITIES, "messages": [], "streams": [], "responses": [], "rstreams": [], "spell": []}

    async def run():
        for body in MESSAGES:
            try:
                p = await A.prepare_request(body, engine, restore_get=restore.restore_get)
                v["messages"].append({"in": body, "body": p.body, "mapping": list(p.mapping.items()), "withheld": p.withheld,
                                      "restored": p.restored, "spellings": list(p.spellings.items())})
            except A.Refused as exc:
                v["messages"].append({"in": body, "refused": str(exc)})
            except Exception as exc:  # noqa: BLE001
                v["messages"].append({"in": body, "failed": type(exc).__name__})
        p = await A.prepare_request(MESSAGES[1], engine, restore_get=restore.restore_get)
        toks = list(p.mapping)
        blocks = [("text", f"Done: {toks[0]} wrote to {toks[-1]}; see `{p.body['messages'][2]['content'][0]['content'].splitlines()[1]}`.", None),
                  ("tool", {"file_path": p.body["messages"][2]["content"][0]["content"].splitlines()[0]}, "Read"),
                  ("tool", {"query": f"{toks[0]} news"}, "WebSearch"),
                  ("text", f"{toks[0].lower()} and {toks[-1]}", None)]
        restorer = A.StreamRestorer(engine, p.mapping, restore.restore_put, spellings=p.spellings)
        outs = []
        for name, data in sse_of(blocks):
            outs.append([name, data, await restorer.event(name, json.loads(json.dumps(data)))])
        v["streams"].append({"body": MESSAGES[1], "events": outs})
        whole = {"type": "message", "content": [{"type": "text", "text": blocks[0][1]}, {"type": "tool_use", "id": "x", "name": "Read", "input": blocks[1][1]},
                                                {"type": "tool_use", "id": "y", "name": "WebSearch", "input": blocks[2][1]}]}
        v["whole"] = {"in": whole, "out": await A.restore_json_response(json.loads(json.dumps(whole)), engine, p.mapping, restore.restore_put,
                                                                        spellings=p.spellings)}
        # the next turn carries the model's own words back
        nxt = json.loads(json.dumps(MESSAGES[1]))
        nxt["messages"].append({"role": "assistant", "content": [{"type": "text", "text": v["whole"]["out"]["content"][0]["text"]},
                                                                 {"type": "tool_use", "id": "x", "name": "Read", "input": v["whole"]["out"]["content"][1]["input"]}]})
        p2 = await A.prepare_request(nxt, engine, restore_get=restore.restore_get)
        v["messages"].append({"in": nxt, "body": p2.body, "mapping": list(p2.mapping.items()), "withheld": p2.withheld, "restored": p2.restored,
                              "spellings": list(p2.spellings.items())})
        headers = {"Content-Type": "application/json", "x-codex-turn-metadata": json.dumps({"cwd": "/work/Harrowgate Freight Ltd"})}
        for body in RESPONSES:
            try:
                p = await O.prepare_responses(body, engine, headers=headers, restore_get=restore.restore_get, drop=frozenset({"collaboration"}))
                v["responses"].append({"in": body, "headers": list(headers.items()), "body": p.body, "mapping": list(p.mapping.items()),
                                       "out_headers": list(p.headers.items()), "withheld": p.withheld, "restored": p.restored,
                                       "spellings": list(p.spellings.items())})
            except A.Refused as exc:
                v["responses"].append({"in": body, "headers": list(headers.items()), "refused": str(exc)})
        p = await O.prepare_responses(RESPONSES[0], engine, headers=headers, restore_get=restore.restore_get)
        toks = list(p.mapping)
        items = [{"type": "message", "id": "m1", "role": "assistant", "content": [{"type": "output_text", "text": f"Reply to {toks[0]} and {toks[-1]}."}]},
                 {"type": "function_call", "id": "f1", "call_id": "c9", "name": "exec_command", "arguments": json.dumps({"cmd": f"ls '/work/{toks[1]}'"})},
                 {"type": "function_call", "id": "f2", "call_id": "c8", "name": "search", "namespace": "mcp__web", "arguments": json.dumps({"q": toks[0]})},
                 {"type": "custom_tool_call", "id": "p1", "call_id": "c7", "name": "apply_patch", "input": f"+{toks[0]} line"},
                 {"type": "custom_tool_call", "id": "p2", "call_id": "c6", "name": "exec", "input": f"text(await tools.exec_command({{cmd: \"echo {toks[0]}\"}}))"}]
        restorer = O.ResponsesRestorer(engine, p.mapping, restore.restore_put, spellings=p.spellings)
        outs = []
        for name, data in responses_sse(items):
            outs.append([name, data, await restorer.event(name, json.loads(json.dumps(data)))])
        v["rstreams"].append({"body": RESPONSES[0], "headers": list(headers.items()), "events": outs})
        v["restore"] = list(restore._d.items())

    asyncio.run(run())
    Path(out, "proxy.json").write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
    print(len(v["messages"]), "requests,", len(v["streams"][0]["events"]), "+", len(v["rstreams"][0]["events"]), "events")


if __name__ == "__main__":
    main(sys.argv[1])
