# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A scripted ACP agent for tests: answers, asks to write a file inside its folder and to touch
a path outside it, and writes the file when allowed. It offers two models as a config option and
names the one it works on in its answer. It reports its window (usage_update) and what the turn
cost (usage on its answer), as ACP has them. Given a name (its first argument), it says it in its
answer: the tests that switch agents tell them apart by it."""

import json
import os
import sys

pending = {}
next_id = 1000
MODELS = [{"value": "default", "name": "Default", "description": "The agent's own default"},
          {"value": "fast", "name": "Fast", "description": "Quicker, lighter"}]
model = "default"
NAME = sys.argv[1] if len(sys.argv) > 1 else None


def config_options():
    return [{"id": "model", "name": "Model", "category": "model", "type": "select",
             "currentValue": model, "options": MODELS}]


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def ask(method, params):
    global next_id
    next_id += 1
    send({"jsonrpc": "2.0", "id": next_id, "method": method, "params": params})
    while True:
        msg = json.loads(sys.stdin.readline())
        if msg.get("id") == next_id and "result" in msg:
            return msg["result"]


def update(session, u):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session, "update": u}})


def permission(session, cid, title, path):
    options = [{"optionId": "allow", "name": "Allow", "kind": "allow_once"},
               {"optionId": "reject", "name": "Reject", "kind": "reject_once"}]
    result = ask("session/request_permission", {"sessionId": session, "toolCall": {
        "toolCallId": cid, "title": title, "kind": "edit", "locations": [{"path": path}]},
        "options": options})
    return (result.get("outcome") or {}).get("optionId") or "cancelled"


cwd = None
meta = None
for line in sys.stdin:
    msg = json.loads(line)
    method, params, rid = msg.get("method"), msg.get("params") or {}, msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "agentInfo": {"name": NAME or "fake"}}})
    elif method == "session/new":
        cwd = params["cwd"]
        meta = params.get("_meta")
        send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": "s1", "configOptions": config_options()}})
    elif method == "session/set_config_option":
        if params.get("configId") == "model" and params.get("value") in {m["value"] for m in MODELS}:
            model = params["value"]
            send({"jsonrpc": "2.0", "id": rid, "result": {"configOptions": config_options()}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "invalid value"}})
    elif method == "session/prompt":
        said = (f"[{NAME}] " if NAME else "") + (f"[meta {json.dumps(meta, sort_keys=True)}] " if meta else "")
        update("s1", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": f"{said}Working on it ({model}). "}})
        target = os.path.join(cwd, "draft.md")
        update("s1", {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Write draft.md",
                      "kind": "edit", "status": "pending", "locations": [{"path": target}]})
        first = permission("s1", "t1", "Write draft.md", target)
        if first == "allow":
            with open(target, "w") as fh:
                fh.write("Draft for the client.\n")
            update("s1", {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"})
        second = permission("s1", "t2", "Edit a system file", "/etc/hosts")
        update("s1", {"sessionUpdate": "agent_message_chunk",
                      "content": {"type": "text", "text": f"first: {first}; second: {second}."}})
        update("s1", {"sessionUpdate": "usage_update", "used": 14200, "size": 32768})
        send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn", "usage": {
            "inputTokens": 3000, "cachedReadTokens": 11000, "cachedWriteTokens": 0,
            "outputTokens": 420, "totalTokens": 14420, "costCents": "n/a"}}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "not supported"}})
