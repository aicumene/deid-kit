# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A scripted ACP agent for tests: answers, asks to write a file inside its folder and to touch
a path outside it, and writes the file when allowed. It offers two models as a config option and
names the one it works on in its answer. It reports its window (usage_update) and what the turn
cost (usage on its answer), as ACP has them. Given a name (its first argument), it says it in its
answer: the tests that switch agents tell them apart by it. A ``session/cancel`` ends the turn at its
next step with ``cancelled``. With ``echo-prompt`` among its arguments it also says which turn of its
session this is and the text blocks of the prompt it was given. With ``review`` among them it answers a
review task (deidkit.agents.review) with notes: one on the document's third line, its apostrophes curled
the way models write them, and one on words the document does not have."""

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


cancelled = False
turns = 0
ECHO = "echo-prompt" in sys.argv[2:]


def ask(method, params):
    global next_id, cancelled
    next_id += 1
    send({"jsonrpc": "2.0", "id": next_id, "method": method, "params": params})
    while True:
        msg = json.loads(sys.stdin.readline())
        if msg.get("method") == "session/cancel":
            cancelled = True
            continue
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
mcp = None
for line in sys.stdin:
    msg = json.loads(line)
    method, params, rid = msg.get("method"), msg.get("params") or {}, msg.get("id")
    if method == "session/cancel":
        cancelled = True                         # between turns: nothing to stop
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "agentInfo": {"name": NAME or "fake"}}})
    elif method == "session/new":
        cwd = params["cwd"]
        meta = params.get("_meta")
        mcp = params.get("mcpServers")
        send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": "s1", "configOptions": config_options()}})
    elif method == "session/set_config_option":
        if params.get("configId") == "model" and params.get("value") in {m["value"] for m in MODELS}:
            model = params["value"]
            send({"jsonrpc": "2.0", "id": rid, "result": {"configOptions": config_options()}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "invalid value"}})
    elif method == "session/prompt":
        cancelled = False
        turns += 1
        asked = " ".join(b.get("text") or "" for b in params.get("prompt") or [])
        if "review" in sys.argv[2:] and "Review the document" in asked:
            source = asked.split('Its text is in the file "', 1)[1].split('"', 1)[0]
            with open(os.path.join(cwd, source), encoding="utf-8") as fh:
                line = fh.read().split("\n")[2]
            notes = {"notes": [
                {"quote": line[:70].replace("'", "’"), "kind": "risk", "aspect": "termination", "importance": 4,
                 "clause_type": "right", "who": "the Tenant", "what": "break the lease",
                 "norm": "Housing Act 1988 s. 21", "note": "A break right on short notice.",
                 "suggestion": "The Tenant may not break the lease"},
                {"quote": "Words this document never had.", "kind": "question", "aspect": "payment",
                 "importance": 2, "note": "Invented."}]}
            update("s1", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "I read it {first}. "}})
            update("s1", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": json.dumps(notes)}})
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
            continue
        said = (f"[{NAME}] " if NAME else "") + (f"[meta {json.dumps(meta, sort_keys=True)}] " if meta else "")
        if ECHO:
            said += f"[turn {turns}] [prompt {json.dumps([b.get('text') for b in params.get('prompt') or []])}] "
        if mcp:
            said += f"[mcp {json.dumps(mcp, sort_keys=True)}] "
        if sys.argv[2:]:
            said += f"[argv {json.dumps(sys.argv[2:])}] "
        if os.environ.get("CODEX_PATH") and os.path.isfile(os.environ["CODEX_PATH"]):
            with open(os.environ["CODEX_PATH"]) as fh:
                said += f"[wrapper {json.dumps(fh.read())}] "
        if os.environ.get("MCP_TOOL_TIMEOUT"):
            said += f"[mcp-timeout {os.environ['MCP_TOOL_TIMEOUT']}] "
        if os.environ.get("MODEL_PROVIDER"):
            said += (f"[provider {os.environ['MODEL_PROVIDER']} {os.environ.get('CODEX_CONFIG', '')} "
                     f"{os.environ.get('CODEX_PATH', '-')} {os.environ.get('ANTHROPIC_API_KEY', '-')} "
                     f"{os.environ.get('INITIAL_AGENT_MODE', '-')}] ")
        update("s1", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": f"{said}Working on it ({model}). "}})
        target = os.path.join(cwd, "draft.md")
        update("s1", {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Write draft.md",
                      "kind": "edit", "status": "pending", "locations": [{"path": target}]})
        first = permission("s1", "t1", "Write draft.md", target)
        if cancelled:
            update("s1", {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "failed"})
            send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "cancelled"}})
            continue
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
