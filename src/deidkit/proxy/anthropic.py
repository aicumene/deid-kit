# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Anthropic's Messages API through deid-kit: the request out in tokens, the answer back in names.

**Out.** Every string that came from the user's side is tokenised: ``system`` text, the text of
each message, the content of ``tool_result`` blocks, and the string values of ``tool_use`` inputs
and earlier assistant text, which reached the user's side detokenised. Left exactly as they are:
the model's reasoning (``thinking``, ``redacted_thinking``), blocks the provider produced itself
(server tools and their results), tool references, tool definitions, and the first system block
when it is the client's attribution block. Images and non-text documents cannot be tokenised: by
default they are replaced with a short note, or the request is refused. A block type this module
does not know refuses the request, since it cannot tell whether it carries text.

**Back.** Text is de-tokenised as it streams, holding back a tail that could be the start of a
token split across two events. A tool call's arguments are collected until the call is complete,
de-tokenised when the tool acts on the user's machine, and emitted in one piece; other tools (web
fetch, MCP servers, subagents) keep the tokens. Reasoning passes untouched.

**The model's own words.** Each text block and local tool call is remembered as the model wrote it,
keyed by a hash of the version handed to the agent. When that block comes back as history, the
original is sent again, so the history is byte-identical to what the model produced.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from deidkit import namefold as nf
from deidkit.proxy.engine import ScopeEngine

#: Tools that act on the user's machine: their arguments get the real values back.
LOCAL_TOOLS = frozenset({
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead", "Bash", "BashOutput",
    "KillShell", "KillBash", "Monitor", "Glob", "Grep", "LS", "TodoWrite", "AskUserQuestion",
    "ExitPlanMode",
})
#: Blocks passed through untouched: produced by the model or the provider, or carrying no text.
_UNTOUCHED = frozenset({
    "thinking", "redacted_thinking", "tool_reference", "server_tool_use", "web_search_tool_result",
    "web_fetch_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result", "mcp_tool_use", "mcp_tool_result",
    "container_upload",
})
_BINARY = frozenset({"image", "document"})
_ATTRIBUTION_PREFIX = "x-anthropic-billing-header"


class Refused(ValueError):
    """The request cannot be de-identified; nothing is sent."""


def _key(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass
class _Slot:
    """A string in the request to tokenise, and where to put the result."""
    holder: dict | list
    key: str | int


@dataclass
class Prepared:
    body: dict
    mapping: dict[str, str]
    withheld: list[str] = field(default_factory=list)
    restored: int = 0


def _strings(obj, slots: list[_Slot]) -> None:
    """Every string value inside a JSON value (keys untouched)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                slots.append(_Slot(obj, k))
            else:
                _strings(v, slots)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                slots.append(_Slot(obj, i))
            else:
                _strings(v, slots)


def _withhold(block: dict, what: str, binary: str, withheld: list[str]) -> dict:
    if binary == "refuse":
        raise Refused(f"the request carries a {what} block; deid-kit de-identifies text only. "
                      "Convert it to text on this machine first.")
    withheld.append(what)
    note = {"type": "text", "text": f"[{what} withheld by the de-identification proxy: "
                                    "only text is sent to the model]"}
    if "cache_control" in block:
        note["cache_control"] = block["cache_control"]
    return note


def _block(blocks: list, i: int, slots: list[_Slot], binary: str, withheld: list[str],
           where: str) -> None:
    """One content block, in place: its strings become slots, or it is withheld or refused."""
    block = blocks[i]
    if not isinstance(block, dict):
        raise Refused(f"unexpected content in {where}")
    t = block.get("type")
    if t == "text":
        slots.append(_Slot(block, "text"))
    elif t in _UNTOUCHED:
        return
    elif t == "document" and (block.get("source") or {}).get("type") == "text":
        slots.append(_Slot(block["source"], "data"))
        for k in ("title", "context"):
            if isinstance(block.get(k), str):
                slots.append(_Slot(block, k))
    elif t in _BINARY:
        blocks[i] = _withhold(block, t, binary, withheld)
    elif t == "search_result":
        for k in ("title", "source"):
            if isinstance(block.get(k), str):
                slots.append(_Slot(block, k))
        inner = block.get("content") or []
        for j in range(len(inner)):
            _block(inner, j, slots, binary, withheld, where)
    else:
        raise Refused(f"unknown content block type {t!r} in {where}")


async def prepare_request(body: dict, engine: ScopeEngine, *, restore_get: Callable[[str], str | None],
                          binary: str = "withhold") -> Prepared:
    """The request as it may cross: tokenised, with the mapping to read the answer back."""
    body = copy.deepcopy(body)
    slots: list[_Slot] = []
    withheld: list[str] = []
    restores: list[tuple[_Slot, str]] = []       # (slot, original) — applied after tokenising

    system = body.get("system")
    if isinstance(system, str):
        slots.append(_Slot(body, "system"))
    elif isinstance(system, list):
        for i, block in enumerate(system):
            if i == 0 and isinstance(block, dict) and \
                    str(block.get("text", "")).startswith(_ATTRIBUTION_PREFIX):
                continue
            _block(system, i, slots, binary, withheld, "system")

    for m, msg in enumerate(body.get("messages") or []):
        content = msg.get("content")
        role = msg.get("role")
        if isinstance(content, str):
            slots.append(_Slot(msg, "content"))
            continue
        if not isinstance(content, list):
            raise Refused(f"message {m} has no content list")
        for i, block in enumerate(content):
            t = block.get("type") if isinstance(block, dict) else None
            if t == "tool_use":
                inner: list[_Slot] = []
                _strings(block.get("input", {}), inner)
                slots.extend(inner)
                original = restore_get(_key("u:", _canonical(block.get("input", {}))))
                if role == "assistant" and original is not None:
                    restores.append((_Slot(block, "input"), original))
            elif t == "tool_result":
                c = block.get("content")
                if isinstance(c, str):
                    slots.append(_Slot(block, "content"))
                elif isinstance(c, list):
                    for j in range(len(c)):
                        _block(c, j, slots, binary, withheld, f"message {m} tool_result")
            else:
                _block(content, i, slots, binary, withheld, f"message {m}")
                if role == "assistant" and t == "text":
                    original = restore_get(_key("t:", block.get("text", "")))
                    if original is not None:
                        restores.append((_Slot(block, "text"), original))

    texts = [s.holder[s.key] for s in slots]
    reds = await engine.tokenize_all(texts)
    for slot, red in zip(slots, reds):
        slot.holder[slot.key] = red.text
    for slot, original in restores:
        slot.holder[slot.key] = json.loads(original) if slot.key == "input" else original

    sent = json.dumps(body, ensure_ascii=False)
    mapping = await engine.mapping_for(reds, sent)
    return Prepared(body=body, mapping=mapping, withheld=withheld, restored=len(restores))


async def _detokenize_obj(obj, engine: ScopeEngine, mapping: dict[str, str]):
    if isinstance(obj, str):
        return await engine.detokenize(obj, mapping)
    if isinstance(obj, list):
        return [await _detokenize_obj(v, engine, mapping) for v in obj]
    if isinstance(obj, dict):
        return {k: await _detokenize_obj(v, engine, mapping) for k, v in obj.items()}
    return obj


async def restore_json_response(data: dict, engine: ScopeEngine, mapping: dict[str, str],
                                restore_put: Callable[[str, str], None],
                                local_tools: frozenset[str] = LOCAL_TOOLS) -> dict:
    """A non-streamed answer, back in names."""
    for block in data.get("content") or []:
        t = block.get("type")
        if t == "text":
            original = block.get("text", "")
            block["text"] = await engine.detokenize(original, mapping)
            restore_put(_key("t:", block["text"]), original)
        elif t == "tool_use" and block.get("name") in local_tools:
            original = block.get("input", {})
            block["input"] = await _detokenize_obj(original, engine, mapping)
            restore_put(_key("u:", _canonical(block["input"])), json.dumps(original, ensure_ascii=False))
    return data


@dataclass
class _Block:
    type: str
    local: bool = False
    orig: str = ""
    emitted_orig: int = 0
    emitted: str = ""
    parts: list[str] = field(default_factory=list)


class StreamRestorer:
    """Rewrites a Messages API event stream from tokens back to names, event by event."""

    def __init__(self, engine: ScopeEngine, mapping: dict[str, str],
                 restore_put: Callable[[str, str], None],
                 local_tools: frozenset[str] = LOCAL_TOOLS) -> None:
        self.engine = engine
        self.mapping = mapping
        self.restore_put = restore_put
        self.local_tools = local_tools
        self.prefixes = {t.casefold()[:i] for t in mapping for i in range(1, len(t) + 1)}
        self.longest = max((len(t) for t in mapping), default=0)
        self.blocks: dict[int, _Block] = {}

    def _holdback(self, text: str) -> int:
        for n in range(min(len(text), self.longest), 0, -1):
            if nf.deconfuse_ascii(text[-n:]).casefold() in self.prefixes:
                return n
        return 0

    async def _text(self, b: _Block, final: bool) -> str:
        end = len(b.orig) if final else len(b.orig) - self._holdback(b.orig)
        if end <= b.emitted_orig:
            return ""
        full = await self.engine.detokenize(b.orig[:end], self.mapping)
        if not full.startswith(b.emitted):             # cannot happen: replacements are local
            raise RuntimeError("de-tokenised prefix moved")
        out, b.emitted, b.emitted_orig = full[len(b.emitted):], full, end
        return out

    async def event(self, name: str, data: dict) -> list[tuple[str, dict]]:
        t = data.get("type")
        if t == "content_block_start":
            cb = data.get("content_block") or {}
            b = _Block(type=cb.get("type", ""))
            self.blocks[data["index"]] = b
            if b.type == "text" and cb.get("text"):
                b.orig = cb["text"]
                data = {**data, "content_block": {**cb, "text": ""}}
                first = await self._text(b, final=False)
                out = [(name, data)]
                if first:
                    out.append(("content_block_delta", {"type": "content_block_delta",
                                "index": data["index"], "delta": {"type": "text_delta", "text": first}}))
                return out
            if b.type == "tool_use" and cb.get("name") in self.local_tools:
                b.local = True
            return [(name, data)]
        if t == "content_block_delta":
            b = self.blocks.get(data.get("index"))
            d = data.get("delta") or {}
            if b and b.type == "text" and d.get("type") == "text_delta":
                b.orig += d.get("text", "")
                out = await self._text(b, final=False)
                return [(name, {**data, "delta": {**d, "text": out}})] if out else []
            if b and b.local and d.get("type") == "input_json_delta":
                b.parts.append(d.get("partial_json", ""))
                return []
            return [(name, data)]
        if t == "content_block_stop":
            idx = data.get("index")
            b = self.blocks.get(idx)
            out: list[tuple[str, dict]] = []
            if b and b.type == "text":
                rest = await self._text(b, final=True)
                if rest:
                    out.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                        "delta": {"type": "text_delta", "text": rest}}))
                self.restore_put(_key("t:", b.emitted), b.orig)
            elif b and b.local:
                raw = "".join(b.parts)
                try:
                    original = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    original = None
                if original is None:
                    new = raw                    # tokens only: safe, if not useful
                else:
                    restored = await _detokenize_obj(original, self.engine, self.mapping)
                    new = json.dumps(restored, ensure_ascii=False)
                    self.restore_put(_key("u:", _canonical(restored)),
                                     json.dumps(original, ensure_ascii=False))
                out.append(("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "input_json_delta",
                                                              "partial_json": new}}))
            out.append((name, data))
            return out
        return [(name, data)]


__all__ = ["LOCAL_TOOLS", "Prepared", "Refused", "StreamRestorer", "prepare_request",
           "restore_json_response"]
