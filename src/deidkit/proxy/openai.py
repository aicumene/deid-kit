# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""OpenAI's Responses API through deid-kit, as Codex uses it.

**Out.** Every string from the user's side is tokenised: ``instructions``, the text of every
message (developer, user, and earlier assistant text, which reached the user's side detokenised),
the arguments of earlier function calls and the input of custom tool calls (``apply_patch``
patches), tool outputs, and the request metadata, which can name the repository's path and git
remote. Left as they are: ``reasoning`` and ``compaction`` items (produced by the model from
tokens, and encrypted), the tool list (``additional_tools``), hosted tool calls and item
references. Images and files are replaced with a note, or refuse the request. An item type this
module does not know refuses the request.

**Back.** Codex builds its history and runs tools from the complete item in
``response.output_item.done``; the deltas are for display. The proxy de-tokenises both: text and
patch deltas with a held-back tail, and each completed item, where the arguments of local tools
(``exec_command``, ``write_stdin``, ``apply_patch``, …) get real values and other tools (MCP
servers, hosted search) keep the tokens. In code mode the model writes one ``exec`` script that
calls tools (``text(await tools.apply_patch("…"))``): the script gets real values only when every
tool it calls is local, and the values are escaped for the string literal they land in. Every event it emits is renumbered, so the sequence stays
gapless when a flushed tail adds an event.

**The model's own words.** Each assistant text, function-call argument string and patch is
remembered as the model wrote it, keyed by a hash of the version handed to Codex, and sent back
byte for byte when it returns as history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from deidkit import namefold as nf
from deidkit.proxy.anthropic import Refused, _Slot, _strings
from deidkit.proxy.engine import ScopeEngine
from deidkit.proxy.spelling import spellings as _spellings

#: Tools that act on the user's machine: their arguments get the real values back.
LOCAL_TOOLS = frozenset({"exec_command", "write_stdin", "shell", "local_shell", "apply_patch",
                         "update_plan", "view_image", "read_file", "list_dir", "grep_files"})
_UNTOUCHED = frozenset({"reasoning", "compaction", "compaction_summary", "additional_tools",
                        "web_search_call", "item_reference", "image_generation_call",
                        "tool_search_call", "tool_search_output"})
#: Code mode: one custom tool whose input is a script calling ``tools.<name>(…)``.
CODE_TOOL = "exec"
_SCRIPT_CALL = re.compile(r"\btools\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)")
_TEXT_PARTS = frozenset({"input_text", "output_text", "text", "summary_text"})
_BINARY_PARTS = frozenset({"input_image", "input_file", "input_audio", "image", "file"})


def _key(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class PreparedResponses:
    body: dict
    mapping: dict[str, str]
    headers: dict[str, str] = field(default_factory=dict)
    withheld: list[str] = field(default_factory=list)
    restored: int = 0
    #: File and folder names as they were written, for a tool call's arguments.
    spellings: dict[str, str] = field(default_factory=dict)


def _note(part: dict, binary: str, withheld: list[str]) -> dict:
    what = part.get("type", "file")
    if binary == "refuse":
        raise Refused(f"the request carries an {what} part; deid-kit de-identifies text only. "
                      "Convert it to text on this machine first.")
    withheld.append(what)
    return {"type": "input_text", "text": f"[{what} withheld by the de-identification proxy: "
                                          "only text is sent to the model]"}


def _parts(parts: list, slots: list[_Slot], binary: str, withheld: list[str], where: str,
           restores: list | None = None, restore_get: Callable | None = None) -> None:
    for i, part in enumerate(parts):
        if isinstance(part, str):
            slots.append(_Slot(parts, i))
            continue
        t = part.get("type")
        if t in _TEXT_PARTS:
            slots.append(_Slot(part, "text"))
            if restores is not None and t == "output_text":
                original = restore_get(_key("t:", part.get("text", "")))
                if original is not None:
                    restores.append((_Slot(part, "text"), original))
        elif t == "refusal":
            slots.append(_Slot(part, "refusal"))
        elif t in _BINARY_PARTS:
            parts[i] = _note(part, binary, withheld)
        else:
            raise Refused(f"unknown content part {t!r} in {where}")


def _output(holder: dict, key: str, slots, binary, withheld, where) -> None:
    out = holder.get(key)
    if isinstance(out, str):
        slots.append(_Slot(holder, key))
    elif isinstance(out, list):
        _parts(out, slots, binary, withheld, where)
    elif out is not None:
        raise Refused(f"unexpected output in {where}")


async def prepare_responses(body: dict, engine: ScopeEngine, *, headers: dict[str, str],
                            restore_get: Callable[[str], str | None],
                            binary: str = "withhold") -> PreparedResponses:
    """The request as it may cross, the headers that carry metadata, and the mapping back."""
    body = copy.deepcopy(body)
    headers = dict(headers)
    slots: list[_Slot] = []
    withheld: list[str] = []
    restores: list[tuple[_Slot, str]] = []

    if isinstance(body.get("instructions"), str):
        slots.append(_Slot(body, "instructions"))
    if isinstance(body.get("client_metadata"), dict):
        _strings(body["client_metadata"], slots)
    for name in list(headers):
        if name.lower() == "x-codex-turn-metadata":
            slots.append(_Slot(headers, name))

    items = body.get("input")
    if isinstance(items, str):
        slots.append(_Slot(body, "input"))
        items = []
    for n, item in enumerate(items or []):
        if not isinstance(item, dict):
            raise Refused(f"unexpected input item {n}")
        t = item.get("type", "message" if "role" in item else None)
        where = f"input item {n} ({t})"
        if t == "message":
            content = item.get("content")
            if isinstance(content, str):
                slots.append(_Slot(item, "content"))
            elif isinstance(content, list):
                assistant = item.get("role") == "assistant"
                _parts(content, slots, binary, withheld, where,
                       restores if assistant else None, restore_get)
        elif t == "function_call":
            args = item.get("arguments", "")
            original = restore_get(_key("f:", args))
            if original is not None:
                restores.append((_Slot(item, "arguments"), original))
            else:
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    slots.append(_Slot(item, "arguments"))       # not JSON: tokenise as text
                else:
                    inner: list[_Slot] = []
                    _strings(parsed, inner)
                    # Tokenised inside the parsed arguments, re-serialised after tokenising.
                    item["arguments"] = parsed
                    slots.extend(inner)
        elif t == "custom_tool_call":
            original = restore_get(_key("c:", item.get("input", "")))
            if original is not None:
                restores.append((_Slot(item, "input"), original))
            else:
                slots.append(_Slot(item, "input"))
        elif t in ("function_call_output", "custom_tool_call_output", "local_shell_call_output",
                   "mcp_tool_call_output"):
            _output(item, "output", slots, binary, withheld, where)
        elif t == "local_shell_call":
            _strings(item.get("action", {}), slots)
        elif t in _UNTOUCHED:
            continue
        else:
            raise Refused(f"unknown input item type {t!r}")

    texts = [s.holder[s.key] for s in slots]
    reds = await engine.tokenize_all(texts)
    for slot, red in zip(slots, reds):
        slot.holder[slot.key] = red.text
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "function_call" and \
                not isinstance(item.get("arguments"), str):
            item["arguments"] = _compact(item["arguments"])
    for slot, original in restores:
        slot.holder[slot.key] = original

    sent = json.dumps(body, ensure_ascii=False) + json.dumps(headers, ensure_ascii=False)
    mapping = await engine.mapping_for(reds, sent)
    return PreparedResponses(body=body, mapping=mapping, headers=headers, withheld=withheld,
                             restored=len(restores), spellings=_spellings(texts, reds))


async def _detok(obj, engine: ScopeEngine, mapping, spellings=None):
    if isinstance(obj, str):
        return await engine.detokenize(obj, mapping, spellings)
    if isinstance(obj, list):
        return [await _detok(v, engine, mapping, spellings) for v in obj]
    if isinstance(obj, dict):
        return {k: await _detok(v, engine, mapping, spellings) for k, v in obj.items()}
    return obj


def script_is_local(script: str, local_tools: frozenset[str]) -> bool:
    """Does a code-mode script call only tools that act on this machine?"""
    calls = _SCRIPT_CALL.findall(script or "")
    return bool(calls) and all(c in local_tools for c in calls)


def literal_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """The mapping with each value escaped for a string literal (a script's, a JSON string's): a
    name with a quote or a backslash must not end the literal it is put into."""
    return {t: json.dumps(v, ensure_ascii=False)[1:-1] for t, v in mapping.items()}


async def restore_item(item: dict, engine: ScopeEngine, mapping: dict[str, str],
                       restore_put: Callable[[str, str], None] | None,
                       local_tools: frozenset[str] = LOCAL_TOOLS,
                       spellings: dict[str, str] | None = None) -> dict:
    """A completed output item, back in names (and remembered as the model wrote it)."""
    item = copy.deepcopy(item)
    t = item.get("type")
    if t == "message":
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                original = part.get("text", "")
                part["text"] = await engine.detokenize(original, mapping)
                if restore_put:
                    restore_put(_key("t:", part["text"]), original)
    elif t == "function_call" and item.get("name") in local_tools:
        original = item.get("arguments", "")
        try:
            restored = _compact(await _detok(json.loads(original), engine, mapping, spellings))
        except json.JSONDecodeError:
            restored = await engine.detokenize(original, mapping, spellings)
        item["arguments"] = restored
        if restore_put:
            restore_put(_key("f:", restored), original)
    elif t == "custom_tool_call" and (
            item.get("name") in local_tools or
            (item.get("name") == CODE_TOOL and script_is_local(item.get("input", ""), local_tools))):
        original = item.get("input", "")
        code = item.get("name") == CODE_TOOL
        values = literal_mapping(mapping) if code else mapping
        item["input"] = await engine.detokenize(original, values, spellings, literal=code)
        if restore_put:
            restore_put(_key("c:", item["input"]), original)
    return item


@dataclass
class _Stream:
    orig: str = ""
    emitted_orig: int = 0
    emitted: str = ""


class ResponsesRestorer:
    """Rewrites a Responses API event stream from tokens back to names, event by event."""

    def __init__(self, engine: ScopeEngine, mapping: dict[str, str],
                 restore_put: Callable[[str, str], None],
                 local_tools: frozenset[str] = LOCAL_TOOLS,
                 spellings: dict[str, str] | None = None) -> None:
        self.engine = engine
        self.mapping = mapping
        self.restore_put = restore_put
        self.local_tools = local_tools
        self.spellings = spellings or {}
        self.prefixes = {t.casefold()[:i] for t in mapping for i in range(1, len(t) + 1)}
        self.longest = max((len(t) for t in mapping), default=0)
        self.streams: dict[tuple, _Stream] = {}
        self.local_items: set[str] = set()
        self.seq = 0

    def _holdback(self, text: str) -> int:
        for n in range(min(len(text), self.longest), 0, -1):
            if nf.deconfuse_ascii(text[-n:]).casefold() in self.prefixes:
                return n
        return 0

    async def _advance(self, s: _Stream, final: bool) -> str:
        end = len(s.orig) if final else len(s.orig) - self._holdback(s.orig)
        if end <= s.emitted_orig:
            return ""
        full = await self.engine.detokenize(s.orig[:end], self.mapping)
        if not full.startswith(s.emitted):
            raise RuntimeError("de-tokenised prefix moved")
        out, s.emitted, s.emitted_orig = full[len(s.emitted):], full, end
        return out

    def _numbered(self, data: dict) -> dict:
        if "sequence_number" in data:
            data = {**data, "sequence_number": self.seq}
        self.seq += 1
        return data

    async def event(self, name: str, data: dict) -> list[tuple[str, dict]]:
        return [(n, self._numbered(d)) for n, d in await self._event(name, data)]

    async def _event(self, name: str, data: dict) -> list[tuple[str, dict]]:
        t = data.get("type", "")
        if t == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") in ("function_call", "custom_tool_call") and \
                    (item.get("name") in self.local_tools or item.get("name") == CODE_TOOL):
                # For code mode the script is judged when it is complete (output_item.done);
                # its deltas are only shown on this machine, so they are shown in names.
                self.local_items.add(item.get("id") or str(data.get("output_index")))
            return [(name, data)]
        if t in ("response.output_text.delta", "response.custom_tool_call_input.delta"):
            if t == "response.custom_tool_call_input.delta" and \
                    (data.get("item_id") or str(data.get("output_index"))) not in self.local_items:
                return [(name, data)]
            key = (t, data.get("item_id"), data.get("content_index"))
            s = self.streams.setdefault(key, _Stream())
            s.orig += data.get("delta", "")
            out = await self._advance(s, final=False)
            return [(name, {**data, "delta": out})] if out else []
        if t in ("response.output_text.done", "response.custom_tool_call_input.done"):
            field_name = "text" if t == "response.output_text.done" else "input"
            delta_type = t.replace(".done", ".delta")
            key = (delta_type, data.get("item_id"), data.get("content_index"))
            s = self.streams.pop(key, None)
            out: list[tuple[str, dict]] = []
            if s is not None:
                rest = await self._advance(s, final=True)
                if rest:
                    delta = {k: v for k, v in data.items() if k not in ("text", "input")}
                    out.append((name.replace(".done", ".delta") if name else "",
                                {**delta, "type": delta_type, "delta": rest}))
            if isinstance(data.get(field_name), str) and (
                    t == "response.output_text.done" or s is not None):
                data = {**data, field_name: await self.engine.detokenize(
                    data[field_name], self.mapping,
                    self.spellings if field_name == "input" else None)}
            return out + [(name, data)]
        if t == "response.content_part.done":
            part = data.get("part") or {}
            if part.get("type") == "output_text":
                part = {**part, "text": await self.engine.detokenize(part.get("text", ""),
                                                                     self.mapping)}
                data = {**data, "part": part}
            return [(name, data)]
        if t == "response.output_item.done":
            item = await restore_item(data.get("item") or {}, self.engine, self.mapping,
                                      self.restore_put, self.local_tools, self.spellings)
            return [(name, {**data, "item": item})]
        if t == "response.completed" and isinstance(data.get("response"), dict):
            resp = dict(data["response"])
            if isinstance(resp.get("output"), list):
                resp["output"] = [await restore_item(i, self.engine, self.mapping, None,
                                                     self.local_tools, self.spellings)
                                  for i in resp["output"]]
            return [(name, {**data, "response": resp})]
        return [(name, data)]


__all__ = ["LOCAL_TOOLS", "PreparedResponses", "ResponsesRestorer", "prepare_responses",
           "restore_item"]
