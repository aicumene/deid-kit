# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Tolerant JSON extraction from LLM output (handles code fences, surrounding prose, and
output truncated mid-structure by the model's token budget)."""

from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in ``text``; return {} if none/invalid.

    Tolerant of (a) code fences / surrounding prose and (b) truncation — when the model
    runs out of ``num_predict`` mid-object the JSON is incomplete; we salvage the longest
    valid prefix by closing the still-open containers. Entity-rich documents routinely
    overflow the token budget, so without this the whole extraction would be discarded."""
    text = _strip_fences(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    # Either no closing brace at all, or the substring didn't parse (truncation). Salvage
    # from the first '{' to the end of the string.
    if start != -1:
        repaired = _repair_truncated_json(text[start:])
        if isinstance(repaired, dict):
            return repaired
    return {}


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        # drop the opening fence line (``` or ```json) and any trailing fence
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _repair_truncated_json(s: str):
    """Best-effort parse of JSON that was cut off mid-structure.

    Single pass tracking string state and a container stack. We remember the last position
    at which the document-so-far was a *complete value* (safe to truncate), then close the
    open containers there and parse. Returns the parsed object/array, or None."""
    stack: list[list] = []  # each entry: ['obj'|'arr', state]
    in_str = False
    esc = False
    last_safe_idx: int | None = None
    last_safe_types: list[str] | None = None

    def mark_safe(idx: int) -> None:
        nonlocal last_safe_idx, last_safe_types
        last_safe_idx = idx
        last_safe_types = [c[0] for c in stack]

    def value_completed(idx: int) -> None:
        # A value just finished; update the enclosing container's state and mark a safe cut.
        if stack:
            top = stack[-1]
            if top[0] == "obj":
                top[1] = "value_done"
                mark_safe(idx)
            else:  # arr
                top[1] = "value_done"
                mark_safe(idx)
        else:
            mark_safe(idx)

    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
                # String finished. In an object it may be a key (not a safe cut) or a value.
                if stack and stack[-1][0] == "obj" and stack[-1][1] == "expect_key":
                    stack[-1][1] = "key_done"
                else:
                    value_completed(i + 1)
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c in " \t\r\n":
            i += 1
            continue
        if c == "{":
            stack.append(["obj", "expect_key"])
            i += 1
            continue
        if c == "[":
            stack.append(["arr", "expect_value"])
            i += 1
            continue
        if c in "}]":
            if stack:
                stack.pop()
            value_completed(i + 1)
            i += 1
            continue
        if c == ":":
            if stack and stack[-1][0] == "obj":
                stack[-1][1] = "expect_value"
            i += 1
            continue
        if c == ",":
            if stack:
                stack[-1][1] = "expect_key" if stack[-1][0] == "obj" else "expect_value"
            i += 1
            continue
        # Bare literal: number / true / false / null. Scan to its end.
        j = i
        while j < n and s[j] not in ',}]: \t\r\n"':
            j += 1
        if j < n:
            value_completed(j)
            i = j
        else:
            break  # literal runs off the end → truncated; stop here
    if last_safe_idx is None or last_safe_types is None:
        return None
    closers = "".join("}" if t == "obj" else "]" for t in reversed(last_safe_types))
    try:
        return json.loads(s[:last_safe_idx] + closers)
    except json.JSONDecodeError:
        return None
