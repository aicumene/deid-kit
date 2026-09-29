# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-proxy check``: did any known value cross? Read the proxy's record and look.

Every string the proxy sent is decoded first: JSON inside a text field is parsed (a coding
agent's tool output often is JSON), so an escape such as ``\\n`` becomes the line break it
stands for. Then every surface the vault knows for the scope is searched as a plain substring,
case-insensitive. A search that respects word boundaries would miss a name glued to an escape.
That miss is how a leak went unnoticed once.

The report gives counts per token kind, never a value:

* **standalone** — the value stands as a word of its own: a leak. The exit status is 1.
* **inside other words** — the value occurs only inside a longer word (``filename`` contains
  ``lena``). The containing word is shown with the value masked, so a person can tell noise from
  a leak without seeing the value.

Surfaces shorter than three characters (initials) are skipped and counted.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from deidkit.namefold import after_escape


def decoded_strings(obj, depth: int = 0) -> Iterator[str]:
    """Every string in a JSON value, with JSON found inside strings decoded as well."""
    if isinstance(obj, str):
        yield obj
        s = obj.strip()
        if depth < 4 and s[:1] in "{[" and s[-1:] in "}]":
            try:
                inner = json.loads(s)
            except ValueError:
                return
            yield from decoded_strings(inner, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from decoded_strings(v, depth)
    elif isinstance(obj, list):
        for v in obj:
            yield from decoded_strings(v, depth)


def crossed_text(record: Path) -> tuple[str, int]:
    """All text the proxy sent to the provider, decoded, and the number of requests."""
    parts, requests = [], 0
    for line in record.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("dir") != "out":
            continue
        requests += 1
        parts.extend(decoded_strings({k: row.get(k) for k in ("body", "turn_metadata")}))
    return "\n".join(parts), requests


def _word_around(text: str, start: int, end: int) -> tuple[int, int]:
    a, b = start, end
    while a > 0 and (text[a - 1].isalnum() or text[a - 1] in "_-"):
        a -= 1
    while b < len(text) and (text[b].isalnum() or text[b] in "_-"):
        b += 1
    return a, b


def find(text: str, surfaces: dict[str, str]) -> dict[str, dict]:
    """Per kind: standalone hits, and masked shapes of the longer words a value sits inside."""
    low = text.casefold()
    report: dict[str, dict] = defaultdict(lambda: {"standalone": 0, "inside": defaultdict(int),
                                                   "skipped": 0, "values": 0})
    for surface, kind in surfaces.items():
        entry = report[kind]
        needle = surface.casefold()
        if len(needle) < 3:
            entry["skipped"] += 1
            continue
        entry["values"] += 1
        i = low.find(needle)
        while i != -1:
            j = i + len(needle)
            a, b = _word_around(low, i, j)
            # A value glued to an escape ("\\nHarrowgate", JSON that was not decoded because
            # text surrounds it) stands alone: the escape is the line break it encodes.
            left_ok = i == 0 or not (low[i - 1].isalnum() or low[i - 1] == "_") or \
                after_escape(low, i)
            right_ok = j == len(low) or not (low[j].isalnum() or low[j] == "_")
            if (left_ok and right_ok) or (a == i and b == j):
                entry["standalone"] += 1
            else:
                shape = text[a:i] + "*" * (j - i) + text[j:b]
                entry["inside"][shape] += 1
            i = low.find(needle, i + 1)
    return report


async def known_surfaces(vault_path: Path, scopes: list[str]) -> dict[str, str]:
    """Every surface the vault matches for the scopes, and the kind of its token."""
    from deidkit.sqlite_store import SQLiteTokenStore
    store = SQLiteTokenStore(vault_path)
    out: dict[str, str] = {}
    for scope in scopes:
        tokens = {r.token: r for r in await store.load_tokens(scope)}
        for row in tokens.values():
            if row.real_value:
                out.setdefault(row.real_value, row.kind)
        for alias in await store.load_aliases(scope):
            row = tokens.get(alias.token)
            if alias.surface and row is not None:
                out.setdefault(alias.surface, row.kind)
    store.close()
    return out


def run(record: Path, vault_path: Path, scopes: list[str]) -> int:
    text, requests = crossed_text(record)
    surfaces = asyncio.run(known_surfaces(vault_path, scopes))
    report = find(text, surfaces)
    leaks = sum(e["standalone"] for e in report.values())
    print(f"record: {requests} requests sent; vault: {len(surfaces)} known values "
          f"in scope {', '.join(scopes)}")
    for kind in sorted(report):
        e = report[kind]
        inside = sum(e["inside"].values())
        line = f"  {kind:8} {e['values']:4} values  standalone {e['standalone']:3}  " \
               f"inside other words {inside:3}"
        if e["skipped"]:
            line += f"  (skipped {e['skipped']} shorter than 3)"
        print(line)
        for shape, n in sorted(e["inside"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"           inside {shape!r} x{n}")
    print("RESULT: " + ("LEAK — a known value crossed as a word of its own" if leaks
                        else "no known value crossed as a word of its own"))
    return 1 if leaks else 0


__all__ = ["crossed_text", "decoded_strings", "find", "known_surfaces", "run"]
