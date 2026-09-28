# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Server-sent events: read them from a byte stream, write them back."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator


async def events(lines: AsyncIterator[bytes]) -> AsyncIterator[tuple[str, str]]:
    """``(event name, data)`` for every event; ``("", ":comment")`` for a comment line."""
    name, data = "", []
    async for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data or name:
                yield name, "\n".join(data)
            name, data = "", []
        elif line.startswith(":"):
            yield "", line
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[6:] if line[5:6] == " " else line[5:])
    if data or name:
        yield name, "\n".join(data)


def encode(name: str, data: dict | str) -> bytes:
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False,
                                                          separators=(",", ":"))
    head = f"event: {name}\n" if name else ""
    return (head + "".join(f"data: {line}\n" for line in body.split("\n")) + "\n").encode("utf-8")


__all__ = ["encode", "events"]
