# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""File names, paths and copied lines come back spelled the way they were written.

The return leg puts one value back for each token: the document's own spelling when the whole
request used only one, otherwise the vault's canonical value. That reads well in prose and breaks
anything that must match the disk exactly. MEASURED 29.09.2026: a matter file was named with
the client's short name, the agent saw it as ``03-Reply-to-ORG_1-Freight.md``, and its call to
read the file came back as ``03-Reply-to-Harrowgate Freight Ltd-Freight.md``, a file that does
not exist; the agent had to fall back to a shell glob. ``letters/04 Brenner, Ada - note.md``
would have come back as ``letters/04 Ada Brenner, Ada - note.md``, and a line an edit must find
in a file the same way.

So, per request, the text around each token is remembered as it was written before tokenising
(:func:`spellings`), at two sizes:

* a **segment**, the text between two separators: a path separator, a line break or a tab (also
  as an escape, the way a JSON string or a script writes them), or the arrow a file viewer puts
  between a line number and the line. A segment is a file or folder name, spaces and commas
  included, or one line;
* a **run** of the characters a name is made of, such as ``03-Reply-to-ORG_1-Freight.md`` inside
  a shell command.

The arguments of a tool that acts on this machine get remembered segments back verbatim, then
remembered quoted names inside a segment (a path in a shell command), then runs, before the
remaining tokens are restored as usual (:func:`respell`). A line of a diff is looked up without
its sign. A token that is a whole folder name (``/work/Brenner/``) is remembered as well, but put
back only between two slashes: on its own line it is a name in prose. Text the model
made up, such as the name of a new file, is not in the table and gets the usual value; so does a
bare token, which is a name in prose rather than part of one.
"""

from __future__ import annotations

import bisect
import json
import re

from deidkit.gateway import Redaction

#: A token as the vault mints it.
TOKEN = re.compile(r"(?<![0-9A-Za-z])[A-Z]+_\d+(?![0-9A-Za-z])")
#: A run of the characters one file or folder name is made of.
RUN = re.compile(r"[\w.\-+~#@%=&]+")
_RUN_CHAR = re.compile(r"[\w.\-+~#@%=&]")
#: What ends a segment: a path separator, a line break or a tab, written out or as an escape
#: (a backslash takes the character after it, so a Windows path splits the same way every
#: time), and the arrow between a line number and the line in a file viewer.
_SEP = re.compile(r"\\(?:u[0-9A-Fa-f]{4}|.)|[/\n\r\t→]", re.S)
#: Decoration around a name or a line, which is not part of it: spaces, quotes, brackets.
_WRAP = " \"'`()[]<>{},;:"
_EDGE = ".-_"
_QUOTE = re.compile(r"([\"'`])")
_LAST_RUN = re.compile(r"[\w.\-+~#@%=&]+$")
#: A longer segment is a paragraph, which no tool call will copy whole.
_LONGEST = 2000


def _align(original: str, red: Redaction) -> list[tuple[int, int, int, int]] | None:
    """For each token that replaced words in ``original``: its span in ``red.text`` and the span
    of the words it replaced. None when the two texts do not line up."""
    queues = {t: list(reversed(v)) for t, v in (red.surfaces or {}).items()}
    spans: list[tuple[int, int, int, int]] = []
    i = j = 0
    for m in TOKEN.finditer(red.text):
        gap = red.text[i:m.start()]
        if not original.startswith(gap, j):
            return None
        j += len(gap)
        queue = queues.get(m.group(0))
        if queue and original.startswith(queue[-1], j):
            surface = queue.pop()
            spans.append((m.start(), m.end(), j, j + len(surface)))
            j += len(surface)
        elif original.startswith(m.group(0), j):     # a token that was in the text already
            j += len(m.group(0))
        else:
            return None
        i = m.end()
    return spans if original[j:] == red.text[i:] else None


class _Positions:
    """Positions in the tokenised text, outside any token, as positions in the original."""

    def __init__(self, spans: list[tuple[int, int, int, int]]) -> None:
        self.ends = [te for _, te, _, _ in spans]
        self.shift = [0]
        for ts, te, os_, oe in spans:
            self.shift.append(self.shift[-1] + (oe - os_) - (te - ts))

    def __call__(self, pos: int) -> int:
        return pos + self.shift[bisect.bisect_right(self.ends, pos)]


def spellings(texts: list[str], reds: list[Redaction]) -> dict[str, str]:
    """One request's table: the segment and the run around each token, as tokenised, mapped to
    the same text as it was written."""
    table: dict[str, str] = {}
    for original, red in zip(texts, reds):
        if not red.surfaces or red.text == original:
            continue
        spans = _align(original, red)
        if not spans:
            continue
        text, where = red.text, _Positions(spans)
        seps = [(m.start(), m.end()) for m in _SEP.finditer(text)]
        starts = [s for s, _ in seps]

        def remember(a: int, b: int) -> None:
            key = text[a:b]
            if key and key not in table and not TOKEN.fullmatch(key.strip(_EDGE)):
                table[key] = original[where(a):where(b)]

        def remember_folder(a: int, b: int) -> None:
            a += len(text[a:b]) - len(text[a:b].lstrip(_EDGE))
            b -= len(text[a:b]) - len(text[a:b].rstrip(_EDGE))
            table.setdefault("/" + text[a:b], original[where(a):where(b)])

        for ts, te, _, _ in spans:
            a, b = ts, te
            while a > 0 and _RUN_CHAR.fullmatch(text[a - 1]):
                a -= 1
            while b < len(text) and _RUN_CHAR.fullmatch(text[b]):
                b += 1
            remember(a, b)
            k = bisect.bisect_right(starts, ts) - 1
            s0 = seps[k][1] if k >= 0 else 0
            s1 = starts[k + 1] if k + 1 < len(starts) else len(text)
            seg = text[s0:s1]
            lo, hi = s0 + len(seg) - len(seg.lstrip(_WRAP)), s0 + len(seg.rstrip(_WRAP))
            if not 0 < hi - lo <= _LONGEST:
                continue
            if not TOKEN.fullmatch(text[lo:hi].strip(_EDGE)):
                remember(lo, hi)
            elif text[s0 - 1:s0] == "/" or text[s1:s1 + 1] == "/":
                remember_folder(lo, hi)
    return table


def respell(text: str, table: dict[str, str], *, literal: bool = False) -> str:
    """``text`` with each remembered segment, or else run, put back as it was written (escaped
    for a string literal when ``literal``); every other token is left for the usual mapping."""
    if not table or not text or not TOKEN.search(text):
        return text

    def put(written: str) -> str:
        return json.dumps(written, ensure_ascii=False)[1:-1] if literal else written

    def run(m: re.Match) -> str:
        core, tail = m.group(0), ""
        if not TOKEN.search(core):
            return core
        while core not in table and core and core[-1] in ".-":      # sentence punctuation
            core, tail = core[:-1], core[-1] + tail
        return put(table[core]) + tail if core in table else m.group(0)

    def remembered(piece: str, folder: bool) -> str | None:
        lo, hi = len(piece) - len(piece.lstrip(_WRAP)), len(piece.rstrip(_WRAP))
        key = piece[lo:hi]
        if key in table:
            return piece[:lo] + put(table[key]) + piece[hi:]
        if key[:1] in "+-":                       # a line of a diff: the sign stays
            inner = key[1:].lstrip(_WRAP)
            if inner in table:
                return piece[:lo] + key[:len(key) - len(inner)] + put(table[inner]) + piece[hi:]
        core = key.strip(_EDGE)
        if folder and "/" + core in table and TOKEN.fullmatch(core):
            at = key.index(core)
            return piece[:lo + at] + put(table["/" + core]) + piece[lo + at + len(core):]
        return None

    def folder_name(run_: str) -> str | None:
        core = run_.strip(_EDGE)
        if not TOKEN.fullmatch(core) or "/" + core not in table:
            return None
        at = run_.index(core)
        return run_[:at] + put(table["/" + core]) + run_[at + len(core):]

    def segment(seg: str, after_slash: bool, before_slash: bool) -> str:
        if not TOKEN.search(seg):
            return seg
        got = remembered(seg, after_slash or before_slash)
        if got is not None:
            return got
        head = tail = ""
        first = RUN.match(seg) if after_slash else None     # "/work/ORG_1 && ls"
        if first and (name := folder_name(first.group(0))) is not None:
            head, seg = name, seg[first.end():]
        last = _LAST_RUN.search(seg) if before_slash else None    # "cd ORG_1/letters"
        if last and (name := folder_name(last.group(0))) is not None:
            seg, tail = seg[:last.start()], name
        parts = _QUOTE.split(seg)                 # a quoted name inside a command
        return head + "".join(
            p if not TOKEN.search(p) else (remembered(p, False) or RUN.sub(run, p))
            for p in parts) + tail

    out: list[str] = []
    cursor = 0
    for m in [*_SEP.finditer(text), None]:
        end = m.start() if m else len(text)
        out.append(segment(text[cursor:end], text[cursor - 1:cursor] == "/",
                           text[end:end + 1] == "/"))
        if m:
            out.append(m.group(0))
            cursor = m.end()
    return "".join(out)


__all__ = ["RUN", "TOKEN", "respell", "spellings"]
