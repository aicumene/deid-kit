"""A matter document under review, as the desktop shows it: its text, and over the text what is
de-identified before a model sees it, the norms the text refers to, and the agents' notes.

The text is the document's in Bernio's matter format (bernio-desktop-chat ``document/FORMAT.md``): a
Markdown or plain-text document is its own text; any other has its text in ``.legix/text/<path>.md``,
which the app writes. Positions are characters (code points) from the start of that text, as in the
W3C Web Annotation model the markup is kept in.

- **De-identified.** The scope's own engine, the one the proxy runs on every request, reads the text;
  where each token stands in the text is found by walking the tokenised text against the original.
- **Norms.** :mod:`deidkit.agents.cite` — the server's extraction, without its knowledge base.
- **Notes.** An agent reviews the document and answers with notes, each on a verbatim quote. The quote
  is looked up in the text by code — a note whose quote is not in the document is not kept, whatever
  the model says. A quote not found as it is written is looked up again in tokens (:func:`find_quotes`):
  the model saw a token where the document has a party's short name ("HFL", "DW", another case), and the
  proxy gave it back the full one. A kept note is one annotation a file, ``.legix/markup/<path>/<id>.json``: the matter's
  notes folder, which leGix versions, signs and syncs with the documents. Nothing in such a file is
  trusted on reading: the quote is looked up in the text again, so a note shows where its words are now,
  or that they are gone.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from deidkit.agents import cite
from deidkit.proxy.spelling import TOKEN

#: Documents that are their own text (bernio-document ``kind``).
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
NOTES = ".legix"
MAX_TEXT = 4 * 1024 * 1024

KINDS = ("risk", "question", "point", "clause")
ASPECTS = ("parties", "subject", "term", "termination", "payment", "liability", "deadlines", "notices",
           "data protection", "dispute resolution", "compliance", "other")
CLAUSE_TYPES = ("definition", "obligation", "right", "condition", "deadline", "payment", "notice",
                "proviso", "warranty", "other")
#: How a note's kind is said in the Web Annotation model.
MOTIVATION = {"risk": "assessing", "question": "questioning", "point": "commenting", "clause": "classifying"}
MAX_QUOTE = 600
MAX_NOTE = 1200
MAX_NOTES = 80
MAX_REPLACEMENT = 4000
#: What a person decides of a note or a proposed change: of a change, accepted or rejected; of an agent's
#: note, whether it helped. Kept as the person's own annotations — what the RL and harness data are made of.
VERDICTS = ("accepted", "rejected", "useful", "not useful", "wrong")

PROMPT = """Review the document "{path}" for the lawyer. Its text is in the file "{source}" in this folder: \
read that file, all of it. Do not change any file.

Write down what a lawyer reviewing this document must see. For each point, copy the document's own words \
that it is about — exactly, character for character, at most 300 characters, enough to be found in the \
text — and say:
- "kind": "risk" (could hurt the client), "question" (must be asked or checked before this goes further), \
"point" (important to know), or "clause" (what a clause does — the document's structure: give one for each \
clause that matters);
- "aspect": one of {aspects};
- "importance": 1 (minor) to 5 (critical) for the client;
- "clause_type", for a clause: one of {clause_types};
- "who", "what", "to_whom", "when": for a clause, who must or may do what, for whom, and when — from the \
text only, empty where it does not say;
- "norm": the statute or regulation the point turns on, as it is usually cited (for example "§ 573c BGB", \
"Housing Act 1988 s. 21", "Art. 28 GDPR"), or empty;
- "suggestion": where you would fix the point by changing the words, the words to put in place of the quote, \
exactly as they should read — the lawyer accepts or rejects it; otherwise empty;
- "note": one or two sentences for the lawyer, in the language they write to you in — under 300 characters.

At most 20 points, the most important first. Answer with one JSON object and nothing else:
{{"notes": [{{"quote": "...", "kind": "risk", "aspect": "termination", "importance": 4, "clause_type": "", \
"who": "", "what": "", "to_whom": "", "when": "", "norm": "", "suggestion": "", "note": "..."}}]}}"""


def prompt(path: str, source: str) -> str:
    return PROMPT.format(path=path, source=source, aspects=", ".join(ASPECTS),
                         clause_types=", ".join(CLAUSE_TYPES))


# ── the document's text ───────────────────────────────────────────────────────────────────────────
def inside(folder: Path, rel: str) -> Path | None:
    """``folder / rel`` if it is a file in ``folder`` (no way out by ``..`` or a link)."""
    if not rel or rel.startswith("/") or "\x00" in rel:
        return None
    try:
        target = (folder / rel).resolve()
        target.relative_to(folder.resolve())
    except (OSError, ValueError):
        return None
    return target if target.is_file() else None


def text_source(path: str) -> str:
    """Where the text of the document ``path`` is, relative to the matter folder."""
    return path if Path(path).suffix.lower() in TEXT_SUFFIXES else f"{NOTES}/text/{path}.md"


def read_text(folder: Path, path: str) -> tuple[str, str] | None:
    """``(source, text)`` of the document ``path``, or None: no such document, or its text not written yet.
    The notes folder itself is not a document."""
    if path.split("/", 1)[0] == NOTES or inside(folder, path) is None:
        return None
    source = text_source(path)
    target = inside(folder, source)
    if target is None or target.stat().st_size > MAX_TEXT:
        return None
    return source, target.read_text("utf-8", errors="replace")


# ── what is de-identified ─────────────────────────────────────────────────────────────────────────
def _walk(original: str, tokenised: str, surfaces: dict[str, list[str]], mapping: dict[str, str]):
    """The tokenised text walked against the original. For each token placed: ``(after, start, end, piece,
    at, to, token)`` — in the tokenised text the token is ``[start, end)`` and the literal piece before it
    ``[after, start)``; in the original the piece begins at ``piece`` and the token's words are ``[at, to)``.

    The tokenised text is the original with surfaces put in tokens' places; what lies between tokens is
    the original's own. So the two are walked together: the literal piece before a token is matched in
    the original, and the token takes the surface — one the engine recorded for it in this text — that
    stands there. Should the walk lose step (a piece not where it was expected), it finds the piece again
    further on, so one odd place does not cost the rest of the document."""
    at = 0                                         # in the original
    last = 0                                       # in the tokenised text
    tokens = list(TOKEN.finditer(tokenised))
    for i, m in enumerate(tokens):
        token = m.group(0)
        if token not in mapping and token not in surfaces:
            continue                               # a token-like word of the document itself
        piece = tokenised[last:m.start()]
        if original.startswith(piece, at):
            begins = at
        else:
            found = original.find(piece, at) if piece else at
            if found < 0:
                last = m.end()
                continue
            begins = found
        at = begins + len(piece)
        candidates = sorted({*surfaces.get(token, []), mapping.get(token, "")} - {""}, key=len, reverse=True)
        surface = next((s for s in candidates if original.startswith(s, at)), None)
        if surface is None:
            # Not one of the recorded spellings: the token covers what lies up to the next literal piece.
            nxt = tokens[i + 1].start() if i + 1 < len(tokens) else len(tokenised)
            follow = tokenised[m.end():nxt][:40]
            end = original.find(follow, at) if follow else len(original)
            if end <= at:
                last = m.end()
                continue
            surface = original[at:end]
        yield last, m.start(), m.end(), begins, at, at + len(surface), token
        at += len(surface)
        last = m.end()


def deid_spans(original: str, tokenised: str, surfaces: dict[str, list[str]],
               mapping: dict[str, str]) -> list[dict]:
    """Where each token of ``tokenised`` stands in ``original``: ``{start, end, type, token}``."""
    return [{"start": at, "end": to, "type": token.rsplit("_", 1)[0], "token": token}
            for _, _, _, _, at, to, token in _walk(original, tokenised, surfaces, mapping)]


def _alignment(original: str, tokenised: str, surfaces: dict[str, list[str]],
               mapping: dict[str, str]) -> list[tuple[int, int, int, int, bool]]:
    """The tokenised text in pieces, each with its place in the original: ``(t0, t1, o0, o1, token)`` — a
    literal piece (the same characters) or a token and the words it stands for. What the walk could not
    place is in no piece."""
    out: list[tuple[int, int, int, int, bool]] = []
    tail_t = tail_o = 0
    for after, start, end, piece, at, to, _ in _walk(original, tokenised, surfaces, mapping):
        out.append((after, start, piece, piece + start - after, False))
        out.append((start, end, at, to, True))
        tail_t, tail_o = end, to
    tail = tokenised[tail_t:]
    found = tail_o if original.startswith(tail, tail_o) else original.find(tail, tail_o)
    if found >= 0:
        out.append((tail_t, len(tokenised), found, found + len(tail), False))
    return [seg for seg in out if seg[0] < seg[1]]


def _to_original(pieces: list[tuple[int, int, int, int, bool]], a: int, b: int) -> tuple[int, int] | None:
    """``[a, b)`` of the tokenised text as a span of the original: a token taken whole; None — an end falls
    where the walk placed nothing."""
    start = end = None
    for t0, t1, o0, o1, token in pieces:
        if start is None and t0 <= a < t1:
            start = o0 if token else o0 + (a - t0)
        if t0 < b <= t1:
            end = o1 if token else o0 + (b - t0)
    return (start, end) if start is not None and end is not None and start < end else None


async def find_quotes(text: str, quotes: list[str], engine=None) -> list[tuple[int, int] | None]:
    """Where each quote is in ``text`` (:meth:`Folded.find`), or None. A quote with a name in it that is not
    found as written is looked up again in tokens, with the scope's ``engine``: the quote and the text
    tokenised, every spelling of a party is its one token — the model saw "ORG_…" where the document says
    "HFL", and the proxy gave the quote back "Harrowgate Freight Ltd". Found there, it is the document's own
    words at that place."""
    folded = Folded.of(text)
    found = [folded.find(q) if q else None for q in quotes]
    missing = [i for i, q in enumerate(quotes) if q and found[i] is None]
    if not missing or engine is None:
        return found
    try:
        reds = await engine.tokenize_all([text, *(quotes[i] for i in missing)])
    except Exception:                              # noqa: BLE001 — the vault did not settle: as written only
        return found
    doc = reds[0]
    pieces = _alignment(text, doc.text, doc.surfaces or {}, doc.mapping or {})
    tokenised = Folded.of(doc.text)
    for i, red in zip(missing, reds[1:]):
        if red.text == quotes[i]:
            continue                               # no name in it: tokens change nothing
        hit = tokenised.find(red.text)
        if hit:
            found[i] = _to_original(pieces, *hit)
    return found


# ── norms ─────────────────────────────────────────────────────────────────────────────────────────
def norm_mentions(text: str) -> list[dict]:
    """The norms ``text`` refers to: ``{start, end, citation, jurisdiction}``, in the order of the text.
    One mention citing two sections ("§§ 133, 157 BGB") is two entries over the same place."""
    return [{"start": s.start, "end": s.end, "citation": s.citation, "jurisdiction": cite.jurisdiction_of(s.citation)}
            for s in sorted(cite.extract_citation_spans(text), key=lambda s: (s.start, s.citation))]


# ── finding a quote ───────────────────────────────────────────────────────────────────────────────
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "`": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
})
_DROP = {"\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026|\[\.\.\.\]|\[\u2026\])\s*")
_ESCAPED = re.compile(r"\\([!-/:-@\[-`{-~])")    # Markdown's backslash before punctuation
_MAX_GAP = 2000


def _fold(text: str, *, unescape: bool) -> tuple[str, list[int]]:
    """``text`` folded for comparison — typographic quotes and dashes plain, any run of white space one
    space, invisible characters gone, and with ``unescape`` Markdown's escapes dropped — with, for each
    character of the result, where it came from."""
    out: list[str] = []
    back: list[int] = []
    space = False
    i = 0
    while i < len(text):
        ch = text[i]
        if unescape and ch == "\\" and i + 1 < len(text) and _ESCAPED.match(text, i):
            i += 1
            continue
        if ch in _DROP:
            i += 1
            continue
        if ch.isspace():
            if not space and out:
                out.append(" ")
                back.append(i)
            space = True
            i += 1
            continue
        space = False
        out.append(ch.translate(_FOLD))
        back.append(i)
        i += 1
    if out and out[-1] == " ":
        out.pop()
        back.pop()
    return "".join(out), back


@dataclass(frozen=True)
class Folded:
    """A text folded once, for looking many quotes up in it."""
    text: str
    folded: str
    back: list[int]

    @classmethod
    def of(cls, text: str) -> "Folded":
        folded, back = _fold(text, unescape=True)
        return cls(text, folded, back)

    def find(self, quote: str, near: int | None = None) -> tuple[int, int] | None:
        """Where ``quote`` is in the text: ``(start, end)``, or None. Exact first; then folded; a quote
        with an ellipsis is its parts in order, each within a page of the one before. Of several places,
        the one nearest ``near``."""
        quote = quote.strip()
        if not quote:
            return None
        exact = [m.start() for m in re.finditer(re.escape(quote), self.text)]
        if exact:
            start = min(exact, key=lambda s: abs(s - near)) if near is not None else exact[0]
            return start, start + len(quote)
        parts = [p for p in (_fold(p, unescape=True)[0] for p in _ELLIPSIS.split(quote)) if p]
        if not parts or sum(len(p) for p in parts) < 8:
            return None
        hits: list[tuple[int, int]] = []
        for first in _all(self.folded, parts[0]):
            end = first + len(parts[0])
            for part in parts[1:]:
                nxt = self.folded.find(part, end, end + _MAX_GAP + len(part))
                if nxt < 0:
                    break
                end = nxt + len(part)
            else:
                hits.append((self.back[first], self.back[end - 1] + 1))
        if not hits:
            return None
        return min(hits, key=lambda h: abs(h[0] - near)) if near is not None else hits[0]


def _all(haystack: str, needle: str) -> list[int]:
    out, at = [], haystack.find(needle)
    while at >= 0:
        out.append(at)
        at = haystack.find(needle, at + 1)
    return out


# ── clause addresses ──────────────────────────────────────────────────────────────────────────────
_ADDRESS = re.compile(r"(?<!\\)\{#([a-z0-9_-]+)\}[ \t]*$", re.M)


def clause_at(text: str, at: int) -> str | None:
    """The address of the clause the place ``at`` is in: of its own line if the line has one, else of
    the nearest line before it that has."""
    line_end = text.find("\n", at)
    line_end = len(text) if line_end < 0 else line_end
    found = None
    for m in _ADDRESS.finditer(text, 0, line_end):
        found = m.group(1)
    return found


# ── the agent's answer ────────────────────────────────────────────────────────────────────────────
_NOTES_OBJECT = re.compile(r'\{\s*"notes"\s*:')


def _cut_notes(answer: str) -> list | None:
    """The whole notes of an answer cut short — its length ran out in the middle of the list: each note that was
    written to its end, in order; None — no list of notes begins in it."""
    starts = list(_NOTES_OBJECT.finditer(answer))
    if not starts:
        return None
    at = answer.find("[", starts[-1].end())
    if at < 0 or answer[starts[-1].end():at].strip():
        return None
    decoder = json.JSONDecoder(strict=False)
    notes, at = [], at + 1
    while True:
        while at < len(answer) and answer[at] in " \t\r\n,":
            at += 1
        try:
            value, at = decoder.raw_decode(answer, at)
        except ValueError:
            return notes
        notes.append(value)


def _json_object(answer: str) -> dict | None:
    """The notes object in the agent's answer. The answer is everything the agent said in the task —
    "I'll read the file first", then the object, perhaps fenced — so the object is looked for where it
    starts, and the last whole one wins. A line break left raw inside a string is taken as one."""
    try:
        value = json.loads(answer, strict=False)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    decoder = json.JSONDecoder(strict=False)
    found = None
    for m in _NOTES_OBJECT.finditer(answer):
        try:
            value, _ = decoder.raw_decode(answer, m.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            found = value
    return found


def _text(value, limit: int) -> str:
    return " ".join(str(value).split())[:limit] if isinstance(value, (str, int, float)) else ""


def parse_notes(answer: str) -> list[dict] | None:
    """The notes in an agent's answer, each with its fields in our vocabulary; None — no notes object in it.
    An answer cut short in the middle of its notes gives the ones written to their end (:func:`notes_of`)."""
    return notes_of(answer)[0]


def notes_of(answer: str) -> tuple[list[dict] | None, bool]:
    """``(notes, cut)``: the notes, as :func:`parse_notes` gives them, and whether the answer was cut short —
    then its whole notes are kept, and the one it was writing is lost."""
    data = _json_object(answer)
    cut = False
    if data is None or not isinstance(data.get("notes"), list):
        whole = _cut_notes(answer) if data is None else None
        if not whole:
            return None, False
        data, cut = {"notes": whole}, True
    return _notes(data), cut


def _notes(data: dict) -> list[dict]:
    out = []
    for raw in data["notes"][:MAX_NOTES]:
        if not isinstance(raw, dict):
            continue
        quote = raw.get("quote") if isinstance(raw.get("quote"), str) else ""
        kind = _text(raw.get("kind"), 20).lower()
        aspect = _text(raw.get("aspect"), 40).lower()
        clause_type = _text(raw.get("clause_type"), 40).lower()
        try:
            importance = max(1, min(5, int(raw.get("importance"))))
        except (TypeError, ValueError):
            importance = 3
        out.append({
            "quote": quote.strip()[:MAX_QUOTE],
            "kind": kind if kind in KINDS else "point",
            "aspect": aspect if aspect in ASPECTS else "other",
            "importance": importance,
            "clause_type": clause_type if clause_type in CLAUSE_TYPES else ("other" if kind == "clause" else ""),
            **{k: _text(raw.get(k), 200) for k in ("who", "what", "to_whom", "when")},
            "norm": _text(raw.get("norm"), 120),
            "note": _text(raw.get("note"), MAX_NOTE),
            # The wording itself, line breaks and all: it goes into the document as written.
            "suggestion": raw["suggestion"].strip()[:MAX_REPLACEMENT] if isinstance(raw.get("suggestion"), str) else "",
        })
    return out


# ── the markup ────────────────────────────────────────────────────────────────────────────────────
def annotation(note: dict, *, path: str, text: str, start: int, end: int, creator: dict, run: str,
               created: str | None = None) -> dict:
    """A kept note as a W3C Web Annotation (FORMAT.md, "The markup"): the quote with the words around it,
    the position, the clause; the note as its body; what the agent said of it under ``bernio``."""
    selectors: list[dict] = [
        {"type": "TextQuoteSelector", "exact": text[start:end], "prefix": text[max(0, start - 32):start],
         "suffix": text[end:end + 32]},
        {"type": "TextPositionSelector", "start": start, "end": end},
    ]
    clause = clause_at(text, start)
    if clause:
        selectors.append({"type": "FragmentSelector", "value": clause})
    body: list[dict] = []
    if note.get("note"):
        body.append({"type": "TextualBody", "value": note["note"], "purpose": "commenting"})
    if note.get("clause_type"):
        body.append({"type": "TextualBody", "value": note["clause_type"], "purpose": "classifying"})
    if note.get("norm"):
        body.append({"type": "TextualBody", "value": note["norm"], "purpose": "linking"})
    if note.get("suggestion"):
        body.append({"type": "TextualBody", "value": note["suggestion"], "purpose": "replacing"})
    return {
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": uuid.uuid4().hex,
        "type": "Annotation",
        "motivation": MOTIVATION[note["kind"]],
        "creator": creator,
        "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"type": "Software", "name": "deid-agent review", "run": run},
        "target": {"source": path, "selector": selectors},
        "body": body,
        "bernio": {k: note[k] for k in ("kind", "aspect", "importance", "clause_type", "who", "what", "to_whom",
                                        "when", "norm") if note.get(k) not in (None, "")},
    }


def markup_dir(folder: Path, path: str) -> Path:
    return folder / NOTES / "markup" / path


def store(folder: Path, path: str, ann: dict) -> Path:
    """One annotation a file — two devices adding notes never write the same file."""
    target = markup_dir(folder, path) / f"{ann['id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ann, ensure_ascii=False, indent=2) + "\n", "utf-8")
    tmp.replace(target)
    return target


def _selector(ann: dict, kind: str) -> dict:
    target = ann.get("target") if isinstance(ann.get("target"), dict) else {}
    for s in target.get("selector") or []:
        if isinstance(s, dict) and s.get("type") == kind:
            return s
    return {}


def place(ann: dict, folded: Folded) -> tuple[str, int, int] | None:
    """Where the annotation's words are in the text now: ``("here", start, end)`` — where it was put;
    ``("moved", …)`` — found again elsewhere; None — the words are no longer in the text. A proposed
    insertion has no words of its own: it stands between the words before and after it."""
    quote = _selector(ann, "TextQuoteSelector")
    exact = quote.get("exact") if isinstance(quote.get("exact"), str) else ""
    pos = _selector(ann, "TextPositionSelector")
    start, end = pos.get("start"), pos.get("end")
    text = folded.text
    if not exact:
        prefix = quote.get("prefix") if isinstance(quote.get("prefix"), str) else ""
        suffix = quote.get("suffix") if isinstance(quote.get("suffix"), str) else ""
        if not (prefix or suffix):
            return None
        if isinstance(start, int) and text[max(0, start - len(prefix)):start] == prefix \
                and text[start:start + len(suffix)] == suffix:
            return "here", start, start
        # The words around it may have changed too: the nearer words first, then fewer of them; of several
        # places, the one nearest where it was.
        for p, q in ((len(prefix), len(suffix)), (16, 16), (16, 0), (0, 16)):
            before, after = prefix[max(0, len(prefix) - p):] if p else "", suffix[:q]
            if not (before or after):
                continue
            hits = [m + len(before) for m in _all(text, before + after)]
            if hits:
                at = min(hits, key=lambda h: abs(h - start)) if isinstance(start, int) else hits[0]
                return "moved", at, at
        return None
    if isinstance(start, int) and isinstance(end, int) and text[start:end] == exact:
        return "here", start, end
    found = folded.find(exact, near=start if isinstance(start, int) else None)
    return ("moved", *found) if found else None


def _now() -> str:
    """To the microsecond: decisions made one after the other keep their order."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _base(motivation: str, creator: dict, target: dict, body: list[dict], bernio: dict) -> dict:
    return {"@context": "http://www.w3.org/ns/anno.jsonld", "id": uuid.uuid4().hex, "type": "Annotation",
            "motivation": motivation, "creator": creator, "created": _now(), "target": target, "body": body,
            "bernio": bernio}


def person(name: str) -> dict:
    return {"type": "Person", "name": name or "person"}


def suggestion(*, path: str, text: str, start: int, end: int, replacement: str, creator: dict) -> dict:
    """A change a person proposes, as Word's tracked change: these words (none, for an insertion) to read
    as ``replacement`` (empty, for a deletion). It changes nothing until it is accepted."""
    selectors: list[dict] = [
        {"type": "TextQuoteSelector", "exact": text[start:end], "prefix": text[max(0, start - 32):start],
         "suffix": text[end:end + 32]},
        {"type": "TextPositionSelector", "start": start, "end": end},
    ]
    clause = clause_at(text, start)
    if clause:
        selectors.append({"type": "FragmentSelector", "value": clause})
    return _base("editing", creator, {"source": path, "selector": selectors},
                 [{"type": "TextualBody", "value": replacement, "purpose": "replacing"}], {"kind": "edit"})


def decision(*, path: str, target: str, verdict: str, creator: dict, comment: str = "", correction: str = "",
             applied: bool = False) -> dict:
    """What a person decided of a note or a proposed change (``target``, its annotation's id): ``verdict``,
    and what they said and would have written instead."""
    body = [{"type": "TextualBody", "value": verdict, "purpose": "assessing"}]
    if comment:
        body.append({"type": "TextualBody", "value": comment, "purpose": "commenting"})
    if correction:
        body.append({"type": "TextualBody", "value": correction, "purpose": "replacing"})
    return _base("assessing", creator, {"source": path, "annotation": target}, body,
                 {"kind": "decision", "verdict": verdict, **({"applied": True} if applied else {})})


_WORD = re.compile(r"\s+|\w+|[^\w\s]")


def hunks(before: str, after: str, limit: int = 200) -> list[dict]:
    """How ``after`` differs from ``before``, word by word: ``{at, old, new}``, ``at`` the place in ``before``."""
    import difflib
    a, b = _WORD.findall(before), _WORD.findall(after)
    starts, at = [], 0
    for w in a:
        starts.append(at)
        at += len(w)
    starts.append(at)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append({"at": starts[i1], "old": "".join(a[i1:i2]), "new": "".join(b[j1:j2])})
            if len(out) >= limit:
                break
    return out


def revision(*, path: str, before: str, after: str, creator: dict) -> dict:
    """The person's own writing, saved from the editor: how the text read before and after, as changes word
    by word. Already made, nothing to accept; it is the record of who wrote what."""
    return _base("editing", creator, {"source": path}, [], {
        "kind": "revision", "before": sha(before), "after": sha(after), "hunks": hunks(before, after)})


def written(*, path: str, creator: dict, how: str, run: str = "") -> dict:
    """Who wrote the document: an agent in a task (``how``: added or changed), or a person."""
    return _base("describing", creator, {"source": path}, [], {"kind": "written", "how": how,
                                                               **({"run": run} if run else {})})


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def writable(path: str) -> bool:
    """A document whose text is the document itself (Markdown, plain text): changes are made in it. A Word
    document's text is its picture, and so is a PDF's; they are changed in their own programs."""
    return Path(path).suffix.lower() in TEXT_SUFFIXES and path.split("/", 1)[0] != NOTES


def annotations(folder: Path, path: str) -> list[dict]:
    """The document's annotations as they are in its files; what is not one of this document is left out."""
    root = markup_dir(folder, path)
    out = []
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            ann = json.loads(f.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(ann, dict) and isinstance(ann.get("target"), dict) and ann["target"].get("source") == path:
            ann.setdefault("id", f.stem)
            out.append(ann)
    return out


def _who(creator) -> dict:
    if isinstance(creator, dict):
        return {"name": _text(creator.get("name"), 80), "model": _text(creator.get("model"), 80),
                "person": creator.get("type") == "Person"}
    return {"name": _text(creator, 80), "model": "", "person": False}


def _body(ann: dict, purpose: str, limit: int = MAX_NOTE) -> str:
    for b in ann.get("body") or []:
        if isinstance(b, dict) and b.get("purpose") == purpose and isinstance(b.get("value"), str):
            return b["value"][:limit] if purpose == "replacing" else _text(b["value"], limit)
    return ""


def _kind(ann: dict) -> str:
    bernio = ann.get("bernio") if isinstance(ann.get("bernio"), dict) else {}
    return bernio.get("kind") if isinstance(bernio.get("kind"), str) else ""


def load(folder: Path, path: str, text: str) -> list[dict]:
    """The document's notes and proposed changes, each placed in the text as it is now, with what people
    decided of it. Nothing in a file is trusted: the words are looked up again, and only our own fields
    are read."""
    return view(folder, path, text)["notes"]


def view(folder: Path, path: str, text: str) -> dict:
    """``notes`` (the agents' notes and the proposed changes), ``revisions`` (what people wrote in the
    editor) and ``written`` (who wrote the document)."""
    folded = Folded.of(text)
    anns = annotations(folder, path)
    decided: dict[str, list[dict]] = {}
    revisions, authors, notes = [], [], []
    for ann in anns:
        kind = _kind(ann)
        if kind == "decision":
            target = ann["target"].get("annotation")
            verdict = (ann.get("bernio") or {}).get("verdict")
            if isinstance(target, str) and verdict in VERDICTS:
                decided.setdefault(target, []).append({
                    "verdict": verdict, "by": _who(ann.get("creator"))["name"], "at": _text(ann.get("created"), 40),
                    "comment": _body(ann, "commenting"), "correction": _body(ann, "replacing", MAX_REPLACEMENT),
                    "applied": (ann.get("bernio") or {}).get("applied") is True})
        elif kind == "revision":
            hs = (ann.get("bernio") or {}).get("hunks")
            revisions.append({"id": _text(ann.get("id"), 64), "by": _who(ann.get("creator")),
                              "at": _text(ann.get("created"), 40), "changes": len(hs) if isinstance(hs, list) else 0})
        elif kind == "written":
            how = (ann.get("bernio") or {}).get("how")
            authors.append({"by": _who(ann.get("creator")), "at": _text(ann.get("created"), 40),
                            "how": how if how in ("added", "changed", "wrote", "saved") else ""})
    for ann in anns:
        kind = _kind(ann)
        if kind not in (*KINDS, "edit"):
            continue
        where = place(ann, folded)
        bernio = ann.get("bernio") or {}
        decisions = sorted(decided.get(str(ann.get("id")), []), key=lambda d: d["at"])
        state = next((d["verdict"] for d in reversed(decisions) if d["verdict"] in ("accepted", "rejected")), "")
        notes.append({
            "id": _text(ann.get("id"), 64),
            "created": _text(ann.get("created"), 40),
            "creator": {k: v for k, v in _who(ann.get("creator")).items() if k != "person"},
            "by_person": _who(ann.get("creator"))["person"],
            "run": _text((ann.get("generator") or {}).get("run"), 64) if isinstance(ann.get("generator"), dict) else "",
            "status": where[0] if where else "gone",
            "start": where[1] if where else None,
            "end": where[2] if where else None,
            "quote": _text(_selector(ann, "TextQuoteSelector").get("exact"), MAX_QUOTE),
            "clause": _text(_selector(ann, "FragmentSelector").get("value"), 80),
            "note": _body(ann, "commenting"),
            "kind": kind,
            "aspect": bernio.get("aspect") if bernio.get("aspect") in ASPECTS else ("" if kind == "edit" else "other"),
            "importance": bernio.get("importance") if bernio.get("importance") in (1, 2, 3, 4, 5) else 3,
            **{k: _text(bernio.get(k), 200) for k in ("clause_type", "who", "what", "to_whom", "when", "norm")},
            "suggestion": _body(ann, "replacing", MAX_REPLACEMENT),
            "decisions": decisions,
            "state": state,
        })
    notes.sort(key=lambda n: (n["start"] is None, n["start"] or 0, -n["importance"]))
    return {"notes": notes, "revisions": sorted(revisions, key=lambda r: r["at"]),
            "written": sorted(authors, key=lambda a: a["at"])}


def keep(folder: Path, path: str, text: str, notes: list[dict], *, creator: dict, run: str,
         found: list[tuple[int, int] | None] | None = None) -> dict:
    """Keep the notes whose quotes are in the text; say of the others why not. ``found``: where each
    note's quote is (:func:`find_quotes`, which can look in tokens); else looked up as written."""
    folded = Folded.of(text)
    kept, refused = [], []
    for n, note in enumerate(notes):
        if found is not None:
            where = found[n]
        else:
            where = folded.find(note["quote"]) if note["quote"] else None
        if where is None:
            refused.append({"quote": note["quote"][:120], "why": "not in the document" if note["quote"] else "no quote"})
            continue
        ann = annotation(note, path=path, text=text, start=where[0], end=where[1], creator=creator, run=run)
        store(folder, path, ann)
        kept.append(ann["id"])
    return {"kept": kept, "refused": refused}


def find(folder: Path, path: str, ident: str) -> dict | None:
    """The document's annotation by its id."""
    if not re.fullmatch(r"[0-9a-f]{8,64}", ident or ""):
        return None
    return next((a for a in annotations(folder, path) if a.get("id") == ident), None)


def apply(text: str, ann: dict, replacement: str) -> str | None:
    """The text with the annotation's words read as ``replacement``; None — they are no longer in it."""
    where = place(ann, Folded.of(text))
    if where is None:
        return None
    return text[:where[1]] + replacement + text[where[2]:]


def write_text(folder: Path, path: str, text: str) -> None:
    """The document's new text, whole or not at all."""
    target = folder / path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(target)


def new_path_ok(folder: Path, path: str) -> bool:
    """A new document may be written here: Markdown or plain text, inside the folder, not in its notes,
    not over a document that is there."""
    if not path or path.startswith("/") or "\x00" in path or not writable(path):
        return False
    parts = Path(path).parts
    if any(p in ("..", ".") or p.startswith(".") for p in parts):
        return False
    try:
        (folder / path).resolve().relative_to(folder.resolve())
    except (OSError, ValueError):
        return False
    return not (folder / path).exists()


# ── votes on the chat's answers ───────────────────────────────────────────────────────────────────
CHAT_FEEDBACK = "chat-feedback"
MAX_ANSWER = 20000
MAX_QUESTION = 4000


def vote(*, key: str, question: str, answer: str, agent: str, model: str, verdict: str, creator: dict,
         comment: str = "", correction: str = "") -> dict:
    """A person's 👍 or 👎 on an answer of the agent in the chat, with the question it answered, the answer, and
    what they said and would have had it say. One per answer: ``key`` (the chat's and the answer's ids) names
    it, and a later vote replaces it."""
    body = [{"type": "TextualBody", "value": verdict, "purpose": "assessing"}]
    if comment:
        body.append({"type": "TextualBody", "value": comment, "purpose": "commenting"})
    if correction:
        body.append({"type": "TextualBody", "value": correction, "purpose": "replacing"})
    ann = _base("assessing", creator, {"source": "chat"}, body, {
        "kind": "vote", "verdict": verdict, "question": question[:MAX_QUESTION], "answer": answer[:MAX_ANSWER],
        "agent": _text(agent, 80), "model": _text(model, 80)})
    ann["id"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return ann


def store_vote(folder: Path, ann: dict) -> Path:
    target = folder / NOTES / CHAT_FEEDBACK / f"{ann['id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ann, ensure_ascii=False, indent=2) + "\n", "utf-8")
    tmp.replace(target)
    return target


def votes(folder: Path) -> list[dict]:
    """The votes on the chat's answers kept in this folder, as records."""
    root = folder / NOTES / CHAT_FEEDBACK
    out = []
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            ann = json.loads(f.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        bernio = ann.get("bernio") if isinstance(ann, dict) and isinstance(ann.get("bernio"), dict) else {}
        if bernio.get("kind") != "vote" or bernio.get("verdict") not in ("up", "down"):
            continue
        out.append({"type": "answer_vote", "at": _text(ann.get("created"), 40),
                    "agent": _text(bernio.get("agent"), 80), "model": _text(bernio.get("model"), 80),
                    "question": str(bernio.get("question") or "")[:MAX_QUESTION],
                    "answer": str(bernio.get("answer") or "")[:MAX_ANSWER],
                    "verdict": bernio["verdict"], "comment": _body(ann, "commenting"),
                    "correction": _body(ann, "replacing", MAX_REPLACEMENT)})
    return out


# ── what people decided, for the harness and for learning ─────────────────────────────────────────
def records(folder: Path) -> list[dict]:
    """What people decided of the agents' work in this folder, one record each — with the agent and its
    model, the note or change it was about, the verdict and the person's words — what people wrote over
    documents an agent wrote, and their votes on the agent's answers in the chat. Real text: it goes out
    only tokenized (``dataset``)."""
    root = folder / NOTES / "markup"
    out: list[dict] = votes(folder)
    if not root.is_dir():
        return out
    docs = sorted({str(f.parent.relative_to(root)) for f in root.rglob("*.json")})
    for path in docs:
        anns = annotations(folder, path)
        by_id = {a.get("id"): a for a in anns}
        agent_wrote = [a for a in anns if _kind(a) == "written" and not _who(a.get("creator"))["person"]]
        for ann in anns:
            kind = _kind(ann)
            if kind == "decision":
                target = by_id.get(ann["target"].get("annotation"))
                if target is None:
                    continue
                who = _who(target.get("creator"))
                bernio = target.get("bernio") or {}
                out.append({
                    "type": "decision", "document": path, "at": _text(ann.get("created"), 40),
                    "about": {"by": "person" if who["person"] else "agent",
                              **({} if who["person"] else {"agent": who["name"], "model": who["model"]}),
                              "run": _text((target.get("generator") or {}).get("run"), 64)
                              if isinstance(target.get("generator"), dict) else "",
                              "kind": _kind(target), "quote": _text(_selector(target, "TextQuoteSelector").get("exact"),
                                                                    MAX_QUOTE),
                              "note": _body(target, "commenting"),
                              "suggestion": _body(target, "replacing", MAX_REPLACEMENT),
                              **{k: bernio[k] for k in ("aspect", "importance", "clause_type", "who", "what",
                                                        "to_whom", "when", "norm") if bernio.get(k) not in (None, "")}},
                    "verdict": (ann.get("bernio") or {}).get("verdict"),
                    "comment": _body(ann, "commenting"),
                    "correction": _body(ann, "replacing", MAX_REPLACEMENT),
                })
            elif kind == "revision" and agent_wrote:
                author = _who(agent_wrote[-1].get("creator"))
                out.append({"type": "revision", "document": path, "at": _text(ann.get("created"), 40),
                            "written_by": {"agent": author["name"], "model": author["model"]},
                            "changes": (ann.get("bernio") or {}).get("hunks") or []})
    return out


async def dataset(folder: Path, engine) -> list[dict]:
    """``records`` with every text in them tokenized by the scope's engine — the names in them read as
    the tokens a model sees, so the records can leave this Mac."""
    out = []
    for rec in records(folder):
        texts: list[str] = []

        def collect(v):
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    collect(x)
            elif isinstance(v, list):
                for x in v:
                    collect(x)
        collect(rec)
        reds = await engine.tokenize_all(texts) if texts else []
        swap = iter(r.text for r in reds)

        def rebuild(v):
            if isinstance(v, str):
                return next(swap)
            if isinstance(v, dict):
                return {k: rebuild(x) for k, x in v.items()}
            if isinstance(v, list):
                return [rebuild(x) for x in v]
            return v
        out.append(rebuild(rec))
    return out
