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
  the model says. A kept note is one annotation a file, ``.legix/markup/<path>/<id>.json``: the matter's
  notes folder, which leGix versions, signs and syncs with the documents. Nothing in such a file is
  trusted on reading: the quote is looked up in the text again, so a note shows where its words are now,
  or that they are gone.
"""
from __future__ import annotations

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
- "note": one or two sentences for the lawyer, in the language they write to you in.

Answer with one JSON object and nothing else:
{{"notes": [{{"quote": "...", "kind": "risk", "aspect": "termination", "importance": 4, "clause_type": "", \
"who": "", "what": "", "to_whom": "", "when": "", "norm": "", "note": "..."}}]}}"""


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
def deid_spans(original: str, tokenised: str, surfaces: dict[str, list[str]],
               mapping: dict[str, str]) -> list[dict]:
    """Where each token of ``tokenised`` stands in ``original``: ``{start, end, type, token}``.

    The tokenised text is the original with surfaces put in tokens' places; what lies between tokens is
    the original's own. So the two are walked together: the literal piece before a token is matched in
    the original, and the token takes the surface — one the engine recorded for it in this text — that
    stands there. Should the walk lose step (a piece not where it was expected), it finds the piece again
    further on, so one odd place does not cost the rest of the document."""
    out: list[dict] = []
    at = 0                                         # in the original
    last = 0                                       # in the tokenised text
    tokens = list(TOKEN.finditer(tokenised))
    for i, m in enumerate(tokens):
        token = m.group(0)
        if token not in mapping and token not in surfaces:
            continue                               # a token-like word of the document itself
        piece = tokenised[last:m.start()]
        if original.startswith(piece, at):
            at += len(piece)
        else:
            found = original.find(piece, at) if piece else at
            if found < 0:
                last = m.end()
                continue
            at = found + len(piece)
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
        out.append({"start": at, "end": at + len(surface), "type": token.rsplit("_", 1)[0], "token": token})
        at += len(surface)
        last = m.end()
    return out


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
    """The notes in an agent's answer, each with its fields in our vocabulary; None — no notes object in it."""
    data = _json_object(answer)
    if data is None or not isinstance(data.get("notes"), list):
        return None
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
    ``("moved", …)`` — found again elsewhere; None — the words are no longer in the text."""
    quote = _selector(ann, "TextQuoteSelector")
    exact = quote.get("exact") if isinstance(quote.get("exact"), str) else ""
    pos = _selector(ann, "TextPositionSelector")
    start, end = pos.get("start"), pos.get("end")
    if not exact:
        return None
    if isinstance(start, int) and isinstance(end, int) and folded.text[start:end] == exact:
        return "here", start, end
    found = folded.find(exact, near=start if isinstance(start, int) else None)
    return ("moved", *found) if found else None


def load(folder: Path, path: str, text: str) -> list[dict]:
    """The document's notes, each placed in the text as it is now. A file that is not an annotation of
    this document is left out; so is anything but our own fields of one."""
    folded = Folded.of(text)
    root = markup_dir(folder, path)
    out = []
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            ann = json.loads(f.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(ann, dict) or (ann.get("target") or {}).get("source") != path:
            continue
        where = place(ann, folded)
        bernio = ann.get("bernio") if isinstance(ann.get("bernio"), dict) else {}
        creator = ann.get("creator")
        body = [b for b in ann.get("body") or [] if isinstance(b, dict)]
        out.append({
            "id": _text(ann.get("id"), 64) or f.stem,
            "created": _text(ann.get("created"), 40),
            "creator": {"name": _text(creator.get("name"), 80), "model": _text(creator.get("model"), 80)}
            if isinstance(creator, dict) else {"name": _text(creator, 80), "model": ""},
            "run": _text((ann.get("generator") or {}).get("run"), 64) if isinstance(ann.get("generator"), dict) else "",
            "status": where[0] if where else "gone",
            "start": where[1] if where else None,
            "end": where[2] if where else None,
            "quote": _text(_selector(ann, "TextQuoteSelector").get("exact"), MAX_QUOTE),
            "clause": _text(_selector(ann, "FragmentSelector").get("value"), 80),
            "note": next((_text(b.get("value"), MAX_NOTE) for b in body if b.get("purpose") == "commenting"), ""),
            "kind": bernio.get("kind") if bernio.get("kind") in KINDS else "point",
            "aspect": bernio.get("aspect") if bernio.get("aspect") in ASPECTS else "other",
            "importance": bernio.get("importance") if bernio.get("importance") in (1, 2, 3, 4, 5) else 3,
            **{k: _text(bernio.get(k), 200) for k in ("clause_type", "who", "what", "to_whom", "when", "norm")},
        })
    return sorted(out, key=lambda n: (n["start"] is None, n["start"] or 0, -n["importance"]))


def keep(folder: Path, path: str, text: str, notes: list[dict], *, creator: dict, run: str) -> dict:
    """Keep the notes whose quotes are in the text; say of the others why not."""
    folded = Folded.of(text)
    kept, refused = [], []
    for note in notes:
        found = folded.find(note["quote"]) if note["quote"] else None
        if found is None:
            refused.append({"quote": note["quote"][:120], "why": "not in the document" if note["quote"] else "no quote"})
            continue
        ann = annotation(note, path=path, text=text, start=found[0], end=found[1], creator=creator, run=run)
        store(folder, path, ann)
        kept.append(ann["id"])
    return {"kept": kept, "refused": refused}
