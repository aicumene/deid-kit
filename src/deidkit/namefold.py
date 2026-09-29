# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Name folding, offset-preserving matching, and the vault's admission predicate.

WHY THIS EXISTS. The vault used to find its values with
``re.compile(re.escape(value), re.IGNORECASE).subn`` over the raw text. That predicate is
wrong in three separate ways at once, all three measured on real documents:

  * no word boundary — "Corporation" fires inside "Certificate of in|corporation|", and the
    payload that crosses to the cloud reads "Certificate of inPERSON_28";
  * no pipeline fold — "José Muñoz" is missed while "Jose Munoz" two lines up is caught, and
    "Sea ﬁnder" (U+FB01 ligature) is missed for an enrolled vessel;
  * no reconciliation — one space ("SEAFINDER" vs "SEA FINDER") or one codepoint (a U+0450
    "ѐ" in place of "е") mints a *new* token for a referent that already has one.

THE FOLD IS NOT A NEW ONE. :func:`deidkit.textmatch.normalize` is the pipeline-invariant
comparison form; a verbatim quote check decides what counts as the same text through it. A
second, private fold here would be a second answer to the same question. So :func:`fold`
reproduces ``normalize`` EXACTLY — ``fold(raw).text == normalize(raw)`` is asserted as a test —
and adds the one thing ``normalize`` cannot give a *replacer*: an offset per folded character
back to the raw byte range it came from, so the token lands on the original text.

WHAT THIS MODULE ADDS ON TOP, AND WHY IT IS NOT IN ``normalize``. :func:`confusable` folds
diacritics (š→s, ñ→n, ѐ→е). ``normalize`` deliberately does not, and must not: a quote check
built on it decides whether a quote was FABRICATED, and a checker that treats "Muñoz" and
"Munoz" as the same characters has stopped being a verbatim check. The privacy layer is asking
a different question — "might this string be this person?" — where a near-miss must be treated
as a hit, because the cost of a false positive is a redundant token and the cost of a false
negative is a name in clear text on someone else's server. Two questions, two predicates, one
shared base.

Pure: stdlib only. No ORM, no settings, no model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# 1. The fold, with offsets
# ─────────────────────────────────────────────────────────────────────────────
# Same two constants ``textmatch.normalize`` applies, imported by value so the two cannot
# drift silently: ``test_fold_reproduces_textmatch_normalize`` re-checks equality.
from deidkit.textmatch import _LINEBREAK_HYPHEN, _PUNCT_FOLD
from deidkit.textmatch import normalize as canonical_normalize

_WS_RE = re.compile(r"\s")

# 1→1 replacements for letters whose NFD carries no combining mark (so the generic
# mark-stripping below cannot reach them). Only length-preserving entries may appear here:
# the confusable fold must not move any offset. 'æ'/'œ'/'ß' are therefore absent.
_CONFUSABLE_SINGLETON = {
    "ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "t", "ħ": "h", "ı": "i", "ŧ": "t",
}

# Visually identical letters from other scripts, mapped onto their Latin twin. All 1:1, so
# the length-preserving contract holds and the offset arrays stay valid.
#
# MEASURED on real documents: a company registration identifier written with Latin "HE" appeared
# in one document as "ΗΕ" (U+0397 U+0395), so the identifier crossed
# in clear beside the token of the company it identifies. The Greek entries are keyed on the
# CAPITAL's appearance and written in lowercase because ``fold`` casefolds first: Η→H means
# η→h here. That is not how a Greek reader sees η, and it does not have to be — both the
# needle and the haystack go through this same map, so a genuinely Greek word still matches
# itself. What the map buys is that a Latin string spelled with Greek letters stops hiding.
_HOMOGLYPH = {
    # Greek → Latin
    "α": "a", "β": "b", "γ": "y", "ε": "e", "ζ": "z", "η": "h", "ι": "i", "κ": "k",
    "μ": "m", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "υ": "y", "χ": "x", "ϲ": "c",
    # Cyrillic → Latin. Only the letters a Latin reader cannot tell apart.
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "ѕ": "s",
    "і": "i", "ј": "j", "ԁ": "d", "ѵ": "v", "ԛ": "q", "ԝ": "w",
}

# Cyrillic letters whose "diacritic" is a letter of its own. 'ё'→'е' is a fold Russian
# typography performs freely; 'й'→'и' is NOT — 'й' is a distinct letter and folding it
# merges unrelated names that differ only in it, so it is pinned through unchanged.
_CONFUSABLE_KEEP = {"й"}


@dataclass(slots=True)
class Folded:
    """A folded string plus, per folded character, the raw half-open span it came from."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    raw_len: int

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Raw half-open span covering folded[start:end]."""
        if start >= end:
            raise ValueError("empty span")
        return self.starts[start], self.ends[end - 1]


def _clusters(raw: str, skip: set[int]) -> list[tuple[str, int, int]]:
    """Split ``raw`` into (text, start, end) clusters: a base character plus any combining
    marks that follow it. Clustering is what makes per-piece NFKC safe — composition only
    happens between a base and its marks, so normalising each cluster gives the same string
    as normalising the whole, while every output character keeps a raw span."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(raw)
    while i < n:
        if i in skip:
            i += 1
            continue
        start = i
        i += 1
        while i < n and i not in skip and (
            unicodedata.combining(raw[i]) != 0 or unicodedata.category(raw[i]) in ("Mn", "Mc", "Me")
        ):
            i += 1
        out.append((raw[start:i], start, i))
    return out


def fold(raw: str) -> Folded:
    """``textmatch.normalize(raw)``, with a raw span recorded for every folded character.

    The steps, in ``normalize``'s order: drop hyphen-before-linebreak, NFKC, drop format
    characters, fold punctuation shapes, collapse whitespace, casefold.
    """
    raw = raw or ""
    skip: set[int] = set()
    for m in _LINEBREAK_HYPHEN.finditer(raw):
        skip.update(range(m.start(), m.end()))

    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending_ws: tuple[int, int] | None = None

    def emit(ch: str, s: int, e: int) -> None:
        chars.append(ch)
        starts.append(s)
        ends.append(e)

    for piece, s, e in _clusters(raw, skip):
        for ch in unicodedata.normalize("NFKC", piece):
            if unicodedata.category(ch) == "Cf":
                continue
            ch = _PUNCT_FOLD.get(ord(ch), ch)
            if _WS_RE.match(ch):
                # collapse: remember the run, emit one space when a non-space arrives
                pending_ws = (pending_ws[0], e) if pending_ws else (s, e)
                continue
            if pending_ws is not None:
                if chars:  # leading whitespace is stripped, not emitted
                    emit(" ", pending_ws[0], pending_ws[1])
                pending_ws = None
            for out_ch in ch.casefold():
                emit(out_ch, s, e)
    # trailing whitespace: ``normalize`` strips it, so ``pending_ws`` is dropped.
    return Folded(text="".join(chars), starts=tuple(starts), ends=tuple(ends), raw_len=len(raw))


def diacritics(folded_text: str) -> str:
    """Length-preserving DIACRITIC fold only (š→s, ć→c, ѐ→е, ł→l). Script is preserved.

    This is the fold the IDENTITY layer wants. :func:`skeleton_part` transliterates Cyrillic
    into Latin with its own table ("в"→"v"), and the glossary reads Russian surname morphology
    ("-ов" is masculine); both would be corrupted by a fold that had already rewritten "в" as
    the Latin letter it merely resembles. Two questions, two folds — see :func:`confusable`.
    """
    out = []
    for ch in folded_text:
        if ch in _CONFUSABLE_KEEP:
            out.append(ch)
            continue
        rep = _CONFUSABLE_SINGLETON.get(ch)
        if rep is not None:
            out.append(rep)
            continue
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if unicodedata.combining(c) == 0)
        out.append(base[0] if len(base) == 1 else ch)
    return "".join(out)


def confusable(folded_text: str) -> str:
    """The MATCH-path fold, applied to an already-:func:`fold`ed string.

    CONTRACT (load-bearing, asserted by ``test_confusable_is_one_to_one``):

      * **1:1 and length-preserving.** Exactly one output character per input character, in
        order, so a caller may keep the offset arrays from :func:`fold` and index them with
        positions in *this* string. Nothing is inserted, nothing is deleted.
      * **What it does.** Diacritics are dropped where NFD yields a single base character;
        letters whose NFD carries no mark are replaced through
        :data:`_CONFUSABLE_SINGLETON`; and Greek/Cyrillic letters that are visually
        indistinguishable from a Latin letter are replaced by that Latin letter
        (:data:`_HOMOGLYPH`). "й" is pinned through unchanged — it is a letter of its own and
        folding it merges two names that differ only in it.
      * **What it deliberately does NOT do.** It cannot remove a COMBINING MARK, because that
        would change the length. Turkish "İ".casefold() is 'i' + U+0307, so the mark exists by
        the time this function runs; removing it is :func:`strip_marks`, which runs afterwards
        and repairs the offsets. This function must never grow a case where one input
        character maps to zero or two output characters.
      * **Every constant compared against its output is folded through it** at import time
        (legal forms, patronymic tails, generic/structure/role words), so a table written in
        Cyrillic still matches text this function has rewritten.
    """
    out = []
    for ch in folded_text:
        if ch in _CONFUSABLE_KEEP:
            out.append(ch)
            continue
        rep = _HOMOGLYPH.get(ch) or _CONFUSABLE_SINGLETON.get(ch)
        if rep is not None:
            out.append(rep)
            continue
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if unicodedata.combining(c) == 0)
        if len(base) == 1:
            out.append(_HOMOGLYPH.get(base, base))
        else:
            out.append(ch)
    return "".join(out)


_MARK_CATS = ("Mn", "Mc", "Me")


def strip_marks(text: str, folded: Folded | None = None) -> tuple[str, Folded | None]:
    """Drop combining marks, absorbing each into the PRECEDING character's raw span.

    :func:`confusable` cannot do this: it is 1:1 by contract and a mark has no base to merge
    into without shortening the string. But the marks must go, because ``str.casefold``
    CREATES them — Turkish "İ" (U+0130) expands to 'i' + U+0307, so
    ``key("TARVELO DENİZCİLİK")`` and ``key("Tarvelo Denizcilik")`` differed by five
    invisible characters and the registered owner of a tokenised vessel crossed in clear.

    Offsets stay valid because a dropped mark's raw span is folded into the span of the
    character it decorates: the replacement still covers every raw byte it consumed.
    """
    if not any(unicodedata.category(c) in _MARK_CATS for c in text):
        return text, folded
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch) in _MARK_CATS and chars:
            if folded is not None:
                ends[-1] = max(ends[-1], folded.ends[i])
            continue
        chars.append(ch)
        if folded is not None:
            starts.append(folded.starts[i])
            ends.append(folded.ends[i])
    out = "".join(chars)
    if folded is None:
        return out, None
    return out, Folded(text=out, starts=tuple(starts), ends=tuple(ends),
                       raw_len=folded.raw_len)


def key(value: str) -> str:
    """The vault's match key for a needle: canonical fold, confusable fold, marks dropped."""
    return strip_marks(confusable(canonical_normalize(value)))[0]


def fold_haystack(raw: str) -> tuple[str, Folded]:
    """(searchable text, offsets) for a document, folded exactly as :func:`key` folds a needle."""
    f = fold(raw)
    text, f2 = strip_marks(confusable(f.text), f)
    return text, (f2 or f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. The boundary predicate
# ─────────────────────────────────────────────────────────────────────────────

def _is_word_char(ch: str) -> bool:
    """Word character for boundary purposes: any letter, digit, or combining mark.

    ``\\b`` is not used. In Python ``\\b`` on ``str`` patterns is in fact Unicode-aware, so
    Cyrillic would work here — but the predicate we need is not ``\\b``: it must be applied
    only on the side where the VALUE's own edge is a word character, or every value that
    legitimately ends in punctuation ("Northwind Investments Ltd.", "S.B. Management") stops
    matching the moment a letter follows it.
    """
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


_ESCAPE_LETTERS = frozenset("ntrfbv")
_HEX = frozenset("0123456789abcdefABCDEF")


def _odd_backslashes_before(hay: str, i: int) -> bool:
    """Is ``hay[i]`` a backslash that starts an escape (preceded by an even run of backslashes)?"""
    n = 0
    while i >= 0 and hay[i] == "\\":
        n += 1
        i -= 1
    return n % 2 == 1


def after_escape(hay: str, start: int) -> bool:
    """Does the text before ``start`` end in a backslash escape (``\\n``, ``\\t``, ``\\u00a0``)?

    MEASURED 28.09.2026: a coding agent's tool output arrived as JSON inside a text field, so a
    letter's line breaks were the two characters ``\\`` and ``n``. "…Dear Ms Voss,\\n\\nBrightwater
    Maritime Ltd asks…" put the letter ``n`` right before the company's name, the left boundary
    read it as part of a word, and the name crossed in clear four times. An escape stands for
    the whitespace it encodes and separates words the same way. An escaped backslash followed by
    a letter (``\\\\n``) is a backslash and a letter, not a line break."""
    if start >= 2 and hay[start - 1] in _ESCAPE_LETTERS and \
            _odd_backslashes_before(hay, start - 2):
        return True
    return start >= 6 and hay[start - 5] in "uU" and all(c in _HEX for c in hay[start - 4:start]) \
        and _odd_backslashes_before(hay, start - 6)


def boundary_ok(hay: str, start: int, end: int, needle: str) -> bool:
    """True when hay[start:end] is a standalone occurrence of ``needle``.

    Asymmetric by design: the boundary is enforced only on a side whose edge character in the
    NEEDLE is a word character. "Certificate of incorporation" contains "corporation" with a
    letter before it and the needle starts with a letter → rejected. "(Hallberg)," has
    punctuation on both sides → accepted. "Ltd." followed by a letter → accepted, because the
    needle's own last character is a full stop and demanding a boundary after it would reject
    every real occurrence. A backslash escape before the needle (``\\n``) is a boundary: see
    :func:`after_escape`.
    """
    if needle and _is_word_char(needle[0]) and start > 0 and _is_word_char(hay[start - 1]) \
            and not after_escape(hay, start):
        return False
    if needle and _is_word_char(needle[-1]) and end < len(hay) and _is_word_char(hay[end]):
        return False
    return True


def find_all(hay: str, needle: str) -> list[tuple[int, int]]:
    """Every standalone occurrence of ``needle`` in ``hay`` (both already folded)."""
    if not needle:
        return []
    out: list[tuple[int, int]] = []
    i = hay.find(needle)
    while i != -1:
        if boundary_ok(hay, i, i + len(needle), needle):
            out.append((i, i + len(needle)))
        i = hay.find(needle, i + 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Identity: script-independent skeletons for MERGE decisions
# ─────────────────────────────────────────────────────────────────────────────

_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "iu",
    "я": "ia", "і": "i", "ї": "i", "є": "e", "ґ": "g",
}

# Applied to the transliterated form, in order. Each rule erases a distinction that Russian
# romanisation systems disagree about and nothing else: for one final letter BGN/PCGN writes
# "-iy" where ISO writes "-ij" and many passport offices write "-y".
_SKEL_RULES = (
    ("kh", "h"), ("zh", "j"), ("shch", "sh"), ("tch", "ch"), ("ck", "k"),
    ("yo", "e"), ("ye", "e"), ("ya", "ia"), ("yu", "iu"),
    ("j", "i"), ("y", "i"), ("ij", "i"), ("ii", "i"), ("ie", "e"), ("ei", "i"),
)

_NONALNUM = re.compile(r"[^0-9a-zA-Zа-яёA-ZА-ЯЁ]+", re.UNICODE)


def translit(s: str) -> str:
    return "".join(_CYR2LAT.get(ch, ch) for ch in s)


def skeleton_part(part: str) -> str:
    """Script- and romanisation-independent skeleton of ONE name part."""
    s = diacritics(canonical_normalize(part))
    s = translit(s)
    s = _NONALNUM.sub("", s)
    for a, b in _SKEL_RULES:
        s = s.replace(a, b)
    # collapse doubled letters ("hallberg" → "halberg")
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def name_parts(name: str) -> list[str]:
    """Name split into parts, punctuation-insensitive, order preserved."""
    return [p for p in re.split(r"[\s,./\\()\"'’]+", (name or "").strip()) if p]


def skeleton(name: str) -> tuple[str, ...]:
    return tuple(sorted(p for p in (skeleton_part(x) for x in name_parts(name)) if p))


def stem_match(a: str, b: str) -> bool:
    """Two skeleton parts denote the same name part, allowing an inflectional tail.

    Russian inflects names (a surname in "-ов" takes "-ова", "-овым", …), so an exact
    comparison of parts under-merges. The allowance is a bounded PREFIX: the shorter must be a
    prefix of the longer, at least 4 characters long, and the tail at most 3 — enough for
    every Russian case ending, short of merging two different names that happen to share a
    stem of four.
    """
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return len(lo) >= 4 and hi.startswith(lo) and len(hi) - len(lo) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. Surface variants enrolled at seed time
# ─────────────────────────────────────────────────────────────────────────────

# Legal-form words, grouped by the form they denote. Members of a group are interchangeable
# surfaces of the SAME company; the presence of a form is also what distinguishes
# "Quorvane Shipping" (a bare name, safe to enrol as an alias) from "Red River Trading Ltd"
# vs "Red River Trading B.V." (two legal persons that must never merge).
_LEGAL_FORMS: tuple[tuple[str, ...], ...] = (
    ("ltd", "ltd.", "limited", "co ltd", "co. ltd", "co.ltd"),
    ("llc", "l.l.c.", "l.l.c", "lc"),
    ("inc", "inc.", "incorporated"),
    ("corp", "corp.", "corporation"),
    ("plc", "p.l.c."),
    ("gmbh",), ("ug",), ("ag",), ("kg",), ("se",),
    # German partnerships and the English LLP — a partnership and a limited company are treated
    # differently by many rules, so the glossary must be able to name the form (23.09.2026)
    ("gbr",), ("ohg",), ("kgaa",), ("partg", "partgmbb"), ("llp", "l.l.p."), ("lp", "l.p."),
    ("bv", "b.v.", "b v"), ("nv", "n.v."),
    ("sa", "s.a."), ("sl", "s.l."), ("srl", "s.r.l."), ("spa", "s.p.a."),
    ("dmcc",), ("fzco",), ("fze",), ("pjsc",), ("jsc",), ("pte",), ("oy",), ("ab",),
    ("as", "a/s"), ("doo", "d.o.o.", "d.o.o"), ("ad",), ("sirketi",), ("anonim",),
    ("ооо",), ("оао",), ("зао",), ("пао",), ("ао",),
)
# Folded through ``key`` at import: ``legal_form_index`` looks the folded part up here, and a
# table written in Cyrillic ("ооо") must survive the homoglyph fold that rewrites its letters.
_FORM_OF: dict[str, int] = {}
for _i, _grp in enumerate(_LEGAL_FORMS):
    for _w in _grp:
        _FORM_OF[key(_w)] = _i

_STRUCTURE_WORDS = {
    # Places and structures. A PERSON span containing one of these is a location that the
    # NER mis-typed, and locations must survive the crossing untouched.
    "centre", "center", "tower", "building", "house", "street", "road", "avenue", "plaza",
    "square", "floor", "district", "region", "city", "port", "airport", "hotel", "island",
    "county", "province", "bay", "harbour", "harbor", "station", "park", "bridge",
    "улица", "дом", "здание", "центр", "башня", "площадь", "район", "город", "порт",
    "область", "край", "проспект", "набережная", "остров",
    # German (23.09.2026: the vault had never seen a German file)
    "straße", "strasse", "str", "weg", "platz", "allee", "ufer", "damm", "ring", "gasse",
    "chaussee", "wohnung", "etage", "stock", "og", "eg", "haus", "gebäude", "stadt",
}

# Role nouns. A role is not a name: nine tokens in one real vault were declensions of two
# Russian role nouns.
_ROLE_STEMS = (
    "истец", "истц", "ответчик", "ответчиц", "заявител", "заинтересован", "третье лицо",
    "должник", "взыскател", "потерпевш", "свидетел", "представител", "доверител",
    "claimant", "respondent", "plaintiff", "defendant", "applicant", "appellant",
    "petitioner", "witness", "trustee", "liquidator", "administrator",
    "kläger", "beklagt", "antragstell", "antragsgegn", "schuldner", "gläubiger", "zeug",
    "vermieter", "mieter", "geschäftsführ", "gesellschaft", "insolvenzverwalt", "betreuer",
)

_GENERIC_WORDS = {
    "client", "husband", "wife", "spouse", "company", "entity", "management", "shipping",
    "corporation", "holding", "group", "draft", "roadmap", "overview", "shareholding",
    "shareholdings", "director", "manager", "owner", "shareholder", "unknown", "n/a",
    "клиент", "муж", "жена", "супруг", "супруга", "компания", "общество", "подача",
    "уплатить", "предварительное", "давности",
}

#: Forms of address and academic titles. They PRECEDE a name and are not part of it: from
#: "Herrn Paul Winter" the splitter enrolled "Herrn" as Winter's given name, so every "Herrn"
#: in the scope would have crossed as PERSON_2 (23.09.2026).
_HONORIFICS = {
    "herr", "herrn", "frau", "fräulein", "dr", "prof", "professor", "mr", "mrs", "ms", "miss",
    "mme", "mlle", "sr", "sra", "sig", "sig.ra", "sir", "dame", "lord", "lady", "hon",
    "господин", "госпожа", "г-н", "г-жа",
}

_PATRONYMIC_TAILS = (
    "ович", "евич", "ьич", "овна", "евна", "ична", "инична",
    "ovich", "evich", "yevich", "ayevich", "ovna", "evna", "ichna",
)

# Ordinary vocabulary. A residual span whose every part is one of these is a phrase, not a
# name — MEASURED: "Forwarded Message", "Container Ship", "Machine Translated" and
# "C.  Verifying" all became PEOPLE, and the glossary then certified each as an "individual
# (natural person)". It is also the guard on enrolling an organisation's leading word: coined
# words ("Quorvane", "Tarvelo", "Brevanco") are safe to enrol on their own, dictionary words
# ("Silver", "Red", "Prosperity") are not.
_COMMON_WORDS = {
    # structure of a document / an email
    "forwarded", "message", "original", "subject", "from", "sent", "to", "cc", "bcc",
    "attachment", "attached", "reply", "regards", "best", "dear", "kind", "sincerely",
    "translated", "machine", "translation", "page", "note", "notes", "summary", "memo",
    "confidential", "privileged", "draft", "final", "version", "appendix", "annex",
    "br", "fyi", "asap", "re", "fwd", "ps", "nb",
    "table", "figure", "section", "article", "clause", "schedule", "exhibit",
    # generic commercial nouns
    "container", "ship", "vessel", "cargo", "bulk", "carrier", "fleet", "charter",
    "trading", "trade", "trades", "invoice", "payment", "account", "bank", "loan",
    "purchase", "sale", "seller", "buyer", "lease", "asset", "assets", "share", "shares",
    "capital", "profit", "loss", "value", "price", "total", "amount", "balance",
    "registration", "register", "certificate", "licence", "license", "agreement",
    "contract", "letter", "notice", "claim", "court", "case", "matter", "law", "legal",
    "verifying", "keeping", "checking", "confirming", "pending", "unconfirmed",
    "yes", "no", "and", "or", "the", "of", "in", "on", "at", "for", "with", "without",
    "debts", "debt", "assets", "annual", "year", "years", "other", "profit", "loss",
    "taxes", "tax", "rules", "code", "cash", "equity", "reserves", "income", "expenses",
    # colours and adjectives that turn up in company names and must NOT be enrolled alone
    "white", "black", "green", "blue", "red", "golden", "silver", "royal", "grand",
    "global", "international", "national", "general", "united", "new", "old", "great",
    "first", "second", "third", "north", "south", "east", "west", "central", "prosperity",
    "ocean", "sea", "river", "lake", "rock", "star", "sun", "moon", "sky", "wave",
    # German articles, pronouns and the vocabulary of a letter and a tenancy — a span made only
    # of these is a phrase ("Die Wohnung", "Eine Kündigung", "Ihr Recht"), not a name
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "ihr", "ihre", "ihrer", "ihren", "ihrem", "sie", "wir", "unser", "unsere", "sehr", "geehrte",
    "geehrter", "mit", "freundlichen", "grüßen", "grüssen", "anlage", "schreiben", "betreff",
    "kündigung", "wohnung", "recht", "rechte", "mietvertrag", "mietverhältnis", "miete",
    "eigenbedarf", "widerspruch", "härte", "frist", "termin", "urteil", "beschluss", "antrag",
    "klage", "vertrag", "vereinbarung", "gesellschafterin", "gesellschafter", "hinweis",
    # template placeholders — "Insert Title", "Insert Rate" became people on 19.09.2026
    "insert", "title", "rate", "vat", "address", "date", "name", "placeholder", "tbd", "tbc",
}
_COMMON_FOLDED = {key(w) for w in _COMMON_WORDS}
_GENERIC_FOLDED = {key(w) for w in _GENERIC_WORDS}
_STRUCTURE_FOLDED = {key(w) for w in _STRUCTURE_WORDS}
_ROLE_STEMS_FOLDED = tuple(key(w) for w in _ROLE_STEMS)
_HONORIFICS_FOLDED = {key(w) for w in _HONORIFICS}


def is_honorific(part: str) -> bool:
    return key(part.rstrip(".")) in _HONORIFICS_FOLDED


def is_common_word(part: str) -> bool:
    """True when ``part`` is ordinary vocabulary rather than a coined name."""
    k = key(part)
    return bool(k) and (k in _COMMON_FOLDED or k in _GENERIC_FOLDED or k in _STRUCTURE_FOLDED
                        or any(k.startswith(stem) for stem in _ROLE_STEMS_FOLDED))


def legal_form_index(part: str) -> int | None:
    return _FORM_OF.get(key(part))


def has_legal_form(name: str) -> bool:
    return any(legal_form_index(p) is not None for p in name_parts(name))


def strip_legal_form(name: str) -> list[str]:
    """Name parts with legal-form words removed ("Quorvane Shipping Ltd" → [Quorvane, Shipping])."""
    return [p for p in name_parts(name) if legal_form_index(p) is None]


_PATRONYMIC_TAILS_FOLDED = tuple(key(t) for t in _PATRONYMIC_TAILS)


def is_patronymic(part: str) -> bool:
    p = key(part)
    return len(p) >= 6 and p.endswith(_PATRONYMIC_TAILS_FOLDED)


def is_initial(part: str) -> bool:
    p = part.strip().rstrip(".")
    return len(p) == 1 and p.isalpha()


_WORDISH = re.compile(r"[^\W_]+(?:\.[^\W_]+)*\.?", re.UNICODE)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in (re.sub(r"\s{2,}", " ", x).strip() for x in values):
        k = key(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def org_variants(name: str) -> list[str]:
    """Extra surfaces of one organisation name that a document plausibly uses.

    Three families, each measured on real documents:
      * legal-form synonyms — a document writes "QUORVANE SHIPPING LIMITED" for a vault value
        of "Quorvane Shipping Ltd";
      * the bare name with the legal form dropped;
      * the space-free compact form — a document writes "SEAFINDER" for "SEA FINDER", and
        publishes its IMO token on the same line, which is how a token gets burned.

    Surgery is done on the ORIGINAL string, never on re-joined parts: "S.B. Management"
    re-joined is "S B Management", which matches nothing in documents that write the dots.
    """
    out: list[str] = []
    if not name.strip():
        return out
    spans = [(m.start(), m.end(), m.group()) for m in _WORDISH.finditer(name)]
    for s, e, word in spans:
        g = legal_form_index(word)
        if g is None:
            continue
        for syn in _LEGAL_FORMS[g]:
            out.append((name[:s] + syn + name[e:]).strip())
        bare = (name[:s] + name[e:]).strip().strip(",;.-").strip()
        if len(bare) >= 4:
            out.append(bare)
    base = out[-1] if out and len(out[-1]) < len(name) else name
    # The initialism and the plural are generated from the full name AND from the bare name,
    # because documents write both ("S.B. Shipping Ltd" and the domain label "sbshipping",
    # "SB ASIA" behind sbasia.example).
    heads: list[str] = []
    for src in _dedupe([name, base]):
        out += _plural_variants(src)
        out += _shortened_variants(src)
        heads += _initialism_variants(src)
    out += heads
    for src in _dedupe([name, base] + heads):
        compact = re.sub(r"\s+", "", src)
        if len(compact) >= 6 and compact != src:
            out.append(compact)
    return [v for v in _dedupe(out) if key(v) != key(name)]


# English number inflection on the SUBSTANTIVE words of a company name. MEASURED: the entity
# graph holds "ARVEN HOLDING LIMITED" and the corporate-structure table writes "Arven Holdings
# Ltd" — every other row of that table is a token and this one is the plaintext name.
# ``boundary_ok`` correctly refuses "arven holding" inside "arven holdings", so the gap is
# enrolment, not matching. Bounded and deterministic: one SUBSTANTIVE word at a time (the legal
# form is never touched — "Ltd" must not become "Ltds"), and only the regular -s.
def _plural_variants(name: str) -> list[str]:
    out: list[str] = []
    spans = [(m.start(), m.end(), m.group()) for m in _WORDISH.finditer(name)]
    subs = [(s, e, w) for s, e, w in spans if legal_form_index(w) is None]
    if len(subs) < 2:
        return out
    for s, e, w in subs:
        if len(w) < 4 or not w.isalpha() or not w.isascii():
            continue
        alt = w[:-1] if w.casefold().endswith("s") else w + ("S" if w.isupper() else "s")
        if len(alt) >= 3:
            out.append((name[:s] + alt + name[e:]).strip())
    return out


# Initialism contraction of the LEADING words. MEASURED: the vault holds "Silver Bay Shipping
# Ltd" and "S.B. Management", so the documents themselves prove S.B. = Silver Bay; they then
# write "S.B. Shipping Ltd", "S.B Shipping ltd" and the domain "sb-shipping.example" — nine
# occurrences, none of them enrolled, in the same sentence as PERSON_1's 100% holding. All
# three punctuation shapes are generated because documents use all three, and a
# punctuation-insensitive KEY would be a second, looser fold that the rest of the module does
# not have.
# The short form a document uses once it has introduced the company: "Red River Trading
# B.V." becomes "Red River BV" in the shareholder lists of three chunks. Exactly one word is
# dropped — the LAST substantive word — and the legal form is kept, which is what keeps the two
# Red River companies apart: "Red River BV" and "Red River Ltd" are different surfaces for
# different legal persons, and if two referents ever did generate the same one, the ambiguity
# path gives it a shared token rather than a guess.
def _shortened_variants(name: str) -> list[str]:
    spans = [(m.start(), m.end(), m.group()) for m in _WORDISH.finditer(name)]
    subs = [(s, e, w) for s, e, w in spans if legal_form_index(w) is None]
    if len(subs) < 3:
        return []
    s, e, _w = subs[-1]
    short = (name[:s] + name[e:]).strip()
    return [re.sub(r"\s{2,}", " ", short)] if len(short) >= 6 else []


def _initialism_variants(name: str) -> list[str]:
    spans = [(m.start(), m.end(), m.group()) for m in _WORDISH.finditer(name)]
    subs = [(s, e, w) for s, e, w in spans if legal_form_index(w) is None]
    out: list[str] = []
    for n in (2, 3):
        if len(subs) <= n:
            continue
        head = subs[:n]
        if any(len(w) < 3 or not w[:1].isupper() or not w.isalpha() for _, _, w in head):
            continue
        letters = [w[0].upper() for _, _, w in head]
        tail = name[head[-1][1]:]
        for joiner in (".".join(letters) + ".", ".".join(letters), "".join(letters)):
            out.append((joiner + tail).strip())
    return out


# Bounded Russian case endings for an enrolled Cyrillic name part. MEASURED: one person's
# surname was 9/9 covered in the nominative and 0/3 in the genitive — the genitive ("-ова"
# after "-ов") crossed in clear together with his initials, which is a complete Russian
# designation of the man. ``boundary_ok`` is right to refuse the nominative inside the genitive
# and must not be loosened; the surface simply has to be enrolled.
#
# The families are the ones Russian surnames actually belong to, and each rule replaces a
# known ending rather than appending to an arbitrary stem, so nothing is generated for a word
# whose shape says it is not a name of that family. A generated form no document contains
# costs one unused alias row — the same trade ``cyrillic_form`` already makes.
_RU_NOUN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # -ов / -ев / -ёв / -ин / -ын: «…ов» → «…ова», «…ову», «…овым», …
    ("ов", ("ова", "ову", "овым", "ове", "овой", "овы", "овых", "овыми")),
    ("ев", ("ева", "еву", "евым", "еве", "евой", "евы", "евых", "евыми")),
    ("ёв", ("ёва", "ёву", "ёвым", "ёве", "ёвой")),
    ("ин", ("ина", "ину", "иным", "ине", "иной", "ины", "иных")),
    ("ын", ("ына", "ыну", "ыным", "ыне", "ыной")),
    # adjectival surnames -ский / -цкий / -ый: «…ский» → «…ского», «…скому», …
    ("ский", ("ского", "скому", "ским", "ском", "ская", "ской", "скую")),
    ("цкий", ("цкого", "цкому", "цким", "цком", "цкая", "цкой", "цкую")),
    ("ый", ("ого", "ому", "ым", "ом", "ая", "ой", "ую")),
    # given names in -ий / -й: «…ий» → «…ия», «…ию», «…ием»; «…ай» → «…ая»
    ("ий", ("ия", "ию", "ием", "ии")),
    ("й", ("я", "ю", "ем", "е")),
    # feminine -ова / -ева / -ина and given names in -а/-я: «…а» → «…ы», «…е», «…у», …
    ("а", ("ы", "е", "у", "ой", "ою")),
    ("я", ("и", "е", "ю", "ей")),
    # given names / surnames ending in a consonant handled by the generic tail below
    ("ь", ("я", "ю", "ем", "е")),
)
_CYRILLIC = re.compile(r"^[а-яёА-ЯЁ][а-яёА-ЯЁ-]+$")


def russian_inflections(part: str) -> list[str]:
    """Declined surfaces of one Cyrillic name part (bounded, deterministic, never a word)."""
    p = (part or "").strip()
    if not _CYRILLIC.match(p) or len(p) < 5 or is_patronymic(p) or is_common_word(p):
        return []
    low = p.casefold()
    cased = (lambda s: s.upper()) if p.isupper() else (lambda s: s[:1].upper() + s[1:])
    out: list[str] = []
    for tail, repls in _RU_NOUN_RULES:
        if not low.endswith(tail):
            continue
        stem = low[: -len(tail)]
        if len(stem) < 3:
            continue
        out += [cased(stem + r) for r in repls]
        break
    else:
        # A bare consonant stem ("-ь" is caught above; "-ев" has its own rule). Nothing is
        # generated for a shape no rule recognises: guessing here is what enrols a word.
        return []
    return [v for v in _dedupe(out) if key(v) != key(p) and not is_common_word(v)]


# A Latin acronym's Cyrillic spelling. "NRTK" is written "НРТК" in a Russian-language document
# and the two are the same company; ``cyrillic_form`` refuses it because it is under five
# characters and an acronym is not a name to transliterate phonetically — it is letter for letter.
_LAT2CYR_LETTER = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "у", "z": "з",
}


def acronym_cyrillic(part: str) -> str | None:
    p = (part or "").strip()
    if not (3 <= len(p) <= 6) or not p.isascii() or not p.isalpha() or not p.isupper():
        return None
    return "".join(_LAT2CYR_LETTER[ch] for ch in p.casefold()).upper()


# Romanisation rules that disagree with each other. Each pair is a substitution the passport
# office, BGN/PCGN and ISO 9 make differently for the SAME Russian letter, so both spellings
# turn up in one document set: the entity graph holds a given name ending in "-y" and the
# documents write it with "-iy"; one source writes a surname with "-yev-" and another with
# "-ev-". Applied to whole name parts, one rule at a time, both directions.
# Contractions are applied everywhere in the part; expansions only at the edges. That
# asymmetry is what keeps the generator honest: "-yev-"→"-ev-" (ye→e, a real disagreement
# between romanisation systems) and a final y→iy are spellings real documents contain, while an
# interior expansion ("e"→"ye" inside a word) invents a spelling no document has ever written.
_CONTRACT_RULES = (
    ("iy", "y"), ("iy", "i"), ("ii", "i"), ("yi", "i"), ("ye", "e"), ("yo", "e"),
    ("kh", "h"), ("ij", "i"), ("j", "y"), ("ts", "c"), ("ck", "k"),
)
_FINAL_RULES = (("y", "iy"), ("y", "i"), ("i", "iy"), ("iy", "y"), ("ii", "y"))
_INITIAL_RULES = (("e", "ye"), ("ya", "ia"), ("ia", "ya"), ("yu", "iu"), ("iu", "yu"),
                  ("i", "y"), ("y", "i"))


def translit_variants(part: str) -> list[str]:
    """Alternative romanisations of one Latin name part (bounded, deterministic)."""
    low = part.casefold()
    if not low.isascii() or not low.isalpha() or len(low) < 4:
        return []
    cased = str.capitalize if part[:1].isupper() else str.lower
    out: list[str] = []
    for a, b in _CONTRACT_RULES:
        if a in low:
            v = low.replace(a, b)
            if len(v) >= 4 and v != low:
                out.append(cased(v))
    for a, b in _FINAL_RULES:
        if low.endswith(a):
            v = low[: -len(a)] + b
            if len(v) >= 4 and v != low:
                out.append(cased(v))
    for a, b in _INITIAL_RULES:
        if low.startswith(a):
            v = b + low[len(a):]
            if len(v) >= 4 and v != low:
                out.append(cased(v))
    return _dedupe(out)[:8]


# Latin → Cyrillic, longest digraph first. Enrols the Cyrillic spelling of a name the scope's
# graph only holds in Latin, so a Russian document is covered even where the Russian NER misses
# the span. A wrong guess enrols a string no document contains and costs nothing; the length
# floor keeps a wrong guess from being a short common word.
_LAT2CYR = (
    ("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("ch", "ч"), ("sh", "ш"),
    ("ts", "ц"), ("yu", "ю"), ("ya", "я"), ("ye", "е"), ("yo", "ё"), ("iy", "ий"),
    ("y", "й"), ("a", "а"), ("b", "б"), ("v", "в"), ("g", "г"), ("d", "д"), ("e", "е"),
    ("z", "з"), ("i", "и"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"),
    ("p", "п"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("f", "ф"), ("h", "х"),
    ("c", "к"), ("j", "й"), ("w", "в"), ("x", "кс"), ("q", "к"),
)
_VOWELS = set("aeiou")


def cyrillic_form(part: str) -> str | None:
    """The Cyrillic spelling of a Latin name part, or None when it is not worth guessing."""
    low = part.casefold()
    if not low.isascii() or not low.isalpha() or len(low) < 5:
        return None
    # A final "-y" is "-ий" after a consonant and "-й" after a vowel ("-ay" → "-ай"); without
    # this the table alone writes "-й" straight after a consonant.
    if low.endswith("y") and len(low) > 3 and low[-2] not in _VOWELS:
        low = low[:-1] + "iy"
    i, out = 0, []
    while i < len(low):
        for a, b in _LAT2CYR:
            if low.startswith(a, i):
                out.append(b)
                i += len(a)
                break
        else:
            return None
    s = "".join(out)
    return s.capitalize() if part[:1].isupper() else s if len(s) >= 5 else None


def person_variants(name: str) -> list[str]:
    """Extra whole-name surfaces of one person: surname-first order and initialised forms.

    These are the orders documents actually use ("Surname Given", "G. Surname").
    Order matters to a substring matcher in a way it does not to a human reader.
    """
    parts = [p for p in name_parts(name) if not is_initial(p)]
    if len(parts) < 2:
        return []
    out = [" ".join(parts[::-1])]
    given, surname = parts[0], parts[-1]
    if len(parts) >= 2 and len(surname) >= 4:
        out += [f"{given[0]}. {surname}", f"{surname} {given[0]}.", f"{given[0]}.{surname}"]
    if len(parts) == 3:  # given patronymic surname → given surname
        out += [f"{parts[0]} {parts[2]}", f"{parts[2]} {parts[0]}"]
    # Romanisation variants: substitute one part at a time, so "Henry Zielinski" also covers
    # "Henri Zielinski" without generating the cartesian product of every spelling.
    respelled: list[str] = []
    bparts = name_parts(name)
    for i, p in enumerate(bparts):
        if is_patronymic(p):
            continue
        for alt in translit_variants(p):
            respelled.append(" ".join(bparts[:i] + [alt] + bparts[i + 1:]))
    cyr = [c for c in (cyrillic_form(p) for p in parts) if c]
    if len(cyr) == len(parts) >= 2:
        respelled += [" ".join(cyr), " ".join(cyr[::-1]),
                      f"{cyr[0][0]}. {cyr[-1]}", f"{cyr[-1]} {cyr[0][0]}.",
                      f"{cyr[0][0]}.{cyr[-1]}"]
    return [v for v in _dedupe(out + respelled) if key(v) != key(name)]


_UMLAUT_OUT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})
_UMLAUT_IN = re.compile(r"([AaOoUu])([Ee])")
_UMLAUT_OF = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
#: Names ending so take an apostrophe in the genitive ("Albers'"), and an apostrophe is a boundary.
_NO_GENITIVE_S = ("s", "ß", "x", "z", "ce")


def spelling_variants(surface: str, *, genitive: bool = False) -> list[str]:
    """Other ways German text writes a name the vault holds (25.09.2026).

    * Without its umlauts — "Müller" as "Mueller", "MUELLER, JOERG": file names, text that came
      through e-mail or an older system, capitals. The key folds ü to u, so "Müller" always met
      "Muller"; "Mueller" it never met, and on the way to the cloud model it crossed as written.
    * With them — a "Mueller" enrolled from such a source meets the "Müller" of the letter.
    * For a person, the genitive — "Albrechts Wohnung": the s is part of the word, so the boundary
      rejected the match and the name crossed. It is consumed INTO the placeholder rather than
      left after it: "PERSON_1s" is no placeholder the return leg can see (a Latin letter after the
      digits), "PERSON_1" is restored to the surface the text used.

    Matching keys only — the vault never enrols these (``vault._Index.variants``).
    """
    s = (surface or "").strip()
    if not s:
        return []
    forms = {s}
    if any(c in s for c in "äöüÄÖÜ"):
        forms.add(s.translate(_UMLAUT_OUT))
    if _UMLAUT_IN.search(s):
        forms.add(_UMLAUT_IN.sub(lambda m: _UMLAUT_OF[m.group(1)], s))
    if genitive:
        for f in list(forms):
            last = f.split()[-1]
            if last[-1:].isalpha() and not last.casefold().endswith(_NO_GENITIVE_S):
                forms.add(f + "s")
    return [f for f in _dedupe(sorted(forms)) if key(f) != key(s)]


def person_fragments(name: str) -> list[tuple[str, str]]:
    """Distinctive single parts of a person name, as (surface, role).

    Role is 'surname' for the last part and 'given' for the others. Initials, particles and
    patronymics are excluded: a patronymic is shared by thousands of people, so enrolling a
    patronymic ("-ovich") as a surface of one man is a false positive waiting for the next
    document.

    DECLENSIONS ARE NOT GENERATED HERE. Russian inflects names, and a token APPENDED with an
    inflection already round-trips ("PERSON_1ым"), so the gap was purely enrolment — but an
    inflected spelling is not a NEW fragment to be adjudicated. A genitive in "-ова" has a
    different skeleton from its nominative in "-ов", so offering it here made the shared-surname
    logic mint a second shared token per case ending: fourteen PERSON rows where there should be
    six. The caller decides who a fragment belongs to ONCE, from the nominative, and then enrols
    that fragment's declensions as aliases of the same token — see ``vault._register_fragments``.
    """
    parts = [p for p in name_parts(name) if not is_initial(p) and not is_honorific(p)]
    parts = [p for p in parts if len(p) >= 4 and p[:1].isalpha()]
    out: list[tuple[str, str]] = []
    for i, p in enumerate(parts):
        if is_patronymic(p):
            continue
        if key(p) in _GENERIC_FOLDED:
            continue
        role = "surname" if i == len(parts) - 1 else "given"
        out.append((p, role))
        for alt in translit_variants(p):
            out.append((alt, role))
        cyr = cyrillic_form(p)
        if cyr:
            out.append((cyr, role))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. The admission predicate for residual (NER) detections
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_SHAPE = re.compile(r"^[A-Za-z]+_\d+$")

# The same shape, as a SEARCH rather than a match. ``_TOKEN_SHAPE`` is anchored, so it only
# ever caught a surface that is nothing but a token; a surface that CONTAINS one
# ("PERSON_1@example.org") walked straight past it and the vault stored a row whose real value
# is a live token. Anything that writes to the vault tests with this.
TOKEN_IN_TEXT = re.compile(r"(?<![0-9A-Za-z])[A-Za-z]+_\d+(?![0-9A-Za-z])")

_BRACKETS = {"(": ")", "[": "]", "{": "}"}

# Characters a model may substitute for an ASCII letter or digit inside a TOKEN it echoes: the
# Cyrillic/Greek capitals that are visually identical, and the fullwidth forms. 1:1 by
# construction, so ``str.translate`` leaves every offset where it was and a replacement can be
# spliced back into the ORIGINAL answer text. A model asked about Russian or Greek text often
# answers in that language, so a Cyrillic О inside "PERSON_1" is a plausible slip — and an
# unrecognised token reaches the reader as a raw token instead of a name.
_ASCII_CONFUSABLES = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Ԛ": "Q",
    "Ԝ": "W", "Г": "F", "Д": "D", "Л": "L", "Ф": "F",
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "о": "o", "р": "p", "с": "c",
    "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "y", "χ": "x",
}
for _d in range(10):  # fullwidth digits and the fullwidth low line
    _ASCII_CONFUSABLES[chr(0xFF10 + _d)] = str(_d)
_ASCII_CONFUSABLES["＿"] = "_"
for _c in range(26):
    _ASCII_CONFUSABLES[chr(0xFF21 + _c)] = chr(ord("A") + _c)
    _ASCII_CONFUSABLES[chr(0xFF41 + _c)] = chr(ord("a") + _c)
_ASCII_CONFUSABLE_TABLE = str.maketrans(_ASCII_CONFUSABLES)


def deconfuse_ascii(text: str) -> str:
    """1:1 map of ASCII-lookalike characters onto ASCII. Same length, same offsets."""
    return (text or "").translate(_ASCII_CONFUSABLE_TABLE)


def distinctive_head(name: str) -> str | None:
    """The leading word of an organisation name when that word identifies the company alone.

    "Quorvane Shipping Ltd" is referred to as "Quorvane", "NRTK Asia M6 Limited" as "NRTK",
    "Tarvelo Denizcilik Ticaret Anonim Sirketi" as "Tarvelo" — the pattern of three companies in
    one real document set whose bare name the OLD vault covered (as junk single-word PERSON rows)
    and the precision fix dropped. The guard is that the word must be COINED: "Silver Bay
    Shipping" must not enrol "Silver", and "Red River Trading" must not enrol "Red".
    """
    parts = [p for p in strip_legal_form(name) if p]
    if len(parts) < 2:
        return None
    head = parts[0].strip(".,;:-")
    if len(head) < 4 or not head[:1].isalpha() or is_common_word(head):
        return None
    if not all(ch.isalpha() or ch in "-'’" for ch in head):
        return None
    return head


def _balanced(s: str) -> bool:
    stack: list[str] = []
    for ch in s:
        if ch in _BRACKETS:
            stack.append(_BRACKETS[ch])
        elif ch in ")]}":
            if not stack or stack.pop() != ch:
                return False
    return not stack


def admit_person(surface: str) -> tuple[bool, str]:
    """Should a residual PERSON span become a token of its own? Returns (ok, reason).

    A real vault says what this has to stop: a country ("Argentina"), a building ("Marlow
    Centre"), a Russian infinitive, a Russian sentence fragment, nine declensions of two Russian
    role nouns, two spans with an unclosed bracket, and eleven common
    nouns. The gate is deliberately structural rather than a blocklist of the words that
    happened to leak — a blocklist would pass "Shareholdings" tomorrow, which is exactly what
    the live re-run of the OLD tokenizer minted.

    A SINGLE-WORD span is never admitted here. Every class-C row in one real vault but two is
    a single word, and a lone capitalised word is not evidence of a person. This does not
    stop bare surnames being *detected*: a bare surname resolves through the identity index
    to the person it belongs to (or to a shared-surname token), which is a different path.

    TWO CAPITALISED WORDS ARE NOT ENOUGH EITHER, and the second measured round says why: over
    real documents this gate admitted "Forwarded Message" (a UI string), "SEA RUNNER
    Container Ship" (a hull that is VESSEL_8 twenty lines up), "Machine Translated",
    "C.  Verifying" and "Agίoy Nikoλάoy" (a Greek street) — and the glossary then
    described each of them as an "individual (natural person)". Three structural additions,
    none of them a blocklist of the words that happened to leak:

      * a span that carries a LINE BREAK or a run of two spaces is a layout artefact, not a
        name — seven of the eight junk rows have one;
      * a span whose every part is ordinary vocabulary is a phrase (:func:`is_common_word`);
      * a span mixing two scripts inside one word is an OCR/homoglyph artefact.

    The two remaining classes — a span that IS an already-enrolled vessel or organisation, and
    a span in a script the document's NER was not running in — need the scope's index and the
    document language, so they are enforced by the caller (``deidkit.vault._admit``).
    """
    v = (surface or "").strip()
    if not v or len(v) < 3 or len(v) > 80:
        return False, "length"
    if _TOKEN_SHAPE.match(v):
        return False, "token-shaped"
    if TOKEN_IN_TEXT.search(v):
        return False, "contains a token"
    if not _balanced(v) or any(c in v for c in "()[]{}/\\|<>@"):
        return False, "malformed span"
    if any(ch.isdigit() for ch in v) or "~" in v:
        return False, "contains digits"
    if "\n" in surface or "\r" in surface or "\t" in surface or "  " in surface:
        return False, "line break or column gap in span"
    parts = [p for p in name_parts(v) if not is_honorific(p)]
    if len(parts) < 2:
        return False, "single word"
    if len(parts) > 4:
        return False, "too many parts for a name"
    substantive = [p for p in parts if not is_initial(p)]
    if substantive and all(is_common_word(p) for p in substantive):
        return False, "every part is an ordinary word"
    for p in parts:
        low = key(p)
        if low in _GENERIC_FOLDED or low in _STRUCTURE_FOLDED:
            return False, f"generic/structure word: {p}"
        if any(low.startswith(stem) for stem in _ROLE_STEMS_FOLDED):
            return False, f"procedural role noun: {p}"
        if not p[:1].isupper() and not is_initial(p):
            return False, f"uncapitalised part: {p}"
        if not all(ch.isalpha() or ch in "-'’." for ch in p):
            return False, f"non-name characters: {p}"
        if mixed_script(p):
            return False, f"mixed script within one word: {p}"
    return True, "ok"


def script_of(text: str) -> str | None:
    """The dominant alphabetic script of ``text``: 'latin', 'cyrillic', 'greek' or None."""
    counts = {"latin": 0, "cyrillic": 0, "greek": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        if ch.isascii():
            counts["latin"] += 1
        elif "Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿":
            counts["greek"] += 1
        elif "Ѐ" <= ch <= "ӿ":
            counts["cyrillic"] += 1
    top = max(counts, key=lambda k: counts[k])
    return top if counts[top] else None


def mixed_script(word: str) -> bool:
    """True when one WORD draws its letters from more than one alphabet.

    "Agίoy Nikoλάoy" is a Greek street name typed with Latin consonants and Greek accented
    vowels; no real name is written that way, and admitting it gave a street a PERSON token.
    """
    seen = set()
    for ch in word:
        if not ch.isalpha():
            continue
        if ch.isascii():
            seen.add("latin")
        elif "Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿":
            seen.add("greek")
        elif "Ѐ" <= ch <= "ӿ":
            seen.add("cyrillic")
    return len(seen) > 1


def is_generic(value: str) -> bool:
    """Role references that are already de-identified ("the husband") — never tokenised."""
    v = (value or "").strip().lower()
    return (not v) or v.startswith(("the ", "a ", "an ", "unknown")) or len(v) < 2
