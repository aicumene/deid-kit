# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The token glossary — what a token MEANS, sent out-of-band with the de-identified payload.

THE PROBLEM IT SOLVES. A de-identified payload strips exactly the things a reader reasons
with. "PERSON_1 holds 70% of ORG_2" tells a cloud model nothing about whether PERSON_1 is a
person or a company, which country's rules about him apply, or whether a sentence about
PERSON_1 should be read as about a man or a woman — and a language with grammatical gender
(Russian, where the past tense of a verb agrees with its subject) cannot even be composed
without knowing. Models fill that vacuum by guessing, and a guess about who is a natural person
is an error of substance, not of style.

THE SHAPE OF THE ANSWER. Attributes, not names. Every line describes ONE token using facts the
system already holds in the scope's seed graph (entities and parties) — nothing is asked of a
model, and nothing is invented. The return leg is untouched: the glossary is prepended to the
outbound prompt and the answer still comes back as tokens, so ``detokenize`` works exactly as
before.

WHAT IS DELIBERATELY NOT IN IT, and why each would undo the vault. A "de-identified" payload
that carries a re-identifier is worse than one that carries the name, because it reads as safe:

  * **exact shareholdings.** "70 / 20 / 5 / 5 of a Hong Kong shipping company" is a search
    query against any public registry. Bands are emitted instead — and the bands are drawn at
    10 / 25 / 50%, which are the thresholds EU beneficial-ownership and control tests actually
    turn on, so the legal reasoning survives and the fingerprint does not.
  * **identifiers of every kind** — IMO, MMSI, call sign, IBAN, company/tax registration
    number, passport. Each is unique by construction; each already has its own token.
  * **vessel flag, year built, valuation.** A bulker of a given build year, under a given flag,
    valued within a stated range is one ship in the world.
  * **addresses, postcodes, emails, phone numbers.** Street addresses survive in the payload
    by design (locations are not tokenised) — repeating one in a line that also says "this is
    PERSON_1's" is what turns a preserved location into an identification.
  * **dates of any kind, birth dates above all.** The design keeps dates in the text; the
    glossary must not bind one to a token.
  * **free-text notes** from an entity's notes and ``relationships[].note``. They are
    LLM-extracted prose that quotes the source document ("Manager with mandate starting
    2020-01-01"), so they smuggle in exactly the identifiers and dates the list above excludes.
  * **family relationships** ("son of PERSON_1"). They are only present in the graph as free
    text, and see above.

FAIL-CLOSED. Before returning, every line is run back through the scope's own match index. If
a line contains any enrolled real value — a name that leaked in through an attribute — the
line is DROPPED. A glossary that leaks the thing it describes is the one failure mode that
would make this feature worse than not having it.
"""

from __future__ import annotations

from dataclasses import dataclass

import re

from deidkit import namefold as nf

# Surname endings whose grammatical gender is unambiguous in Russian. This is morphology, not
# inference about a person: "-ova" is a feminine surname form, full stop. Endings that carry
# no gender (Serbian "-ić", Latin surnames generally) yield nothing, and nothing is guessed.
_FEMALE_TAILS = ("ova", "eva", "ina", "aya", "skaya", "ская", "ова", "ева", "ина", "ая")
_MALE_TAILS = ("ov", "ev", "in", "sky", "skiy", "ski", "ich", "ский", "ов", "ев", "ин", "ий")

_KIND_WORD = {
    "PERSON": "individual (natural person)",
    "ORG": "organisation",
    "VESSEL": "vessel",
    "ACCOUNT": "bank account",
    "ID": "registration/identity number",
    "IBAN": "bank account number",
    "EMAIL": "email address",
    "PHONE": "telephone number",
    "CARD": "payment card number",
    "IP": "IP address",
    "PROPERTY": "real property",
}

_RELATION_WORD = {
    "shareholder": "shareholder in",
    "beneficial_owner": "beneficial owner of",
    "owns": "owns",
    "controls": "controls",
    "parent_of": "parent of",
    "linked": "linked to",
}


def legal_form_of(name: str) -> str | None:
    """The legal form of an organisation's name as written ("GmbH", "Ltd", "LLP"), or None.

    A compound form is returned whole: "GmbH & Co. KG" is a limited partnership, not a GmbH,
    and naming only its first word would state the wrong form."""
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    idx = [i for i, p in enumerate(parts) if nf.legal_form_index(p.strip(",.;:()")) is not None]
    if not idx:
        return None
    return " ".join(parts[idx[0]: idx[-1] + 1]).strip(",;:()")


def _band(pct: object) -> str | None:
    """Shareholding band. Cut at the thresholds law uses, never the exact figure."""
    try:
        p = float(pct)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    if p > 100:
        return "a recorded stake outside 0–100% (source figure is inconsistent)"
    if p >= 50:
        return "≥50%"
    if p >= 25:
        return "25–49%"
    if p >= 10:
        return "10–24%"
    return "<10%"


def _gender_from_surfaces(surfaces: list[str]) -> str | None:
    for s in surfaces:
        parts = nf.name_parts(s)
        if not parts:
            continue
        # ``diacritics``, not ``confusable``: the match fold rewrites Cyrillic letters that
        # look Latin, and this is Russian MORPHOLOGY — "-ов" must still read as "-ов".
        last = nf.diacritics(parts[-1].casefold())
        if last.endswith(_FEMALE_TAILS):
            return "female"
        if last.endswith(_MALE_TAILS):
            return "male"
    return None


@dataclass(frozen=True)
class Wording:
    """How the glossary names what the tokens come from. A deployment sets its own once, with
    :func:`set_wording`; the defaults are neutral."""

    #: what the text was taken from, as in "derived from the source documents"
    source: str = "the source documents"
    #: the collection a token is shared within, as in "shared by 2 individuals in these documents"
    scope: str = "these documents"


_WORDING = Wording()


def set_wording(wording: Wording) -> None:
    """Set the words the glossary uses for its source and its scope (process-wide)."""
    global _WORDING
    _WORDING = wording


def wording() -> Wording:
    return _WORDING


def _script_note(surfaces: list[str]) -> str | None:
    scripts = set()
    for s in surfaces:
        for ch in s:
            if "Ѐ" <= ch <= "ӿ":
                scripts.add("Cyrillic")
            elif ch.isalpha() and ch.isascii():
                scripts.add("Latin")
    if len(scripts) > 1:
        return f"the name occurs in both Cyrillic and Latin script in {_WORDING.source}"
    return None


def build_glossary(index, mapping: dict[str, str], text: str = "") -> list[str]:
    """One line per token in ``mapping``, describing it without identifying it.

    ``index`` is the vault's :class:`deidkit.vault._Index` for the scope (duck-typed here so
    this module stays importable on its own). ``text`` is the payload the
    glossary will travel with — see :func:`_visible` and :func:`_banding_is_defeated`.

    THE LEGEND DESCRIBES WHAT CROSSED, NOT THE SCOPE. Measured over real documents, the first
    version emitted 286 lines naming 182 tokens that do not occur in the payload's own text —
    64% of every token reference it made — and 220 of those tokens were absent from ``mapping``,
    so the return leg could not re-hydrate a single one of them. One chunk with two tokens in
    its text handed over eight. Two rules follow, and they are the same rule twice:

      * a token gets a LINE only if it occurs in the text being sent;
      * a token is NAMED in someone else's line — as a relation target, an identifier owner, a
        shared-name member — only if it occurs in the text being sent.

    That makes the glossary total with respect to ``mapping`` by construction, which is what
    the return leg needs: a model that echoes a token it read in the glossary is echoing a
    token this request actually sent.
    """
    visible = _visible(mapping, text)
    # OBSERVED surfaces only. The seeder also generates spellings a document might use
    # (source='variant'); a glossary line that said "written in both scripts" on the strength of
    # a spelling WE invented would be the vault asserting a fact about the source documents
    # that they do not contain.
    #
    # 'fragment' is not observed either: the splitter adds a Cyrillic form of every name part
    # (a Cyrillic spelling of a German surname), and a German name was then certified as written
    # "in both Cyrillic and Latin script" (23.09.2026). Observed = the scope's graph
    # ('entity') and what the detector found in a document ('residual').
    surfaces_by_token: dict[str, list[str]] = {}
    for a in getattr(index, "aliases", []):
        if a.source in ("entity", "residual"):
            surfaces_by_token.setdefault(a.token, []).append(a.surface)

    lines: list[str] = []
    for token in sorted(visible, key=lambda t: (t.split("_")[0], int(t.split("_")[-1]))):
        row = index.by_token.get(token)
        if row is None:
            continue
        attrs = row.attributes or {}
        surfaces = [row.real_value] + surfaces_by_token.get(token, [])
        bits: list[str] = []

        shared = attrs.get("shared_by") or []
        if shared:
            # Resolve every member through ``canonical_token`` and keep only ACTIVE, VISIBLE
            # tokens. The raw list is a merge-ordering artefact: it named PERSON_20, which is
            # retired INTO PERSON_9, beside PERSON_9 — so the line asserted that one man is two
            # people, and a model dutifully wrote a table row saying so. If the resolved set has
            # fewer than two members the surface is not ambiguous at all and the line is not
            # emitted: an ambiguity nobody has is a false statement about the source documents.
            resolved: list[str] = []
            for t in shared:
                c = index.canonical(t) if hasattr(index, "canonical") else t
                crow = index.by_token.get(c)
                if (crow is not None and crow.status == "active"
                        and c != token and c not in resolved):
                    resolved.append(c)
            if len(resolved) < 2:
                continue
            what = attrs.get("fragment") or "name"
            kept = [t for t in resolved if t in visible]
            if len(kept) >= 2:
                lines.append(
                    f"{token} = the {what} is shared by {', '.join(kept)}; the source text does "
                    f"not say which of them is meant — do not attribute it to one of them"
                )
            else:
                # The members did not cross in THIS payload, so naming them would hand the
                # model tokens the return leg cannot reverse. The warning is the part that
                # matters and it survives without them: what the reader must not do is pin the
                # sentence on one referent.
                noun = {"PERSON": "individuals"}.get(row.kind, "referents")
                lines.append(
                    f"{token} = a {what} shared by {len(resolved)} different {noun} in "
                    f"{_WORDING.scope}; the source text does not say which of them is meant — do not "
                    f"attribute it to any one of them"
                )
            continue

        if row.kind == "PERSON" and not _person_is_corroborated(row, surfaces):
            # "PERSON_4 = individual (natural person)" was emitted for the string "Forwarded
            # Message", and "PERSON_64 = individual (natural person)" for a hull that is
            # VESSEL_8 in the same payload. A glossary line is an ASSERTION; with no entity
            # behind the token and no name-shaped morphology there is nothing to assert, and
            # emitting nothing is strictly better than emitting something false.
            continue

        bits.append(_KIND_WORD.get(row.kind, row.kind.lower()))
        if row.kind == "ORG":
            # THE LEGAL FORM CROSSES. It is a fact about the organisation, not an identifier — a
            # GmbH is one of 1.3 million — and it can decide the answer: a partnership and a
            # limited company are treated differently by many rules. "organisation" alone left
            # the model to guess which of the two ORG_1 was (23.09.2026).
            form = legal_form_of(row.real_value)
            if form:
                bits[-1] = f"{bits[-1]} ({form})"
        if attrs.get("also_kind"):
            bits.append(
                f"a {_KIND_WORD.get(attrs['also_kind'], attrs['also_kind'].lower())} of the "
                f"same name also appears in {_WORDING.scope}"
            )
        if row.kind == "PERSON":
            g = _gender_from_surfaces(surfaces)
            if g:
                bits.append(f"{g} (from surname morphology)")
        if attrs.get("jurisdiction"):
            bits.append(f"jurisdiction {attrs['jurisdiction']}")
        roles = [r for r in (attrs.get("roles") or []) if r]
        if roles:
            bits.append("role: " + "; ".join(roles[:3]))
        note = _script_note(surfaces)
        if note:
            bits.append(note)
        if attrs.get("identifier_of_name"):
            owner = index.token_for(str(attrs["identifier_of_name"]))
            if owner and owner in visible:
                bits.append(f"identifier of {owner}")
            else:
                bits.append(f"identifier of a party in {_WORDING.scope}")

        banded_out = _banding_is_defeated(text, token)
        seen_rel: set[str] = set()
        for rel in (attrs.get("relations") or [])[:8]:
            target = index.token_for(str(rel.get("target_name") or ""))
            if not target or target not in visible:
                continue  # only ever name a counterparty that crossed in THIS payload
            verb = _RELATION_WORD.get(str(rel.get("type")), str(rel.get("type")))
            band = _band(rel.get("pct"))
            if band and banded_out:
                # A band beside its own raw number is decoration. The payload that carried
                # "PERSON_1 = … shareholder in ORG_2 (≥50%)" also carried "PERSON_1 70%" four
                # lines down in clear, so the 70/20/5/5 fingerprint the band was drawn to
                # withhold was in the same envelope. Where the text supplies the figure, the
                # clause is dropped rather than dressed up.
                continue
            phrase = f"{verb} {target}" + (f" ({band})" if band else "")
            if phrase in seen_rel or f"{verb} {target}" in seen_rel:
                continue
            seen_rel.add(phrase)
            seen_rel.add(f"{verb} {target}")
            bits.append(phrase)

        lines.append(f"{token} = " + ", ".join(bits))

    return _drop_unmappable_lines(_drop_leaky_lines(index, lines), visible)


def _visible(mapping: dict[str, str], text: str) -> set[str]:
    """The tokens that actually occur in the payload being sent.

    ``mapping`` is already exactly the tokens ``_apply_index`` substituted into ``text``, so
    the two agree; ``text`` is checked anyway because it is the thing that crosses, and the
    invariant "everything the glossary names is re-hydratable" must not depend on a caller
    getting the pairing right.
    """
    if not text:
        return set(mapping)
    present = {m.group(0) for m in nf.TOKEN_IN_TEXT.finditer(text)}
    return {t for t in mapping if t in present}


def _person_is_corroborated(row, surfaces: list[str]) -> bool:
    """Is there anything behind a PERSON token beyond one NER span saying so?

    Two kinds of evidence count, and neither is a guess: the scope's graph typed this referent
    (``entity_type``/``roles``/``jurisdiction`` came from a seed entity or a party), or the
    name itself has the morphology of a personal name in a language the system reads.
    """
    attrs = row.attributes or {}
    if attrs.get("entity_type") or attrs.get("roles") or attrs.get("jurisdiction"):
        return True
    if attrs.get("source") in ("intake", "mailbox"):
        return True
    for s in surfaces:
        parts = [p for p in nf.name_parts(s) if not nf.is_initial(p)]
        if any(nf.is_patronymic(p) for p in parts):
            return True
        if len(parts) >= 2 and _gender_from_surfaces([s]):
            return True
    return False


# A percentage figure in the payload, near the token the glossary is about to band.
_PCT = re.compile(r"\d{1,3}(?:[.,]\d+)?\s*%")
_NEAR = 200


def _banding_is_defeated(text: str, token: str) -> bool:
    """True when the payload states a percentage right next to ``token``.

    The band exists to withhold the exact shareholding. It withholds nothing when the exact
    shareholding is four lines below it in the same envelope — measured on one real chunk,
    whose de-identified text reads "PERSON_1 70%", "PERSON_4 (20%)", "PERSON_7 (5%)",
    "PERSON_9 (5%)" while the glossary bands all four.
    """
    if not text:
        return False
    for m in nf.TOKEN_IN_TEXT.finditer(text):
        if m.group(0) != token:
            continue
        window = text[max(0, m.start() - _NEAR): m.end() + _NEAR]
        if _PCT.search(window):
            return True
    return False


def _drop_unmappable_lines(lines: list[str], visible: set[str]) -> list[str]:
    """Last gate: a line may not name a token the return leg cannot reverse.

    The header instructs the model to echo these tokens back. A token the glossary names but
    the request never sent is a token ``detokenize(mapping=...)`` will not touch — and on the
    ``mapping=None`` path it is worse, because the model is then handed a real name for a
    referent the cloud never received. Measured before this gate: one payload's glossary named
    14 tokens absent from its mapping, and a real model echoed up to 13 of them in one answer.
    """
    out = []
    for line in lines:
        named = {m.group(0) for m in nf.TOKEN_IN_TEXT.finditer(line)}
        if named <= visible:
            out.append(line)
    return out


def _drop_leaky_lines(index, lines: list[str]) -> list[str]:
    """Fail-closed: a glossary line that contains an enrolled real value is discarded."""
    keys = getattr(index, "keys", {})
    if not keys:
        return lines
    out = []
    for line in lines:
        hay, _ = nf.fold_haystack(line)
        if any(nf.find_all(hay, k) for k in keys):
            continue
        out.append(line)
    return out


def render(lines: list[str]) -> str:
    """The block that is prepended to the outbound prompt."""
    if not lines:
        return ""
    return (
        "TOKEN GLOSSARY (out-of-band; these describe the pseudonyms used below. They are "
        f"derived from {_WORDING.source}, contain no identifying detail, and must be echoed back "
        "as tokens — never expanded, guessed at, or replaced with real names):\n"
        + "\n".join(f"  {ln}" for ln in lines)
        + "\n\n"
    )
