"""References to norms in a document's text: where each is, and its canonical form ("§ 823 BGB",
"Art. 6 DSGVO", "8 U.S.C. § 1427", "Cap. 113 s. 105", "ГК РФ ст. 450").

The extraction is Bernio's server's, copied without its database: ``apps/api/app/knowledge/cite.py`` and
``cite_de.py`` of LegalTech (b3a0d49, 13.10.2026) — the same patterns, so that a mention the desktop marks
is the one the server's knowledge base would resolve. Change it there first, then here. The server's
reasons for being strict (a contract's own "§ 5" is not a statute) are in its docstrings, kept below.

Only finding is here: whether the knowledge base holds the norm is the server's to say, and until it
can be asked a mention is shown as found in the text, not checked.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ── German statutes (cite_de.py) ────────────────────────────────────────────────────────────

#: German typography puts a no-break space after "§" and a narrow one inside abbreviations, and
#: copy-paste from a PDF adds zero-width joiners. To Postgres those are ordinary characters: a
#: citation carrying one is neither equal to nor ILIKE the seeded provision, so a *correct* cite
#: comes back unresolved and the gate objects to good law. Fold them before matching — NFKC also
#: composes a decomposed "ü" (so "Brüssel Ia-VO" matches however it was typed) and turns the
#: Unicode Roman numerals into ASCII — then strip the zero-widths and the soft hyphen, which NFKC leaves alone.
_INVISIBLE = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff\u00ad")

# ── Code tables ──────────────────────────────────────────────────────────────
# Two tables, because the marker is part of the citation: German codes are cited with "§", the
# Grundgesetz, the EGBGB and EU regulations with "Artikel". Keeping them disjoint is what stops
# "Art. 229 § 5 EGBGB" (a transitional provision) from being read as "§ 5 EGBGB", which does not
# exist.

#: Cited with § — the abbreviation as practitioners write it, read off the A-Z index at
#: gesetze-im-internet.de. Wide on purpose: an abbreviation missing from this table is a citation
#: the gate never sees, while a spare one costs nothing, because a § and a section number are still
#: required in front of it.
PARAGRAPH_CODES: tuple[str, ...] = (
    # civil, procedural, family, land, tenancy
    "BGB", "ZPO", "EGZPO", "FamFG", "GVG", "EGGVG", "ZVG", "EGZVG", "GBO", "GBV", "RPflG",
    "BeurkG", "AnfG", "IntFamRVG", "LPartG", "VersAusglG", "UhVorschG", "AUG", "SachenRBerG",
    "SchuldRAnpG", "ErbbauRG", "WEG", "HeizkostenV", "BetrKV", "WoBindG", "WoFG",
    # costs and the legal professions
    "RVG", "GKG", "GNotKG", "JVEG", "JVKostG", "BRAO", "BORA", "BNotO", "StBerG", "WPO", "PAO",
    "RDG",
    # criminal and regulatory
    "StGB", "EGStGB", "StPO", "EGStPO", "OWiG", "JGG", "StVollzG", "StrEG", "BtMG", "KCanG",
    "WaffG", "SchwarzArbG",
    # commercial, corporate, insolvency, capital markets
    "HGB", "EGHGB", "AktG", "GmbHG", "EGGmbHG", "GenG", "PartGG", "UmwG", "InsO", "EGInsO",
    "StaRUG", "PublG", "DrittelbG", "MitbestG", "BörsG", "DepotG", "ScheckG", "GüKG", "BinSchG",
    # competition and intellectual property
    "UWG", "GWB", "UrhG", "VGG", "MarkenG", "PatG", "GebrMG", "DesignG",
    # employment
    "BetrVG", "SprAuG", "BPersVG", "KSchG", "TVG", "ArbGG", "TzBfG", "ArbZG", "ArbSchG", "ASiG",
    "BUrlG", "MuSchG", "BEEG", "PflegeZG", "FPfZG", "AGG", "EntgTranspG", "EntgFG", "NachwG",
    "AÜG", "AEntG", "MiLoG", "BBiG", "BBG", "BeamtStG", "SGG",
    # public law, information, environment, planning
    "VwGO", "VwVfG", "VwZG", "VwVG", "BVerfGG", "IFG", "UIG", "VIG", "ROG", "BauGB", "BauNVO",
    "BImSchG", "KrWG", "WHG", "BNatSchG", "UVPG", "UmweltHG", "GenTG", "EnWG",
    # tax
    "AO", "EGAO", "FGO", "EStG", "KStG", "GewStG", "UStG", "UStDV", "ErbStG", "GrEStG", "GrStG",
    "BewG", "SolZG", "AStG", "InvStG", "UmwStG",
    # transport
    "StVG", "StVO", "StVZO", "FeV", "FZV", "PflVG", "PBefG", "LuftVG", "HaftPflG",
    # banking, insurance, money laundering
    "VVG", "VAG", "KWG", "WpHG", "WpÜG", "ZAG", "KAGB", "GwG",
    # data, media, telecoms
    "BDSG", "TDDDG", "TTDSG", "DDG", "TMG", "TKG",
    # trade, health, food, product safety
    "GewO", "HandwO", "GastG", "ApoG", "BÄO", "PsychThG", "AMG", "MPDG", "IfSG", "TFG", "TPG",
    "LFGB", "ProdHaftG", "ProdSG", "TierSchG", "JuSchG", "KHG", "KHEntgG",
    # migration, civil status, identity
    "AufenthG", "AsylG", "StAG", "AZRG", "BeschV", "PStG", "PassG", "PAuswG", "AdVermiG",
    "AdWirkG", "FreizügG/EU",
    # the ministry's own spellings of the Einführungsgesetze — normalised away on output
    "ZPOEG", "HGBEG", "GVGEG", "ZVGEG", "StPOEG", "StGBEG", "InsOEG", "AOEG", "HwO",
)

#: Cited with Artikel — the constitution, the EGBGB, EU and Council-of-Europe law.
ARTICLE_CODES: tuple[str, ...] = (
    "GG", "EGBGB", "BGBEG",
    "AEUV", "EUV", "GRCh", "EMRK", "CISG",
    "DSGVO", "DS-GVO",
    "EuGVVO", "EuGüVO", "EuPartVO", "EuUntVO", "EuErbVO", "EuInsVO", "EuVTVO", "EuMahnverfVO",
    "EuZVO", "EuBVO", "HUP", "HKÜ",
)

#: The Einführungsgesetze have two spellings — practitioners write the EG prefix, the ministry
#: publishes the suffix — and the Handwerksordnung has two abbreviations. Both spellings are
#: accepted and both emit one form, because a second spelling of one law is a second key in a
#: knowledge base whose citation column is unique.
_CODE_ALIASES: dict[str, str] = {
    "ZPOEG": "EGZPO", "BGBEG": "EGBGB", "HGBEG": "EGHGB", "GVGEG": "EGGVG", "ZVGEG": "EGZVG",
    "StPOEG": "EGStPO", "StGBEG": "EGStGB", "InsOEG": "EGInsO", "AOEG": "EGAO", "HwO": "HandwO",
}

#: The Sozialgesetzbuch carries a Roman book number that is part of the citation ("§ 33 SGB V"), so
#: it needs a fragment rather than a literal. Longest first — "SGB XII" must win over "SGB X". A
#: bare "SGB" is deliberately *not* in the table above: SGB V, VI and VII are different codes, and
#: a citation stored without its book numeral resolves to whichever of them the resolver's
#: unanchored ilike reaches first.
_SGB = r"SGB\s+(?:XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)"
#: One instrument goes by four names, two of which carry a space and a hyphen that drafters swap at
#: will ("Brüssel Ia-VO", "Brüssel-Ia-VO"). "Brüssel" and "Rom" are ordinary words, so only the full
#: pattern is safe to match; the separator is normalised to a space on output.
_EU_NAMED = r"(?:Brüssel|Rom)[\s-]I{1,3}[ab]?-VO"
#: Немецкие кодексы, названные по-русски: юристы фирмы ведут немецкие дела на русском, и в
#: переводе решения суда стоит «§ 126 ГГУ» / «§ 305 Германского Гражданского Уложения» (13.09.2026:
#: разметка документа Arrow нашла 1 цитату из 6 упоминаний §). Норма та же — BGB/HGB/ZPO.
_RU_NAMED = (r"(?:ГГУ|ГТУ|ГПК\s+(?:ФРГ|Германии)|"
             r"[Гг]ерманск\w{1,4}\s+[Гг]ражданск\w{1,4}\s+[Уу]ложени\w{1,2}|"
             r"[Гг]ерманск\w{1,4}\s+[Тт]оргов\w{1,4}\s+[Уу]ложени\w{1,2})")


def _alternation(codes: tuple[str, ...], *fragments: str) -> str:
    """Fragments first, then literals longest-first, so no abbreviation is eaten by its own prefix."""
    return "|".join([*fragments, *(re.escape(c) for c in sorted(codes, key=len, reverse=True))])


def _normalize_code(raw: str) -> str:
    """One law, one spelling — collapse the typographic variants and fold the ministry's aliases."""
    code = re.sub(r"^(Brüssel|Rom)[\s-]", r"\1 ", " ".join(raw.split()))
    if code == "ГГУ" or "ражданск" in code:
        return "BGB"
    if code == "ГТУ" or "оргов" in code:
        return "HGB"
    if code.startswith("ГПК"):
        return "ZPO"
    return _CODE_ALIASES.get(code, code)


# ── Cluster grammar ──────────────────────────────────────────────────────────
# Between the section number and the code only these may appear. Anything else — a noun, an
# article, "dieses Vertrages" — ends the cluster, and no citation is recognised. Note what is
# absent: a free-standing lowercase letter. It would make "§ 5 im BGB" a citation, so a letter
# enumerator is only read where its own marker introduces it ("lit. f", "Buchst. a").
_SUB_WORDS: tuple[str, ...] = (
    "Absatz", "Abs", "Unterabsatz", "Unterabs", "UAbs", "Satz", "S", "Halbsatz", "Halbs", "Hs",
    "Teilsatz", "Teils", "Nummer", "Nr", "Ziffer", "Ziff", "Spiegelstrich", "Alt", "Var", "Fall",
    "ff", "f",
)
_LETTERED = r"(?:Doppelbuchstabe|Doppelbuchst|Buchstabe|Buchst|sublit|lit)\.?\s*[a-z]{1,2}(?![a-z])\)?"
#: A section number may carry ONE letter — but "305ff." is § 305 with a range marker, not
#: § 305f, so the letter is only taken when no further letter follows it.
_SECTION = r"\d+(?!\d)(?:[a-z](?![a-z]))?"
_TOKEN = (rf"(?:{_LETTERED}|a\.E\.|\(\d+[a-z]?\)|{_SECTION}\.?|[IVX]{{1,4}}(?![IVX])|"
          rf"(?:{'|'.join(sorted(_SUB_WORDS, key=len, reverse=True))})\.?|[a-z]\))")
#: Whitespace between tokens is optional: "§ 823 Abs.1 S.1 BGB" and "§§ 305ff. BGB" are both
#: idiomatic, and a PDF-extracted Schriftsatz breaks citations across lines.
_ITEM = rf"{_TOKEN}(?:\s*{_TOKEN})*"
#: Separators inside a citation list, including the two range forms ("bis", a dash).
_SEP = r"(?:\s*[,;]\s*|\s+(?:und|sowie|bis)\s+|\s*[-–—]\s*)"
_LIST = rf"{_ITEM}(?:{_SEP}(?:§§?\s*)?{_ITEM}){{0,60}}"
#: "§" is spelled out in dictated and accessibility-converted text; "Art" loses its period in the
#: ministry's own house style.
_PARA_MARK = r"§§?|Paragrap[fh]en|Paragrap[fh]|Par\."
_ART_MARK = r"Artt?\.?|Artikel"

#: ``(?!-\w)`` keeps a compound out: BGB-Vorschriften, AGG-Beschwerdestelle, StGB-Tatbestände.
_PARA = re.compile(
    rf"({_PARA_MARK})\s*({_LIST})\s*\b({_alternation(PARAGRAPH_CODES, _SGB, _RU_NAMED)})\b(?!-\w)")
_ART = re.compile(
    rf"({_ART_MARK})\s*({_LIST})\s*\b({_alternation(ARTICLE_CODES, _EU_NAMED)})\b(?!-\w)")
#: "Art. 229 § 5 EGBGB" — in the Einführungsgesetze an Artikel is subdivided into §§, so the § is
#: part of the norm's address and gesetze-im-internet serves it under both. Matched separately so
#: the § is not read as a free-standing section of a code that has none. The Artikel may carry a
#: letter: Art. 246a–246e and 247a EGBGB (information duties in consumer contracts) are among
#: the most-cited provisions of the act, and until 13.09.2026 "Art. 246a § 1 EGBGB" matched
#: nothing here — the connector loading bgbeg found 14 such provisions with no extractor form.
_EGBGB = re.compile(
    rf"(?:{_ART_MARK})\s*(\d+[a-z]?)\s*§\s*({_SECTION})(?:\s*{_TOKEN})*\s*\b(EGBGB|BGBEG)\b")
#: The fee schedules are numbered, not sectioned: the RVG's Vergütungsverzeichnis is cited
#: "Nr. 3100 VV RVG" and the GKG's Kostenverzeichnis "Nr. 1210 KV GKG". Both are annexes, so
#: neither is reachable through a § rule — and "Nr." inside a § citation is a subdivision, which
#: is why this needs the instrument name to fire at all.
_SCHEDULE = re.compile(r"\bNr\.?\s*(\d{4})\s*(VV\s*RVG|KV\s*GKG)\b")

_HAS_SUB = re.compile(r"\b(?:Absatz|Abs|Unterabsatz|Unterabs|UAbs|Satz|Halbsatz|Halbs|Hs|"
                      r"Teilsatz|Teils|Nummer|Nr|Ziffer|Ziff|Spiegelstrich|Alt|Var|Fall|"
                      r"Buchstabe|Buchst|sublit|lit)\b")
#: Pinpoint of a § citation: the Absatz and Nummer named right after the section number —
#: "§ 3 Abs. 1 Nr. 26 EStG", "§ 3 Nr. 26 EStG", "§ 3 (1) Nr. 26", "§ 6 Absatz 1 Nummer 4 Satz 2".
#: Only the first Absatz/Nummer after the section is read: a chain "§§ 280 Abs. 1, 3, 281 BGB"
#: gets the pinpoint of its first section and nothing for the rest — a coarser hint, never a
#: wrong one (the second section's number is not mistaken for a Nummer of the first).
_PINPOINT = re.compile(
    rf"(?:{_PARA_MARK}|{_ART_MARK})\s*(?P<sec>{_SECTION})"
    rf"(?:\s*(?:Abs(?:atz|\.)?\s*(?P<abs>\d+[a-z]?)|\((?P<abs2>\d+[a-z]?)\)))?"
    rf"(?:\s*(?:Satz|S\.)\s*\d+)?"
    rf"(?:\s*(?:Nummer|Nr\.?)\s*(?P<nr>\d+[a-z]?))?")


def de_pinpoints(text: str) -> dict[str, tuple[str | None, str | None]]:
    """Section number → (Absatz, Nummer) named in ``text`` for German citations.

    Feeds retrieval's pinning: a question that names "§ 3 Nr. 26 EStG" wants the PART of § 3
    whose anchor is «§3 Nr. 26», not the section's first k chunks in file order (15.09.2026: the
    list-item anchors made that part addressable; before, all 31 parts of § 3 EStG were «§3»).
    Keyed by section number only — the code is resolved by :func:`extract_de_citations`; a query
    naming § 3 of two codes with different pinpoints keeps the first (coarser, not wrong).
    """
    out: dict[str, tuple[str | None, str | None]] = {}
    for m in _PINPOINT.finditer(text):
        abs_, nr = m.group("abs") or m.group("abs2"), m.group("nr")
        if not (abs_ or nr):
            continue
        out.setdefault(m.group("sec"), (abs_, nr))
    return out


_MARKER = re.compile(rf"^\s*(?:{_PARA_MARK}|{_ART_MARK})\s*")
_NUMBER = re.compile(rf"^\s*({_SECTION})")


def _sections(marker: str, body: str) -> list[str]:
    """The section numbers a cluster cites. See "The comma rule" in the module docstring.

    The plural marker alone is not enough. "§§ 280 Abs. 1, 3, 281 BGB" cites two sections and three
    Absätze: the bare "3" belongs to Abs., the bare "281" is a section, and nothing in the syntax
    separates them. German chains are written in ascending order, so a bare number that goes
    *backwards* after a subsection word has been seen is a subsection continuing that list — which
    reads every reported case correctly and errs toward omission, not invention, when it is wrong:

        §§ 280 Abs. 1, 3, 281 BGB        3 < 280 → Absatz;  281 > 280 → section
        §§ 223 Abs. 1, 224 Abs. 1 Nr. 2, 5 StGB     5 < 224 → Nummer
        § 823 Abs. 1, 2 sowie § 826 BGB  2 < 823 → Absatz;  "§ 826" repeats the marker → section
        §§ 133, 157 BGB                  no subsection word yet → both sections
    """
    parts = re.split(_SEP, body)
    plural = marker.startswith(("§§", "Artt"))
    repeats_marker = any(_MARKER.match(p) for p in parts[1:])
    if not plural and not repeats_marker and _HAS_SUB.search(body):
        parts = parts[:1]           # singular marker + a subsection word → one section, many Absätze
    out: list[str] = []
    seen_sub = False
    for i, part in enumerate(parts):
        stripped = _MARKER.sub("", part)
        m = _NUMBER.match(stripped)
        if not m:                   # a part that is only "ff." carries no section of its own
            seen_sub = seen_sub or bool(_HAS_SUB.search(part))
            continue
        repeats_own_marker = bool(_MARKER.match(part))
        # A part carrying its OWN subsection word is a section beyond doubt — "241 Abs. 2" in
        # "§§ 280 Abs. 1, 311 Abs. 2 Nr. 2, 241 Abs. 2 BGB" — so the ascending test below, which
        # exists only to resolve a BARE number, must not touch it.
        carries_own_sub = bool(_HAS_SUB.search(stripped[m.end():]))
        if i and seen_sub and not repeats_own_marker and not carries_own_sub and out:
            try:
                descends = int(re.sub(r"\D", "", m.group(1))) <= int(re.sub(r"\D", "", out[-1]))
            except ValueError:      # pragma: no cover — _NUMBER guarantees digits
                descends = False
            if descends:            # a subsection continuing the previous list, not a new section
                seen_sub = True
                continue
        # "§ 5   280 BGB" — two bare numbers with nothing between them is not a citation shape.
        rest = stripped[m.end():].lstrip()
        if re.match(r"\d", rest):
            continue
        out.append(m.group(1))
        seen_sub = seen_sub or bool(_HAS_SUB.search(part))
    return out


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """NFKC + удаление невидимых — как раньше, но с картой «индекс в нормализованном →
    индекс в исходном». Нужна разметке документа (13.09.2026): координаты цитаты должны
    указывать в ТОТ текст, что лежит в чанках, а NFKC меняет длину (лигатуры, «…»)."""
    out: list[str] = []
    back: list[int] = []
    # Нормализуем не по символу, а по кластеру «база + комбинирующие знаки»: разложенный
    # умлаут (u + ¨) должен СЛОЖИТЬСЯ в ü, а посимвольный NFKC этого не сделает.
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        piece = unicodedata.normalize("NFKC", text[i:j]).translate(_INVISIBLE)
        out.append(piece)
        back.extend([i] * len(piece))
        i = j
    return "".join(out), back


def extract_de_citation_spans(text: str) -> list[tuple[int, int, str]]:
    """Как :func:`extract_de_citations`, но с координатами ``(start, end)`` в ИСХОДНОМ тексте.
    Одно вхождение «§§ 133, 157 BGB» даёт две записи с одним и тем же span."""
    norm, back = _normalized_with_map(text)
    found: list[tuple[int, int, str]] = []

    def span(m: re.Match) -> tuple[int, int]:
        a, b = m.start(), m.end()
        if not back or a >= len(back):
            return a, b
        return back[a], back[min(b, len(back)) - 1] + 1

    for m in _EGBGB.finditer(norm):
        found.append((*span(m), f"Art. {m.group(1)} § {m.group(2)} EGBGB"))
    for m in _SCHEDULE.finditer(norm):
        found.append((*span(m), f"Nr. {m.group(1)} {' '.join(m.group(2).split())}"))
    for pattern, form in ((_PARA, "§ {} {}"), (_ART, "Art. {} {}")):
        for m in pattern.finditer(norm):
            code = _normalize_code(m.group(3))
            for section in _sections(m.group(1), m.group(2)):
                found.append((*span(m), form.format(section, code)))
    return sorted(found, key=lambda t: t[0])


def extract_de_citations(text: str) -> list[str]:
    """German citations in ``text``, normalized to "§ 823 BGB" / "Art. 2 GG" / "Nr. 3100 VV RVG",
    in order of appearance. Duplicates are kept — :func:`app.knowledge.cite.extract_citations`
    de-dupes across all jurisdictions."""
    return [c for _, _, c in extract_de_citation_spans(text)]


# ── the other jurisdictions (cite.py) ───────────────────────────────────────────────────────
# RU — "ст. 39 СК РФ" / "ч. 7 ст. 38 СК РФ" (article-first) and "СК РФ ст. 39" (code-first, the KB form).
_RU_ART_FIRST = re.compile(r"ст\.?\s*(\d+)\s+(СК РФ|ГК РФ)")
_RU_CODE_FIRST = re.compile(r"(СК РФ|ГК РФ)\s+ст\.?\s*(\d+)")
_CPR = re.compile(r"\bCPR\s+\d+(?:\.\d+)?\b")

# England & Wales — the Matrimonial Causes Act 1973 (the core financial-remedy statute). Handles
# "s.25 MCA 1973", "section 25 of the Matrimonial Causes Act 1973", "MCA 1973, s 25"; a subsection
# ("s.25(2)") normalizes to its section. → KB canonical form "MCA 1973 s. N".
_MCA = r"(?:MCA\s*1973|Matrimonial Causes Act\s*1973)"
_MCA_SEC_FIRST = re.compile(rf"(?:section|s)\.?\s*(\d+)(?:\(\d+[A-Za-z]?\))?\s*(?:of\s+the\s+)?{_MCA}", re.I)
_MCA_ACT_FIRST = re.compile(rf"{_MCA}[,\s]+(?:section|s)\.?\s*(\d+)", re.I)

# US — "8 U.S.C. § 1427(a)" / "8 USC 1427" / "8 U.S.C. §1427" (any title). The subsection in parens is
# dropped (section-level, like the MCA pattern) → "N U.S.C. § M", which matches the KB authority's
# title form ("8 U.S.C. § 1427 — …") via the resolver's ilike.
_USC = re.compile(r"\b(\d+)\s*U\.?\s*S\.?\s*C\.?\s*(?:§+\s*|sec(?:tion)?s?\.?\s*)?(\d+)")
# Cyprus — the Companies Law, Cap. 113. "Cap. 113 s. 105", "Cap 113, s.105",
# "section 105 of Cap. 113". Sections inserted by amendment carry an alphabetic suffix, and it is
# a GREEK letter: the series runs Α, Β, Γ, Δ, Ε, Ζ, Η, Θ, Ι, Κ, Λ, Μ, Ν — Greek ordinals, exactly
# as Greek legislative drafting numbers inserted provisions. Six of the 523 sections carry a LATIN
# "A" instead; CyLaw is inconsistent there and the corpus preserves what the source published.
#
# The suffix therefore accepts BOTH alphabets, because Α and A are one keystroke apart and
# indistinguishable on screen. Folding for comparison lives in :func:`fold_citation`; what is
# emitted here stays as written, so a citation never silently changes its spelling.
_CY_CAP = r"Cap\.?\s*113"
_CY_SUFFIX = r"[0-9]+[A-Za-zΑ-Ωα-ω]?"
_CY_CAP_FIRST = re.compile(rf"{_CY_CAP}[,\s]+(?:section|sec\.?|s)\.?\s*({_CY_SUFFIX})", re.I)
_CY_SEC_FIRST = re.compile(rf"(?:section|sec\.?|s)\.?\s*({_CY_SUFFIX})\s+of\s+(?:the\s+)?{_CY_CAP}", re.I)

#: Греческие заглавные, неотличимые от латинских. Только они: Γ, Δ, Θ, Λ, Ξ, Π, Σ, Φ, Ψ, Ω
#: двойников не имеют, и «сворачивать» их не к чему.
_HOMOGLYPHS = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def fold_citation(citation: str) -> str:
    """Ключ для СРАВНЕНИЯ, не для показа.

    Кипрская «Cap. 113 s. 4Α» (греческая альфа) и «Cap. 113 s. 4A» (латинская) выглядят
    одинаково и набираются вслепую одна вместо другой. Хранится то, что опубликовал источник;
    сравнивается — свёрнутая форма, иначе шесть статей из 523 находились бы только при точном
    попадании в алфавит, а промах читался бы как «нормы нет».
    """
    return (citation or "").translate(_HOMOGLYPHS)


#: Обратное отображение: латинская → греческая. Нужно, потому что искать приходится в ОБЕ
#: стороны — пользователь может набрать латиницей то, что хранится греческим, и наоборот.
_UNFOLD = {v: k for k, v in {chr(k): v for k, v in _HOMOGLYPHS.items()}.items()}


def citation_variants(citation: str) -> list[str]:
    """Написание как есть плюс его двойник в другом алфавите, без повторов.

    Для подавляющего большинства цитат вернёт одну строку — греко-латинские двойники встречаются
    только в кипрских суффиксах. Существует ради запроса: сравнение по ``==``/``ilike`` омоглиф
    не переживает, и статья «не находится» при совершенно верной ссылке.
    """
    c = citation or ""
    out = [c]
    folded = c.translate(_HOMOGLYPHS)
    if folded != c:
        out.append(folded)
    unfolded = "".join(_UNFOLD.get(ch, ch) for ch in c)
    if unfolded != c and unfolded not in out:
        out.append(unfolded)
    return out


# US CFR — "8 CFR § 214.2(o)" / "8 C.F.R. 245.1" → "N CFR § P.S" (subsection dropped).
_CFR_US = re.compile(r"\b(\d+)\s*C\.?\s*F\.?\s*R\.?\s*(?:§+\s*|part\s*|sec(?:tion)?s?\.?\s*)?(\d+\.\d+)", re.I)


@dataclass(slots=True, frozen=True)
class CitationSpan:
    """Одно вхождение цитаты в тексте: координаты в ИСХОДНОМ тексте и каноническая форма."""

    start: int
    end: int
    citation: str


def extract_citation_spans(text: str) -> list[CitationSpan]:
    """Все вхождения цитат с координатами, в порядке появления; дубликаты сохранены (одна и та же
    норма, процитированная дважды, — два вхождения). Нужна разметке документа (13.09.2026):
    ссылка на норму привязывается к чанку и смещению, а не к документу целиком."""
    found: list[CitationSpan] = []
    for m in _RU_ART_FIRST.finditer(text):      # "ст. N CODE" → groups (N, CODE)
        found.append(CitationSpan(m.start(), m.end(), f"{m.group(2)} ст. {m.group(1)}"))
    for m in _RU_CODE_FIRST.finditer(text):     # "CODE ст. N" → groups (CODE, N)
        found.append(CitationSpan(m.start(), m.end(), f"{m.group(1)} ст. {m.group(2)}"))
    for m in _CPR.finditer(text):
        found.append(CitationSpan(m.start(), m.end(), m.group(0)))
    for m in _MCA_SEC_FIRST.finditer(text):     # "s. N ... MCA 1973"
        found.append(CitationSpan(m.start(), m.end(), f"MCA 1973 s. {m.group(1)}"))
    for m in _MCA_ACT_FIRST.finditer(text):     # "MCA 1973 ... s. N"
        found.append(CitationSpan(m.start(), m.end(), f"MCA 1973 s. {m.group(1)}"))
    for m in _USC.finditer(text):               # "N U.S.C. § M" (any US Code title)
        found.append(CitationSpan(m.start(), m.end(), f"{m.group(1)} U.S.C. § {m.group(2)}"))
    for m in _CFR_US.finditer(text):            # "N CFR § P.S"
        found.append(CitationSpan(m.start(), m.end(), f"{m.group(1)} CFR § {m.group(2)}"))
    for m in _CY_CAP_FIRST.finditer(text):      # "Cap. 113 s. N"
        found.append(CitationSpan(m.start(), m.end(), f"Cap. 113 s. {m.group(1)}"))
    for m in _CY_SEC_FIRST.finditer(text):      # "section N of Cap. 113"
        found.append(CitationSpan(m.start(), m.end(), f"Cap. 113 s. {m.group(1)}"))
    for a, b, c in extract_de_citation_spans(text):   # "§ N CODE" / "Art. N CODE"
        found.append(CitationSpan(a, b, c))
    return found


def extract_citations(text: str) -> list[str]:
    """Pull citations out of free text, normalized to the KB's canonical form
    ("СК РФ ст. N", "ГК РФ ст. N", "CPR N.N", "MCA 1973 s. N", "§ N BGB", "Art. N GG").
    Order-preserving + de-duplicated. Порядок — тот же, что раньше: по семействам, немецкие
    последними (не по позиции в тексте) — на этот порядок опираются тесты и пиннинг."""
    seen: set[str] = set()
    out: list[str] = []
    for sp in extract_citation_spans(text):
        if sp.citation not in seen:
            seen.add(sp.citation)
            out.append(sp.citation)
    return out


def jurisdiction_of(citation: str) -> str:
    """The jurisdiction a canonical citation belongs to, by its form (the server's ``ingest/citations.py``)."""
    c = citation
    if " U.S.C. " in c or " CFR " in c:
        return "US"
    if c.startswith(("CPR ", "MCA 1973")):
        return "GB"
    if c.startswith("Cap. 113"):
        return "CY"
    if c.startswith(("СК РФ", "ГК РФ")):
        return "RU"
    if c.startswith(("§ ", "§§ ", "Art. ", "Nr. ")):
        return "DE"
    return ""
