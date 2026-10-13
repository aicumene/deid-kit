"""Ported from LegalTech apps/api/tests/test_cite_de.py (b3a0d49): the desktop finds norms as the server does.

German citation extraction — section-level, and strict about what counts as a citation.

The regression this file exists for is at the bottom: before the German patterns, a Klageschrift
citing entirely fabricated §§ passed ``gate_draft`` with zero objections, because no pattern
matched, ``extract_citations`` returned ``[]`` and the gate's "no citations → pass" short-circuit
certified it. The anti-hallucination guarantee the English and US work gets did not exist for
German work, and it failed silently.
"""
from __future__ import annotations

import unicodedata

from deidkit.agents.cite import extract_citations
from deidkit.agents.cite import extract_de_citations


def test_subsection_is_dropped_like_the_us_forms():
    assert extract_de_citations("§ 823 Abs. 1 Satz 2 BGB") == ["§ 823 BGB"]
    assert extract_de_citations("§ 253 Abs. 2 Nr. 2 ZPO") == ["§ 253 ZPO"]
    assert extract_de_citations("§823 BGB") == ["§ 823 BGB"]            # no space is idiomatic too


def test_roman_absatz_style_is_understood():
    assert extract_de_citations("§ 823 I 1 BGB") == ["§ 823 BGB"]       # Absatz 1, Satz 1
    assert extract_de_citations("§ 346 II 1 Nr. 3 1. Var. BGB") == ["§ 346 BGB"]
    # A Roman numeral before the code is an Absatz and goes; after the code stem it is a
    # Sozialgesetzbuch book and stays. "§ 7 I SGB IV" carries both.
    assert extract_de_citations("§ 7 I SGB IV") == ["§ 7 SGB IV"]


def test_the_whole_subdivision_ladder_is_dropped():
    for text, expected in (
        ("§ 4 Nr. 21 Buchst. a Doppelbuchst. bb UStG", "§ 4 UStG"),
        ("§ 309 Nr. 8 lit. b sublit. dd BGB", "§ 309 BGB"),
        ("§ 1 Abs. 1 Satz 1 Halbs. 1 VwVfG", "§ 1 VwVfG"),
        ("§ 14 S. 1 1. Hs. BImSchG", "§ 14 BImSchG"),
        ("§ 812 Abs. 1 S. 1 1. Var. BGB", "§ 812 BGB"),
        ("§ 48 Abs. 2 Alt. 2 GmbHG", "§ 48 GmbHG"),
        ("§ 992 Abs. 1 a.E. BGB", "§ 992 BGB"),
        ("§ 132 Abs. 2 Ziff. 1 VwGO", "§ 132 VwGO"),
    ):
        assert extract_de_citations(text) == [expected], text


def test_sloppy_but_idiomatic_spacing_still_reads():
    # Real filings: the abbreviation period goes missing, so does the space after it, and a PDF
    # extraction breaks the citation across a line.
    assert extract_de_citations("§ 823 Abs 1 BGB") == ["§ 823 BGB"]
    assert extract_de_citations("§ 823 Abs.1 S.1 BGB") == ["§ 823 BGB"]
    assert extract_de_citations("§ 823\nAbs. 1 BGB") == ["§ 823 BGB"]
    assert extract_de_citations("Art 6 Abs. 1 GG") == ["Art. 6 GG"]     # ministry style, no period
    assert extract_de_citations("Paragraph 823 BGB") == ["§ 823 BGB"]   # dictated text


def test_letter_suffixed_sections_survive():
    # § 3 RVG and § 3a RVG are different provisions; truncating the letter cites the wrong one.
    assert extract_de_citations("§ 3a Abs. 1 RVG und § 130a ZPO") == ["§ 3a RVG", "§ 130a ZPO"]
    assert extract_de_citations("§ 7f SGB IV") == ["§ 7f SGB IV"]
    # …but the letter must not eat the "f" of a range marker: "§§ 305ff." is § 305, not § 305f.
    assert extract_de_citations("§§ 305ff. BGB") == ["§ 305 BGB"]


def test_one_trailing_code_governs_the_whole_chain_until_the_next_one():
    # The structural inverse of the US form: in "8 U.S.C. § 1427" the code leads and each citation
    # stands alone; in German one trailing abbreviation stamps every section pending before it.
    assert extract_de_citations("§§ 280 Abs. 1, 311 Abs. 2 Nr. 2, 241 Abs. 2 BGB") == \
        ["§ 280 BGB", "§ 311 BGB", "§ 241 BGB"]
    assert extract_de_citations("§§ 823 Abs. 1, 831 BGB, § 7 StVG, § 115 VVG") == \
        ["§ 823 BGB", "§ 831 BGB", "§ 7 StVG", "§ 115 VVG"]
    # A comma followed by a subdivision word continues the section before it, so this is two
    # sections and not three.
    assert extract_de_citations("§§ 929 S. 1, 932 Abs. 1 S. 1, Abs. 2 BGB") == \
        ["§ 929 BGB", "§ 932 BGB"]


def test_plural_marker_announces_a_list_of_sections():
    assert extract_de_citations("§§ 133, 157 BGB") == ["§ 133 BGB", "§ 157 BGB"]
    assert extract_de_citations("§§ 823 Abs. 1, 1004 BGB analog") == ["§ 823 BGB", "§ 1004 BGB"]


def test_singular_marker_with_an_absatz_is_one_section():
    # "§ 823 Abs. 1, 2 BGB" is one section and two Absätze — not §§ 823 and 2.
    assert extract_de_citations("§ 823 Abs. 1, 2 BGB") == ["§ 823 BGB"]
    assert extract_de_citations("Art. 6 Abs. 1, 2 DSGVO") == ["Art. 6 DSGVO"]
    # …but with no subsection word the comma can only be separating sections.
    assert extract_de_citations("Art. 4, 6 DSGVO") == ["Art. 4 DSGVO", "Art. 6 DSGVO"]


def test_a_repeated_marker_is_always_its_own_section():
    assert extract_de_citations("§ 5 Abs. 2 und § 6 Abs. 1 BGB") == ["§ 5 BGB", "§ 6 BGB"]


def test_ranges_are_read_as_their_endpoints():
    # Never the interior: "§§ 78 bis 78c StGB" spans 78, 78a, 78b, 78c, which no arithmetic
    # produces, and expanding would put sections in the citation list the author never wrote.
    assert extract_de_citations("§§ 305 bis 310 BGB") == ["§ 305 BGB", "§ 310 BGB"]
    assert extract_de_citations("§§ 305-310 BGB") == ["§ 305 BGB", "§ 310 BGB"]
    assert extract_de_citations("§§ 305 – 310 BGB") == ["§ 305 BGB", "§ 310 BGB"]   # en dash
    assert extract_de_citations("§§ 78 bis 78c StGB") == ["§ 78 StGB", "§ 78c StGB"]
    assert extract_de_citations("§§ 195 ff. BGB") == ["§ 195 BGB"]      # open-ended → the anchor
    assert extract_de_citations("§§ 346 f. BGB") == ["§ 346 BGB"]
    assert extract_de_citations("§§ 249 ff., 253 BGB") == ["§ 249 BGB", "§ 253 BGB"]


def test_artikel_codes_use_the_artikel_form():
    assert extract_de_citations("Art. 2 Abs. 1 GG") == ["Art. 2 GG"]
    assert extract_de_citations("Artikel 14 GG") == ["Art. 14 GG"]
    assert extract_de_citations("Art. 6 Abs. 1 lit. f DSGVO") == ["Art. 6 DSGVO"]
    assert extract_de_citations("Art. 7 Nr. 1 Brüssel Ia-VO") == ["Art. 7 Brüssel Ia-VO"]


def test_sgb_is_refused_without_its_book_numeral():
    # SGB V, VI and VII are different codes. A citation stored without the numeral resolves to
    # whichever of them the gate's unanchored ilike reaches first, so it must never be emitted.
    assert extract_de_citations("§ 7 SGB") == []
    assert extract_de_citations("§ 7 Abs. 1 SGB IV") == ["§ 7 SGB IV"]
    assert extract_de_citations("§ 44 SGB XII") == ["§ 44 SGB XII"]     # not SGB X


def test_egbgb_keeps_both_markers():
    # The EGBGB's transitional provisions are Artikel and § at once; "§ 5 EGBGB" does not exist.
    assert extract_de_citations("Art. 229 § 5 EGBGB") == ["Art. 229 § 5 EGBGB"]
    assert extract_de_citations("Art. 229 § 5 Satz 2 EGBGB") == ["Art. 229 § 5 EGBGB"]
    assert extract_de_citations("Art. 3 EGBGB") == ["Art. 3 EGBGB"]
    assert extract_de_citations("Art. 14 I Nr. 2 EGBGB") == ["Art. 14 EGBGB"]
    # A lettered Artikel with §§ — Art. 246a ff. EGBGB, the consumer information duties. The
    # connector serves them as "Art. 246a § 1 EGBGB" (13.09.2026); an extractor that reads only a
    # bare number here returns nothing and the pin never fires.
    assert extract_de_citations("Art. 246a § 1 Abs. 2 EGBGB") == ["Art. 246a § 1 EGBGB"]


def test_the_fee_schedules_are_cited_by_number():
    # Annexes to the RVG and the GKG — numbered, not sectioned, so no § rule reaches them.
    assert extract_de_citations("Nr. 3100 VV RVG") == ["Nr. 3100 VV RVG"]
    assert extract_de_citations("Nr. 1210 KV GKG") == ["Nr. 1210 KV GKG"]


def test_one_law_gets_one_spelling():
    # citation is a unique column, so a second spelling of one law is a second key. The ministry's
    # suffix form and the practitioners' prefix form must land on the same string.
    assert extract_de_citations("§ 26 ZPOEG") == ["§ 26 EGZPO"]
    assert extract_de_citations("Art. 3 BGBEG") == ["Art. 3 EGBGB"]
    assert extract_de_citations("§ 15 HwO") == ["§ 15 HandwO"]
    # …and the EU instruments whose names carry a space or a hyphen interchangeably.
    for raw in ("Art. 7 Brüssel Ia-VO", "Art. 7 Brüssel-Ia-VO"):
        assert extract_de_citations(raw) == ["Art. 7 Brüssel Ia-VO"], raw
    assert extract_de_citations("Art. 4 Rom-I-VO") == ["Art. 4 Rom I-VO"]


def test_german_typography_does_not_hide_a_citation():
    # A no-break space, a zero-width character or a decomposed umlaut makes a correct citation
    # unequal to the seeded provision in Postgres — the gate would then object to good law.
    assert extract_de_citations("\u00a7\u00a0823 Abs.\u00a01 BGB") == ["\u00a7 823 BGB"]
    assert extract_de_citations("\u00a7 823\u200b BGB") == ["\u00a7 823 BGB"]
    decomposed = unicodedata.normalize("NFD", "Art. 7 Br\u00fcssel Ia-VO")
    assert extract_de_citations(decomposed) == ["Art. 7 Br\u00fcssel Ia-VO"]


def test_contract_clauses_are_not_statutory_citations():
    # German contracts, Satzungen and AGB number their own clauses with § — the largest source of
    # false objections, and the reason a known code abbreviation is required.
    for text in ("gemäß § 5 dieses Vertrages",
                 "§ 12 der Satzung bleibt von der vorstehenden Änderung unberührt.",
                 "Abweichend von § 3 Abs. 2 des Mietvertrages trägt der Vermieter die Kosten.",
                 "§ 2 der Anlage 3",
                 "Im Übrigen gelten die Regelungen in § 7 AGB des Auftragnehmers.",
                 "die Ansprüche nach dem BGB",
                 "Art. 5 des Abkommens",
                 "§ 1 Name, Sitz, Geschäftsjahr",                 # a clause heading, not a cite
                 "Die Höhe der Abfindung ergibt sich aus den Regelungen des § 14.",
                 "Die Höhe seiner Entschädigung richtet sich nach § 14, Abs. 1.",
                 "Zu § 8 (4) Beschlussfähigkeit Mitgliederversammlung",
                 "Inhaltsverzeichnis: § 1 Präambel .......... 2   § 2 Vertragsgegenstand ..... 3",
                 "Sollte eine Bestimmung dieses Vertrages unwirksam sein, gilt § 15 entsprechend.",
                 "Die Vertragsparteien vereinbaren, dass § 4 Abs. 3 ersatzlos gestrichen wird.",
                 "Wie in Paragraph 3 dieses Vertrages vereinbart, ist die Vergütung fällig.",
                 "Wir verweisen auf § 11 Abs. 1 unserer Einkaufsbedingungen (Stand 01/2026).",
                 "§ 5 im BGB",                                    # a word is not a subdivision
                 "Palandt/Heinrichs, § 278 Rn. 7",                # a commentary, not a statute
                 "zuletzt geändert durch Art. 5 des Gesetzes vom 29. August 2013"):
        assert extract_de_citations(text) == [], text


def test_an_internal_clause_does_not_inherit_the_code_beside_it():
    """The dangerous shape: a real statutory citation earlier in the sentence, a clause reference
    to the parties' own contract later. Only the statute may be extracted."""
    assert extract_de_citations(
        "Die Kündigung richtet sich nach § 5 dieses Vertrages; im Übrigen bleibt § 626 BGB "
        "unberührt.") == ["§ 626 BGB"]
    assert extract_de_citations(
        "Der Mieter trägt die Betriebskosten gemäß § 556 Abs. 1 BGB; die Verteilung auf die "
        "Mietparteien regelt § 4 dieses Vertrages.") == ["§ 556 BGB"]


def test_case_law_is_out_of_scope_by_decision():
    # Documented, not accidental: nothing in the KB could back a BGH docket or an NJW reference,
    # so extracting them would raise an objection no reviewer can clear.
    assert extract_de_citations("BGH, Urteil vom 12.03.2019 – VI ZR 123/19, NJW 2019, 1234") == []


def test_german_coexists_with_the_other_jurisdictions():
    cites = extract_citations("§ 823 BGB, СК РФ ст. 39, CPR 25.1, MCA 1973 s. 25, 8 U.S.C. § 1182")
    for c in ("§ 823 BGB", "СК РФ ст. 39", "CPR 25.1", "MCA 1973 s. 25", "8 U.S.C. § 1182"):
        assert c in cites


def test_order_preserving_dedupe_across_the_whole_extractor():
    assert extract_citations("§ 823 BGB; erneut § 823 Abs. 2 BGB") == ["§ 823 BGB"]


def test_a_bare_number_after_a_subsection_word_is_not_a_new_section():
    """The plural marker alone was taken as "everything after a comma is a section", so an Absatz
    or Nummer written bare produced a citation to a real but unrelated provision — the worst kind
    of false positive, because nothing downstream can tell it was never written."""
    assert extract_de_citations("§§ 280 Abs. 1, 3, 281 BGB") == ["§ 280 BGB", "§ 281 BGB"]
    assert extract_de_citations("§§ 223 Abs. 1, 224 Abs. 1 Nr. 2, 5 StGB") == \
        ["§ 223 StGB", "§ 224 StGB"]
    assert extract_de_citations("§ 823 Abs. 1, 2 sowie § 826 BGB") == ["§ 823 BGB", "§ 826 BGB"]
    assert extract_de_citations("§§ 305, 307 Abs. 1, 2 BGB") == ["§ 305 BGB", "§ 307 BGB"]


def test_a_part_with_its_own_subsection_word_is_a_section_however_the_numbers_run():
    """The ascending test resolves a BARE number only. "241 Abs. 2" carries its own Absatz, so it
    is a section even though 241 < 311."""
    assert extract_de_citations("§§ 280 Abs. 1, 311 Abs. 2 Nr. 2, 241 Abs. 2 BGB") == \
        ["§ 280 BGB", "§ 311 BGB", "§ 241 BGB"]


def test_a_german_compound_is_a_word_not_a_citation():
    """A sentence ends at "§ 14." and the next one opens with a compound. German forms compounds
    freely, so the code was being taken from the first word of the following sentence — and from
    the heading under a Satzung clause number."""
    assert extract_de_citations(
        "Die Höhe der Abfindung ergibt sich aus § 14.\nBGB-Vorschriften bleiben unberührt.") == []
    assert extract_de_citations("§ 12\nAGG-Beschwerdestelle") == []
    assert extract_de_citations("a) Es gilt § 7.\nb) StGB-Tatbestände bleiben unberührt.") == []


def test_two_bare_numbers_in_one_item_are_not_a_citation():
    """"§ 5   280 BGB" emitted § 5 — the left-hand number — and silently dropped the right-hand
    one. Fabricating a citation AND losing the real one is strictly worse than reading neither."""
    assert extract_de_citations("§ 5   280 BGB") == []


def test_a_long_chain_cannot_hang_the_extractor():
    """A denial of service, not a formatting nit: no token was maximal and whitespace between
    tokens was optional, so "305" could be read as "305", "30"+"5" or "3"+"0"+"5" — 2^(n-1)
    tokenisations per number, multiplied along the chain. Twelve sections whose code is spelled out
    rather than abbreviated (so the match never completes and every combination is tried) took 92
    seconds at 100% CPU. This text arrives from OCR of client documents and from PST mail bodies,
    so an adversary — or a bad scan — controls it.
    """
    import time

    chain = "§§ " + ", ".join(str(300 + i) for i in range(40)) + " des Buergerlichen Gesetzbuchs."
    started = time.perf_counter()
    assert extract_de_citations(chain) == []
    assert time.perf_counter() - started < 1.0, "extraction is super-linear again"


def test_german_codes_named_in_russian_resolve_to_the_same_norms():
    # Переводы немецких решений и договоров: «§ 126 ГГУ» — это § 126 BGB (13.09.2026, документ
    # Arrow: из шести упоминаний § разметка находила одно). Кодекс тот же, написание другое.
    assert extract_de_citations("На основании § 126 ГГУ под письменной формой") == ["§ 126 BGB"]
    assert extract_de_citations("Согласно  §  305  Германского  Гражданского  Уложения") == ["§ 305 BGB"]
    assert extract_de_citations("п. 1 и 2 § 288, § 247 Германского гражданского уложения") == ["§ 288 BGB", "§ 247 BGB"]
    assert extract_de_citations("§ 377 ГТУ") == ["§ 377 HGB"]
    assert extract_de_citations("§ 253 ГПК ФРГ") == ["§ 253 ZPO"]
    # «ст. 5 ГТУ» — не § и не наш маркер; «§ 5 ГГУ дальше» без кодекса сразу за номером — не цитата.
    assert extract_de_citations("ст. 5 ГТУ") == []


def test_de_pinpoints_read_absatz_and_nummer_after_the_section():
    from deidkit.agents.cite import de_pinpoints
    assert de_pinpoints("Übungsleiter nach § 3 Nr. 26 EStG") == {"3": (None, "26")}
    assert de_pinpoints("§ 6 Abs. 1 Nr. 3a EStG") == {"6": ("1", "3a")}
    assert de_pinpoints("§ 6 Absatz 1 Satz 2 Nummer 4 EStG") == {"6": ("1", "4")}
    assert de_pinpoints("§ 3 (1) Nr. 2 UStG") == {"3": ("1", "2")}
    assert de_pinpoints("Art. 250 § 6 Abs. 2 Nr. 7 EGBGB") == {"250": (None, None)} or \
        de_pinpoints("Art. 250 § 6 Abs. 2 Nr. 7 EGBGB").get("6") == ("2", "7")
    assert de_pinpoints("Was ist nach § 3 EStG steuerfrei?") == {}
    assert de_pinpoints("§§ 280 Abs. 1, 3, 281 BGB") == {"280": ("1", None)}
