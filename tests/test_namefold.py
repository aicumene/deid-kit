# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The vault's folding, boundary and admission predicates.

These are the three decisions behind every defect measured in a real vault, and each is tested
against a string of the same shape as the one that produced it (the names are invented):

  * the fold — "José Muñoz" was missed while "Jose Munoz" was caught; "Sea ﬁnder" (U+FB01)
    was missed for an enrolled vessel;
  * the boundary — "Certificate of incorporation" became "Certificate of inPERSON_28";
  * the admission predicate — a country, a building, an infinitive and nine declensions of two
    role nouns were each given a PERSON token.

Cyrillic appears here only as code points or as spellings the module itself generates from a
Latin name: what is tested is the mechanism, not a language.

Pure: no store, no detector.
"""

from __future__ import annotations

import pytest

from deidkit import namefold as nf
from deidkit.textmatch import normalize

#: The Russian lower-case alphabet, as code points (U+0430..U+044F and U+0451).
CYRILLIC_LOWER = "".join(chr(c) for c in range(0x0430, 0x0450)) + chr(0x0451)

# ── the fold IS the comparison form ──────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "Certificate of incorporation",
    "Sea ﬁnder sailed",                      # U+FB01 ligature
    "José Muñoz",                            # combining-capable diacritics
    "Ἀθῆναι Ελένη",                          # Greek with breathings and accents
    "a  b\n\tc   ",                          # whitespace collapse + strip
    "physi-\ncally",                         # line-break hyphen
    "«Beispiel» — GmbH ­ Muster",       # guillemets, em dash, soft hyphen
    "ＳＥＡ ＦＩＮＤＥＲ",                        # fullwidth (NFKC)
    "",
])
def test_fold_reproduces_textmatch_normalize(raw):
    """One fold, not two. If this drifts, the name matcher and a quote check built on
    ``normalize`` no longer agree on what the same text is."""
    f = nf.fold(raw)
    assert f.text == normalize(raw)
    assert len(f.starts) == len(f.ends) == len(f.text)


def test_fold_offsets_map_back_to_the_original_bytes():
    raw = "the vessel  Sea ﬁnder, flagged MT"
    hay, folded = nf.fold_haystack(raw)
    (start, end), = nf.find_all(hay, nf.key("SEA FINDER"))
    rs, re_ = folded.raw_span(start, end)
    assert raw[rs:re_] == "Sea ﬁnder"       # the ORIGINAL bytes, ligature intact


def test_confusable_fold_is_length_preserving():
    """Offsets are indexed with positions in the confusable-folded string; if the fold could
    change a length, every replacement after the first diacritic would land off by one."""
    for s in ["josé muñoz", "łódź", "ﬁ", "straße".casefold(), "ἀθῆναι", CYRILLIC_LOWER]:
        assert len(nf.confusable(s)) == len(s)


def test_diacritics_and_ligatures_match_the_enrolled_spelling():
    assert nf.key("José Muñoz") == nf.key("Jose Munoz")
    assert nf.key("Sea ﬁnder") == nf.key("SEA FINDER")
    assert nf.key("Łukasz Wróbel") == nf.key("Lukasz Wrobel")


# ── the boundary predicate ───────────────────────────────────────────────────

@pytest.mark.parametrize("haystack,needle,expected", [
    ("Certificate of incorporation", "Corporation", []),      # THE measured defect
    ("Silver Rock Corporation, Ltd", "Corporation", ["Corporation"]),
    ("He visited the Silver Mine", "Silver Bay Shipping", []),
    ("(Zielinski), and Zielinski.", "Zielinski", ["Zielinski", "Zielinski"]),
    ("Zielinskis Wohnung", "Zielinski", []),                  # inflected: not a bare match
    ("signed by Zielinski.", "Zielinski", ["Zielinski"]),
    ("IMO: 12345670", "1234567", []),                          # not inside a longer number
    ("IMO: 1234567.", "1234567", ["1234567"]),
    ("Northwind Investments Ltd. was", "Northwind Investments Ltd.", ["Northwind Investments Ltd."]),
    ("shareholdings", "Shareholding", []),
])
def test_boundary_predicate(haystack, needle, expected):
    hay, folded = nf.fold_haystack(haystack)
    spans = nf.find_all(hay, nf.key(needle))
    assert [haystack[slice(*folded.raw_span(a, b))] for a, b in spans] == expected


def test_boundary_is_asymmetric_where_the_needle_ends_in_punctuation():
    """A value whose own edge is punctuation must not demand a boundary on that side, or
    "S.B. Management" stops matching the moment a word follows the dot."""
    assert nf.boundary_ok("x ltd. was", 2, 6, "ltd.")
    assert not nf.boundary_ok("xltd was", 1, 4, "ltd")


# ── identity skeletons ───────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("Henry Zielinski", "Henri Zielinski"),
    ("Henry Zielinski", "Zielinski Henry"),
    ("Henry Zielinski", "Henry Zielinskiy"),
    ("Jonas Peter Mayer", "Jonas Peter Maier"),
    ("Jonas Peter Mayer", "Mayer Jonas Peter"),
])
def test_skeleton_is_romanisation_and_order_independent(a, b):
    assert nf.skeleton(a) == nf.skeleton(b)


def test_skeleton_is_script_independent_for_a_generated_cyrillic_spelling():
    """A name enrolled in Latin and the Cyrillic spelling the splitter generates for it are one
    person — and the homoglyph fold must not get in the way (``skeleton_part`` transliterates
    Cyrillic ve, U+0432, as "v"; the homoglyph table would have made it "b")."""
    cyr = " ".join(nf.cyrillic_form(p) for p in ("Henry", "Zielinski"))
    assert nf.script_of(cyr) == "cyrillic"
    assert nf.skeleton(cyr) == nf.skeleton("Henry Zielinski")
    word = nf.cyrillic_form("Novakova")
    assert nf.diacritics(word.casefold()) == word.casefold()     # morphology survives
    assert nf.confusable(word.casefold()) != word.casefold()     # the match fold does rewrite


def test_skeleton_keeps_different_people_apart():
    assert nf.skeleton("Nils Ostrander") != nf.skeleton("Nils Brandvold")
    assert nf.skeleton("Henry Zielinski") != nf.skeleton("Marek Zielinski")


def test_stem_match_tolerates_a_case_ending_and_nothing_more():
    assert nf.stem_match("ostrander", "ostrandera")
    assert nf.stem_match("ostrander", "ostranderom")
    assert not nf.stem_match("ostrander", "ostranderville")   # tail > 3
    assert not nf.stem_match("nil", "nils")                   # stem < 4


# ── admission ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    "Management", "Silver", "Bay", "Shipping", "Argentina", "Rock", "Corporation",
    "client", "husband", "Administrator", "Roadmap", "Draft",
    "Shareholdings", "Marlow Centre", "limitation ~3", "of the partition",
    "Claimant (SILVER BAY ASIA LIMITED", "Claimant (",
    "Zielinski Henry (Henry Zielinski / Henri Zielinski)",
])
def test_every_junk_shape_from_the_real_vault_is_refused(value):
    """The acceptance criterion: re-seeding must mint none of these."""
    ok, _reason = nf.admit_person(value)
    assert not ok, f"{value!r} would still be minted as a PERSON"


@pytest.mark.parametrize("value", [
    "Henry Zielinski", "Nils Ostrander", "Marek Zielinski", "Jonas Peter Mayer",
    "Mira Novakova", "Nils Brandvold", "José Muñoz",
])
def test_every_real_person_shape_is_admitted(value):
    ok, reason = nf.admit_person(value)
    assert ok, f"{value!r} refused: {reason}"


def test_org_variants_cover_the_spellings_documents_use():
    v = {nf.key(x) for x in nf.org_variants("Quorvane Shipping Ltd")}
    assert nf.key("QUORVANE SHIPPING LIMITED") in v      # measured: 'PERSON_29 LIMITED'
    assert nf.key("Quorvane Shipping") in v
    assert nf.key("SEAFINDER") in {nf.key(x) for x in nf.org_variants("SEA FINDER")}


def test_person_variants_cover_romanisation_and_order():
    v = {nf.key(x) for x in nf.person_variants("Henry Zielinski")}
    assert nf.key("Henri Zielinski") in v             # a respelling measured in real documents
    assert nf.key("Zielinski Henry") in v
    assert nf.key("H. Zielinski") in v
    cyr = [nf.cyrillic_form(p) for p in ("Zielinski", "Henry")]
    assert nf.key(" ".join(cyr)) in v                 # the Cyrillic spelling, surname first


# ── the confusable fold's contract, and the scripts that defeated it ─────────

_SAMPLE_CHARS = ("abcxyz0189 .-'ñšćłøđİi̇ıαβγεζηικμνορτυχ" + CYRILLIC_LOWER
                 + chr(0x0450) + chr(0x0439))


def test_confusable_is_one_to_one_and_length_preserving():
    """The contract the offset arrays depend on. ``fold`` records one raw span per folded
    character and the caller indexes those arrays with positions in ``confusable``'s output, so
    a single entry that emitted zero or two characters would silently move every replacement
    after it. A combining mark is therefore NOT removed here — that is ``strip_marks``, which
    repairs the offsets as it goes."""
    for ch in _SAMPLE_CHARS:
        assert len(nf.confusable(ch)) == 1, repr(ch)
    assert len(nf.confusable(_SAMPLE_CHARS)) == len(_SAMPLE_CHARS)


def test_the_cyrillic_short_i_is_kept_and_yo_is_folded():
    """U+0451 → U+0435 is a fold Russian typography makes; U+0439 → U+0438 is not — the short i
    is a letter of its own, and folding it merges names that differ only in it."""
    assert nf.key(chr(0x0451)) == nf.key(chr(0x0435))          # yo → ye
    assert nf.key(chr(0x0439)) != nf.key(chr(0x0438))          # short i stays short i
    assert nf.key(chr(0x0450)) == nf.key(chr(0x0435))          # ye with grave → ye


def test_turkish_dotted_capital_i_no_longer_hides_a_company_name():
    """MEASURED: entity "… Denizcilik Ticaret Anonim Sirketi", document "… DENİZCİLİK TİCARET
    ANONİM ŞİRKETİ" — the registered owner of a tokenised vessel, in clear, on the vessel's own
    line. ``str.casefold`` EXPANDS U+0130 to 'i' + U+0307, so the two keys differed by five
    invisible combining marks. ş→s and ç→c always folded; only the dotted I broke."""
    assert nf.key("TARVELO DENİZCİLİK TİCARET ANONİM ŞİRKETİ") == \
           nf.key("Tarvelo Denizcilik Ticaret Anonim Sirketi")
    raw = "as of 2024 the owner is TARVELO DENİZCİLİK TİCARET ANONİM ŞİRKETİ SEA FINDER"
    hay, folded = nf.fold_haystack(raw)
    hits = nf.find_all(hay, nf.key("Tarvelo Denizcilik Ticaret Anonim Sirketi"))
    assert len(hits) == 1
    # ...and the raw span is exactly the company name: the dropped marks were absorbed into the
    # span of the character they decorate, so the replacement covers every byte it consumed.
    start, end = folded.raw_span(*hits[0])
    assert raw[start:end] == "TARVELO DENİZCİLİK TİCARET ANONİM ŞİRKETİ"


def test_greek_homoglyphs_no_longer_hide_a_registration_number():
    """MEASURED: a company's registration number is "HE …" (Latin H, E); the document writes
    "ΗΕ …" (U+0397 U+0395) and it crossed in clear beside the token of the company it
    identifies."""
    raw = "Marlow House Registration Number: ΗΕ 999817 Registration Date: 23.01.2008"
    hay, folded = nf.fold_haystack(raw)
    hits = nf.find_all(hay, nf.key("HE 999817"))
    assert len(hits) == 1
    start, end = folded.raw_span(*hits[0])
    assert raw[start:end] == "ΗΕ 999817"


def test_russian_case_endings_are_generated_for_a_surname_but_not_for_a_latin_word():
    """MEASURED: an enrolled surname was 9/9 covered in the nominative and 0/3 in the genitive.
    ``boundary_ok`` is right to refuse the nominative inside the genitive; the surface simply
    had to be enrolled. Nothing is generated for a word that is not Cyrillic."""
    surname = nf.cyrillic_form("Zielinskiy")
    forms = nf.russian_inflections(surname)
    assert forms and all(f[:6] == surname[:6] for f in forms)
    assert len({nf.key(f) for f in forms}) == len(forms)
    assert nf.russian_inflections("Company") == []


def test_org_variants_generate_the_initialism_and_the_plural_documents_write():
    """MEASURED: "S.B. Shipping Ltd" / "S.B Shipping ltd" (9 occurrences, 0% covered) and
    "Arven Holdings Ltd" (document plural, entity singular). The documents themselves prove
    S.B. = Silver Bay: the vault holds "S.B. Management" beside "Silver Bay Shipping Ltd"."""
    v = {nf.key(x) for x in nf.org_variants("Silver Bay Shipping Ltd")}
    assert nf.key("S.B. Shipping Ltd") in v
    assert nf.key("S.B Shipping ltd") in v
    assert nf.key("SB Shipping Ltd") in v
    assert nf.key("SBShipping") in {nf.key(x) for x in nf.org_variants("Silver Bay Shipping")}
    assert nf.key("ARVEN HOLDINGS LIMITED") in {nf.key(x)
                                                for x in nf.org_variants("ARVEN HOLDING LIMITED")}
    assert nf.key("Red River B.V.") in {nf.key(x)
                                        for x in nf.org_variants("Red River Trading B.V.")}


def test_only_a_coined_leading_word_identifies_a_company_on_its_own():
    """Companies are referred to by their leading word alone, and the OLD tokenizer covered
    them. "Silver" and "Red" must never be enrolled that way — that is the case the single-word
    rule was written to close, and it stays closed."""
    assert nf.distinctive_head("Quorvane Shipping Ltd") == "Quorvane"
    assert nf.distinctive_head("NRTK Asia M6 Limited") == "NRTK"
    assert nf.distinctive_head("Tarvelo Denizcilik Ticaret Anonim Sirketi") == "Tarvelo"
    assert nf.distinctive_head("Silver Bay Shipping Ltd") is None
    assert nf.distinctive_head("Red River Trading Ltd") is None
    acronym = nf.acronym_cyrillic("NRTK")
    assert nf.script_of(acronym) == "cyrillic" and nf.translit(acronym.casefold()) == "nrtk"


@pytest.mark.parametrize("span,why", [
    ("SEA RUNNER\nContainer Ship", "line break or column gap in span"),
    ("C.  Verifying", "line break or column gap in span"),
    ("Forwarded Message", "every part is an ordinary word"),
    ("Machine Translated", "every part is an ordinary word"),
    ("Agίoy Nikoλάoy", "mixed script within one word: Agίoy"),
    ("PERSON_1@sbasia.example", "contains a token"),
])
def test_the_structural_gate_refuses_the_second_round_of_non_people(span, why):
    """Every one of these shapes acquired a PERSON token on real documents AFTER the first fix,
    and the glossary then described each as an "individual (natural person)"."""
    ok, reason = nf.admit_person(span)
    assert not ok and reason == why


def test_a_real_name_still_passes_the_hardened_gate():
    for name in ("Nils Brandvold", "José Muñoz", "Henry Zielinski", "Mira Novakova"):
        assert nf.admit_person(name)[0], name


def test_a_token_wearing_a_homoglyph_or_a_fullwidth_digit_is_still_a_token():
    """The inbound leg is a raw ASCII regex while the outbound leg folds everything. A model
    answering in a Greek- or Cyrillic-script language slips a look-alike capital into a token;
    an unrecognised token reaches the reader as a raw token."""
    assert nf.deconfuse_ascii("ΡERSON_1") == "PERSON_1"        # Greek Rho
    assert nf.deconfuse_ascii("PERSΟN_2") == "PERSON_2"        # Greek Omicron
    assert nf.deconfuse_ascii(chr(0x0420) + "ERSON_3") == "PERSON_3"  # Cyrillic Er
    assert nf.deconfuse_ascii("PERSON_１") == "PERSON_1"        # fullwidth one
    # 1:1, so a match position in the folded copy is a position in the original.
    for s in ("ΡERSON_1", "PERSΟN_2", "PERSON_１", "ordinary text", CYRILLIC_LOWER):
        assert len(nf.deconfuse_ascii(s)) == len(s)


# ── German spellings of a held name (25.09.2026) ─────────────────────────────

@pytest.mark.parametrize("held,written", [
    ("Jörg Müller", "Joerg Mueller"), ("Müller", "Mueller"), ("Mueller", "Müller"),
    ("Ölmühle Böhm GmbH", "Oelmuehle Boehm GmbH"),
])
def test_a_name_is_met_with_and_without_its_umlauts(held, written):
    assert nf.key(written) in {nf.key(v) for v in nf.spelling_variants(held)}


def test_a_persons_genitive_is_a_spelling_and_an_apostrophe_name_needs_none():
    assert "Albrechts" in nf.spelling_variants("Albrecht", genitive=True)
    assert "Mira Albrechts" in nf.spelling_variants("Mira Albrecht", genitive=True)
    assert "Muellers" in nf.spelling_variants("Müller", genitive=True)
    assert nf.spelling_variants("Albers", genitive=True) == []          # "Albers'" — the ' is a boundary
    assert nf.spelling_variants("Albrecht") == []                       # no genitive unless asked (companies)


def test_a_backslash_escape_before_a_name_is_a_boundary():
    """MEASURED 28.09.2026: a coding agent's tool output came as JSON text, so a line break before
    a company's name was the two characters backslash and n, and the name crossed in clear."""
    raw = '{"output":"Dear Ms Voss,\\n\\nBrightwater Maritime Ltd asks\\tTamsin Okafor\\u00a0Ltd"}'
    hay, folded = nf.fold_haystack(raw)
    for name in ("Brightwater Maritime Ltd", "Tamsin Okafor"):
        assert len(nf.find_all(hay, nf.key(name))) == 1, name


def test_an_escaped_backslash_before_a_letter_is_not_a_line_break():
    hay, folded = nf.fold_haystack("C:\\\\nBrightwater")          # the text  C:\\nBrightwater
    assert nf.find_all(hay, nf.key("Brightwater")) == []
