"""Ported from LegalTech apps/api/tests/test_cite_us.py (b3a0d49): the desktop finds norms as the server does.

US citation extraction (USC + CFR) — normalized to the KB's title form, subsection dropped."""
from __future__ import annotations

from deidkit.agents.cite import extract_citations


def test_usc_forms_normalize_and_drop_subsection():
    assert extract_citations("under 8 U.S.C. § 1427(a) and 8 USC 1439 and 8 U.S.C. §1182") == \
        ["8 U.S.C. § 1427", "8 U.S.C. § 1439", "8 U.S.C. § 1182"]


def test_cfr_forms_normalize():
    assert extract_citations("see 8 CFR § 214.2(o) and 8 C.F.R. 245.1") == \
        ["8 CFR § 214.2", "8 CFR § 245.1"]


def test_us_and_other_jurisdictions_coexist():
    cites = extract_citations("8 U.S.C. § 1182, СК РФ ст. 39, CPR 25.1, MCA 1973 s. 25")
    for c in ("8 U.S.C. § 1182", "СК РФ ст. 39", "CPR 25.1", "MCA 1973 s. 25"):
        assert c in cites


def test_no_false_positive_from_lookalikes():
    assert extract_citations("Paid $8 USD; see 8 US states; the U.S.C. generally.") == []   # no section number


def test_order_preserving_dedupe():
    assert extract_citations("8 U.S.C. § 1427; again 8 USC 1427") == ["8 U.S.C. § 1427"]
