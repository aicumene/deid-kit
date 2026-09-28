# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The vault's identity layer: precision, recall, the case seam, migration, and the glossary.

Every case here is a defect that was measured in a real vault (117 rows, 54 of them defective)
or in chunks run through the old tokenizer. The fixture reproduces the SHAPE of that data with
invented names — the same roles, two brothers sharing a surname, a person spelled two ways by
two sources, a company/vessel name collision, parties whose ROLE says they are companies —
because the defects were shape-dependent, not string-dependent.

No database: ``harness.Scope`` keeps rows in memory, and the residual detector is scripted, so
"residual detection" here means "whatever the NER would have handed us", which is the only part
of the pipeline this package does not own.
"""

from __future__ import annotations

from deidkit import glossary, vault
from deidkit import namefold as nf
from harness import M, Scope, alias_row, ent, token_row

HENRY = "Henry Zielinski"


def _fleet() -> Scope:
    """Two brothers, a person two sources spell differently, a company/vessel collision,
    parties whose ROLE says they are companies, and two generic role references."""
    return Scope(
        entities=[
            ent("individual", HENRY, jurisdiction_country="BE",
                role="Manager, Beneficial Owner",
                relationships=[{"type": "shareholder", "target": "SILVER BAY ASIA LIMITED",
                                "shareholding_pct": 70, "note": "Direct ownership"}]),
            ent("individual", "Marek Zielinski", role="Shareholder / Director"),
            ent("individual", "Nils Ostrander", role="Shareholder"),
            ent("individual", "Jonas Peter Mayer", role="Shareholder"),
            ent("individual", "Mira Novakova", role="Shareholder"),
            ent("company", "SILVER BAY ASIA LIMITED", "5820418", identifier_type="company_no",
                jurisdiction_country="HK", role="Ship Owner"),
            ent("company", "Silver Bay Shipping Ltd", jurisdiction_country="CY"),
            ent("company", "Silver Rock Corporation, Ltd", jurisdiction_country="VG",
                role="Vessel Owner"),
            ent("company", "Quorvane Shipping Ltd", jurisdiction_country="CY"),
            ent("company", "Red River Trading Ltd", jurisdiction_country="CY"),
            ent("company", "Red River Trading B.V.", jurisdiction_country="NL"),
            ent("company", "Tarvelo Denizcilik Ticaret Anonim Sirketi", jurisdiction_country="TR"),
            ent("company", "S.B. Management", "0461937258", jurisdiction_country="BE"),
            ent("company", "S.B.Management Nordic", jurisdiction_country="SE"),
            ent("company", "S.B. Argentina S.R.L.", jurisdiction_country="AR"),
            ent("company", "Marlix"),
            ent("vessel", "MARLIX"),
            ent("vessel", "SILVER ROCK", "2345674", identifier_type="imo"),
            ent("vessel", "SEA FINDER", "1234567", identifier_type="imo",
                attributes={"imo": "1234567", "mmsi": "200000123", "callsign": "QX7Z3",
                            "year_built": "2011", "valuation_usd_low": 13800000}),
            ent("property", "Suburban house in Dayton, Ohio"),
        ],
        parties=[
            {"name": "S.B. Management", "role": "Company, Counterparty / Company"},
            {"name": HENRY, "role": "Manager, Beneficial Owner / Shareholder"},
            {"name": "Jonas Peter Maier", "role": "Shareholder / Director"},
            {"name": "Silver Bay Shipping Ltd", "role": "Entity"},
            {"name": "Tarvelo", "role": "Manager/Entity, Vessel Owner"},
            {"name": "Silver Rock Corporation, Ltd", "role": "Vessel Owner"},
            {"name": "Quorvane Shipping", "role": "Company / Shipowner"},
            {"name": "the client", "role": "Client / Spouse"},
            {"name": "the husband", "role": "Spouse"},
        ],
    )


def _is_mayer(value: str) -> bool:
    """Either spelling of Jonas Peter Mayer's surname, in any script."""
    return nf.skeleton_part("Mayer") in nf.skeleton(value)


# ── STEP 1: precision ────────────────────────────────────────────────────────

async def test_party_roles_are_honoured_so_org_names_are_never_split():
    """The alias path minted "Silver", "Bay", "Shipping", "Management", "Corporation",
    "client", "husband" and the COUNTRY "Argentina" — every one of them from splitting a party
    that the party list says is a company."""
    s = _fleet()
    await s.seed()
    surfaces = {r.real_value for r in s.rows} | {a.surface for a in s.aliases}
    for junk in ("Silver", "Bay", "Shipping", "Management", "Corporation", "Rock",
                 "Argentina", "client", "husband"):
        assert junk not in surfaces, f"{junk!r} was enrolled as a surface"


async def test_a_country_name_is_never_tokenised():
    """Tokenising "Argentina" destroys the jurisdiction signal the design preserves."""
    s = _fleet()
    red = await s.tokenize("The subsidiary in Argentina and its office in Cyprus.", language="en")
    assert "Argentina" in red.text and "Cyprus" in red.text


async def test_no_word_boundary_defect():
    """The measured payload said "Certificate of inPERSON_28"."""
    s = _fleet()
    red = await s.tokenize("Certificate of incorporation: 5820418", language="en")
    assert red.text.startswith("Certificate of incorporation:")
    assert "ID_" in red.text                      # the number IS tokenised


async def test_a_property_description_is_not_a_designator():
    s = _fleet()
    await s.seed()
    assert not any(r.kind == "PROPERTY" for r in s.rows)


async def test_residual_detections_that_are_not_people_are_refused():
    """Measured on a re-run: one English chunk minted 'Shareholdings' and a BUILDING as
    PEOPLE, another chunk minted five more rows. The defect is generative, so the gate has to
    be structural, not a blocklist."""
    s = _fleet()
    s.detections = [("Shareholdings", "PERSON"), ("Marlow Centre", "PERSON"),
                    ("Forwarded Message", "PERSON"), ("Claimant (", "PERSON"),
                    ("of the partition", "PERSON"), ("Nils Brandvold", "PERSON")]
    text = ("Shareholdings in companies at Marlow Centre; Forwarded Message; "
            "Nils Brandvold signed.")
    red = await s.tokenize(text, language="en")
    assert "Shareholdings" in red.text and "Marlow Centre" in red.text
    assert "Forwarded Message" in red.text
    assert "Nils Brandvold" not in red.text        # a real person IS caught
    assert s.token_of("Nils Brandvold") is not None


# ── STEP 1: fragments ────────────────────────────────────────────────────────

async def test_a_name_part_is_an_alias_not_a_token():
    """Five individuals + ONE shared surname; not five + given + surname + variants each —
    and the Russian case endings the splitter enrols for the surname's Cyrillic spelling are
    aliases of the shared token, not a second shared token per case ending."""
    s = _fleet()
    await s.seed()
    person_tokens = {r.token for r in s.rows if r.kind == "PERSON"}
    assert len(person_tokens) == 6
    assert {a.token for a in s.aliases if a.surface == "Henry"} == {s.token_of(HENRY)}


async def test_a_surname_shared_by_two_people_gets_its_own_token_not_a_guess():
    """"Zielinski" is both brothers'. Attributing it to one is a silent factual error in a
    payload somebody will rely on."""
    s = _fleet()
    red = await s.tokenize("The assets of Zielinski are frozen.", language="en")
    assert "Zielinski" not in red.text
    tok = next(iter(red.mapping))
    row = next(r for r in s.rows if r.token == tok)
    assert set(row.attributes["shared_by"]) == {s.token_of(HENRY), s.token_of("Marek Zielinski")}
    assert any("shared by" in line for line in red.glossary)


async def test_an_unambiguous_surname_resolves_to_its_owner():
    s = _fleet()
    red = await s.tokenize("Ostrander holds 20%.", language="en")
    assert red.mapping[next(iter(red.mapping))] == "Ostrander"
    assert next(iter(red.mapping)) == s.token_of("Nils Ostrander")


# ── STEP 2: recall and identity ──────────────────────────────────────────────

async def test_one_referent_one_token_across_spelling_and_order():
    """One man held TEN tokens in the real vault, twelve after three chunks."""
    s = _fleet()
    text = (f"{HENRY}, Henri Zielinski, Zielinski Henry, H. Zielinski "
            "and HENRY ZIELINSKI are all the same man.")
    red = await s.tokenize(text, language="en")
    tok = s.token_of(HENRY)
    assert red.text.count(tok) == 5
    # three rows carry the surname and that is the correct number: the two brothers, plus the
    # one shared-surname token.
    surname_rows = [r for r in s.rows if r.kind == "PERSON" and "ielinsk" in r.real_value]
    assert len(surname_rows) == 3
    assert sum(1 for r in surname_rows if (r.attributes or {}).get("shared_by")) == 1


async def test_the_cyrillic_spelling_of_a_known_name_is_the_same_token():
    """A Russian-language document writes a Latin-script name in Cyrillic. The spelling is
    generated here, not written into the test: the point is the enrolment, not the language."""
    s = _fleet()
    cyrillic = " ".join(nf.cyrillic_form(p) for p in ("Zielinski", "Henry"))
    red = await s.tokenize(f"{cyrillic} signed.", language="ru")
    assert red.text == f"{s.token_of(HENRY)} signed."


async def test_the_russian_genitive_of_a_shared_surname_goes_to_the_shared_token():
    """The genitive of an enrolled surname crossed in clear beside the person's initials. The
    declension is enrolled — as an alias of the SHARED token, because the surname is shared by
    two brothers and the fix must not silently pick one of them."""
    s = _fleet()
    genitive = nf.russian_inflections(nf.cyrillic_form("Zielinskiy"))[0]
    red = await s.tokenize(f"{genitive} signed.", language="ru")
    assert genitive not in red.text
    row = next(r for r in s.rows if r.token == next(iter(red.mapping)))
    assert set(row.attributes["shared_by"]) == {s.token_of(HENRY), s.token_of("Marek Zielinski")}


async def test_two_sources_that_disagree_on_a_spelling_reconcile_to_one_token():
    """The entities say "Mayer", the party list says "Maier"; the old vault made two tokens and
    neither source ever reconciled."""
    s = _fleet()
    await s.seed()
    assert len({r.token for r in s.rows if _is_mayer(r.real_value)}) == 1


async def test_legal_form_synonyms_and_compact_forms_are_matched():
    """Measured: documents write "QUORVANE SHIPPING LIMITED" and "SEAFINDER"; the vault held
    "Quorvane Shipping Ltd" and "SEA FINDER" and matched neither."""
    s = _fleet()
    red = await s.tokenize("QUORVANE SHIPPING LIMITED chartered SEAFINDER IMO: 1234567.",
                           language="en")
    assert "QUORVANE" not in red.text and "SEAFINDER" not in red.text
    assert red.text.count("ORG_") == 1 and "VESSEL_" in red.text


async def test_mmsi_and_callsign_cross_as_tokens_not_in_clear():
    """The IMO was tokenised while the MMSI and call sign of the SAME hull crossed in clear on
    the same line, which re-identifies the token that was protecting it."""
    s = _fleet()
    red = await s.tokenize("IMO: 1234567 MMSI: 200000123 Callsign: QX7Z3", language="en")
    assert "200000123" not in red.text and "QX7Z3" not in red.text


async def test_diacritics_are_matched_against_the_ascii_enrolment():
    s = Scope(entities=[ent("individual", "Jose Munoz", jurisdiction_country="ES")])
    red = await s.tokenize("Saludos cordiales, José Muñoz, Accountant", language="en")
    assert "Muñoz" not in red.text


async def test_a_name_only_the_mailbox_knows_is_still_caught():
    """One scope had NO individual among its entities and the English NER does not label a
    name in a Spanish email — so the accountant crossed in clear on the line above his own
    (tokenised) address."""
    s = Scope(entities=[ent("company", "Cor-Vida S.L.", "B12345678")])
    s.detections = [("jose.munoz@corvida.example", "EMAIL")]
    red = await s.tokenize("José Muñoz\nAccountant\njose.munoz@corvida.example", language="en")
    assert "Muñoz" not in red.text and "@corvida.example" not in red.text


async def test_a_mailbox_that_is_a_function_does_not_become_a_person():
    s = Scope()
    s.detections = [("sales.office@example.com", "EMAIL")]
    await s.tokenize("write to sales.office@example.com", language="en")
    assert not any(r.kind == "PERSON" for r in s.rows)


async def test_distinct_legal_persons_of_one_corporate_family_never_merge():
    """"Red River Trading Ltd" and "Red River Trading B.V." are two companies; a fix that
    normalised harder would collapse them, and the question may be exactly which one owes what.

    The bare form is enrolled for NEITHER of them — but it is not enrolled for nothing either.
    Abstaining left the bare name in clear five times in real documents, on lines that also
    carry other companies' tokens, so the remedy the design applies to a surname shared by two
    brothers applies here: a token of its own, glossed as shared."""
    s = _fleet()
    await s.seed()
    ltd, bv = s.token_of("Red River Trading Ltd"), s.token_of("Red River Trading B.V.")
    assert ltd != bv
    bare = next(a for a in s.aliases if a.normalized == "red river trading")
    assert bare.token not in (ltd, bv), "the bare form was attributed to one of the two"
    row = next(r for r in s.rows if r.token == bare.token)
    assert row.kind == "ORG"
    assert set(row.attributes["shared_by"]) == {ltd, bv}


async def test_a_company_and_a_vessel_of_the_same_name_do_not_collapse_silently():
    """A company "Marlix" and a vessel "MARLIX": a unique key let only one row exist and the
    vessel silently had no token at all."""
    s = _fleet()
    await s.seed()
    row = next(r for r in s.rows if r.real_value.casefold() == "marlix")
    assert row.attributes.get("also_kind") == "VESSEL"
    assert s.token_of("Silver Rock Corporation, Ltd") != s.token_of("SILVER ROCK")


async def test_one_person_written_as_two_adjacent_spans_is_one_token():
    """A name written surname-first crossed as "PERSON_67 PERSON_68" — a cloud model reading
    that sees two shareholders where there is one."""
    s = _fleet()
    red = await s.tokenize("shareholder Mayer Jonas Peter (5%)", language="en")
    tok = s.token_of("Jonas Peter Mayer")
    assert red.text.count(tok) == 1


# ── STEP 2: the case seam ────────────────────────────────────────────────────

async def test_detokenize_is_case_insensitive_because_the_model_chooses_the_case():
    """Outbound has always been case-insensitive; inbound was ``str.replace``. Anything the
    model echoed lower-cased reached the reader as a raw token."""
    s = _fleet()
    red = await s.tokenize(f"{HENRY} and SILVER BAY ASIA LIMITED", language="en")
    tok = s.token_of(HENRY)
    for spelling in (tok, tok.lower(), tok.capitalize(), tok.title()):
        out = await s.detokenize(f"Per {spelling}, the answer is yes.", mapping=red.mapping)
        assert HENRY in out, spelling


async def test_detokenize_does_not_confuse_org_1_with_org_12():
    s = Scope()
    out = await s.detokenize("ORG_12 and ORG_1 and ORG_1x",
                             mapping={"ORG_1": "Acme", "ORG_12": "Globex"})
    assert out == "Globex and Acme and ORG_1x"


async def test_roundtrip_is_the_identity_when_one_spelling_was_used():
    s = _fleet()
    text = "QUORVANE SHIPPING LIMITED and Henri Zielinski in Cyprus on 2024-03-01, IMO 1234567."
    red = await s.tokenize(text, language="en")
    assert await s.detokenize(red.text, mapping=red.mapping) == text


# ── STEP 2: migration ────────────────────────────────────────────────────────

def _legacy_rows():
    """The shape of an old vault: a good row, a duplicate, a fragment, junk, a second spelling,
    and a token historical crossings refer to by name."""
    return [
        token_row("PERSON_1", "PERSON", HENRY),
        token_row("PERSON_2", "PERSON", "Henry"),
        token_row("PERSON_3", "PERSON", "Zielinski"),
        token_row("PERSON_7", "PERSON", "Marek Zielinski"),
        token_row("PERSON_19", "PERSON", "Management"),
        token_row("PERSON_28", "PERSON", "Corporation"),
        token_row("PERSON_41", "PERSON", "Administrator"),
        token_row("PERSON_49", "PERSON", "Henri Zielinski"),
        token_row("PERSON_53", "PERSON", "Nils Brandvold"),
        token_row("ORG_2", "ORG", "SILVER BAY ASIA LIMITED"),
    ]


async def test_reconcile_retires_junk_merges_duplicates_and_deletes_nothing():
    s = _fleet()
    s.rows = _legacy_rows()
    before = {r.token for r in s.rows}
    report = await s.reconcile()

    assert {r.token for r in s.rows} >= before, "a row was deleted"
    by_token = {r.token: r for r in s.rows}
    assert by_token["PERSON_19"].status == "retired"          # junk
    assert by_token["PERSON_28"].status == "retired"
    assert by_token["PERSON_41"].status == "retired"
    assert by_token["PERSON_19"].canonical_token is None      # junk has no referent
    assert by_token["PERSON_2"].status == "retired"           # duplicate
    assert by_token["PERSON_2"].canonical_token == "PERSON_1"
    assert by_token["PERSON_49"].canonical_token == "PERSON_1"
    assert by_token["PERSON_1"].status == "active"
    assert by_token["PERSON_53"].status == "active"           # a genuine residual catch
    assert by_token["PERSON_3"].attributes["shared_by"] == ["PERSON_1", "PERSON_7"]
    assert {m["token"] for m in report["merged"]} == {"PERSON_2", "PERSON_49"}


async def test_a_retired_token_still_re_hydrates_because_historical_crossings_reference_it():
    """...and a retired DUPLICATE re-hydrates to the referent's current name, not to the
    fragment it used to hold on its own. "Henry" was never a person; it was this man's given
    name. A retired JUNK row, which has no referent, still returns its own literal value."""
    s = _fleet()
    s.rows = _legacy_rows()
    await s.reconcile()
    out = await s.detokenize("PERSON_28 and PERSON_2 were named in a June answer.")
    assert out == f"Corporation and {HENRY} were named in a June answer."


async def test_a_retired_token_can_never_fire_outbound_again():
    s = _fleet()
    s.rows = _legacy_rows()
    await s.reconcile()
    red = await s.tokenize("A Corporation is not a person.", language="en")
    assert "PERSON_28" not in red.text and "Corporation" in red.text


async def test_an_alias_left_behind_by_a_retired_token_does_not_fire():
    """Retirement has to reach the alias index, not just the token row. A row kept by one
    reconcile pass carries alias rows; if a later pass retires it, those aliases are still in
    the store and would keep firing outbound — which is precisely the state retirement exists
    to make impossible."""
    s = Scope(rows=[token_row("PERSON_9", "PERSON", "Corporation", status="retired")],
              aliases=[alias_row("PERSON_9", "Corporation", "corporation", "legacy")])
    red = await s.tokenize("A Corporation is not a person.", language="en", seed=False)
    assert red.text == "A Corporation is not a person."
    # ...and it still re-hydrates, because historical crossings used it
    assert await s.detokenize("PERSON_9") == "Corporation"


async def test_a_retired_number_is_never_reissued():
    s = _fleet()
    s.rows = _legacy_rows()
    await s.reconcile()
    minted = {r.token for r in s.rows}
    assert len(minted) == len(s.rows)
    legacy = {r.token for r in _legacy_rows()}
    assert all(int(t.split("_")[1]) > 53 for t in minted
               if t.startswith("PERSON_") and t not in legacy)


async def test_reconcile_is_idempotent():
    s = _fleet()
    s.rows = _legacy_rows()
    await s.reconcile()
    tokens = len(s.rows)
    report = await s.reconcile()
    assert len(s.rows) == tokens
    assert report["merged"] == [] and report["retired"] == []


async def test_tokenizing_the_same_text_twice_does_not_grow_the_vault():
    """The old vault grew an orphan row per crossing: a token minted and never applied."""
    s = _fleet()
    text = "Henri Zielinski and QUORVANE SHIPPING LIMITED."
    r1 = await s.tokenize(text, language="en")
    n = len(s.rows)
    r2 = await s.tokenize(text, language="en")
    assert len(s.rows) == n and r1.text == r2.text
    assert set(r1.mapping) <= {r.token for r in s.rows}


# ── STEP 3: the glossary ─────────────────────────────────────────────────────

async def test_glossary_describes_the_tokens_that_crossed():
    s = _fleet()
    red = await s.tokenize(f"{HENRY} holds shares in SILVER BAY ASIA LIMITED.", language="en")
    person = s.token_of(HENRY)
    line = next(ln for ln in red.glossary if ln.startswith(person + " ="))
    assert "individual (natural person)" in line
    assert "male (from surname morphology)" in line
    assert "jurisdiction BE" in line
    assert f"shareholder in {s.token_of('SILVER BAY ASIA LIMITED')}" in line


async def test_glossary_bands_a_shareholding_and_never_prints_the_figure():
    """"70 / 20 / 5 / 5 of a Hong Kong shipping company" is a registry search query. The bands
    are cut where ownership thresholds are cut, so the reasoning survives and the fingerprint
    does not."""
    s = _fleet()
    red = await s.tokenize(f"{HENRY} and SILVER BAY ASIA LIMITED", language="en")
    blob = "\n".join(red.glossary)
    assert "≥50%" in blob
    assert "70%" not in blob and "70 %" not in blob


async def test_glossary_never_names_anything():
    s = _fleet()
    red = await s.tokenize(f"{HENRY}, SILVER BAY ASIA LIMITED, SEA FINDER, 1234567.",
                           language="en")
    blob = "\n".join(red.glossary)
    for real in ("Henry", "Zielinski", "SILVER BAY", "Silver Bay", "FINDER", "1234567",
                 "Quorvane", "200000123", "QX7Z3"):
        assert real not in blob, f"glossary leaked {real!r}"


async def test_glossary_excludes_identifiers_valuations_and_notes():
    s = _fleet()
    red = await s.tokenize("SEA FINDER and its owner SILVER BAY ASIA LIMITED", language="en")
    blob = "\n".join(red.glossary)
    for excluded in ("2011", "13800000", "QX7Z3", "200000123", "Direct ownership", "5820418"):
        assert excluded not in blob


async def test_glossary_is_empty_when_nothing_crossed():
    s = _fleet()
    red = await s.tokenize("A question about Cyprus tax residence.", language="en")
    assert red.glossary == []


async def test_glossary_line_containing_a_real_value_is_dropped(monkeypatch):
    """Fail-closed. A glossary that leaks the thing it describes is worse than no glossary."""
    s = _fleet()
    await s.seed()
    idx = await s.index()
    monkeypatch.setattr(glossary, "_KIND_WORD",
                        dict(glossary._KIND_WORD, PERSON=f"individual, {HENRY}"))
    tok = s.token_of(HENRY)
    assert glossary.build_glossary(idx, {tok: HENRY}) == []


async def test_the_glossary_does_not_change_the_return_leg():
    """The glossary is supposed to cost the return leg nothing."""
    s = _fleet()
    text = "QUORVANE SHIPPING LIMITED and Henri Zielinski."
    with_g = await s.tokenize(text, language="en")
    without_g = await s.tokenize(text, language="en", glossary=False)
    assert with_g.text == without_g.text
    assert with_g.mapping == without_g.mapping
    assert without_g.glossary == []
    assert await s.detokenize(with_g.text, mapping=with_g.mapping) == text


# ═════════════════════════════════════════════════════════════════════════════
# ROUND TWO. Everything below is a defect the first fix INTRODUCED or left, each
# measured on real data before it was closed.
# ═════════════════════════════════════════════════════════════════════════════

# ── coverage: a precision fix that loses coverage is a leak ──────────────────

def _legacy_company_rows():
    """Single-word PERSON rows that are really companies. The OLD tokenizer covered them; the
    precision fix retired them as "single word" with no canonical and no alias, and real
    company names went back into clear text."""
    return [
        token_row("PERSON_25", "PERSON", "Tarvelo"),
        token_row("PERSON_30", "PERSON", "Quorvane"),
        token_row("PERSON_31", "PERSON", "NRTK"),
        # "Brevanco Ltd" is ONE word plus a legal form, so there is no leading word to enrol —
        # only the organisation resolver reaches this one.
        token_row("PERSON_32", "PERSON", "Brevanco"),
        token_row("PERSON_19", "PERSON", "Management"),        # still junk, still retired
    ]


async def test_a_single_word_company_row_is_merged_into_its_company_not_retired_as_junk():
    """MEASURED REGRESSION: after reconcile, the company rows were retired with canonical_token
    None and ``token_for`` of the company word returned None — so a sentence naming the
    company crossed in clear where the OLD algorithm had written a token. Each resolves
    uniquely onto an ORG by legal-form-stripped prefix; nothing about them is ambiguous."""
    s = _fleet()
    s.seeds.add_entity(M, ent("company", "NRTK Asia M6 Limited", jurisdiction_country="HK"))
    s.seeds.add_entity(M, ent("company", "Brevanco Ltd", jurisdiction_country="CY"))
    s.rows = _legacy_company_rows()
    await s.reconcile()
    by_token = {r.token: r for r in s.rows}
    for tok, org_name in (("PERSON_25", "Tarvelo Denizcilik Ticaret Anonim Sirketi"),
                          ("PERSON_30", "Quorvane Shipping Ltd"),
                          ("PERSON_31", "NRTK Asia M6 Limited"),
                          ("PERSON_32", "Brevanco Ltd")):
        assert by_token[tok].status == "retired"
        assert by_token[tok].canonical_token == s.token_of(org_name), tok
    assert by_token["PERSON_19"].canonical_token is None      # junk is still junk
    idx = await s.index()
    for surface, org_name in (("Tarvelo", "Tarvelo Denizcilik Ticaret Anonim Sirketi"),
                              ("Quorvane", "Quorvane Shipping Ltd"),
                              ("NRTK", "NRTK Asia M6 Limited"),
                              (nf.acronym_cyrillic("NRTK"), "NRTK Asia M6 Limited"),
                              ("Brevanco", "Brevanco Ltd")):
        assert idx.token_for(surface) == s.token_of(org_name), surface


async def test_the_bare_company_word_crosses_as_the_company_token():
    s = _fleet()
    red = await s.tokenize("Vessels managed by Tarvelo and chartered from Quorvane.",
                           language="en")
    assert "Tarvelo" not in red.text and "Quorvane" not in red.text
    assert red.mapping[s.token_of("Tarvelo Denizcilik Ticaret Anonim Sirketi")] == "Tarvelo"


async def test_a_coined_company_word_is_enrolled_but_an_ordinary_one_is_not():
    """The guard on the fix above. "Silver Bay Shipping" must not enrol "Silver"."""
    s = _fleet()
    await s.seed()
    idx = await s.index()
    assert idx.token_for("Quorvane") == s.token_of("Quorvane Shipping Ltd")
    assert idx.token_for("Silver") is None
    assert idx.token_for("Red") is None


# ── recall: people the residual pass found ───────────────────────────────────

async def test_a_person_the_residual_pass_found_gets_fragment_aliases_too():
    """MEASURED: a scope with ZERO individual entities and an EMPTY party list has only
    residual people — and fragments were enrolled only from the seed graph. The salutation
    "José, saludos" crossed in clear one line above his own tokenised address and his own
    token, which publishes the mapping."""
    s = Scope(entities=[ent("company", "Cor-Vida S.L.", "B12345678")])
    s.detections = [("Jose Munoz", "PERSON")]
    await s.tokenize("Best regards, Jose Munoz", language="en")
    idx = await s.index()
    tok = s.token_of("Jose Munoz")
    assert idx.token_for("Jose") == tok
    assert idx.token_for("Munoz") == tok
    red2 = await s.tokenize("Jose, saludos\nhere it is.", language="en", seed=False)
    assert "Jose" not in red2.text


# ── one man, one token ───────────────────────────────────────────────────────

def _duplicate_man_rows():
    return [
        token_row("PERSON_9", "PERSON", "Jonas Peter Mayer"),
        token_row("PERSON_20", "PERSON", "Jonas Peter Maier"),
        token_row("PERSON_21", "PERSON", "Maier"),
    ]


async def test_one_man_ends_with_exactly_one_active_token():
    """MEASURED: reconcile turned the single-word row into a NEW person in the same run in which
    the second spelling merged correctly, because the single-word row was resolved against an
    index in which the multi-part row had not yet been merged. The man then held two active
    tokens and the payload named him twice on one line."""
    s = _fleet()
    s.rows = _duplicate_man_rows()
    await s.reconcile()
    idx = await s.index()
    active = [r for r in s.rows
              if r.status == "active" and r.kind == "PERSON" and _is_mayer(r.real_value)]
    assert len(active) == 1, [r.token for r in active]
    for tok in ("PERSON_20", "PERSON_21"):
        assert idx.canonical(tok) == active[0].token


async def test_a_shared_name_never_names_a_token_that_was_retired_into_another():
    """MEASURED: shared_by = ["PERSON_20", "PERSON_9"] where PERSON_20 is retired INTO PERSON_9 —
    the payload line asserted that one man is two people, and a real model wrote a table row
    saying exactly that. ``shared_by`` is a merge-ordering artefact unless it is re-derived
    after every merge has landed."""
    s = _fleet()
    s.rows = _duplicate_man_rows()
    await s.reconcile()
    idx = await s.index()
    for row in s.rows:
        shared = (row.attributes or {}).get("shared_by") or []
        if row.status != "active" or not shared:
            continue
        canon = [idx.canonical(t) for t in shared]
        assert len(set(canon)) == len(canon), (row.token, shared)
        for t in shared:
            assert idx.by_token[t].status == "active", (row.token, t)


async def test_no_two_tokens_in_one_crossing_are_fragments_of_one_person():
    """A round-trip check cannot see this: the round trip was TRUE for the payload that read
    "PERSON_61 PERSON_9 (5%)" — two tokens, two names, one man."""
    s = _fleet()
    red = await s.tokenize("shareholder Mayer Jonas Peter (5%) - a friend of the son",
                           language="en")
    idx = await s.index()
    toks = list(red.mapping)
    for i, a in enumerate(toks):
        for b in toks[i + 1:]:
            assert not idx.same_referent(a, b), (a, red.mapping[a], b, red.mapping[b])


async def test_two_adjacent_tokens_of_one_man_are_merged_into_the_more_specific_one():
    """MEASURED: one man's name crossed as two tokens on one line, and the equality test
    ``token == last_token`` could not fire because the tokens differ. Here the surname (in the
    German genitive) is the SHARED-surname token of two brothers and the given name is one
    brother's: the merge has to fire on the REFERENT, and it has to keep the specific token,
    because a shared-name token swallowing the person it is shared with is the same error
    upside down."""
    s = _fleet()
    red = await s.tokenize("signed: Zielinskis Henry", language="en")
    henry = s.token_of(HENRY)
    assert red.text.strip().endswith(henry)
    assert len(list(red.mapping)) == 1
    assert red.mapping[henry] == "Zielinskis Henry"


# ── the glossary describes what crossed, and nothing else ───────────────────

async def test_the_glossary_names_only_tokens_that_are_in_the_payload():
    """MEASURED: across 35 real chunks the glossary emitted 286 lines naming 182 tokens absent
    from the chunk's own text — 64% of every token reference — and 220 of them were absent from
    ``mapping``, so the return leg could not reverse a single one."""
    s = _fleet()
    red = await s.tokenize(f"{HENRY} is a director.", language="en")
    named = {m.group(0) for line in red.glossary for m in nf.TOKEN_IN_TEXT.finditer(line)}
    assert named, "the glossary said nothing at all"
    in_text = {m.group(0) for m in nf.TOKEN_IN_TEXT.finditer(red.text)}
    assert named <= in_text, sorted(named - in_text)
    assert named <= set(red.mapping), sorted(named - set(red.mapping))


async def test_a_banded_shareholding_is_dropped_when_the_payload_states_the_figure():
    """MEASURED: the payload carrying "PERSON_1 = … shareholder in ORG_2 (≥50%)" also carried
    "PERSON_1 70%" in clear four lines down, so the fingerprint the band exists to withhold
    travelled in the same envelope. A band beside its own raw number is decoration."""
    s = _fleet()
    quiet = await s.tokenize(f"{HENRY} and SILVER BAY ASIA LIMITED", language="en")
    assert any("≥50%" in ln for ln in quiet.glossary)
    loud = await s.tokenize(f"{HENRY} 70% of SILVER BAY ASIA LIMITED", language="en")
    assert not any("≥50%" in ln for ln in loud.glossary)
    assert not any("shareholder in" in ln for ln in loud.glossary)


async def test_the_glossary_refuses_to_call_an_uncorroborated_span_a_natural_person():
    """MEASURED: "PERSON_4 = individual (natural person)" for the string "Forwarded Message",
    and the same line for a hull that is VESSEL_8 in the same payload. A glossary line is an
    assertion; with nothing behind the token there is nothing to assert."""
    s = Scope()
    s.detections = [("Konto Kom", "PERSON")]
    red = await s.tokenize("Konto Kom appears in the ledger.", language="en")
    assert red.mapping, "nothing crossed, so the test proves nothing"
    assert not any("natural person" in ln for ln in red.glossary)


async def test_a_shared_name_line_survives_without_naming_unmappable_tokens():
    """The warning is the part that matters. When the people who share the name did not cross,
    naming them would hand the model tokens the return leg cannot reverse — so the line says
    how many there are and refuses to name them."""
    s = _fleet()
    red = await s.tokenize("The assets of Zielinski are frozen.", language="en")
    line = next(ln for ln in red.glossary if "shared" in ln)
    named = {m.group(0) for m in nf.TOKEN_IN_TEXT.finditer(line)}
    assert named <= set(red.mapping)
    assert "do not attribute it" in line


# ── de-tokenisation reverses only what this request tokenised ───────────────

async def test_a_glossary_line_naming_an_unmappable_token_is_dropped(monkeypatch):
    """The fail-closed backstop behind the visibility rule: a token the request never sent is
    one the return leg cannot reverse — measured before the rule, one payload named 14 such
    tokens and a real model echoed 13 of them in a single answer."""
    s = _fleet()
    await s.seed()
    idx = await s.index()
    tok = s.token_of(HENRY)
    monkeypatch.setattr(glossary, "_KIND_WORD",
                        dict(glossary._KIND_WORD, PERSON="individual, associated with ORG_99"))
    assert glossary.build_glossary(idx, {tok: HENRY}, f"{tok} signed.") == []


async def test_the_full_vault_path_resolves_a_retired_duplicate_to_its_canonical_row():
    """MEASURED: an answer naming two retired duplicates of one man, neither of which crossed,
    came back naming him as two separate parties."""
    s = _fleet()
    s.rows = _legacy_rows()
    await s.reconcile()
    out = await s.detokenize("PERSON_2 and PERSON_49 and PERSON_1.")
    assert out == f"{HENRY} and {HENRY} and {HENRY}."


async def test_a_token_the_model_invented_is_not_given_a_name():
    """The scope rule. ``mapping`` is the tokens THIS request sent; anything else stays a
    token, visibly, rather than being handed a real name the cloud never received."""
    s = _fleet()
    red = await s.tokenize(f"{HENRY} is a director.", language="en")
    out = await s.detokenize("Consider PERSON_1 and also ORG_29 and PERSON_53.",
                             mapping=red.mapping)
    assert HENRY in out
    assert "ORG_29" in out and "PERSON_53" in out


async def test_a_token_wrapped_in_markdown_emphasis_comes_back():
    """``_`` was in both guard classes — it is the token's own separator AND markdown's
    emphasis marker, so ``_PERSON_1_`` and ``__PERSON_1__`` reached the reader as raw tokens."""
    s = Scope()
    mapping = {"ORG_1": "Acme", "ORG_12": "Globex"}
    out = await s.detokenize("_ORG_1_ and __ORG_1__ and ORG_1_ORG_12 and ORG_12 and ORG_1x",
                             mapping=mapping)
    assert out == "_Acme_ and __Acme__ and Acme_Globex and Globex and ORG_1x"


async def test_a_token_the_model_spelled_with_a_homoglyph_comes_back():
    """A model answering in a Greek- or Cyrillic-script language slips a look-alike capital
    into the token; a fullwidth digit does the same. Greek Rho and Omicron here."""
    s = Scope()
    out = await s.detokenize("Share of ΡERSON_1 and PERSΟN_1 and PERSON_１ is 70 %.",
                             mapping={"PERSON_1": "Zielinski"})
    assert out.count("Zielinski") == 3 and "PERSON" not in out


async def test_one_call_that_tokenises_several_texts_names_a_man_one_way():
    """MEASURED: a chat merged per-call mappings with ``dict.update``, and each had chosen the
    spelling its OWN text used, so one conversation named the same man two ways."""
    s = _fleet()
    first = await s.tokenize("Henri Zielinski holds 70%.", language="en")
    second = await s.tokenize("Zielinski Henry owns 70%.", language="en")
    tok = s.token_of(HENRY)
    assert first.mapping[tok] != second.mapping[tok]        # each text used its own spelling
    merged = vault.merge_mappings([first, second])
    assert merged[tok] == HENRY                             # the canonical value, once


# ── the residual pass may not mint a token for a token ──────────────────────

async def test_the_structural_path_refuses_a_surface_that_contains_a_token():
    """MEASURED: an anchored token-shape test only caught a surface that IS a token.
    "PERSON_1@…" walked past it and the vault stored a row whose real value is a live token;
    the whole-vault re-hydration path would have handed that to the reader."""
    s = _fleet()
    s.detections = [("PERSON_1@sbasia.example", "EMAIL")]
    await s.tokenize("Contact PERSON_1 at PERSON_1@sbasia.example.", language="en")
    assert not any(nf.TOKEN_IN_TEXT.search(r.real_value) for r in s.rows)
    assert not any(nf.TOKEN_IN_TEXT.search(a.surface) for a in s.aliases)


def test_the_admission_gate_refuses_a_structural_surface_containing_a_token():
    """Proven on its own, because the write guard below would otherwise mask it: the two are
    deliberately redundant, and a redundant guard that is never tested is one guard."""
    assert vault._admit("PERSON_1@sbasia.example", "EMAIL") == (False, "contains a token")
    assert vault._admit("jose.munoz@corvida.example", "EMAIL")[0]


async def test_the_write_guard_refuses_a_token_bearing_value_even_if_admission_let_it_past():
    """...and proven on its own too. The invariant is enforced at the WRITE, so that a future
    detector cannot reintroduce it by relaxing one admission rule."""
    s = Scope()
    idx = await s.index()
    assert vault._new_token(s.store, idx, M, "PERSON_1@sbasia.example", "EMAIL") is None
    assert s.rows == []


async def test_a_residual_span_that_is_an_enrolled_vessel_is_not_a_person():
    """MEASURED: a hull followed by the words "Container Ship" became a PERSON while the same
    hull was VESSEL_8, and because the residual key is LONGER, longest-first matching made the
    ship cross as a natural person."""
    s = _fleet()
    s.detections = [("SEA FINDER Container Ship", "PERSON")]
    red = await s.tokenize("SEA FINDER Container Ship, 2014.", language="en")
    assert not any(r.kind == "PERSON" and "FINDER" in r.real_value for r in s.rows)
    assert red.text.startswith("VESSEL_")


async def test_a_town_inside_a_companys_registered_name_is_not_a_person():
    """MEASURED: a town inside a company's registered name became PERSON_1. Locations are
    preserved by design; a location that acquires a PERSON token is that design inverted."""
    s = Scope(entities=[ent("company", "COR-VIDA S.L. Monte Alto", "B12345678")])
    s.detections = [("Monte Alto", "PERSON")]
    red = await s.tokenize("Registered office: Monte Alto, Spain.", language="en")
    assert "Monte Alto" in red.text
    assert not any(r.kind == "PERSON" for r in s.rows)


# ── recall passes over the outbound payload ─────────────────────────────────

async def test_a_secondary_mmsi_and_call_sign_beside_a_tokenised_one_do_not_cross():
    """MEASURED: the primary identifier a token, the alternate in clear beside it. Six MMSIs and
    one call sign leaked this way, and one of them was a PRIMARY, not an alternate."""
    s = _fleet()
    red = await s.tokenize(
        "SEA FINDER\nIMO: 1234567\nMMSI: 200000123 (200000735, 200000987)\n"
        "Call sign: QX7Z3 (QX7Z4)\nFlag: PANAMA\nYear built: 2011", language="en")
    for literal in ("200000735", "200000987", "QX7Z4"):
        assert literal not in red.text, literal
    # ...and inside that same identifier record, a word is still a word: a call sign carries
    # digits; a flag name does not.
    assert "PANAMA" in red.text and "Year built" in red.text


async def test_a_figure_inside_an_identifier_record_is_not_an_imo():
    """The second guard on the recall pass. A vessel record legitimately says "IMO" and
    "MMSI", so the context gate passes — and a valuation on the next line is seven digits like
    an IMO. An IMO carries a check digit and an MMSI begins with a Maritime Identification
    Digit; a sum of money satisfies neither."""
    s = _fleet()
    red = await s.tokenize(
        "SEA FINDER\nIMO: 1234567\nMMSI: 200000123\nValuation: 1380000 USD\n"
        "Purchase price 8250000", language="en")
    assert "1234567" not in red.text          # the real identifier is still caught
    assert "1380000" in red.text and "8250000" in red.text


async def test_a_balance_sheet_is_not_a_fleet_register():
    """The guard on the fix above. An earlier draft read every all-caps word as an ITU call sign
    and every seven-digit figure as an IMO, and tokenised the word "Cyprus" — a JURISDICTION
    the design deliberately preserves — inside a company's annual accounts."""
    s = _fleet()
    red = await s.tokenize(
        "S.B. Management 0461937258\nTOTAL ASSETS 2050000\nDEBTS 1500679\n"
        "Registered in CYPRUS", language="en")
    for survivor in ("TOTAL ASSETS", "DEBTS", "CYPRUS", "2050000", "1500679"):
        assert survivor in red.text, survivor


async def test_a_domain_that_is_an_enrolled_company_does_not_cross():
    """MEASURED: the tokenised mailbox and the plaintext domain on adjacent lines. "Cor-Vida
    S.L." is one word plus a legal form, so there is no leading word to enrol and nothing but
    the company's own mailbox connects the domain to it. The evidence was already in the
    vault."""
    s = Scope(entities=[ent("company", "Cor-Vida S.L.", "B12345678")])
    s.detections = [("jose.munoz@corvida.example", "EMAIL")]
    red = await s.tokenize(
        "jose.munoz@corvida.example <http://www.corvida.example/> and corvida", language="en")
    assert "corvida.example" not in red.text and "corvida" not in red.text


async def test_a_public_service_domain_is_left_alone():
    """The guard: only a domain whose own label resolves to a company the vault already holds."""
    s = _fleet()
    red = await s.tokenize("sent from mail.outlook.com via facebook.com", language="en")
    assert "outlook.com" in red.text and "facebook.com" in red.text


# ── the index may not admit an alias whose token row is gone ────────────────

async def test_an_alias_whose_token_row_does_not_exist_never_fires():
    """The guard was ``a.token in by_token and status != 'active'``: when the row is ABSENT the
    first conjunct is False, so the key was admitted and ``tokenize`` emitted a token that is in
    neither ``mapping`` nor the glossary — a crossing that is not even auditable as a
    redaction, because ``redacted_count`` does not count it. No foreign key stands behind it."""
    s = Scope(aliases=[alias_row("PERSON_999", "Veltra Marine Holdings",
                                 "veltra marine holdings", "legacy")])
    red = await s.tokenize("Addressed to Veltra Marine Holdings Ltd.", language="en", seed=False)
    assert "PERSON_999" not in red.text
    assert red.text == "Addressed to Veltra Marine Holdings Ltd."
