# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The pattern-only detector: what it reports, and what it leaves alone in code and logs."""

from deidkit.patterns import RegexDetector, card_ok, iban_ok


def kinds(text: str) -> dict[str, str]:
    return {span: kind for span, kind in RegexDetector().detect(text, "en")}


def test_an_iban_is_found_even_when_a_capitalised_word_follows_it():
    # The ECBS documentation IBAN.
    assert kinds("Pay DE89 3704 0044 0532 0130 00 EUR by Friday") == {
        "DE89 3704 0044 0532 0130 00": "IBAN"}


def test_an_iban_with_a_wrong_check_is_not_reported():
    assert not iban_ok("DE89 3704 0044 0532 0130 01")
    assert kinds("DE89 3704 0044 0532 0130 01") == {}


def test_a_card_needs_luhn_and_a_network_prefix():
    assert card_ok("4111 1111 1111 1111")                 # the well-known test Visa
    assert not card_ok("1727539200000")                   # a millisecond timestamp
    assert kinds("card 4111 1111 1111 1111, ts 1727539200000") == {"4111 1111 1111 1111": "CARD"}


def test_phones_only_in_international_form():
    found = kinds("call +44 20 7946 0958 or +1 (312) 555-0100; port 8787, build 20260928")
    assert found == {"+44 20 7946 0958": "PHONE", "+1 (312) 555-0100": "PHONE"}


def test_personal_mailboxes_but_not_role_mailboxes():
    found = kinds("ada.brenner@harrowgate.example, noreply@example.org, postmaster@host.example")
    assert found == {"ada.brenner@harrowgate.example": "EMAIL"}


def test_code_and_versions_are_left_alone():
    code = 'version = "3.12.11"\nx = y+123456789\nurl = "http://127.0.0.1:8787/v1"\nid = 9876543210'
    assert kinds(code) == {}


def test_a_token_is_never_reported():
    assert kinds("write to PERSON_1@EMAIL_2.example") == {}


def test_kinds_narrow_the_report():
    det = RegexDetector(("EMAIL",))
    assert det.detect("a.b@c.example +44 20 7946 0958", "en") == [("a.b@c.example", "EMAIL")]
