# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A pseudonym is derived from the referent, not from the order of enrolment.

Tokens used to be issued by a counter, ``max+1``. A counter has two properties a pseudonym must
not have, and both are pinned here.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from deidkit import namefold as nf
from deidkit.vault import _derive_token, _normalize

SALT = b"\x11" * 32
OTHER = b"\x22" * 32


def test_the_same_referent_always_gets_the_same_token():
    """This is the whole point of replacing the counter.

    With a counter, rebuilding the vault in another order made ``PERSON_28`` a different person
    — the old docstring said outright that re-issuing would "silently redirect 85 historical
    crossings", which is why retired tokens were kept as taken forever. Here a redirection is
    impossible by construction.
    """
    a = _derive_token(SALT, "PERSON", "Henry Zielinski", set())
    b = _derive_token(SALT, "PERSON", "Henry Zielinski", set())
    assert a == b


def test_order_of_enrolment_changes_nothing():
    first_then_second = [_derive_token(SALT, "PERSON", n, set()) for n in ("Ann", "Ben")]
    second_then_first = [_derive_token(SALT, "PERSON", n, set()) for n in ("Ben", "Ann")]
    assert first_then_second == second_then_first[::-1]


def test_surfaces_of_one_name_normalise_to_one_token():
    """The vault already joins surfaces through aliases; the derivation must not contradict it."""
    assert _normalize("  HENRY   ZIELINSKI ") == _normalize("Henry Zielinski")
    assert (_derive_token(SALT, "PERSON", "  HENRY   ZIELINSKI ", set())
            == _derive_token(SALT, "PERSON", "Henry Zielinski", set()))


def test_different_people_get_different_tokens():
    assert _derive_token(SALT, "PERSON", "Ann", set()) != _derive_token(SALT, "PERSON", "Ben", set())


def test_the_kind_is_part_of_the_identity():
    """One name as a person and as an organisation are different referents; the tokens differ —
    in the derived number, not only in the prefix the kind supplies anyway."""
    person = _derive_token(SALT, "PERSON", "Nord", set())
    org = _derive_token(SALT, "ORG", "Nord", set())
    assert person.split("_")[1] != org.split("_")[1]


def test_a_different_scope_gives_a_different_token_for_the_same_person():
    """A salt per scope: one leak does not link a person across every scope at once.

    That is a choice, not a side effect — a shared salt would allow matching people across
    de-identified texts, and introducing one has to be a separate decision.
    """
    assert (_derive_token(SALT, "PERSON", "Ann", set())
            != _derive_token(OTHER, "PERSON", "Ann", set()))


def test_the_token_is_not_a_verifier_for_a_guessed_name():
    """A bare sha256(name) would confirm a guess to anyone holding a list of candidates.

    Tokens cross to the cloud, so that would be WORSE than a counter, which says nothing.
    Checked by the token not being predictable without the salt.
    """
    naive = f"PERSON_{int(hashlib.sha256(b'Ann').hexdigest()[:8], 16) % 90_000_000 + 10_000_000}"
    assert _derive_token(SALT, "PERSON", "Ann", set()) != naive


def test_the_shape_is_unchanged_because_eight_places_depend_on_it():
    """``TOKEN_IN_TEXT``, ``_TOKEN_SHAPE``, the homoglyph table and the glossary know the
    ``KIND_DIGITS`` shape. Changing it would mean changing all of them, and a miss would give a
    token tokenised a second time — the ``PERSON_12 → PERSON_1`` defect the code guards against."""
    tok = _derive_token(SALT, "PERSON", "Ann", set())
    assert re.fullmatch(r"PERSON_\d{8}", tok)
    assert nf.TOKEN_IN_TEXT.fullmatch(tok)


def test_the_derivation_is_pinned():
    """The tokens already issued must stay valid: HMAC-SHA256(salt, kind ‖ 0x00 ‖ key), the
    first five bytes big-endian, modulo 90 000 000, plus 10 000 000."""
    import hmac
    digest = hmac.new(SALT, b"PERSON\x00" + nf.key("Ann").encode(), hashlib.sha256).digest()
    expected = f"PERSON_{int.from_bytes(digest[:5], 'big') % 90_000_000 + 10_000_000}"
    assert _derive_token(SALT, "PERSON", "Ann", set()) == expected


@pytest.mark.parametrize("kind,value,token", [
    ("PERSON", "Henry Zielinski", "PERSON_78716317"),
    ("PERSON", "  HENRY   ZIELINSKI ", "PERSON_78716317"),
    ("PERSON", "José Muñoz", "PERSON_24470782"),
    ("ORG", "Orvalis Shipping", "ORG_14351219"),
    ("ADDRESS", "Musterweg 12", "ADDRESS_31938007"),
    ("EMAIL", "jane@example.com", "EMAIL_87573269"),
])
def test_golden_tokens_stay_what_they_were(kind, value, token):
    """Fixed salt, fixed values, fixed tokens — computed once and written down. A change in the
    derivation OR in the key fold feeding it would silently re-map every token already issued,
    and only a golden value notices the second."""
    assert _derive_token(SALT, kind, value, set()) == token


def test_a_collision_resolves_deterministically_not_by_taking_the_next_free_number():
    """Taking the next free number would bring order-dependence back through the back door."""
    first = _derive_token(SALT, "PERSON", "Ann", set())
    a = _derive_token(SALT, "PERSON", "Ann", {first})
    b = _derive_token(SALT, "PERSON", "Ann", {first})
    assert a == b and a != first


def test_it_refuses_rather_than_hands_out_someone_elses_token():
    """Sixty-four taken in a row in a space of 90 million is a broken salt, not a collision.
    Failing is more honest than handing out another referent's pseudonym."""
    taken = {_derive_token(SALT, "PERSON", "Ann", set())}
    for _ in range(1, 64):
        taken.add(_derive_token(SALT, "PERSON", "Ann", set(taken)))
    with pytest.raises(RuntimeError):
        _derive_token(SALT, "PERSON", "Ann", taken)
