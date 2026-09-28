# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The anonymization vault — reversible, per-scope pseudonymization.

The two-plane model only lets *de-identified* text cross to the cloud. A lossy redactor that
replaces every name with a bare ``[PERSON]`` is irreversible, so a cloud answer reading
"[PERSON] owes [PERSON]" can't be turned back into real names. This module is a **reversible
vault** instead: each real referent gets a stable token (``PERSON_48170392``, ``ORG_2``) kept in
a :class:`~deidkit.store.TokenStore`, so the same entity always maps to the same token, and the
trusted plane can reverse the mapping to restore real names in the answer it shows the user.

What leaves the trusted store gets tokenized through here: cloud model payloads and exported
documents. The working data stays plaintext on the trusted plane — this is a *boundary* vault,
not encrypt-at-rest.

Design choices:
  * **Seeded from the scope's own graph.** We pull the known entity names/identifiers and the
    party names first (:class:`~deidkit.seeds.SeedSource`), so detection is high-quality and
    deterministic for the entities we already know, then run the residual detector
    (:class:`~deidkit.detect.PiiDetector`, e.g. Presidio) for residual PII (emails, IBANs,
    passport numbers, stray names).
  * **Locations and dates are deliberately NOT tokenized.** Jurisdictions ("Cyprus", "BVI") and
    dates are what cross-border questions and deadline arithmetic run on — tokenizing them
    would gut the answer. We hide *who*, not *where/when*.
  * **Fail-closed on the crossing, fail-open on enrichment.** If the detector is missing we
    still tokenize the known graph entities; the gateway's eligibility check remains the hard
    wall.

WHAT THE 2026-08 PRECISION/RECALL FIX CHANGED, and why each change is not optional. One real
vault held 117 rows for one scope, of which 54 were defective — 26 tokens for things that are
not referents at all ("Management", "Argentina", a Russian infinitive, nine declensions of two
Russian role nouns) and 28 duplicates of ten referents (one man held ten tokens, twelve after
three more chunks). Both defects were *generative*: one English chunk minted "Shareholdings"
and "Marlow Centre" as PEOPLE on the spot.

  1. **Roles are honoured, not ignored.** A party carries a ``role`` ("Company", "Vessel
     Owner", "Director/Individual"). The old seeder alias-split EVERY party as if it were a
     person, which is where "Silver", "Bay", "Shipping", "Management", "Corporation" and the
     country "Argentina" came from. Roles are now read; only a party the role identifies as an
     individual is split at all.
  2. **A name part is an ALIAS, not a token.** "Henry", "Zielinski", "Henri Zielinski" and the
     same name in Cyrillic, surname first, are four surfaces of one man and now resolve to ONE
     token (:class:`~deidkit.store.AliasRow`). A part shared by two referents — "Zielinski",
     the surname of two brothers — gets a token of its own, glossed as shared, because
     assigning it to either brother is a silent factual error in the payload.
  3. **Matching goes through the canonical fold** (``deidkit.textmatch.normalize``, via
     ``deidkit.namefold``) with offsets mapped back to the raw text, plus a word boundary. That
     is what stops "Certificate of in|corporation|" → "Certificate of inPERSON_28" and what
     starts catching "José Muñoz" and "Sea ﬁnder".
  4. **Detokenisation is case-insensitive.** It used to be ``str.replace``, while the outbound
     side was IGNORECASE — so a token the model echoed lower-cased came back to the reader as
     a raw token. See :func:`detokenize`.
  5. **Nothing is deleted.** Historical crossings reference tokens by name; a defective row is
     RETIRED (dropped from the match index, kept for the return leg) by
     :func:`reconcile_scope`, never removed. See that function for the migration story.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

import logging
import re
from collections import Counter

from deidkit import namefold as nf
from deidkit.detect import ANALYZED_ENTITIES as _ANALYZED_ENTITIES  # noqa: F401 (old name)
from deidkit.detect import PRESIDIO_ENTITIES as _PRESIDIO_ENTITIES  # noqa: F401 (old name)
from deidkit.detect import PRESIDIO_KIND as _PRESIDIO_KIND  # noqa: F401 (old name)
from deidkit.detect import STRUCTURAL_KINDS as _STRUCTURAL_KINDS
from deidkit.detect import VETO_KIND as _VETO_KIND
from deidkit.detect import PiiDetector
from deidkit.gateway import Redaction
from deidkit.seeds import SeedSource
from deidkit.store import ScopeId, TokenStore

log = logging.getLogger(__name__)

# entity.type → token prefix
_ENTITY_KIND: dict[str, str] = {
    "individual": "PERSON",
    "company": "ORG",
    "vessel": "VESSEL",
    "account": "ACCOUNT",
    "property": "PROPERTY",
}

# The residual detector's vocabulary — Presidio label → token kind, the LOCATION veto, the
# structural kinds — lives in ``deidkit.detect`` and is imported above under its old names.

# A value shaped like one of our own tokens (PERSON_1, ORG_12). The residual PII pass must
# never mint a token for a token — that's how you get PERSON_12 → "PERSON_1".
_TOKEN_SHAPE = nf._TOKEN_SHAPE

# A surface shorter than this is not enrolled as a match key: a two-character key fires
# somewhere in every document.
_MIN_KEY_LEN = 3

# Role words in a party's ``role`` (``SeedSource.parties``). ORG markers are tested FIRST and
# win: real data has "Vessel Owner" and "Counterparty / Company" on organisations while people
# carry "Beneficial Owner" — so the ORG test matches the phrase "vessel owner", never the
# bare word "owner".
_ORG_ROLE_MARKERS = (
    "company", "entity", "corporation", "corp", "holding", "vessel owner", "shipowner",
    "ship owner", "lessor", "lessee", "bank", "employer", "firm", "llc", "ltd", "limited",
    "insurer", "charterer", "operator", "registry", "counterparty / company", "subsidiary",
)
_PERSON_ROLE_MARKERS = (
    "individual", "person", "director", "shareholder", "beneficial owner", "manager",
    "subject", "applicant", "respondent", "claimant", "spouse", "husband", "wife", "client",
    "employee", "accountant", "witness", "officer", "signatory", "heir", "son", "daughter",
)


def _normalize(value: str) -> str:
    """The match key for a surface — the product's canonical fold plus the diacritic fold."""
    return nf.key(value)


def _is_generic(value: str) -> bool:
    return nf.is_generic(value)


def _person_aliases(name: str) -> list[str]:
    """Distinctive single-name parts of a person name. These are ALIAS surfaces now, not
    tokens of their own — see the module docstring, point 2."""
    return [surface for surface, _role in nf.person_fragments(name)]


def _next_index(tokens_in_use: set[str], kind: str) -> int:
    prefix = kind + "_"
    mx = 0
    for tok in tokens_in_use:
        if tok.startswith(prefix):
            try:
                mx = max(mx, int(tok[len(prefix):]))
            except ValueError:
                pass
    return mx + 1


def _party_kind(role: str, name: str) -> tuple[str, bool]:
    """(token kind, is_individual) for a party, from its ROLE.

    Returns ``is_individual=False`` for anything not positively identified as a person: a
    party we cannot classify is still tokenised as a whole, but it is never split into name
    parts. Splitting is the operation that produced "Silver", "Bay", "Shipping",
    "Management", "Corporation", "client", "husband" and the country "Argentina" — all six of
    the class-C rows that came from the alias path — so it now requires positive evidence.
    """
    r = (role or "").casefold()
    if any(m in r for m in _ORG_ROLE_MARKERS):
        return "ORG", False
    if any(m in r for m in _PERSON_ROLE_MARKERS):
        return "PERSON", True
    if nf.has_legal_form(name):
        return "ORG", False
    return "PERSON", False


# ─────────────────────────────────────────────────────────────────────────────
# The index: surface → token
# ─────────────────────────────────────────────────────────────────────────────

class _Index:
    """The scope's match index, loaded once per call."""

    __slots__ = (
        "_people",
        "alias_keys",
        "aliases",
        "by_token",
        "keys",
        "legacy_keys",
        "norms",
        "rows",
        "salt",
        "tokens_in_use",
        "variants",
    )

    def __init__(self, rows: list, aliases: list,
                 salt: bytes | None = None) -> None:
        # Пустая соль — законное состояние: сейф, собранный до её появления, продолжает
        # работать на счётчике, и уже выданные токены никуда не переезжают.
        self.salt = salt
        self.rows = rows
        self.by_token = {r.token: r for r in rows}
        self.tokens_in_use = {r.token for r in rows}
        self.aliases = aliases
        self.keys: dict[str, str] = {}
        # Every key that already HAS an alias row, whether or not it is currently in ``keys``.
        # ``keys`` answers "what does this surface resolve to?" and is mutated as rows retire;
        # this answers "does a row already exist?", which is what the store's UNIQUE
        # (scope, normalized) actually constrains. Reconciliation retires a row, drops its
        # key, and re-enrols the surface under the canonical token — without this set the
        # second enrolment inserts a second row for one key and the whole transaction fails.
        self.alias_keys: set[str] = {a.normalized for a in aliases}
        self.legacy_keys: set[str] = set()
        self._people: list | None = None
        self.norms: set[str] = {r.normalized for r in rows}
        # Legacy fallback: an ACTIVE row with no alias row (a store that predates
        # ``reconcile_scope``) still matches on its own value. Retired rows never do.
        for r in rows:
            if r.status == "active" and r.real_value.strip():
                k = _normalize(r.real_value)
                if len(k) >= _MIN_KEY_LEN and k not in self.keys:
                    self.keys[k] = r.token
                    self.legacy_keys.add(k)
        for a in aliases:
            row = self.by_token.get(a.token)
            # An alias whose TOKEN ROW does not exist mints a token that can be neither
            # mapped nor described: ``tokenize`` emits it, ``by_token.get`` returns None, so
            # it is skipped for both ``mapping`` and the glossary and does not even count as a
            # redaction. The old guard's first conjunct was False in exactly that case, which
            # admitted the key. There is no FK behind this, so the guard is the guarantee.
            if row is None or row.status != "active":
                continue
            self.keys[a.normalized] = a.token
            self.legacy_keys.discard(a.normalized)
        # Other spellings of the names held above (namefold.spelling_variants: "Mueller" for
        # "Müller", "Albrechts" for "Albrecht") — keys for MATCHING only: never a row, never seen by
        # enrolment (which reads ``keys``), and a key of the vault always wins over a variant.
        self.variants: dict[str, str] = {}
        held = [(r.real_value, r.token) for r in rows] + [(a.surface, a.token) for a in aliases]
        for surface, token in held:
            row = self.by_token.get(token)
            if row is None or row.status != "active" or row.kind not in ("PERSON", "ORG", "ENTITY"):
                continue
            for v in nf.spelling_variants(surface, genitive=row.kind == "PERSON"):
                k = _normalize(v)
                if len(k) >= _MIN_KEY_LEN and k not in self.keys and k not in _FRAME_KEYS:
                    self.variants.setdefault(k, token)

    def token_for(self, surface: str) -> str | None:
        return self.keys.get(_normalize(surface))

    def match_keys(self) -> dict[str, str]:
        """What text is matched on: ``keys`` and the spellings of the names they hold."""
        return {**self.variants, **self.keys} if self.variants else self.keys

    def canonical(self, token: str | None) -> str | None:
        """Follow ``canonical_token`` to the row that carries the referent today.

        A retired duplicate points at the token that absorbed it. Everything that reasons
        about WHO a token is — the glossary's ``shared_by``, the adjacency merge, the
        full-vault re-hydration path — has to ask this rather than the raw token, or it
        republishes the duplicate the merge just retired.
        """
        seen: set[str] = set()
        while token and token not in seen:
            seen.add(token)
            row = self.by_token.get(token)
            if row is None or row.status == "active" or not row.canonical_token:
                return token
            token = row.canonical_token
        return token

    def same_referent(self, a: str, b: str) -> bool:
        """Two tokens denote one referent (identical, or one canonicalises onto the other)."""
        if a == b:
            return True
        ca, cb = self.canonical(a), self.canonical(b)
        if ca == cb:
            return True
        for x, y in ((a, b), (b, a)):
            shared = (self.by_token.get(x).attributes or {}).get("shared_by") \
                if self.by_token.get(x) else None
            if shared and self.canonical(y) in {self.canonical(t) for t in shared}:
                return True
        return False

    def active(self) -> list:
        return [r for r in self.rows if r.status == "active"]

    def people(self) -> list:
        """One representative row per DISTINCT person the vault currently knows.

        Not the same as "every active PERSON row", and the difference is the whole reason the
        old vault could not tell a duplicate from a new person: before reconciliation one real
        scope held ten rows for one man, so asking "which people is the fragment of his given
        name shared by?" of the raw rows answered "six" and turned one man into a
        shared-surname token. Rows that reduce to a person already in the list (a declension,
        a transliteration, an initialised form) are folded away here; malformed and
        single-word rows are not people at all and never represent one."""
        if self._people is None:
            cands = [
                r for r in self.active()
                if r.kind == "PERSON" and not (r.attributes or {}).get("shared_by")
                and len(nf.name_parts(r.real_value)) >= 2 and nf.admit_person(r.real_value)[0]
            ]
            cands.sort(key=lambda r: (
                0 if (r.attributes or {}).get("entity_type") else 1,
                -len(nf.skeleton(r.real_value)),
                _token_index(r.token),
            ))
            reps: list = []
            for r in cands:
                sk = nf.skeleton(r.real_value)
                if any(_same_referent(nf.skeleton(x.real_value), sk) for x in reps):
                    continue
                reps.append(r)
            self._people = reps
        return self._people

    def invalidate(self) -> None:
        self._people = None


async def _load(store: TokenStore, scope_id: ScopeId) -> list:
    return list(await store.load_tokens(scope_id))


async def _load_aliases(store: TokenStore, scope_id: ScopeId) -> list:
    return list(await store.load_aliases(scope_id))


async def _salt_of(store: TokenStore, scope_id: ScopeId) -> bytes | None:
    """Соль области, если она уже заведена. ЧИТАЕТ И ТОЛЬКО ЧИТАЕТ.

    Сперва я заводил её прямо здесь — и это была ошибка того же рода, что аудит нашёл в других
    модулях: путь чтения, который пишет. Тесты сейфа подсовывают сюда хранилище-заглушку без
    записи, и упали шестьдесят девять из них — заглушка оказалась права насчёт формы, а не
    просто бедна. Заведение вынесено в :func:`ensure_salt`, которую зовёт та операция, что и так
    пишет.
    """
    return await store.load_salt(scope_id)


async def ensure_salt(store: TokenStore, scope_id: ScopeId) -> bytes:
    """Соль области, заводя её при первом обращении. Зовётся с ПИШУЩЕГО пути.

    Генерируется сама, а не берётся из конфига НАМЕРЕННО: секрет, который надо не забыть
    поменять, забывают.

    БЕЗ ``commit``, и это не мелочь. Соль попадает в хранилище вместе с остальной единицей работы
    пересечения; отдельная фиксация заставила бы вызывающего писать раньше, чем он решил писать.
    Возвращённая соль отдаётся дальше ЗНАЧЕНИЕМ, поэтому повторное чтение в той же транзакции —
    которое в транзакционном хранилище не увидело бы ещё не сброшенную строку и завело бы
    вторую — просто не нужно.
    """
    existing = await _salt_of(store, scope_id)
    if existing is not None:
        return existing
    salt = secrets.token_bytes(32)
    store.add_salt(scope_id, salt)
    return salt


async def _index_of(store: TokenStore, scope_id: ScopeId,
                    salt: bytes | None = None) -> _Index:
    """``salt`` передаётся значением, когда вызывающий её уже завёл: строка, добавленная в
    транзакционное хранилище без сброса, ещё не видна собственному чтению, и повторное чтение
    завело бы вторую."""
    return _Index(await _load(store, scope_id), await _load_aliases(store, scope_id),
                  salt if salt is not None else await _salt_of(store, scope_id))


#: Псевдоним выводится ИЗ РЕФЕРЕНТА, а не из порядка обхода. Восемь цифр без ведущего нуля:
#: форма ``ВИД_ЦИФРЫ`` сохранена дословно, потому что она несущая — её знают ``TOKEN_IN_TEXT``,
#: ``_TOKEN_SHAPE``, таблица гомоглифов (модель подставляет кириллическую О внутрь токена) и
#: восемь мест в сейфе и глоссарии. Сменить форму значило бы менять их все разом, а промах дал
#: бы ровно тот дефект, от которого код и защищается: токен, токенизированный второй раз.
_TOKEN_SPAN = 90_000_000
_TOKEN_FLOOR = 10_000_000


def _derive_token(salt: bytes, kind: str, surface: str, taken: set[str]) -> str:
    """``HMAC(соль, вид ‖ нормализованное значение)`` → ``PERSON_48170392``.

    ЗАЧЕМ НЕ СЧЁТЧИК. У счётчика два свойства, обоих не должно быть у псевдонима.

    Он **зависит от порядка**: пересоберите сейф иначе — и ``PERSON_28`` окажется другим
    человеком. Прежняя докстрока этой функции прямо говорила, что перевыдача «молча
    перенаправила бы 85 исторических пересечений», и потому отставленные токены навсегда
    числились занятыми. Здесь перенаправление невозможно ПО ПОСТРОЕНИЮ: токен — функция от
    референта, и другого он обозначать не может.

    И он **ничего не говорит о референте**: два сейфа из одних и тех же данных не сойдутся ни в
    одном токене, и сверить их нечем.

    ПОЧЕМУ С СОЛЬЮ. Токены пересекают границу в облако. Голый ``sha256(имя)`` превратил бы токен
    в проверялку догадок — имея список кандидатов, посторонний подтвердил бы, кто из них в области.
    Это хуже счётчика, который не говорит ничего. Соль случайна и живёт на доверенной стороне
    рядом с подлинниками, так что нового места утечки не создаёт.

    Столкновение разрешается ДЕТЕРМИНИРОВАННО — доминой в сообщении, а не подбором следующего
    свободного: иначе зависимость от порядка вернулась бы через заднюю дверь.
    """
    base = f"{kind}\x00{_normalize(surface)}".encode()
    for attempt in range(64):
        msg = base if attempt == 0 else base + b"\x00" + str(attempt).encode()
        digest = hmac.new(salt, msg, hashlib.sha256).digest()
        n = int.from_bytes(digest[:5], "big") % _TOKEN_SPAN + _TOKEN_FLOOR
        tok = f"{kind}_{n}"
        if tok not in taken:
            return tok
    # 64 подряд занятых номера при пространстве в 90 миллионов — это не столкновение, а поломка
    # соли (например, нулевые байты). Падать здесь честнее, чем выдать чужой токен.
    raise RuntimeError(f"vault: could not derive a free token for {kind}")


def _mint_token(idx: _Index, kind: str, surface: str | None = None) -> str:
    """Псевдоним для нового референта.

    Без соли — старое поведение со счётчиком: сейф, собранный до появления соли, продолжает
    работать, и уже выданные ``PERSON_1`` никуда не переезжают. Перевыдавать существующие
    токены в новой форме НЕЛЬЗЯ: это и есть то самое перенаправление истории.
    """
    if idx.salt and surface:
        tok = _derive_token(idx.salt, kind, surface, idx.tokens_in_use)
    else:
        tok = f"{kind}_{_next_index(idx.tokens_in_use, kind)}"
    idx.tokens_in_use.add(tok)
    return tok


def _add_alias(store: TokenStore, idx: _Index, scope_id: ScopeId,
               surface: str, token: str, source: str) -> bool:
    """Enrol one surface → token. Refuses to move a key that already resolves elsewhere: a
    surface resolving to two referents is the ambiguity case and is handled by the caller."""
    k = _normalize(surface)
    if len(k) < _MIN_KEY_LEN or nf.TOKEN_IN_TEXT.search(surface):
        return False
    owner = idx.keys.get(k)
    if owner is not None and owner != token and k not in idx.legacy_keys:
        return False
    if owner == token and k not in idx.legacy_keys:
        return True
    stored = k[:500]
    if stored in idx.alias_keys:
        # A row for this key exists (possibly pointing at a token this pass has since retired).
        # Re-point it instead of inserting a duplicate a UNIQUE constraint would reject.
        for a in idx.aliases:
            if a.normalized == stored:
                a.token, a.surface, a.source = token, surface.strip(), source
                break
        idx.keys[k] = token
        idx.legacy_keys.discard(k)
        return True
    row = store.add_alias(scope_id, token=token, surface=surface.strip(),
                          normalized=stored, source=source)
    idx.aliases.append(row)
    idx.alias_keys.add(stored)
    idx.keys[k] = token
    idx.legacy_keys.discard(k)
    return True


def _new_token(store: TokenStore, idx: _Index, scope_id: ScopeId,
               surface: str, kind: str, *, attributes: dict | None = None,
               source: str = "entity") -> str | None:
    # No row's real value may CONTAIN one of our own tokens. Measured: the residual pass
    # enrolled "PERSON_1@example.org" as an EMAIL, so the vault held a row whose value is a live
    # token and the full-vault re-hydration path would hand the reader "PERSON_1@example.org".
    # The invariant is enforced at the WRITE, not in one detector, and it refuses rather than
    # raises: a crossing must not 500 because one detection was malformed.
    if nf.TOKEN_IN_TEXT.search(surface):
        log.warning("vault: refusing a token-bearing value for a new %s row", kind)
        return None
    tok = _mint_token(idx, kind, surface)
    # A token row's ``normalized`` is the OLD dedup key (a casefold) and carries a UNIQUE
    # constraint in a relational store. The match index lives in the alias rows, so this field
    # no longer decides anything — but a new row whose fold happens to equal a legacy row's key
    # would still be refused by the constraint, so it is disambiguated by token. Nothing reads it.
    norm = _normalize(surface)[:500]
    if norm in idx.norms:
        norm = f"{norm[:500 - len(tok) - 1]}#{tok}"
    idx.norms.add(norm)
    row = store.add_token(scope_id, token=tok, kind=kind, real_value=surface.strip(),
                          normalized=norm, status="active",
                          attributes=attributes or None)
    idx.rows.append(row)
    idx.by_token[tok] = row
    idx.invalidate()
    _add_alias(store, idx, scope_id, surface, tok, source)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Identity resolution
# ─────────────────────────────────────────────────────────────────────────────

def _token_index(token: str) -> int:
    try:
        return int(token.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _referent_score(canon: tuple[str, ...], cand: tuple[str, ...]) -> int:
    """How many parts of ``cand`` line up with distinct parts of ``canon`` (skeletons)."""
    used: set[int] = set()
    score = 0
    # Full parts are matched BEFORE initials. An initial is a one-character wildcard: let it
    # go first and an initial consumes the slot of a surname that starts with the same letter,
    # after which the surname has nothing left to match and one man reads as two.
    for c in sorted(cand, key=lambda x: len(x) <= 1):
        for i, k in enumerate(canon):
            if i in used:
                continue
            hit = k.startswith(c) if len(c) <= 1 else nf.stem_match(k, c)
            if hit:
                used.add(i)
                score += 1
                break
    return score


def _same_referent(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Two name skeletons denote one person: two parts line up, in either direction."""
    return max(_referent_score(a, b), _referent_score(b, a)) >= 2


def _resolve_person(idx: _Index, surface: str) -> tuple[str | None, bool]:
    """(token, ambiguous) for a person surface not found verbatim in the index.

    Two parts of a name must line up for a merge — "Nils Brandvold" does NOT become
    "Nils Ostrander" — and an initial counts as a part only when it disambiguates
    ("H. Zielinski" is Henry, not Marek). A single part that fits two people is
    ambiguous and the caller mints a shared token rather than guessing.
    """
    cand = nf.skeleton(surface)
    if not cand:
        return None, False
    hits: list[str] = []
    for r in idx.people():
        if _referent_score(nf.skeleton(r.real_value), cand) >= 2:
            hits.append(r.token)
    if len(hits) == 1:
        return hits[0], False
    if len(hits) > 1:
        return None, True
    return None, False


def _fragment_owners(idx: _Index, fragment: str) -> list[str]:
    """Active PERSON tokens whose name contains ``fragment`` as a part."""
    f = nf.skeleton_part(fragment)
    if not f:
        return []
    out = []
    for r in idx.people():
        if any(nf.stem_match(p, f) for p in nf.skeleton(r.real_value)):
            out.append(r.token)
    return sorted(set(out), key=_token_index)


def _propose_variant(pending: dict, surface: str, token: str) -> None:
    slot = pending.setdefault(_normalize(surface), {"surface": surface, "tokens": set()})
    slot["tokens"].add(token)


def _flush_variants(store: TokenStore, idx: _Index, scope_id: ScopeId,
                    pending: dict) -> None:
    """Enrol generated surfaces, but only the UNAMBIGUOUS ones.

    Dropping the legal form off "Red River Trading Ltd" and off "Red River Trading B.V."
    produces the same string for two different legal persons. Giving it to whichever was seeded
    first is the silent-misattribution failure this whole change exists to remove.

    ABSTAINING IS NOT THE ANSWER EITHER, and measured documents are what say so: leaving the
    bare form unenrolled left "Red River Trading" in clear five times, on lines that also
    carry ORG_9's and PERSON_16's tokens. The design already has the right remedy and had only
    ever applied it to PEOPLE — a surface claimed by two referents gets a token of its OWN,
    glossed as shared, so the payload says "the name is shared by ORG_6 and ORG_23; the source
    text does not say which is meant" instead of either naming the company or guessing which
    one. A surname shared by two brothers and a name shared by two legal persons are the same
    problem.
    """
    for k, slot in sorted(pending.items()):
        tokens = slot["tokens"]
        if len(tokens) == 1:
            _add_alias(store, idx, scope_id, slot["surface"], next(iter(tokens)), "variant")
            continue
        owner = idx.keys.get(k)
        if owner is None:
            kinds = {idx.by_token[t].kind for t in tokens if t in idx.by_token}
            if len(kinds) == 1 and kinds <= {"ORG", "VESSEL", "ACCOUNT"}:
                _new_token(store, idx, scope_id, slot["surface"], next(iter(kinds)),
                           attributes={"fragment": "name",
                                       "shared_by": sorted(tokens, key=_token_index)},
                           source="variant")
            continue
        for a in idx.aliases:
            if a.normalized == k and a.source == "variant":
                # Neutralise in place (no delete): the key can never match again, the row
                # remains visible as the record of a retracted guess, and it is reversible.
                a.source, a.normalized = "retracted", f"~retracted:{k}"[:500]
                idx.alias_keys.discard(k)
                idx.alias_keys.add(a.normalized)
                idx.keys.pop(k, None)


def _register_surface(store: TokenStore, idx: _Index, scope_id: ScopeId,
                      surface: str, kind: str, *, attributes: dict | None = None,
                      source: str = "entity", variants: bool = True,
                      pending: dict | None = None) -> str | None:
    """Get-or-create the token for one canonical surface, and enrol its variant surfaces."""
    surface = (surface or "").strip()
    if _is_generic(surface) or len(_normalize(surface)) < _MIN_KEY_LEN:
        return None
    tok = idx.token_for(surface)
    if tok is None:
        tok = _new_token(store, idx, scope_id, surface, kind,
                         attributes=attributes, source=source)
    else:
        row = idx.by_token.get(tok)
        if row is not None and attributes:
            merged = dict(row.attributes or {})
            for k, v in attributes.items():
                merged.setdefault(k, v)
            # A company and a vessel of the same name collapse onto one key (a real scope has
            # both "Marlix" and "MARLIX"). Keep one token; record the second kind so the
            # glossary can say so rather than the payload implying there is only a company.
            if row.kind != kind:
                merged.setdefault("also_kind", kind)
            row.attributes = merged
    if variants and tok:
        if kind == "PERSON":
            gen = nf.person_variants(surface)
        else:
            gen = nf.org_variants(surface)
            # The company's own COINED word. Three companies in one real document set are
            # referred to by their leading word alone, and the OLD vault covered all three (as
            # junk single-word PERSON rows). Dropping those rows without
            # re-enrolling the surface traded a precision win for a recall LEAK, which is the
            # wrong trade. ``distinctive_head`` refuses "Silver" and "Red", so this does not
            # reopen the class the single-word rule was written to close.
            head = nf.distinctive_head(surface)
            if head:
                gen = gen + [head]
                cyr = nf.cyrillic_form(head) or nf.acronym_cyrillic(head)
                if cyr:
                    gen = gen + [cyr]
        for v in gen:
            if pending is None:
                _add_alias(store, idx, scope_id, v, tok, "variant")
            else:
                _propose_variant(pending, v, tok)
    return tok


def _register_fragments(store: TokenStore, idx: _Index, scope_id: ScopeId,
                        people: list[tuple[str, str]]) -> None:
    """Enrol the name parts of known individuals.

    A part belonging to exactly one person becomes an alias of that person. A part shared by
    two ("Zielinski", "Nils") becomes a token of its OWN, flagged ``shared_by`` — the payload
    then says "the surname shared by PERSON_1 and PERSON_7", which is what the source document
    actually supports, instead of silently attributing the sentence to one brother.
    """
    proposals: dict[str, dict] = {}
    for surface, token in people:
        for part, role in nf.person_fragments(surface):
            k = _normalize(part)
            if len(k) < 4:
                continue
            slot = proposals.setdefault(
                k, {"surface": part, "role": role, "tokens": set(),
                    "skel": nf.skeleton_part(part)})
            slot["tokens"].add(token)
    # All spellings of ONE shared fragment share ONE token: a surname in Latin, its Cyrillic
    # spelling and another romanisation of it are the same shared surname, and minting a token
    # per spelling would recreate, inside the fix, the duplication the fix exists to remove.
    shared_tokens: dict[tuple, str] = {}

    def _also_inflections(surface: str, token: str) -> None:
        """Enrol the declensions of a fragment as aliases of the token it was just given.

        The decision "whose name is this?" is made once, on the nominative. The declensions
        follow it — they are the same fragment in another case, not another candidate — which
        is what keeps a genitive out of the shared-token adjudication while still closing the
        leak that put the genitive of one person's surname in clear text.
        """
        for infl in nf.russian_inflections(surface):
            _add_alias(store, idx, scope_id, infl, token, "fragment")
        cyr = nf.cyrillic_form(surface)
        if cyr:
            for infl in nf.russian_inflections(cyr):
                _add_alias(store, idx, scope_id, infl, token, "fragment")

    for k, slot in sorted(proposals.items()):
        # Ownership is computed over EVERY active person in the vault, not only the ones this
        # seeding pass saw: "Nils" belongs to Nils Ostrander *and* to the Nils the residual pass
        # found, and a token that silently means the first is a false statement about the second.
        tokens = sorted(set(slot["tokens"]) | set(_fragment_owners(idx, slot["surface"])))
        owner_now = idx.keys.get(k)
        existing = idx.by_token.get(owner_now) if owner_now else None
        if len(tokens) == 1:
            _add_alias(store, idx, scope_id, slot["surface"], tokens[0], "fragment")
            _also_inflections(slot["surface"], tokens[0])
        elif (grouped := shared_tokens.get((slot["skel"], tuple(tokens)))) is not None:
            _add_alias(store, idx, scope_id, slot["surface"], grouped, "fragment")
            _also_inflections(slot["surface"], grouped)
        elif (existing is not None and existing.kind == "PERSON"
              and _normalize(existing.real_value) == k
              and not (existing.attributes or {}).get("entity_type")):
            # A legacy row already holds this bare fragment as if it were a person of its own
            # ("PERSON_3 = Zielinski"). Adopt it as the shared-surname token rather than mint a
            # second one: the token is already in 85 historical crossings.
            existing.attributes = dict(existing.attributes or {},
                                       fragment=slot["role"],
                                       shared_by=[t for t in tokens if t != existing.token])
            idx.invalidate()
            shared_tokens[(slot["skel"], tuple(tokens))] = existing.token
            _also_inflections(slot["surface"], existing.token)
        elif owner_now is None or k in idx.legacy_keys:
            minted = _new_token(
                store, idx, scope_id, slot["surface"], "PERSON",
                attributes={"fragment": slot["role"], "shared_by": tokens}, source="fragment")
            if minted:
                shared_tokens[(slot["skel"], tuple(tokens))] = minted
                _also_inflections(slot["surface"], minted)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding from the scope's graph
# ─────────────────────────────────────────────────────────────────────────────

def _entity_attributes(e) -> dict:
    """Non-identifying, glossary-bound facts about an entity. See deidkit/glossary.py for
    what is deliberately NOT here (identifiers, addresses, valuations, exact percentages)."""
    attrs: dict = {"entity_type": e.type}
    if e.jurisdiction_country:
        attrs["jurisdiction"] = e.jurisdiction_country
    roles = [r.strip() for r in (e.role or "").split(",") if r.strip()]
    if roles:
        attrs["roles"] = roles[:4]
    rels = []
    for rel in (e.relationships or [])[:6]:
        if not isinstance(rel, dict) or not rel.get("target"):
            continue
        rels.append({"type": str(rel.get("type") or "linked"),
                     "target_name": str(rel["target"]),
                     "pct": rel.get("shareholding_pct")})
    if rels:
        attrs["relations"] = rels
    if e.type == "vessel":
        attrs["vessel_type"] = "vessel"
    return attrs


async def seed_from_scope(store: TokenStore, scope_id: ScopeId,
                          seeds: SeedSource | None) -> None:
    """Pre-populate the vault from the scope's known entities + parties (``seeds``).

    ``seeds=None`` is a scope with nothing known in advance: nothing is seeded."""
    idx = await _index_of(store, scope_id)
    before = len(idx.rows) + len(idx.aliases)
    people: list[tuple[str, str]] = []
    pending: dict = {}

    ents = await seeds.entities(scope_id) if seeds is not None else []
    for e in ents:
        kind = _ENTITY_KIND.get(e.type, "ENTITY")
        name = (e.name or "").strip()
        if name and kind == "PROPERTY" and not (e.identifier or "").strip():
            # "Suburban house in Dayton, Ohio" is a description, not a designator: it
            # will never match verbatim, and if it did it would swallow the jurisdiction the
            # design deliberately preserves. Real property is keyed on a title number or
            # address identifier, or not at all.
            log.info("vault: property entity %s has no identifier; not enrolled", e.id)
            name = ""
        if name:
            tok = _register_surface(store, idx, scope_id, name, kind,
                                    attributes=_entity_attributes(e), source="entity",
                                    pending=pending)
            if tok and kind == "PERSON":
                people.append((name, tok))
        if (e.identifier or "").strip():
            _register_surface(
                store, idx, scope_id, e.identifier,
                "ACCOUNT" if kind == "ACCOUNT" else "ID",
                attributes={"identifier_of_name": name,
                            "identifier_type": e.identifier_type or ""},
                source="entity", variants=False,
            )
        # Identifiers that live in ``attributes`` rather than in ``identifier``. Measured leak:
        # a vessel's IMO crossed as a token while its MMSI and call sign — each unique to the
        # same hull — crossed in clear on the same line, which re-identifies the token.
        for field in ("imo", "mmsi", "callsign", "iban", "registration_number"):
            v = str((e.attributes or {}).get(field) or "").strip()
            if len(v) >= 4:
                _register_surface(
                    store, idx, scope_id, v, "ID",
                    attributes={"identifier_of_name": name, "identifier_type": field},
                    source="entity", variants=False,
                )

    parties = await seeds.parties(scope_id) if seeds is not None else None
    for p in parties or []:
        if not isinstance(p, dict):
            continue
        name = (p.get("name") or "").strip()
        if not name or _is_generic(name):
            continue
        kind, individual = _party_kind(str(p.get("role") or ""), name)
        tok = idx.token_for(name)
        if tok is None and individual:
            # "Jonas Peter Maier" (parties) and "Jonas Peter Mayer" (entities) are one man
            # spelled two ways by two sources; the old seeder made two tokens and neither
            # source ever reconciled.
            resolved, ambiguous = _resolve_person(idx, name)
            if resolved is not None:
                _add_alias(store, idx, scope_id, name, resolved, "intake")
                for v in nf.person_variants(name):
                    _propose_variant(pending, v, resolved)
                tok = resolved
            elif ambiguous:
                tok = _new_token(store, idx, scope_id, name, "PERSON",
                                 attributes={"source": "intake"}, source="intake")
        if tok is None and not individual:
            tok = _merge_org(store, idx, scope_id, name, pending)
        if tok is None:
            tok = _register_surface(store, idx, scope_id, name, kind,
                                    attributes={"roles": [str(p.get("role") or "")][:1]},
                                    source="intake", variants=individual or kind == "ORG",
                                    pending=pending)
        if tok and individual:
            people.append((name, tok))

    _flush_variants(store, idx, scope_id, pending)
    _register_fragments(store, idx, scope_id, people)
    if len(idx.rows) + len(idx.aliases) != before:
        await store.commit()


def _merge_org(store: TokenStore, idx: _Index, scope_id: ScopeId,
               name: str, pending: dict | None = None) -> str | None:
    """Resolve a short organisation surface onto an existing organisation, or return None.

    "Tarvelo" (a party) is "Tarvelo Denizcilik Ticaret Anonim Sirketi" (an entity); "NRTK" is
    "NRTK Asia M6 Limited"; "Quorvane Shipping" is "Quorvane Shipping Ltd". The merge is
    deliberately narrow:

      * only when the SHORT name carries no legal form of its own — "Red River Trading Ltd"
        and "Red River Trading B.V." are two legal persons and must never merge;
      * only onto the SAME kind — "Silver Rock Corporation, Ltd" (owner) and the vessel
        "SILVER ROCK" are different things that share a name;
      * only when exactly one candidate matches;
      * never between two rows that came from entities. Whoever resolved the entities has
        already made that call and this module is not the place to overrule it (a pair like
        "S.B. Management" / "S.B.Management Nordic" is exactly such a case and is left alone).
    """
    parts = [nf.skeleton_part(p) for p in nf.strip_legal_form(name)]
    parts = [p for p in parts if p]
    if not parts or nf.has_legal_form(name):
        return None
    hits = []
    for r in idx.active():
        if r.kind not in ("ORG", "VESSEL", "ACCOUNT") or r.kind == "PERSON":
            continue
        rparts = [nf.skeleton_part(p) for p in nf.strip_legal_form(r.real_value)]
        rparts = [p for p in rparts if p]
        if len(rparts) > len(parts) and rparts[:len(parts)] == parts or rparts == parts and _normalize(r.real_value) != _normalize(name):
            hits.append(r.token)
    hits = sorted(set(hits))
    if len(hits) != 1:
        return None
    _add_alias(store, idx, scope_id, name, hits[0], "intake")
    for v in nf.org_variants(name):
        if pending is None:
            _add_alias(store, idx, scope_id, v, hits[0], "variant")
        else:
            _propose_variant(pending, v, hits[0])
    return hits[0]


# ─────────────────────────────────────────────────────────────────────────────
# Residual PII
# ─────────────────────────────────────────────────────────────────────────────

def _detect(detector: PiiDetector | None, text: str, language: str) -> list[tuple[str, str]]:
    """The residual pass's ``(span_text, kind)`` detections (see :mod:`deidkit.detect`).

    Without a detector the residual pass is skipped, exactly as when Presidio is unavailable:
    the known-entity pass already ran, and the gateway is the hard wall."""
    if detector is None:
        log.warning("vault: Presidio unavailable, residual PII pass skipped (%s)",
                    "no detector configured")
        return []
    return list(detector.detect(text, language))


# Mailbox local parts that are a function, not a person.
_ROLE_MAILBOXES = {
    "info", "office", "admin", "sales", "support", "contact", "mail", "accounts",
    "accounting", "finance", "hr", "legal", "noreply", "no-reply", "service", "help",
    "team", "billing", "invoice", "reception", "secretariat", "post", "general",
}


def _person_from_email(address: str) -> str | None:
    """"jose.munoz@corvida.example" → "Jose Munoz".

    One scope had NO individual among its entities, and the English NER does not label names in
    a Spanish email — so the accountant's name crossed in full clear text on the line above his
    own (tokenised) address. The mailbox is the only structured evidence of his name the system
    holds, and it is evidence: ``first.last@`` is a naming convention, not a guess about who he
    is. Enrolling the Latin spelling is what lets the diacritic fold catch "José Muñoz" two
    lines below.
    """
    local = address.split("@", 1)[0]
    parts = [p for p in re.split(r"[._-]+", local) if p]
    if len(parts) != 2 or any(not p.isalpha() or len(p) < 3 for p in parts):
        return None
    if any(p.casefold() in _ROLE_MAILBOXES for p in parts):
        return None
    return " ".join(p.capitalize() for p in parts)


def _trim_frame_words(surface: str) -> str:
    """Strip the sign-off/framing word a NER span dragged in with the name.

    "BR Oskar" is a man's given name with a sign-off ("BR", best regards) glued to the front,
    and the span is what a residual detection offers. Keeping the whole span as the row's VALUE
    means the answer comes back naming him "BR Oskar"; dropping the frame word keeps the
    coverage — the span itself is still enrolled as an alias — and makes the name right.
    """
    parts = nf.name_parts(surface)
    while len(parts) > 1 and nf.is_common_word(parts[0]) and len(parts[0]) <= 4:
        parts = parts[1:]
    while len(parts) > 1 and nf.is_common_word(parts[-1]) and len(parts[-1]) <= 4:
        parts = parts[:-1]
    trimmed = " ".join(parts)
    return trimmed if len(trimmed) >= 4 else surface


def _admit(surface: str, kind: str, idx: "_Index | None" = None,
           language: str | None = None, vetoed: set[str] | None = None) -> tuple[bool, str]:
    """Should a residual detection be enrolled at all? (see namefold.admit_person).

    ``idx`` and ``language`` are the two things ``namefold`` cannot see. With them this gate
    also refuses the classes the structural predicate alone let through, every one measured on
    real documents:

      * a span that CONTAINS an already-enrolled vessel or organisation — "SEA RUNNER\nContainer
        Ship" is VESSEL_8, and because the residual row's key is longer, ``_apply_index``'s
        longest-first rule made the hull cross as a natural person while nine other VESSEL_n
        tokens sat correctly in the same payload;
      * a span whose script is not the one the NER was RUNNING in — every Greek row in one real
        vault came from the English model reading a Greek-language document;
      * a span Presidio itself also labelled a LOCATION ("Monte Alto", "Marlow House"). The
        design preserves locations deliberately; a location that acquires a PERSON token is the
        design being inverted, and Presidio's own verdict is the cheapest evidence available.
    """
    if kind in _STRUCTURAL_KINDS:
        # SEARCH, not match: an anchored test only ever caught a surface that IS a token, so
        # "PERSON_1@example.org" was admitted and stored as an EMAIL row.
        if not surface.strip():
            return False, "empty"
        if nf.TOKEN_IN_TEXT.search(surface):
            return False, "contains a token"
        return True, "structural"
    if kind == "PROPERTY":
        # A property token was only ever minted from a prose description ("Suburban house in
        # Dayton, Ohio"): it never matches verbatim, and if it did it would swallow a
        # jurisdiction. Real property is keyed on a title number, which is an ID.
        return False, "prose description, not a designator"
    if kind != "PERSON":
        if not surface.strip():
            return False, "empty"
        return (False, "contains a token") if nf.TOKEN_IN_TEXT.search(surface) else (True, "non-person")
    ok, reason = nf.admit_person(surface)
    if not ok:
        return ok, reason
    if vetoed and _normalize(surface) in vetoed:
        return False, "Presidio also labelled this span a location"
    if language:
        want = {"ru": "cyrillic", "en": "latin"}.get(language)
        got = nf.script_of(surface)
        if want and got and got != want:
            return False, f"span is {got}, NER language is {language}"
    if idx is not None:
        owner = _covering_referent(idx, surface)
        if owner is not None:
            return False, f"span contains the enrolled surface of {owner}"
    return True, "ok"


def _covering_referent(idx: "_Index", surface: str) -> str | None:
    """An enrolled ORG/VESSEL/ACCOUNT that this span is part of, in either direction.

    Inside-out: "SEA RUNNER\nContainer Ship" CONTAINS VESSEL_8's surface, so it is that hull,
    not a new natural person — and because the residual key is longer, longest-first matching
    made the ship cross as a person while nine correct VESSEL_n tokens sat in the same payload.

    Outside-in: "Monte Alto" is CONTAINED IN the organisation "CORVIDA S.L. Monte Alto" — it is
    the town in that company's registered name. A span that is part of a company's own name is
    not independently a person, and this is the check that keeps a town out of the PERSON
    namespace without a gazetteer. It is only ever consulted for a span with no corroboration
    from the scope's graph, so a real party whose name the graph knows is never refused by it.
    """
    hay, _ = nf.fold_haystack(surface)
    if len(hay) < 4:
        return None
    for k, tok in idx.keys.items():
        row = idx.by_token.get(tok)
        if row is None or row.kind not in ("ORG", "VESSEL", "ACCOUNT") or k == hay:
            continue
        if len(k) >= 5 and nf.find_all(hay, k):
            return tok
        if len(k) > len(hay) and nf.find_all(k, hay):
            return tok
    return None


async def _absorb(store: TokenStore, scope_id: ScopeId,
                  detected: list[tuple[str, str]], *, language: str | None = None) -> None:
    """Resolve each residual detection onto an existing referent, or enrol a new one."""
    idx = await _index_of(store, scope_id)
    before = len(idx.rows) + len(idx.aliases)
    pending: dict = {}
    vetoed = {_normalize(v) for v, k in detected if k == _VETO_KIND}
    # People this pass discovers get the same fragment treatment as people from the graph.
    # Without it a residual-only scope — one real scope had no individual entities and no
    # parties — has no bare given name and no bare surname enrolled at all, so the salutation
    # "José, saludos" crossed in clear one line above his own tokenised address and his own
    # token, which publishes the mapping.
    found_people: list[tuple[str, str]] = []
    for surface, kind in sorted(detected, key=lambda x: len(x[0]), reverse=True):
        surface = surface.strip()
        if kind == _VETO_KIND:
            continue
        ok, reason = _admit(surface, kind, idx, language, vetoed)
        if not ok:
            log.debug("vault: residual %r (%s) not admitted: %s", surface, kind, reason)
            continue
        if idx.token_for(surface) is not None:
            continue
        if kind == "PERSON":
            resolved, ambiguous = _resolve_person(idx, surface)
            if resolved is not None:
                _add_alias(store, idx, scope_id, surface, resolved, "residual")
                found_people.append((surface, resolved))
                continue
            if ambiguous:
                _new_token(store, idx, scope_id, surface, "PERSON",
                           attributes={"shared_by": _fragment_owners(idx, surface)},
                           source="residual")
                continue
            owners = _fragment_owners(idx, surface) if len(nf.name_parts(surface)) == 1 else []
            if len(owners) == 1:
                _add_alias(store, idx, scope_id, surface, owners[0], "residual")
                continue
            if len(owners) > 1:
                _new_token(store, idx, scope_id, surface, "PERSON",
                           attributes={"shared_by": owners, "fragment": "name part"},
                           source="residual")
                continue
        value = _trim_frame_words(surface) if kind == "PERSON" else surface
        tok = _register_surface(store, idx, scope_id, value, kind,
                                attributes={"source": "residual"}, source="residual",
                                variants=(kind in ("PERSON", "ORG", "VESSEL")),
                                pending=pending)
        if tok and value != surface:
            _add_alias(store, idx, scope_id, surface, tok, "residual")
        if kind == "PERSON" and tok:
            found_people.append((surface, tok))
        if kind == "EMAIL" and tok:
            derived = _person_from_email(surface)
            if derived and _admit(derived, "PERSON")[0] and idx.token_for(derived) is None:
                resolved, ambiguous = _resolve_person(idx, derived)
                if resolved is not None:
                    _add_alias(store, idx, scope_id, derived, resolved, "residual")
                    found_people.append((derived, resolved))
                elif not ambiguous:
                    dtok = _register_surface(store, idx, scope_id, derived, "PERSON",
                                             attributes={"source": "mailbox"}, source="residual",
                                             pending=pending)
                    if dtok:
                        found_people.append((derived, dtok))
    _flush_variants(store, idx, scope_id, pending)
    if found_people:
        _register_fragments(store, idx, scope_id, found_people)
    if len(idx.rows) + len(idx.aliases) != before:
        await store.commit()



# ─────────────────────────────────────────────────────────────────────────────
# Recall passes over the OUTBOUND text
# ─────────────────────────────────────────────────────────────────────────────

# A nine-digit MMSI/IMO-shaped literal, and an ITU call sign (letters and digits, no lower
# case). Both are only ever looked for on a line that ALREADY carries a tokenised identifier of
# the same referent, so this is a recall pass over the payload rather than a new recogniser
# turned loose on every document.
# An IMO (7 digits) or MMSI (9 digits), and an ITU call sign — which has BOTH letters and
# digits, because that is what distinguishes "ZX9Q4" and "4KQD2" from the all-caps headings
# ("DEBTS", "ASSETS", "CYPRUS") of a Belgian annual-accounts filing. Without the digit
# requirement this pass tokenised the word Cyprus, which is a JURISDICTION the design
# deliberately preserves, in a document whose company number is legitimately enrolled.
# Exactly 7 (IMO) or 9 (MMSI) digits, standing alone. The boundary classes are plain
# alphanumerics: a comma legitimately FOLLOWS an identifier in a list — "(211000001,
# 351000002)" is two MMSIs — and excluding it silently dropped the first of every such pair,
# which is how a primary MMSI stayed in clear next to its own hull's token.
_ID_LITERAL = re.compile(r"(?<![0-9A-Za-z])(\d{9}|\d{7})(?![0-9A-Za-z])")
_CALLSIGN = re.compile(r"(?<![0-9A-Za-z])(?=[A-Z0-9]{4,7}(?![0-9A-Za-z]))"
                       r"(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])([A-Z0-9]{4,7})")
# ...and the RECORD has to say it is talking about identifiers. A seven-digit number in a
# balance sheet is a sum of money; the same seven digits under the word "IMO" are a hull.
_ID_CONTEXT = re.compile(
    r"\b(imo|mmsi|call\s*sign|callsign|позывн|ммси|номер\s+имо|flag|флаг)\b", re.IGNORECASE)
_DOMAIN = re.compile(r"(?<![0-9A-Za-z@.])((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(?![0-9A-Za-z])")


def _is_identifier_shaped(digits: str) -> bool:
    """Does this number carry the internal structure of an IMO or an MMSI?

    An IMO number has a check digit — sum of the first six digits weighted 7…2, modulo 10 — and
    an MMSI begins with a Maritime Identification Digit in 2–7. Both are free, deterministic
    tests, and they are what separates a hull from a figure in a balance sheet that happens to
    sit near a tokenised company number.
    """
    if len(digits) == 7:
        return sum(int(d) * w for d, w in zip(digits[:6], range(7, 1, -1))) % 10 == int(digits[6])
    return len(digits) == 9 and digits[0] in "234567"


def _identifier_owner(idx: _Index, token: str) -> str | None:
    row = idx.by_token.get(token)
    return str((row.attributes or {}).get("identifier_of_name") or "") or None if row else None


def _identifier_segments(text: str) -> list[str]:
    """The text cut into RECORD-sized pieces for the identifier recall pass.

    A "line" is the wrong unit twice over in real documents. One document is a markdown table
    whose every row is a single 2000-character line carrying four different hulls, so a
    line-scoped anchor is ambiguous and the pass refuses to act; another writes each field of
    one vessel on its own line, so the anchor and the alternate identifier are three lines
    apart. Segments are therefore table cells, and a sliding window of three consecutive lines.
    """
    out: list[str] = []
    lines = text.splitlines()
    for i in range(len(lines)):
        window = "\n".join(lines[i:i + 3])
        if len(window) <= 400:
            out.append(window)
        else:
            out.extend(c for c in window.split("|") if len(c) <= 400)
    return out


def _recall_identifiers(store: TokenStore, idx: _Index, scope_id: ScopeId,
                        text: str) -> bool:
    """Enrol identifier literals that sit beside an already-tokenised identifier of one hull.

    MEASURED: "VESSEL_2 IMO: ID_5 MMSI: ID_13 (351000469) Flag: … Call sign: ID_14" — the
    primary MMSI is a token and the alternate, in parentheses beside it, is in clear; the same
    shape leaked six MMSIs and one call sign across one document set, and one of them was a
    PRIMARY, not an alternate. The glossary's own exclusion list says flag, year built and all
    identifiers are withheld; the text beside the tokens supplied all three.

    This is a recall pass over the payload, not a new recogniser: a literal is enrolled only
    when the segment it sits in already carries a tokenised identifier of EXACTLY ONE referent,
    so a number with no anchor, or with two candidate hulls, is left alone.
    """
    id_keys = [(k, tok) for k, tok in idx.keys.items()
               if len(k) >= 4 and (r := idx.by_token.get(tok)) is not None and r.kind == "ID"]
    if not id_keys:
        return False
    grew = False
    for seg in _identifier_segments(text):
        if not any(ch.isdigit() for ch in seg):
            continue
        hay, _ = nf.fold_haystack(seg)
        owners = set()
        for k, tok in id_keys:
            if k in hay and nf.find_all(hay, k):
                owner = _identifier_owner(idx, tok)
                if owner:
                    owners.add(owner)
        if len(owners) != 1 or not _ID_CONTEXT.search(seg):
            continue          # no anchor, two hulls in one record, or not an identifier record
        owner = next(iter(owners))
        cands = [m.group(1) for m in _ID_LITERAL.finditer(seg)
                 if _is_identifier_shaped(m.group(1))]
        cands += [m.group(1) for m in _CALLSIGN.finditer(seg)
                  if not nf.is_common_word(m.group(1))]
        for c in cands:
            if idx.token_for(c) is not None or len(c) < 4:
                continue
            tok = _register_surface(
                store, idx, scope_id, c, "ID",
                attributes={"identifier_of_name": owner, "identifier_type": "secondary"},
                source="residual", variants=False)
            grew = grew or bool(tok)
    return grew


def _recall_domains(store: TokenStore, idx: _Index, scope_id: ScopeId,
                    text: str) -> bool:
    """Enrol a domain whose registrable label IS an enrolled organisation.

    MEASURED: "https://orvalis.example/" sits beside ORG_5 = ORVALIS SHIPPING; "sbasia.example"
    beside the token of SILVER BAY ASIA LIMITED; "www.corvida.example" beside EMAIL_2 =
    jose.munoz@corvida.example and ORG_1 = Corvida S.L. The evidence is already in the vault,
    which is why this is not a general domain recogniser: a domain is enrolled only when its own
    label resolves to an organisation the vault already holds, so public services
    ("outlook.com", "facebook.com") are untouched.
    """
    grew = False
    hosts: dict[str, str] = {}
    for row in idx.active():
        if row.kind == "EMAIL" and "@" in row.real_value:
            hosts[row.real_value.split("@", 1)[1].casefold()] = row.token
    for m in _DOMAIN.finditer(text):
        dom = m.group(1)
        if idx.token_for(dom) is not None:
            continue
        labels = dom.split(".")
        if len(labels) < 2 or len(labels[0]) < 3:
            continue
        label = labels[-2] if labels[0].casefold() in ("www", "mail", "web") else labels[0]
        owner = idx.token_for(label) or idx.token_for(re.sub(r"[^0-9A-Za-z]", "", label))
        if owner is None:
            email_tok = hosts.get(".".join(labels[-2:]).casefold()) or hosts.get(dom.casefold())
            if email_tok is None:
                continue
            owner_row = None
            for r in idx.active():
                if r.kind in ("ORG", "VESSEL") and nf.skeleton_part(label) and \
                        nf.skeleton_part(label) in {nf.skeleton_part(p)
                                                    for p in nf.strip_legal_form(r.real_value)}:
                    owner_row = r
                    break
            if owner_row is None:
                continue
            owner = owner_row.token
        row = idx.by_token.get(owner)
        if row is None or row.kind not in ("ORG", "VESSEL", "ACCOUNT"):
            continue
        if _add_alias(store, idx, scope_id, dom, owner, "residual"):
            grew = True
        if len(label) >= 5 and idx.token_for(label) is None:
            grew = _add_alias(store, idx, scope_id, label, owner, "residual") or grew
    return grew


# ─────────────────────────────────────────────────────────────────────────────
# The crossing
# ─────────────────────────────────────────────────────────────────────────────

def _apply_index(text: str, keys: dict[str, str],
                 idx: "_Index | None" = None) -> tuple[str, dict[str, list[str]]]:
    """Replace every enrolled surface in ``text`` with its token, in ONE pass.

    Matching happens on the folded form (``namefold``) and the replacement is spliced into the
    RAW text through the offset map, so "José Muñoz" and "Sea ﬁnder" are found without the
    output losing a single original byte outside the matched spans. Longest key first, and an
    overlap map, so "QUORVANE SHIPPING LIMITED" is consumed by the company's token instead of
    being half-eaten by a shorter alias — the defect that produced "PERSON_29 LIMITED".

    ADJACENT SPANS OF ONE REFERENT ARE MERGED, and the test is the REFERENT, not the token.
    One man's name, written surname first in Cyrillic, was detected as two spans carrying two
    different tokens, so the equality test ``token == last_token`` could not fire and one man
    crossed as two shareholders on the same line — in a Latin-script chunk of the same scope he
    is a single token, so the split was script- and word-order dependent. When the two tokens denote one referent
    the merge now fires and keeps the MORE SPECIFIC of the two: a shared-fragment token
    ("the surname shared by …") must never swallow the token of the person it is shared with.
    """
    if not text.strip() or not keys:
        return text, {}
    hay, folded = nf.fold_haystack(text)
    taken = bytearray(len(hay))
    hits: list[tuple[int, int, str]] = []
    for k in sorted(keys, key=len, reverse=True):
        for start, end in nf.find_all(hay, k):
            if any(taken[start:end]):
                continue
            taken[start:end] = b"\x01" * (end - start)
            hits.append((start, end, keys[k]))
    if not hits:
        return text, {}
    def _one_referent(a: str, b: str) -> bool:
        return a == b or (idx is not None and idx.same_referent(a, b))

    def _more_specific(a: str, b: str) -> str:
        """Of two tokens for one referent, the one that names a person rather than a fragment."""
        if idx is None:
            return a
        for x, y in ((a, b), (b, a)):
            if (idx.by_token.get(x, None) is not None
                    and (idx.by_token[x].attributes or {}).get("shared_by")
                    and not (idx.by_token.get(y) and
                             (idx.by_token[y].attributes or {}).get("shared_by"))):
                return y
        return a

    hits.sort()
    out: list[str] = []
    found: dict[str, list[str]] = {}
    cursor = 0
    last_token: str | None = None
    for start, end, token in hits:
        rs, re_ = folded.raw_span(start, end)
        if rs < cursor:
            continue
        gap = text[cursor:rs]
        if last_token is not None and not gap.strip() and _one_referent(token, last_token):
            keep = _more_specific(last_token, token)
            surfaces = found.pop(last_token)
            merged = surfaces.pop() + gap + text[rs:re_]
            if surfaces:
                found[last_token] = surfaces
            found.setdefault(keep, []).append(merged)
            if keep != last_token:
                out[-1] = keep
            cursor = re_
            last_token = keep
            continue
        out.append(gap)
        out.append(token)
        found.setdefault(token, []).append(text[rs:re_])
        cursor = re_
        last_token = token
    out.append(text[cursor:])
    return "".join(out), found


async def tokenize(
    store: TokenStore,
    scope_id: ScopeId,
    text: str,
    *,
    seeds: SeedSource | None = None,
    detector: PiiDetector | None = None,
    language: str | None = None,
    seed: bool = True,
    glossary: bool = True,
) -> Redaction:
    """Replace real values in ``text`` with stable per-scope tokens (reversible).

    Returns a :class:`~deidkit.gateway.Redaction` whose ``mapping`` (token→real) lets the
    caller re-hydrate the cloud answer on the trusted plane afterwards, and whose ``glossary``
    carries the out-of-band, non-identifying description of each token that actually crossed.

    ``seeds`` — the scope's known entities and parties, enrolled first (with ``seed=True``);
    ``detector`` — the residual PII detector (see :mod:`deidkit.detect`); without one the
    residual pass is skipped. ``language`` overrides the detected one ('en', 'ru', 'de', …).
    """
    if not text or not text.strip():
        return Redaction(text=text or "")

    # Соль ЧИТАЕТСЯ, а не заводится: tokenize зовут из многих мест, и часть вызовов по
    # смыслу — чтение. Писать в пути, который может быть чтением, — та самая ошибка, что
    # аудит нашёл в других модулях; здесь её поймали тесты сейфа, подсовывающие хранилище-
    # заглушку. Соли выдаёт ensure_salt на пишущем пути; без соли _mint_token
    # работает по-старому, на счётчике.
    salt = await _salt_of(store, scope_id)

    if seed:
        await seed_from_scope(store, scope_id, seeds)

    # Residual PII is detected on the ORIGINAL text and absorbed into the index BEFORE the
    # replacement pass, so a detection that turns out to be a known person becomes an alias of
    # that person's token instead of a new row — and every row that IS minted is guaranteed to
    # be applied. (The old order minted "Henri Zielinski" and then never used it, because a
    # shorter alias had already consumed the surname: one orphan row per crossing.)
    lang = language or _detect_language(text)
    detected = _detect(detector, text, lang)
    if detected:
        await _absorb(store, scope_id, detected, language=lang)

    idx = await _index_of(store, scope_id, salt)
    # Two recall passes over THIS payload, both anchored on evidence already in the vault: an
    # identifier literal beside a tokenised identifier of the same hull, and a domain whose
    # label is an enrolled company. Neither costs anything on text that has no such pair.
    if _recall_identifiers(store, idx, scope_id, text) | \
            _recall_domains(store, idx, scope_id, text):
        await store.commit()
        idx = await _index_of(store, scope_id, salt)

    out, found = _apply_index(text, idx.match_keys(), idx)

    mapping: dict[str, str] = {}
    surfaces_seen: dict[str, list[str]] = {}
    canonical: dict[str, str] = {}
    kinds: Counter[str] = Counter()
    for token, surfaces in found.items():
        row = idx.by_token.get(token)
        if row is None:
            continue
        # Re-hydrate to the surface the DOCUMENT used when it used only one — the answer then
        # reads in the document's own spelling and tokenize→detokenize is the identity. Where
        # one referent appeared under two spellings there is no identity to preserve, so the
        # canonical value wins.
        distinct = {_normalize(s) for s in surfaces}
        mapping[token] = surfaces[0] if len(distinct) == 1 else row.real_value
        surfaces_seen[token] = list(surfaces)
        canonical[token] = row.real_value
        kinds[row.kind] += len(surfaces)

    gloss: list[str] = []
    if glossary and mapping:
        from deidkit.glossary import build_glossary
        gloss = build_glossary(idx, mapping, out)

    return Redaction(
        text=out,
        entity_types=sorted(kinds),
        redacted_count=int(sum(kinds.values())),
        mapping=mapping,
        glossary=gloss,
        surfaces=surfaces_seen,
        canonical=canonical,
    )


def merge_mappings(reds: list[Redaction]) -> dict[str, str]:
    """One re-hydration mapping for a call that tokenised several pieces of text.

    ``mapping[token] = surfaces[0]`` is a per-CALL decision, and merging several per-call
    mappings with ``dict.update`` makes the last one win: measured on one real scope, PERSON_1
    was written one way in a Latin-script chunk and another way in a Cyrillic-script one, and
    so was PERSON_4, so one chat conversation named the same man two ways across turns. The rule the single-text path already applies — keep the document's
    own spelling only when there is exactly ONE of them — is applied here to the UNION.
    """
    surfaces: dict[str, list[str]] = {}
    canonical: dict[str, str] = {}
    fallback: dict[str, str] = {}
    for red in reds:
        for tok, vals in (red.surfaces or {}).items():
            surfaces.setdefault(tok, []).extend(vals)
        canonical.update(red.canonical or {})
        fallback.update(red.mapping or {})
    out: dict[str, str] = dict(fallback)
    for tok, vals in surfaces.items():
        distinct = {_normalize(v) for v in vals}
        out[tok] = vals[0] if len(distinct) == 1 else canonical.get(tok, vals[0])
    return out


def with_glossary(red: Redaction) -> Redaction:
    """The payload as it should CROSS: the token glossary, then the de-identified text.

    Returned as a new :class:`Redaction` so the gateway hashes what actually crossed — an
    audit row whose ``content_hash`` covers only part of the payload is not an audit row. The
    ``mapping`` is carried through untouched, so the return leg is byte-for-byte what it was
    before this feature existed.
    """
    from deidkit.glossary import render
    block = render(red.glossary)
    if not block:
        return red
    return Redaction(text=block + red.text, entity_types=red.entity_types,
                     redacted_count=red.redacted_count, mapping=red.mapping,
                     glossary=red.glossary, surfaces=red.surfaces, canonical=red.canonical)


# A token reference in a cloud answer. Case-insensitive (the model echoes "person_3" and
# "Person_3" as readily as "PERSON_3"); the guards stop ORG_1 matching inside ORG_12 and stop a
# token being recognised inside a longer word.
#
# ``_`` IS NOT A GUARD CHARACTER. It is the token's own separator and markdown's emphasis
# marker, so a model that writes ``_PERSON_1_`` or ``__PERSON_1__`` for emphasis — or
# ``PERSON_1_ORG_2`` for a pair — produced a token the return leg could not see, and the reader
# saw a raw token. Dropping it from both classes does not weaken the two guarantees the guards
# exist for: ``ORG_1`` still does not match inside ``ORG_12`` (the next character is a digit,
# which is alphanumeric) and ``ORG_1x`` is still left alone.
def _token_pattern(tokens: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return re.compile(rf"(?<![0-9A-Za-z])(?:{alts})(?![0-9A-Za-z])", re.IGNORECASE)


async def detokenize(
    store: TokenStore,
    scope_id: ScopeId,
    text: str,
    *,
    mapping: dict[str, str] | None = None,
) -> str:
    """Restore real values from tokens.

    SCOPE. ``mapping`` is the tokens THIS request actually sent, and passing it is the correct
    call: de-tokenisation must only ever reverse what this request tokenised. A token the model
    invents, or copies out of a glossary, then stays visibly a token instead of being handed a
    real name the cloud never received.

    The ``mapping=None`` path exists for the 85 historical crossings whose answers must still
    re-hydrate, and it is deliberately NARROWER than the raw table: every token resolves
    through ``canonical_token`` to the row that carries the referent TODAY. Without that,
    PERSON_37 and PERSON_49 — two retired duplicates of one man — came back as two separately
    named parties, one of them carrying a parenthetical alias list that is itself a
    re-identification artefact.

    CASE. The outbound side has always matched case-insensitively (documents write
    "QUORVANE SHIPPING LIMITED" for a value stored as "Quorvane Shipping Ltd"), while this
    side used ``str.replace`` — exact case. Anything the model echoed in another case came
    back to the reader as a raw token, and it is the model, not us, that chooses the case in
    the answer. The INBOUND side moved: making the outbound side case-SENSITIVE instead would
    have meant missing the name in the document, which is a leak, and the pair must agree.
    Longest token first so ``ORG_12`` is replaced before ``ORG_1``.

    SCRIPT. The outbound leg folds homoglyphs; this leg now does too. A Cyrillic О inside
    "PERSON_1" — a plausible slip for a model answering in Russian — and a fullwidth digit both
    left the token unrecognised. ``deconfuse_ascii`` is 1:1, so the match positions are
    positions in the ORIGINAL answer and the replacement is spliced back into it unchanged.
    """
    if not text:
        return text
    if mapping is None:
        rows = await _load(store, scope_id)
        idx = _Index(rows, [])
        mapping = {}
        for r in rows:
            canon = idx.by_token.get(idx.canonical(r.token) or r.token)
            mapping[r.token] = (canon or r).real_value
    if not mapping:
        return text
    lookup = {t.casefold(): v for t, v in mapping.items()}
    probe = nf.deconfuse_ascii(text)
    pat = _token_pattern(list(mapping))
    out: list[str] = []
    cursor = 0
    for m in pat.finditer(probe):
        out.append(text[cursor:m.start()])
        out.append(lookup[m.group(0).casefold()])
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out)


def _detect_language(text: str) -> str:
    try:
        from deidkit.lang import detect_language
        return detect_language(text) or "en"
    except Exception:  # noqa: BLE001
        return "en"


# ─────────────────────────────────────────────────────────────────────────────
# Migration of an existing vault
# ─────────────────────────────────────────────────────────────────────────────

async def reconcile_scope(store: TokenStore, scope_id: ScopeId,
                          *, seeds: SeedSource | None = None, dry_run: bool = False) -> dict:
    """Bring an EXISTING vault in line with the fixed detector. Reversible by construction.

    Three things happen to the rows that are already there, and nothing is deleted:

      * **kept** — a row the current detector would still admit stays active and gains its
        alias surfaces (variants, fragments).
      * **merged** — a row that denotes a referent another row already holds (nine of ten
        duplicates of one man, "Quorvane Shipping" under a PERSON prefix, a Latin acronym beside
        its Cyrillic spelling) is RETIRED with ``canonical_token`` set, and its surface is
        re-enrolled as an
        alias of the canonical token. Outbound, the text now yields the canonical token;
        inbound, the retired token still resolves to its own original value, so every one of
        the 85 historical crossings re-hydrates exactly as it did before.
      * **retired** — a row the current detector would refuse ("Management", "Argentina", a
        Russian infinitive, the declensions of two Russian role nouns, the two malformed spans)
        is retired with no canonical and no alias: it can never fire outbound again, and still
        detokenises.

    Reversal is a single update (``status='active'``, ``canonical_token=None``) plus dropping
    the alias rows this pass added — no other field of an existing row changes. A snapshot of
    the token rows taken before the run restores them wholesale if that is preferred.

    ``seeds`` is the scope's seed source (see :func:`seed_from_scope`). With ``dry_run`` the
    final commit is skipped; seeding commits on its own as always, and a transactional store's
    caller rolls the rest back — :class:`~deidkit.store.InMemoryTokenStore` has no rollback.
    """
    idx = await _index_of(store, scope_id)
    report = {"kept": [], "merged": [], "retired": []}

    # 1. Rebuild the canonical picture from the scope's graph, in a scratch index that only
    #    contains rows already marked active, so seeding decides who the referents are.
    await seed_from_scope(store, scope_id, seeds)
    idx = await _index_of(store, scope_id)

    graph_tokens = {a.token for a in idx.aliases if a.source in ("entity", "intake")}

    def _retire(row, canonical, bucket, **extra):
        row.status, row.canonical_token = "retired", canonical
        idx.invalidate()
        k = _normalize(row.real_value.strip())
        if idx.keys.get(k) == row.token:
            idx.keys.pop(k, None)
            idx.legacy_keys.discard(k)
        report[bucket].append({"token": row.token, "value": row.real_value.strip(), **extra})

    # Row order is deliberately NOT load-bearing. Processing multi-part rows before single-word
    # ones was tried as the fix for "one man, two active tokens" and measured to change nothing
    # on either real scope: the duplicate survives merge ORDER, not merge order alone. What
    # actually closes it is the ``shared_by`` re-derivation below, which runs after every merge
    # has landed. Ordering by token keeps the report stable and reproducible.
    for row in sorted(idx.rows, key=lambda r: r.token):
        if row.status != "active":
            continue
        surface = row.real_value.strip()
        attrs = row.attributes or {}
        if row.token in graph_tokens or attrs.get("shared_by") or attrs.get("entity_type"):
            report["kept"].append(row.token)
            _add_alias(store, idx, scope_id, surface, row.token, "legacy")
            continue
        if row.kind in _STRUCTURAL_KINDS or row.kind in ("ID", "ACCOUNT", "VESSEL"):
            report["kept"].append(row.token)
            _add_alias(store, idx, scope_id, surface, row.token, "legacy")
            continue
        # Does another, kept referent already own this surface? A key that resolves to the row
        # ITSELF is the legacy fallback talking, not an answer — ask the identity resolver.
        owner = idx.keys.get(_normalize(surface))
        if owner == row.token:
            owner = None
        if owner is None and row.kind == "PERSON":
            resolved, _amb = _resolve_person(idx, surface)
            owner = resolved
            if owner is None:
                frag = _fragment_owners(idx, surface)
                owner = frag[0] if len(frag) == 1 else None
        if owner is None:
            # BEFORE calling a single-word row junk, ask the ORGANISATION resolver. Rows like
            # PERSON_25='Tarvelo', PERSON_30='Quorvane' and PERSON_31='NRTK' are real companies
            # the OLD tokenizer covered; retiring them as "single word" with no canonical and no
            # alias put three company names back into clear text. They resolve uniquely by
            # legal-form-stripped prefix onto ORG_10, ORG_12 and ORG_13.
            owner = _merge_org(store, idx, scope_id, surface)
        if owner is not None and owner != row.token:
            _retire(row, owner, "merged", into=owner)
            _add_alias(store, idx, scope_id, surface, owner, "legacy")
            continue
        ok, reason = _admit(surface, row.kind, idx)
        if not ok:
            _retire(row, None, "retired", reason=reason)
            continue
        _add_alias(store, idx, scope_id, surface, row.token, "legacy")
        report["kept"].append(row.token)

    # RE-DERIVE ``shared_by`` NOW THAT EVERY MERGE HAS LANDED. A shared-name token records the
    # tokens the name could denote; that list is computed while merging is still in progress,
    # so it can name a token this same pass has since retired onto another. Measured:
    # PERSON_61.shared_by = ["PERSON_20", "PERSON_9"] where PERSON_20 is retired INTO PERSON_9
    # — a payload line asserting that one man is two people, which is precisely what the shared
    # token exists to prevent. Resolve through ``canonical_token``, keep only active tokens,
    # de-duplicate; and if the set collapses to ONE, the surface was never ambiguous, so the
    # token becomes a retired alias of that member instead of an ambiguity nobody has.
    for row in sorted(idx.rows, key=lambda r: r.token):
        shared = (row.attributes or {}).get("shared_by")
        if row.status != "active" or not shared:
            continue
        resolved = []
        for t in shared:
            c = idx.canonical(t)
            crow = idx.by_token.get(c) if c else None
            if crow is not None and crow.status == "active" and c != row.token and c not in resolved:
                resolved.append(c)
        resolved.sort(key=_token_index)
        if len(resolved) >= 2:
            if resolved != list(shared):
                row.attributes = dict(row.attributes or {}, shared_by=resolved)
                report.setdefault("shared_rewritten", []).append(
                    {"token": row.token, "was": list(shared), "now": resolved})
            continue
        target = resolved[0] if resolved else None
        if target is None:
            # EVERY name that shared this part is gone. The token is then a part of names that no
            # longer exist, and left active it rewrites the word wherever it occurs: a demo
            # scope kept PERSON_5 = "Insert" after "Insert Title", "Insert Rate", "Insert VAT",
            # "Insert Address" and "Insert Date" were retired, so every "Insert" in a template
            # would have crossed as a person (23.09.2026).
            _retire(row, None, "retired", reason="a part shared only by retired names")
            continue
        surface = row.real_value.strip()
        _retire(row, target, "merged", into=target)
        _add_alias(store, idx, scope_id, surface, target, "legacy")
        report.setdefault("shared_collapsed", []).append({"token": row.token, "into": target})

    if not dry_run:
        await store.commit()
    return report


# ── Чтение для обезличенного фрагмента и проверок «отказ закрыт» (15.09.2026) ────────────
async def token_attributes(store: TokenStore, scope_id: ScopeId,
                           tokens: list[str] | None = None) -> dict[str, tuple[str, dict]]:
    """Токен → (вид, атрибуты) — например, ``attributes["roles"]`` сущности для меток ролей.

    Только чтение: ни соли, ни дозачисления. Ушедший в отставку токен отвечает атрибутами
    своего канонического — на выходе всегда тот токен, что стоит в тексте."""
    rows = await _load(store, scope_id)
    by_token = {r.token: r for r in rows}
    out: dict[str, tuple[str, dict]] = {}
    for tok in (tokens if tokens is not None else list(by_token)):
        row = by_token.get(tok)
        if row is None:
            continue
        attrs = row.attributes or {}
        if not attrs and row.canonical_token and row.canonical_token in by_token:
            attrs = by_token[row.canonical_token].attributes or {}
        out[tok] = (row.kind, dict(attrs))
    return out


async def active_people(store: TokenStore, scope_id: ScopeId) -> dict[str, str]:
    """``{token: real name}`` of the scope's ACTIVE natural persons. Read-only."""
    idx = await _index_of(store, scope_id)
    return {r.token: r.real_value for r in idx.rows
            if r.kind == "PERSON" and r.status == "active" and r.real_value}


async def residual_surfaces(store: TokenStore, scope_id: ScopeId, text: str) -> list[str]:
    """Зачисленные в сейф реквизиты, которые ЕЩЁ стоят в ``text`` — проверка «отказ закрыт»
    для обезличенного фрагмента тем же индексом, что применяет сейф, но БЕЗ дозачисления:
    это проверка результата, а не переход границы, и писать в сейф она не должна."""
    if not text or not text.strip():
        return []
    idx = await _index_of(store, scope_id)
    _, found = _apply_index(text, idx.match_keys(), idx)
    return sorted({s for surfaces in found.values() for s in surfaces})


async def graph_index(store: TokenStore, scope_id: ScopeId) -> "_Index":
    """Индекс сейфа ТОЛЬКО по графу области: сущности, стороны и структурные виды (IBAN,
    e-mail, телефон — их распознавание не суждение, а разбор). Без остаточных догадок Presidio.

    ЗАЧЕМ (пилот обезличивания, 16.09.2026): на немецком тексте остаточный проход зачислил как
    ЛЮДЕЙ «Birkenallee», «Seestraße», «Februar», «Wohnung», «Rechnung Nr.» — улица съедена
    токеном, номер дома и этаж остались («[person 1] 17, 3. OG links»), и семантическая
    разметка адреса уже не увидела. Для фрагмента имена вне графа находит модель, а не Presidio."""
    # Признак графа — на САМОЙ строке (entity_type / roles / identifier_of_name / source=intake):
    # псевдонимы остаточной строки тоже носят источники «fragment» и «variant» (части и варианты
    # написания догадки), и по ним графовую строку от остаточной не отличить — 16.09.2026 так
    # «Wohnung Birkenallee» (PERSON, residual) прошла в индекс фрагмента.
    rows = await _load(store, scope_id)
    aliases = await _load_aliases(store, scope_id)
    keep = [r for r in rows if r.status == "active" and _from_graph(r)]
    tokens = {r.token for r in keep}
    idx = _Index(keep, [a for a in aliases if a.token in tokens], await _salt_of(store, scope_id))
    # Обращение, зачисленное частью имени («Herrn» у «Herrn Paul Winter»), метит любого
    # «Herrn» в тексте токеном этого человека; в индексе фрагмента такие ключи не участвуют.
    for key in [k for k in idx.keys if k in _FRAME_KEYS]:
        del idx.keys[key]
    return idx


#: Слова обращения и артикли, которые не могут быть ключом совпадения сами по себе.
_FRAME_KEYS = frozenset({
    "herr", "herrn", "frau", "fraeulein", "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam",
    "der", "die", "das", "dem", "den", "des", "the", "firma", "gmbh", "ltd", "limited", "ag", "kg",
})


def _from_graph(row) -> bool:
    attrs = row.attributes or {}
    if attrs.get("source") in ("residual", "mailbox"):
        return False
    return bool(row.kind in _STRUCTURAL_KINDS or attrs.get("entity_type") or attrs.get("roles")
                or attrs.get("identifier_of_name") or attrs.get("source") == "intake")


async def graph_tokenize(store: TokenStore, scope_id: ScopeId, text: str, *,
                         seeds: SeedSource | None = None) -> Redaction:
    """Как :func:`tokenize`, но только по графу области (см. :func:`graph_index`) и без записи в
    сейф: ничего не дозачисляется, соль не заводится. Для обезличенного фрагмента."""
    if not text or not text.strip():
        return Redaction(text=text or "")
    await seed_from_scope(store, scope_id, seeds)
    idx = await graph_index(store, scope_id)
    out, found = _apply_index(text, idx.match_keys(), idx)
    mapping: dict[str, str] = {}
    surfaces_seen: dict[str, list[str]] = {}
    canonical: dict[str, str] = {}
    kinds: Counter[str] = Counter()
    for token, surfaces in found.items():
        row = idx.by_token.get(token)
        if row is None:
            continue
        distinct = {_normalize(s) for s in surfaces}
        mapping[token] = surfaces[0] if len(distinct) == 1 else row.real_value
        surfaces_seen[token] = list(surfaces)
        canonical[token] = row.real_value
        kinds[row.kind] += len(surfaces)
    return Redaction(text=out, entity_types=sorted(kinds), redacted_count=int(sum(kinds.values())),
                     mapping=mapping, surfaces=surfaces_seen, canonical=canonical)


async def graph_residual_surfaces(store: TokenStore, scope_id: ScopeId, text: str) -> list[str]:
    """Реквизиты ГРАФА области, ещё стоящие в ``text`` — проверка «отказ закрыт» для фрагмента."""
    if not text or not text.strip():
        return []
    idx = await graph_index(store, scope_id)
    _, found = _apply_index(text, idx.match_keys(), idx)
    return sorted({s for surfaces in found.values() for s in surfaces})
