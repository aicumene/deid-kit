# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Where the vault keeps its tokens, aliases and per-scope salts.

A *scope* is the unit of pseudonymisation: one project, case, customer or tenant — whatever a
deployment wants a person to have ONE token in, and a different, unlinkable token outside of.
Every read and write the vault makes is scoped by a ``scope_id`` (any hashable value).

THE CONTRACT (:class:`TokenStore`). Three tables' worth of state:

  * token rows (:class:`TokenRow`) — one per referent: the token, its kind, the real value the
    return leg restores, a status (``"active"`` / ``"retired"``), the token a retired duplicate
    now resolves to, and non-identifying attributes for the glossary;
  * alias rows (:class:`AliasRow`) — the match index: one surface → one token;
  * one salt per scope — the key the tokens are derived from.

Loads are ``async`` and return LISTS OF LIVE ROWS: the vault edits a loaded row in place
(``status``, ``canonical_token``, ``attributes``; an alias's ``token``, ``surface``, ``source``
and ``normalized``), exactly as it edits an ORM object in a session. Adds are synchronous and
return the new row, which the vault keeps using. ``commit()`` is called where the vault has
finished a unit of work; a persistent store saves the added rows AND the in-place edits there.
Each ``load_*`` call must return a NEW list (the vault appends to it), holding the same row
objects each time.

A store may use any row class with the same attribute names — an SQLAlchemy model satisfies the
contract as it stands. Nothing in the vault ever deletes a row: a defective token is retired, so
every answer that was ever sent with it can still be re-hydrated.

:class:`InMemoryTokenStore` is the reference implementation (and the one the tests run on). It
has no transactions: what the vault did before an uncommitted ``reconcile_scope(dry_run=True)``
returns is kept.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Protocol

#: Anything hashable identifies a scope: a UUID, an integer, a string.
ScopeId = Hashable


@dataclass(slots=True, eq=False)
class TokenRow:
    """One pseudonym and the real value it stands for."""

    scope_id: ScopeId
    token: str
    kind: str
    real_value: str
    # The old dedup key (a fold of the value). Kept unique per scope; nothing reads it.
    normalized: str
    status: str = "active"
    # For a retired duplicate: the token that now carries this referent.
    canonical_token: str | None = None
    # Derived, non-identifying facts rendered into the glossary. Never an identifier.
    attributes: dict | None = None


@dataclass(slots=True, eq=False)
class AliasRow:
    """One surface that resolves to a token — the vault's match index."""

    scope_id: ScopeId
    token: str
    surface: str
    # The match key (``deidkit.namefold.key``), unique per scope.
    normalized: str
    # entity | intake | fragment | variant | residual | legacy | retracted
    source: str = "entity"


class TokenStore(Protocol):
    """What the vault reads and writes. See the module docstring for the contract."""

    async def load_tokens(self, scope_id: ScopeId) -> list[Any]:
        """Every token row of the scope, active and retired."""
        ...

    async def load_aliases(self, scope_id: ScopeId) -> list[Any]:
        """Every alias row of the scope."""
        ...

    async def load_salt(self, scope_id: ScopeId) -> bytes | None:
        """The scope's salt, or ``None`` when it has none yet. Must not create one."""
        ...

    def add_token(self, scope_id: ScopeId, *, token: str, kind: str, real_value: str,
                  normalized: str, status: str, attributes: dict | None) -> Any:
        """Add a token row; return it."""
        ...

    def add_alias(self, scope_id: ScopeId, *, token: str, surface: str, normalized: str,
                  source: str) -> Any:
        """Add an alias row; return it."""
        ...

    def add_salt(self, scope_id: ScopeId, salt: bytes) -> None:
        """Record the scope's salt (called once per scope, by ``ensure_salt``)."""
        ...

    async def commit(self) -> None:
        """Persist everything added and edited since the last commit."""
        ...


class InMemoryTokenStore:
    """A :class:`TokenStore` in process memory."""

    def __init__(self) -> None:
        self._tokens: dict[ScopeId, list[TokenRow]] = {}
        self._aliases: dict[ScopeId, list[AliasRow]] = {}
        self._salts: dict[ScopeId, bytes] = {}
        #: How many times the vault committed — a unit of work that changed nothing does not.
        self.commits = 0

    # ── TokenStore ───────────────────────────────────────────────────────────
    async def load_tokens(self, scope_id: ScopeId) -> list[TokenRow]:
        return list(self._tokens.get(scope_id, ()))

    async def load_aliases(self, scope_id: ScopeId) -> list[AliasRow]:
        return list(self._aliases.get(scope_id, ()))

    async def load_salt(self, scope_id: ScopeId) -> bytes | None:
        return self._salts.get(scope_id)

    def add_token(self, scope_id: ScopeId, *, token: str, kind: str, real_value: str,
                  normalized: str, status: str = "active",
                  attributes: dict | None = None) -> TokenRow:
        row = TokenRow(scope_id=scope_id, token=token, kind=kind, real_value=real_value,
                       normalized=normalized, status=status, attributes=attributes)
        self._tokens.setdefault(scope_id, []).append(row)
        return row

    def add_alias(self, scope_id: ScopeId, *, token: str, surface: str, normalized: str,
                  source: str = "entity") -> AliasRow:
        row = AliasRow(scope_id=scope_id, token=token, surface=surface, normalized=normalized,
                       source=source)
        self._aliases.setdefault(scope_id, []).append(row)
        return row

    def add_salt(self, scope_id: ScopeId, salt: bytes) -> None:
        self._salts[scope_id] = salt

    async def commit(self) -> None:
        self.commits += 1

    # ── inspection and loading (not part of the protocol) ────────────────────
    def token_rows(self, scope_id: ScopeId) -> list[TokenRow]:
        """The scope's token rows — the live list, for inspection or for loading existing rows."""
        return self._tokens.setdefault(scope_id, [])

    def alias_rows(self, scope_id: ScopeId) -> list[AliasRow]:
        """The scope's alias rows — the live list."""
        return self._aliases.setdefault(scope_id, [])


__all__ = ["AliasRow", "InMemoryTokenStore", "ScopeId", "TokenRow", "TokenStore"]
