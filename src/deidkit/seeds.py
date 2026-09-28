# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""What the vault knows about a scope BEFORE it reads a word of text: its entities and parties.

Seeding is what makes detection deterministic for the referents a deployment already knows. The
vault enrols every seed entity (a person, company, vessel, account, property — with identifiers,
roles, a country and ownership edges) and every party (a name with a role) before the residual
detector sees the text, so a known name becomes an alias of its referent's token instead of a
new guess. See ``deidkit.vault.seed_from_scope``.

THE CONTRACT (:class:`SeedSource`), two reads:

  * ``entities(scope_id)`` — objects with the attributes of :class:`SeedEntity`
    (``type``, ``name``, ``identifier``, ``identifier_type``, ``jurisdiction_country``, ``role``,
    ``attributes``, ``relationships``, ``id``). ``None`` in any optional attribute is read as empty.
    ``type`` is one of ``individual``, ``company``, ``vessel``, ``account``, ``property``; anything
    else is enrolled as a generic ``ENTITY``. ``role`` is a comma-separated list.
    ``relationships`` is a list of ``{"type", "target", "shareholding_pct"}`` (``target`` is the
    other entity's name); ``attributes`` may carry the identifiers ``imo``, ``mmsi``,
    ``callsign``, ``iban``, ``registration_number``.
  * ``parties(scope_id)`` — a list of ``{"name": ..., "role": ...}`` or ``None``. The role
    decides whether a party is split into name parts: only a party whose role says it is an
    individual is ("Director", "Shareholder", "Individual"…); a role that says company
    ("Company", "Vessel Owner", "Holding"…) never is.

:class:`InMemorySeedSource` is the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from deidkit.store import ScopeId


@dataclass(slots=True, eq=False)
class SeedEntity:
    """A known referent of a scope."""

    type: str
    name: str
    identifier: str = ""
    identifier_type: str = ""
    jurisdiction_country: str = ""
    role: str = ""
    attributes: dict | None = None
    relationships: list | None = None
    id: Any = None


class SeedSource(Protocol):
    """Where the vault reads a scope's entities and parties from."""

    async def entities(self, scope_id: ScopeId) -> list[Any]:
        """The scope's entities (see :class:`SeedEntity` for the attributes read)."""
        ...

    async def parties(self, scope_id: ScopeId) -> list[dict] | None:
        """The scope's parties as ``[{"name", "role"}]``, or ``None``."""
        ...


class InMemorySeedSource:
    """A :class:`SeedSource` in process memory."""

    def __init__(self, entities: dict[ScopeId, list[Any]] | None = None,
                 parties: dict[ScopeId, list[dict] | None] | None = None) -> None:
        self._entities: dict[ScopeId, list[Any]] = {k: list(v) for k, v in (entities or {}).items()}
        self._parties: dict[ScopeId, list[dict] | None] = dict(parties or {})

    async def entities(self, scope_id: ScopeId) -> list[Any]:
        return list(self._entities.get(scope_id, ()))

    async def parties(self, scope_id: ScopeId) -> list[dict] | None:
        return self._parties.get(scope_id)

    def add_entity(self, scope_id: ScopeId, entity: Any) -> None:
        self._entities.setdefault(scope_id, []).append(entity)

    def set_parties(self, scope_id: ScopeId, parties: list[dict] | None) -> None:
        self._parties[scope_id] = parties


__all__ = ["InMemorySeedSource", "SeedEntity", "SeedSource"]
