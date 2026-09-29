# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Known people and organisations of a scope, read from a TOML file kept beside the files.

The vault matches known names deterministically, with their spellings, transliterations and
inflected forms; a detector only finds what it recognises. So the names that matter should be
enrolled before the first request, and the simplest place to list them is a file::

    # deid.toml
    scope = "client-a"                  # optional; the caller's scope applies when absent
    paths = ["~/matters/client-a"]      # optional: this client's project folders

    [[entity]]
    type = "individual"                 # individual | company | vessel | account | property
    name = "Ada Brenner"
    role = "Director"                   # optional

    [[entity]]
    type = "company"
    name = "Harrowgate Freight Ltd"
    jurisdiction_country = "GB"         # optional

    [[party]]                           # optional: names with a role, as an intake lists them
    name = "Harrowgate Freight Ltd"
    role = "Counterparty / Company"

The file holds real names: keep it out of the repository (for example in ``.git/info/exclude``)
or outside the working tree.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from deidkit.seeds import InMemorySeedSource, SeedEntity

_ENTITY_FIELDS = ("identifier", "identifier_type", "jurisdiction_country", "role")


def load_seed_files(paths: list[str | os.PathLike], default_scope: str,
                    into: InMemorySeedSource | None = None) -> InMemorySeedSource:
    """Read seed files into one :class:`~deidkit.seeds.InMemorySeedSource`, keyed by scope."""
    src = into or InMemorySeedSource()
    for path in paths:
        with open(Path(path).expanduser(), "rb") as fh:
            data = tomllib.load(fh)
        scope = str(data.get("scope") or default_scope)
        for e in data.get("entity", []):
            if not e.get("name") or not e.get("type"):
                raise ValueError(f"{path}: every [[entity]] needs a type and a name")
            src.add_entity(scope, SeedEntity(
                type=str(e["type"]), name=str(e["name"]),
                **{f: str(e.get(f, "")) for f in _ENTITY_FIELDS},
                attributes=e.get("attributes")))
        parties = data.get("party")
        if parties:
            src.set_parties(scope, [{"name": str(p["name"]), "role": str(p.get("role", ""))}
                                    for p in parties])
    return src


def load_scope_paths(paths: list[str | os.PathLike], default_scope: str) -> list[tuple[str, str]]:
    """``(folder, scope)`` pairs from the seed files' ``paths`` lists, folders resolved (symbolic
    links followed, so ``/tmp`` and ``/private/tmp`` agree), longest first."""
    out: list[tuple[str, str]] = []
    for path in paths:
        with open(Path(path).expanduser(), "rb") as fh:
            data = tomllib.load(fh)
        scope = str(data.get("scope") or default_scope)
        for folder in data.get("paths", []):
            out.append((os.path.realpath(os.path.expanduser(str(folder))), scope))
    return sorted(out, key=lambda pair: -len(pair[0]))


__all__ = ["load_scope_paths", "load_seed_files"]
