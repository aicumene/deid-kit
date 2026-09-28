# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""deid-kit: reversible, per-scope de-identification of text on its way to a cloud model.

A vault replaces the people, organisations and identifiers in a text with salted, deterministic
tokens (``PERSON_48170392``) and restores them in the answer; a glossary tells the model what
each token is without saying who; a gateway checks eligibility and audits every crossing; and a
probe measures whether a de-identified excerpt can still be traced back to its source.

The vault works through three interfaces a deployment implements — a token store
(:class:`~deidkit.store.TokenStore`), a seed source (:class:`~deidkit.seeds.SeedSource`) and a
residual PII detector (:class:`~deidkit.detect.PiiDetector`) — with in-memory implementations
included. Start with :func:`deidkit.vault.tokenize` and :func:`deidkit.vault.detokenize`.
"""

from deidkit.gateway import PrivacyGateway, Redaction
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

__all__ = [
    "InMemorySeedSource",
    "InMemoryTokenStore",
    "PrivacyGateway",
    "Redaction",
    "SeedEntity",
]
