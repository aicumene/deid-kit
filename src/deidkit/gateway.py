# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The Privacy Gateway — the SOLE sanctioned egress from trusted → cloud.

For any cross-plane request, the gateway:
  1. **Checks eligibility** — refuses anything above ``INTERNAL`` (PrivacyViolation).
  2. **De-identifies** the payload (Presidio Analyzer + Anonymizer by default) — facts, not
     names — unless the caller hands in text the vault already tokenised.
  3. **Audits** — hands one :class:`CrossingRecord` to the :class:`AuditSink`: the plane, the
     sensitivity, the entity types seen, the redaction count, and a SHA-256 of the
     *de-identified* payload (the raw payload is never part of the record).

The redactor and the audit sink are injected so the policy can be tested without Presidio and
so a deployment decides where audit records live (an append-only table, a log stream).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from deidkit.classification import Sensitivity, may_cross_to_cloud
from deidkit.model import PrivacyViolation
from deidkit.tracing import set_attributes, start_span


@dataclass(slots=True)
class Redaction:
    text: str
    entity_types: list[str] = field(default_factory=list)
    redacted_count: int = 0
    # token → real-value map for the reversible vault path (empty for the lossy redactor).
    # Lets the trusted plane re-hydrate real names in the cloud answer.
    mapping: dict[str, str] = field(default_factory=dict)
    # One line per token that actually crossed, describing what the token IS without
    # identifying it ("PERSON_1 = individual (natural person), male, jurisdiction BE,
    # beneficial owner of ORG_2 (>=50%)"). Built by deidkit/glossary.py from the scope's
    # graph — never by a model — and sent OUT-OF-BAND alongside the de-identified text, so the
    # cloud plane can reason in context while the return leg is unchanged: the answer still
    # comes back in tokens and detokenize() still reverses it.
    glossary: list[str] = field(default_factory=list)
    # Per token, the raw surfaces THIS text actually used, and the vault's canonical value.
    # ``mapping`` collapses both into one string per token, which is a per-call decision; a
    # caller that tokenises several pieces of text for ONE crossing (a chat sends a system
    # block, the history and the question) has to re-derive the mapping over the union or the
    # same man is named two ways across turns. See ``vault.merge_mappings``.
    surfaces: dict[str, list[str]] = field(default_factory=dict)
    canonical: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CrossingRecord:
    """What the gateway audits for one crossing. Identifiers and counts only — never content."""

    actor_id: Any
    action: str
    resource_type: str        # always "privacy.cross"
    plane: str                # always "cloud"
    sensitivity: str          # the label, e.g. "Internal"
    content_hash: str         # SHA-256 (hex) of the de-identified text that crossed
    detail: dict              # {"target", "entity_types", "redacted_count"}


class AuditSink(Protocol):
    """Where crossings are recorded. ``context`` is whatever the caller passed to
    :meth:`PrivacyGateway.cross_to_cloud` (a database session, a request) — the gateway does
    not look at it."""

    async def record(self, context: Any, record: CrossingRecord) -> None:
        ...


class InMemoryAuditSink:
    """An :class:`AuditSink` that keeps ``(context, record)`` pairs in a list."""

    def __init__(self) -> None:
        self.records: list[tuple[Any, CrossingRecord]] = []

    async def record(self, context: Any, record: CrossingRecord) -> None:
        self.records.append((context, record))


def content_hash(text: str) -> str:
    """SHA-256 (hex) of the UTF-8 text — the hash a crossing is audited under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _presidio_redact(text: str) -> Redaction:
    """Default redactor using Microsoft Presidio (Analyzer + Anonymizer).

    Imported lazily; if Presidio isn't available, we REFUSE rather than send unredacted
    text — the trusted-plane guarantee is more important than convenience."""
    try:
        from presidio_anonymizer import AnonymizerEngine

        from deidkit.presidio import PII_ENTITIES, analyzer  # reuse the cached engine
    except Exception as exc:
        raise PrivacyViolation(
            "Presidio is not installed; refusing to cross the privacy boundary without "
            "an active redactor."
        ) from exc

    results = analyzer().analyze(
        text=text, language="en", entities=PII_ENTITIES, score_threshold=0.5
    )
    anonymized = AnonymizerEngine().anonymize(text=text, analyzer_results=results)
    counts = Counter(r.entity_type for r in results)
    return Redaction(
        text=anonymized.text,
        entity_types=sorted(counts),
        redacted_count=int(sum(counts.values())),
    )


class PrivacyGateway:
    """The one sanctioned trusted → cloud egress."""

    def __init__(self, *, audit: AuditSink,
                 redact_fn: Callable[[str], Redaction] | None = None) -> None:
        self._audit = audit
        self._redact = redact_fn or _presidio_redact

    async def cross_to_cloud(
        self,
        context: Any,
        *,
        actor_id: Any,
        action: str,
        sensitivity: Sensitivity,
        payload: str,
        target: str,
        pre_redacted: Redaction | None = None,
    ) -> Redaction:
        """Redact ``payload`` and audit the crossing. Raise if not eligible.

        Returns the :class:`Redaction` for the caller to send onward. The caller MUST send
        only ``redaction.text`` (never the raw payload) to the cloud plane.

        ``pre_redacted`` lets a caller that already de-identified the text through the
        reversible vault hand the result in directly — the gateway then skips its own redactor
        but still enforces eligibility and writes the audit record (hashing the tokenized text).

        ``context`` is passed to the audit sink untouched.
        """
        with start_span(
            "privacy.cross_to_cloud",
            **{
                "privacy.action": action,
                "privacy.target": target,
                "privacy.sensitivity": Sensitivity(sensitivity).label,
            },
        ):
            if not may_cross_to_cloud(sensitivity):
                raise PrivacyViolation(
                    f"Refusing to cross {Sensitivity(sensitivity).label} payload to cloud."
                )

            redaction = pre_redacted if pre_redacted is not None else self._redact(payload)
            set_attributes(**{
                "privacy.redacted_count": redaction.redacted_count,
                "privacy.entity_types": redaction.entity_types,
            })

            await self._audit.record(context, CrossingRecord(
                actor_id=actor_id, action=action, resource_type="privacy.cross",
                plane="cloud", sensitivity=Sensitivity(sensitivity).label,
                content_hash=content_hash(redaction.text),
                detail={
                    "target": target,
                    "entity_types": redaction.entity_types,
                    "redacted_count": redaction.redacted_count,
                },
            ))
            return redaction


__all__ = ["AuditSink", "CrossingRecord", "InMemoryAuditSink", "PrivacyGateway", "Redaction",
           "content_hash"]
