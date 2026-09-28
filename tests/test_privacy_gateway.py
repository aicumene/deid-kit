# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Privacy Gateway — the leak guarantee.

The guard case (sensitivity > INTERNAL → block) is exercised with a stub redactor; the success
path audits into an in-memory sink. Neither needs Presidio.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from deidkit import tracing
from deidkit.classification import Sensitivity
from deidkit.gateway import InMemoryAuditSink, PrivacyGateway, Redaction, content_hash
from deidkit.model import PrivacyViolation


def _stub_redactor(seen: list[str]):
    def _r(text: str) -> Redaction:
        seen.append(text)
        return Redaction(text="[redacted]", entity_types=["PERSON"], redacted_count=1)
    return _r


@pytest.mark.parametrize("sensitivity", [Sensitivity.CONFIDENTIAL, Sensitivity.RESTRICTED])
async def test_blocks_too_sensitive_before_redaction_or_audit(sensitivity):
    seen: list[str] = []
    audit = InMemoryAuditSink()
    gateway = PrivacyGateway(redact_fn=_stub_redactor(seen), audit=audit)
    with pytest.raises(PrivacyViolation):
        await gateway.cross_to_cloud(
            None, actor_id=None, action="t", sensitivity=sensitivity,
            payload="Jane Doe SSN 123-45-6789", target="research-agent",
        )
    assert seen == [], "redactor must NOT run for blocked payloads"
    assert audit.records == [], "a refused crossing is not a crossing"


async def test_internal_passes_through_redactor_and_audits():
    seen: list[str] = []
    audit = InMemoryAuditSink()
    gateway = PrivacyGateway(redact_fn=_stub_redactor(seen), audit=audit)
    ctx = object()
    redaction = await gateway.cross_to_cloud(
        ctx, actor_id="user-7", action="research.query",
        sensitivity=Sensitivity.INTERNAL,
        payload="Does Jane Doe have a claim under the policy?",
        target="research-agent",
    )
    assert seen and "Jane Doe" in seen[0]              # raw text reached only the redactor
    assert redaction.text == "[redacted]"
    assert len(audit.records) == 1
    context, rec = audit.records[0]
    assert context is ctx                               # handed to the sink untouched
    assert rec.plane == "cloud" and rec.resource_type == "privacy.cross"
    assert rec.sensitivity == "Internal" and rec.actor_id == "user-7"
    assert rec.content_hash == content_hash("[redacted]")  # hash is of REDACTED text
    assert rec.detail == {"target": "research-agent", "entity_types": ["PERSON"],
                          "redacted_count": 1}
    assert "Jane Doe" not in str(rec)                   # no PII in the audit record


async def test_a_pre_redacted_payload_skips_the_redactor_but_not_the_audit():
    seen: list[str] = []
    audit = InMemoryAuditSink()
    gateway = PrivacyGateway(redact_fn=_stub_redactor(seen), audit=audit)
    pre = Redaction(text="PERSON_1 asks about ORG_2.", entity_types=["ORG", "PERSON"],
                    redacted_count=2, mapping={"PERSON_1": "x", "ORG_2": "y"})
    out = await gateway.cross_to_cloud(None, actor_id=None, action="chat",
                                       sensitivity=Sensitivity.PUBLIC, payload="ignored",
                                       target="model", pre_redacted=pre)
    assert out is pre and seen == []
    assert audit.records[0][1].content_hash == content_hash("PERSON_1 asks about ORG_2.")


async def test_without_presidio_the_default_redactor_refuses(monkeypatch):
    """No active redactor → no crossing. The trusted-plane guarantee is worth more than
    convenience."""
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", None)
    audit = InMemoryAuditSink()
    gateway = PrivacyGateway(audit=audit)
    with pytest.raises(PrivacyViolation, match="Presidio is not installed"):
        await gateway.cross_to_cloud(None, actor_id=None, action="t",
                                     sensitivity=Sensitivity.INTERNAL, payload="Jane Doe",
                                     target="model")
    assert audit.records == []


async def test_the_crossing_is_traced_with_identifiers_only():
    spans, attrs = [], []

    @contextmanager
    def factory(name, **a):
        spans.append((name, a))
        yield None

    tracing.set_span_factory(factory)
    tracing.set_attribute_setter(lambda **a: attrs.append(a))
    try:
        gateway = PrivacyGateway(redact_fn=_stub_redactor([]), audit=InMemoryAuditSink())
        await gateway.cross_to_cloud(None, actor_id=None, action="research.query",
                                     sensitivity=Sensitivity.INTERNAL, payload="Jane Doe",
                                     target="model")
    finally:
        tracing.set_span_factory(None)
        tracing.set_attribute_setter(None)
    assert spans == [("privacy.cross_to_cloud", {"privacy.action": "research.query",
                                                 "privacy.target": "model",
                                                 "privacy.sensitivity": "Internal"})]
    assert attrs == [{"privacy.redacted_count": 1, "privacy.entity_types": ["PERSON"]}]
    assert "Jane" not in repr(spans) + repr(attrs)
