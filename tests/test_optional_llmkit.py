# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""deid-kit does not need llm-kit — and when both are installed, they share one vocabulary.

Without llm-kit (this test environment) the package defines its own ``Sensitivity``,
``PrivacyViolation`` and request types, value- and field-compatible with llm-kit's. With llm-kit
installed they ARE llm-kit's objects, so one ``except PrivacyViolation`` catches a refusal from
the router and from the gateway alike. The second case runs in a subprocess against a stand-in
``llmkit`` package, so nothing leaks into this interpreter.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap

from deidkit import classification, model


def test_without_llmkit_the_local_definitions_are_used():
    assert "llmkit" not in sys.modules
    assert classification.Sensitivity.__module__ == "deidkit.classification"
    assert [(s.name, int(s), s.label) for s in classification.Sensitivity] == [
        ("PUBLIC", 0, "Public"), ("INTERNAL", 1, "Internal"),
        ("CONFIDENTIAL", 2, "Confidential"), ("RESTRICTED", 3, "Restricted")]
    assert classification.max_crossable() is classification.Sensitivity.INTERNAL
    assert classification.may_cross_to_cloud(classification.Sensitivity.PUBLIC)
    assert not classification.may_cross_to_cloud(classification.Sensitivity.CONFIDENTIAL)
    assert issubclass(model.PrivacyViolation, RuntimeError)
    fields = {f.name: f.default for f in dataclasses.fields(model.ModelRequest)
              if f.default is not dataclasses.MISSING}
    assert fields == {"sensitivity": classification.Sensitivity.CONFIDENTIAL, "model": None,
                      "temperature": 0.2, "max_tokens": None, "think": None,
                      "response_format": None, "allow_cloud": False}
    assert (model.QUERY, model.DOCUMENT) == ("query", "document")


_FAKE_LLMKIT = {
    "llmkit/__init__.py": "",
    "llmkit/classification.py": """
        from enum import IntEnum
        class Sensitivity(IntEnum):
            PUBLIC = 0
            INTERNAL = 1
            CONFIDENTIAL = 2
            RESTRICTED = 3
            @property
            def label(self):
                return self.name.capitalize()
        def max_crossable():
            return Sensitivity.INTERNAL
        def may_cross_to_cloud(s):
            return s <= max_crossable()
    """,
    "llmkit/base.py": """
        from dataclasses import dataclass, field
        DOCUMENT = "document"
        QUERY = "query"
        @dataclass
        class LLMMessage:
            role: str
            content: str
        @dataclass
        class LLMRequest:
            messages: list
            sensitivity: int = 2
            model: str = None
            temperature: float = 0.2
            max_tokens: int = None
            think: bool = None
            response_format: dict = None
            allow_cloud: bool = False
            metadata: dict = field(default_factory=dict)
    """,
    "llmkit/router.py": """
        class PrivacyViolation(RuntimeError):
            pass
    """,
}


def test_with_llmkit_installed_the_objects_are_llmkits(tmp_path):
    for rel, body in _FAKE_LLMKIT.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    probe = textwrap.dedent("""
        import asyncio
        import llmkit.base, llmkit.classification, llmkit.router
        from deidkit import classification, model
        from deidkit.gateway import InMemoryAuditSink, PrivacyGateway
        assert classification.Sensitivity is llmkit.classification.Sensitivity
        assert model.PrivacyViolation is llmkit.router.PrivacyViolation
        assert model.ModelRequest is llmkit.base.LLMRequest
        assert model.ModelMessage is llmkit.base.LLMMessage
        gw = PrivacyGateway(audit=InMemoryAuditSink(), redact_fn=lambda t: None)
        try:
            asyncio.run(gw.cross_to_cloud(None, actor_id=None, action="t", target="m",
                                          sensitivity=classification.Sensitivity.RESTRICTED,
                                          payload="x"))
        except llmkit.router.PrivacyViolation:
            print("caught by llm-kit's class")
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), *sys.path]))
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                         check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "caught by llm-kit's class"
