# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The narrow model interface the gateway and the re-identification probe need.

deid-kit calls a model in exactly two places — :func:`deidkit.reid.probe_retrieval` embeds a
text and :func:`deidkit.reid.probe_judge` asks for one completion — and refuses a crossing in
one (:class:`deidkit.gateway.PrivacyGateway`). So the whole dependency is:

  * :class:`ModelRouter` — any object with ``embed(texts, *, sensitivity, kind)`` and
    ``generate(request)`` whose reply has a ``.text``;
  * :class:`ModelRequest` / :class:`ModelMessage` — what ``generate`` receives;
  * :class:`PrivacyViolation` — what a refused crossing raises;
  * ``QUERY`` / ``DOCUMENT`` — the role of a text for an asymmetric embedder.

When llm-kit is installed these names ARE llm-kit's objects (``LLMRequest``, ``LLMMessage``,
``PrivacyViolation``, ``QUERY``, ``DOCUMENT``), so an ``LLMRouter`` is a ``ModelRouter`` as it
stands, and one ``except PrivacyViolation`` catches a refusal from the router and from the
gateway alike. Without llm-kit the definitions below are used: the same field names and
defaults, so any router that reads llm-kit's request reads this one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from deidkit.classification import Sensitivity

try:
    from llmkit.base import DOCUMENT, QUERY
    from llmkit.base import LLMMessage as ModelMessage
    from llmkit.base import LLMRequest as ModelRequest
    from llmkit.router import PrivacyViolation
except ImportError:
    #: The role of a text in an asymmetric embedder.
    DOCUMENT: str = "document"  # type: ignore[no-redef]
    QUERY: str = "query"  # type: ignore[no-redef]

    class PrivacyViolation(RuntimeError):  # type: ignore[no-redef]
        """Raised when a crossing would send too-sensitive data off the trusted plane."""

    @dataclass(slots=True)
    class ModelMessage:  # type: ignore[no-redef]
        role: str  # "system" | "user" | "assistant"
        content: str

    @dataclass(slots=True)
    class ModelRequest:  # type: ignore[no-redef]
        """One inference request, carrying the sensitivity of its payload."""

        messages: list[ModelMessage]
        sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL  # safe default: assume private data
        model: str | None = None
        temperature: float = 0.2
        max_tokens: int | None = None
        # Reasoning-model control; None leaves the model's default.
        think: bool | None = None
        # A JSON schema for constrained decoding, where the backend supports it.
        response_format: dict | None = None
        # Caller opt-in: this request MAY be served by the cloud plane if eligible.
        allow_cloud: bool = False
        metadata: dict[str, str] = field(default_factory=dict)


class ModelReply(Protocol):
    """What ``generate`` returns: at least the completion text."""

    text: str


class ModelRouter(Protocol):
    """The two calls deid-kit makes of a model router."""

    async def embed(self, texts: Sequence[str], *, sensitivity: Sensitivity,
                    kind: str) -> list[list[float]]:
        """One vector per text."""
        ...

    async def generate(self, request: Any) -> ModelReply:
        """One completion for a :class:`ModelRequest`."""
        ...


__all__ = ["DOCUMENT", "QUERY", "ModelMessage", "ModelReply", "ModelRequest", "ModelRouter",
           "PrivacyViolation"]
