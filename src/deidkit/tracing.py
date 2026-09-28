# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Optional tracing hooks.

The gateway opens a span around every crossing (``privacy.cross_to_cloud``) whose attributes are
identifiers only — action, target, sensitivity label, then the redaction count and entity types —
never content. The package itself carries no tracing dependency: until hooks are installed,
:func:`start_span` is a no-op that yields ``None`` and :func:`set_attributes` does nothing.

A deployment installs them at process start, for example a facade over OpenTelemetry::

    from deidkit import tracing

    tracing.set_span_factory(start_span)          # start_span(name, **attrs) -> context manager
    tracing.set_attribute_setter(set_attributes)  # set_attributes(**attrs) on the current span

Both hooks are process-wide on purpose: gateways are built in several places, and a per-gateway
setting would leave the ones built elsewhere untraced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

#: ``factory(name, **attrs)`` → a context manager around the span (it may yield the span object).
SpanFactory = Callable[..., AbstractContextManager[Any]]
#: ``setter(**attrs)`` → sets attributes on the current span.
AttributeSetter = Callable[..., None]

_factory: SpanFactory | None = None
_setter: AttributeSetter | None = None


def set_span_factory(factory: SpanFactory | None) -> None:
    """Install the process-wide span factory; ``None`` removes it."""
    global _factory
    _factory = factory


def set_attribute_setter(setter: AttributeSetter | None) -> None:
    """Install the process-wide attribute setter; ``None`` removes it."""
    global _setter
    _setter = setter


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[Any]:
    """Open a span through the installed factory; a no-op yielding ``None`` without one."""
    factory = _factory
    if factory is None:
        yield None
        return
    with factory(name, **attrs) as span:
        yield span


def set_attributes(**attrs: Any) -> None:
    """Set attributes on the current span through the installed setter; a no-op without one."""
    setter = _setter
    if setter is not None:
        setter(**attrs)


__all__ = ["AttributeSetter", "SpanFactory", "set_attribute_setter", "set_attributes",
           "set_span_factory", "start_span"]
