# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Data-sensitivity labels.

The four labels govern what may cross from the trusted plane to the cloud plane. Ordering
matters: higher value == more sensitive. Only data at or below ``max_crossable()`` (INTERNAL)
may ever leave the trusted plane, and only after de-identification.

They are the same labels, with the same values, as llm-kit's (``llmkit.classification``). When
llm-kit is installed this module IS that one's contents — the objects are imported from it — so
a deployment that uses both packages has one ``Sensitivity`` class for the router and for the
gateway. Without llm-kit the definitions below are used; llm-kit compares labels by value
(``Sensitivity(x)``), so a label from either side is understood by the other.

Dependency-free (stdlib only).
"""

from __future__ import annotations

from enum import IntEnum

try:
    from llmkit.classification import Sensitivity, max_crossable, may_cross_to_cloud
except ImportError:

    class Sensitivity(IntEnum):  # type: ignore[no-redef]
        """Sensitivity label assigned to every piece of data."""

        PUBLIC = 0        # already public
        INTERNAL = 1      # non-public but non-confidential (de-identified facts, generic questions)
        CONFIDENTIAL = 2  # content of a scope, work product
        RESTRICTED = 3    # personal data, privileged communications, health/financial records

        @property
        def label(self) -> str:
            return self.name.capitalize()

    def max_crossable() -> Sensitivity:  # type: ignore[no-redef]
        """The highest sensitivity that may cross to the cloud plane (after de-identification)."""
        return Sensitivity.INTERNAL

    def may_cross_to_cloud(sensitivity: Sensitivity) -> bool:  # type: ignore[no-redef]
        """True iff data at this sensitivity is *eligible* to cross the gateway.

        Eligibility is necessary but not sufficient — the gateway still de-identifies and
        audits. CONFIDENTIAL / RESTRICTED data is never eligible.
        """
        return sensitivity <= max_crossable()


__all__ = ["Sensitivity", "max_crossable", "may_cross_to_cloud"]
