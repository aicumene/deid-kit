# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Microsoft Presidio, for deployments that want it — the ``presidio`` extra.

Nothing here is imported at module load: this module can be imported (and ``PII_ENTITIES`` read)
without Presidio installed, and :func:`analyzer` raises ``ImportError`` only when called. Both
places that use it treat that as intended: :class:`deidkit.detect.PresidioDetector` fails open
(logs and returns no detections), the gateway's default redactor fails closed (refuses the
crossing).

Install::

    pip install 'deid-kit[presidio]'
    python -m spacy download en_core_web_sm

A deployment with its own engine (more languages, custom recognisers) passes it instead:
``PresidioDetector(my_engine_factory)`` and ``PrivacyGateway(redact_fn=...)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

#: The entity set the gateway's default redactor asks for (English recognisers).
PII_ENTITIES: list[str] = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
    "US_DRIVER_LICENSE", "US_PASSPORT", "US_BANK_NUMBER", "IBAN_CODE",
    "LOCATION", "DATE_TIME", "MEDICAL_LICENSE", "IP_ADDRESS",
]

#: ``(language code, spaCy model)`` pairs the default engine loads.
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (("en", "en_core_web_sm"),)


@lru_cache
def analyzer(models: tuple[tuple[str, str], ...] = DEFAULT_MODELS) -> Any:
    """A Presidio ``AnalyzerEngine`` over spaCy ``models``, built once per process."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": code, "model_name": name} for code, name in models],
        }
    )
    return AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=[code for code, _ in models],
    )


__all__ = ["DEFAULT_MODELS", "PII_ENTITIES", "analyzer"]
