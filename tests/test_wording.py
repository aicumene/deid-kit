# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The words the package puts in front of a model are neutral by default and set by a deployment.

A deployment in a particular field (a law firm, a clinic) sets its own words once — the glossary's
source and scope, the re-identification judge's brief — and gets exactly the text it had before.
"""

from __future__ import annotations

import pytest

from deidkit import glossary, reid


@pytest.fixture(autouse=True)
def _restore_defaults():
    yield
    glossary.set_wording(glossary.Wording())
    reid.set_judge_wording(reid.JudgeWording())


def test_the_glossary_header_is_neutral_by_default():
    block = glossary.render(["PERSON_1 = individual (natural person)"])
    assert "derived from the source documents" in block
    assert "case file" not in block and "matter" not in block


def test_a_deployment_sets_the_glossary_words_once():
    glossary.set_wording(glossary.Wording(source="the case file", scope="this matter"))
    block = glossary.render(["PERSON_1 = individual (natural person)"])
    assert "derived from the case file, contain no identifying detail" in block
    assert glossary.wording().scope == "this matter"


class _Router:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)

        class R:  # noqa: D401
            text = '{"reference": "UNKNOWN", "clue": ""}'
        return R()


async def test_the_judge_is_briefed_neutrally_by_default():
    router = _Router()
    await reid.probe_judge(router, "excerpt", [reid.Candidate("A", "first record")])
    system, user = (m.content for m in router.requests[0].messages)
    assert "lawyer" not in system and "matter" not in system
    assert user.startswith("Records:\n- A: first record")


async def test_a_deployment_sets_the_judges_brief():
    reid.set_judge_wording(reid.JudgeWording(system="You are a clinician.", candidates="Patient files"))
    router = _Router()
    await reid.probe_judge(router, "excerpt", [reid.Candidate("A", "first file")])
    system, user = (m.content for m in router.requests[0].messages)
    assert system == "You are a clinician."
    assert user.startswith("Patient files:\n- A: first file")
