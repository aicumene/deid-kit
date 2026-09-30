# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A short name a document defines for a party ("TW") is that party's token. MEASURED 29.09.2026
on a matter agent: the party's name crossed as a token and its two-letter short form crossed in
clear, 40 to 53 times per task; on 30.09.2026 a task that used the short form crossed with it
before the agent had read the document that defines it."""

import re
import sys

import pytest

from deidkit.patterns import RegexDetector
from deidkit.proxy.engine import ScopeEngine
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

TOKEN = re.compile(r"\b[A-Z]+_\d+\b")
PARTIES = (
    '(1) KESTREL VENTURES LLP, a limited liability partnership registered in England and Wales '
    '("Kestrel");\n'
    '(2) TOBIAS WREN of the address set out in Schedule 1 ("TW"); and\n'
    '(3) BRACKEN HOLDINGS LTD (the "Company").\n'
)


def seeds() -> InMemorySeedSource:
    s = InMemorySeedSource()
    s.add_entity("case-1", SeedEntity("individual", "TOBIAS WREN", role="Shareholder"))
    s.add_entity("case-1", SeedEntity("company", "KESTREL VENTURES LLP", role="Shareholder"))
    s.add_entity("case-1", SeedEntity("company", "BRACKEN HOLDINGS LTD", role="Company"))
    s.add_entity("case-1", SeedEntity("individual", "ANNA BECKER", role="Tenant"))
    return s


@pytest.fixture
def engine():
    return ScopeEngine(InMemoryTokenStore(), "case-1", seeds=seeds(), detector=RegexDetector())


async def tokenise(engine, *texts):
    return [r.text for r in await engine.tokenize_all(list(texts))]


async def test_the_defined_short_name_is_the_party_s_token(engine):
    (parties, body) = await tokenise(
        engine, PARTIES, "The issued shares are held 6,500 by Kestrel and 3,500 by TW. A quorum "
                         "needs a director appointed by TW.")
    assert "TW" not in body and "Kestrel" not in body
    person = re.search(r"\(2\) (PERSON_\d+) of", parties).group(1)
    assert re.search(rf'\("{person}"\)', parties)                     # the definition itself
    assert body.count(person) == 2
    org = re.search(r"\(1\) (ORG_\d+), a limited", parties).group(1)
    assert f"6,500 by {org}" in body


async def test_a_short_name_learnt_once_is_a_token_in_every_later_text(engine):
    await tokenise(engine, PARTIES)
    (later,) = await tokenise(engine, "- 3.2 A quorum is three directors, one appointed by TW.")
    assert re.fullmatch(r"- 3\.2 A quorum is three directors, one appointed by PERSON_\d+\.", later)


async def test_a_short_name_is_matched_as_written(engine):
    """Folded, a two-letter name fires on words: "ab" is German for "from"."""
    (_, text) = await tokenise(
        engine, 'ANNA BECKER, Musterweg 12 ("AB"), und TOBIAS WREN ("TW").',
        "Die Frist läuft ab 1. Mai; AB zahlt. Tw and tw stay, TWO and TWP stay, TW goes.")
    assert text.startswith("Die Frist läuft ab 1. Mai; PERSON_")
    assert "Tw and tw stay, TWO and TWP stay, PERSON_" in text
    assert "AB" not in text and not re.search(r"\bTW\b", text)


async def test_a_term_that_is_not_the_party_s_short_name_stays_in_clear(engine):
    """ "the Company", "the Sale Shares" name a role or a thing, not who the party is."""
    (_, text) = await tokenise(
        engine, PARTIES + 'The shares held by TOBIAS WREN (the "Sale Shares") are offered.',
        "The Company shall not transfer the Sale Shares.")
    assert text == "The Company shall not transfer the Sale Shares."


async def test_a_definition_with_no_party_before_it_is_not_learnt(engine):
    """Control: the short name must follow a party the vault knows, on the same line."""
    (_, text) = await tokenise(engine, 'The address set out in Schedule 1 ("TW").', "Signed by TW.")
    assert text == "Signed by TW."


async def test_the_return_leg_gives_the_short_name_back(engine):
    await tokenise(engine, PARTIES)
    reds = await engine.tokenize_all(["Appointed by TW."])
    assert re.fullmatch(r"Appointed by PERSON_\d+\.", reds[0].text)       # it crossed as a token
    mapping = await engine.mapping_for(reds, reds[0].text)
    assert await engine.detokenize(reds[0].text, mapping) == "Appointed by TW."


async def test_one_request_hides_a_short_name_used_before_its_definition(engine):
    """The task comes first in a request, the document that defines the name later."""
    (task, _) = await tokenise(engine, "Draft a note on a sale of shares by TW.", PARTIES)
    assert re.fullmatch(r"Draft a note on a sale of shares by PERSON_\d+\.", task)


async def test_the_agent_learns_the_folder_s_short_names_before_the_first_task(tmp_path):
    pytest.importorskip("aiohttp")
    from deidkit.agents.host import AgentHost, HostConfig
    from deidkit.proxy.server import Proxy, ProxyConfig

    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "01-agreement.md").write_text("## PARTIES\n\n" + PARTIES)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "notes.md").write_text('ZOE QUILL ("ZQ")')      # hidden: not read
    proxy = Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1", seeds=seeds(),
                                                    detector=RegexDetector()))
    host = AgentHost(HostConfig(folder=tmp_path, scope="case-1", title="Case 1",
                                agent_command=[sys.executable, "-c", "pass"], dev_login=True), proxy)

    assert await host.warm_vault() == 1
    (task,) = [r.text for r in await proxy.engine("case-1").tokenize_all(
        ["Draft a note on a sale of shares by TW."])]
    assert re.fullmatch(r"Draft a note on a sale of shares by PERSON_\d+\.", task)
