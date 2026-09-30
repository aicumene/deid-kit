# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""What a matter agent's requests carried in clear on 29–30.09.2026, with every known name already
a token: a command reaching into other matters, the machine's account name in every absolute
path, and the tenant's street address."""

import re
import sys
from pathlib import Path

import pytest

from deidkit.patterns import RegexDetector
from deidkit.proxy.engine import ScopeEngine
from deidkit.proxy.spelling import spellings
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

pytest.importorskip("aiohttp")
from deidkit.agents.cli import account_name                         # noqa: E402
from deidkit.agents.host import AgentHost, HostConfig, outside_paths  # noqa: E402
from deidkit.proxy.server import Proxy, ProxyConfig                  # noqa: E402


@pytest.fixture
def matters(tmp_path, monkeypatch):
    """Two matters side by side, a file of known names beside them, and a home of our own."""
    home = tmp_path / "home"
    (home / "notes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "matters"
    for m in ("A-1", "B-2"):
        (root / m / "documents").mkdir(parents=True)
        (root / m / "documents" / "01-letter.md").write_text("A letter.\n")
    (tmp_path / "seeds").mkdir()
    return root / "A-1", root, home


# ── a command that names a path outside the matter folder ────────────────────────────────────

@pytest.mark.parametrize("command", [
    'grep -rni "clause 4" {root}/ 2>/dev/null | head -20',   # the one a matter agent asked for
    "cat ../B-2/documents/01-letter.md",
    "ls ~/notes",
    "cd $HOME/notes && ls",
    "cp drafts/a.md {tmp}/a.md",
])
def test_a_command_naming_a_path_outside_the_folder_is_caught(matters, command, tmp_path):
    folder, root, _ = matters
    assert outside_paths(command.format(root=root, tmp=tmp_path), folder)


@pytest.mark.parametrize("command", [
    'grep -rni "clause 4" {folder}/documents/ 2>/dev/null; echo "--- exit: $?"',
    'find . -type f -not -path "*/.git/*" | head -100',
    'grep -n "/api/v1" documents/*.md',                      # a pattern, not a path
    "/usr/bin/wc -l documents/*.md > /dev/null",
    "sed -n 1,40p documents/01-letter.md",
])
def test_a_command_inside_the_folder_is_left_to_the_person(matters, command):
    """Control: the check must not refuse the agent's ordinary work."""
    folder, _, _ = matters
    assert outside_paths(command.format(folder=folder), folder) == []


async def test_the_host_refuses_such_a_command_without_asking(matters):
    folder, root, _ = matters
    host = AgentHost(HostConfig(folder=folder, scope="a-1", title="A-1",
                                agent_command=[sys.executable, "-c", "pass"], dev_login=True),
                     Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="a-1")))
    command = f'grep -rni "clause 4" {root}/ 2>/dev/null'
    answer = await host.on_permission({
        # a permission request for a command carries no locations
        "toolCall": {"kind": "execute", "title": command, "rawInput": {"command": command},
                     "locations": []},
        "options": [{"optionId": "allow-once", "name": "Yes", "kind": "allow_once"},
                    {"optionId": "reject", "name": "No", "kind": "reject_once"}]})
    assert answer == {"outcome": "selected", "optionId": "reject"}
    assert not host.permissions and not any(e["kind"] == "permission" for e in host.events)
    assert any(e["kind"] == "notice" and "outside the matter folder" in e["text"] for e in host.events)


# ── the machine's account name in absolute paths ─────────────────────────────────────────────

def test_the_account_is_named_by_its_home_folder_unless_the_name_is_generic():
    assert account_name(Path("/Users/tobiaswren")) == "tobiaswren"
    assert account_name(Path("/Users/admin")) is None
    assert account_name(Path("/home/tw")) is None                          # too short to be a key


async def test_absolute_paths_cross_without_the_account_and_come_back_as_on_disk():
    seeds = InMemorySeedSource()
    seeds.add_entity("a-1", SeedEntity("account", "tobiaswren"))
    eng = ScopeEngine(InMemoryTokenStore(), "a-1", seeds=seeds, detector=RegexDetector())
    texts = ["Primary working directory: /Users/tobiaswren/Work/matters/A-1",
             "memory at `/Users/tobiaswren/.claude/projects/-Users-tobiaswren-Work-matters-A-1/memory/`"]
    reds = await eng.tokenize_all(texts)
    assert not any("tobiaswren" in r.text for r in reds)
    tok = re.search(r"ACCOUNT_\d+", reds[0].text).group(0)
    assert f"-Users-{tok}-Work-matters-A-1" in reds[1].text
    mapping = await eng.mapping_for(reds, "\n".join(r.text for r in reds))
    back = await eng.detokenize(f"/Users/{tok}/Work/matters/A-1/documents/01-letter.md", mapping,
                                spellings(texts, reds))
    assert back == "/Users/tobiaswren/Work/matters/A-1/documents/01-letter.md"


# ── street addresses through the proxy ───────────────────────────────────────────────────────

@pytest.fixture
def tenancy():
    seeds = InMemorySeedSource()
    seeds.add_entity("nor", SeedEntity("individual", "Jonas Brandt", role="Tenant"))
    return ScopeEngine(InMemoryTokenStore(), "nor", seeds=seeds, detector=RegexDetector())


async def test_a_street_address_crosses_as_a_token_and_the_city_stays(tenancy):
    (red,) = await tenancy.tokenize_all(
        ["Jonas Brandt, Musterweg 12, 2. OG links, 22303 Hamburg, kündigt zum 31. Mai 2026."])
    assert "Musterweg" not in red.text and "22303" not in red.text
    assert re.fullmatch(r"PERSON_\d+, ADDRESS_\d+, 2\. OG links, ADDRESS_\d+ Hamburg, "
                        r"kündigt zum 31\. Mai 2026\.", red.text)
    mapping = await tenancy.mapping_for([red], red.text)
    assert await tenancy.detokenize(red.text, mapping) == \
        "Jonas Brandt, Musterweg 12, 2. OG links, 22303 Hamburg, kündigt zum 31. Mai 2026."


async def test_one_address_is_one_token_across_requests_and_in_a_file_name(tenancy):
    texts = ["documents/01-Mietvertrag-Musterweg-12-Kuendigung.md", "die Wohnung Musterweg 12"]
    first = await tenancy.tokenize_all(texts)
    (later,) = await tenancy.tokenize_all(["Mietvertrag Musterweg 12, Anlage 2"])
    assert "Musterweg" not in first[0].text and "Musterweg" not in later.text
    token = re.search(r"ADDRESS_\d+", first[1].text).group(0)
    assert token in later.text
    # a call to read the file names it the way the listing did, and opens it on disk
    mapping = await tenancy.mapping_for(first, "\n".join(r.text for r in first))
    assert await tenancy.detokenize(first[0].text, mapping, spellings(texts, first)) == texts[0]


async def test_a_date_and_an_amount_are_not_addresses(tenancy):
    """Control: places, dates and amounts stay readable."""
    text = "Frist ab 1. Mai 2026; Miete 1250 Euro; Hamburg-Barmbek."
    (red,) = await tenancy.tokenize_all([text])
    assert red.text == text
