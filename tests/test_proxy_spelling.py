# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""File names, paths and copied lines come back as they were written, not in the vault's canonical
spelling. MEASURED 29.09.2026: an agent could not open a matter file whose name held the client's
short name; the call to read it came back with the canonical name in its place."""

import json

import pytest

from deidkit.gateway import Redaction
from deidkit.patterns import RegexDetector
from deidkit.proxy.engine import ScopeEngine
from deidkit.proxy.openai import literal_mapping
from deidkit.proxy.spelling import respell, spellings
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

HEADING = "<cwd>/work/Brenner</cwd> Matter: HARROWGATE FREIGHT LTD, claim by Ada Brenner"
LISTING = "letters/03-Reply-to-Harrowgate-Freight.md\nletters/04 Brenner, Ada - note.md"
LETTER = "     1→Dear Ms Brenner,\n     2→Harrowgate asks for an extension until 30 June.\n"
SIGNED = "Signed,\nAda Brenner\n"


@pytest.fixture
async def request_seen():
    """One request: a heading with the working folder, a listing, a file as a viewer shows it."""
    seeds = InMemorySeedSource()
    seeds.add_entity("case-1", SeedEntity("company", "Harrowgate Freight Ltd"))
    seeds.add_entity("case-1", SeedEntity("individual", "Ada Brenner", role="Director"))
    eng = ScopeEngine(InMemoryTokenStore(), "case-1", seeds=seeds, detector=RegexDetector())
    pieces = [HEADING, LISTING, LETTER, SIGNED]
    reds = await eng.tokenize_all(pieces)
    mapping = await eng.mapping_for(reds, "\n".join(r.text for r in reds))
    person = next(t for t in mapping if t.startswith("PERSON"))
    org = next(t for t in mapping if t.startswith("ORG"))
    return eng, mapping, spellings(pieces, reds), [r.text for r in reds], person, org


async def test_a_path_comes_back_as_it_is_on_disk(request_seen):
    eng, mapping, table, sent, person, _ = request_seen
    assert not any(n in sent[1] for n in ("Harrowgate", "Brenner"))        # it crossed in tokens
    for line, real in zip(sent[1].splitlines(), LISTING.splitlines()):
        path = f"/work/{person}/{line}"
        assert await eng.detokenize(path, mapping, table) == f"/work/Brenner/{real}"
        # the defect: without the table the vault's canonical spelling lands in the path
        assert await eng.detokenize(path, mapping) != f"/work/Brenner/{real}"


async def test_a_command_gets_its_file_and_folder_names_back(request_seen):
    eng, mapping, table, sent, person, _ = request_seen
    glued, spaced = sent[1].splitlines()
    cases = {
        f"cd /work/{person} && ls": "cd /work/Brenner && ls",
        f"cd {person}/letters": "cd Brenner/letters",
        f'cat "{spaced}" | head -5': 'cat "letters/04 Brenner, Ada - note.md" | head -5',
        f"wc -c {glued.split('/')[1]} notes.md": "wc -c 03-Reply-to-Harrowgate-Freight.md notes.md",
    }
    for command, real in cases.items():
        assert await eng.detokenize(command, mapping, table) == real


async def test_an_edit_finds_the_line_it_copied(request_seen):
    eng, mapping, table, sent, _, _ = request_seen
    line = sent[2].split("→")[2].strip()                             # as the model read it
    assert await eng.detokenize(line, mapping, table) == \
        "Harrowgate asks for an extension until 30 June."


async def test_a_patch_in_a_script_keeps_its_string_literal(request_seen):
    eng, mapping, table, sent, _, _ = request_seen
    line = sent[2].split("→")[2].strip()
    patch = (f"*** Update File: {sent[1].splitlines()[0]}\n@@\n-{line}\n"
             f"+{line.replace('30 June', '31 July')}\n")
    script = f"text(await tools.apply_patch({json.dumps(patch)}));"
    back = await eng.detokenize(script, literal_mapping(mapping), table, literal=True)
    restored = json.loads(back[back.index("(", back.index("apply_patch")) + 1:back.rindex("));")])
    assert restored.splitlines() == [
        "*** Update File: letters/03-Reply-to-Harrowgate-Freight.md",
        "@@",
        "-Harrowgate asks for an extension until 30 June.",           # copied: as in the file
        "+Harrowgate Freight Ltd asks for an extension until 31 July.",   # new: the usual value
    ]


async def test_prose_and_new_names_keep_the_usual_value(request_seen):
    eng, mapping, table, _, person, org = request_seen
    for text in (f"{org} asks for more time.", f"Dear {person},", f"Signed,\n{person}\n",
                 f"drafts/{org}-summary.md"):
        assert await eng.detokenize(text, mapping, table) == await eng.detokenize(text, mapping)


def test_a_remembered_name_is_escaped_for_a_literal_and_keeps_its_punctuation():
    table = {"ORG_1-a.md": 'Say "hi"-a.md'}
    assert respell('open("ORG_1-a.md")', table, literal=True) == 'open("Say \\"hi\\"-a.md")'
    assert respell("see ORG_1-a.md.", table) == 'see Say "hi"-a.md.'
    assert respell("ORG_1-b.md", table) == "ORG_1-b.md"                # not seen: left for the mapping


def test_a_token_already_in_the_text_is_passed_over_and_a_mismatch_is_dropped():
    red = Redaction(text="ISO_9001 file ORG_1-Freight.md", surfaces={"ORG_1": ["Harrowgate"]})
    table = spellings(["ISO_9001 file Harrowgate-Freight.md"], [red])
    assert table["ORG_1-Freight.md"] == "Harrowgate-Freight.md"
    wrong = Redaction(text="ORG_1-Freight.md", surfaces={"ORG_1": ["Elsewhere"]})
    assert spellings(["Harrowgate-Freight.md"], [wrong]) == {}
