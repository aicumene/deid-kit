# SPDX-License-Identifier: AGPL-3.0-only
"""The review of a matter document (deidkit.agents.review): what is de-identified, where, the norms the
text refers to, and the agents' notes — kept only on quotes the text has, read back as they stand now."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiohttp.test_utils import TestClient, TestServer       # noqa: E402

from deidkit.agents import review                            # noqa: E402
from deidkit.agents.host import AgentChoice, AgentHost, HostConfig  # noqa: E402
from deidkit.patterns import RegexDetector                  # noqa: E402
from deidkit.proxy.server import Proxy, ProxyConfig         # noqa: E402
from deidkit.store import InMemoryTokenStore                # noqa: E402

FAKE = Path(__file__).with_name("fake_acp_agent.py")
LEASE = ("# Lease\n\n"
         "The Tenant may break the lease on 14 August 2025 on six months' written notice. {#sec_5_2}\n\n"
         "Notices go to 14 Harbour Lane, Bristol BS1 4QA, or to t.wren@example.com, phone +44 117 496 0321.\n"
         "Section 21 of the Housing Act 1988 and § 573c Abs. 1 BGB do not apply; Art. 6 DSGVO does.\n")


async def wait_for(pred, timeout=10.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


# ── what is de-identified ─────────────────────────────────────────────────────────────────────────
async def test_each_token_is_placed_on_the_words_it_stands_for():
    proxy = Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="m", detector=RegexDetector()))
    red = (await proxy.engine("m").tokenize_all([LEASE]))[0]
    spans = review.deid_spans(LEASE, red.text, red.surfaces or {}, red.mapping or {})
    placed = {LEASE[s["start"]:s["end"]]: s["type"] for s in spans}
    assert placed == {"14 Harbour Lane": "ADDRESS", "BS1 4QA": "ADDRESS", "t.wren@example.com": "EMAIL",
                      "+44 117 496 0321": "PHONE"}
    assert all(s["token"] in red.text for s in spans)


def test_a_place_the_walk_lost_does_not_cost_the_rest():
    original = "Write to ann@example.org or bob@example.org today."
    # the engine's text with a piece that is not the original's (as if normalised) before the first token
    tokenised = "Write  to EMAIL_1 or EMAIL_2 today."
    spans = review.deid_spans(original, tokenised, {"EMAIL_1": ["ann@example.org"], "EMAIL_2": ["bob@example.org"]},
                              {"EMAIL_1": "ann@example.org", "EMAIL_2": "bob@example.org"})
    assert [original[s["start"]:s["end"]] for s in spans] == ["bob@example.org"]


def test_a_document_s_own_token_like_word_is_not_de_identified():
    spans = review.deid_spans("See FORM_12 by mail.", "See FORM_12 by mail.", {}, {})
    assert spans == []


# ── norms ─────────────────────────────────────────────────────────────────────────────────────────
def test_norms_are_found_where_they_are():
    found = review.norm_mentions(LEASE)
    assert [(LEASE[n["start"]:n["end"]], n["citation"], n["jurisdiction"]) for n in found] == [
        ("§ 573c Abs. 1 BGB", "§ 573c BGB", "DE"), ("Art. 6 DSGVO", "Art. 6 DSGVO", "DE")]


# ── quotes ────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("quote", [
    "six months' written notice",                       # as it is
    "six months\u2019 written notice",                  # a model's apostrophe
    "six  months'\nwritten notice",                     # its white space
    "The Tenant may break ... six months' written notice",   # an ellipsis
])
def test_a_quote_is_found_as_models_write_it(quote):
    found = review.Folded.of(LEASE).find(quote)
    assert found is not None
    assert LEASE[found[0]:found[1]].endswith("written notice")


def test_a_markdown_escape_does_not_hide_a_quote():
    text = "5\\.2 The rent is \\*net\\* of VAT.\n"
    found = review.Folded.of(text).find("5.2 The rent is *net* of VAT.")
    assert found == (0, len(text) - 1)


def test_words_the_text_does_not_have_are_not_found():
    folded = review.Folded.of(LEASE)
    assert folded.find("The Landlord may break the lease") is None
    assert folded.find("...") is None and folded.find("  ") is None


def test_of_two_places_the_nearer_one():
    text = "notice here. " + "x" * 500 + " notice there."
    folded = review.Folded.of(text)
    assert folded.find("notice") == (0, 6)
    assert folded.find("notice", near=480)[0] == text.rindex("notice")


def test_the_clause_of_a_place():
    text = "# Lease\n\n## 5.2 Break {#sec_5_2}\n\nThe Tenant may break.\n\n## 6 Rent {#sec_6}\n\nRent is due.\n"
    assert review.clause_at(text, text.index("Tenant")) == "sec_5_2"
    assert review.clause_at(text, text.index("Rent is")) == "sec_6"
    assert review.clause_at(text, 0) is None
    assert review.clause_at("A brace \\{#not_one}\nx", 21) is None


# ── the agent's answer ────────────────────────────────────────────────────────────────────────────
def test_notes_are_read_from_whatever_the_agent_said_around_them():
    answer = 'I read it {first}. ```json\n{"notes": [{"quote": "q", "kind": "Risk", "aspect": "nonsense", ' \
             '"importance": 9, "note": "  two\n words "}]}\n``` Done.'
    [note] = review.parse_notes(answer)
    assert note["kind"] == "risk" and note["aspect"] == "other" and note["importance"] == 5
    assert note["note"] == "two words" and note["clause_type"] == ""
    assert review.parse_notes("No notes today.") is None
    assert review.parse_notes('{"notes": "none"}') is None


# ── the text, and what is kept ────────────────────────────────────────────────────────────────────
def test_the_text_of_a_document(tmp_path):
    (tmp_path / "lease.md").write_text(LEASE)
    (tmp_path / "Supply.docx").write_bytes(b"PK")
    assert review.read_text(tmp_path, "lease.md") == ("lease.md", LEASE)
    assert review.read_text(tmp_path, "Supply.docx") is None                # its text is not written yet
    (tmp_path / ".legix/text").mkdir(parents=True)
    (tmp_path / ".legix/text/Supply.docx.md").write_text("<!-- bernio-text/1 -->\nSupply.\n")
    assert review.read_text(tmp_path, "Supply.docx")[0] == ".legix/text/Supply.docx.md"
    (tmp_path / "Lease.pdf").write_bytes(b"%PDF-1.4")                       # a PDF: its text is written beside it too
    (tmp_path / ".legix/text/Lease.pdf.md").write_text("<!-- bernio-text/1 -->\nLease.\n")
    assert review.read_text(tmp_path, "Lease.pdf") == (".legix/text/Lease.pdf.md", "<!-- bernio-text/1 -->\nLease.\n")
    assert not review.writable("Lease.pdf")
    for path in ("../lease.md", "/etc/hosts", ".legix/text/Supply.docx.md", "missing.md", ""):
        assert review.read_text(tmp_path, path) is None, path


def test_a_kept_note_is_read_back_where_its_words_are_now(tmp_path):
    notes = review.parse_notes(json.dumps({"notes": [
        {"quote": "six months\u2019 written notice", "kind": "risk", "aspect": "termination", "importance": 4,
         "note": "Short."},
        {"quote": "never there", "kind": "point", "note": "x"}]}))
    kept = review.keep(tmp_path, "lease.md", LEASE, notes, creator={"type": "Software", "name": "Claude Code",
                                                                    "model": "claude-opus-5-5"}, run="r1")
    assert len(kept["kept"]) == 1 and kept["refused"] == [{"quote": "never there", "why": "not in the document"}]
    [f] = list((tmp_path / ".legix/markup/lease.md").glob("*.json"))
    ann = json.loads(f.read_text())
    assert ann["type"] == "Annotation" and ann["motivation"] == "assessing"
    assert {s["type"] for s in ann["target"]["selector"]} == {"TextQuoteSelector", "TextPositionSelector",
                                                              "FragmentSelector"}
    [note] = review.load(tmp_path, "lease.md", LEASE)
    assert note["status"] == "here" and LEASE[note["start"]:note["end"]] == "six months' written notice"
    assert note["clause"] == "sec_5_2" and note["creator"] == {"name": "Claude Code", "model": "claude-opus-5-5"}

    edited = "# Lease\n\nA new first line.\n\n" + LEASE[len("# Lease\n\n"):]
    [moved] = review.load(tmp_path, "lease.md", edited)
    assert moved["status"] == "moved" and edited[moved["start"]:moved["end"]] == "six months' written notice"
    [gone] = review.load(tmp_path, "lease.md", "# Lease\n\nRewritten entirely.\n")
    assert gone["status"] == "gone" and gone["start"] is None


def test_a_file_that_is_not_this_document_s_note_is_left_out(tmp_path):
    root = tmp_path / ".legix/markup/lease.md"
    root.mkdir(parents=True)
    (root / "a.json").write_text("not json")
    (root / "b.json").write_text(json.dumps({"target": {"source": "other.md"}}))
    (root / "c.json").write_text(json.dumps([1, 2]))
    assert review.load(tmp_path, "lease.md", LEASE) == []


# ── through the host ──────────────────────────────────────────────────────────────────────────────
async def test_a_second_reviewer_takes_the_document_and_hands_back(tmp_path):
    folder = tmp_path / "matter"
    folder.mkdir()
    (folder / "lease.md").write_text(LEASE)
    claude = AgentChoice("Claude Code", [sys.executable, str(FAKE), "Claude Code", "review"])
    ours = AgentChoice("MatterAgent", [sys.executable, str(FAKE), "MatterAgent", "review"])
    gone = AgentChoice("Codex", [sys.executable, str(FAKE), "Codex"], unavailable="not signed in")
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1", dev_login=True, agent_command=claude.command,
                     agents=[claude, ours, gone])
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1", detector=RegexDetector())))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    try:
        await wait_for(lambda: host.status == "ready")
        headers = {"x-deid-agent-token": host.cfg.token}
        post = lambda body: client.post("/deid-agent/api/review", json=body, headers=headers)  # noqa: E731
        assert (await post({"path": "lease.md", "agent": "Nobody"})).status == 400
        assert (await post({"path": "lease.md", "agent": "Codex"})).status == 409

        # Claude Code first, then MatterAgent as the second reviewer.
        for who in ("Claude Code", "MatterAgent"):
            r = await post({"path": "lease.md", "agent": who})
            assert r.status == 202, await r.text()
            run = (await r.json())["run"]
            await wait_for(lambda: any(e["kind"] == "review" and e.get("run") == run and e["state"] in ("done", "failed")
                                       for e in host.events))
            await wait_for(lambda: not host.busy and host.status == "ready")
            done = next(e for e in host.events if e["kind"] == "review" and e.get("run") == run and e["state"] != "started")
            assert done["state"] == "done" and done["agent"] == who and done["kept"] == 1, done
            # the agent at work is the one that was, after either review
            assert host.current.name == "Claude Code"
        notes = review.load(folder, "lease.md", LEASE)
        assert sorted(n["creator"]["name"] for n in notes) == ["Claude Code", "MatterAgent"]
        # the chat heard the hand-over both ways
        switched = [e["current"] for e in host.events if e["kind"] == "agents"]
        assert switched[-2:] == ["MatterAgent", "Claude Code"], switched
    finally:
        await client.close()


@pytest.fixture
async def rig(tmp_path):
    folder = tmp_path / "matter"
    folder.mkdir()
    (folder / "lease.md").write_text(LEASE)
    command = [sys.executable, str(FAKE), "Claude Code", "review"]
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1", dev_login=True, agent_command=command,
                     agents=[AgentChoice("Claude Code", command)])
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1", detector=RegexDetector())))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    await wait_for(lambda: host.status == "ready")
    yield host, client, folder
    await client.close()


async def test_the_review_of_a_document_through_the_host(rig):
    host, client, folder = rig
    headers = {"x-deid-agent-token": host.cfg.token}
    r = await client.get("/deid-agent/api/review?path=lease.md", headers=headers)
    view = await r.json()
    assert r.status == 200 and view["source"] == "lease.md" and view["notes"] == []
    assert {view["text"][s["start"]:s["end"]] for s in view["deid"]} >= {"t.wren@example.com", "14 Harbour Lane"}
    assert [n["citation"] for n in view["norms"]] == ["§ 573c BGB", "Art. 6 DSGVO"]

    r = await client.post("/deid-agent/api/review", json={"path": "lease.md"}, headers=headers)
    assert r.status == 202
    run = (await r.json())["run"]
    # a second one, or a task, waits for this one
    assert (await client.post("/deid-agent/api/review", json={"path": "lease.md"}, headers=headers)).status == 409
    await wait_for(lambda: any(e["kind"] == "review" and e["state"] == "done" for e in host.events))
    done = next(e for e in host.events if e["kind"] == "review" and e["state"] == "done")
    assert done["run"] == run and done["kept"] == 1 and done["agent"] == "Claude Code"
    assert done["refused"] == [{"quote": "Words this document never had.", "why": "not in the document"}]
    # the page says what was asked, not the whole prompt
    assert next(e for e in host.events if e["kind"] == "user")["text"] == "Review lease.md"

    view = await (await client.get("/deid-agent/api/review?path=lease.md", headers=headers)).json()
    [note] = view["notes"]
    assert note["status"] == "here" and note["kind"] == "risk" and note["importance"] == 4
    assert note["norm"] == "Housing Act 1988 s. 21" and note["who"] == "the Tenant"
    assert note["creator"]["name"] == "Claude Code" and note["creator"]["model"] == "Default"
    # the notes are the app's, not the agent's work
    state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
    assert state["added"] == [] and not any(p.startswith(".legix") for p in state["files"])


async def test_a_second_review_cannot_slip_in_before_the_first_starts(rig):
    host, client, folder = rig
    first = await host._review_start("lease.md")             # no await inside: the run has not started
    second = await host._review_start("lease.md")
    assert first.status == 202 and second.status == 409
    await wait_for(lambda: any(e["kind"] == "review" and e["state"] == "done" for e in host.events))
    assert sum(e["kind"] == "review" and e["state"] == "started" for e in host.events) == 1


def test_an_agent_without_a_name_is_called_what_it_is():
    assert AgentChoice("", ["/x/claude-acp"]).label == "Claude Code"
    assert AgentChoice("", ["python", "-m", "matteragent"]).label == "MatterAgent"
    assert AgentChoice("", ["node", "codex-acp"]).label == "Codex"
    assert AgentChoice("", ["some-agent"]).label == "Agent"
    assert AgentChoice("Our reviewer", ["/x/claude-acp"]).label == "Our reviewer"


async def test_a_document_without_its_text_is_said_so(rig):
    host, client, folder = rig
    (folder / "Supply.docx").write_bytes(b"PK")
    headers = {"x-deid-agent-token": host.cfg.token}
    for r in (await client.get("/deid-agent/api/review?path=Supply.docx", headers=headers),
              await client.post("/deid-agent/api/review", json={"path": "Supply.docx"}, headers=headers)):
        assert r.status == 404 and (await r.json())["source"] == ".legix/text/Supply.docx.md"
    assert (await client.get("/deid-agent/api/review?path=../x.md", headers=headers)).status == 404
    assert (await client.get("/deid-agent/api/review?path=lease.md")).status == 401
