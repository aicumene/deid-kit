# SPDX-License-Identifier: AGPL-3.0-only
"""The person's side of the review (deidkit.agents.review): changes proposed as Word's tracked changes and
accepted or rejected, decisions and feedback on the agents' notes, the person's own writing in the editor,
who wrote each document, and the records of all that for the harness and for learning — tokenized."""
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
         "Notices go to t.wren@example.com.\n")


async def wait_for(pred, timeout=10.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
async def rig(tmp_path):
    folder = tmp_path / "matter"
    folder.mkdir()
    (folder / "lease.md").write_text(LEASE)
    command = [sys.executable, str(FAKE), "Claude Code", "review"]
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1", dev_login=True, agent_command=command,
                     agents=[AgentChoice("Claude Code", command)], person="ada")
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1", detector=RegexDetector())))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    await wait_for(lambda: host.status == "ready")
    headers = {"x-deid-agent-token": host.cfg.token}

    async def call(name, body=None, method="POST"):
        r = await (client.post(f"/deid-agent/api/{name}", json=body, headers=headers) if method == "POST"
                   else client.get(f"/deid-agent/api/{name}", params=body, headers=headers))
        return r.status, await r.json()
    yield host, call, folder
    await client.close()


def cp(text, s):
    return text.index(s)


# ── tracked changes ───────────────────────────────────────────────────────────────────────────────
async def test_a_proposed_change_changes_nothing_until_it_is_accepted(rig):
    host, call, folder = rig
    at = cp(LEASE, "six months'")
    status, made = await call("review-suggest", {"path": "lease.md", "start": at, "end": at + 10, "text": "three months"})
    assert status == 200 and (folder / "lease.md").read_text() == LEASE
    _, view = await call("review", {"path": "lease.md"}, "GET")
    [edit] = view["notes"]
    assert edit["kind"] == "edit" and edit["by_person"] and edit["creator"]["name"] == "ada"
    assert edit["quote"] == "six months" and edit["suggestion"] == "three months" and edit["state"] == ""
    status, done = await call("review-decide", {"path": "lease.md", "target": made["id"], "verdict": "accepted"})
    assert status == 200 and done["applied"] is True
    assert "three months' written notice" in (folder / "lease.md").read_text()
    _, view = await call("review", {"path": "lease.md"}, "GET")
    [edit] = view["notes"]
    assert edit["state"] == "accepted" and edit["decisions"][0]["by"] == "ada" and edit["status"] == "gone"


async def test_an_insertion_and_a_deletion(rig):
    host, call, folder = rig
    at = cp(LEASE, "written notice")
    _, ins = await call("review-suggest", {"path": "lease.md", "start": at, "end": at, "text": "prior "})
    gone = cp(LEASE, " on 14 August 2025")
    _, dele = await call("review-suggest", {"path": "lease.md", "start": gone, "end": gone + len(" on 14 August 2025"),
                                           "text": ""})
    # the deletion first: the insertion's place moves, and it is found by the words around it
    assert (await call("review-decide", {"path": "lease.md", "target": dele["id"], "verdict": "accepted"}))[1]["applied"]
    assert (await call("review-decide", {"path": "lease.md", "target": ins["id"], "verdict": "accepted"}))[1]["applied"]
    assert "may break the lease on six months' prior written notice" in (folder / "lease.md").read_text()


async def test_a_rejected_change_and_one_whose_words_are_gone(rig):
    host, call, folder = rig
    at = cp(LEASE, "Notices go")
    _, a = await call("review-suggest", {"path": "lease.md", "start": at, "end": at + 7, "text": "Letters"})
    _, b = await call("review-suggest", {"path": "lease.md", "start": at, "end": at + 7, "text": "Mail"})
    assert (await call("review-decide", {"path": "lease.md", "target": a["id"], "verdict": "rejected",
                                         "comment": "keep the defined word"}))[1]["applied"] is False
    (folder / "lease.md").write_text(LEASE.replace("Notices go", "Letters go"))
    status, body = await call("review-decide", {"path": "lease.md", "target": b["id"], "verdict": "accepted"})
    assert status == 409 and "no longer" in body["error"]
    _, view = await call("review", {"path": "lease.md"}, "GET")
    rejected = next(n for n in view["notes"] if n["id"] == a["id"])
    assert rejected["state"] == "rejected" and rejected["decisions"][0]["comment"] == "keep the defined word"


async def test_nothing_to_propose_and_nothing_to_decide(rig):
    host, call, folder = rig
    for body in ({"path": "lease.md", "start": 5, "end": 5, "text": ""},          # inserts nothing
                 {"path": "lease.md", "start": 2, "end": 7, "text": "Lease"},      # the same words
                 {"path": "lease.md", "start": 50, "end": 10, "text": "x"},
                 {"path": "lease.md", "start": 0, "end": 10_000, "text": "x"}):
        assert (await call("review-suggest", body))[0] == 400, body
    assert (await call("review-suggest", {"path": "lease.md", "start": 2, "end": 7, "text": "Let", "sha": "0" * 64}))[0] == 409
    assert (await call("review-decide", {"path": "lease.md", "target": "ab" * 16, "verdict": "accepted"}))[0] == 404
    assert (await call("review-decide", {"path": "lease.md", "target": "../x", "verdict": "accepted"}))[0] == 404
    assert (await call("review-decide", {"path": "lease.md", "target": "ab" * 16, "verdict": "maybe"}))[0] == 400


async def test_a_word_document_keeps_the_decision_and_is_changed_in_word(rig):
    host, call, folder = rig
    (folder / "Supply.docx").write_bytes(b"PK")
    (folder / ".legix/text").mkdir(parents=True)
    (folder / ".legix/text/Supply.docx.md").write_text("Delivery within fourteen days.\n")
    _, s = await call("review-suggest", {"path": "Supply.docx", "start": 16, "end": 24, "text": "ten"})
    status, done = await call("review-decide", {"path": "Supply.docx", "target": s["id"], "verdict": "accepted"})
    assert status == 200 and done["applied"] is False and "Word" in done["why"]
    assert (folder / ".legix/text/Supply.docx.md").read_text() == "Delivery within fourteen days.\n"
    assert (folder / "Supply.docx").read_bytes() == b"PK"


# ── the agents' notes: accepted wording, and feedback ─────────────────────────────────────────────
async def test_an_agent_s_wording_accepted_and_its_note_judged(rig):
    host, call, folder = rig
    assert (await call("review", {"path": "lease.md"}))[0] == 202
    await wait_for(lambda: any(e["kind"] == "review" and e.get("state") == "done" for e in host.events))
    _, view = await call("review", {"path": "lease.md"}, "GET")
    [note] = view["notes"]
    assert note["suggestion"] == "The Tenant may not break the lease" and not note["by_person"]
    status, done = await call("review-decide", {"path": "lease.md", "target": note["id"], "verdict": "accepted"})
    assert done["applied"] and (folder / "lease.md").read_text().count("The Tenant may not break the lease") == 1
    await call("review-decide", {"path": "lease.md", "target": note["id"], "verdict": "wrong",
                                 "comment": "the break is mutual", "correction": "Either party may break the lease"})
    _, view = await call("review", {"path": "lease.md"}, "GET")
    [note] = view["notes"]
    assert [d["verdict"] for d in note["decisions"]] == ["accepted", "wrong"] and note["state"] == "accepted"
    assert note["decisions"][1]["correction"] == "Either party may break the lease"


# ── the editor ────────────────────────────────────────────────────────────────────────────────────
async def test_the_person_writes_over_the_text_it_was_opened_with(rig):
    host, call, folder = rig
    _, view = await call("review", {"path": "lease.md"}, "GET")
    edited = view["text"].replace("six months'", "three months'") + "\nSigned in Hamburg.\n"
    status, saved = await call("review-text", {"path": "lease.md", "text": edited, "sha": view["sha"]})
    assert status == 200 and (folder / "lease.md").read_text() == edited
    # again over the old text: someone wrote in between
    assert (await call("review-text", {"path": "lease.md", "text": "x", "sha": view["sha"]}))[0] == 409
    _, view = await call("review", {"path": "lease.md"}, "GET")
    assert view["sha"] == saved["sha"] and view["revisions"][0]["by"]["name"] == "ada"
    [ann] = [a for a in review.annotations(folder, "lease.md") if review._kind(a) == "revision"]
    assert {(h["old"], h["new"]) for h in ann["bernio"]["hunks"]} >= {("six", "three")}


async def test_a_new_document_from_what_an_agent_wrote(rig):
    host, call, folder = rig
    body = {"path": "Drafts/Memo.md", "text": "# Memo\n\nThe break is mutual.\n", "sha": "",
            "written_by": {"agent": "Claude Code", "model": "claude-opus-5-5"}}
    assert (await call("review-text", body))[0] == 200
    assert (folder / "Drafts/Memo.md").read_text().startswith("# Memo")
    _, view = await call("review", {"path": "Drafts/Memo.md"}, "GET")
    assert [(w["by"]["name"], w["by"]["model"], w["how"]) for w in view["written"]] == [
        ("Claude Code", "claude-opus-5-5", "wrote"), ("ada", "", "saved")]
    for path in ("Drafts/Memo.md", "../out.md", ".legix/x.md", "Memo.docx", "Drafts/.hidden.md", "/tmp/x.md"):
        assert (await call("review-text", {**body, "path": path}))[0] == 400, path


async def test_what_the_agent_writes_in_a_task_is_noted_as_its(rig):
    host, call, folder = rig
    host.cfg.agents[0].command = [sys.executable, str(FAKE), "Claude Code"]     # the plain fake: writes draft.md
    task = asyncio.create_task(host.run_prompt("Draft a reply"))
    await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
    ask = next(e for e in host.events if e["kind"] == "permission")
    host.permissions[ask["id"]].set_result({"outcome": "selected", "optionId": "allow"})
    await task                                       # its write outside the folder is refused unasked
    [w] = review.view(folder, "draft.md", (folder / "draft.md").read_text())["written"]
    assert w["by"]["name"] == "Claude Code" and w["how"] == "added"
    assert review.view(folder, "lease.md", LEASE)["written"] == []                 # it did not touch the lease


# ── the records, for the harness and for learning ─────────────────────────────────────────────────
async def test_the_records_go_out_tokenized(rig):
    host, call, folder = rig
    (folder / "draft.md").write_text("Write to the landlord.\n")
    review.store(folder, "draft.md", review.written(path="draft.md", how="added",
                                                    creator={"type": "Software", "name": "Claude Code", "model": "m"}))
    _, view = await call("review", {"path": "draft.md"}, "GET")
    await call("review-text", {"path": "draft.md", "text": "Write to t.wren@example.com.\n", "sha": view["sha"]})
    at = cp(LEASE, "Notices")
    _, s = await call("review-suggest", {"path": "lease.md", "start": at, "end": at + 7, "text": "Letters"})
    await call("review-decide", {"path": "lease.md", "target": s["id"], "verdict": "rejected",
                                 "comment": "ask t.wren@example.com first"})
    status, out = await call("review-dataset", {}, "GET")
    assert status == 200 and out["count"] == 2
    dumped = json.dumps(out["records"])
    assert "t.wren@example.com" not in dumped and "EMAIL_" in dumped
    kinds = {r["type"]: r for r in out["records"]}
    assert kinds["revision"]["written_by"] == {"agent": "Claude Code", "model": "m"}
    assert kinds["decision"]["about"]["by"] == "person" and kinds["decision"]["verdict"] == "rejected"
    assert "ada" not in dumped                                                    # the person is a role here


def test_a_person_s_own_document_gives_no_revision_record(tmp_path):
    (tmp_path / "notes.md").write_text("a\n")
    review.store(tmp_path, "notes.md", review.written(path="notes.md", how="wrote", creator=review.person("ada")))
    review.store(tmp_path, "notes.md", review.revision(path="notes.md", before="a\n", after="b\n",
                                                       creator=review.person("ada")))
    assert review.records(tmp_path) == []
