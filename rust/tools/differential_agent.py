# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: the Python deid-agent and the Rust one, side by side, driven the same way.

    RT=…/agent-runtime/python/bin/python3.13
    $RT rust/tools/differential_agent.py [--rust rust/target/release/deid-agent] [-v]

Each run gets the same folder (same path), a vault made beforehand with one salt, the same seed file and a
home of its own; the agents are tests/fake_acp_agent.py, the model matteragent's tests/fake_model.py. The
script is a person at the page: tasks, permissions, model and agent switches, a hand-over, a cancel, a
review by the agent at work and by a second reviewer, proposed changes and decisions, the person's own
writing, votes, the dataset, files, errors. Compared: every API answer, the event stream (read over
HTTP/1.1 and, as the desktop reads it, HTTP/1.0), the folder's files with the markup, the usage log, the
audit and record, what the model got — random ids, times and temp names normalised — and that no agent
process is left once deid-agent is stopped.
"""
import argparse
import asyncio
import itertools
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
DEID = HERE.parents[2]
sys.path.insert(0, str(DEID.parent / "matteragent" / "tests"))
sys.path.insert(0, str(DEID.parent / "matteragent" / "src"))
import fake_model  # noqa: E402
from fake_model import FakeModel  # noqa: E402

FAKE = DEID / "tests" / "fake_acp_agent.py"
TOKEN = "tok-differential-1"
SALT = bytes(range(32))
LEASE = ("# Lease\n\n"
         "The Tenant may break the lease on 14 August 2025 on six months' written notice. {#sec_5_2}\n\n"
         "Notices go to 14 Harbour Lane, Bristol BS1 4QA, or to t.wren@example.com, phone +44 117 496 0321.\n"
         "Section 21 of the Housing Act 1988 and § 573c Abs. 1 BGB do not apply; Art. 6 DSGVO does.\n")
SEED = """scope = "case-1"
[[entity]]
type = "individual"
name = "Ada Brenner"
[[entity]]
type = "company"
name = "Harrowgate Freight Ltd"
[[party]]
name = "Tobias Wren"
role = "Tenant"
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def http(port, method, path, body=None, token=TOKEN, raw=None):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    head = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
    if token:
        head.append(f"x-deid-agent-token: {token}")
    if method in ("POST", "PUT"):
        head += ["Content-Type: application/json", f"Content-Length: {len(data)}"]
    w.write(("\r\n".join(head) + "\r\n\r\n").encode() + data)
    await w.drain()
    status = int((await r.readline()).split()[1])
    hdrs = {}
    while (line := await r.readline()) not in (b"\r\n", b""):
        k, _, v = line.decode().partition(":")
        hdrs[k.strip().lower()] = v.strip()
    out = b""
    if "chunked" in hdrs.get("transfer-encoding", ""):
        while True:
            size = int((await r.readline()).strip() or b"0", 16)
            if size == 0:
                await r.readline()
                break
            out += await r.readexactly(size)
            await r.readexactly(2)
    elif "content-length" in hdrs:
        out = await r.readexactly(int(hdrs["content-length"]))
    else:
        out = await r.read()
    w.close()
    ctype = hdrs.get("content-type", "").split(";")[0]
    try:
        parsed = json.loads(out)
    except ValueError:
        parsed = out.decode("utf-8", "replace")
    return {"status": status, "type": ctype, "body": parsed}


class Stream:
    """The page's event stream, read as it comes."""

    def __init__(self, port, version="1.1"):
        self.events: list[dict] = []
        self.port, self.version = port, version
        self.task = asyncio.create_task(self.read())

    async def read(self):
        r, w = await asyncio.open_connection("127.0.0.1", self.port)
        w.write(f"GET /deid-agent/api/events HTTP/{self.version}\r\nHost: x\r\nx-deid-agent-token: {TOKEN}\r\n\r\n".encode())
        await w.drain()
        while (line := await r.readline()) not in (b"\r\n", b""):
            pass
        buf = b""
        chunked = self.version == "1.1"
        try:
            while True:
                if chunked:
                    size_line = await r.readline()
                    if not size_line:
                        return
                    size = int(size_line.strip() or b"0", 16)
                    if size == 0:
                        return
                    buf += await r.readexactly(size)
                    await r.readexactly(2)
                else:
                    piece = await r.read(65536)
                    if not piece:
                        return
                    buf += piece
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    for line in block.decode().split("\n"):
                        if line.startswith("data: "):
                            self.events.append(json.loads(line[6:]))
        except (ConnectionError, asyncio.IncompleteReadError):
            return

    def has(self, pred):
        return any(pred(e) for e in self.events)

    async def until(self, pred, timeout=20.0, after=0):
        for _ in range(int(timeout / 0.02)):
            if any(pred(e) for e in self.events[after:]):
                return next(e for e in self.events[after:] if pred(e))
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out waiting; last events: {[e.get('kind') for e in self.events[-8:]]}")


def kind(k, **fields):
    return lambda e: e.get("kind") == k and all(e.get(f) == v for f, v in fields.items())


def agent_cmd(py, fake, name, *flags):
    return f"{name}={py} {fake} {name.replace(' ', '')} {' '.join(flags)}".strip()


async def scenario(port, s: Stream, log, folder: Path, fake: FakeModel, multi: bool):
    """The person at the page. Every API answer goes to `log`."""
    async def call(what, method, path, body=None, **kw):
        a = await http(port, method, path, body, **kw)
        log.append([what, a])
        return a

    await call("no token", "GET", "/deid-agent/api/state", token=None)
    await call("token in the address", "GET", f"/deid-agent/api/state?t={TOKEN}", token=None)
    await s.until(kind("status", status="ready"))
    await call("state", "GET", "/deid-agent/api/state")
    await call("page", "GET", "/")
    await call("style", "GET", "/deid-agent/style.css")
    await call("health", "GET", "/deid/health", token=None)
    await call("unknown api", "GET", "/deid-agent/api/nothing")
    await call("empty task", "POST", "/deid-agent/api/prompt", {"text": "   "})
    await call("bad json", "POST", "/deid-agent/api/prompt", raw=b"{not json")
    await call("text not a string", "POST", "/deid-agent/api/prompt", {"text": 5})
    await call("no such permission", "POST", "/deid-agent/api/permission", {"id": "nope", "optionId": "allow"})

    # a task, a model answer counted into it, the person allowing the edit
    mark = len(s.events)
    await call("task 1", "POST", "/deid-agent/api/prompt", {"text": "Draft a reply to Ada Brenner of Harrowgate Freight Ltd"})
    ask = await s.until(kind("permission"), after=mark)
    await call("busy", "POST", "/deid-agent/api/prompt", {"text": "Another"})
    fake.script = [{"type": "message", "role": "assistant", "id": "msg_1", "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1200, "output_tokens": 80, "cache_read_input_tokens": 9000,
                              "cache_creation_input_tokens": 400}}]
    await call("model request", "POST", "/v1/messages", {
        "model": "claude-sonnet-5-5", "max_tokens": 10, "system": f"Primary working directory: {folder}",
        "messages": [{"role": "user", "content": "Write to Ada Brenner at ada@example.org"}]}, token=None)
    await call("allow", "POST", "/deid-agent/api/permission", {"id": ask["id"], "optionId": "allow"})
    await call("answered twice", "POST", "/deid-agent/api/permission", {"id": ask["id"], "optionId": "allow"})
    await s.until(kind("files"), after=mark)
    await call("state after task 1", "GET", "/deid-agent/api/state")
    await call("file", "GET", "/deid-agent/api/file?path=draft.md")
    await call("file outside", "GET", "/deid-agent/api/file?path=../x")
    await call("file absolute", "GET", "/deid-agent/api/file?path=/etc/hosts")
    await call("model fast", "POST", "/deid-agent/api/model", {"value": "fast"})
    await call("model unknown", "POST", "/deid-agent/api/model", {"value": "nope"})

    if multi:
        await call("agent unknown", "POST", "/deid-agent/api/agent", {"value": "Nobody"})
        mark = len(s.events)
        await call("to Codex", "POST", "/deid-agent/api/agent", {"value": "Codex"})
        await s.until(kind("status", status="ready"), after=mark)
        mark = len(s.events)
        await call("task 2", "POST", "/deid-agent/api/prompt", {"text": "Now check the draft"})
        ask = await s.until(kind("permission"), after=mark)
        await call("switch while busy", "POST", "/deid-agent/api/agent", {"value": "Claude Code"})
        await call("refuse", "POST", "/deid-agent/api/permission", {"id": ask["id"], "optionId": "reject"})
        await s.until(kind("files"), after=mark)
        mark = len(s.events)
        await call("back to Claude Code", "POST", "/deid-agent/api/agent", {"value": "Claude Code"})
        await s.until(kind("status", status="ready"), after=mark)
        mark = len(s.events)
        await call("task 3", "POST", "/deid-agent/api/prompt", {"text": "Send it"})
        await s.until(kind("permission"), after=mark)
        await call("cancel", "POST", "/deid-agent/api/cancel", {})
        await s.until(kind("files"), after=mark)
        # a hand-over in the middle of a task
        mark = len(s.events)
        await call("task 4", "POST", "/deid-agent/api/prompt", {"text": "Redraft the letter"})
        await s.until(kind("permission"), after=mark)
        await call("hand over", "POST", "/deid-agent/api/agent", {"value": "MatterAgent", "handover": True})
        await call("hand over again", "POST", "/deid-agent/api/agent", {"value": "Codex", "handover": True})
        await s.until(kind("handover"), after=mark, timeout=30)
        handed = [i for i, e in enumerate(s.events) if e.get("kind") == "handover"][-1]
        ask = await s.until(kind("permission"), after=handed)
        await call("allow carried", "POST", "/deid-agent/api/permission", {"id": ask["id"], "optionId": "allow"})
        await s.until(kind("files"), after=handed, timeout=30)
        await call("state after hand-over", "GET", "/deid-agent/api/state")
        mark = len(s.events)
        await call("back to Claude Code again", "POST", "/deid-agent/api/agent", {"value": "Claude Code"})
        await s.until(kind("status", status="ready"), after=mark)

    # review: by the agent at work, then (with several agents) by a second reviewer
    mark = len(s.events)
    r = await call("review", "POST", "/deid-agent/api/review", {"path": "lease.md"})
    await call("review again at once", "POST", "/deid-agent/api/review", {"path": "lease.md"})
    await s.until(lambda e: e.get("kind") == "review" and e.get("state") in ("done", "failed"), after=mark, timeout=30)
    await s.until(lambda e: True, after=mark)
    await asyncio.sleep(0.3)
    if multi:
        mark = len(s.events)
        await call("second reviewer", "POST", "/deid-agent/api/review", {"path": "lease.md", "agent": "MatterAgent"})
        await s.until(lambda e: e.get("kind") == "review" and e.get("state") in ("done", "failed"), after=mark, timeout=30)
        await s.until(kind("status", status="ready"), after=mark, timeout=30)
        await asyncio.sleep(0.3)
        await call("reviewer unknown", "POST", "/deid-agent/api/review", {"path": "lease.md", "agent": "Nobody"})
    mark = len(s.events)
    await call("review of the deal", "POST", "/deid-agent/api/review", {"path": "deal.md"})
    await s.until(lambda e: e.get("kind") == "review" and e.get("state") in ("done", "failed"), after=mark, timeout=30)
    await s.until(kind("files"), after=mark)
    await asyncio.sleep(0.3)
    await call("view of the deal", "GET", "/deid-agent/api/review?path=deal.md")
    await call("review of nothing", "POST", "/deid-agent/api/review", {"path": "missing.md"})
    await call("review of a word file", "GET", "/deid-agent/api/review?path=deal.docx")
    view = await call("view", "GET", "/deid-agent/api/review?path=lease.md")
    notes = view["body"].get("notes", []) if isinstance(view["body"], dict) else []
    sha = view["body"].get("sha") if isinstance(view["body"], dict) else ""
    await call("suggest", "POST", "/deid-agent/api/review-suggest", {"path": "lease.md", "start": 2, "end": 7, "text": "LEASE", "sha": sha})
    await call("suggest nothing", "POST", "/deid-agent/api/review-suggest", {"path": "lease.md", "start": 2, "end": 2, "text": ""})
    await call("suggest stale", "POST", "/deid-agent/api/review-suggest", {"path": "lease.md", "start": 0, "end": 1, "text": "X", "sha": "0"})
    view = await call("view 2", "GET", "/deid-agent/api/review?path=lease.md")
    notes = view["body"].get("notes", []) if isinstance(view["body"], dict) else []
    # notes on one quote by two reviewers come in a random order (their files' names): pick by who wrote them
    by_who = sorted(notes, key=lambda n: (json.dumps(n.get("creator"), sort_keys=True), n.get("start") or 0))
    edit = next((n for n in by_who if n["kind"] == "edit"), None)
    risk = next((n for n in by_who if n["kind"] == "risk"), None)
    if edit:
        await call("accept edit", "POST", "/deid-agent/api/review-decide", {"path": "lease.md", "target": edit["id"], "verdict": "accepted"})
    if risk:
        await call("accept wording", "POST", "/deid-agent/api/review-decide", {"path": "lease.md", "target": risk["id"],
                                                                               "verdict": "accepted", "comment": " good  one ",
                                                                               "correction": "  Better words  "})
        await call("judge note", "POST", "/deid-agent/api/review-decide", {"path": "lease.md", "target": risk["id"], "verdict": "useful"})
    await call("decide unknown", "POST", "/deid-agent/api/review-decide", {"path": "lease.md", "target": "0" * 32, "verdict": "accepted"})
    await call("decide bad verdict", "POST", "/deid-agent/api/review-decide", {"path": "lease.md", "target": "0" * 32, "verdict": "maybe"})
    await call("decide not a dict", "POST", "/deid-agent/api/review-decide", [1, 2])
    view = await call("view 3", "GET", "/deid-agent/api/review?path=lease.md")
    sha = view["body"].get("sha") if isinstance(view["body"], dict) else ""
    text = view["body"].get("text", "") if isinstance(view["body"], dict) else ""
    await call("person writes", "POST", "/deid-agent/api/review-text", {"path": "lease.md", "text": text + "\nAdded by the person.\n", "sha": sha})
    await call("person writes stale", "POST", "/deid-agent/api/review-text", {"path": "lease.md", "text": "x", "sha": sha})
    await call("new document", "POST", "/deid-agent/api/review-text", {"path": "notes/new.md", "text": "Fresh for Ada Brenner.\n",
                                                                     "written_by": {"agent": "Codex", "model": "gpt-5"}})
    await call("new over old", "POST", "/deid-agent/api/review-text", {"path": "lease.md", "text": "x"})
    await call("new hidden", "POST", "/deid-agent/api/review-text", {"path": ".hidden.md", "text": "x"})
    await call("vote", "POST", "/deid-agent/api/review-vote", {"key": "chat-1:a-1", "question": "What notice?", "answer": "Six months for Ada Brenner.",
                                                             "verdict": "down", "agent": "Codex", "model": "gpt-5", "comment": "too short",
                                                             "correction": " Six months' written notice. "})
    await call("vote again", "POST", "/deid-agent/api/review-vote", {"key": "chat-1:a-1", "question": "What notice?", "answer": "Six months.",
                                                                   "verdict": "up"})
    await call("bad vote", "POST", "/deid-agent/api/review-vote", {"key": "", "answer": "x", "question": "q", "verdict": "up"})
    await call("view 4", "GET", "/deid-agent/api/review?path=lease.md")
    await call("view new", "GET", "/deid-agent/api/review?path=notes/new.md")
    await call("dataset", "GET", "/deid-agent/api/review-dataset")
    await call("final state", "GET", "/deid-agent/api/state")


def make_folder(base: Path) -> Path:
    folder = base / "Case 1"
    (folder / "docs").mkdir(parents=True)
    (folder / "letter.md").write_text("Dear Ada Brenner,\nHarrowgate Freight Ltd (\"HFL\") writes.\n")
    (folder / "lease.md").write_text(LEASE)
    # a party called by its short name: a quote naming it in full is found in tokens
    (folder / "deal.md").write_text('This agreement is between HARROWGATE FREIGHT LTD ("HFL") and Ada Brenner.\n\n'
                                    "HFL shall deliver the goods by 1 May.\n")
    (folder / "deal.docx").write_bytes(b"PK\x03\x04 not really")
    (folder / ".legix" / "text").mkdir(parents=True)
    (folder / ".legix" / "text" / "deal.docx.md").write_text("The deal between HFL and Ada Brenner.\n")
    (folder / "docs" / "memo.txt").write_text("TOBIAS WREN (\"TW\") is the tenant.\n")
    return folder


def snapshot_files(folder: Path) -> dict:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(folder))
            data = p.read_bytes()
            try:
                text = data.decode()
                try:
                    out[rel] = json.loads(text) if p.suffix == ".json" else text
                except ValueError:
                    out[rel] = text
            except UnicodeDecodeError:
                out[rel] = repr(data[:40])
    return out


def jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


async def run(cmd, base: Path, py: str, multi: bool, verbose: bool):
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    home = base / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "auth.json").write_text("{}")
    tmp = base / "tmp"
    tmp.mkdir()
    folder = make_folder(base)
    fake = base / "fake.py"
    shutil.copy(FAKE, fake)
    vault = base / "vault.sqlite"
    db = sqlite3.connect(vault)
    db.executescript("create table if not exists salts (scope text primary key, salt blob not null);")
    db.execute("insert into salts values ('case-1', ?)", (SALT,))
    db.commit()
    db.close()
    (base / "seeds.toml").write_text(SEED)
    (base / "style.css").write_text("body { color: teal; }\n")
    (base / "hosts.toml").write_text("[[host]]\nname = \"thor\"\n")
    port = free_port()
    flags = ["--folder", str(folder), "--scope", "case-1", "--title", "Case 1", "--port", str(port), "--secrets-stdin",
             "--dev-login", "--seeds", str(base / "seeds.toml"), "--vault", str(vault), "--audit", str(base / "audit.jsonl"),
             "--record", str(base / "record.jsonl"), "--usage-log", str(base / "usage.jsonl"), "--style", str(base / "style.css"),
             "--no-guard", "--person", "ann", "--claude-executable", "/usr/bin/true"]
    if multi:
        flags += ["--agent", agent_cmd(py, fake, "Claude Code", "claude-agent-acp", "echo-prompt", "review"),
                  "--agent", agent_cmd(py, fake, "Codex", "codex-acp", "echo-prompt"),
                  "--agent", agent_cmd(py, fake, "MatterAgent", "matteragent", "echo-prompt", "review"),
                  "--agent-unavailable", "Gemini=not on this Mac",
                  "--agent-meta", 'MatterAgent={"own": {"x": 1}}',
                  "--codex-executable", "/usr/bin/true",
                  "--subagents", str(base / "hosts.toml"), "--subagents-command", f"{py} -m matteragent.mcp"]
    else:
        flags += ["--agent-command", f"{py} {fake} Solo review", "--session-meta", '{"claudeCode": {"options": {"tools": ["Read"]}}}']
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "TMPDIR": str(tmp) + "/", "USER": "ann", "LANG": "en_US.UTF-8"}
    fake_model._ids = itertools.count(1)
    log: list = []
    async with FakeModel([]) as model:
        flags += ["--upstream", model.url]
        proc = await asyncio.create_subprocess_exec(*cmd, *flags, env=env, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        proc.stdin.write(json.dumps({"token": TOKEN, "api_key": None}).encode() + b"\n")
        await proc.stdin.drain()
        first = (await asyncio.wait_for(proc.stdout.readline(), 30)).decode().strip()
        for _ in range(200):
            try:
                await http(port, "GET", "/deid/health")
                break
            except OSError:
                await asyncio.sleep(0.05)
        s11, s10 = Stream(port, "1.1"), Stream(port, "1.0")
        try:
            await scenario(port, s11, log, folder, model, multi)
        except AssertionError as e:
            log.append(["scenario stopped", str(e)])
        await asyncio.sleep(0.5)
        procs_before = live(fake)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 20)
        except asyncio.TimeoutError:
            proc.kill()
        err = (await proc.stderr.read()).decode(errors="replace")
        await asyncio.sleep(0.5)
        left = live(fake)
        for t in (s11.task, s10.task):
            t.cancel()
        requests = [q["body"] for q in model.requests]
    result = {"first line": first, "api": log, "events": s11.events, "events over HTTP/1.0": len(s10.events) == len(s11.events),
              "files": snapshot_files(folder), "usage": jsonl(base / "usage.jsonl"), "audit": jsonl(base / "audit.jsonl"),
              "record": jsonl(base / "record.jsonl"), "model got": requests,
              "agents while running": procs_before, "agents left": left,
              "codex home": sorted(str(p.relative_to(home)) for p in home.rglob("*"))}
    return normalise(result, base), err


def live(fake: Path) -> int:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if str(fake) in l and "ps -axo" not in l)


def normalise(v, base: Path):
    """Paths, ports, times and temp names made neutral; a random id named by where it first shows in the
    API answers and the events (in their order), one seen only in files or records left as <ID>; the markup
    files (named by random ids) and the dataset's records (in the order of those names) as sorted lists."""
    v = dict(v)
    # notes on the same words with the same importance (two reviewers on one quote) come in the order of
    # their files' random names, in both programs: put them in one order before the ids are named
    for step in v["api"]:
        body = step[1]["body"] if isinstance(step[1], dict) else None
        if isinstance(body, dict) and isinstance(body.get("notes"), list):
            body["notes"] = sorted(body["notes"], key=lambda n: (n.get("start") is None, n.get("start") or 0,
                                                                 -(n.get("importance") or 0),
                                                                 json.dumps(n.get("creator"), sort_keys=True),
                                                                 n.get("kind", ""), n.get("note", ""), n.get("quote", "")))
    # the markup files are named by their random ids: kept as (folder, contents), the names left out
    v["markup"] = [[k.rsplit("/", 1)[0], x] for k, x in v["files"].items() if k.startswith(".legix/")]
    v["files"] = {k: x for k, x in v["files"].items() if not k.startswith(".legix/")}
    text = json.dumps(v, ensure_ascii=False).replace(str(base), "<BASE>").replace(os.path.realpath(base), "<BASE>")
    text = re.sub(r"deid-codex-[a-z0-9_]{8}", "deid-codex-X", text)
    text = re.sub(r"127\.0\.0\.1:\d+", "127.0.0.1:<PORT>", text)
    text = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", "<T>", text)
    out = json.loads(text)
    ident = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{32}|[0-9a-f]{16}|[0-9a-f]{12})(?![0-9a-f])")
    names: dict[str, str] = {}
    for m in ident.finditer(json.dumps([out["api"], out["events"]], ensure_ascii=False)):
        names.setdefault(m.group(0), f"<ID{len(names) + 1}>")
    out = json.loads(ident.sub(lambda m: names.get(m.group(0), "<ID>"), json.dumps(out, ensure_ascii=False)))
    out["markup"] = sorted(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in out["markup"])
    for step in out["api"]:
        body = step[1]["body"] if isinstance(step[1], dict) else None
        if isinstance(body, dict) and isinstance(body.get("records"), list):
            body["records"] = sorted(body["records"], key=lambda r: json.dumps(r, sort_keys=True))

    def drop(x):
        if isinstance(x, dict):
            return {k: drop(y) for k, y in x.items() if k not in ("at", "seconds")}
        if isinstance(x, list):
            return [drop(y) for y in x]
        return x
    return drop(out)


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return path, a, b
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if a.get(k, "<missing>") != b.get(k, "<missing>"):
                return first_difference(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}")
        return path, a, b
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_difference(x, y, f"{path}[{i}]")
        return path + " (length)", len(a), len(b)
    return path, a, b


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", default=str(DEID / "rust" / "target" / "release" / "deid-agent"))
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--only", choices=["solo", "multi"])
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="deid-agent-diff-"))
    py = sys.executable
    python_cmd = [py, "-c", "import sys; sys.path.insert(0, %r); from deidkit.agents.cli import main; main()" % str(DEID / "src")]
    same = True
    for name, multi in (("solo", False), ("multi", True)):
        if args.only and args.only != name:
            continue
        print(f"-- {name}")
        a, a_err = await run(python_cmd, work / "run", py, multi, args.v)
        b, b_err = await run([args.rust], work / "run", py, multi, args.v)
        for part in a:
            if a[part] != b[part]:
                same = False
                where, x, y = first_difference(a[part], b[part])
                print(f"DIFF {part}{where}\n  python: {json.dumps(x, ensure_ascii=False)[:900]}\n  rust:   {json.dumps(y, ensure_ascii=False)[:900]}")
            else:
                n = len(a[part]) if isinstance(a[part], (list, dict)) else a[part]
                print(f"same {part} ({n})")
        if args.v:
            print("python stderr:", a_err[-3000:])
            print("rust stderr:", b_err[-3000:])
            (work.parent / "agent-diff-python.json").write_text(json.dumps(a, ensure_ascii=False, indent=1))
            (work.parent / "agent-diff-rust.json").write_text(json.dumps(b, ensure_ascii=False, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
