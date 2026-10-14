# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: the Python deid-proxy and the Rust one, side by side, on the same requests.

    RT=…/agent-runtime/python/bin/python3.13
    $RT rust/tools/differential_proxy.py [--rust rust/target/release/deid-proxy] [-v]

Each proxy gets a vault file made beforehand with one salt (so both mint the same tokens, and the Rust
one reads a file Python's sqlite3 wrote), the same seed file, and a scripted model upstream
(matteragent's tests/fake_model.py). Compared: every request the model got (path, the headers that
matter, the body), every answer the client got (status, content type, body or events), and the audit
and record files (times dropped).
"""
import argparse
import asyncio
import itertools
import re
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
DEID = HERE.parents[2]
sys.path.insert(0, str(DEID.parent / "matteragent" / "tests"))
sys.path.insert(0, str(DEID.parent / "matteragent" / "src"))
import fake_model  # noqa: E402
from fake_model import FakeModel, calls, text  # noqa: E402

SALT = bytes(range(32))
SEED = """scope = "case-1"
paths = ["{folder}"]

[[entity]]
type = "individual"
name = "Ada Brenner"

[[entity]]
type = "company"
name = "Harrowgate Freight Ltd"

[[party]]
name = "Tobias Wren"
role = "Director"
"""

LETTER = "Dear Ada Brenner,\nHarrowgate Freight Ltd (\"HFL\") writes about Musterweg 12, 20095 Hamburg. Tobias Wren signs; lena.voss@example.org, +44 20 7946 0958, IBAN GB82 WEST 1234 5698 7654 32."


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def http(port, method, path, headers=None, body=b""):
    """One request; (status, headers, body) — the body read whole."""
    r, w = await asyncio.open_connection("127.0.0.1", port)
    head = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
    head += [f"{k}: {v}" for k, v in (headers or {}).items()]
    if body or method == "POST":
        head.append(f"Content-Length: {len(body)}")
    w.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
    await w.drain()
    status = int((await r.readline()).split()[1])
    hdrs = {}
    while (line := await r.readline()) not in (b"\r\n", b""):
        k, _, v = line.decode().partition(":")
        hdrs[k.strip().lower()] = v.strip()
    data = b""
    if "chunked" in hdrs.get("transfer-encoding", ""):
        while True:
            size = int((await r.readline()).strip() or b"0", 16)
            if size == 0:
                await r.readline()
                break
            data += await r.readexactly(size)
            await r.readexactly(2)
    elif "content-length" in hdrs:
        data = await r.readexactly(int(hdrs["content-length"]))
    else:
        data = await r.read()
    w.close()
    return status, hdrs, data


def answer(status, hdrs, data):
    ctype = hdrs.get("content-type", "").split(";")[0]
    if "event-stream" in ctype:
        events, name, buf = [], "", []
        for line in data.decode().split("\n"):
            if not line:
                if buf or name:
                    d = "\n".join(buf)
                    try:
                        d = json.loads(d)
                    except ValueError:
                        pass
                    events.append([name, d])
                name, buf = "", []
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                buf.append(line[6:] if line[5:6] == " " else line[5:])
        return {"status": status, "type": ctype, "events": events}
    try:
        body = json.loads(data)
    except ValueError:
        body = data.decode("utf-8", "replace")
    return {"status": status, "type": ctype, "body": body}


def messages(stream, extra=None, content=LETTER, where="Primary working directory: {folder}"):
    body = {"model": "claude", "max_tokens": 100, "system": [{"type": "text", "text": "x-anthropic-billing-header: v"},
                                                             {"type": "text", "text": where}],
            "messages": [{"role": "user", "content": content}], **(extra or {})}
    if stream:
        body["stream"] = True
    return body


SCENARIO = [
    # (method, path, headers, body or None, model script items)
    ("GET", "/deid/health", {}, None, []),
    ("POST", "/v1/messages", {"x-api-key": "k", "anthropic-version": "2023-06-01"}, messages(True),
     [text("Reply to PERSON_{0} at ORG_{1}: done.")]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(False),
     [calls(("Read", {"file_path": "/work/ORG_{1}/a.md"}), ("WebSearch", {"q": "PERSON_{0}"}), say="Reading for person_{0}.")]),
    ("POST", "/v1/messages/count_tokens", {"x-api-key": "k"}, messages(False), [{"input_tokens": 42}]),
    ("POST", "/v1/messages", {"x-deid-scope": "other", "x-api-key": "k"}, messages(True), [text("Other scope: PERSON_{0}?")]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(False, content=[{"type": "mystery"}]), []),
    ("POST", "/v1/messages", {"x-api-key": "k"}, b"not json", []),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(True, content=[{"type": "text", "text": LETTER},
                                                                          {"type": "image", "source": {"type": "base64", "data": "AA"}}]),
     [calls(("Read", {"file_path": "/work/ORG_{1}/a.md"}))]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(True), [(500, {"type": "error", "error": {"type": "api_error", "message": "PERSON_{0}"}})]),
    ("POST", "/v1/responses", {"authorization": "Bearer t", "chatgpt-account-id": "a", "originator": "codex_cli_rs",
                               "x-codex-turn-metadata": json.dumps({"cwd": "/work/Harrowgate Freight Ltd"})},
     {"model": "gpt", "stream": True, "instructions": "Help Ada Brenner.",
      "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": LETTER + " <cwd>{folder}</cwd>"}]}]},
     [calls(("exec_command", {"cmd": "ls /work/ORG_{1}"}), say="Listing for PERSON_{0}.")]),
    ("POST", "/v1/responses", {"authorization": "Bearer t"}, {"model": "gpt", "input": [{"role": "user", "content": LETTER}]},
     [text("ok PERSON_{0}")]),
    # the same compressed, as Codex sends it to ChatGPT: the runtime's aiohttp cannot read zstd (it wants
    # backports.zstd) and answers 400 itself, so only the Rust proxy is checked here, against the twin above
    ("POST", "/v1/responses", {"authorization": "Bearer t", "content-encoding": "zstd"}, "zstd:" + json.dumps(
        {"model": "gpt", "input": [{"role": "user", "content": LETTER}]}), [text("ok PERSON_{0}")]),
    ("GET", "/v1/models", {"x-api-key": "k"}, None, []),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(True, where="No folder here."), [text("Default: PERSON_{0}")]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(True), [("cut", text("Cut short for PERSON_{0} at ORG_{1}, sorry."))]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(True), [("stream_error", "overloaded_error")]),
    ("POST", "/v1/messages", {"x-api-key": "k"}, messages(False, where="Primary working directory: /elsewhere"),
     [text("Elsewhere: PERSON_{0}")]),
    ("PUT", "/v1/files", {}, None, []),
    ("GET", "/v1/messages", {"upgrade": "websocket", "connection": "Upgrade"}, None, []),
]


CONFIGS = {"folders": ["--scope", "default"],
           "strict": ["--scope", "default", "--binary", "refuse", "--require-scope"]}


async def run(cmd, base: Path, flags):
    shutil.rmtree(base, ignore_errors=True)
    folder = base / "matter"
    folder.mkdir(parents=True)
    vault = base / "vault.sqlite"
    db = sqlite3.connect(vault)
    db.executescript("create table if not exists salts (scope text primary key, salt blob not null);")
    for scope in ("case-1", "other", "default"):
        db.execute("insert into salts values (?, ?)", (scope, SALT))
    db.commit()
    db.close()
    (base / "seeds.toml").write_text(SEED.replace("{folder}", str(folder)))
    port = free_port()
    fake_model._ids = itertools.count(1)
    async with FakeModel([]) as fake:
        fake.models = [{"id": "m", "meta": {"n_ctx": 1000}}]
        proc = await asyncio.create_subprocess_exec(
            *cmd, "--port", str(port), "--vault", str(vault), *flags, "--seeds", str(base / "seeds.toml"),
            "--audit", str(base / "audit.jsonl"), "--record", str(base / "record.jsonl"),
            "--upstream", fake.url, "--upstream-openai", fake.url + "/v1", "--upstream-chatgpt", fake.url + "/codex",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        for _ in range(100):
            try:
                await http(port, "GET", "/deid/health")
                break
            except OSError:
                await asyncio.sleep(0.05)
        answers, sent, audits, records = [], [], [], []
        for n, (method, path, headers, body, script) in enumerate(SCENARIO):
            fake_model._ids = itertools.count(100 * n)     # the fake's ids, the same on both sides
            folder_s = str(folder)
            if isinstance(body, dict):
                raw = json.dumps(body).replace("{folder}", folder_s).encode()
            elif isinstance(body, str) and body.startswith("zstd:"):
                import zstandard
                raw = zstandard.ZstdCompressor().compress(body[5:].encode())
            else:
                raw = body or b""
            # the model answers in the tokens the vault minted for Ada Brenner and Harrowgate Freight Ltd,
            # read when the request arrives (the proxy has enrolled the seeds by then)
            def fill(x):
                def now(body, x=x):
                    toks = known_tokens(vault)
                    out = json.loads(re.sub(r"[A-Z]+_\{(\d)\}", lambda m: toks[int(m.group(1))], json.dumps(x)))
                    return tuple(out) if isinstance(x, tuple) else out
                return now
            fake.script = [fill(x) for x in script]
            h = dict(headers)
            if raw:
                h["content-type"] = "application/json"
            before = len(fake.requests)
            st, hd, data = await http(port, method, path, h, raw if method in ("POST", "PUT") else b"")
            answers.append(answer(st, hd, data))
            sent.append(fake.requests[before:])
            await asyncio.sleep(0.05)       # the audit line is written after the answer
            for name, got in (("audit", audits), ("record", records)):
                rows = lines(base / f"{name}.jsonl")
                got.append(rows[sum(map(len, got)):])
        proc.terminate()
        _, err = await proc.communicate()
        keep = ("x-api-key", "authorization", "anthropic-version", "x-codex-turn-metadata", "chatgpt-account-id",
                "accept-encoding", "content-type", "content-encoding", "originator")
        reqs = [[{"path": q["path"], "headers": {k: v for k, v in q["headers"].items() if k in keep}, "body": q["body"]}
                 for q in qs] for qs in sent]
        gets = list(fake.gets)
        upstream = fake.url
    out = {"requests": reqs, "gets": gets, "answers": answers, "audit": audits, "record": records}
    return json.loads(json.dumps(out).replace(upstream, "<UPSTREAM>")), err.decode(errors="replace"), str(folder)


def lines(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        o = json.loads(line)
        o.pop("at", None)
        out.append(o)
    return out


def known_tokens(vault):
    db = sqlite3.connect(vault)
    try:
        rows = db.execute("select token, real_value from tokens where scope = 'case-1' order by seq").fetchall()
    except sqlite3.Error:
        rows = []
    db.close()
    want = {"Ada Brenner": 0, "Harrowgate Freight Ltd": 1}
    out = ["X_0", "X_0"]
    for tok, real in rows:
        if real in want:
            out[want[real]] = tok
    return out


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
    ap.add_argument("--rust", default=str(DEID / "rust" / "target" / "release" / "deid-proxy"))
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="deid-proxy-diff-"))
    env_py = [sys.executable, "-c", "import sys; sys.path.insert(0, %r); from deidkit.proxy.cli import main; main()" % str(DEID / "src")]
    same = True
    for config, flags in CONFIGS.items():
        print(f"-- {config}: {' '.join(flags)}")
        a, py_err, _ = await run(env_py, work / "run", flags)
        b, rs_err, _ = await run([args.rust], work / "run", flags)
        z = next(i for i, sc in enumerate(SCENARIO) if isinstance(sc[3], str) and sc[3].startswith("zstd:"))
        out = lambda rows: [o for o in rows if o.get("dir") != "in"]
        twin = b["requests"][z] == b["requests"][z - 1] and b["answers"][z]["status"] == b["answers"][z - 1]["status"] \
            and b["audit"][z] == b["audit"][z - 1] and out(b["record"][z]) == out(b["record"][z - 1])
        print(("same" if twin else "DIFF") + f" rust zstd request = its plain twin ({b['answers'][z]['status']})")
        same = same and twin
        for part in ("requests", "answers", "audit", "record"):
            a[part][z] = b[part][z] = None
        for part in ("requests", "gets", "answers", "audit", "record"):
            if a[part] != b[part]:
                same = False
                where, x, y = first_difference(a[part], b[part])
                print(f"DIFF {part}{where}\n  python: {json.dumps(x, ensure_ascii=False)[:700]}\n  rust:   {json.dumps(y, ensure_ascii=False)[:700]}")
            else:
                print(f"same {part} ({len(a[part])})")
        if args.v:
            print(json.dumps(b["answers"], ensure_ascii=False, indent=1)[:6000])
        if args.v or not same:
            print("python stderr:", py_err[-1500:])
            print("rust stderr:", rs_err[-1500:])
    shutil.rmtree(work, ignore_errors=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
