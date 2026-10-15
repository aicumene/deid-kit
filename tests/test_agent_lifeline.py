# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""deid-agent goes with the program that started it (``--lifeline``): when its stdin closes — the system closes it
when that program ends, even killed — it stops its agents and exits, freeing its port. Without the flag a closed
stdin changes nothing. Either way the agents are stopped with all they started: each runs in a process group of its
own. Both deid-agents, as processes: Python's, and Rust's when it is built (rust/target/release/deid-agent).

LegalAiz 14.10.2026: a killed desktop app left deid-agent, Claude Code and MatterAgent running, holding the port,
and the next agent could not start."""

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_acp_agent.py")
RUST = ROOT / "rust" / "target" / "release" / "deid-agent"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def listening(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def until(check, seconds: float, what: str) -> None:
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, what
        time.sleep(0.05)


@pytest.fixture(params=["python", "rust"])
def deid_agent(request) -> list[str]:
    if request.param == "rust":
        if not RUST.is_file():
            pytest.skip("Rust's deid-agent is not built (cargo build --release --bin deid-agent in rust/)")
        return [str(RUST)]
    return [sys.executable, "-m", "deidkit.agents"]


def start(deid_agent: list[str], tmp_path: Path, *flags: str):
    """deid-agent over a folder, its agent the scripted one under a shell that first starts a grandchild (a
    sleep): the agent's and the grandchild's numbers are written down. The secrets go in on stdin, left open."""
    (tmp_path / "matter").mkdir()
    script = (f"echo $$ > {shlex.quote(str(tmp_path / 'agent.pid'))}; echo 'the agent says: oops' >&2; sleep 300 & "
              f"echo $! > {shlex.quote(str(tmp_path / 'grandchild.pid'))}; exec {shlex.quote(sys.executable)} "
              f"{shlex.quote(str(FAKE))}")
    port = free_port()
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    env.pop("ANTHROPIC_API_KEY", None)
    proc = subprocess.Popen(
        [*deid_agent, "--folder", str(tmp_path / "matter"), "--scope", "lifeline-test", "--port", str(port),
         "--vault", str(tmp_path / "vault.sqlite"), "--audit", str(tmp_path / "audit.jsonl"), "--usage-log", "",
         "--dev-login", "--no-guard", "--secrets-stdin", *flags,
         "--agent-command", shlex.join(["/bin/sh", "-c", script])],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(tmp_path / "deid-agent.log", "wb"), env=env,
        cwd=tmp_path)
    proc.stdin.write(json.dumps({"token": "lifeline-test-token", "api_key": None}).encode() + b"\n")
    proc.stdin.flush()
    until(lambda: listening(port), 30, "deid-agent did not listen")
    pids = [tmp_path / "agent.pid", tmp_path / "grandchild.pid"]
    until(lambda: all(p.is_file() and p.read_text().strip() for p in pids), 30, "the agent did not start")
    agent, grandchild = (int(p.read_text()) for p in pids)
    assert alive(agent) and alive(grandchild)
    return proc, port, agent, grandchild


def gone(proc, port: int, agent: int, grandchild: int) -> None:
    until(lambda: proc.poll() is not None, 15, "deid-agent is still running")
    until(lambda: not listening(port), 5, "the port is still taken")
    until(lambda: not alive(agent), 5, "the agent outlived deid-agent")
    until(lambda: not alive(grandchild), 5, "what the agent started outlived it")


def stop_leftovers(proc, *pids: int) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    for pid in pids:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_with_a_lifeline_deid_agent_goes_when_its_stdin_closes(deid_agent, tmp_path):
    proc, port, agent, grandchild = start(deid_agent, tmp_path, "--lifeline")
    try:
        proc.stdin.close()                    # as the system does when the program that started it is killed
        gone(proc, port, agent, grandchild)
        log = (tmp_path / "deid-agent.log").read_text()
        # what the agent says on stderr is in deid-agent's log (the desktop app's Console shows it), and why it stopped
        assert "INFO agent stderr: the agent says: oops" in log, log
        assert "its stdin closed" in log, log
    finally:
        stop_leftovers(proc, agent, grandchild)


def test_without_one_a_closed_stdin_changes_nothing_and_sigterm_stops_the_whole_agent(deid_agent, tmp_path):
    proc, port, agent, grandchild = start(deid_agent, tmp_path)
    try:
        proc.stdin.close()
        time.sleep(1.5)
        assert proc.poll() is None and listening(port) and alive(agent) and alive(grandchild)
        proc.send_signal(signal.SIGTERM)
        gone(proc, port, agent, grandchild)
    finally:
        stop_leftovers(proc, agent, grandchild)
