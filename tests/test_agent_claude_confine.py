# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Claude Code in a matter: its shell commands in its sandbox (no network, people's files closed but
the folder, no command outside it, no start without it) and its tool calls decided and journalled by
agent-guard — given as the session's flag-tier settings, added to what the session already gives.
Codex and matteragent get none of it: Codex is confined by deid-agent's own profile, matteragent has
no shell."""

import json
import os
import stat
import sys

import pytest

pytest.importorskip("aiohttp")

from deidkit.agents.host import (GUARD_EVENTS, PEOPLES_FILES, AgentChoice, HostConfig,  # noqa: E402
                                 claude_settings, with_claude_settings)
from deidkit.proxy.server import ProxyConfig                                           # noqa: E402

from test_agent_host import FAKE, started, wait_for                                     # noqa: E402


def test_the_sandbox_closes_the_network_and_peoples_files_but_the_folder(tmp_path):
    sandbox = claude_settings(tmp_path / "matter")["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]          # no start without it
    assert sandbox["allowUnsandboxedCommands"] is False                  # no command outside it
    assert sandbox["autoAllowBashIfSandboxed"] is False                  # the person still asked
    assert sandbox["filesystem"] == {"denyRead": list(PEOPLES_FILES), "allowRead": [str(tmp_path / "matter")]}
    assert "/Users" in PEOPLES_FILES and "/Volumes" in PEOPLES_FILES
    assert sandbox["network"] == {"allowedDomains": [], "strictAllowlist": True}


def test_the_guard_hears_every_event_and_journals_where_it_is_told(tmp_path):
    guard, journal = "/Applications/Bernio Chat.app/x/agent-guard", tmp_path / "guard" / "case 1.jsonl"
    settings = claude_settings(tmp_path, confine=False, guard=[guard], guard_log=journal)
    assert "sandbox" not in settings and set(settings["hooks"]) == set(GUARD_EVENTS)
    for event in GUARD_EVENTS:
        hook = settings["hooks"][event][0]["hooks"][0]
        assert hook["command"] == (f"AGENT_GUARD_LOG='{journal}' "
                                   f"'/Applications/Bernio Chat.app/x/agent-guard' hook")
        assert hook["type"] == "command" and hook["timeout"] == 10


def test_without_confinement_or_guard_the_session_is_as_given(tmp_path):
    meta = {"claudeCode": {"options": {"tools": ["Read"]}}}
    assert claude_settings(tmp_path, confine=False) == {}
    assert with_claude_settings(meta, {}, tmp_path) == meta


def test_the_matters_settings_are_added_to_those_given(tmp_path):
    given = {"env": {"A": "1"}, "sandbox": {"enabled": False},
             "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "mine"}]}]}}
    meta = {"claudeCode": {"options": {"tools": ["Read"], "settings": given}}}
    ours = claude_settings(tmp_path, guard=["/x/agent-guard"])
    settings = with_claude_settings(meta, ours, tmp_path)["claudeCode"]["options"]["settings"]
    assert settings["env"] == {"A": "1"}                                  # kept
    assert settings["sandbox"] == ours["sandbox"]                         # the matter's replaces it
    assert [h["hooks"][0]["command"] for h in settings["hooks"]["PreToolUse"]] == ["mine", "/x/agent-guard hook"]
    assert meta["claudeCode"]["options"]["settings"]["sandbox"] == {"enabled": False}   # the given untouched


def test_a_settings_file_given_is_read_from_the_folder(tmp_path):
    (tmp_path / "conf.json").write_text(json.dumps({"env": {"B": "2"}}))
    meta = {"claudeCode": {"options": {"settings": "conf.json"}}}
    settings = with_claude_settings(meta, claude_settings(tmp_path), tmp_path)["claudeCode"]["options"]["settings"]
    assert settings["env"] == {"B": "2"} and settings["sandbox"]["enabled"]
    with pytest.raises(ValueError):
        with_claude_settings({"claudeCode": {"options": {"settings": ["x"]}}}, claude_settings(tmp_path), tmp_path)


async def session_meta(tmp_path, agent, **kw):
    folder = tmp_path / "matter"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=[agent], codex_confine=False, **kw)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Read the lease"},
                          headers={"x-deid-agent-token": cfg.token})
        await wait_for(lambda: any(e["kind"] in ("permission", "error") for e in host.events))
        said = "".join(e["text"] for e in host.events if e["kind"] == "message")
    finally:
        await client.close()
    return json.loads(said.split("[meta ", 1)[1].split("] ", 1)[0]) if "[meta " in said else None, folder


async def test_claude_code_starts_confined_and_guarded(tmp_path):
    meta = {"claudeCode": {"options": {"tools": ["Read", "Write", "Edit", "Bash"], "strictMcpConfig": True}}}
    agent = AgentChoice("Claude Code", [sys.executable, str(FAKE), "claude-agent-acp"], meta=meta)
    got, folder = await session_meta(tmp_path, agent, guard=["/x/agent-guard"], guard_log=tmp_path / "g.jsonl")
    options = got["claudeCode"]["options"]
    assert options["tools"] == ["Read", "Write", "Edit", "Bash"] and options["strictMcpConfig"] is True
    assert options["settings"]["sandbox"]["filesystem"]["allowRead"] == [str(folder)]
    assert "/x/agent-guard hook" in options["settings"]["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "settings" not in meta["claudeCode"]["options"]                # the given options untouched


async def test_codex_and_matteragent_get_none_of_it(tmp_path):
    for name in ("codex-acp", "matteragent"):
        agent = AgentChoice(name, [sys.executable, str(FAKE), name])
        got, _ = await session_meta(tmp_path / name, agent, guard=["/x/agent-guard"],
                                    codex_executable="/opt/codex/bin/codex")
        assert "settings" not in json.dumps(got or {}), name


def executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_guard_and_the_sandbox_come_from_the_command_line(tmp_path, monkeypatch):
    from deidkit.agents import cli
    from deidkit.agents.host import AgentHost

    seen = []
    monkeypatch.setattr(cli, "AgentHost", lambda cfg, proxy: seen.append(cfg) or AgentHost(cfg, proxy))
    monkeypatch.setattr(cli.web, "run_app", lambda *a, **k: None)
    folder = tmp_path / "matter"
    folder.mkdir()
    guard = executable(tmp_path / "agent-guard")
    base = ["--folder", str(folder), "--scope", "case 1/a", "--dev-login",
            "--vault", str(tmp_path / "vault.sqlite"), "--audit", str(tmp_path / "logs" / "audit.jsonl")]
    cli.main(base + ["--guard", str(guard)])
    assert seen[0].guard == [str(guard)] and seen[0].claude_confine is True
    assert seen[0].guard_log == tmp_path / "logs" / "guard" / "case_1_a.jsonl"
    assert stat.S_IMODE(os.stat(tmp_path / "logs" / "guard").st_mode) == 0o700
    cli.main(base + ["--guard", str(guard), "--no-guard", "--claude-unconfined"])
    assert seen[1].guard == [] and seen[1].guard_log is None and seen[1].claude_confine is False
    monkeypatch.setenv("DEID_AGENT_GUARD", str(guard))      # the bundle's wrapper names its own guard
    cli.main(base)
    assert seen[2].guard == [str(guard)]
    with pytest.raises(SystemExit):
        cli.main(base + ["--guard", str(tmp_path / "nowhere")])
