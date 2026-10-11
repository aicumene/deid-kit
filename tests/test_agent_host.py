# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""deid-agent end to end with a scripted ACP agent: the token, a permission answered by a person,
a path outside the folder refused without asking, the changed files reported."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer       # noqa: E402

from deidkit.agents.host import AgentChoice, AgentHost, HostConfig, model_choice  # noqa: E402
from deidkit.patterns import RegexDetector                  # noqa: E402
from deidkit.proxy.server import Proxy, ProxyConfig         # noqa: E402
from deidkit.store import InMemoryTokenStore                # noqa: E402

FAKE = Path(__file__).with_name("fake_acp_agent.py")


async def wait_for(pred, timeout=10.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    """The person's own settings file is read; the tests' is empty."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
async def rig(tmp_path):
    (tmp_path / "letter.md").write_text("A letter.\n")
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1",
                     agent_command=[sys.executable, str(FAKE)], dev_login=True)
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1")))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    await wait_for(lambda: host.status == "ready")
    yield host, client, tmp_path
    await client.close()


async def test_the_api_needs_the_token(rig):
    host, client, folder = rig
    assert (await client.get("/deid-agent/api/state")).status == 401
    # never in the address, where access logs keep it
    assert (await client.get(f"/deid-agent/api/state?t={host.cfg.token}")).status == 401
    ok = await client.get("/deid-agent/api/state", headers={"x-deid-agent-token": host.cfg.token})
    body = await ok.json()
    assert ok.status == 200 and body["status"] == "ready" and body["files"] == ["letter.md"]
    assert body["auth"].startswith("personal")
    assert (await client.get("/deid/health")).status == 200           # the proxy is mounted


async def test_a_turn_with_a_person_answering(rig):
    host, client, folder = rig
    headers = {"x-deid-agent-token": host.cfg.token}
    r = await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
    assert r.status == 202
    await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
    ask = next(e for e in host.events if e["kind"] == "permission")
    assert ask["paths"] == ["draft.md"]
    r = await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "allow"},
                          headers=headers)
    assert r.status == 200
    await wait_for(lambda: any(e["kind"] == "turn_end" for e in host.events))
    await wait_for(lambda: any(e["kind"] == "files" for e in host.events))
    text = "".join(e["text"] for e in host.events if e["kind"] == "message")
    assert "first: allow; second: reject." in text                   # outside: refused unasked
    assert any(e["kind"] == "notice" for e in host.events)
    assert (folder / "draft.md").read_text() == "Draft for the client.\n"
    files = next(e for e in host.events if e["kind"] == "files")
    assert files["added"] == ["draft.md"]
    doc = await client.get("/deid-agent/api/file?path=draft.md", headers=headers)
    assert (await doc.json())["text"] == "Draft for the client.\n"
    outside = await client.get("/deid-agent/api/file?path=../x", headers=headers)
    assert outside.status == 404


async def test_the_agent_holds_a_stand_in_not_the_key(tmp_path):
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1",
                     agent_command=[sys.executable, str(FAKE)], api_key="sk-org-held-by-proxy")
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(
        default_scope="case-1", upstream_key=cfg.api_key, agent_key=cfg.agent_key)))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    try:
        await wait_for(lambda: host.status == "ready")
        assert host.agent.env["ANTHROPIC_API_KEY"] == cfg.agent_key
        assert host.agent.env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1"     # settings can't reroute
        assert "sk-org-held-by-proxy" not in "".join(host.agent.env.values())
    finally:
        await client.close()


def test_secrets_come_as_one_json_line():
    import io

    from deidkit.agents.cli import read_secrets
    assert read_secrets(io.StringIO('{"token": "t1", "api_key": null}\n')) == \
        {"token": "t1", "api_key": None}
    assert read_secrets(io.StringIO("")) == {}
    with pytest.raises(SystemExit):
        read_secrets(io.StringIO("token=t1\n"))


def test_model_requests_can_go_to_the_organizations_own_server(tmp_path, monkeypatch):
    """--upstream reaches the proxy: the agent's requests can go to an open model the organization
    runs itself, still through the proxy, instead of Anthropic. Without it, Anthropic as before."""
    from deidkit.agents import cli

    seen = []
    real_proxy = cli.Proxy

    def capture(store, cfg):
        seen.append(cfg)
        return real_proxy(store, cfg)

    monkeypatch.setattr(cli, "Proxy", capture)
    monkeypatch.setattr(cli.web, "run_app", lambda *a, **k: None)
    folder = tmp_path / "matter"
    folder.mkdir()
    base = ["--folder", str(folder), "--scope", "case-1", "--dev-login",
            "--vault", str(tmp_path / "vault.sqlite"), "--audit", str(tmp_path / "audit.jsonl")]
    cli.main(base)
    cli.main(base + ["--upstream", "http://10.0.0.7:8092/"])
    assert seen[0].upstream == "https://api.anthropic.com"
    assert seen[1].upstream == "http://10.0.0.7:8092"


async def test_a_refused_key_is_told_at_once(tmp_path):
    from aiohttp import web

    async def refuse(request):
        return web.json_response({"type": "error", "error": {"type": "authentication_error",
                                                             "message": "invalid x-api-key"}},
                                 status=401)
    provider = web.Application()
    provider.router.add_post("/v1/messages/count_tokens", refuse)
    upstream = TestServer(provider)
    await upstream.start_server()
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1",
                     agent_command=[sys.executable, str(FAKE)], api_key="sk-org-revoked")
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(
        default_scope="case-1", upstream=str(upstream.make_url("")).rstrip("/"),
        upstream_key=cfg.api_key, agent_key=cfg.agent_key)))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    try:
        await wait_for(lambda: host.status == "ready")
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        for _ in range(2):
            r = await client.post("/v1/messages/count_tokens", json=body,
                                  headers={"x-api-key": cfg.agent_key})
            assert r.status == 401
        told = [e for e in host.events if e["kind"] == "error"]
        assert len(told) == 1 and "refuses the organization's key" in told[0]["text"]
    finally:
        await client.close()
        await upstream.close()


async def test_a_settings_file_that_reroutes_the_agent_stops_the_local_sign_in(tmp_path):
    folder = tmp_path / "case"
    (folder / ".claude").mkdir(parents=True)
    (folder / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"https_proxy": "http://127.0.0.1:9", "EDITOR": "vi"}}))
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1",
                     agent_command=[sys.executable, str(FAKE)], dev_login=True)
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1")))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    try:
        await wait_for(lambda: host.status == "failed")
        told = next(e for e in host.events if e["kind"] == "error")["text"]
        assert "settings.json: https_proxy" in told and "EDITOR" not in told
        assert host.agent is None                                          # never started
    finally:
        await client.close()


def test_with_a_key_the_host_manages_the_provider():
    from deidkit.agents.acp import agent_env
    held = agent_env(base_url="http://127.0.0.1:1", scope="s", api_key="deid-agent-x")
    local = agent_env(base_url="http://127.0.0.1:1", scope="s", api_key=None)
    assert held["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1"      # settings can't reroute it
    assert "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST" not in local      # it would stop the sign-in


# ── the model ────────────────────────────────────────────────────────────────────────────────

async def test_the_page_offers_the_agents_models_and_switches_between_tasks(rig):
    host, client, folder = rig
    headers = {"x-deid-agent-token": host.cfg.token}
    state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
    assert state["models"]["current"] == "default"
    assert [o["value"] for o in state["models"]["options"]] == ["default", "fast"]

    r = await client.post("/deid-agent/api/model", json={"value": "fast"}, headers=headers)
    assert r.status == 200 and (await r.json())["models"]["current"] == "fast"
    assert [e["current"] for e in host.events if e["kind"] == "models"] == ["default", "fast"]

    # the next task runs on it: the agent names the model it works on
    await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
    await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
    assert "Working on it (fast)." in "".join(e["text"] for e in host.events if e["kind"] == "message")

    # not in the middle of a task, and not a model the agent does not offer
    busy = await client.post("/deid-agent/api/model", json={"value": "default"}, headers=headers)
    assert busy.status == 409
    ask = next(e for e in host.events if e["kind"] == "permission")
    await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "reject"},
                      headers=headers)
    await wait_for(lambda: not host.busy)
    unknown = await client.post("/deid-agent/api/model", json={"value": "huge"}, headers=headers)
    assert unknown.status == 400 and host.models["current"] == "fast"
    assert (await client.post("/deid-agent/api/model", json={"value": "default"})).status == 401


async def test_a_model_the_agent_changes_itself_reaches_the_page(rig):
    host, client, folder = rig
    await host.on_update({"update": {"sessionUpdate": "config_option_update", "configOptions": [
        {"id": "model", "category": "model", "type": "select", "currentValue": "fast",
         "options": [{"value": "default", "name": "Default"}, {"value": "fast", "name": "Fast"}]}]}})
    assert host.models["current"] == "fast" and host.events[-1]["kind"] == "models"


def test_the_model_choice_reads_both_shapes_acp_has_used():
    grouped = {"configOptions": [
        {"id": "mode", "type": "select", "currentValue": "ask", "options": [{"value": "ask"}]},
        {"id": "model", "category": "model", "type": "select", "currentValue": "b", "options": [
            {"value": "a", "name": "A"},
            {"group": "more", "name": "More", "options": [{"value": "b", "name": "B", "description": "d"}]}]}]}
    assert model_choice(grouped) == {"via": "config", "id": "model", "current": "b", "options": [
        {"value": "a", "name": "A", "description": ""}, {"value": "b", "name": "B", "description": "d"}]}
    older = {"models": {"currentModelId": "x", "availableModels": [{"modelId": "x", "name": "X"},
                                                                  {"modelId": "y"}]}}
    assert model_choice(older) == {"via": "models", "id": None, "current": "x", "options": [
        {"value": "x", "name": "X", "description": ""}, {"value": "y", "name": "y", "description": ""}]}
    assert model_choice({"configOptions": [{"id": "mode", "type": "select", "options": []}]}) is None
    assert model_choice({}) is None



async def test_what_a_task_cost_reaches_the_page_and_the_usage_log(tmp_path):
    folder = tmp_path / "Client A Ltd"                       # a folder name may be a client's
    folder.mkdir()
    usage_log = tmp_path / "logs" / "usage.jsonl"
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1",
                     agent_command=[sys.executable, str(FAKE)], dev_login=True, usage_log=usage_log)
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1")))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    try:
        await wait_for(lambda: host.status == "ready")
        headers = {"x-deid-agent-token": host.cfg.token}
        await client.post("/deid-agent/api/prompt", json={"text": "Draft"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        ask = next(e for e in host.events if e["kind"] == "permission")    # the path outside: refused unasked
        await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "allow"},
                          headers=headers)
        await wait_for(lambda: any(e["kind"] == "turn_end" for e in host.events))
        state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
    finally:
        await client.close()
    window = next(e for e in host.events if e["kind"] == "usage")
    assert (window["used"], window["size"]) == (14200, 32768) and state["context"] == {"used": 14200, "size": 32768}
    end = next(e for e in host.events if e["kind"] == "turn_end")
    assert end["usage"] == {"inputTokens": 3000, "cachedReadTokens": 11000, "cachedWriteTokens": 0,
                            "outputTokens": 420, "totalTokens": 14420}      # counts only
    assert isinstance(end["seconds"], float)
    assert state["usage"] == {"tasks": 1, **end["usage"]}
    line = json.loads(usage_log.read_text())
    assert (line["scope"], line["model"], line["stopReason"], line["usage"]) == \
        ("case-1", "default", "end_turn", end["usage"])
    assert "Client A" not in usage_log.read_text() and oct(usage_log.stat().st_mode & 0o777) == "0o600"


def test_turn_usage_reads_both_places_agents_put_it():
    from deidkit.agents.host import turn_usage
    assert turn_usage({"stopReason": "end_turn"}) == {}
    assert turn_usage({"usage": {"inputTokens": 5, "outputTokens": True, "totalTokens": -1}}) == {"inputTokens": 5}
    assert turn_usage({"_meta": {"usage": {"outputTokens": 7}}}) == {"outputTokens": 7}
    assert turn_usage({"usage": "lots"}) == {}


def two_agents(*, upstream=None, unavailable=None):
    return [AgentChoice("Claude Code", [sys.executable, str(FAKE), "Claude Code"], unavailable=unavailable),
            AgentChoice("MatterAgent", [sys.executable, str(FAKE), "MatterAgent"], upstream=upstream)]


async def started(cfg, proxy_cfg):
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), proxy_cfg))
    client = TestClient(TestServer(host.app()))
    await client.start_server()
    await wait_for(lambda: host.status in ("ready", "failed"))
    return host, client


async def test_the_page_switches_agents_between_tasks(tmp_path):
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=two_agents())
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
        assert state["agents"] == {"current": "Claude Code", "options": [
            {"value": "Claude Code", "name": "Claude Code"}, {"value": "MatterAgent", "name": "MatterAgent"}]}
        assert state["handover"] is None
        first = host.agent

        r = await client.post("/deid-agent/api/agent", json={"value": "MatterAgent"}, headers=headers)
        assert r.status == 202
        await wait_for(lambda: host.status == "ready" and host.agent is not first)
        assert first.proc.returncode is None                           # the other one keeps its seat
        assert host.current.name == "MatterAgent" and host.models["current"] == "default"
        kinds = [e["kind"] for e in host.events]
        assert kinds.count("agents") == 2 and {"kind": "models", "options": []} in \
            [{k: v for k, v in e.items() if k != "at"} for e in host.events]

        # the next task goes to it, in its own session
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        assert "[MatterAgent] Working on it" in "".join(e["text"] for e in host.events if e["kind"] == "message")
        busy = await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        assert busy.status == 409 and host.current.name == "MatterAgent"   # a switch is between tasks…
        assert "hand the task over" in (await busy.json())["error"]          # …or a handover
        ask = next(e for e in host.events if e["kind"] == "permission")
        await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "reject"},
                          headers=headers)
        await wait_for(lambda: not host.busy)
        done = next(e for e in host.events if e["kind"] == "permission_done" and e["id"] == ask["id"])
        assert done["outcome"] == "refused"

        # back to the first: the same process and session, at once
        r = await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        assert r.status == 202 and host.agent is first and host.status == "ready"

        unknown = await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)
        assert unknown.status == 400
        assert (await client.post("/deid-agent/api/agent", json={"value": "Claude Code"})).status == 401
    finally:
        await client.close()


def echoing_agents():
    """Two scripted agents that say which turn of their session it is and what their prompt held."""
    return [AgentChoice("Claude Code", [sys.executable, str(FAKE), "Claude Code", "echo-prompt"]),
            AgentChoice("Codex", [sys.executable, str(FAKE), "Codex", "echo-prompt"])]


def said(host, after=0):
    return "".join(e["text"] for e in host.events[after:] if e["kind"] == "message")


def prompt_of(text: str) -> list[str]:
    """The prompt's text blocks, as the scripted agent echoed them (the first echo in ``text``)."""
    return json.JSONDecoder().raw_decode(text.split("[prompt ", 1)[1])[0]


async def answer_all(client, host, headers, option="allow"):
    """Answer the asks of the task under way until it ends."""
    answered = set()
    while host.busy or host.handover is not None:
        for e in [e for e in host.events if e["kind"] == "permission" and e["id"] not in answered]:
            answered.add(e["id"])
            await client.post("/deid-agent/api/permission", json={"id": e["id"], "optionId": option}, headers=headers)
        await asyncio.sleep(0.02)


async def test_an_agent_picked_again_goes_on_in_its_session_told_only_what_it_has_not_seen(tmp_path):
    """Caching and the separation of contexts: the agent that worked keeps its session; the other is told,
    in a brief at the head of its prompt, the work it has not seen — the asks, what was said and done, how
    it ended — and not what it has; nobody's reasoning goes across."""
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=echoing_agents())
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        claude = host.agent
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply to the landlord"}, headers=headers)
        await wait_for(lambda: host.busy)
        await answer_all(client, host, headers)
        host.emit({"kind": "thought", "text": "SECRET REASONING of the first model"})
        first = prompt_of(said(host))
        assert first == ["Draft a reply to the landlord"]                   # nothing to tell at first

        await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)
        await wait_for(lambda: host.status == "ready" and host.current.name == "Codex")
        mark = len(host.events)
        await client.post("/deid-agent/api/prompt", json={"text": "Now check the draft"}, headers=headers)
        await wait_for(lambda: host.busy)
        await answer_all(client, host, headers, option="reject")
        told, task = prompt_of(said(host, mark))
        assert "[turn 1]" in said(host, mark) and task == "Now check the draft"
        assert "Task 1 — the person asked Claude Code: Draft a reply to the landlord" in told
        assert "Claude Code said: [Claude Code]" in told and "· Write draft.md — done" in told
        assert "the person allowed it" in told and "Ended: finished." in told
        assert "draft.md (new)" in told and "SECRET REASONING" not in told
        shown = next(e for e in host.events[mark:] if e["kind"] == "brief")
        assert shown == {**shown, "agent": "Codex", "text": told}            # the page shows what was told

        # back to Claude Code: its own session goes on (its second turn), told only Codex's task
        await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        await wait_for(lambda: host.status == "ready" and host.current.name == "Claude Code")
        assert host.agent is claude
        mark = len(host.events)
        await client.post("/deid-agent/api/prompt", json={"text": "Send it"}, headers=headers)
        await wait_for(lambda: host.busy)
        await answer_all(client, host, headers)
        text = said(host, mark)
        told, task = prompt_of(text)
        assert "[turn 2]" in text and task == "Send it"
        # only Codex's task: the one it did itself it is not told (an agent's words may quote anything)
        assert [l for l in told.splitlines() if l.startswith("Task ")] == \
            ["Task 1 — the person asked Codex: Now check the draft"]
        assert "the person refused it" in told

        # a task with no switch before it carries no brief
        mark = len(host.events)
        await client.post("/deid-agent/api/prompt", json={"text": "Thanks"}, headers=headers)
        await wait_for(lambda: host.busy)
        await answer_all(client, host, headers)
        assert prompt_of(said(host, mark)) == ["Thanks"]
        assert not any(e["kind"] == "brief" for e in host.events[mark:])
    finally:
        await client.close()


async def test_a_task_under_way_is_handed_over_and_carried_on(tmp_path):
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=echoing_agents())
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        claude = host.agent
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply to the landlord"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))     # it waits on the person
        ask = next(e for e in host.events if e["kind"] == "permission")

        # the chat's switch (no handover) is refused while it works; the person's handover is not
        assert (await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)).status == 409
        r = await client.post("/deid-agent/api/agent", json={"value": "Codex", "handover": True}, headers=headers)
        assert r.status == 202 and (await r.json())["handover"] == "Codex"
        state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
        assert state["handover"] == "Codex"
        assert (await client.post("/deid-agent/api/prompt", json={"text": "x"}, headers=headers)).status == 409

        await wait_for(lambda: any(e["kind"] == "handover" for e in host.events))
        assert next(e for e in host.events if e["kind"] == "permission_done" and e["id"] == ask["id"])["outcome"] \
            == "cancelled"
        ended = [e for e in host.events if e["kind"] == "turn_end"]
        assert ended[0]["stopReason"] == "cancelled"
        assert claude.proc.returncode is None                                # it keeps its session
        handed = next(e for e in host.events if e["kind"] == "handover")
        assert handed == {**handed, "from": "Claude Code", "to": "Codex"}

        await wait_for(lambda: "[Codex]" in said(host))
        mark = next(i for i, e in enumerate(host.events) if e["kind"] == "handover")
        told, carry = prompt_of(said(host, mark))
        assert "Task 1 — the person asked Claude Code: Draft a reply to the landlord" in told
        assert "· Write draft.md — failed" in told and "did not answer before it stopped" in told
        assert "Ended: stopped so that the person could hand the task over." in told
        assert "Then the person handed the task to you." in told and "Handing the task over" not in told
        assert carry.startswith("The person has handed you the last task above, which Claude Code was doing")
        assert not any(e["kind"] == "user" for e in host.events[mark:])        # the person gave no new task
        await answer_all(client, host, headers)
        assert host.current.name == "Codex" and not host.busy
    finally:
        await client.close()


def test_a_brief_is_the_same_for_the_same_work_and_carries_no_reasoning():
    from deidkit.agents.host import BRIEF_SAID, BRIEF_STEPS, brief
    events = [{"kind": "agents"}, {"kind": "user", "text": "Review the lease", "agent": "Claude Code"},
              {"kind": "thought", "text": "the model's own reasoning"},
              {"kind": "message", "text": "I will read "}, {"kind": "message", "text": "the lease."},
              *[{"kind": "tool", "id": f"r{i}", "title": f"Read part {i}", "status": "pending"} for i in range(50)],
              *[{"kind": "tool_update", "id": f"r{i}", "status": "completed"} for i in range(49)],
              {"kind": "message", "text": "x" * (BRIEF_SAID + 500) + " THE END"},
              {"kind": "usage", "used": 9000, "size": 32768},
              {"kind": "turn_end", "stopReason": "end_turn", "seconds": 12.5, "cost": {"usd": 0.31}},
              {"kind": "files", "changed": ["lease.md"], "added": [], "removed": []}]
    text = brief(events, "Codex")
    assert text == brief(list(events), "Codex")                            # the same work, the same words
    assert "reasoning" not in text.replace("Their reasoning", "") and "12.5" not in text and "0.31" not in text
    assert "Claude Code said: I will read the lease." in text and text.count("Claude Code said:") == 2
    assert "THE END" in text and "x" * (BRIEF_SAID + 1) not in text      # an agent's words: their end
    assert f"({50 - BRIEF_STEPS} earlier steps not listed)" in text and "· Read part 49 — not finished" in text
    assert "Read part 0 —" not in text and "lease.md (changed)" in text
    assert text.endswith("The person's new task follows; the work above is what came before it.")
    assert brief([{"kind": "agents"}, {"kind": "status", "status": "ready"}], "Codex") is None


async def test_an_agent_with_its_own_server_gets_no_key_and_its_requests_go_there(tmp_path):
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[],
                     api_key="sk-org-held-by-proxy", agents=two_agents(upstream="http://10.0.0.7:8092"))
    host, client = await started(cfg, ProxyConfig(default_scope="case-1", upstream_key=cfg.api_key,
                                                  agent_key=cfg.agent_key))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        assert host.proxy.cfg.upstream == "https://api.anthropic.com"
        assert host.proxy.cfg.upstream_key == cfg.api_key and host.agent.env["ANTHROPIC_API_KEY"] == cfg.agent_key

        await client.post("/deid-agent/api/agent", json={"value": "MatterAgent"}, headers=headers)
        await wait_for(lambda: host.status == "ready" and host.current.name == "MatterAgent")
        assert host.proxy.cfg.upstream == "http://10.0.0.7:8092"
        assert host.proxy.cfg.upstream_key is None                     # the key is for Anthropic
        assert "ANTHROPIC_API_KEY" not in host.agent.env

        await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        await wait_for(lambda: host.status == "ready" and host.current.name == "Claude Code")
        assert host.proxy.cfg.upstream == "https://api.anthropic.com"
        assert host.proxy.cfg.upstream_key == cfg.api_key and host.agent.env["ANTHROPIC_API_KEY"] == cfg.agent_key
    finally:
        await client.close()


async def test_an_agent_that_cannot_start_here_is_listed_with_why_and_not_offered(tmp_path):
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=two_agents(unavailable="Claude Code is not on this Mac"))
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        assert host.status == "ready" and host.current.name == "MatterAgent"   # the first that can
        state = await (await client.get("/deid-agent/api/state", headers=headers)).json()
        assert state["agents"]["options"][0] == {"value": "Claude Code", "name": "Claude Code",
                                                 "unavailable": "Claude Code is not on this Mac"}
        r = await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        assert r.status == 409 and "not on this Mac" in (await r.json())["error"]
        assert host.current.name == "MatterAgent"
    finally:
        await client.close()


async def test_one_agent_offers_no_choice(rig):
    host, client, folder = rig
    state = await (await client.get("/deid-agent/api/state",
                                    headers={"x-deid-agent-token": host.cfg.token})).json()
    assert state["agents"] is None
    r = await client.post("/deid-agent/api/agent", json={"value": "fake"},
                          headers={"x-deid-agent-token": host.cfg.token})
    assert r.status == 400


def test_the_agents_come_from_the_command_line_in_their_order(tmp_path, monkeypatch):
    from deidkit.agents import cli

    seen = []
    monkeypatch.setattr(cli, "AgentHost", lambda cfg, proxy: seen.append(cfg) or AgentHost(cfg, proxy))
    monkeypatch.setattr(cli.web, "run_app", lambda *a, **k: None)
    folder = tmp_path / "matter"
    folder.mkdir()
    base = ["--folder", str(folder), "--scope", "case-1", "--dev-login",
            "--vault", str(tmp_path / "vault.sqlite"), "--audit", str(tmp_path / "audit.jsonl")]
    cli.main(base + ["--agent", "Claude Code='/Applications/Bernio Chat Test.app/x/claude-acp'",
                     "--agent", "MatterAgent=/x/matteragent --server llama --model m",
                     "--agent-upstream", "MatterAgent=http://10.0.0.7:8092/",
                     "--agent-unavailable", "Claude Code=Claude Code is not on this Mac"])
    agents = seen[0].agents
    assert [a.name for a in agents] == ["Claude Code", "MatterAgent"]
    assert agents[0].command == ["/Applications/Bernio Chat Test.app/x/claude-acp"]
    assert agents[0].unavailable == "Claude Code is not on this Mac" and agents[0].upstream is None
    assert agents[1].command == ["/x/matteragent", "--server", "llama", "--model", "m"]
    assert agents[1].upstream == "http://10.0.0.7:8092" and agents[1].unavailable is None
    cli.main(base)
    assert seen[1].agents == []                                        # --agent-command alone, as before
    with pytest.raises(SystemExit):
        cli.main(base + ["--agent-upstream", "Nobody=http://x"])
    with pytest.raises(SystemExit):                                    # nothing here could start
        cli.main(base + ["--agent-unavailable", "Claude Code=not here"])


async def test_the_page_takes_the_look_of_the_program_that_opens_it(tmp_path):
    css = tmp_path / "look.css"
    css.write_text(":root { --accent: #8a6a3c; }\n")
    folder = tmp_path / "matter"
    folder.mkdir()
    plain = HostConfig(folder=folder, scope="case-1", title="Case 1",
                       agent_command=[sys.executable, str(FAKE)], dev_login=True)
    styled = HostConfig(folder=folder, scope="case-1", title="Case 1",
                        agent_command=[sys.executable, str(FAKE)], dev_login=True, style=css)
    for cfg, linked in ((plain, False), (styled, True)):
        host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
        try:
            page = await (await client.get("/")).text()
            assert ('href="/deid-agent/style.css"' in page) is linked
            sheet = await client.get("/deid-agent/style.css")             # no token: it is not a secret
            assert sheet.status == (200 if linked else 404)
            if linked:
                assert sheet.content_type == "text/css" and "--accent: #8a6a3c" in await sheet.text()
        finally:
            await client.close()


async def test_a_task_says_what_it_cost_with_the_cache_and_without_it(tmp_path):
    usage_log = tmp_path / "usage.jsonl"
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[sys.executable, str(FAKE)],
                     dev_login=True, usage_log=usage_log)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        # two answers of the model cross the proxy during the task: a cold write, then a read
        host.proxy.on_usage("claude-opus-5-5", {"input_tokens": 2, "cache_creation_input_tokens": 30294,
                                                "cache_creation": {"ephemeral_1h_input_tokens": 30294}, "output_tokens": 194})
        host.proxy.on_usage("claude-opus-5-5", {"input_tokens": 5, "cache_read_input_tokens": 30294, "output_tokens": 60})
        ask = next(e for e in host.events if e["kind"] == "permission")
        await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "reject"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "turn_end" for e in host.events))
        cost = next(e for e in host.events if e["kind"] == "turn_end")["cost"]
        assert cost["requests"] == 2 and cost["cache_write_1h"] == 30294 and cost["cache_read"] == 30294
        assert cost["usd"] == pytest.approx((2 * 4 + 30294 * 8 + 194 * 20 + 5 * 4 + 30294 * 0.2 + 60 * 20) / 1e6, abs=1e-4)
        assert cost["usd_without_cache"] == pytest.approx(((2 + 30294 + 5 + 30294) * 4 + (194 + 60) * 20) / 1e6, abs=1e-4)
        assert json.loads(usage_log.read_text().splitlines()[-1])["cost"] == cost
        # the next task counts its own answers only
        await client.post("/deid-agent/api/prompt", json={"text": "Again"}, headers=headers)
        await wait_for(lambda: sum(e["kind"] == "permission" for e in host.events) == 2)
        ask = [e for e in host.events if e["kind"] == "permission"][-1]
        await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "reject"}, headers=headers)
        await wait_for(lambda: sum(e["kind"] == "turn_end" for e in host.events) == 2)
        assert "cost" not in [e for e in host.events if e["kind"] == "turn_end"][-1]
    finally:
        await client.close()


async def test_an_agent_gets_its_own_session_options(tmp_path):
    meta = {"claudeCode": {"options": {"tools": ["Read", "Write", "Edit", "Bash"], "strictMcpConfig": True}}}
    agents = two_agents()
    agents[0].meta = meta
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True, agents=agents)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        said = "".join(e["text"] for e in host.events if e["kind"] == "message")
        assert f"[meta {json.dumps(meta, sort_keys=True)}]" in said
    finally:
        await client.close()


def test_session_options_come_from_the_command_line(tmp_path, monkeypatch):
    from deidkit.agents import cli

    seen = []
    monkeypatch.setattr(cli, "AgentHost", lambda cfg, proxy: seen.append(cfg) or AgentHost(cfg, proxy))
    monkeypatch.setattr(cli.web, "run_app", lambda *a, **k: None)
    folder = tmp_path / "matter"
    folder.mkdir()
    base = ["--folder", str(folder), "--scope", "case-1", "--dev-login",
            "--vault", str(tmp_path / "vault.sqlite"), "--audit", str(tmp_path / "audit.jsonl")]
    cli.main(base + ["--agent", "Claude Code=/x/claude-acp", "--agent", "MatterAgent=/x/matteragent",
                     "--agent-meta", 'Claude Code={"claudeCode": {"options": {"tools": ["Read"]}}}'])
    assert seen[0].agents[0].meta == {"claudeCode": {"options": {"tools": ["Read"]}}} and seen[0].agents[1].meta is None
    cli.main(base + ["--session-meta", '{"claudeCode": {"options": {"strictMcpConfig": true}}}'])
    assert seen[1].session_meta == {"claudeCode": {"options": {"strictMcpConfig": True}}}
    with pytest.raises(SystemExit):
        cli.main(base + ["--session-meta", "[1, 2]"])


async def test_codex_goes_through_the_proxy_with_its_own_sign_in(tmp_path):
    agents = [AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])]
    assert agents[0].codex and not two_agents()[0].codex
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], api_key="sk-org",
                     codex_executable="/opt/codex/bin/codex", agents=agents, codex_confine=False)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        said = "".join(e["text"] for e in host.events if e["kind"] == "message")
        provider = said.split("[provider ", 1)[1].split("] ", 1)[0]
        name, config, path, key, mode = provider.split(" ", 1)[0], *provider.split(" ", 1)[1].rsplit(" ", 3)
        assert name == "deid" and path == "/opt/codex/bin/codex"
        assert mode == "read-only"            # Codex asks the person before every edit
        assert key == "-"                     # the organization's key is Anthropic's: Codex is not lent it
        deid = json.loads(config)["model_providers"]["deid"]
        assert deid["base_url"] == f"http://127.0.0.1:{cfg.port}/v1" and deid["requires_openai_auth"] is True
        assert deid["http_headers"] == {"x-deid-scope": "case-1"} and deid["wire_api"] == "responses"
        assert json.loads(config)["features"] == {"enable_request_compression": False}
    finally:
        await client.close()


def test_the_persons_codex_comes_from_the_command_line(tmp_path, monkeypatch):
    from deidkit.agents import cli

    seen = []
    monkeypatch.setattr(cli, "AgentHost", lambda cfg, proxy: seen.append(cfg) or AgentHost(cfg, proxy))
    monkeypatch.setattr(cli.web, "run_app", lambda *a, **k: None)
    folder = tmp_path / "matter"
    folder.mkdir()
    cli.main(["--folder", str(folder), "--scope", "case-1", "--dev-login", "--vault", str(tmp_path / "v.sqlite"),
              "--audit", str(tmp_path / "a.jsonl"), "--agent", "Codex=/x/codex-acp",
              "--codex-executable", "/Applications/Some.app/codex"])
    assert seen[0].codex_executable == "/Applications/Some.app/codex" and seen[0].agents[0].codex
    assert seen[0].codex_confine                      # confined unless the person says otherwise
    cli.main(["--folder", str(folder), "--scope", "case-1", "--dev-login", "--vault", str(tmp_path / "v.sqlite"),
              "--audit", str(tmp_path / "a.jsonl"), "--agent", "Codex=/x/codex-acp", "--codex-unconfined"])
    assert seen[1].codex_confine is False


async def test_codex_reaches_openai_with_its_own_sign_in_while_the_host_holds_the_organizations_key(tmp_path):
    # MEASURED 08.10.2026, in review: holding the key, the proxy refused every request without the
    # stand-in (401) and stripped Authorization — Codex on a ChatGPT sign-in could not answer at all.
    from aiohttp import web
    from test_proxy_openai import HEADERS, Backend, request, seeds
    backend = Backend()
    app = web.Application()
    app.router.add_post("/codex/responses", backend.responses)
    upstream = TestServer(app)
    await upstream.start_server()
    agents = [AgentChoice("Claude Code", [sys.executable, str(FAKE), "Claude Code"]),
              AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])]
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], api_key="sk-org", agents=agents,
                     codex_confine=False)
    proxy_cfg = ProxyConfig(default_scope="case-1", upstream_chatgpt=str(upstream.make_url("/codex")),
                            detector=RegexDetector(), seeds=seeds())
    host, client = await started(cfg, proxy_cfg)
    headers = {"x-deid-agent-token": cfg.token}
    try:
        assert host.proxy.cfg.upstream_key == "sk-org"          # Claude Code works first, with the key
        r = await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)
        assert r.status == 202
        await wait_for(lambda: host.status == "ready")
        assert host.proxy.cfg.upstream_key is None
        r = await client.post("/v1/responses", json=request(), headers=HEADERS)
        assert r.status == 200
        await r.read()
        assert backend.headers[0]["Authorization"] == "Bearer t"
        r = await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        await wait_for(lambda: host.status == "ready")
        assert host.proxy.cfg.upstream_key == "sk-org"          # and the key is back for Claude Code
    finally:
        await client.close()
        await upstream.close()


async def test_an_agent_in_openais_shape_with_its_own_server_gets_its_responses_there(tmp_path):
    agents = [AgentChoice("MatterAgent", [sys.executable, str(FAKE), "MatterAgent", "--wire", "responses"],
                          upstream="http://127.0.0.1:9/"),
              AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])]
    assert agents[0].responses and agents[1].responses and not two_agents()[0].responses
    assert AgentChoice("x", ["m", "--wire=responses"]).responses
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True, agents=agents,
                     codex_confine=False)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    headers = {"x-deid-agent-token": cfg.token}
    try:
        assert host.proxy.cfg.upstream_openai == "http://127.0.0.1:9/v1"
        assert host.proxy.cfg.upstream == "http://127.0.0.1:9"
        await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)
        await wait_for(lambda: host.status == "ready" and host.current.name == "Codex")
        assert host.proxy.cfg.upstream_openai == "https://api.openai.com/v1"     # Codex goes to OpenAI again
    finally:
        await client.close()

