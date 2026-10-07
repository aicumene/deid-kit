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
        first = host.agent

        r = await client.post("/deid-agent/api/agent", json={"value": "MatterAgent"}, headers=headers)
        assert r.status == 202
        await wait_for(lambda: host.status == "ready" and host.agent is not first)
        assert first.proc.returncode is not None                       # the other one stopped
        assert host.current.name == "MatterAgent" and host.models["current"] == "default"
        kinds = [e["kind"] for e in host.events]
        assert kinds.count("agents") == 2 and {"kind": "models", "options": []} in \
            [{k: v for k, v in e.items() if k != "at"} for e in host.events]

        # the next task goes to it, in its own session
        await client.post("/deid-agent/api/prompt", json={"text": "Draft a reply"}, headers=headers)
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        assert "[MatterAgent] Working on it" in "".join(e["text"] for e in host.events if e["kind"] == "message")
        busy = await client.post("/deid-agent/api/agent", json={"value": "Claude Code"}, headers=headers)
        assert busy.status == 409 and host.current.name == "MatterAgent"
        ask = next(e for e in host.events if e["kind"] == "permission")
        await client.post("/deid-agent/api/permission", json={"id": ask["id"], "optionId": "reject"},
                          headers=headers)
        await wait_for(lambda: not host.busy)

        unknown = await client.post("/deid-agent/api/agent", json={"value": "Codex"}, headers=headers)
        assert unknown.status == 400
        assert (await client.post("/deid-agent/api/agent", json={"value": "Claude Code"})).status == 401
    finally:
        await client.close()


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
