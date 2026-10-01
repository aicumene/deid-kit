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

from deidkit.agents.host import AgentHost, HostConfig, model_choice  # noqa: E402
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

