# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""deid-agent end to end with a scripted ACP agent: the token, a permission answered by a person,
a path outside the folder refused without asking, the changed files reported."""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer       # noqa: E402

from deidkit.agents.host import AgentHost, HostConfig       # noqa: E402
from deidkit.proxy.server import Proxy, ProxyConfig         # noqa: E402
from deidkit.store import InMemoryTokenStore                # noqa: E402

FAKE = Path(__file__).with_name("fake_acp_agent.py")


async def wait_for(pred, timeout=10.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


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
