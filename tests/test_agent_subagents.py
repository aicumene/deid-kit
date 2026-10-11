# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Sub-agents on the organization's own servers (matteragent's): Claude Code and Codex get them as
the MCP server "matter" and lose their own cloud sub-agents, matteragent gets its hosts file, and
without sub-agents nothing changes. Their questions reach this side in names, through either
provider's shape; Codex's confinement opens what the server must read and nothing wide."""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from deidkit.agents.confine import profile                               # noqa: E402
from deidkit.agents.host import AgentChoice, HostConfig, with_subagents  # noqa: E402
from deidkit.proxy.anthropic import LOCAL_TOOLS, SUBAGENT_TOOLS, restore_json_response  # noqa: E402
from deidkit.proxy.openai import LOCAL_TOOLS as OPENAI_LOCAL, restore_item, tool_name  # noqa: E402
from deidkit.proxy.server import MemoryRestore, ProxyConfig             # noqa: E402

from test_agent_host import FAKE, started, wait_for                    # noqa: E402
from test_proxy_openai import engine, request                           # noqa: E402
from deidkit.proxy.openai import prepare_responses                      # noqa: E402

COMMAND = ["/opt/runtime/bin/python", "-m", "matteragent.mcp"]


@pytest.fixture
def hosts(tmp_path):
    path = tmp_path / "hosts.toml"
    path.write_text('[[host]]\nname = "thor"\nurl = "http://10.0.0.1:8092"\nmodel = "m"\nkinds = ["read"]\n')
    return path


def config(tmp_path, hosts, agent, **kw):
    folder = tmp_path / "matter"
    folder.mkdir(exist_ok=True)
    return HostConfig(folder=folder, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                      agents=[agent], subagents=hosts, subagents_command=COMMAND, codex_confine=False, **kw)


async def said_by(cfg):
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Read the lease"},
                          headers={"x-deid-agent-token": cfg.token})
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        return "".join(e["text"] for e in host.events if e["kind"] == "message")
    finally:
        await client.close()


def bracket(said, tag):
    return json.loads(said.split(f"[{tag} ", 1)[1].split("] ", 1)[0])


async def test_claude_gets_the_matter_server_and_its_own_sub_agents_are_off(tmp_path, hosts):
    meta = {"claudeCode": {"options": {"tools": ["Read", "Write", "Edit", "Bash"], "strictMcpConfig": True}}}
    agent = AgentChoice("Claude Code", [sys.executable, str(FAKE), "claude-agent-acp"], meta=meta)
    cfg = config(tmp_path, hosts, agent)
    said = await said_by(cfg)
    server = bracket(said, "mcp")[0]
    assert server["name"] == "matter" and server["command"] == COMMAND[0] and server["env"] == []
    assert server["args"] == COMMAND[1:] + ["--folder", str(cfg.folder), "--hosts", str(hosts),
                                            "--author-family", "claude"]
    options = bracket(said, "meta")["claudeCode"]["options"]
    assert options["disallowedTools"] == ["Task"] and options["tools"] == ["Read", "Write", "Edit", "Bash"]
    # the sub-agents only read: allowed without asking each time
    assert "mcp__matter__read" in options["allowedTools"] and "mcp__matter__consult" in options["allowedTools"]
    assert "[mcp-timeout 900000]" in said
    assert meta["claudeCode"]["options"].get("disallowedTools") is None      # the given options untouched


async def test_codex_gets_the_matter_server_where_it_starts_and_waits_for_it(tmp_path, hosts):
    agent = AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])
    cfg = config(tmp_path, hosts, agent, codex_executable="/opt/codex/bin/codex")
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    try:
        await client.post("/deid-agent/api/prompt", json={"text": "Read the lease"},
                          headers={"x-deid-agent-token": cfg.token})
        await wait_for(lambda: any(e["kind"] == "permission" for e in host.events))
        said = "".join(e["text"] for e in host.events if e["kind"] == "message")
        # Codex's own cloud sub-agents are not offered to its model: the proxy takes them out
        assert host.proxy.cfg.drop_tool_namespaces == frozenset({"collaboration"})
    finally:
        await client.close()
    provider = said.split("[provider ", 1)[1].split("] ", 1)[0]
    settings = json.loads(provider.split(" ", 1)[1].rsplit(" ", 3)[0])
    assert "mcp_servers" not in settings and settings["features"] == {"enable_request_compression": False}
    # defined whole where Codex starts, with the timeouts the adapter's own entry lacks
    wrapper = bracket(said, "wrapper")
    for part in (f"mcp_servers.matter.command={json.dumps(COMMAND[0])}", "mcp_servers.matter.tool_timeout_sec=900",
                 "mcp_servers.matter.startup_timeout_sec=30", '--author-family", "gpt"', "developer_instructions=",
                 'mcp_servers.matter.default_tools_approval_mode="approve"'):
        assert part in wrapper, part
    assert "mcp__matter__<kind>" in wrapper                     # told of the tools it keeps deferred
    # named in the session too, so the adapter waits for it before the first task
    assert bracket(said, "mcp")[0]["name"] == "matter"
    assert bracket(said, "meta") == {"mcpStartupAwaitTimeoutMs": 30000}


async def test_switching_to_an_agent_without_sub_agents_offers_codex_everything_again(tmp_path):
    folder = tmp_path / "matter"
    folder.mkdir()
    cfg = HostConfig(folder=folder, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     agents=[AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])], codex_confine=False)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1", drop_tool_namespaces=frozenset({"x"})))
    try:
        assert host.proxy.cfg.drop_tool_namespaces == frozenset()
    finally:
        await client.close()


def test_the_proxy_takes_codexs_own_sub_agents_out_of_a_request():
    from deidkit.proxy.openai import drop_namespaces
    body = {"input": [{"type": "additional_tools", "role": "developer", "tools": [
        {"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "exec"}]},
        {"type": "namespace", "name": "collaboration", "tools": [{"type": "function", "name": "spawn_agent"}]},
        {"type": "namespace", "name": "mcp__matter", "tools": [{"type": "function", "name": "read"}]}]},
        {"type": "message", "role": "user", "content": "Read the lease"}],
        "tools": [{"type": "namespace", "name": "collaboration", "tools": []}, {"type": "function", "name": "x"}]}
    assert drop_namespaces(body, frozenset({"collaboration"})) == 2
    assert [t["name"] for t in body["input"][0]["tools"]] == ["functions", "mcp__matter"]
    assert body["tools"] == [{"type": "function", "name": "x"}]
    assert drop_namespaces(body, frozenset()) == 0


async def test_matteragent_gets_its_hosts_and_no_mcp_server(tmp_path, hosts):
    agent = AgentChoice("MatterAgent", [sys.executable, str(FAKE), "matteragent"])
    said = await said_by(config(tmp_path, hosts, agent))
    assert bracket(said, "argv") == ["--subagents", str(hosts)]
    assert "[mcp " not in said and "[mcp-timeout" not in said


async def test_without_sub_agents_nothing_changes(tmp_path):
    meta = {"claudeCode": {"options": {"tools": ["Read"]}}}
    agent = AgentChoice("Claude Code", [sys.executable, str(FAKE), "claude-agent-acp"], meta=meta)
    folder = tmp_path / "matter"
    folder.mkdir()
    said = await said_by(HostConfig(folder=folder, scope="case-1", title="Case 1", agent_command=[],
                                    dev_login=True, agents=[agent]))
    assert "[mcp " not in said and "[mcp-timeout" not in said and bracket(said, "meta") == meta


def test_the_task_tool_and_the_matter_tools_are_added_to_what_is_already_there():
    meta = {"claudeCode": {"options": {"disallowedTools": ["WebFetch", "Task"], "allowedTools": ["Read"]}}}
    options = with_subagents(meta)["claudeCode"]["options"]
    assert options["disallowedTools"] == ["WebFetch", "Task"]
    assert options["allowedTools"][0] == "Read" and "mcp__matter__check" in options["allowedTools"]
    assert with_subagents(None)["claudeCode"]["options"]["disallowedTools"] == ["Task"]


def test_the_kinds_of_agent_are_told_apart():
    assert AgentChoice("a", ["npx", "-y", "@agentclientprotocol/claude-agent-acp@0.85.0"]).claude
    assert AgentChoice("a", ["/App/Contents/Resources/agent-runtime/bin/claude-acp"]).claude
    assert AgentChoice("a", ["/runtime/bin/matteragent", "--model", "m"]).matteragent
    assert AgentChoice("a", ["/venv/bin/python", "-m", "matteragent", "--model", "m"]).matteragent
    assert not AgentChoice("a", ["/venv/bin/python", "-m", "matteragent.mcp"]).matteragent
    codex = AgentChoice("a", ["npx", "@agentclientprotocol/codex-acp"])
    assert codex.codex and not codex.claude and not codex.matteragent


# ── names come back in a sub-agent's question ─────────────────────────────────────────────────

def test_every_sub_agent_tool_is_local_for_both_providers():
    assert SUBAGENT_TOOLS <= LOCAL_TOOLS and SUBAGENT_TOOLS <= OPENAI_LOCAL
    assert {"mcp__matter__read", "mcp__matter__consult", "Subagent", "subagent"} <= SUBAGENT_TOOLS
    assert "mcp__crm__lookup" not in LOCAL_TOOLS                  # another server's tool is not


async def _mapping():
    eng = engine()
    prep = await prepare_responses(request(), eng, headers={}, restore_get=MemoryRestore().restore_get)
    token = next(t for t, v in prep.mapping.items() if "Brenner" in v and t.startswith("PERSON"))
    return eng, prep.mapping, token


async def test_claudes_call_to_a_sub_agent_carries_the_real_name():
    eng, mapping, token = await _mapping()
    data = {"content": [
        {"type": "tool_use", "id": "t1", "name": "mcp__matter__read",
         "input": {"question": f"What does {token} owe?", "paths": ["documents/lease.md"]}},
        {"type": "tool_use", "id": "t2", "name": "mcp__crm__lookup", "input": {"q": token}}]}
    out = await restore_json_response(data, eng, mapping, MemoryRestore().restore_put)
    assert "Brenner" in out["content"][0]["input"]["question"]
    assert out["content"][1]["input"]["q"] == token                 # another server keeps the token


async def test_codexs_call_to_a_sub_agent_carries_the_real_name_namespaced_or_in_a_script():
    eng, mapping, token = await _mapping()
    spaced = {"type": "function_call", "namespace": "mcp__matter", "name": "read", "call_id": "c1",
              "arguments": json.dumps({"question": f"What does {token} owe?"})}
    assert tool_name(spaced) == "mcp__matter__read"
    assert "Brenner" in json.loads((await restore_item(spaced, eng, mapping, None))["arguments"])["question"]
    script = {"type": "custom_tool_call", "name": "exec",
              "input": f'text(await tools.mcp__matter__read({{question: "What does {token} owe?"}}));'}
    assert "Brenner" in (await restore_item(script, eng, mapping, None))["input"]
    mixed = {"type": "custom_tool_call", "name": "exec", "input":
             f'text(await tools.mcp__matter__read({{question: "{token}"}}));'
             f'text(await tools.mcp__crm__lookup({{q: "{token}"}}));'}
    assert (await restore_item(mixed, eng, mapping, None))["input"] == mixed["input"]


# ── Codex's confinement opens what the server reads ───────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS's")
def test_confined_codex_starts_with_the_settings_it_takes_only_at_its_start(tmp_path):
    from deidkit.agents.confine import confined_codex
    wrapper = confined_codex(tmp_path / "case", sys.executable, tmp_path / "w", tmp_path / "h",
                             args=["--disable", "multi_agent"])
    assert wrapper.read_text().rstrip().endswith(' --disable multi_agent "$@"')


def test_confined_codex_may_read_what_the_matter_server_needs_and_nothing_wide(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "lib").mkdir(parents=True)
    hosts = tmp_path / "hosts.toml"
    hosts.write_text("")
    text = profile(tmp_path / "matters" / "case", "/opt/codex/bin/codex", tmp_path / "codex-home",
                   extra_read=[runtime, hosts, "/", os.path.expanduser("~"), tmp_path / "missing"])
    real = lambda p: os.path.realpath(str(p))                       # noqa: E731
    assert f'(allow file-read* (subpath "{real(runtime)}"))' in text
    assert f'(allow file-read* (literal "{real(hosts)}"))' in text
    assert '(subpath "/"))' not in text and f'(subpath "{real(os.path.expanduser("~"))}"))' not in text
    assert "missing" not in text
    assert text.index("(deny ") < text.index(real(runtime))         # the later rule wins


def test_a_wrapper_in_an_installation_opens_the_installation(tmp_path, hosts):
    from deidkit.agents.host import AgentHost
    from deidkit.proxy.server import Proxy
    from deidkit.store import InMemoryTokenStore
    runtime = tmp_path / "agent-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "python" / "bin").mkdir(parents=True)
    wrapper = runtime / "bin" / "matteragent-mcp"
    wrapper.write_text('#!/bin/sh\nexec "$(dirname "$0")/../python/bin/python3" -m matteragent.mcp "$@"\n')
    wrapper.chmod(0o755)
    cfg = config(tmp_path, hosts, AgentChoice("Codex", ["codex-acp"]))
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1")))
    assert os.path.realpath(runtime) in host.subagent_paths({"command": str(wrapper), "args": []})


def test_the_paths_the_server_needs_include_its_interpreter_and_hosts(tmp_path, hosts):
    from deidkit.agents.host import AgentHost
    from deidkit.proxy.server import Proxy
    from deidkit.store import InMemoryTokenStore
    cfg = config(tmp_path, hosts, AgentChoice("Codex", ["codex-acp"]))
    host = AgentHost(cfg, Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="case-1")))
    paths = host.subagent_paths({"command": sys.executable, "args": ["-m", "matteragent.mcp"]})
    assert str(hosts) in paths
    assert os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))) in paths
    assert Path(sys.prefix).resolve().as_posix() in {Path(p).resolve().as_posix() for p in paths}
