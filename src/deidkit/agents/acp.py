# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A client for the Agent Client Protocol (ACP, version 1): run a coding agent as a child process.

ACP is JSON-RPC 2.0 over the agent's stdin and stdout, one message per line. The client starts
the agent, calls ``initialize``, opens a session on a folder (``session/new``) and sends prompts
(``session/prompt``). While a prompt runs, the agent streams ``session/update`` notifications:
message and thought chunks, tool calls and their updates, plans. It may also call the client
back with ``session/request_permission`` before a tool acts. Claude Code and Codex both speak
ACP through open adapters (``@agentclientprotocol/claude-agent-acp``,
``@agentclientprotocol/codex-acp``).

This client advertises neither file-system nor terminal capabilities: the agent reads and
writes with its own tools, inside its own permission checks, and every permission request is
decided by the ``on_permission`` callback, which a host hands to a person.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("deidkit.agents.acp")

PROTOCOL_VERSION = 1

Update = dict[str, Any]
PermissionDecider = Callable[[dict], Awaitable[dict]]


class AcpError(RuntimeError):
    def __init__(self, error: dict) -> None:
        self.error = error
        super().__init__(f"ACP error {error.get('code')}: {error.get('message')}")


class AcpAgent:
    """One agent process and its JSON-RPC connection."""

    def __init__(self, command: list[str], *, env: dict[str, str], cwd: str,
                 on_update: Callable[[Update], Awaitable[None]],
                 on_permission: PermissionDecider) -> None:
        self.command = command
        self.env = env
        self.cwd = cwd
        self.on_update = on_update
        self.on_permission = on_permission
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._stderr: asyncio.Task | None = None
        self.agent_info: dict = {}

    async def start(self) -> dict:
        self.proc = await asyncio.create_subprocess_exec(
            *self.command, cwd=self.cwd, env=self.env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=64 * 1024 * 1024)
        self._reader = asyncio.create_task(self._read())
        self._stderr = asyncio.create_task(self._drain_stderr())
        self.agent_info = await self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False},
                                   "terminal": False},
            "clientInfo": {"name": "deid-agent", "version": "0.1"},
        })
        return self.agent_info

    async def new_session(self, cwd: str) -> str:
        result = await self.request("session/new", {"cwd": cwd, "mcpServers": []})
        return result["sessionId"]

    async def prompt(self, session_id: str, text: str) -> dict:
        return await self.request("session/prompt", {
            "sessionId": session_id, "prompt": [{"type": "text", "text": text}]})

    async def cancel(self, session_id: str) -> None:
        await self._send({"jsonrpc": "2.0", "method": "session/cancel",
                          "params": {"sessionId": session_id}})

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
        for task in (self._reader, self._stderr):
            if task:
                task.cancel()

    # ── JSON-RPC ─────────────────────────────────────────────────────────────
    async def request(self, method: str, params: dict) -> Any:
        self._next_id += 1
        rid = self._next_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return await fut

    async def _send(self, message: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def _read(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    log.warning("agent wrote a line that is not JSON-RPC")
                    continue
                await self._dispatch(msg)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AcpError({"code": -32000, "message": "the agent exited"}))

    async def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                if "error" in msg:
                    fut.set_exception(AcpError(msg["error"]))
                else:
                    fut.set_result(msg.get("result"))
            return
        method, params = msg.get("method"), msg.get("params") or {}
        if method == "session/update":
            await self.on_update(params)
            return
        if "id" not in msg:
            return                                   # another notification: not needed here
        if method == "session/request_permission":
            asyncio.create_task(self._answer(msg["id"], self.on_permission(params)))
            return
        await self._send({"jsonrpc": "2.0", "id": msg["id"],
                          "error": {"code": -32601, "message": f"method not supported: {method}"}})

    async def _answer(self, rid, pending: Awaitable[dict]) -> None:
        try:
            outcome = await pending
        except Exception:                            # noqa: BLE001 — a failed decision is a refusal
            log.exception("permission decision failed")
            outcome = {"outcome": "cancelled"}
        await self._send({"jsonrpc": "2.0", "id": rid, "result": {"outcome": outcome}})

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            log.debug("agent stderr: %s", line.decode("utf-8", "replace").rstrip()[:300])


def agent_env(*, base_url: str, scope: str, api_key: str | None, extra: dict | None = None) -> dict:
    """The environment for a Claude agent behind the proxy: a minimal inherited base, the proxy
    as its only endpoint, and the scope. Without ``api_key`` the agent falls back to the local
    sign-in, which a product must not offer to its users (see the Agent SDK's terms)."""
    keep = ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG", "SHELL", "TERM")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_CUSTOM_HEADERS": f"x-deid-scope: {scope}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    env.update(extra or {})
    return env


__all__ = ["AcpAgent", "AcpError", "PROTOCOL_VERSION", "agent_env"]
