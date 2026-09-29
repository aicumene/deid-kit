# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-agent``: a coding agent over one folder, behind deid-proxy, with a page to drive it.

One process on one loopback port holds three things:

* **the proxy** (``/v1/…``): the agent's only endpoint. Everything it sends to its model is
  tokenised, and everything that comes back is put back into names (``deidkit.proxy``);
* **the agent**, started as a child over ACP (``deidkit.agents.acp``) with the folder as its
  working directory, the proxy as its base URL and the folder's scope as its header;
* **the page and its API** (``/`` and ``/deid-agent/api/…``): a person gives the task, watches the
  work, answers each permission request (an edit, a command), and reads the files that changed.

**The API needs a token.** The agent runs on the same machine and could otherwise call the API
itself and approve its own edit. The token is made at start and handed to the page in the URL
fragment (``#t=…``), which a browser never sends to a server; the page sends it back in a header
only, never in an address, so it appears in no request line, log or file the agent can read.

**The agent never holds the organization's key.** The proxy keeps it and gives the agent a
stand-in made for this run (``agent_key``), swapping one for the other on the way out. A program
that starts ``deid-agent`` hands over the token and the key on stdin (``--secrets-stdin``): other
processes of the same user, the agent's commands among them, can read a process's environment
and arguments (``ps -E``), not what came through its stdin.

A permission request that names a path outside the folder is refused without asking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from deidkit.agents.acp import AcpAgent, agent_env
from deidkit.proxy.server import Proxy

log = logging.getLogger("deidkit.agents.host")

_SKIP_DIRS = {".git", ".claude", ".codex", "node_modules", "__pycache__", ".venv"}
_MAX_FILE = 2 * 1024 * 1024


@dataclass
class HostConfig:
    folder: Path
    scope: str
    title: str
    agent_command: list[str]
    api_key: str | None = None
    dev_login: bool = False
    claude_executable: str | None = None
    port: int = 8790
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    #: What the agent sends as its key; the proxy swaps it for ``api_key``, which stays there.
    agent_key: str = field(default_factory=lambda: "deid-agent-" + secrets.token_urlsafe(24))


def snapshot(folder: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for the folder's files, hidden tool folders left out."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            p = Path(root, name)
            try:
                if p.stat().st_size <= _MAX_FILE:
                    out[str(p.relative_to(folder))] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                continue
    return out


def inside(folder: Path, path: str) -> bool:
    try:
        target = Path(path)
        target = target if target.is_absolute() else folder / target
        return os.path.realpath(target).startswith(os.path.realpath(folder) + os.sep) or \
            os.path.realpath(target) == os.path.realpath(folder)
    except (OSError, ValueError):
        return False


class AgentHost:
    def __init__(self, cfg: HostConfig, proxy: Proxy) -> None:
        self.cfg = cfg
        self.proxy = proxy
        self.agent: AcpAgent | None = None
        self.session_id: str | None = None
        self.events: list[dict] = []
        self.listeners: set[asyncio.Queue] = set()
        self.permissions: dict[str, asyncio.Future] = {}
        self.busy = False
        self.baseline = snapshot(cfg.folder)
        self.status = "starting"
        self.key_refused_told = False
        proxy.on_key_refused = self.key_refused

    # ── events ───────────────────────────────────────────────────────────────
    def emit(self, event: dict) -> None:
        event = {"at": time.time(), **event}
        self.events.append(event)
        for q in list(self.listeners):
            q.put_nowait(event)

    def changes(self) -> dict:
        now = snapshot(self.cfg.folder)
        return {"changed": sorted(p for p in now if p in self.baseline and now[p] != self.baseline[p]),
                "added": sorted(p for p in now if p not in self.baseline),
                "removed": sorted(p for p in self.baseline if p not in now),
                "files": sorted(now)}

    # ── the agent ────────────────────────────────────────────────────────────
    async def start_agent(self) -> None:
        extra = {"CLAUDE_CODE_EXECUTABLE": self.cfg.claude_executable} if self.cfg.claude_executable else {}
        env = agent_env(base_url=f"http://127.0.0.1:{self.cfg.port}", scope=self.cfg.scope,
                        api_key=self.cfg.agent_key if self.cfg.api_key else None, extra=extra)
        self.agent = AcpAgent(self.cfg.agent_command, env=env, cwd=str(self.cfg.folder),
                              on_update=self.on_update, on_permission=self.on_permission)
        try:
            info = await self.agent.start()
            self.session_id = await self.agent.new_session(str(self.cfg.folder))
            self.status = "ready"
            self.emit({"kind": "status", "status": "ready",
                       "agent": (info.get("agentInfo") or {}).get("name", "agent")})
        except Exception as exc:                     # noqa: BLE001
            self.status = "failed"
            log.exception("the agent did not start")
            self.emit({"kind": "error", "text": f"The agent did not start: {exc}"})

    async def on_update(self, params: dict) -> None:
        u = params.get("update") or {}
        kind = u.get("sessionUpdate")
        content = u.get("content") or {}
        if kind in ("agent_message_chunk", "agent_thought_chunk") and content.get("type") == "text":
            self.emit({"kind": "message" if kind == "agent_message_chunk" else "thought",
                       "text": content.get("text", "")})
        elif kind == "tool_call":
            self.emit({"kind": "tool", "id": u.get("toolCallId"), "title": u.get("title", ""),
                       "tool": u.get("kind"), "status": u.get("status", "pending")})
        elif kind == "tool_call_update":
            self.emit({"kind": "tool_update", "id": u.get("toolCallId"),
                       "status": u.get("status"), "title": u.get("title")})
        elif kind == "plan":
            self.emit({"kind": "plan", "entries": [
                {"content": e.get("content", ""), "status": e.get("status", "")}
                for e in u.get("entries") or []]})

    async def on_permission(self, params: dict) -> dict:
        call = params.get("toolCall") or {}
        options = params.get("options") or []
        paths = [loc.get("path", "") for loc in call.get("locations") or [] if loc.get("path")]
        outside = [p for p in paths if not inside(self.cfg.folder, p)]
        if outside:
            reject = next((o for o in options if o.get("kind", "").startswith("reject")), None)
            self.emit({"kind": "notice", "text": "Refused without asking: the action names a "
                                                 "path outside the matter folder."})
            return {"outcome": "selected", "optionId": reject["optionId"]} if reject \
                else {"outcome": "cancelled"}
        rid = secrets.token_hex(8)
        fut = asyncio.get_running_loop().create_future()
        self.permissions[rid] = fut
        self.emit({"kind": "permission", "id": rid, "title": call.get("title", "An action"),
                   "tool": call.get("kind"), "paths": [os.path.relpath(p, self.cfg.folder)
                                                      if os.path.isabs(p) else p for p in paths],
                   "options": [{"optionId": o.get("optionId"), "name": o.get("name"),
                                "kind": o.get("kind")} for o in options]})
        try:
            return await fut
        finally:
            self.permissions.pop(rid, None)
            self.emit({"kind": "permission_done", "id": rid})

    def key_refused(self) -> None:
        """The provider refused the organization's key. The agent would retry for minutes behind
        a silent page, so the person is told, and a running turn is stopped."""
        if self.key_refused_told:
            return
        self.key_refused_told = True
        if self.busy and self.agent and self.session_id:
            asyncio.get_running_loop().create_task(self.agent.cancel(self.session_id))
            text = "Anthropic refused the organization's key, so the turn was stopped."
        else:
            text = "Anthropic refuses the organization's key."
        self.emit({"kind": "error", "text": text + " The key is wrong or revoked: replace it, "
                                                   "then open the agent again."})

    async def run_prompt(self, text: str) -> None:
        self.busy = True
        self.key_refused_told = False
        self.emit({"kind": "user", "text": text})
        try:
            result = await self.agent.prompt(self.session_id, text)
            self.emit({"kind": "turn_end", "stopReason": result.get("stopReason")})
        except Exception as exc:                     # noqa: BLE001
            self.emit({"kind": "error", "text": f"The turn failed: {exc}"})
        finally:
            self.busy = False
            self.emit({"kind": "files", **self.changes()})

    # ── HTTP ─────────────────────────────────────────────────────────────────
    def _authorised(self, request: web.Request) -> bool:
        given = request.headers.get("x-deid-agent-token", "")          # a header, never the address
        return secrets.compare_digest(given.encode(), self.cfg.token.encode())

    async def page(self, request: web.Request) -> web.Response:
        html = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def api(self, request: web.Request) -> web.StreamResponse:
        if not self._authorised(request):
            return web.json_response({"error": "unauthorised"}, status=401)
        name = request.match_info["name"]
        if name == "state" and request.method == "GET":
            auth = "organization key" if self.cfg.api_key else "personal sign-in (development)"
            return web.json_response({"title": self.cfg.title, "folder": str(self.cfg.folder),
                                      "scope": self.cfg.scope, "status": self.status,
                                      "busy": self.busy, "auth": auth, **self.changes()})
        if name == "events" and request.method == "GET":
            return await self._events(request)
        if name == "prompt" and request.method == "POST":
            text = ((await request.json()).get("text") or "").strip()
            if not text:
                return web.json_response({"error": "empty task"}, status=400)
            if self.busy or self.status != "ready":
                return web.json_response({"error": "the agent is busy or not ready"}, status=409)
            asyncio.create_task(self.run_prompt(text))
            return web.json_response({"ok": True}, status=202)
        if name == "permission" and request.method == "POST":
            body = await request.json()
            fut = self.permissions.get(body.get("id", ""))
            if fut is None or fut.done():
                return web.json_response({"error": "no such request"}, status=404)
            option = body.get("optionId")
            fut.set_result({"outcome": "selected", "optionId": option} if option
                           else {"outcome": "cancelled"})
            return web.json_response({"ok": True})
        if name == "cancel" and request.method == "POST":
            if self.agent and self.session_id:
                await self.agent.cancel(self.session_id)
            return web.json_response({"ok": True})
        if name == "file" and request.method == "GET":
            rel = request.query.get("path", "")
            target = self.cfg.folder / rel
            if not inside(self.cfg.folder, str(target)) or not target.is_file():
                return web.json_response({"error": "no such file"}, status=404)
            data = target.read_bytes()[:_MAX_FILE]
            return web.json_response({"path": rel, "text": data.decode("utf-8", "replace")})
        return web.json_response({"error": "not found"}, status=404)

    async def _events(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                           "Cache-Control": "no-store"})
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        for event in self.events:
            q.put_nowait(event)
        self.listeners.add(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), 15)
                    await resp.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.listeners.discard(q)
        return resp

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_get("/", self.page)
        app.router.add_route("*", "/deid-agent/api/{name}", self.api)
        app.router.add_route("*", "/{tail:.*}", self.proxy.handle)      # the agent's endpoint

        async def startup(app):
            await self.proxy._start(app)
            asyncio.create_task(self.start_agent())

        async def cleanup(app):
            if self.agent:
                await self.agent.stop()
            await self.proxy._stop(app)

        app.on_startup.append(startup)
        app.on_cleanup.append(cleanup)
        return app


__all__ = ["AgentHost", "HostConfig", "inside", "snapshot"]
