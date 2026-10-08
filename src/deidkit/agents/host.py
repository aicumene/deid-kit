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

A permission request that names a path outside the folder — among its locations, or in the command
it would run — is refused without asking.

**The model** is the agent's own choice list (:func:`model_choice`): the page shows it and switches
between tasks, through the agent (``session/set_config_option``). Every model the agent offers goes
through the same proxy.

**What a task cost** is taken as ACP carries it: the agent's ``usage_update`` (the tokens now in its
window, and the window's size) is shown as it comes, and the turn's tokens on the answer to
``session/prompt`` (``usage``, ACP's End-Turn Token Usage draft; ``_meta.usage`` too) are shown
when the task ends, with the seconds the host measured. Each task adds one line to the usage log
(``usage_log``): when, the scope, the model, how it ended, the seconds and the tokens — what a firm
can bill a matter against. The line names the scope, never the folder, whose name may be a
client's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from deidkit.agents.acp import AcpAgent, agent_env
from deidkit.agents.pricing import RequestUsage, task_cost
from deidkit.proxy.server import Proxy

log = logging.getLogger("deidkit.agents.host")

_SKIP_DIRS = {".git", ".claude", ".codex", "node_modules", "__pycache__", ".venv"}
_MAX_FILE = 2 * 1024 * 1024
#: What :meth:`AgentHost.warm_vault` reads: the folder's text documents, not its binaries.
_WARM_SUFFIXES = {".md", ".txt", ".csv", ".json", ".html", ".xml", ".eml"}
_WARM_MAX_BYTES = 512 * 1024
_WARM_MAX_FILES = 200
#: Set in a settings file's ``env``, these send the agent's requests somewhere other than the
#: proxy: another endpoint, another provider, or an HTTP proxy that sees them before they are
#: de-identified.
_REROUTE = re.compile(r"ANTHROPIC_\w*BASE_URL|CLAUDE_CODE_USE_\w+|HTTPS?_PROXY|ALL_PROXY", re.I)


@dataclass
class AgentChoice:
    """An agent the person can pick on the page (``--agent``): its name there and the command that
    starts it (ACP over stdio). ``upstream`` — a server of the organization's own that answers its
    model requests (the Messages API), instead of where the proxy sends them; no key goes there,
    since the organization's key is for Anthropic. ``unavailable`` — why it cannot start on this
    machine: the page lists it and does not offer it."""
    name: str
    command: list[str] = field(default_factory=list)
    upstream: str | None = None
    unavailable: str | None = None
    #: The session's `_meta` for this agent (``--agent-meta``): its own options.
    meta: dict | None = None


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
    #: One JSON line per task: what it cost (:func:`turn_usage`). None: kept nowhere.
    usage_log: Path | None = None
    #: The agents the page switches between, the first that can start working first. Empty: the
    #: one ``agent_command`` starts, and the page offers no choice.
    agents: list[AgentChoice] = field(default_factory=list)
    #: A stylesheet the page loads after its own (``--style``): the look of the program that opens
    #: the page, which can then match its own windows. Not a secret, so served without the token.
    style: Path | None = None
    #: The session's `_meta` when no agent of ``agents`` names its own (``--session-meta``).
    session_meta: dict | None = None


#: The token counts of ACP's draft ``PromptResponse.usage``; anything else in it is left out.
USAGE_FIELDS = ("inputTokens", "cachedReadTokens", "cachedWriteTokens", "outputTokens",
                "thoughtTokens", "totalTokens")


def turn_usage(result: dict) -> dict:
    """The token counts of a finished turn, as the agent gave them on its answer to
    ``session/prompt``: ``usage`` (ACP's End-Turn Token Usage draft), or ``_meta.usage`` where an
    agent puts it while the draft is one. Counts only; {} when there are none."""
    result = result if isinstance(result, dict) else {}
    meta = result.get("_meta") if isinstance(result.get("_meta"), dict) else {}
    given = result.get("usage") if isinstance(result.get("usage"), dict) else meta.get("usage")
    if not isinstance(given, dict):
        return {}
    return {k: given[k] for k in USAGE_FIELDS
            if isinstance(given.get(k), int) and not isinstance(given.get(k), bool) and given[k] >= 0}


def _append_line(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created owner-only from the first byte, not narrowed afterwards: no moment when it is not.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def model_choice(setup: dict) -> dict | None:
    """The models a session lets the person pick from, in either shape ACP has used: a ``select``
    config option of the model category (``configOptions``, current), or the older ``models`` block
    (``availableModels`` + ``currentModelId``, set with ``session/set_model``). ``None`` when the
    agent offers no choice.

    Returns ``{"via": "config" | "models", "id": <config option id>, "current": <value>,
    "options": [{"value", "name", "description"}, …]}``; grouped options are flattened."""
    for option in setup.get("configOptions") or []:
        if option.get("id") != "model" and option.get("category") != "model":
            continue
        if option.get("type", "select") != "select":
            continue
        options = []
        for entry in option.get("options") or []:
            for item in entry.get("options") if isinstance(entry.get("options"), list) else [entry]:
                if item.get("value"):
                    options.append({"value": item["value"], "name": item.get("name") or item["value"],
                                    "description": item.get("description") or ""})
        return {"via": "config", "id": option.get("id"), "current": option.get("currentValue"),
                "options": options}
    models = setup.get("models") or {}
    if models.get("availableModels"):
        return {"via": "models", "id": None, "current": models.get("currentModelId"),
                "options": [{"value": m["modelId"], "name": m.get("name") or m["modelId"],
                             "description": m.get("description") or ""}
                            for m in models["availableModels"] if m.get("modelId")]}
    return None


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


def settings_files(folder: Path) -> list[Path]:
    """The settings files Claude Code may read for an agent in ``folder``: managed, the person's
    own, and the project's (the folder's and its parents')."""
    out = [Path("/Library/Application Support/ClaudeCode/managed-settings.json"),
           Path("/etc/claude-code/managed-settings.json"), Path.home() / ".claude" / "settings.json"]
    for d in (folder, *folder.parents):
        out += [d / ".claude" / "settings.json", d / ".claude" / "settings.local.json"]
    return list(dict.fromkeys(out))


def rerouting_settings(folder: Path) -> list[str]:
    """``"<file>: <VARIABLE>"`` for each setting that would send the agent around the proxy."""
    found: list[str] = []
    for path in settings_files(folder):
        try:
            env = json.loads(path.read_text(encoding="utf-8")).get("env") or {}
            found += [f"{path}: {name}" for name in env if _REROUTE.fullmatch(name)]
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    return found


def inside(folder: Path, path: str) -> bool:
    try:
        target = Path(path)
        target = target if target.is_absolute() else folder / target
        return os.path.realpath(target).startswith(os.path.realpath(folder) + os.sep) or \
            os.path.realpath(target) == os.path.realpath(folder)
    except (OSError, ValueError):
        return False


#: What a command may name outside the matter folder: devices and the system's own programs.
_DEVICES = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/tty"}
_PROGRAM_DIRS = ("/usr/", "/bin/", "/sbin/", "/opt/homebrew/", "/System/", "/Library/Developer/")
_REDIRECT = re.compile(r"^\d*[<>]+&?")
_OPTION_VALUE = re.compile(r"^(?:--?[\w-]+|\w+)=")


def outside_paths(command: str, folder: Path) -> list[str]:
    """The paths a shell command names outside ``folder``.

    A permission request for a command carries no locations, so the check on them never saw one:
    MEASURED 29.09.2026, a matter agent asked to run ``grep -rni … ~/matters/`` — the
    other matters and their files of known names — and only the person's "No" stopped it. A word
    counts as a path when it starts with ``/``, ``~`` or ``$HOME``, or climbs with ``..``, and it
    (or the folder it would be in) exists on this machine, so a pattern such as ``"/api/v1"`` is
    not taken for one."""
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        words = command.split()
    home = os.path.expanduser("~")
    found: list[str] = []
    for word in words:
        for part in re.split(r"[;&|()]+", word):
            p = _OPTION_VALUE.sub("", _REDIRECT.sub("", part))
            p = p.replace("${HOME}", home).replace("$HOME", home)
            if not p or not (p.startswith(("/", "~")) or p == ".." or p.startswith("../")
                             or "/../" in p):
                continue
            target = Path(os.path.expanduser(p))
            target = target if target.is_absolute() else folder / target
            probe = target if target.exists() else target.parent
            if not probe.exists() or str(probe) == os.sep:
                continue
            real = os.path.realpath(target)
            if str(target) in _DEVICES or real in _DEVICES or real.startswith(_PROGRAM_DIRS):
                continue
            if not inside(folder, str(target)):
                found.append(p)
    return found


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
        self.models: dict | None = None
        self.key_refused_told = False
        self.context: dict | None = None             # the agent's last usage_update
        self.usage: dict = {"tasks": 0}              # the session's tokens, summed per field
        self.agents = cfg.agents or [AgentChoice(name="", command=cfg.agent_command)]
        self.current = next((a for a in self.agents if not a.unavailable), self.agents[0])
        self.upstream = proxy.cfg.upstream           # for an agent without a server of its own
        self.requests: list[RequestUsage] = []       # the model's answers in the task under way
        proxy.on_key_refused = self.key_refused
        proxy.on_usage = self.answer_usage

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
    async def warm_vault(self) -> int:
        """Pass the folder's text documents through the scope's vault before the first task.

        A document defines the short names it uses for the parties (``TOBIAS WREN … ("TW")``),
        and the vault learns one when it first sees the definition. The lawyer's own task may use
        the short name before the agent has read the document — MEASURED 30.09.2026: a task
        written with a party's short name crossed with it in clear in the task's first request.
        Nothing leaves the machine here; the documents are only tokenised. Returns how many were
        read."""
        texts: list[str] = []
        for root, dirs, files in os.walk(self.cfg.folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                p = Path(root) / name
                if p.suffix.lower() in _WARM_SUFFIXES and p.stat().st_size <= _WARM_MAX_BYTES:
                    texts.append(p.read_text("utf-8", errors="replace"))
                if len(texts) >= _WARM_MAX_FILES:
                    break
        if texts:
            await self.proxy.engine(self.cfg.scope).tokenize_all(texts)
        return len(texts)

    async def start_agent(self) -> None:
        try:
            log.info("vault warmed from %d document(s)", await self.warm_vault())
        except Exception:                            # noqa: BLE001 — the agent still starts
            log.exception("the vault could not be warmed from the folder's documents")
        if len(self.agents) > 1:
            self.emit({"kind": "agents", **self.agent_choice()})
        await self.launch(self.current)

    def agent_choice(self) -> dict | None:
        """The agents the page offers, ``None`` when there is one: ``{"current": <name>, "options":
        [{"value", "name", "unavailable"?}, …]}``."""
        if len(self.agents) < 2:
            return None
        return {"current": self.current.name,
                "options": [{"value": a.name, "name": a.name,
                             **({"unavailable": a.unavailable} if a.unavailable else {})}
                            for a in self.agents]}

    async def launch(self, choice: AgentChoice) -> None:
        """Start ``choice`` and open its session over the folder."""
        if choice.unavailable:
            self.status = "failed"
            self.emit({"kind": "error", "text": f"{choice.name} cannot start here: {choice.unavailable}"})
            return
        if not self.cfg.api_key:
            # The local sign-in does not work with a host-managed provider, so settings files keep
            # their say over where the agent connects: refuse to start if one would reroute it.
            rerouted = rerouting_settings(self.cfg.folder)
            if rerouted:
                self.status = "failed"
                self.emit({"kind": "error", "text": "The agent was not started: a settings file "
                                                    "would send its requests around the proxy ("
                                                    + "; ".join(rerouted) + "). Remove the "
                                                    "setting, or give the agent the organization's "
                                                    "key."})
                return
        # Where its model requests go, and whether the organization's key goes with them: not to
        # a server of the organization's own, which needs none — the key is for Anthropic.
        own_server = choice.upstream is not None
        self.proxy.cfg.upstream = (choice.upstream or self.upstream).rstrip("/")
        self.proxy.cfg.upstream_key = None if own_server else self.cfg.api_key
        lend = bool(self.cfg.api_key) and not own_server
        extra = {"CLAUDE_CODE_EXECUTABLE": self.cfg.claude_executable} if self.cfg.claude_executable else {}
        env = agent_env(base_url=f"http://127.0.0.1:{self.cfg.port}", scope=self.cfg.scope,
                        api_key=self.cfg.agent_key if lend else None, extra=extra)
        self.agent = AcpAgent(choice.command, env=env, cwd=str(self.cfg.folder),
                              on_update=self.on_update, on_permission=self.on_permission)
        try:
            info = await self.agent.start()
            self.session_id = await self.agent.new_session(str(self.cfg.folder), choice.meta or self.cfg.session_meta)
            self.models = model_choice(self.agent.session_setup)
            if self.models:
                self.emit({"kind": "models", **self.models})
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
        elif kind == "usage_update":
            used, size = u.get("used"), u.get("size")
            if isinstance(used, int) and isinstance(size, int):
                self.context = {"used": used, "size": size}
                cost = u.get("cost")
                if isinstance(cost, dict) and "amount" in cost and "currency" in cost:
                    self.context["cost"] = {"amount": cost["amount"], "currency": cost["currency"]}
                self.emit({"kind": "usage", **self.context})
        elif kind == "config_option_update":
            choice = model_choice({"configOptions": u.get("configOptions")})
            if choice and choice != self.models:
                self.models = choice
                self.emit({"kind": "models", **choice})

    async def on_permission(self, params: dict) -> dict:
        call = params.get("toolCall") or {}
        options = params.get("options") or []
        paths = [loc.get("path", "") for loc in call.get("locations") or [] if loc.get("path")]
        outside = [p for p in paths if not inside(self.cfg.folder, p)]
        raw = call.get("rawInput") if isinstance(call.get("rawInput"), dict) else {}
        command = raw.get("command") if isinstance(raw.get("command"), str) else None
        if command is None and call.get("kind") == "execute":
            command = call.get("title") or ""
        if command:
            outside += outside_paths(command, self.cfg.folder)
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

    def answer_usage(self, model: str, usage: dict) -> None:
        """An answer of the model crossed the proxy: count it into the task under way."""
        self.requests.append(RequestUsage.from_anthropic(usage, model))

    async def run_prompt(self, text: str) -> None:
        self.busy = True
        self.key_refused_told = False
        self.requests = []
        self.emit({"kind": "user", "text": text})
        started = time.monotonic()
        try:
            result = await self.agent.prompt(self.session_id, text)
            usage = turn_usage(result)
            seconds = round(time.monotonic() - started, 1)
            # What the task cost, from the answers the proxy saw: with the cache and without it.
            cost = task_cost(self.requests) if self.requests else None
            self.emit({"kind": "turn_end", "stopReason": result.get("stopReason"),
                       "usage": usage, "seconds": seconds, **({"cost": cost} if cost else {})})
            self.count(result.get("stopReason"), usage, seconds, cost)
        except Exception as exc:                     # noqa: BLE001
            self.emit({"kind": "error", "text": f"The turn failed: {exc}"})
        finally:
            self.busy = False
            self.emit({"kind": "files", **self.changes()})

    def count(self, stop_reason, usage: dict, seconds: float, cost: dict | None = None) -> None:
        """Add a finished task to the session's totals and to the usage log."""
        self.usage["tasks"] += 1
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + v
        if self.cfg.usage_log is None:
            return
        line = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scope": self.cfg.scope,
                **({"agent": self.current.name} if len(self.agents) > 1 else {}),
                "model": (self.models or {}).get("current"), "stopReason": stop_reason,
                "seconds": seconds, "usage": usage, **({"cost": cost} if cost else {})}
        try:
            _append_line(self.cfg.usage_log, line)
        except OSError:
            log.exception("the usage log could not be written")

    # ── HTTP ─────────────────────────────────────────────────────────────────
    def _authorised(self, request: web.Request) -> bool:
        given = request.headers.get("x-deid-agent-token", "")          # a header, never the address
        return secrets.compare_digest(given.encode(), self.cfg.token.encode())

    async def page(self, request: web.Request) -> web.Response:
        html = (Path(__file__).parent / "page.html").read_text(encoding="utf-8")
        if self.cfg.style:
            html = html.replace("</head>", '<link rel="stylesheet" href="/deid-agent/style.css">\n</head>', 1)
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def stylesheet(self, request: web.Request) -> web.Response:
        if not self.cfg.style:
            return web.json_response({"error": "not found"}, status=404)
        try:
            css = self.cfg.style.read_text(encoding="utf-8")
        except OSError:
            return web.json_response({"error": "not found"}, status=404)
        return web.Response(text=css, content_type="text/css", headers={"Cache-Control": "no-store"})

    async def api(self, request: web.Request) -> web.StreamResponse:
        if not self._authorised(request):
            return web.json_response({"error": "unauthorised"}, status=401)
        name = request.match_info["name"]
        if name == "state" and request.method == "GET":
            auth = "organization key" if self.cfg.api_key else "personal sign-in (development)"
            return web.json_response({"title": self.cfg.title, "folder": str(self.cfg.folder),
                                      "scope": self.cfg.scope, "status": self.status,
                                      "busy": self.busy, "auth": auth, "models": self.models,
                                      "agents": self.agent_choice(),
                                      "usage": self.usage, "context": self.context,
                                      **self.changes()})
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
        if name == "model" and request.method == "POST":
            return await self._set_model(((await request.json()).get("value") or "").strip())
        if name == "agent" and request.method == "POST":
            return await self._set_agent(((await request.json()).get("value") or "").strip())
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

    async def _set_model(self, value: str) -> web.Response:
        """Switch the agent's model between tasks. The agent's own answer is what the page shows:
        the choice it reports back, not the one asked for."""
        if not (self.models and self.agent and self.session_id):
            return web.json_response({"error": "the agent offers no choice of model"}, status=409)
        if self.busy:
            return web.json_response({"error": "the agent is working: change the model between "
                                               "tasks"}, status=409)
        if value not in {o["value"] for o in self.models["options"]} | {self.models.get("current")}:
            return web.json_response({"error": "the agent offers no such model"}, status=400)
        try:
            if self.models["via"] == "config":
                options = await self.agent.set_config_option(self.session_id, self.models["id"], value)
                choice = model_choice({"configOptions": options}) or {**self.models, "current": value}
            else:
                await self.agent.set_model(self.session_id, value)
                choice = {**self.models, "current": value}
        except Exception as exc:                     # noqa: BLE001 — the agent said no
            return web.json_response({"error": f"the model was not changed: {exc}"}, status=502)
        self.models = choice
        self.emit({"kind": "models", **choice})
        return web.json_response({"ok": True, "models": choice})

    async def _set_agent(self, value: str) -> web.Response:
        """Switch to another agent between tasks: the one working stops, the other starts over the
        same folder in a new session — it does not see the earlier tasks; the page keeps them."""
        choice = next((a for a in self.agents if a.name == value), None)
        if len(self.agents) < 2 or choice is None:
            return web.json_response({"error": "there is no such agent"}, status=400)
        if choice.unavailable:
            return web.json_response({"error": f"{choice.name} cannot start here: {choice.unavailable}"},
                                     status=409)
        if self.busy:
            return web.json_response({"error": "the agent is working: switch between tasks"}, status=409)
        if self.status == "starting":
            return web.json_response({"error": "the agent is starting"}, status=409)
        if choice is self.current and self.status == "ready":
            return web.json_response({"ok": True, "agents": self.agent_choice()})
        old, self.agent, self.session_id = self.agent, None, None
        self.models, self.context = None, None
        self.current, self.status = choice, "starting"
        self.emit({"kind": "agents", **self.agent_choice()})
        self.emit({"kind": "models", "options": []})          # the stopped agent's models go
        self.emit({"kind": "status", "status": "starting"})
        if old:
            await old.stop()
        asyncio.get_running_loop().create_task(self.launch(choice))
        return web.json_response({"ok": True, "agents": self.agent_choice()}, status=202)

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
        app.router.add_get("/deid-agent/style.css", self.stylesheet)
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


__all__ = ["AgentChoice", "AgentHost", "HostConfig", "inside", "model_choice", "snapshot", "turn_usage"]
