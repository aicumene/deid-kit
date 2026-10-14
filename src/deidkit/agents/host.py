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

**Agents switch without losing the work** (``--agent``, several). Each agent the person has worked with
keeps its own process and session (:class:`Seat`) while another works: picked again, it goes on in the
same session, its context as it was — and its model's prompt cache with it, since the context is only
ever added to. What it has not seen, done meanwhile by the others, it is told in a brief at the head of
its next prompt (:func:`brief`): only the part since it last took part, and the same words for the same
work, so that a brief does not unsettle what the cache holds. A task under way can be handed over too:
the step under way stops (``session/cancel``, the asks it waits on answered as cancelled), and the one
picked carries the task on in its own session, told what was done.

**Contexts stay apart.** No agent's context is copied into another's: a brief is made by the host from
the record the person sees — the tasks, what the agents said, the steps they took and how each ended,
the person's answers to their asks, the documents changed — and not from what an agent read or how its
model reasoned; the one told reads again what it needs, under its own permissions. It goes in that
agent's prompt, and so through the proxy like everything else the agent sends: the model sees tokens
in it, never the names. The page shows each brief as it was given.

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
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web

from deidkit.agents import review
from deidkit.agents.acp import AcpAgent, agent_env, codex_env
from deidkit.agents.confine import ConfinementUnavailable, codex_with, confined_codex, matter_codex_home
from deidkit.agents.pricing import RequestUsage, request_usage, task_cost
from deidkit.proxy.anthropic import SUBAGENT_KINDS
from deidkit.proxy.server import Proxy

log = logging.getLogger("deidkit.agents.host")

# `.legix` is the matter's notes folder: the app writes the texts of documents and the review's notes there,
# and neither is the agent's work to show as changed.
_SKIP_DIRS = {".git", ".claude", ".codex", ".legix", "node_modules", "__pycache__", ".venv"}
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

    @property
    def codex(self) -> bool:
        """Codex through its ACP adapter (``codex-acp``): it finds the proxy through its own
        settings (:func:`~deidkit.agents.acp.codex_env`), not Claude's."""
        return any("codex-acp" in part for part in self.command)

    @property
    def claude(self) -> bool:
        """Claude Code through its ACP adapter (``claude-agent-acp``, or the desktop's ``claude-acp``)."""
        return any("claude-agent-acp" in part or Path(part).name == "claude-acp" for part in self.command)

    @property
    def label(self) -> str:
        """What the person calls it: its name, or, for the one agent of ``--agent-command``, what it is."""
        if self.name:
            return self.name
        return "Claude Code" if self.claude else "Codex" if self.codex else "MatterAgent" if self.matteragent else "Agent"

    @property
    def matteragent(self) -> bool:
        """matteragent: it hands jobs to the sub-agents itself, with its Subagent tool."""
        return any(Path(part).name == "matteragent" or part == "matteragent" for part in self.command) or any(
            a == "-m" and b == "matteragent" for a, b in zip(self.command, self.command[1:]))

    @property
    def responses(self) -> bool:
        """It asks its model in OpenAI's shape (the Responses API): Codex, or an agent started with
        ``--wire responses`` (matteragent). Its own ``upstream`` then answers ``/v1/responses``."""
        wire = any(a == "--wire=responses" for a in self.command) or any(
            a == "--wire" and b == "responses" for a, b in zip(self.command, self.command[1:]))
        return self.codex or wire


@dataclass
class HostConfig:
    folder: Path
    scope: str
    title: str
    agent_command: list[str]
    api_key: str | None = None
    dev_login: bool = False
    claude_executable: str | None = None
    #: The person's own Codex, for an agent that is Codex (``codex-acp``); none — the adapter's.
    codex_executable: str | None = None
    #: Codex confined by the kernel to the matter folder (:mod:`deidkit.agents.confine`). Where that
    #: cannot be done, Codex does not start unless this is False (``--codex-unconfined``).
    codex_confine: bool = True
    #: Codex's home for this matter (``--codex-home``): its own sessions and settings, the person's
    #: sign-in linked in. None: ``~/.deid/agent/codex/<scope>``.
    codex_home: Path | None = None
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
    #: Sub-agents on the organization's own model servers (``--subagents``): the hosts file of
    #: matteragent's sub-agents. Claude Code and Codex get them as the MCP server "matter"
    #: (``subagents_command``), matteragent as its Subagent tool; the cloud agents' own sub-agents
    #: are then off. None: no sub-agents.
    subagents: Path | None = None
    #: What starts the sub-agents' MCP server (``python -m matteragent.mcp`` or the bundle's
    #: ``matteragent-mcp``); deid-agent adds the folder, the hosts and the main agent's family.
    subagents_command: list[str] = field(default_factory=list)
    #: Seconds a client waits for one sub-agent's report.
    subagents_timeout: float = 900.0
    #: Claude Code's shell commands in its sandbox (:func:`claude_settings`). Where the sandbox cannot
    #: start, Claude Code does not start, unless this is False (``--claude-unconfined``).
    claude_confine: bool = True
    #: agent-guard's command (``--guard``): it decides each of Claude Code's tool calls before it runs
    #: and journals it after. Empty: no guard.
    guard: list[str] = field(default_factory=list)
    #: Where agent-guard journals this matter (``<the audit's folder>/guard/<scope>.jsonl``).
    guard_log: Path | None = None
    #: Who the person at this Mac is, in the notes and decisions they make on documents (``--person``).
    person: str = field(default_factory=lambda: os.environ.get("USER", "") or "person")


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


#: The matter server's tools as Claude Code names them.
SUBAGENT_MCP_TOOLS = tuple(f"mcp__matter__{k}" for k in SUBAGENT_KINDS)
#: What Codex is told about the sub-agents. Codex keeps an MCP server's tools out of its model's tool
#: list — deferred, on the global ``tools`` object of its code mode — and a model that is not told of
#: them reads every document itself (measured 09.10.2026: told, Codex handed the reading to them).
CODEX_SUBAGENTS_NOTE = (
    "This matter has sub-agents on the organization's own model servers, next to its folder. They are "
    "deferred tools on the global `tools` object of exec, named `mcp__matter__<kind>` — read, find, "
    "extract, calc, check, and consult when a head model is up; `ALL_TOOLS` lists those available now. "
    "Each takes a question and, optionally, the paths to read. Hand them the reading of the matter's "
    "documents: a sub-agent reads the files where they are and reports back with exact quotes, each "
    "checked against its file. Prefer them to reading long documents yourself.")


def with_subagents(meta: dict | None) -> dict:
    """The session's `_meta` for Claude Code with sub-agents of the organization's own: its own (the
    Task tool, in Anthropic's cloud) off, so the work beside the main agent stays here; the matter
    server's tools allowed without asking — they only read, inside the folder Claude Code reads anyway."""
    meta = json.loads(json.dumps(meta or {}))
    options = meta.setdefault("claudeCode", {}).setdefault("options", {})
    disallowed = list(options.get("disallowedTools") or [])
    options["disallowedTools"] = disallowed + [t for t in ("Task",) if t not in disallowed]
    allowed = list(options.get("allowedTools") or [])
    options["allowedTools"] = allowed + [t for t in SUBAGENT_MCP_TOOLS if t not in allowed]
    return meta


#: Where people's files live: closed to Claude Code's shell commands (``sandbox.filesystem.denyRead``),
#: but for the matter folder, which ``allowRead`` opens again inside them.
PEOPLES_FILES = ("/Users", "/Volumes", "/home", "/root", "/mnt", "/media")
#: The events agent-guard hears: the request (journalled as its fingerprint, not its text), each tool
#: call before it runs (allow, ask, deny), and how it ended — a call that fails has no PostToolUse.
GUARD_EVENTS = ("UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure")


def claude_settings(folder: Path, *, confine: bool = True, guard: list[str] | None = None,
                    guard_log: Path | None = None) -> dict:
    """Claude Code's settings for a matter. The session gives them as ``claudeCode.options.settings``,
    which the adapter passes on as ``--settings``: the flag tier, which the person's or a folder's own
    settings files cannot loosen.

    ``confine``: its shell commands run in its sandbox (Seatbelt on macOS, a namespace on Linux), so
    the kernel holds them where the host's reading of a command line (:func:`outside_paths`) holds
    only what the command spells out. No network at all; people's files closed, the matter folder
    open; writes to the folder and a temp folder only. No command leaves the sandbox, and Claude
    Code does not start without it. The person is still asked before each command, since one that
    writes changes the matter's files.

    ``guard``: agent-guard's command. Each tool call is decided by it before it runs (allow, ask —
    the person's card — or deny) and journalled after, in ``guard_log``: a file under the person's
    home, which the agent's commands cannot read."""
    settings: dict = {}
    if confine:
        settings["sandbox"] = {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": False,
            "filesystem": {"denyRead": list(PEOPLES_FILES), "allowRead": [str(folder)]},
            "network": {"allowedDomains": [], "strictAllowlist": True},
        }
    if guard:
        command = " ".join(shlex.quote(part) for part in [*guard, "hook"])
        if guard_log is not None:
            command = f"AGENT_GUARD_LOG={shlex.quote(str(guard_log))} {command}"
        settings["hooks"] = {event: [{"matcher": "*", "hooks": [{"type": "command", "command": command,
                                                                 "timeout": 10}]}]
                             for event in GUARD_EVENTS}
    return settings


def with_claude_settings(meta: dict | None, settings: dict, folder: Path) -> dict:
    """The session's `_meta` with ``settings`` (:func:`claude_settings`) added to the settings it
    already gives: the matter's sandbox replaces any sandbox given, its hooks run after any given."""
    meta = json.loads(json.dumps(meta or {}))
    if not settings:
        return meta
    options = meta.setdefault("claudeCode", {}).setdefault("options", {})
    given = options.get("settings") or {}
    if isinstance(given, str):                     # a file, read as the adapter would: from the folder
        given = json.loads(Path(folder, given).read_text(encoding="utf-8"))
    if not isinstance(given, dict):
        raise ValueError("claudeCode.options.settings is neither an object nor a file of one")
    merged = dict(given)
    for key, value in settings.items():
        if key == "hooks":
            hooks = dict(merged.get("hooks") or {})
            for event, entries in value.items():
                hooks[event] = list(hooks.get(event) or []) + entries
            merged["hooks"] = hooks
        else:
            merged[key] = value
    options["settings"] = merged
    return meta


#: Where a Python and the sub-agents' package are installed: the environment, the interpreter's own
#: installation, and the folder the package is imported from — an editable install keeps it in a
#: source tree elsewhere, and the import lists that folder, not only the package in it.
_WHERE_INSTALLED = ("import os, sys\nprint(sys.prefix)\nprint(sys.base_prefix)\ntry:\n    import matteragent\n"
                    "    print(os.path.dirname(matteragent.__path__[0]))\nexcept Exception:\n    pass\n")


def _shebang_python(program: str) -> str | None:
    """The interpreter a script's first line names, when it is a Python (a console script)."""
    try:
        with open(program, "rb") as fh:
            first = fh.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if first.startswith("#!"):
        parts = first[2:].split()
        if parts and Path(parts[0]).name.startswith("python"):
            return parts[0]
        if len(parts) > 1 and Path(parts[0]).name == "env" and parts[1].startswith("python"):
            return shutil.which(parts[1])
    return None


#: How much of the work a brief carries: of an agent's words in one run of a task, the end (where the
#: answer is); of a task's text, the start; of a run's steps, the latest; and the whole brief, the older
#: tasks shortened first.
BRIEF_SAID = 1500
BRIEF_TASK = 3000
BRIEF_STEPS = 40
BRIEF_CHARS = 24000

_ENDED = {"end_turn": "finished", "cancelled": "stopped by the person", "refusal": "refused by the agent",
          "max_tokens": "stopped: its reply ran out of room", "max_turn_requests": "stopped: it ran out of steps"}
_STEP = {"completed": "done", "failed": "failed"}
_ANSWER = {"allowed": "allowed it", "refused": "refused it", "cancelled": "did not answer before it stopped"}

#: What an agent is told when the person hands it a task under way, after the brief.
CARRY_ON = ("The person has handed you the last task above, which {previous} was doing and did not finish. "
            "Carry it on from where it stopped: the documents are as {previous} left them. Read again "
            "whatever you need, and finish the task as the person asked it.")


def _clip_end(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else "…" + text[-n:].lstrip()


def _clip_start(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def brief(events: list[dict], me: str, *, carrying_on: bool = False) -> str | None:
    """What an agent is told of work on the matter that is not in its own context: the tasks and their runs
    in ``events`` — the page's record since it last took part — as the person saw them. What was asked,
    what each agent said, the steps it took and how each ended, the person's answers to its asks, how each
    run ended, and the documents changed. Not the agents' reasoning and not what they read: one agent's
    context is not copied into another's. The same events give the same text, with nothing of time or cost
    in it, so a brief does not unsettle a prompt cache. ``carrying_on``: the agent takes the last task over
    (:data:`CARRY_ON` follows); otherwise the person's next task follows. None when there is no work."""
    tasks: list[dict] = []
    task = run = None
    perms: dict[str, str] = {}
    files = None

    def new_run(agent: str, **extra) -> dict:
        r = {"agent": agent, "items": [], "steps": {}, "plan": None, "end": None, **extra}
        task["runs"].append(r)
        return r

    for e in events:
        kind = e.get("kind")
        if kind == "files":
            files = e
            continue
        if kind == "user":
            task = {"text": e.get("text", ""), "runs": []}
            tasks.append(task)
            run = new_run(e.get("agent") or "the agent")
            continue
        if kind == "handover":
            if task is None:                        # the task under way when this agent left it
                task = {"text": None, "runs": []}
                tasks.append(task)
            run = new_run(e.get("to") or "another agent", taken_from=e.get("from"))
            continue
        if run is None:
            continue
        items = run["items"]
        if kind == "message":
            if items and items[-1][0] == "said":
                items[-1] = ("said", items[-1][1] + e.get("text", ""))
            else:
                items.append(("said", e.get("text", "")))
        elif kind == "tool":
            sid = e.get("id") or f"#{len(run['steps'])}"
            run["steps"][sid] = {"title": e.get("title") or e.get("tool") or "a step",
                                 "status": e.get("status") or "pending"}
            items.append(("step", sid))
        elif kind == "tool_update":
            step = run["steps"].get(e.get("id"))
            if step is not None:
                step["status"] = e.get("status") or step["status"]
                step["title"] = e.get("title") or step["title"]
        elif kind == "plan":
            run["plan"] = e.get("entries") or []
        elif kind == "permission":
            perms[e.get("id", "")] = e.get("title") or "an action"
        elif kind == "permission_done" and e.get("id") in perms and e.get("outcome") in _ANSWER:
            items.append(("line", f"It asked to: {perms[e['id']]} — the person {_ANSWER[e['outcome']]}."))
        elif kind == "notice" and not e.get("handing"):          # the handover is told as how the run ended
            items.append(("line", e.get("text", "")))
        elif kind == "error":
            items.append(("line", "Error: " + e.get("text", "")))
        elif kind == "turn_end":
            run["end"] = e.get("stopReason")

    def render(task: dict, number: int, whole: bool) -> list[str]:
        if task["text"] is None:
            out = ["The task you were doing when it was handed over:"]
        else:
            first = task["runs"][0]["agent"] if task["runs"] else "the agent"
            out = [f"Task {number} — the person asked {'you' if first == me else first}: "
                   f"{_clip_start(task['text'], BRIEF_TASK)}"]
        runs = task["runs"]
        for i, r in enumerate(runs):
            who = "you" if r["agent"] == me else r["agent"]
            if r.get("taken_from") is not None:
                gave = "you" if r["taken_from"] == me else r["taken_from"]
                out.append(f"  Then the person handed the task to {who}." if who == "you" and not r["items"]
                           else f"  {who[:1].upper() + who[1:]} took the task over from {gave}.")
            if whole:
                steps = [x for x in r["items"] if x[0] == "step"]
                skip = max(0, len(steps) - BRIEF_STEPS)
                if skip:
                    out.append(f"  ({skip} earlier steps not listed)")
                for kind, value in r["items"]:
                    if kind == "said":
                        if value.strip():
                            out.append(f"  {who[:1].upper() + who[1:]} said: {_clip_end(value, BRIEF_SAID)}")
                    elif kind == "step":
                        if skip:
                            skip -= 1
                            continue
                        st = r["steps"][value]
                        out.append(f"  · {st['title']} — {_STEP.get(st['status'], 'not finished')}")
                    else:
                        out.append(f"  {value}")
                if r["plan"]:
                    out.append("  Plan: " + "; ".join(("✓ " if p.get("status") == "completed" else "• ")
                                                      + str(p.get("content", "")) for p in r["plan"]))
            if r["end"] is not None:
                handed = i + 1 < len(runs) and runs[i + 1].get("taken_from") is not None
                ended = (f"stopped so that the person could hand the task over"
                         if handed and r["end"] == "cancelled" else _ENDED.get(r["end"], str(r["end"])))
                out.append(f"  Ended: {ended}.")
        return out

    if not any(r["items"] or r["end"] is not None for t in tasks for r in t["runs"]):
        return None
    head = ("The work on this matter that you have not seen — done while another agent worked on it — as the "
            "person saw it: what was asked, what each agent said and did, and how it ended. Their reasoning "
            "and what they read are not part of it: read again whatever you need.")
    tail = [("The person's new task follows; the work above is what came before it." if not carrying_on else "")]
    if files:
        parts = [f"{p} (changed)" for p in files.get("changed") or []] +                 [f"{p} (new)" for p in files.get("added") or []] + [f"{p} (removed)" for p in files.get("removed") or []]
        if parts:
            tail.insert(0, "Documents changed since the session began: " + ", ".join(parts) + ".")
    numbered = [(t, sum(1 for x in tasks[:i + 1] if x["text"] is not None)) for i, t in enumerate(tasks)]
    # whole, then the older tasks shortened to their ask and how they ended until it fits
    for shortened in range(len(tasks) + 1):
        lines = [head, ""]
        for i, (t, n) in enumerate(numbered):
            lines += render(t, n, whole=i >= shortened) + [""]
        text = "\n".join(lines + [x for x in tail if x]).strip()
        if len(text) <= BRIEF_CHARS:
            return text
    return "…" + text[-BRIEF_CHARS:]


@dataclass
class Seat:
    """An agent of the page with its own process and session, kept while the person works with another:
    picked again, it goes on in the same session — its context as it was, and its model's prompt cache
    with it — told only what was done since it last took part."""
    choice: AgentChoice
    agent: AcpAgent | None = None
    session_id: str | None = None
    models: dict | None = None
    context: dict | None = None                  # its last usage_update
    #: The page's events before this index are in its context: its own work, or told in a brief.
    seen: int = 0
    #: Its confined Codex's profile and wrapper, while it runs.
    codex_dir: str | None = None

    @property
    def alive(self) -> bool:
        return bool(self.agent and self.session_id and self.agent.proc and self.agent.proc.returncode is None)


class AgentHost:
    def __init__(self, cfg: HostConfig, proxy: Proxy) -> None:
        self.cfg = cfg
        self.proxy = proxy
        self.events: list[dict] = []
        self.listeners: set[asyncio.Queue] = set()
        self.permissions: dict[str, asyncio.Future] = {}
        self.busy = False
        self.baseline = snapshot(cfg.folder)
        self.status = "starting"
        self.key_refused_told = False
        self.usage: dict = {"tasks": 0}              # the session's tokens, summed per field
        self.agents = cfg.agents or [AgentChoice(name="", command=cfg.agent_command)]
        self.current = next((a for a in self.agents if not a.unavailable), self.agents[0])
        #: Each agent the person has worked with, by name: its process and session, kept.
        self.seats: dict[str, Seat] = {}
        #: The agent the task under way goes to once the one working has stopped (the person handed it over).
        self.handover: AgentChoice | None = None
        self.upstream = proxy.cfg.upstream           # for an agent without a server of its own
        self.upstream_openai = proxy.cfg.upstream_openai
        self.requests: list[RequestUsage] = []       # the model's answers in the task under way
        proxy.on_key_refused = self.key_refused
        proxy.on_usage = self.answer_usage

    @property
    def seat(self) -> Seat:
        """The seat of the agent at work."""
        return self.seats.setdefault(self.current.name, Seat(self.current))

    # The agent at work's process, session, models and window, as the rest of the host reads them.
    agent = property(lambda self: self.seat.agent, lambda self, v: setattr(self.seat, "agent", v))
    session_id = property(lambda self: self.seat.session_id, lambda self, v: setattr(self.seat, "session_id", v))
    models = property(lambda self: self.seat.models, lambda self, v: setattr(self.seat, "models", v))
    context = property(lambda self: self.seat.context, lambda self, v: setattr(self.seat, "context", v))

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

    def route(self, choice: AgentChoice) -> bool:
        """Point the proxy at ``choice``'s model: where its requests go, whether the organization's key goes
        with them, which tool namespaces leave its requests. Routed again at every switch — the agents kept
        while another works send no requests. Returns whether the agent is lent the key (its stand-in)."""
        # Where its model requests go, and whether the organization's key goes with them: not to
        # a server of the organization's own, which needs none — the key is for Anthropic.
        own_server = choice.upstream is not None
        self.proxy.cfg.upstream = (choice.upstream or self.upstream).rstrip("/")
        # An agent in OpenAI's shape with a server of its own: its /v1/responses go there.
        self.proxy.cfg.upstream_openai = choice.upstream.rstrip("/") + "/v1" if own_server and choice.responses \
            else self.upstream_openai
        # Holding the key, the proxy lets only the stand-in through, toward Anthropic only, and strips
        # other sign-ins: an agent in OpenAI's shape (Codex on the person's own sign-in, matteragent
        # --wire responses) would be refused. The key is Anthropic's; they go without it.
        self.proxy.cfg.upstream_key = None if own_server or choice.responses else self.cfg.api_key
        # Codex's own sub-agents (the collaboration tools, in OpenAI's cloud) are not offered to its
        # model when the session has sub-agents of the organization's own: the work beside it stays
        # here. Codex has no setting that takes them away (measured 09.10.2026 with 0.161: with
        # --disable multi_agent they stayed), so the proxy takes them out of its requests.
        self.proxy.cfg.drop_tool_namespaces = frozenset({"collaboration"}) \
            if self.subagent_server(choice) and choice.codex else frozenset()
        # The organization's key is Anthropic's: an agent in OpenAI's shape is not lent it.
        return bool(self.cfg.api_key) and not own_server and not choice.responses

    def take_seat(self, choice: AgentChoice) -> bool:
        """``choice`` becomes the agent at work. The one that worked keeps its seat, with all so far in its
        context. True when ``choice``'s seat was alive and goes on (routed again, ready at once); False
        when it has to start (:meth:`launch`)."""
        if choice.name != self.current.name:
            self.seat.seen = len(self.events)
            self.current = choice
        if len(self.agents) > 1:
            self.emit({"kind": "agents", **self.agent_choice()})
        seat = self.seat
        if seat.alive:
            self.route(choice)
            self.status = "ready"
            self.emit({"kind": "models", **seat.models} if seat.models else {"kind": "models", "options": []})
            self.emit({"kind": "status", "status": "ready",
                       "agent": (seat.agent.agent_info.get("agentInfo") or {}).get("name", "agent")})
            return True
        self.status = "starting"
        self.emit({"kind": "models", "options": []})          # a stopped agent's models go
        self.emit({"kind": "status", "status": "starting"})
        return False

    async def launch(self, choice: AgentChoice) -> None:
        """Start ``choice`` and open its session over the folder: its seat. A new session has none of the
        work in it: what was done before, it is told with its first prompt."""
        seat = self.seats.setdefault(choice.name, Seat(choice))
        seat.seen = 0
        if choice.unavailable:
            self.status = "failed"
            self.emit({"kind": "error", "text": f"{choice.name} cannot start here: {choice.unavailable}"})
            return
        if not self.cfg.api_key and not choice.codex:
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
        lend = self.route(choice)
        server = self.subagent_server(choice)
        command = list(choice.command)
        if self.cfg.subagents and choice.matteragent and "--subagents" not in command:
            command += ["--subagents", str(self.cfg.subagents)]
        if choice.codex:
            # Codex's home for this matter: the person's ~/.codex holds the sessions of all their work.
            home = self.cfg.codex_home or \
                Path("~/.deid/agent/codex").expanduser() / re.sub(r"[^A-Za-z0-9._-]", "_", self.cfg.scope)
            auth = matter_codex_home(home)
            extra = codex_env(base_url=f"http://127.0.0.1:{self.cfg.port}", scope=self.cfg.scope,
                              codex_path=self.cfg.codex_executable, codex_home=str(home))
            codex_args = self.codex_server_args(server) if server else []
            if self.cfg.codex_confine or codex_args:
                self._drop_codex_dir(seat)
                seat.codex_dir = tempfile.mkdtemp(prefix="deid-codex-")
            if not self.cfg.codex_confine and codex_args:
                extra["CODEX_PATH"] = str(codex_with(self.cfg.codex_executable, seat.codex_dir, codex_args))
            if self.cfg.codex_confine:
                try:
                    extra["CODEX_PATH"] = str(confined_codex(
                        self.cfg.folder, self.cfg.codex_executable, seat.codex_dir, codex_home=home,
                        extra_read=self.subagent_paths(server) if server else (), args=codex_args,
                        auth_file=auth))
                except ConfinementUnavailable as exc:
                    self._drop_codex_dir(seat)
                    self.status = "failed"
                    self.emit({"kind": "error", "text": f"Codex was not started: {exc}."})
                    return
        else:
            extra = {"CLAUDE_CODE_EXECUTABLE": self.cfg.claude_executable} if self.cfg.claude_executable else {}
            if server and choice.claude:
                # Claude Code waits this long for an MCP tool; a sub-agent reading a long contract on
                # a slow host takes minutes.
                extra["MCP_TOOL_TIMEOUT"] = str(int(self.cfg.subagents_timeout * 1000))
        env = agent_env(base_url=f"http://127.0.0.1:{self.cfg.port}", scope=self.cfg.scope,
                        api_key=self.cfg.agent_key if lend else None, extra=extra)
        # what it sends is the page's only while it is the agent at work
        seat.agent = AcpAgent(command, env=env, cwd=str(self.cfg.folder),
                              on_update=lambda params, c=choice: self.on_update(params, c),
                              on_permission=lambda params, c=choice: self.on_permission(params, c))
        meta = choice.meta or self.cfg.session_meta
        if server and choice.claude:
            meta = with_subagents(meta)
        if choice.claude:
            try:
                meta = with_claude_settings(meta, claude_settings(
                    self.cfg.folder, confine=self.cfg.claude_confine, guard=self.cfg.guard,
                    guard_log=self.cfg.guard_log), self.cfg.folder)
            except (OSError, ValueError) as exc:
                self.status = "failed"
                self.emit({"kind": "error", "text": f"Claude Code was not started: its settings: {exc}"})
                return
        if server and choice.codex:
            # The adapter starts the first task only once the server is up: a task started before
            # it has no tools of it for the whole task (measured 09.10.2026).
            meta = {**(meta or {}), "mcpStartupAwaitTimeoutMs": 30000}
        try:
            info = await seat.agent.start()
            seat.session_id = await seat.agent.new_session(
                str(self.cfg.folder), meta,
                mcp_servers=[{"name": "matter", "command": server["command"], "args": server["args"], "env": []}]
                if server else None)
            seat.models = model_choice(seat.agent.session_setup)
            if seat.models:
                self.emit({"kind": "models", **seat.models})
            self.status = "ready"
            self.emit({"kind": "status", "status": "ready",
                       "agent": (info.get("agentInfo") or {}).get("name", "agent")})
        except Exception as exc:                     # noqa: BLE001
            self.status = "failed"
            log.exception("the agent did not start")
            self.emit({"kind": "error", "text": f"The agent did not start: {exc}"})

    def subagent_server(self, choice: AgentChoice) -> dict | None:
        """The sub-agents' MCP server for ``choice``, as ``{command, args}`` — none for an agent
        that has its own way to them (matteragent), or when there are no sub-agents."""
        if not self.cfg.subagents or not self.cfg.subagents_command or choice.matteragent:
            return None
        family = "gpt" if choice.codex else "claude" if choice.claude else ""
        command = [*self.cfg.subagents_command, "--folder", str(self.cfg.folder),
                   "--hosts", str(self.cfg.subagents)] + (["--author-family", family] if family else [])
        return {"command": command[0], "args": command[1:]}

    def codex_server_args(self, server: dict) -> list[str]:
        """The matter server for Codex, defined whole where Codex starts (``-c``): the adapter's own
        entry for it has no timeouts, and Codex waits 60 seconds for a tool by default — a sub-agent
        reading a long contract on a slow host takes minutes. The session still names the server,
        so the adapter waits for it to start; seeing it defined, the adapter adds no copy."""
        def toml(value) -> str:
            return json.dumps(value, ensure_ascii=False)          # a JSON string or list is TOML too
        return ["-c", f"mcp_servers.matter.command={toml(server['command'])}",
                "-c", f"mcp_servers.matter.args={toml(server['args'])}",
                "-c", f"mcp_servers.matter.tool_timeout_sec={int(self.cfg.subagents_timeout)}",
                "-c", "mcp_servers.matter.startup_timeout_sec=30",
                # the sub-agents only read, inside the folder Codex may read anyway: not asked each time
                "-c", 'mcp_servers.matter.default_tools_approval_mode="approve"',
                "-c", f"developer_instructions={toml(CODEX_SUBAGENTS_NOTE)}"]

    def subagent_paths(self, server: dict) -> list[str]:
        """What the sub-agents' MCP server reads besides the folder, for Codex's confinement: its
        hosts file, its program, and — for a Python program — the interpreter's installation and the
        package, wherever they are installed (asked of the interpreter itself)."""
        paths = [str(self.cfg.subagents)]
        program = shutil.which(server["command"]) or server["command"]
        paths.append(os.path.realpath(program))
        # a wrapper in an installation's bin/ (the desktop's agent runtime: bin/matteragent-mcp beside
        # python/): the installation with it — the profile skips what is too wide to open
        paths.append(os.path.dirname(os.path.dirname(os.path.realpath(program))))
        if Path(program).name.startswith("python") or _shebang_python(program):
            python = _shebang_python(program) or program
            try:
                done = subprocess.run([python, "-c", _WHERE_INSTALLED], capture_output=True, text=True, timeout=20)
                paths += [line for line in done.stdout.splitlines() if line.strip()]
            except (OSError, subprocess.TimeoutExpired):
                log.warning("could not ask %s where it is installed", python)
            paths.append(os.path.dirname(os.path.dirname(os.path.realpath(python))))
        return list(dict.fromkeys(paths))

    async def on_update(self, params: dict, choice: AgentChoice | None = None) -> None:
        if choice is not None and choice.name != self.current.name:
            log.debug("an update from %s, which is not at work, left out", choice.name)
            return
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

    async def on_permission(self, params: dict, choice: AgentChoice | None = None) -> dict:
        # Only the agent at work asks the person; nor does one whose task is being handed over.
        if (choice is not None and choice.name != self.current.name) or self.handover is not None:
            return {"outcome": "cancelled"}
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
            # how it was answered — allowed, refused, or not at all (the turn stopped) — for the brief
            answer = fut.result() if fut.done() and not fut.cancelled() else {}
            option = next((o for o in options if answer.get("outcome") == "selected"
                           and o.get("optionId") == answer.get("optionId")), None)
            outcome = "cancelled" if option is None else \
                "allowed" if str(option.get("kind", "")).startswith("allow") else "refused"
            self.emit({"kind": "permission_done", "id": rid, "outcome": outcome})

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
        self.requests.append(request_usage(usage, model))

    async def run_prompt(self, text: str, *, carrying_from: str | None = None, shown: str | None = None) -> None:
        """A task for the agent at work — or, ``carrying_from`` another agent, the task under way, handed
        over. What was done that it has not seen goes first in the same prompt, as a brief (:func:`brief`).
        ``shown`` is what the page and the other agents' briefs say of the task, if not its own words."""
        seat = self.seat
        self.busy = True
        self.key_refused_told = False
        self.requests = []
        told = brief(self.events[seat.seen:], self.current.name, carrying_on=carrying_from is not None)
        if carrying_from is None:
            self.emit({"kind": "user", "text": shown or text, "agent": self.current.name})
        if told:
            self.emit({"kind": "brief", "agent": self.current.name, "text": told})
        seat.seen = len(self.events)
        started = time.monotonic()
        before = snapshot(self.cfg.folder)
        try:
            result = await self.agent.prompt(self.session_id, ([told] if told else []) + [text])
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
            self.note_written(before)
            self.emit({"kind": "files", **self.changes()})
            seat.seen = len(self.events)                     # its own work is in its context
            if self.handover is not None:
                asyncio.get_running_loop().create_task(self.carry_on(self.handover, self.current.name))

    def note_written(self, before: dict[str, str]) -> None:
        """Who wrote what in the task just done: each text document the agent added or changed gets a note
        of it — the agent and the model — so that what a person later writes over it is a pair for learning
        (:func:`deidkit.agents.review.records`). Only documents the review can show; nothing in .legix."""
        try:
            now = snapshot(self.cfg.folder)
            model = self._model_name()
            for path, digest in now.items():
                how = "added" if path not in before else "changed" if before[path] != digest else ""
                if how and review.writable(path):
                    creator = {"type": "Software", "name": self.current.label, **({"model": model} if model else {})}
                    review.store(self.cfg.folder, path, review.written(path=path, creator=creator, how=how))
        except OSError:
            log.warning("who wrote the documents of the task was not noted")

    async def carry_on(self, choice: AgentChoice, previous: str) -> None:
        """The task under way, stopped so that the person could hand it to ``choice``, goes on with
        ``choice``: in its own session, told what was done. ``previous`` keeps its session for later."""
        try:
            if not self.take_seat(choice):
                await self.launch(choice)
        except Exception as exc:                     # noqa: BLE001
            log.exception("the agent picked to carry the task on did not start")
            self.status = "failed"
            self.emit({"kind": "error", "text": f"{choice.name} did not start: {exc}"})
        if self.status != "ready":
            self.handover = None                     # the error is on the page; the task waits for the person
            return
        self.emit({"kind": "handover", "from": previous, "to": choice.name})
        self.handover = None                         # no await between this and the task's start
        await self.run_prompt(CARRY_ON.format(previous=previous), carrying_from=previous)

    async def stop_turn(self) -> None:
        """Stop the turn under way: ``session/cancel``, and the asks it waits on answered as cancelled —
        what ACP asks of a client that cancels."""
        if self.agent and self.session_id:
            try:
                await self.agent.cancel(self.session_id)
            except Exception:                        # noqa: BLE001 — a gone agent stops anyway
                log.warning("the cancel did not reach the agent")
        for fut in list(self.permissions.values()):
            if not fut.done():
                fut.set_result({"outcome": "cancelled"})

    async def stop_if_stuck(self, seat: Seat, after: float = 30.0) -> None:
        """An agent that does not stop when asked, so that its task can be handed over, is stopped: the
        task goes on with the other one, and this one starts a new session when it is picked again."""
        await asyncio.sleep(after)
        if self.busy and self.handover is not None and seat is self.seat and seat.agent:
            self.emit({"kind": "notice", "text": f"{seat.choice.name} did not stop when asked: it was stopped."})
            await seat.agent.stop()

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
                                      "handover": self.handover.name if self.handover else None,
                                      "usage": self.usage, "context": self.context,
                                      **self.changes()})
        if name == "events" and request.method == "GET":
            return await self._events(request)
        if name == "prompt" and request.method == "POST":
            text = ((await request.json()).get("text") or "").strip()
            if not text:
                return web.json_response({"error": "empty task"}, status=400)
            if self.busy or self.status != "ready" or self.handover is not None:
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
            body = await request.json()
            return await self._set_agent((body.get("value") or "").strip(), handover=body.get("handover") is True)
        if name == "cancel" and request.method == "POST":
            await self.stop_turn()
            return web.json_response({"ok": True})
        if name == "review" and request.method == "GET":
            return await self._review_view(request.query.get("path", ""))
        if name == "review" and request.method == "POST":
            body = await request.json()
            body = body if isinstance(body, dict) else {}
            return await self._review_start(str(body.get("path") or "").strip(), str(body.get("agent") or "").strip())
        if name in ("review-suggest", "review-decide", "review-text", "review-vote") and request.method == "POST":
            body = await request.json()
            if not isinstance(body, dict):
                return web.json_response({"error": "bad request"}, status=400)
            return await {"review-suggest": self._review_suggest, "review-decide": self._review_decide,
                          "review-text": self._review_text, "review-vote": self._review_vote}[name](body)
        if name == "review-dataset" and request.method == "GET":
            records = await review.dataset(self.cfg.folder, self.proxy.engine(self.cfg.scope))
            return web.json_response({"records": records, "count": len(records)})
        if name == "file" and request.method == "GET":
            rel = request.query.get("path", "")
            target = self.cfg.folder / rel
            if not inside(self.cfg.folder, str(target)) or not target.is_file():
                return web.json_response({"error": "no such file"}, status=404)
            data = target.read_bytes()[:_MAX_FILE]
            return web.json_response({"path": rel, "text": data.decode("utf-8", "replace")})
        return web.json_response({"error": "not found"}, status=404)

    # ── review ───────────────────────────────────────────────────────────────
    async def _review_view(self, path: str) -> web.Response:
        """The document ``path`` under review: its text, and over it what is de-identified, the norms it
        refers to and the agents' notes (:mod:`deidkit.agents.review`)."""
        read = review.read_text(self.cfg.folder, path)
        if read is None:
            return web.json_response({"error": "no text for this document", "source": review.text_source(path)},
                                     status=404)
        source, text = read
        red = (await self.proxy.engine(self.cfg.scope).tokenize_all([text]))[0]
        deid = review.deid_spans(text, red.text, red.surfaces or {}, red.mapping or {})
        return web.json_response({
            "path": path, "source": source, "text": text, "deid": deid,
            "norms": review.norm_mentions(text),
            **review.view(self.cfg.folder, path, text),
            "writable": review.writable(path), "sha": review.sha(text), "person": self.cfg.person,
            "agent": {"name": self.current.label, "busy": self.busy, "ready": self.status == "ready"},
        })

    async def _review_suggest(self, body: dict) -> web.Response:
        """A change the person proposes: positions in characters of the text as the page has it."""
        path, start, end = str(body.get("path") or ""), body.get("start"), body.get("end")
        replacement = body.get("text") if isinstance(body.get("text"), str) else None
        read = review.read_text(self.cfg.folder, path)
        if read is None:
            return web.json_response({"error": "no text for this document"}, status=404)
        text = read[1]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)) \
                or replacement is None or len(replacement) > review.MAX_REPLACEMENT \
                or (start == end and not replacement) or text[start:end] == replacement:
            return web.json_response({"error": "no change to propose"}, status=400)
        if body.get("sha") and body["sha"] != review.sha(text):
            return web.json_response({"error": "the document changed: open it again"}, status=409)
        ann = review.suggestion(path=path, text=text, start=start, end=end, replacement=replacement,
                                creator=review.person(self.cfg.person))
        review.store(self.cfg.folder, path, ann)
        return web.json_response({"ok": True, "id": ann["id"]})

    async def _review_decide(self, body: dict) -> web.Response:
        """What the person decided of a note or a proposed change. Accepted, a change is made in the
        document — one whose text is the document itself; a Word document is changed in Word."""
        path, target, verdict = str(body.get("path") or ""), str(body.get("target") or ""), body.get("verdict")
        comment = review._text(body.get("comment"), review.MAX_NOTE)
        correction = body["correction"].strip()[:review.MAX_REPLACEMENT] if isinstance(body.get("correction"), str) else ""
        if verdict not in review.VERDICTS:
            return web.json_response({"error": "no such decision"}, status=400)
        ann = review.find(self.cfg.folder, path, target)
        if ann is None or review._kind(ann) in ("decision", "revision", "written"):
            return web.json_response({"error": "no such note"}, status=404)
        applied, why = False, ""
        replacement = review._body(ann, "replacing", review.MAX_REPLACEMENT)
        if verdict == "accepted" and (replacement or review._kind(ann) == "edit"):
            if not review.writable(path):
                why = "a Word or PDF document is changed in its own program: the decision is kept"
            else:
                read = review.read_text(self.cfg.folder, path)
                changed = review.apply(read[1], ann, replacement) if read else None
                if changed is None:
                    return web.json_response({"error": "its words are no longer in the document"}, status=409)
                review.write_text(self.cfg.folder, path, changed)
                applied = True
        review.store(self.cfg.folder, path, review.decision(
            path=path, target=target, verdict=verdict, creator=review.person(self.cfg.person),
            comment=comment, correction=correction, applied=applied))
        return web.json_response({"ok": True, "applied": applied, **({"why": why} if why else {})})

    async def _review_vote(self, body: dict) -> web.Response:
        """The person's 👍 or 👎 on an answer of the agent in the chat, kept in the matter's notes folder for
        learning (``review-dataset``); a later vote on the same answer replaces it."""
        key, verdict = body.get("key"), body.get("verdict")
        question, answer = body.get("question"), body.get("answer")
        if verdict not in ("up", "down") or not isinstance(key, str) or not key.strip() or len(key) > 200 \
                or not isinstance(answer, str) or not answer.strip() or not isinstance(question, str):
            return web.json_response({"error": "no such vote"}, status=400)
        correction = body["correction"].strip()[:review.MAX_REPLACEMENT] if isinstance(body.get("correction"), str) else ""
        ann = review.vote(key=key, question=question, answer=answer, agent=str(body.get("agent") or ""),
                          model=str(body.get("model") or ""), verdict=verdict, creator=review.person(self.cfg.person),
                          comment=review._text(body.get("comment"), review.MAX_NOTE), correction=correction)
        review.store_vote(self.cfg.folder, ann)
        return web.json_response({"ok": True, "id": ann["id"]})

    async def _review_text(self, body: dict) -> web.Response:
        """The person's own writing, from the editor: the whole text of a Markdown or text document, over
        the text it was opened with (``sha``; empty — a new document). The change is kept as theirs."""
        path, text, base = str(body.get("path") or ""), body.get("text"), str(body.get("sha") or "")
        if not isinstance(text, str) or len(text.encode("utf-8")) > review.MAX_TEXT:
            return web.json_response({"error": "no text"}, status=400)
        me = review.person(self.cfg.person)
        if not base:
            if not review.new_path_ok(self.cfg.folder, path):
                return web.json_response({"error": "a new document is a .md or .txt file of the folder, "
                                                   "not one that is there"}, status=400)
            review.write_text(self.cfg.folder, path, text)
            by = body.get("written_by") if isinstance(body.get("written_by"), dict) else None
            creator = ({"type": "Software", "name": review._text(by.get("agent"), 80) or "Agent",
                        **({"model": review._text(by.get("model"), 80)} if by.get("model") else {})} if by else me)
            review.store(self.cfg.folder, path, review.written(path=path, creator=creator, how="wrote"))
            if by:
                # The person saved what the agent wrote: the agent wrote it, the person keeps it.
                review.store(self.cfg.folder, path, review.written(path=path, creator=me, how="saved"))
            return web.json_response({"ok": True, "sha": review.sha(text)})
        if not review.writable(path):
            return web.json_response({"error": "a Word or PDF document is changed in its own program"}, status=400)
        read = review.read_text(self.cfg.folder, path)
        if read is None:
            return web.json_response({"error": "no such document"}, status=404)
        if review.sha(read[1]) != base:
            return web.json_response({"error": "the document changed since it was opened: open it again"}, status=409)
        if text != read[1]:
            review.write_text(self.cfg.folder, path, text)
            review.store(self.cfg.folder, path, review.revision(path=path, before=read[1], after=text, creator=me))
        return web.json_response({"ok": True, "sha": review.sha(text)})

    async def _review_start(self, path: str, agent: str = "") -> web.Response:
        """A review by the agent at work, or by ``agent`` — a second reviewer: it takes the document between
        tasks, and the one at work comes back after it."""
        read = review.read_text(self.cfg.folder, path)
        if read is None:
            return web.json_response({"error": "no text for this document", "source": review.text_source(path)},
                                     status=404)
        reviewer = None
        if agent and agent not in (self.current.name, self.current.label):
            reviewer = next((a for a in self.agents if agent in (a.name, a.label)), None)
            if reviewer is None:
                return web.json_response({"error": "there is no such agent"}, status=400)
            if reviewer.unavailable:
                return web.json_response({"error": f"{reviewer.name} cannot start here: {reviewer.unavailable}"},
                                         status=409)
        if self.busy or self.status != "ready" or self.handover is not None:
            return web.json_response({"error": "the agent is busy or not ready"}, status=409)
        run = uuid.uuid4().hex[:12]
        self.busy = True                     # taken now: a second review or a task cannot slip in before it starts
        asyncio.create_task(self.run_review(path, read[0], run, reviewer))
        return web.json_response({"ok": True, "run": run}, status=202)

    async def _at_work(self, choice: AgentChoice) -> bool:
        """``choice`` becomes the agent at work, between tasks (started if it has no seat yet); False — it did
        not start."""
        if not self.take_seat(choice):
            await self.launch(choice)
        return self.status == "ready"

    def _model_name(self) -> str:
        """The model that answered the task just done, as the provider named it; else as the agent does."""
        if self.requests and self.requests[-1].model:
            return self.requests[-1].model
        models = self.models or {}
        return next((o["name"] for o in models.get("options") or [] if o.get("value") == models.get("current")),
                    str(models.get("current") or ""))

    async def run_review(self, path: str, source: str, run: str, reviewer: AgentChoice | None = None) -> None:
        """The agent at work reviews the document — or ``reviewer``, which takes over between tasks and hands
        back to the agent that was at work when it is done; the notes whose quotes are in the text are kept."""
        if reviewer is None:
            return await self._review(path, source, run)
        back = self.current
        try:
            if not await self._at_work(reviewer):
                self.emit({"kind": "review", "state": "failed", "path": path, "run": run, "agent": reviewer.label,
                           "text": f"{reviewer.name} did not start"})
                return
            await self._review(path, source, run)
        finally:
            self.busy = True                 # nothing slips in while the agent at work comes back
            try:
                if self.current.name != back.name:
                    await self._at_work(back)
            finally:
                self.busy = False

    async def _review(self, path: str, source: str, run: str) -> None:
        agent = self.current.label
        self.emit({"kind": "review", "state": "started", "path": path, "run": run, "agent": agent})
        at = len(self.events)
        self.busy = False                    # run_prompt takes it again at once, with no await between
        await self.run_prompt(review.prompt(path, source), shown=f"Review {path}")
        answer = "".join(e.get("text", "") for e in self.events[at:] if e.get("kind") == "message")
        failed = next((e.get("text") for e in self.events[at:] if e.get("kind") == "error"), None)
        notes, cut = review.notes_of(answer)
        read = review.read_text(self.cfg.folder, path)
        if notes is None or read is None:
            # The agent's own words when it said why (MatterAgent: "The model's server did not answer…") — not ours.
            said = " ".join(answer.split())
            why = failed or ("the document's text is gone" if read is None else
                             said if 0 < len(said) <= 400 else "the agent did not answer with notes")
            self.emit({"kind": "review", "state": "failed", "path": path, "run": run, "agent": agent, "text": why})
            return
        model = self._model_name()
        creator = {"type": "Software", "name": agent, **({"model": model} if model else {})}
        kept = review.keep(self.cfg.folder, path, read[1], notes, creator=creator, run=run)
        self.emit({"kind": "review", "state": "done", "path": path, "run": run, "agent": agent, "model": model,
                   "kept": len(kept["kept"]), "refused": kept["refused"], **({"cut": True} if cut else {})})

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

    async def _set_agent(self, value: str, handover: bool = False) -> web.Response:
        """Switch the agent at work. Between tasks, the one picked goes on in its own session (or starts
        one), told with its next task what was done that it has not seen. During a task, only when the
        person hands the task over (``handover``): the step under way stops, and the one picked carries the
        task on, told what was done. The one that worked keeps its session in both cases."""
        choice = next((a for a in self.agents if a.name == value), None)
        if len(self.agents) < 2 or choice is None:
            return web.json_response({"error": "there is no such agent"}, status=400)
        if choice.unavailable:
            return web.json_response({"error": f"{choice.name} cannot start here: {choice.unavailable}"},
                                     status=409)
        if self.handover is not None:
            return web.json_response({"error": f"the task is being handed over to {self.handover.name}"},
                                     status=409)
        if self.busy:
            if not handover:
                return web.json_response({"error": "the agent is working: switch between tasks, or hand "
                                                   "the task over"}, status=409)
            if choice.name == self.current.name:
                return web.json_response({"ok": True, "agents": self.agent_choice()})
            self.handover = choice
            self.emit({"kind": "notice", "handing": True,
                       "text": f"Handing the task over to {choice.name}: {self.current.name} stops at its current step."})
            await self.stop_turn()
            asyncio.get_running_loop().create_task(self.stop_if_stuck(self.seat))
            return web.json_response({"ok": True, "handover": choice.name, "agents": self.agent_choice()},
                                     status=202)
        if self.status == "starting":
            return web.json_response({"error": "the agent is starting"}, status=409)
        if choice.name == self.current.name and self.status == "ready":
            return web.json_response({"ok": True, "agents": self.agent_choice()})
        if not self.take_seat(choice):
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

    def _drop_codex_dir(self, seat: Seat) -> None:
        """A confined Codex's profile and wrapper, once it no longer runs."""
        if seat.codex_dir:
            shutil.rmtree(seat.codex_dir, ignore_errors=True)
            seat.codex_dir = None

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
            for seat in self.seats.values():
                if seat.agent:
                    await seat.agent.stop()
                self._drop_codex_dir(seat)
            await self.proxy._stop(app)

        app.on_startup.append(startup)
        app.on_cleanup.append(cleanup)
        return app


__all__ = ["AgentChoice", "AgentHost", "CARRY_ON", "CODEX_SUBAGENTS_NOTE", "HostConfig", "Seat", "brief", "inside",
           "model_choice", "snapshot", "turn_usage", "with_subagents"]
