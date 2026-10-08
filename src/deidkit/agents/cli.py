# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-agent``: a coding agent over one folder, behind deid-proxy, with a page to drive it.

    deid-agent --folder ~/matters/client-a --scope client-a --seeds ~/.deid/seeds/client-a.toml

The page's address, with its token, is printed on start. The organization's key comes from
ANTHROPIC_API_KEY and stays in the proxy; the agent gets a stand-in. A program that starts
deid-agent and opens the page itself passes ``--secrets-stdin`` and writes one JSON line,
``{"token": "…", "api_key": "…"}``, instead: a process's environment and arguments can be read by
other processes of the same user, its stdin cannot. ``--dev-login`` lets the agent use the
machine's own Claude sign-in, for development only: a product must not offer a personal
subscription sign-in to its users.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path

from aiohttp import web

from deidkit.agents.host import AgentChoice, AgentHost, HostConfig
from deidkit.patterns import RegexDetector
from deidkit.proxy.cli import seed_files
from deidkit.proxy.server import Proxy, ProxyConfig
from deidkit.seedfile import load_scope_paths, load_seed_files
from deidkit.seeds import SeedEntity
from deidkit.sqlite_store import SQLiteTokenStore

#: The Claude adapter at a fixed version. Unpinned, `npx -y` fetches each new release at the
#: agent's start: on 1 October 2026 a new release brought a new Agent SDK with its 216 MB
#: `claude`, and the agent took two minutes to start — with behaviour nobody had checked.
#: Raise the version on purpose, after a run against it.
DEFAULT_AGENT = "npx -y @agentclientprotocol/claude-agent-acp@0.85.0"
#: Account names that say nothing about who works on the machine.
_GENERIC_ACCOUNTS = {"user", "users", "admin", "administrator", "root", "home", "guest", "default",
                     "public", "shared", "owner", "office"}


def account_name(home: Path | None = None) -> str | None:
    """The operating-system account the agent runs under, as its home folder names it — enrolled
    as a known name of the scope, so absolute paths cross as ``/Users/ACCOUNT_…/…``.

    MEASURED 30.09.2026: in one task of a matter agent the account name crossed 103 times, the
    home path 61 — the agent's instructions name its working directory and its memory folder, and
    every absolute path a tool prints begins with them. A generic name ("admin") identifies
    nobody and is left alone."""
    name = (home or Path.home()).name
    return name if len(name) >= 3 and name.casefold() not in _GENERIC_ACCOUNTS else None


def read_secrets(stream) -> dict:
    """The one JSON line a starting program writes: ``token`` and ``api_key`` (either may be null)."""
    line = stream.readline()
    try:
        given = json.loads(line) if line.strip() else {}
    except ValueError:
        given = None
    if not isinstance(given, dict):
        sys.exit("deid-agent: --secrets-stdin expects one JSON line with token and api_key")
    return given


def json_object(text: str, option: str) -> dict:
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        sys.exit(f"deid-agent: {option} expects a JSON object, not {text!r}")
    return value


def agent_choices(agents: list[str], upstreams: list[str], unavailable: list[str],
                  metas: list[str] | None = None) -> list[AgentChoice]:
    """The agents of ``--agent NAME=COMMAND``, in their order, with ``--agent-upstream NAME=URL``,
    ``--agent-unavailable NAME=REASON`` and ``--agent-meta NAME=JSON``; a name only in
    ``--agent-unavailable`` is listed after them."""
    def pairs(specs: list[str], option: str) -> list[tuple[str, str]]:
        out = []
        for spec in specs:
            name, sep, value = spec.partition("=")
            if not sep or not name.strip() or not value.strip():
                sys.exit(f"deid-agent: {option} expects NAME=VALUE, not {spec!r}")
            out.append((name.strip(), value.strip()))
        return out

    choices: dict[str, AgentChoice] = {}
    for name, command in pairs(agents, "--agent"):
        choices[name] = AgentChoice(name=name, command=shlex.split(command))
    for name, url in pairs(upstreams, "--agent-upstream"):
        if name not in choices:
            sys.exit(f"deid-agent: --agent-upstream names no agent: {name}")
        choices[name].upstream = url.rstrip("/")
    for name, reason in pairs(unavailable, "--agent-unavailable"):
        choices.setdefault(name, AgentChoice(name=name)).unavailable = reason
    for name, meta in pairs(metas or [], "--agent-meta"):
        if name not in choices:
            sys.exit(f"deid-agent: --agent-meta names no agent: {name}")
        choices[name].meta = json_object(meta, "--agent-meta")
    return list(choices.values())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="deid-agent", description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", required=True, help="the folder the agent works in")
    ap.add_argument("--scope", required=True, help="the folder's deid scope")
    ap.add_argument("--title", help="shown on the page (default: the folder's name)")
    ap.add_argument("--seeds", action="append", default=[], metavar="TOML")
    ap.add_argument("--seeds-dir", metavar="DIR")
    ap.add_argument("--vault", default="~/.deid/agent/vault.sqlite",
                    help="token store; not the one a running deid-proxy uses")
    ap.add_argument("--audit", default="~/.deid/agent/audit.jsonl")
    ap.add_argument("--record", help="write what crossed (tokens only) to this JSONL file")
    ap.add_argument("--usage-log", default="~/.deid/agent/usage.jsonl",
                    help="one JSON line per task: scope, model, seconds, tokens ('' for none)")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--agent-command", default=DEFAULT_AGENT, help="the ACP agent to start")
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=COMMAND",
                    help="an agent the page switches to, between tasks (repeat for each; the first "
                         "that can start works first); instead of --agent-command")
    ap.add_argument("--agent-upstream", action="append", default=[], metavar="NAME=URL",
                    help="that agent's model requests go to this server of the organization's own "
                         "(Messages API), with no key")
    ap.add_argument("--agent-unavailable", action="append", default=[], metavar="NAME=REASON",
                    help="the page lists that agent but cannot start it on this machine, and says why")
    ap.add_argument("--agent-meta", action="append", default=[], metavar="NAME=JSON",
                    help="that agent's own session options, sent as the session's _meta (the Claude "
                         "adapter reads claudeCode.options: which tools Claude Code offers the model)")
    ap.add_argument("--session-meta", metavar="JSON",
                    help="the session's _meta for an agent that names none of its own")
    ap.add_argument("--style", metavar="CSS",
                    help="a stylesheet the page loads after its own: the look of the program that opens it")
    ap.add_argument("--claude-executable", default=shutil.which("claude"),
                    help="the claude binary the Claude adapter should run")
    ap.add_argument("--upstream", default="https://api.anthropic.com",
                    help="where the agent's model requests go: Anthropic, or a server of the "
                         "organization's own that speaks the Messages API (/v1/messages)")
    ap.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                    help="environment variable holding the organization's key")
    ap.add_argument("--secrets-stdin", action="store_true",
                    help='read the page token and the key as one JSON line on stdin '
                         '({"token": ..., "api_key": ...}) instead of the environment')
    ap.add_argument("--dev-login", action="store_true",
                    help="allow the machine's own Claude sign-in when no API key is set")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"deid-agent: no such folder: {folder}")
    if args.secrets_stdin:
        given = read_secrets(sys.stdin)
        token, api_key = given.get("token") or "", given.get("api_key") or None
    else:
        token, api_key = os.environ.pop("DEID_AGENT_TOKEN", ""), os.environ.get(args.api_key_env) or None
    if not api_key and not args.dev_login:
        sys.exit(f"deid-agent: no organization key (set {args.api_key_env}, or pass it with "
                 "--secrets-stdin), or pass --dev-login for development")

    cfg = HostConfig(folder=folder, scope=args.scope, title=args.title or folder.name,
                     agent_command=shlex.split(args.agent_command), api_key=api_key,
                     dev_login=args.dev_login, claude_executable=args.claude_executable,
                     port=args.port,
                     usage_log=Path(args.usage_log).expanduser() if args.usage_log else None,
                     agents=agent_choices(args.agent, args.agent_upstream, args.agent_unavailable,
                                          args.agent_meta),
                     session_meta=json_object(args.session_meta, "--session-meta") if args.session_meta else None,
                     style=Path(args.style).expanduser() if args.style else None)
    if cfg.agents and all(a.unavailable for a in cfg.agents):
        sys.exit("deid-agent: no agent can start here: "
                 + "; ".join(f"{a.name}: {a.unavailable}" for a in cfg.agents))
    if token:
        cfg.token = token
    files = seed_files(args.seeds, args.seeds_dir)
    store = SQLiteTokenStore(args.vault)
    seeds = load_seed_files(files, args.scope)
    account = account_name()
    if account:
        seeds.add_entity(args.scope, SeedEntity("account", account))
    proxy = Proxy(store, ProxyConfig(
        default_scope=args.scope, seeds=seeds, upstream=args.upstream.rstrip("/"),
        scope_paths=load_scope_paths(files, args.scope), detector=RegexDetector(),
        record=Path(args.record).expanduser() if args.record else None,
        audit=Path(args.audit).expanduser(),
        upstream_key=api_key, agent_key=cfg.agent_key if api_key else None))
    host = AgentHost(cfg, proxy)
    # A token handed over by the program that opens the page is not printed: output may go to a log.
    url = f"http://127.0.0.1:{args.port}/" + ("" if token else f"#t={cfg.token}")
    web.run_app(host.app(), host="127.0.0.1", port=args.port,
                print=lambda _: print(f"deid-agent: {url}", flush=True))


if __name__ == "__main__":
    main()
