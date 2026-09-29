# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-agent``: a coding agent over one folder, behind deid-proxy, with a page to drive it.

    deid-agent --folder ~/matters/client-a --scope client-a --seeds ~/.deid/seeds/client-a.toml

The page's address, with its token, is printed on start (or pass the token in DEID_AGENT_TOKEN
when another program opens the page). The agent's credential comes from ANTHROPIC_API_KEY in
this process's environment. ``--dev-login`` lets it use the machine's own Claude sign-in instead,
for development only: a product must not offer a personal subscription sign-in to its users.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import sys
from pathlib import Path

from aiohttp import web

from deidkit.agents.host import AgentHost, HostConfig
from deidkit.patterns import RegexDetector
from deidkit.proxy.cli import seed_files
from deidkit.proxy.server import Proxy, ProxyConfig
from deidkit.seedfile import load_scope_paths, load_seed_files
from deidkit.sqlite_store import SQLiteTokenStore

DEFAULT_AGENT = "npx -y @agentclientprotocol/claude-agent-acp"


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
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--agent-command", default=DEFAULT_AGENT, help="the ACP agent to start")
    ap.add_argument("--claude-executable", default=shutil.which("claude"),
                    help="the claude binary the Claude adapter should run")
    ap.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                    help="environment variable holding the agent's API key")
    ap.add_argument("--dev-login", action="store_true",
                    help="allow the machine's own Claude sign-in when no API key is set")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"deid-agent: no such folder: {folder}")
    api_key = os.environ.get(args.api_key_env) or None
    if not api_key and not args.dev_login:
        sys.exit(f"deid-agent: set {args.api_key_env} (the organization's key), "
                 "or pass --dev-login for development")

    files = seed_files(args.seeds, args.seeds_dir)
    store = SQLiteTokenStore(args.vault)
    proxy = Proxy(store, ProxyConfig(
        default_scope=args.scope, seeds=load_seed_files(files, args.scope),
        scope_paths=load_scope_paths(files, args.scope), detector=RegexDetector(),
        record=Path(args.record).expanduser() if args.record else None,
        audit=Path(args.audit).expanduser()))
    cfg = HostConfig(folder=folder, scope=args.scope, title=args.title or folder.name,
                     agent_command=shlex.split(args.agent_command), api_key=api_key,
                     dev_login=args.dev_login, claude_executable=args.claude_executable,
                     port=args.port)
    given = os.environ.pop("DEID_AGENT_TOKEN", "")
    if given:
        cfg.token = given
    host = AgentHost(cfg, proxy)
    # A token handed over by the program that opens the page is not printed: output may go to a log.
    url = f"http://127.0.0.1:{args.port}/" + ("" if given else f"#t={cfg.token}")
    web.run_app(host.app(), host="127.0.0.1", port=args.port,
                print=lambda _: print(f"deid-agent: {url}", flush=True))


if __name__ == "__main__":
    main()
