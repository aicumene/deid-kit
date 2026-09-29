# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""``deid-proxy``: run the de-identification proxy on this machine.

    deid-proxy --scope client-a --seeds ./deid.toml
    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude          # Claude Code
    codex -c model_provider=deid ...                          # Codex, see docs/coding-agents.md

The vault (``~/.deid/vault.sqlite``) and the audit (``~/.deid/audit.jsonl``) stay on this machine;
both are readable by their owner only. ``--record FILE`` also writes what crossed, in tokens,
for checking that nothing else did.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import sys
from pathlib import Path

from aiohttp import web

from deidkit.patterns import RegexDetector
from deidkit.proxy.server import Proxy, ProxyConfig
from deidkit.seedfile import load_seed_files
from deidkit.sqlite_store import SQLiteTokenStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="deid-proxy", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default: loopback)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--upstream", default="https://api.anthropic.com",
                    help="Anthropic's API, for Claude Code")
    ap.add_argument("--upstream-openai", default="https://api.openai.com/v1",
                    help="OpenAI's API, for Codex with an API key")
    ap.add_argument("--upstream-chatgpt", default="https://chatgpt.com/backend-api/codex",
                    help="the ChatGPT backend, for Codex signed in with ChatGPT")
    ap.add_argument("--vault", default="~/.deid/vault.sqlite", help="token store (SQLite)")
    ap.add_argument("--scope", default="default",
                    help="scope for requests without an x-deid-scope header")
    ap.add_argument("--seeds", action="append", default=[], metavar="TOML",
                    help="known people and organisations (repeatable)")
    ap.add_argument("--audit", default="~/.deid/audit.jsonl")
    ap.add_argument("--record", help="write what crossed (tokens only) to this JSONL file")
    ap.add_argument("--binary", choices=["withhold", "refuse"], default="withhold",
                    help="images and PDFs: replace with a note (default) or refuse the request")
    ap.add_argument("--kinds", default="EMAIL,IBAN,CARD,PHONE",
                    help="pattern kinds to detect beyond the known names")
    ap.add_argument("--glossary", action="store_true", help="send a glossary line per token")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback:
        print(f"deid-proxy: listening on {args.host}, not on loopback: anyone who reaches it can "
              "read the answers in clear", file=sys.stderr)

    store = SQLiteTokenStore(args.vault)
    seeds = load_seed_files(args.seeds, args.scope)
    kinds = tuple(k.strip().upper() for k in args.kinds.split(",") if k.strip())
    cfg = ProxyConfig(default_scope=args.scope, upstream=args.upstream,
                      upstream_openai=args.upstream_openai, upstream_chatgpt=args.upstream_chatgpt,
                      binary=args.binary,
                      record=Path(args.record).expanduser() if args.record else None,
                      audit=Path(args.audit).expanduser() if args.audit else None,
                      detector=RegexDetector(kinds) if kinds else None,
                      glossary=args.glossary, seeds=seeds)
    web.run_app(Proxy(store, cfg).app(), host=args.host, port=args.port,
                print=lambda msg: print(f"deid-proxy: {args.host}:{args.port} -> {args.upstream}, "
                                        f"{args.upstream_chatgpt}, {args.upstream_openai}",
                                        file=sys.stderr))


if __name__ == "__main__":
    main()
