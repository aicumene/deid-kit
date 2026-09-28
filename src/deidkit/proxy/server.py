# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The local HTTP proxy between a coding agent and Anthropic's API (``aiohttp``, extra ``proxy``).

``POST /v1/messages`` and ``POST /v1/messages/count_tokens`` are de-identified on the way out and,
for messages, re-identified on the way back. ``GET`` and ``HEAD`` requests carry no content and
are forwarded as they are. Anything else is refused: a request this proxy cannot read is not
sent. Every crossing is audited through :class:`~deidkit.gateway.PrivacyGateway` (sensitivity,
token kinds, count, SHA-256 of what crossed — never content). With a record file, the proxy also
writes what crossed, in tokens, for checking.

The agent's own credentials pass through untouched; the proxy stores none of them and writes no
header to the record.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

from deidkit.classification import Sensitivity
from deidkit.gateway import CrossingRecord, InMemoryAuditSink, PrivacyGateway, Redaction
from deidkit.proxy import sse
from deidkit.proxy.anthropic import (LOCAL_TOOLS, Refused, StreamRestorer, prepare_request,
                                     restore_json_response)
from deidkit.proxy.engine import ScopeEngine

log = logging.getLogger("deidkit.proxy")

_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer",
        "upgrade", "proxy-authorization", "proxy-connection", "accept-encoding", "x-deid-scope"}
_BACK_DROP = {"content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


@dataclass
class ProxyConfig:
    default_scope: str = "default"
    upstream: str = "https://api.anthropic.com"
    binary: str = "withhold"
    local_tools: frozenset[str] = LOCAL_TOOLS
    record: Path | None = None
    audit: Path | None = None
    detector: object | None = None
    glossary: bool = False
    seeds: object | None = None
    extra: dict = field(default_factory=dict)


class JsonlAuditSink:
    """An :class:`~deidkit.gateway.AuditSink` appending one JSON line per crossing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, context, record: CrossingRecord) -> None:
        line = {"at": time.time(), "actor": str(record.actor_id), "action": record.action,
                "plane": record.plane, "sensitivity": record.sensitivity,
                "sha256": record.content_hash, **record.detail}
        _append(self.path, line)


def _append(path: Path, obj: dict) -> None:
    fresh = not path.exists()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if fresh:
        os.chmod(path, 0o600)


def _never(text: str) -> Redaction:
    raise RuntimeError("the proxy always hands the gateway a pre-redacted payload")


def _error(status: int, message: str) -> web.Response:
    kind = "invalid_request_error" if status < 500 else "api_error"
    return web.json_response({"type": "error", "error": {"type": kind, "message": message}},
                             status=status)


class MemoryRestore:
    """The model's own words, in process memory (a store without a ``restore`` table)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def restore_get(self, key: str) -> str | None:
        return self._d.get(key)

    def restore_put(self, key: str, value: str) -> None:
        self._d[key] = value


class Proxy:
    def __init__(self, store, config: ProxyConfig) -> None:
        self.store = store
        self.restore = store if hasattr(store, "restore_get") else MemoryRestore()
        self.cfg = config
        self.engines: dict[str, ScopeEngine] = {}
        sink = JsonlAuditSink(config.audit) if config.audit else InMemoryAuditSink()
        self.gateway = PrivacyGateway(audit=sink, redact_fn=_never)
        self.session: ClientSession | None = None

    def engine(self, scope: str) -> ScopeEngine:
        if scope not in self.engines:
            self.engines[scope] = ScopeEngine(self.store, scope, seeds=self.cfg.seeds,
                                              detector=self.cfg.detector,
                                              glossary=self.cfg.glossary)
        return self.engines[scope]

    def _record(self, obj: dict) -> None:
        if self.cfg.record:
            _append(self.cfg.record, {"at": time.time(), **obj})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        app.on_startup.append(self._start)
        app.on_cleanup.append(self._stop)
        return app

    async def _start(self, app) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30,
                                                           sock_read=900),
                                     auto_decompress=True)

    async def _stop(self, app) -> None:
        if self.session:
            await self.session.close()

    def _headers(self, request: web.Request) -> dict[str, str]:
        out = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        out["Accept-Encoding"] = "identity"
        return out

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.method == "POST" and request.path in ("/v1/messages", "/v1/messages/count_tokens"):
            return await self._messages(request)
        if request.method in ("GET", "HEAD"):
            return await self._pass(request)
        log.warning("refused %s %s", request.method, request.path)
        return _error(403, f"deid proxy: {request.method} {request.path} is not forwarded, "
                           "because the proxy cannot de-identify it")

    async def _pass(self, request: web.Request) -> web.Response:
        url = self.cfg.upstream.rstrip("/") + request.path_qs
        async with self.session.request(request.method, url, headers=self._headers(request)) as up:
            body = await up.read()
            headers = {k: v for k, v in up.headers.items() if k.lower() not in _BACK_DROP}
            log.info("%s %s -> %s", request.method, request.path, up.status)
            return web.Response(status=up.status, body=body, headers=headers)

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        counting = request.path.endswith("/count_tokens")
        scope = request.headers.get("x-deid-scope") or self.cfg.default_scope
        if request.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
            return _error(415, "deid proxy: compressed request bodies are not supported")
        try:
            body = json.loads(await request.read())
        except ValueError:
            return _error(400, "deid proxy: the request body is not JSON")
        engine = self.engine(scope)
        try:
            prep = await prepare_request(body, engine, restore_get=self.restore.restore_get,
                                         binary=self.cfg.binary)
        except Refused as exc:
            log.warning("refused %s: %s", request.path, exc)
            return _error(400, f"deid proxy: {exc}")
        except Exception:                                   # noqa: BLE001 — fail closed
            log.exception("de-identification failed; nothing sent")
            return _error(500, "deid proxy: de-identification failed, so nothing was sent")

        sent = json.dumps(prep.body, ensure_ascii=False, separators=(",", ":"))
        kinds = sorted({t.rsplit("_", 1)[0] for t in prep.mapping})
        await self.gateway.cross_to_cloud(
            None, actor_id=scope, action=request.path, sensitivity=Sensitivity.INTERNAL,
            payload=sent, target=self.cfg.upstream,
            pre_redacted=Redaction(text=sent, entity_types=kinds, redacted_count=len(prep.mapping)))
        self._record({"dir": "out", "scope": scope, "path": request.path_qs, "body": prep.body,
                      "withheld": prep.withheld})
        log.info("%s scope=%s tokens=%d withheld=%d restored=%d", request.path, scope,
                 len(prep.mapping), len(prep.withheld), prep.restored)

        url = self.cfg.upstream.rstrip("/") + request.path_qs
        up = await self.session.post(url, data=sent.encode("utf-8"), headers=self._headers(request))
        try:
            headers = {k: v for k, v in up.headers.items() if k.lower() not in _BACK_DROP}
            streamed = "text/event-stream" in up.headers.get("content-type", "")
            if up.status >= 400 or not streamed:
                data = await up.read()
                if up.status < 400 and not counting:
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("type") == "message":
                        self._record({"dir": "in", "scope": scope, "body": obj})
                        obj = await restore_json_response(obj, engine, prep.mapping,
                                                          self.restore.restore_put,
                                                          self.cfg.local_tools)
                        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                return web.Response(status=up.status, body=data, headers=headers)

            resp = web.StreamResponse(status=up.status, headers=headers)
            await resp.prepare(request)
            restorer = StreamRestorer(engine, prep.mapping, self.restore.restore_put,
                                      self.cfg.local_tools)
            try:
                async for name, data in sse.events(up.content):
                    if not name and data.startswith(":"):
                        await resp.write((data + "\n\n").encode("utf-8"))
                        continue
                    self._record({"dir": "in", "scope": scope, "event": name, "data": data})
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        await resp.write(sse.encode(name, data))
                        continue
                    for out_name, out_data in await restorer.event(name, obj):
                        await resp.write(sse.encode(out_name, out_data))
            except Exception:                               # noqa: BLE001
                log.exception("re-identification failed mid-stream")
                await resp.write(sse.encode("error", {"type": "error", "error": {
                    "type": "api_error", "message": "deid proxy: the answer could not be "
                                                    "re-identified; please retry"}}))
            await resp.write_eof()
            return resp
        finally:
            up.release()


__all__ = ["JsonlAuditSink", "MemoryRestore", "Proxy", "ProxyConfig"]
