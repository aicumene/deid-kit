# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""A local proxy that puts deid-kit between a coding agent and its model.

The agent works on the real files; the model receives tokens. See ``docs/coding-agents.md``.
Anthropic's Messages API (Claude Code) is implemented here; it needs the ``proxy`` extra
(``aiohttp``). Run it with ``deid-proxy`` or ``python -m deidkit.proxy``.
"""
