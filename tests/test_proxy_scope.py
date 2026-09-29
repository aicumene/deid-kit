# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""The scope of a request: its header, else the folder the agent works in, else the default."""

import os

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web                                     # noqa: E402
from aiohttp.test_utils import make_mocked_request          # noqa: E402

from deidkit.proxy.server import Proxy, ProxyConfig, scope_for_folder, working_directory  # noqa: E402
from deidkit.seedfile import load_scope_paths               # noqa: E402
from deidkit.store import InMemoryTokenStore                # noqa: E402

CLAUDE = {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=9;"},
                     {"type": "text", "text": "You are Claude Code.\nPrimary working directory: {cwd}\nIs a git repository: false"}],
          "messages": []}
CODEX = {"input": [{"type": "message", "role": "user", "content": [
    {"type": "input_text", "text": "<environment_context>\n  <cwd>{cwd}</cwd>\n  <shell>zsh</shell>\n</environment_context>"}]}]}


def body(template, cwd):
    import json
    return json.loads(json.dumps(template).replace("{cwd}", cwd))


def test_each_agent_names_its_folder(tmp_path):
    assert working_directory(body(CLAUDE, str(tmp_path))) == str(tmp_path)
    assert working_directory(body(CODEX, str(tmp_path))) == str(tmp_path)
    assert working_directory({"messages": []}) is None


def test_seed_files_list_folders_and_the_longest_wins(tmp_path):
    outer, inner = tmp_path / "matters", tmp_path / "matters" / "client-b"
    inner.mkdir(parents=True)
    (tmp_path / "a.toml").write_text(f'scope = "client-a"\npaths = ["{outer}"]\n')
    (tmp_path / "b.toml").write_text(f'scope = "client-b"\npaths = ["{inner}"]\n')
    pairs = load_scope_paths([tmp_path / "a.toml", tmp_path / "b.toml"], "default")
    assert scope_for_folder(str(inner / "sub"), pairs) == "client-b"
    assert scope_for_folder(str(outer / "other"), pairs) == "client-a"
    assert scope_for_folder(str(tmp_path / "elsewhere"), pairs) is None
    assert scope_for_folder(str(tmp_path / "matters-old"), pairs) is None     # a prefix is not a folder


def test_a_symlinked_path_resolves_to_the_same_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    pairs = [(os.path.realpath(real), "client-a")]
    assert scope_for_folder(str(link / "x"), pairs) == "client-a"


def test_header_then_folder_then_default_or_refusal(tmp_path):
    pairs = [(os.path.realpath(tmp_path), "client-a")]
    proxy = Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="default", scope_paths=pairs))
    strict = Proxy(InMemoryTokenStore(), ProxyConfig(default_scope="default", scope_paths=pairs,
                                                     require_scope=True))
    with_header = make_mocked_request("POST", "/v1/messages", headers={"x-deid-scope": "client-z"})
    plain = make_mocked_request("POST", "/v1/messages")
    inside, outside = body(CLAUDE, str(tmp_path / "doc")), body(CODEX, "/nowhere/else")
    assert proxy.scope(with_header, inside) == "client-z"
    assert proxy.scope(plain, inside) == "client-a"
    assert proxy.scope(plain, outside) == "default"
    assert strict.scope(plain, outside) is None
