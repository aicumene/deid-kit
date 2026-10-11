# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Codex confined to the matter folder (deidkit.agents.confine): the profile closes what it should and
opens only what Codex needs; off macOS Codex is not started unconfined without the person's word; and
on macOS the kernel keeps a confined process inside the matter."""

import os
import subprocess
import sys

import pytest

from deidkit.agents import confine
from deidkit.agents.confine import ConfinementUnavailable, confined_codex, profile


def real(p):
    return os.path.realpath(str(p))


def test_the_profile_closes_homes_disks_and_the_neighbours_and_opens_the_matter(tmp_path):
    matter = tmp_path / "matters" / "case-1"
    matter.mkdir(parents=True)
    tools = tmp_path / "tools"
    (tools / "bin").mkdir(parents=True)
    text = profile(matter, tools / "bin" / "codex", tmp_path / "codex-home")
    deny = next(l for l in text.splitlines() if l.startswith("(deny "))
    assert '(subpath "/Users")' in deny and '(subpath "/Volumes")' in deny
    assert f'(subpath "{real(tmp_path / "matters")}")' in deny          # the other matters beside it
    opened = next(l for l in text.splitlines() if l.startswith("(allow file-read* file-write* "))
    assert f'(subpath "{real(matter)}")' in opened and f'(subpath "{real(tmp_path / "codex-home")}")' in opened
    assert f'(allow file-read* (subpath "{real(tools)}"))' in text            # Codex's install
    assert text.index("(deny ") < text.index("(allow file-read* file-write* ")   # the later rule wins


def test_nothing_too_wide_is_closed_or_opened(tmp_path, monkeypatch):
    # A matter at the top of a disk does not close the disk; Codex in the home folder does not open it.
    assert '(subpath "/")' not in profile("/case-1", "/opt/codex/bin/codex", "/opt/codex-home")
    home = real(os.path.expanduser("~"))
    text = profile(tmp_path / "case", os.path.join(home, "codex"), tmp_path / "h")
    assert f'(subpath "{home}")' not in text and f'(literal "{os.path.join(home, "codex")}")' in text


def test_off_macos_codex_is_not_confined_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(confine.sys, "platform", "linux")
    with pytest.raises(ConfinementUnavailable, match="macOS only"):
        confined_codex(tmp_path, "/opt/codex/bin/codex", tmp_path)


@pytest.mark.skipif(not confine.supported(), reason="Seatbelt is macOS's")
def test_the_kernel_keeps_a_confined_process_in_the_matter(tmp_path):
    parent = tmp_path / "matters"
    (parent / "case-1").mkdir(parents=True)
    (parent / "case-1" / "lease.md").write_text("inside\n")
    (parent / "shared").mkdir()
    (parent / "shared" / "fee-note.md").write_text("fee: 9,750\n")
    tools = tmp_path / "tools" / "bin"
    tools.mkdir(parents=True)
    # "Codex": a script that reads what it is given (a copied system binary is killed by macOS itself)
    (tools / "codex").write_text('#!/bin/sh\nexec /bin/cat "$@"\n')
    (tools / "codex").chmod(0o755)
    wrapper = confined_codex(parent / "case-1", str(tools / "codex"), tmp_path / "w", tmp_path / "codex-home")
    assert os.access(wrapper, os.X_OK) and "sandbox-exec -f" in wrapper.read_text()

    def read(path):
        done = subprocess.run([str(wrapper), str(path)], capture_output=True, text=True, timeout=30)
        return done.returncode, done.stdout + done.stderr

    assert read(parent / "case-1" / "lease.md") == (0, "inside\n")
    code, said = read(parent / "shared" / "fee-note.md")
    assert code != 0 and "Operation not permitted" in said and "9,750" not in said
    code, said = read(os.path.expanduser("~/.zshrc") if os.path.exists(os.path.expanduser("~/.zshrc"))
                      else "/Users/Shared")
    assert code != 0


async def test_the_host_does_not_start_codex_where_it_cannot_be_confined(tmp_path, monkeypatch):
    from deidkit.agents.host import AgentChoice, HostConfig
    from deidkit.proxy.server import ProxyConfig
    from test_agent_host import FAKE, started, wait_for

    monkeypatch.setattr(confine.sys, "platform", "linux")
    agents = [AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])]
    cfg = HostConfig(folder=tmp_path, scope="case-1", title="Case 1", agent_command=[], dev_login=True,
                     codex_executable="/opt/codex/bin/codex", agents=agents)
    host, client = await started(cfg, ProxyConfig(default_scope="case-1"))
    try:
        await wait_for(lambda: host.status == "failed")
        said = " ".join(e.get("text", "") for e in host.events if e["kind"] == "error")
        assert "Codex was not started" in said and "--codex-unconfined" in said
        assert host.agent is None
    finally:
        await client.close()
