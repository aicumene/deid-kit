# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Codex confined to the matter folder (deidkit.agents.confine): the profile closes what it should and
opens only what Codex needs; off macOS Codex is not started unconfined without the person's word; and
on macOS the kernel keeps a confined process inside the matter."""

import os
import subprocess
import sys
import tempfile

import pytest

from deidkit.agents import confine
from deidkit.agents.confine import ConfinementUnavailable, confined_codex, matter_codex_home, profile



@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    """The person's home is the test's: a matter's Codex home and sign-in link are made under it."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEX_HOME", raising=False)

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


def test_the_shared_temp_folders_are_closed_and_the_sign_in_is_the_one_file_opened_beside(tmp_path):
    text = profile(tmp_path / "matters" / "case-1", "/opt/codex/bin/codex", tmp_path / "codex-home",
                   auth_file=tmp_path / "person" / ".codex" / "auth.json")
    deny = next(l for l in text.splitlines() if l.startswith("(deny "))
    assert '(subpath "/private/tmp")' in deny and '(subpath "/private/var/folders")' in deny
    opened = text.splitlines()
    assert f'(allow file-read* file-write* (literal "{real(tmp_path / "person" / ".codex" / "auth.json")}"))' in opened
    assert f'(subpath "{real(tmp_path / "person" / ".codex")}")' not in text      # not the rest of it


def test_a_matter_has_its_own_codex_home_with_the_persons_sign_in_linked(tmp_path):
    person = tmp_path / "person" / ".codex"
    home = tmp_path / "deid" / "codex" / "case-1"
    assert matter_codex_home(home, person) is None and not (home / "auth.json").exists()   # not signed in
    person.mkdir(parents=True)
    (person / "auth.json").write_text('{"tokens": "the person\'s"}')
    auth = matter_codex_home(home, person)
    assert auth == person.resolve() / "auth.json" and (home / "auth.json").is_symlink()
    assert (home / "tmp").is_dir() and matter_codex_home(home, person) == auth            # made once
    (home / "auth.json").write_text('{"tokens": "refreshed"}')                  # a write goes to theirs
    assert (person / "auth.json").read_text() == '{"tokens": "refreshed"}'


@pytest.mark.skipif(not confine.supported(), reason="Seatbelt is macOS's")
def test_the_kernel_keeps_codex_out_of_the_shared_temp_and_of_the_persons_other_sessions(tmp_path):
    """Probed 11.10.2026: a confined Codex read /tmp, $TMPDIR and every session in ~/.codex."""
    matter = tmp_path / "matters" / "case-1"
    matter.mkdir(parents=True)
    person = tmp_path / "person" / ".codex"
    (person / "sessions").mkdir(parents=True)
    (person / "auth.json").write_text("signed in\n")
    (person / "sessions" / "other-matter.jsonl").write_text("another matter's transcript\n")
    shared = tempfile.NamedTemporaryFile("w", dir="/private/tmp", prefix="deid-confine-", delete=False)
    shared.write("another program's scratch\n")
    shared.close()
    home = tmp_path / "deid" / "codex" / "case-1"
    auth = matter_codex_home(home, person)
    tools = tmp_path / "tools" / "bin"
    tools.mkdir(parents=True)
    # "Codex": runs the shell line it is given, under the profile, with the wrapper's TMPDIR
    (tools / "codex").write_text('#!/bin/sh\nexec /bin/sh -c "$1"\n')
    (tools / "codex").chmod(0o755)
    wrapper = confined_codex(matter, str(tools / "codex"), tmp_path / "w", home, auth_file=auth)

    def run(line):
        done = subprocess.run([str(wrapper), line], capture_output=True, text=True, timeout=30)
        return done.returncode, done.stdout + done.stderr

    try:
        assert run(f'cat "{home / "auth.json"}"') == (0, "signed in\n")        # its sign-in, through the link
        assert run(f'cat "{person / "sessions" / "other-matter.jsonl"}"')[0] != 0
        code, said = run(f'cat "{shared.name}"')
        assert code != 0 and "scratch" not in said
        code, said = run('echo own > "$TMPDIR/x" && cat "$TMPDIR/x" && echo "$TMPDIR"')
        assert code == 0 and said.startswith("own\n") and said.strip().endswith(str(home.resolve() / "tmp") + "/")
    finally:
        os.unlink(shared.name)


async def test_codex_gets_the_matters_own_home(tmp_path):
    from deidkit.agents.host import AgentChoice, HostConfig
    from deidkit.proxy.server import ProxyConfig
    from test_agent_host import FAKE, started

    person = tmp_path / "home" / ".codex"                    # HOME is tmp_path/home in these tests
    person.mkdir(parents=True)
    (person / "auth.json").write_text("{}")
    agents = [AgentChoice("Codex", [sys.executable, str(FAKE), "codex-acp"])]
    cfg = HostConfig(folder=tmp_path / "m", scope="case/1", title="Case 1", agent_command=[], dev_login=True,
                     codex_executable="/opt/codex/bin/codex", agents=agents, codex_confine=False)
    (tmp_path / "m").mkdir()
    host, client = await started(cfg, ProxyConfig(default_scope="case/1"))
    try:
        home = tmp_path / "home" / ".deid" / "agent" / "codex" / "case_1"
        assert host.agent.env["CODEX_HOME"] == str(home)
        assert (home / "auth.json").is_symlink() and os.path.realpath(home / "auth.json") == real(person / "auth.json")
    finally:
        await client.close()


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
