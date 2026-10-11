# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Codex confined to the matter folder: what it may read and write is decided by the kernel, not by
Codex or by the model.

Codex's own sandbox keeps the commands it runs from writing outside the workspace and from the
network, but lets them read the whole disk: a fee note in the next folder, another client's matter,
the person's mail — and whatever they read goes to the model. Measured 09.10.2026 in the MatterAgent
evaluation (``--agent codex``): asked for "the fee in ../shared/fee-note.md", Codex read it and the
fee reached OpenAI.

So on macOS deid-agent starts Codex under a Seatbelt profile (``sandbox-exec``) that closes every
home folder, every other disk, the matter folder's neighbours and the temp folders every program of
the person shares (``/tmp``, ``/private/var/folders``: probed 11.10.2026, they held the scratch of
other sessions and other programs' files, readable), and opens the matter folder, Codex's home for
this matter, the one sign-in file and Codex's install. The commands Codex runs inherit the profile;
their temp files go to a folder of Codex's home (``TMPDIR``). Codex's own sandbox does not start
inside it: macOS applies no second Seatbelt profile within one that denies anything (probed
11.10.2026 on macOS 26.6.2), so a command fails there and Codex asks the person to run it outside
its sandbox, under this profile alone. Names and sizes may still be looked up on the way to what is
open (``file-read-metadata``); contents may not.

Codex's home is the matter's own (deid-agent gives each matter one): the person's ``~/.codex`` holds
the transcripts of every Codex session on the machine — other matters, other work — and its settings
bring the person's own MCP servers into the matter. The home of a matter holds this matter's sessions
only; the sign-in is the person's, linked (``auth.json``), so that a refreshed token reaches the
person's own file and the profile opens that one file of ``~/.codex``, nothing else.

Where there is no Seatbelt, Codex is not confined, and deid-agent does not start it unless the person
says so (``--codex-unconfined``)."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import sys
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
log = logging.getLogger("deidkit.agents.confine")

#: The temp folders every program of the person shares: closed (Codex and its commands get their own).
_TEMP = ("/private/tmp", "/private/var/folders")
#: Roots that are never closed as "the matter folder's neighbours": closing them would close the
#: system Codex runs on.
_TOO_WIDE = {"/", "/private", "/private/var", "/private/var/folders", "/Users", "/Volumes", "/tmp",
             "/private/tmp", "/var"}


class ConfinementUnavailable(RuntimeError):
    """Codex cannot be confined here."""


def supported() -> bool:
    return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)


def _real(path: str | os.PathLike) -> str:
    # Seatbelt compares real paths: /tmp is /private/tmp, /var is /private/var.
    return os.path.realpath(os.path.expanduser(str(path)))


def _q(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(folder: str | os.PathLike, codex: str | os.PathLike, codex_home: str | os.PathLike,
            extra_read: list[str | os.PathLike] | tuple = (), auth_file: str | os.PathLike | None = None) -> str:
    """The Seatbelt profile for Codex over ``folder``. ``codex_home``: Codex's home for this matter;
    ``auth_file``: the person's sign-in file it links to, the one file opened outside it. ``extra_read``:
    what else Codex's own children must read, as the sub-agents' MCP server does — its interpreter, its
    package and its hosts file — opened for reading only; nothing too wide is opened."""
    matter, home, binary = _real(folder), _real(codex_home), _real(codex)
    person = _real("~")

    def too_wide(path: str) -> bool:     # would open the system root, a home folder, or a disk
        return path in _TOO_WIDE or path == person or person.startswith(path + "/") or path.count("/") < 2

    install = os.path.dirname(os.path.dirname(binary))          # its bin/ and what sits beside it
    if too_wide(install):
        install = os.path.dirname(binary)
    codex_itself = f"(subpath {_q(install)})" if not too_wide(install) else f"(literal {_q(binary)})"
    parent = os.path.dirname(matter)
    closed = ["/Users", "/Volumes", *_TEMP] + ([parent] if parent not in _TOO_WIDE else [])
    sign_in = [f";; the person's sign-in, linked from it: the one file of their own Codex home",
               f"(allow file-read* file-write* (literal {_q(_real(auth_file))}))"] if auth_file else []
    return "\n".join([
        "(version 1)",
        "(allow default)",
        ";; closed: every home folder, every other disk, the shared temp folders, and the matter folder's neighbours",
        "(deny file-read* file-write* " + " ".join(f"(subpath {_q(p)})" for p in closed) + ")",
        ";; names and sizes may be looked up on the way to what is open; contents may not",
        "(allow file-read-metadata)",
        ";; open: the matter folder, and Codex's home for this matter (its settings, sessions, temp files)",
        f"(allow file-read* file-write* (subpath {_q(matter)}) (subpath {_q(home)}))",
        *sign_in,
        ";; Codex itself",
        f"(allow file-read* {codex_itself})",
        *_readable(extra_read, too_wide),
        ""])


def _readable(paths, too_wide) -> list[str]:
    lines = []
    for given in paths:
        real = _real(given)
        if too_wide(real) or not os.path.exists(real):
            continue
        kind = "subpath" if os.path.isdir(real) else "literal"
        lines.append(f"(allow file-read* ({kind} {_q(real)}))")
    return [";; read only: what Codex's children need (the sub-agents' MCP server)", *lines] if lines else []


def confined_codex(folder: str | os.PathLike, codex: str | None, workdir: str | os.PathLike,
                   codex_home: str | os.PathLike | None = None,
                   extra_read: list[str | os.PathLike] | tuple = (), args: list[str] | tuple = (),
                   auth_file: str | os.PathLike | None = None) -> Path:
    """Write the profile and a wrapper that runs ``codex`` under it into ``workdir``; return the
    wrapper, which the Codex adapter starts in Codex's place (``CODEX_PATH``). ``args`` go to Codex
    before the adapter's own (``--disable multi_agent``: settings Codex takes only at its start).
    Codex and its commands keep their temp files in ``tmp`` of its home (``TMPDIR``): the shared ones
    are closed."""
    if not supported():
        raise ConfinementUnavailable("Codex can be confined to the matter folder on macOS only "
                                     "(Seatbelt); start it with --codex-unconfined to run it as it is")
    binary = codex or shutil.which("codex")
    if not binary:
        raise ConfinementUnavailable("no Codex executable was found to confine: give it with "
                                     "--codex-executable")
    codex_home = codex_home or os.environ.get("CODEX_HOME") or "~/.codex"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sb = workdir / "codex.sb"
    sb.write_text(profile(folder, binary, codex_home, extra_read, auth_file), encoding="utf-8")
    temp = Path(_real(codex_home)) / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    wrapper = workdir / "codex"
    extra = "".join(" " + shlex.quote(a) for a in args)
    wrapper.write_text("#!/bin/sh\n# Codex confined to the matter folder (deid-agent, deidkit.agents.confine)\n"
                       f"TMPDIR={shlex.quote(str(temp) + '/')}; export TMPDIR\n"
                       f'exec {SANDBOX_EXEC} -f {shlex.quote(str(sb))} {shlex.quote(_real(binary))}{extra} "$@"\n',
                       encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def codex_with(codex: str | None, workdir: str | os.PathLike, args: list[str] | tuple) -> Path:
    """A wrapper that starts ``codex`` with ``args`` before the adapter's own, for Codex not confined."""
    binary = codex or shutil.which("codex")
    if not binary:
        raise ConfinementUnavailable("no Codex executable was found: give it with --codex-executable")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    wrapper = workdir / "codex"
    wrapper.write_text("#!/bin/sh\n# Codex with settings it takes only at its start (deid-agent)\n"
                       f"exec {shlex.quote(_real(binary))}" + "".join(" " + shlex.quote(a) for a in args)
                       + ' "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def matter_codex_home(home: str | os.PathLike, person_home: str | os.PathLike | None = None) -> Path:
    """Codex's home for a matter (``home``), made if new: its own sessions, settings and temp files;
    the person's sign-in linked in (``auth.json`` → theirs), so that the person signs in once and a
    refreshed token reaches their own file. Returns the person's sign-in file it links to (None when
    they have none: Codex then says it is not signed in)."""
    home = Path(home).expanduser()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    (home / "tmp").mkdir(exist_ok=True)
    person = Path(person_home or os.environ.get("CODEX_HOME") or "~/.codex").expanduser()
    theirs = person / "auth.json"
    link = home / "auth.json"
    if not theirs.is_file():
        return None
    if link.is_symlink() and os.path.realpath(link) == os.path.realpath(theirs):
        return Path(os.path.realpath(theirs))
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            # Codex wrote a file of its own here: the person's own file may be the staler one — keep it.
            log.warning("%s is a file, not the link to %s: left as it is", link, theirs)
            return Path(os.path.realpath(link))
        link.unlink()
    link.symlink_to(theirs)
    return Path(os.path.realpath(theirs))


__all__ = ["ConfinementUnavailable", "codex_with", "confined_codex", "matter_codex_home", "profile", "supported"]
