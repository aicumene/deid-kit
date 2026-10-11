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
home folder, every other disk and the matter folder's neighbours, and opens the matter folder,
Codex's own home (its sign-in, settings and sessions) and Codex's install. Codex's own sandbox still
runs inside it — Seatbelt nests — and the commands Codex runs inherit the profile. Names and sizes
may still be looked up on the way to what is open (``file-read-metadata``); contents may not.

Where there is no Seatbelt, Codex is not confined, and deid-agent does not start it unless the person
says so (``--codex-unconfined``)."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

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
            extra_read: list[str | os.PathLike] | tuple = ()) -> str:
    """The Seatbelt profile for Codex over ``folder``. ``extra_read``: what else Codex's own
    children must read, as the sub-agents' MCP server does — its interpreter, its package and its
    hosts file — opened for reading only; nothing too wide is opened."""
    matter, home, binary = _real(folder), _real(codex_home), _real(codex)
    person = _real("~")

    def too_wide(path: str) -> bool:     # would open the system root, a home folder, or a disk
        return path in _TOO_WIDE or path == person or person.startswith(path + "/") or path.count("/") < 2

    install = os.path.dirname(os.path.dirname(binary))          # its bin/ and what sits beside it
    if too_wide(install):
        install = os.path.dirname(binary)
    codex_itself = f"(subpath {_q(install)})" if not too_wide(install) else f"(literal {_q(binary)})"
    parent = os.path.dirname(matter)
    closed = ["/Users", "/Volumes"] + ([parent] if parent not in _TOO_WIDE else [])
    return "\n".join([
        "(version 1)",
        "(allow default)",
        ";; closed: every home folder, every other disk, and the matter folder's neighbours",
        "(deny file-read* file-write* " + " ".join(f"(subpath {_q(p)})" for p in closed) + ")",
        ";; names and sizes may be looked up on the way to what is open; contents may not",
        "(allow file-read-metadata)",
        ";; open: the matter folder, and Codex's own home (its sign-in, settings and sessions)",
        f"(allow file-read* file-write* (subpath {_q(matter)}) (subpath {_q(home)}))",
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
                   extra_read: list[str | os.PathLike] | tuple = (), args: list[str] | tuple = ()) -> Path:
    """Write the profile and a wrapper that runs ``codex`` under it into ``workdir``; return the
    wrapper, which the Codex adapter starts in Codex's place (``CODEX_PATH``). ``args`` go to Codex
    before the adapter's own (``--disable multi_agent``: settings Codex takes only at its start)."""
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
    sb.write_text(profile(folder, binary, codex_home, extra_read), encoding="utf-8")
    wrapper = workdir / "codex"
    extra = "".join(" " + shlex.quote(a) for a in args)
    wrapper.write_text("#!/bin/sh\n# Codex confined to the matter folder (deid-agent, deidkit.agents.confine)\n"
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


__all__ = ["ConfinementUnavailable", "codex_with", "confined_codex", "profile", "supported"]
