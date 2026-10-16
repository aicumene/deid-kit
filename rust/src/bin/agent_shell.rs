// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The launcher of a confined Claude Code's shell commands (deid-agent, `agents::confine`). Claude Code's own
//! profile lets it start this one program outside that profile, and this puts the command at once into the
//! stricter profile beside it — `command.sb` in its own folder, which the agent cannot write — then runs the shell
//! its own name says: zsh or bash. It takes nothing else from its caller: no option, no profile, no shell named.
//! Started from inside a sandbox (a copy elsewhere, or itself again from a command), it cannot start the second
//! one, and the command does not run.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

fn fail(why: &str) -> ! {
    eprintln!("agent-shell: {why}");
    std::process::exit(126);
}

fn main() {
    let me: PathBuf = std::env::current_exe().and_then(|e| e.canonicalize())
        .unwrap_or_else(|e| fail(&format!("cannot tell where it is: {e}")));
    let shell = match me.file_name().and_then(|n| n.to_str()) {
        Some("zsh") => "/bin/zsh",
        Some("bash") => "/bin/bash",
        _ => fail("it runs as zsh or bash, in the folder deid-agent made for it"),
    };
    let profile = me.with_file_name("command.sb");
    if !profile.is_file() {
        fail(&format!("no profile beside it: {}", profile.display()));
    }
    let err = std::process::Command::new(SANDBOX_EXEC).arg("-f").arg(&profile).arg(shell)
        .args(std::env::args_os().skip(1)).exec();
    fail(&format!("{SANDBOX_EXEC}: {err}"));
}
