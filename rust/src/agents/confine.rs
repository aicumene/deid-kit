// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Codex confined to the matter folder by the kernel (`deidkit/agents/confine.py`): on macOS it starts
//! under a Seatbelt profile that closes every home folder, every other disk, the matter folder's
//! neighbours and the shared temp folders, and opens the matter folder, Codex's home for this matter,
//! the one sign-in file and Codex's install. The reasons, measured, are in the Python module.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use crate::{log, py};

pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
const TEMP: [&str; 2] = ["/private/tmp", "/private/var/folders"];
const TOO_WIDE: [&str; 9] = ["/", "/private", "/private/var", "/private/var/folders", "/Users", "/Volumes", "/tmp",
                             "/private/tmp", "/var"];

/// Codex cannot be confined here.
#[derive(Debug)]
pub struct ConfinementUnavailable(pub String);

pub fn supported() -> bool {
    cfg!(target_os = "macos") && Path::new(SANDBOX_EXEC).exists()
}

/// `os.path.realpath(os.path.expanduser(path))` as a string.
pub fn real(path: &str) -> String {
    py::realpath(&py::expanduser(path)).to_string_lossy().into_owned()
}

/// `os.path.dirname`.
pub fn dirname(p: &str) -> String {
    match p.rfind('/') {
        None => String::new(),
        Some(i) => {
            let head = &p[..=i];
            let trimmed = head.trim_end_matches('/');
            if trimmed.is_empty() { head.to_string() } else { trimmed.to_string() }
        }
    }
}

fn q(path: &str) -> String {
    format!("\"{}\"", path.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `shlex.quote`.
pub fn shquote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = s.chars().all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe { s.to_string() } else { format!("'{}'", s.replace('\'', "'\"'\"'")) }
}

/// The Seatbelt profile for Codex over `folder`.
pub fn profile(folder: &str, codex: &str, codex_home: &str, extra_read: &[String], auth_file: Option<&str>) -> String {
    let (matter, home, binary) = (real(folder), real(codex_home), real(codex));
    let person = real("~");
    let too_wide = |path: &str| -> bool {
        TOO_WIDE.contains(&path) || path == person || person.starts_with(&format!("{path}/")) || path.matches('/').count() < 2
    };
    let mut install = dirname(&dirname(&binary));
    if too_wide(&install) {
        install = dirname(&binary);
    }
    let codex_itself = if !too_wide(&install) { format!("(subpath {})", q(&install)) } else { format!("(literal {})", q(&binary)) };
    let parent = dirname(&matter);
    let mut closed: Vec<String> = ["/Users", "/Volumes"].iter().chain(TEMP.iter()).map(|s| s.to_string()).collect();
    if !TOO_WIDE.contains(&parent.as_str()) {
        closed.push(parent);
    }
    let mut lines = vec![
        "(version 1)".to_string(),
        "(allow default)".to_string(),
        ";; closed: every home folder, every other disk, the shared temp folders, and the matter folder's neighbours".to_string(),
        format!("(deny file-read* file-write* {})", closed.iter().map(|p| format!("(subpath {})", q(p))).collect::<Vec<_>>().join(" ")),
        ";; names and sizes may be looked up on the way to what is open; contents may not".to_string(),
        "(allow file-read-metadata)".to_string(),
        ";; open: the matter folder, and Codex's home for this matter (its settings, sessions, temp files)".to_string(),
        format!("(allow file-read* file-write* (subpath {}) (subpath {}))", q(&matter), q(&home)),
    ];
    if let Some(auth) = auth_file {
        lines.push(";; the person's sign-in, linked from it: the one file of their own Codex home".to_string());
        lines.push(format!("(allow file-read* file-write* (literal {}))", q(&real(auth))));
    }
    lines.push(";; Codex itself".to_string());
    lines.push(format!("(allow file-read* {codex_itself})"));
    let mut readable = Vec::new();
    for given in extra_read {
        let r = real(given);
        if too_wide(&r) || !Path::new(&r).exists() {
            continue;
        }
        let kind = if Path::new(&r).is_dir() { "subpath" } else { "literal" };
        readable.push(format!("(allow file-read* ({kind} {}))", q(&r)));
    }
    if !readable.is_empty() {
        lines.push(";; read only: what Codex's children need (the sub-agents' MCP server)".to_string());
        lines.extend(readable);
    }
    lines.push(String::new());
    lines.join("\n")
}

/// `shutil.which(name)`: a path to the executable, from PATH when the name has no slash.
pub fn which(name: &str) -> Option<String> {
    let executable = |p: &Path| p.is_file() && std::fs::metadata(p).is_ok_and(|m| m.permissions().mode() & 0o111 != 0);
    if name.contains('/') {
        return executable(Path::new(name)).then(|| name.to_string());
    }
    let path = std::env::var("PATH").unwrap_or_else(|_| "/bin:/usr/bin".into());
    path.split(':').map(|d| if d.is_empty() { ".".to_string() } else { d.to_string() })
        .map(|d| Path::new(&d).join(name)).find(|p| executable(p)).map(|p| p.to_string_lossy().into_owned())
}

fn write_exec(path: &Path, text: &str) -> std::io::Result<()> {
    std::fs::write(path, text)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

/// Write the profile and a wrapper that runs Codex under it into `workdir`; return the wrapper.
pub fn confined_codex(folder: &str, codex: Option<&str>, workdir: &Path, codex_home: Option<&str>, extra_read: &[String],
                      args: &[String], auth_file: Option<&str>) -> Result<PathBuf, ConfinementUnavailable> {
    if !supported() {
        return Err(ConfinementUnavailable("Codex can be confined to the matter folder on macOS only (Seatbelt); start it \
                                           with --codex-unconfined to run it as it is".into()));
    }
    let binary = codex.filter(|c| !c.is_empty()).map(str::to_string).or_else(|| which("codex"))
        .ok_or_else(|| ConfinementUnavailable("no Codex executable was found to confine: give it with --codex-executable".into()))?;
    let codex_home = codex_home.map(str::to_string).filter(|h| !h.is_empty())
        .or_else(|| std::env::var("CODEX_HOME").ok().filter(|h| !h.is_empty())).unwrap_or_else(|| "~/.codex".into());
    let io = |e: std::io::Error| ConfinementUnavailable(e.to_string());
    std::fs::create_dir_all(workdir).map_err(io)?;
    let sb = workdir.join("codex.sb");
    std::fs::write(&sb, profile(folder, &binary, &codex_home, extra_read, auth_file)).map_err(io)?;
    let temp = PathBuf::from(real(&codex_home)).join("tmp");
    std::fs::create_dir_all(&temp).map_err(io)?;
    let wrapper = workdir.join("codex");
    let extra: String = args.iter().map(|a| format!(" {}", shquote(a))).collect();
    write_exec(&wrapper, &format!(
        "#!/bin/sh\n# Codex confined to the matter folder (deid-agent, deidkit.agents.confine)\nTMPDIR={}; export TMPDIR\nexec {SANDBOX_EXEC} -f {} {}{extra} \"$@\"\n",
        shquote(&format!("{}/", temp.to_string_lossy())), shquote(&sb.to_string_lossy()), shquote(&real(&binary)))).map_err(io)?;
    Ok(wrapper)
}

/// A wrapper that starts Codex with `args` before the adapter's own, for Codex not confined.
pub fn codex_with(codex: Option<&str>, workdir: &Path, args: &[String]) -> Result<PathBuf, ConfinementUnavailable> {
    let binary = codex.filter(|c| !c.is_empty()).map(str::to_string).or_else(|| which("codex"))
        .ok_or_else(|| ConfinementUnavailable("no Codex executable was found: give it with --codex-executable".into()))?;
    let io = |e: std::io::Error| ConfinementUnavailable(e.to_string());
    std::fs::create_dir_all(workdir).map_err(io)?;
    let wrapper = workdir.join("codex");
    let extra: String = args.iter().map(|a| format!(" {}", shquote(a))).collect();
    write_exec(&wrapper, &format!("#!/bin/sh\n# Codex with settings it takes only at its start (deid-agent)\nexec {}{extra} \"$@\"\n",
                                  shquote(&real(&binary)))).map_err(io)?;
    Ok(wrapper)
}

/// Codex's home for a matter, made if new, the person's sign-in linked in; the sign-in file it links to.
pub fn matter_codex_home(home: &Path, person_home: Option<&str>) -> std::io::Result<Option<PathBuf>> {
    use std::os::unix::fs::DirBuilderExt;
    if !home.is_dir() {
        if let Some(parent) = home.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::DirBuilder::new().mode(0o700).create(home).or_else(|e| if home.is_dir() { Ok(()) } else { Err(e) })?;
    }
    let tmp = home.join("tmp");
    if !tmp.is_dir() {
        std::fs::create_dir(&tmp)?;
    }
    let person = py::expanduser(&person_home.map(str::to_string).filter(|h| !h.is_empty())
        .or_else(|| std::env::var("CODEX_HOME").ok().filter(|h| !h.is_empty())).unwrap_or_else(|| "~/.codex".into()));
    let theirs = person.join("auth.json");
    let link = home.join("auth.json");
    if !theirs.is_file() {
        return Ok(None);
    }
    let is_link = std::fs::symlink_metadata(&link).is_ok_and(|m| m.file_type().is_symlink());
    if is_link && py::realpath(&link) == py::realpath(&theirs) {
        return Ok(Some(py::realpath(&theirs)));
    }
    if link.exists() || is_link {
        if !is_link {
            log::warning(&format!("{} is a file, not the link to {}: left as it is", link.display(), theirs.display()));
            return Ok(Some(py::realpath(&link)));
        }
        std::fs::remove_file(&link)?;
    }
    std::os::unix::fs::symlink(&theirs, &link)?;
    Ok(Some(py::realpath(&theirs)))
}
