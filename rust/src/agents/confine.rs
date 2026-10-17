// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The agents confined to the matter folder by the kernel (`deidkit/agents/confine.py` for Codex): on macOS each
//! starts under a Seatbelt profile that closes every home folder, every other disk, the matter folder's
//! neighbours and the shared temp folders, and opens the matter folder and what the agent itself needs; its network
//! is deid-agent alone. Codex's reasons, measured, are in the Python module; Claude Code's are below.

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
/// `network` is added at the profile's end ([`network`]); none leaves Codex's network as it is.
pub fn confined_codex(folder: &str, codex: Option<&str>, workdir: &Path, codex_home: Option<&str>, extra_read: &[String],
                      args: &[String], auth_file: Option<&str>, network: &[String]) -> Result<PathBuf, ConfinementUnavailable> {
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
    std::fs::write(&sb, with_lines(profile(folder, &binary, &codex_home, extra_read, auth_file), network)).map_err(io)?;
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

// ── Claude Code, and the network ────────────────────────────────────────────────────────────────
//
// Claude Code is confined whole, as Codex is: its own tools (Read, Write, Edit, Grep) and everything it starts run
// under a profile that opens the matter folder and what Claude Code itself needs. Its shell commands go further:
// macOS starts no second sandbox inside a first (sandbox_apply: Operation not permitted, 16.10.2026), so Claude
// Code's own command sandbox cannot run there; instead Claude Code's shell is the launcher (`agent-shell`), the one
// program its profile lets start outside it, and the launcher puts each command into the stricter profile beside it
// — the matter folder and a temp folder, no Keychain, no network. Claude Code reads the person's sign-in with
// /usr/bin/security, so its own profile must allow that; the commands' profile does not. Measured with Claude
// Code 2.1.293 on macOS 26.6.

/// The hosts deid-agent opens a tunnel to for an agent it confines, on port 443: the sign-ins, which the agents
/// renew themselves (Claude Code at platform.claude.com, Codex at auth.openai.com). Every other host is refused.
pub const SIGN_IN: [&str; 2] = ["platform.claude.com", "auth.openai.com"];

/// The system log's socket: a program's own log lines (agent-guard writes there), not the network.
const SYSLOG: &str = "(allow network-outbound (remote unix-socket (path-literal \"/private/var/run/syslog\")))";

/// What a confined agent's environment says of the files it no longer reaches: git without the person's
/// ~/.gitconfig (closed to its commands; git stops at a config file it may not read), and Apple's /usr/bin/git
/// keeping the cache of where the developer tools are in the agent's own temp folder — without it each git takes a
/// second and a half, and the shared cache, written from inside, could send the person's next git elsewhere.
pub fn tools_env(temp: &str) -> Vec<(String, String)> {
    vec![("GIT_CONFIG_GLOBAL".into(), "/dev/null".into()),
         ("xcrun_db".into(), format!("{}/xcrun_db", temp.trim_end_matches('/')))]
}

/// What a confined agent's environment says of the network: every https request through deid-agent, whose
/// tunnels reach only [`SIGN_IN`]; requests to deid-agent itself go straight.
pub fn proxy_env(port: u16) -> Vec<(String, String)> {
    let proxy = format!("http://127.0.0.1:{port}");
    let direct = "127.0.0.1,localhost".to_string();
    vec![("HTTPS_PROXY".into(), proxy.clone()), ("https_proxy".into(), proxy),
         ("NO_PROXY".into(), direct.clone()), ("no_proxy".into(), direct)]
}

/// The network of a confined agent: deid-agent alone, on this Mac (`port`), and the sub-agents' model servers.
pub fn network(port: u16, servers: &[String]) -> Vec<String> {
    let mut lines = vec![
        ";; the network: deid-agent alone, on this Mac — the model, and the sign-in it lets through".to_string(),
        "(deny network-outbound)".to_string(),
        format!("(allow network-outbound (remote ip \"localhost:{port}\"))"),
        SYSLOG.to_string(),
    ];
    if !servers.is_empty() {
        lines.push(";; the sub-agents' model servers (Seatbelt names a server's port, not its address)".into());
        lines.extend(servers.iter().map(|s| format!("(allow network-outbound (remote ip {}))", q(s))));
    }
    lines
}

/// The sub-agents' servers in a hosts file, as Seatbelt can name them: `localhost:<port>` for one on this Mac,
/// `*:<port>` for one elsewhere.
pub fn server_addresses(hosts: &Path) -> Vec<String> {
    let Some(table) = std::fs::read_to_string(hosts).ok().and_then(|t| toml::from_str::<toml::Table>(&t).ok()) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for host in table.get("host").and_then(|h| h.as_array()).into_iter().flatten() {
        if let Some(a) = host.get("url").and_then(|u| u.as_str()).and_then(seatbelt_address) {
            if !out.contains(&a) {
                out.push(a);
            }
        }
    }
    out
}

/// `http://host:port/…` as Seatbelt's `remote ip`: its port, and `localhost` or `*` for its host.
pub fn seatbelt_address(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?.rsplit('@').next()?;
    let (host, port) = match authority.strip_prefix('[') {
        Some(v6) => {
            let (h, after) = v6.split_once(']')?;
            (h, after.strip_prefix(':'))
        }
        None => match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        },
    };
    let port: u16 = match port {
        Some(p) => p.parse().ok()?,
        None if scheme.eq_ignore_ascii_case("https") => 443,
        None => 80,
    };
    let local = matches!(host.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1");
    Some(format!("{}:{port}", if local { "localhost" } else { "*" }))
}

/// A profile with `lines` added at its end.
pub fn with_lines(profile: String, lines: &[String]) -> String {
    if lines.is_empty() {
        return profile;
    }
    format!("{}\n{}\n", profile.trim_end_matches('\n'), lines.join("\n"))
}

/// A string inside Seatbelt's `#"…"` regular expression, its special characters escaped.
fn rx(s: &str) -> String {
    s.chars().map(|c| if "\\.+*?()|[]{}^$".contains(c) { format!("\\{c}") } else { c.to_string() }).collect()
}

/// What a matter folder's neighbours and the person's other files are: closed to both profiles.
fn closed(matter: &str) -> String {
    let parent = dirname(matter);
    let mut closed: Vec<String> = ["/Users", "/Volumes"].iter().chain(TEMP.iter()).map(|s| s.to_string()).collect();
    if !TOO_WIDE.contains(&parent.as_str()) {
        closed.push(parent);
    }
    format!("(deny file-read* file-write* {})", closed.iter().map(|p| format!("(subpath {})", q(p))).collect::<Vec<_>>().join(" "))
}

/// Closed for writing inside the matter folder: what would run outside any sandbox later — git's settings and
/// hooks (git runs them for Claude Code), Claude Code's and Codex's settings for the folder (hooks, MCP servers).
fn kept_settings(matter: &str) -> String {
    let mut parts: Vec<String> = [".git", ".claude", ".codex"].iter()
        .map(|d| format!("(subpath {})", q(&format!("{matter}/{d}")))).collect();
    parts.push(format!("(literal {})", q(&format!("{matter}/.mcp.json"))));
    if !matter.contains('"') {
        parts.push(format!("(regex #\"^{}/(.*/)?\\.(git|claude|codex)(/|$)\")", rx(matter)));
    }
    format!("(deny file-write* {})", parts.join(" "))
}

/// Where Claude Code keeps a working folder's transcripts, under ~/.claude/projects: the path with every character
/// but an ASCII letter or digit made `-` (one for each UTF-16 unit, as JavaScript counts); past 200 characters it
/// is cut and a hash added, so then only its start is named.
pub fn claude_projects(home: &str, folder: &str) -> String {
    let name: String = folder.chars()
        .flat_map(|c| if c.is_ascii_alphanumeric() { vec![c] } else { vec!['-'; c.len_utf16()] }).collect();
    let base = format!("{home}/.claude/projects/");
    if name.len() <= 200 || base.contains('"') {
        format!("(subpath {})", q(&format!("{base}{name}")))
    } else {
        format!("(regex #\"^{}{}\")", rx(&base), &name[..200])
    }
}

/// The profile of Claude Code's shell commands, as the launcher applies it: the matter folder and the session's
/// temp folder, nothing else of the person's, no Keychain, no network.
pub fn command_profile(folder: &str, temp: &str) -> String {
    let (matter, temp) = (real(folder), real(temp));
    let lines = [
        "(version 1)".to_string(),
        "(allow default)".to_string(),
        ";; closed: every home folder, every other disk, the shared temp folders, and the matter folder's neighbours".into(),
        closed(&matter),
        ";; names and sizes may be looked up on the way to what is open; contents may not".into(),
        "(allow file-read-metadata)".into(),
        ";; written nowhere but the matter folder and this session's temp folder".into(),
        "(deny file-write* (subpath \"/\"))".into(),
        format!("(allow file-read* file-write* (subpath {}) (subpath {}))", q(&matter), q(&temp)),
        kept_settings(&matter),
        "(allow file-write* (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/dtracehelper\") (regex #\"^/dev/tty\") (regex #\"^/dev/fd/\"))".into(),
        ";; not the person's sign-ins: the Keychain's command".into(),
        "(deny process-exec (literal \"/usr/bin/security\"))".into(),
        ";; no network".into(),
        "(deny network-outbound)".into(),
        "(deny network-bind)".into(),
        SYSLOG.into(),
        String::new(),
    ];
    lines.join("\n")
}

/// Claude Code's own profile over `folder`, for the person whose home is `home`. `session` holds the launcher
/// (`bin/zsh`, its profile beside it) and the temp folder (`tmp`); `read` and `write` are what else its own
/// children need (agent-guard, its journal, the sub-agents' server).
pub fn claude_profile(home: &str, folder: &str, claude: &str, session: &Path, network: &[String], read: &[String],
                      write: &[String]) -> String {
    let matter = real(folder);
    let home = real(home);
    let bin = real(&session.join("bin").to_string_lossy());
    let temp = real(&session.join("tmp").to_string_lossy());
    let binary = real(claude);
    let too_wide = |path: &str| -> bool {
        TOO_WIDE.contains(&path) || path == home || home.starts_with(&format!("{path}/")) || path.matches('/').count() < 2
    };
    let mut install = dirname(&dirname(&binary));
    if too_wide(&install) {
        install = dirname(&binary);
    }
    let mut itself = vec![if too_wide(&install) { format!("(literal {})", q(&binary)) } else { format!("(subpath {})", q(&install)) }];
    let given = py::expanduser(claude).to_string_lossy().into_owned();
    if given != binary && given.starts_with('/') {
        itself.push(format!("(literal {})", q(&given)));
    }
    let at = |p: &str| q(&format!("{home}/{p}"));
    let mut projects = vec![claude_projects(&home, &matter)];
    let given_folder = py::expanduser(folder).to_string_lossy().into_owned();
    if given_folder != matter {
        projects.push(claude_projects(&home, &given_folder));
    }
    let mut lines = vec![
        "(version 1)".to_string(),
        "(allow default)".to_string(),
        ";; closed: every home folder, every other disk, the shared temp folders, and the matter folder's neighbours".into(),
        closed(&matter),
        ";; names and sizes may be looked up on the way to what is open; contents may not".into(),
        "(allow file-read-metadata)".into(),
        ";; open: the matter folder, and this session's temp folder".into(),
        format!("(allow file-read* file-write* (subpath {}) (subpath {}))", q(&matter), q(&temp)),
        kept_settings(&matter),
        ";; read only: the launcher of the shell commands, and their profile".into(),
        format!("(allow file-read* (subpath {}))", q(&bin)),
        ";; Claude Code itself".into(),
        format!("(allow file-read* {})", itself.join(" ")),
        ";; read only: Claude Code's settings and the person's — a hook or an MCP server written there would run at the".into(),
        ";; person's next claude, outside any sandbox — and what macOS and git read in a home folder".into(),
        format!("(allow file-read* (literal {}) (literal {}) (literal {}) (literal {}) (literal {}) (subpath {}) (literal {}))",
                at(".claude.json"), at(".claude/settings.json"), at(".claude/CLAUDE.md"), at(".CFUserTextEncoding"),
                at(".gitconfig"), at(".config/git"), at("Library/Preferences/com.apple.security.plist")),
        ";; what Claude Code keeps of this session: its transcript, the list of sessions running, its lock".into(),
        format!("(allow file-read* file-write* {} (subpath {}) (subpath {}) (regex #\"^{}/\\.claude\\.lock\"))",
                projects.join(" "), at(".claude/sessions"), at(".claude/session-env"), rx(&home)),
        ";; the person's sign-in, in the login keychain (Claude Code reads it with /usr/bin/security)".into(),
        format!("(allow file-read* file-write* (subpath {}))", at("Library/Keychains")),
    ];
    let present = |p: &&String| !p.is_empty() && !too_wide(&real(p));
    let paths = |given: &[String], what: &str| -> Option<String> {
        let rules: Vec<String> = given.iter().filter(present).map(|p| {
            let r = real(p);
            let kind = if Path::new(&r).is_dir() { "subpath" } else { "literal" };
            format!("({kind} {})", q(&r))
        }).collect();
        (!rules.is_empty()).then(|| format!("(allow {what} {})", rules.join(" ")))
    };
    if let Some(r) = paths(read, "file-read*") {
        lines.push(";; read only: what Claude Code's own children need (agent-guard, the sub-agents' server)".into());
        lines.push(r);
    }
    // a journal, and the files beside it named after it (agent-guard keeps its resource and repeat state there)
    let journals: Vec<String> = write.iter().filter(present).filter(|p| !Path::new(&real(p)).is_dir()).filter_map(|p| {
        let r = real(p);
        let (dir, name) = (dirname(&r), r.rsplit('/').next().unwrap_or("").to_string());
        let stem = name.split('.').next().unwrap_or("");
        (!stem.is_empty() && !r.contains('"')).then(|| format!("(regex #\"^{}/{}\\.[^/]*$\")", rx(&dir), rx(stem)))
    }).collect();
    if let Some(w) = paths(write, "file-read* file-write*") {
        lines.push(";; written: agent-guard's journal of this matter, and the files beside it named after it".into());
        lines.push(w);
        if !journals.is_empty() {
            lines.push(format!("(allow file-read* file-write* {})", journals.join(" ")));
        }
    }
    lines.push(";; the shell commands start through the launcher, which puts them into the stricter profile beside it".into());
    lines.push(format!("(allow process-exec (literal {}) (with no-sandbox))", q(&format!("{bin}/zsh"))));
    lines.extend(network.iter().cloned());
    lines.push(String::new());
    lines.join("\n")
}

/// A confined Claude Code: the program its adapter starts, the shell it runs commands with, its temp folder, and
/// the profile it runs under.
#[derive(Debug)]
pub struct ConfinedClaude {
    pub executable: PathBuf,
    pub shell: PathBuf,
    pub temp: PathBuf,
    pub profile: PathBuf,
    /// Claude Code's own program, the wrapper's (to read the sign-in outside the profile, for comparison).
    pub binary: PathBuf,
}

/// Take macOS's quarantine off a program deid-agent copied out of the app (16.10.2026): copied out of an app that
/// came in a downloaded disk image, it carries the mark without the person's approval of the app, and macOS kills
/// it at its start (code 137) — every shell command would fail.
pub fn unquarantine(path: &Path) {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        if let (Ok(p), Ok(n)) = (std::ffi::CString::new(path.as_os_str().as_bytes()), std::ffi::CString::new("com.apple.quarantine")) {
            unsafe { libc::removexattr(p.as_ptr(), n.as_ptr(), libc::XATTR_NOFOLLOW) };
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = path;
}

/// The folders Claude Code keeps a session's files in, made before it starts: inside its profile it may write in
/// them but not make them, and a person who signed in and never ran a session has none yet.
pub fn claude_folders(home: &str) -> std::io::Result<()> {
    let home = PathBuf::from(real(home));
    for d in [".claude/projects", ".claude/sessions", ".claude/session-env"] {
        std::fs::create_dir_all(home.join(d))?;
    }
    Ok(())
}

/// Claude Code tried under its profile before it starts, with the environment it will have: it starts there, it
/// reads the person's sign-in there (when it reads it outside), and its shell commands start through the launcher.
/// Err says what failed, in a few words. What a newer Claude Code, an older macOS, or a Claude Code installed where
/// its profile does not reach (under the home folder, as with nvm) would break is found here, not in the first task.
pub async fn check_claude(c: &ConfinedClaude, folder: &Path, env: &[(String, String)], sign_in: bool) -> Result<(), String> {
    use tokio::process::Command;
    let run = |program: &Path, args: &[&str], sandboxed: bool| {
        let mut cmd = if sandboxed {
            let mut c2 = Command::new(SANDBOX_EXEC);
            c2.arg("-f").arg(&c.profile).arg(program);
            c2
        } else {
            Command::new(program)
        };
        cmd.args(args).current_dir(folder).env_clear().envs(env.iter().map(|(k, v)| (k, v)))
            .stdin(std::process::Stdio::null()).kill_on_drop(true);
        async move {
            match tokio::time::timeout(std::time::Duration::from_secs(20), cmd.output()).await {
                Ok(Ok(out)) => Ok(out),
                Ok(Err(e)) => Err(e.to_string()),
                Err(_) => Err("no answer in 20 s".into()),
            }
        }
    };
    let first_line = |b: &[u8]| String::from_utf8_lossy(b).lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
    let started = run(&c.executable, &["--version"], false).await.map_err(|e| format!("Claude Code did not start in it: {e}"))?;
    if !started.status.success() {
        return Err(format!("Claude Code did not start in it ({}): {}", started.status, first_line(&started.stderr)));
    }
    let shell = run(&c.shell, &["-c", "exit 0"], true).await.map_err(|e| format!("its shell commands could not start: {e}"))?;
    if !shell.status.success() {
        return Err(format!("its shell commands could not start ({}): {}", shell.status, first_line(&shell.stderr)));
    }
    if sign_in {
        let signed_in = |out: &std::process::Output| String::from_utf8_lossy(&out.stdout).contains("\"loggedIn\": true");
        let inside = run(&c.executable, &["auth", "status"], false).await.map_err(|e| format!("Claude Code's sign-in could not be read in it: {e}"))?;
        if !signed_in(&inside) {
            if run(&c.binary, &["auth", "status"], false).await.is_ok_and(|o| signed_in(&o)) {
                return Err("inside it Claude Code could not read the person's sign-in".into());
            }
        }
    }
    Ok(())
}


/// Lay out a confined Claude Code in `session`: the launcher copied in as `bin/zsh` with the commands' profile
/// beside it, Claude Code's own profile, and the wrapper that starts Claude Code under it.
pub fn confined_claude(home: &str, folder: &str, claude: Option<&str>, launcher: &Path, session: &Path, network: &[String],
                       read: &[String], write: &[String]) -> Result<ConfinedClaude, ConfinementUnavailable> {
    if !supported() {
        return Err(ConfinementUnavailable("Claude Code can be confined to the matter folder on macOS only (Seatbelt)".into()));
    }
    let binary = claude.filter(|c| !c.is_empty()).map(str::to_string).or_else(|| which("claude"))
        .ok_or_else(|| ConfinementUnavailable("no Claude Code executable was found to confine: give it with --claude-executable".into()))?;
    let io = |e: std::io::Error| ConfinementUnavailable(e.to_string());
    let (bin, temp) = (session.join("bin"), session.join("tmp"));
    std::fs::create_dir_all(&bin).map_err(io)?;
    std::fs::create_dir_all(&temp).map_err(io)?;
    let shell = bin.join("zsh");
    std::fs::copy(launcher, &shell).map_err(io)?;
    unquarantine(&shell);
    std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o555)).map_err(io)?;
    claude_folders(home).map_err(io)?;
    std::fs::write(bin.join("command.sb"), command_profile(folder, &temp.to_string_lossy())).map_err(io)?;
    let sb = session.join("claude.sb");
    std::fs::write(&sb, claude_profile(home, folder, &binary, session, network, read, write)).map_err(io)?;
    let wrapper = session.join("claude");
    write_exec(&wrapper, &format!(
        "#!/bin/sh\n# Claude Code confined to the matter folder (deid-agent); its shell commands start through bin/zsh\nTMPDIR={}; export TMPDIR\nexec {SANDBOX_EXEC} -f {} {} \"$@\"\n",
        shquote(&format!("{}/", real(&temp.to_string_lossy()))), shquote(&sb.to_string_lossy()), shquote(&real(&binary)))).map_err(io)?;
    Ok(ConfinedClaude { executable: wrapper, shell, temp, profile: sb, binary: PathBuf::from(real(&binary)) })
}
