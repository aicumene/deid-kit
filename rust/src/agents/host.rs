// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! `deid-agent`: a coding agent over one folder, behind deid-proxy, with a page to drive it
//! (`deidkit/agents/host.py`, whose docstrings say why each part is as it is).
//!
//! One process on one loopback port holds the proxy (`/v1/…`, the agent's only endpoint), the agent
//! (a child over ACP, the folder its working directory) and the page with its API (`/` and
//! `/deid-agent/api/…`, behind a token the page holds in its URL fragment). The agent never holds the
//! organization's key; the proxy lends it a stand-in. Agents switch without losing their sessions; one
//! told what it has not seen gets a brief made from the record the person sees.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use crate::agents::acp::{self, AcpAgent, AcpError};
use crate::agents::confine;
use crate::agents::pricing::{self, RequestUsage};
use crate::agents::review;
use crate::http::{Request, Response};
use crate::proxy::server::Proxy;
use crate::pyre::{self, Regex};
use crate::{log, py, pyjson};

const SKIP_DIRS: [&str; 7] = [".git", ".claude", ".codex", ".legix", "node_modules", "__pycache__", ".venv"];
const MAX_FILE: u64 = 2 * 1024 * 1024;
const WARM_SUFFIXES: [&str; 7] = [".md", ".txt", ".csv", ".json", ".html", ".xml", ".eml"];
const WARM_MAX_BYTES: u64 = 512 * 1024;
const WARM_MAX_FILES: usize = 200;
const PAGE: &str = include_str!("../../../src/deidkit/agents/page.html");

/// The sub-agents' kinds (`deidkit.proxy.anthropic.SUBAGENT_KINDS`).
pub fn subagent_kinds() -> Vec<String> {
    crate::proxy::anthropic::SUBAGENT_KINDS.iter().map(|k| k.to_string()).collect()
}

// ── the agents ────────────────────────────────────────────────────────────────────────────────────

/// An agent the person can pick on the page (`--agent`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentChoice {
    pub name: String,
    pub command: Vec<String>,
    pub upstream: Option<String>,
    pub unavailable: Option<String>,
    pub meta: Option<Value>,
}

fn base_name(part: &str) -> &str {
    part.trim_end_matches('/').rsplit('/').next().unwrap_or("")
}

impl AgentChoice {
    pub fn new(name: &str, command: Vec<String>) -> AgentChoice {
        AgentChoice { name: name.into(), command, ..Default::default() }
    }

    pub fn codex(&self) -> bool {
        self.command.iter().any(|p| p.contains("codex-acp"))
    }

    pub fn claude(&self) -> bool {
        self.command.iter().any(|p| p.contains("claude-agent-acp") || base_name(p) == "claude-acp")
    }

    pub fn label(&self) -> String {
        if !self.name.is_empty() {
            return self.name.clone();
        }
        (if self.claude() { "Claude Code" } else if self.codex() { "Codex" } else if self.matteragent() { "MatterAgent" } else { "Agent" }).into()
    }

    pub fn matteragent(&self) -> bool {
        self.command.iter().any(|p| base_name(p) == "matteragent" || p == "matteragent")
            || self.command.windows(2).any(|w| w[0] == "-m" && w[1] == "matteragent")
    }

    pub fn responses(&self) -> bool {
        let wire = self.command.iter().any(|a| a == "--wire=responses")
            || self.command.windows(2).any(|w| w[0] == "--wire" && w[1] == "responses");
        self.codex() || wire
    }
}

pub struct HostConfig {
    pub folder: PathBuf,
    pub scope: String,
    pub title: String,
    pub agent_command: Vec<String>,
    pub api_key: Option<String>,
    pub dev_login: bool,
    pub claude_executable: Option<String>,
    pub codex_executable: Option<String>,
    pub codex_confine: bool,
    pub codex_home: Option<PathBuf>,
    pub port: u16,
    pub token: String,
    pub agent_key: String,
    pub usage_log: Option<PathBuf>,
    pub agents: Vec<AgentChoice>,
    pub style: Option<PathBuf>,
    pub session_meta: Option<Value>,
    pub subagents: Option<PathBuf>,
    pub subagents_command: Vec<String>,
    pub subagents_timeout: f64,
    pub claude_confine: bool,
    /// The launcher of a confined Claude Code's shell commands (`agent-shell`, beside deid-agent): with it Claude
    /// Code is confined whole; without it only its shell commands are, in its own sandbox.
    pub claude_shell: Option<PathBuf>,
    pub guard: Vec<String>,
    pub guard_log: Option<PathBuf>,
    pub person: String,
}

impl HostConfig {
    pub fn new(folder: PathBuf, scope: &str, title: &str, agent_command: Vec<String>) -> HostConfig {
        HostConfig { folder, scope: scope.into(), title: title.into(), agent_command, api_key: None, dev_login: false,
                     claude_executable: None, codex_executable: None, codex_confine: true, codex_home: None, port: 8790,
                     token: py::token_urlsafe(24), agent_key: format!("deid-agent-{}", py::token_urlsafe(24)),
                     usage_log: None, agents: Vec::new(), style: None, session_meta: None, subagents: None,
                     subagents_command: Vec::new(), subagents_timeout: 900.0, claude_confine: true, claude_shell: None, guard: Vec::new(),
                     guard_log: None,
                     person: std::env::var("USER").ok().filter(|u| !u.is_empty()).unwrap_or_else(|| "person".into()) }
    }
}

const USAGE_FIELDS: [&str; 6] = ["inputTokens", "cachedReadTokens", "cachedWriteTokens", "outputTokens", "thoughtTokens", "totalTokens"];

/// The token counts of a finished turn: `usage`, or `_meta.usage`; counts only.
pub fn turn_usage(result: &Value) -> Map<String, Value> {
    let empty = Map::new();
    let r = result.as_object().unwrap_or(&empty);
    let meta = r.get("_meta").and_then(Value::as_object).unwrap_or(&empty);
    let given = match r.get("usage") {
        Some(Value::Object(u)) => Some(u),
        _ => meta.get("usage").and_then(Value::as_object),
    };
    let Some(given) = given else { return Map::new() };
    let mut out = Map::new();
    for k in USAGE_FIELDS {
        if let Some(Value::Number(n)) = given.get(k) {
            if n.as_i64().is_some_and(|v| v >= 0) || n.is_u64() {
                out.insert(k.into(), Value::Number(n.clone()));
            }
        }
    }
    out
}

/// One JSON line appended to a file made owner-only from its first byte.
pub fn append_line(path: &Path, obj: &Value) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(path)?;
    f.write_all((py::dumps(obj) + "\n").as_bytes())
}

/// The models a session lets the person pick from, in either shape ACP has used; None — no choice.
pub fn model_choice(setup: &Value) -> Option<Value> {
    let truthy = |v: Option<&Value>| py::truthy(v);
    let get = |v: &Value, k: &str| v.get(k).cloned().unwrap_or(Value::Null);
    let options_list = match setup.get("configOptions") {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    for option in &options_list {
        if option.get("id") != Some(&json!("model")) && option.get("category") != Some(&json!("model")) {
            continue;
        }
        if option.get("type").unwrap_or(&json!("select")) != &json!("select") {
            continue;
        }
        let mut options = Vec::new();
        for entry in option.get("options").and_then(Value::as_array).into_iter().flatten() {
            let items = match entry.get("options") {
                Some(Value::Array(a)) => a.clone(),
                _ => vec![entry.clone()],
            };
            for item in items {
                if truthy(item.get("value")) {
                    let value = item["value"].clone();
                    let name = if truthy(item.get("name")) { item["name"].clone() } else { value.clone() };
                    let description = if truthy(item.get("description")) { item["description"].clone() } else { json!("") };
                    options.push(json!({"value": value, "name": name, "description": description}));
                }
            }
        }
        return Some(json!({"via": "config", "id": get(option, "id"), "current": get(option, "currentValue"), "options": options}));
    }
    let models = setup.get("models").filter(|m| truthy(Some(m))).cloned().unwrap_or(json!({}));
    if truthy(models.get("availableModels")) {
        let mut options = Vec::new();
        for m in models["availableModels"].as_array().into_iter().flatten() {
            if truthy(m.get("modelId")) {
                let id = m["modelId"].clone();
                options.push(json!({"value": id, "name": if truthy(m.get("name")) { m["name"].clone() } else { id.clone() },
                                    "description": if truthy(m.get("description")) { m["description"].clone() } else { json!("") }}));
            }
        }
        return Some(json!({"via": "models", "id": null, "current": get(&models, "currentModelId"), "options": options}));
    }
    None
}

fn walk(folder: &Path, dir: &Path, skip: &dyn Fn(&str) -> bool, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        let ft = e.file_type();
        let is_link = ft.as_ref().is_ok_and(|t| t.is_symlink());
        let is_dir = if is_link { p.is_dir() } else { ft.is_ok_and(|t| t.is_dir()) };
        if is_dir {
            if !is_link && !skip(&name) {
                walk(folder, &p, skip, out);
            }
        } else {
            out.push(p);
        }
    }
}

/// `{relative path: sha256}` for the folder's files, hidden tool folders left out.
pub fn snapshot(folder: &Path) -> BTreeMap<String, String> {
    let mut files = Vec::new();
    walk(folder, folder, &|d| SKIP_DIRS.contains(&d), &mut files);
    let mut out = BTreeMap::new();
    for p in files {
        let Ok(meta) = std::fs::metadata(&p) else { continue };
        if meta.len() <= MAX_FILE {
            if let Ok(data) = std::fs::read(&p) {
                let rel = p.strip_prefix(folder).unwrap_or(&p).to_string_lossy().into_owned();
                out.insert(rel, hex::encode(Sha256::digest(&data)));
            }
        }
    }
    out
}

/// The settings files Claude Code may read for an agent in `folder`.
pub fn settings_files(folder: &Path) -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("/Library/Application Support/ClaudeCode/managed-settings.json"),
                       PathBuf::from("/etc/claude-code/managed-settings.json"), py::expanduser("~").join(".claude").join("settings.json")];
    for d in folder.ancestors() {
        out.push(d.join(".claude").join("settings.json"));
        out.push(d.join(".claude").join("settings.local.json"));
    }
    let mut seen = Vec::new();
    out.retain(|p| if seen.contains(p) { false } else { seen.push(p.clone()); true });
    out
}

fn reroute() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"ANTHROPIC_\w*BASE_URL|CLAUDE_CODE_USE_\w+|HTTPS?_PROXY|ALL_PROXY", pyre::I)
}

/// `"<file>: <VARIABLE>"` for each setting that would send the agent around the proxy.
pub fn rerouting_settings(folder: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for path in settings_files(folder) {
        let Ok(Some(text)) = py::read_text_strict(&path) else { continue };
        let Some(Value::Object(data)) = pyjson::loads(&text) else { continue };
        let env = data.get("env").filter(|e| py::truthy(Some(e))).cloned().unwrap_or(json!({}));
        let names: Option<Vec<String>> = match &env {
            Value::Object(m) => Some(m.keys().cloned().collect()),
            Value::String(s) => Some(s.chars().map(String::from).collect()),
            Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_string)).collect(),
            _ => None,
        };
        let Some(names) = names else { continue };
        let display = path.to_string_lossy();
        for name in names {
            if reroute().fullmatch(&name.chars().collect::<Vec<_>>()).is_some() {
                found.push(format!("{display}: {name}"));
            }
        }
    }
    found
}

/// `os.path.realpath(path) == folder or under it`; a relative path is taken in the folder.
pub fn inside(folder: &Path, path: &str) -> bool {
    if path.contains('\0') {
        return false;
    }
    let target = PathBuf::from(path);
    let target = if target.is_absolute() { target } else { folder.join(target) };
    let (real, root) = (py::realpath(&target), py::realpath(folder));
    real == root || real.to_string_lossy().starts_with(&format!("{}/", root.to_string_lossy()))
}

const DEVICES: [&str; 5] = ["/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/tty"];
const PROGRAM_DIRS: [&str; 6] = ["/usr/", "/bin/", "/sbin/", "/opt/homebrew/", "/System/", "/Library/Developer/"];

/// `shlex.split(text)` (POSIX mode, no comments); Err where Python raises ValueError.
pub fn shlex_split(text: &str) -> Result<Vec<String>, ()> {
    #[derive(PartialEq, Clone, Copy)]
    enum St { Ws, Word, Quote(char), Esc(char) }
    let mut out = Vec::new();
    let mut st = St::Ws;
    let mut token = String::new();
    let mut quoted = false;
    let mut escaped_from = St::Word;
    let ws = |c: char| " \t\r\n".contains(c);
    for c in text.chars() {
        match st {
            St::Ws => {
                if ws(c) {
                    if !token.is_empty() || quoted {
                        out.push(std::mem::take(&mut token));
                        quoted = false;
                    }
                } else if c == '\\' {
                    escaped_from = St::Word;
                    st = St::Esc('\\');
                } else if c == '\'' || c == '"' {
                    st = St::Quote(c);
                } else {
                    token.push(c);
                    st = St::Word;
                }
            }
            St::Quote(q) => {
                quoted = true;
                if c == q {
                    st = St::Word;
                } else if c == '\\' && q == '"' {
                    escaped_from = st;
                    st = St::Esc('\\');
                } else {
                    token.push(c);
                }
            }
            St::Esc(e) => {
                if let St::Quote(q) = escaped_from {
                    if c != e && c != q {
                        token.push(e);
                    }
                }
                token.push(c);
                st = escaped_from;
            }
            St::Word => {
                if ws(c) {
                    st = St::Ws;
                    if !token.is_empty() || quoted {
                        out.push(std::mem::take(&mut token));
                        quoted = false;
                    }
                } else if c == '\'' || c == '"' {
                    st = St::Quote(c);
                } else if c == '\\' {
                    escaped_from = St::Word;
                    st = St::Esc('\\');
                } else {
                    token.push(c);
                }
            }
        }
    }
    match st {
        St::Quote(_) | St::Esc(_) => return Err(()),
        _ => {}
    }
    if !token.is_empty() || quoted {
        out.push(token);
    }
    Ok(out)
}

/// PurePosixPath's spelling of a path: no empty or `.` parts, no trailing slash (`..` kept).
pub fn pure(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let root = if path.starts_with("//") && !path.starts_with("///") { "//" } else if path.starts_with('/') { "/" } else { "" };
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    let joined = parts.join("/");
    if root.is_empty() && joined.is_empty() { ".".into() } else { format!("{root}{joined}") }
}

fn pure_parent(path: &str) -> String {
    let p = pure(path);
    match p.rfind('/') {
        None => ".".into(),
        Some(i) => {
            let head = &p[..i];
            if head.is_empty() || head.chars().all(|c| c == '/') { p[..=i].to_string() } else { head.to_string() }
        }
    }
}

fn exists(p: &str) -> bool {
    !p.contains('\0') && Path::new(p).exists()
}

/// `os.path.expanduser`, `~user` too (from the password database; unchanged when there is no such user).
fn expanduser(p: &str) -> String {
    if let Some(rest) = p.strip_prefix('~') {
        let (user, tail) = rest.split_once('/').map_or((rest, ""), |(u, t)| (u, t));
        if !user.is_empty() {
            let Ok(name) = std::ffi::CString::new(user) else { return p.to_string() };
            let pw = unsafe { libc::getpwnam(name.as_ptr()) };
            if pw.is_null() {
                return p.to_string();
            }
            let dir = unsafe { std::ffi::CStr::from_ptr((*pw).pw_dir) }.to_string_lossy().trim_end_matches('/').to_string();
            return if rest.contains('/') { format!("{dir}/{tail}") } else { dir };
        }
    }
    py::expanduser(p).to_string_lossy().into_owned()
}

/// The paths a shell command names outside `folder`.
pub fn outside_paths(command: &str, folder: &Path) -> Vec<String> {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    static REDIRECT: OnceLock<Regex> = OnceLock::new();
    static OPTION: OnceLock<Regex> = OnceLock::new();
    let words = shlex_split(command).unwrap_or_else(|_| py::split_ws(command).into_iter().map(str::to_string).collect());
    let home = expanduser("~");
    let folder_s = folder.to_string_lossy().into_owned();
    let mut found = Vec::new();
    for word in words {
        let wt: Vec<char> = word.chars().collect();
        for part in pyre::lazy(&SPLIT, r"[;&|()]+", pyre::NONE).split(&wt, 0).into_iter().flatten() {
            let pt: Vec<char> = part.chars().collect();
            let p = pyre::lazy(&REDIRECT, r"^\d*[<>]+&?", pyre::NONE).sub_with(&pt, 0, |_| String::new()).0;
            let qt: Vec<char> = p.chars().collect();
            let p = pyre::lazy(&OPTION, r"^(?:--?[\w-]+|\w+)=", pyre::NONE).sub_with(&qt, 0, |_| String::new()).0;
            let p = p.replace("${HOME}", &home).replace("$HOME", &home);
            if p.is_empty() || !(p.starts_with('/') || p.starts_with('~') || p == ".." || p.starts_with("../") || p.contains("/../")) {
                continue;
            }
            let expanded = pure(&expanduser(&p));
            let target = if expanded.starts_with('/') { expanded } else { pure(&format!("{folder_s}/{expanded}")) };
            let probe = if exists(&target) { target.clone() } else { pure_parent(&target) };
            if !exists(&probe) || probe == "/" {
                continue;
            }
            let real = py::realpath(Path::new(&target)).to_string_lossy().into_owned();
            if DEVICES.contains(&target.as_str()) || DEVICES.contains(&real.as_str()) || PROGRAM_DIRS.iter().any(|d| real.starts_with(d)) {
                continue;
            }
            if !inside(folder, &target) {
                found.push(p);
            }
        }
    }
    found
}

/// The matter server's tools as Claude Code names them.
pub fn subagent_mcp_tools() -> Vec<String> {
    subagent_kinds().iter().map(|k| format!("mcp__matter__{k}")).collect()
}

pub const CODEX_SUBAGENTS_NOTE: &str = "This matter has sub-agents on the organization's own model servers, next to its folder. They are \
deferred tools on the global `tools` object of exec, named `mcp__matter__<kind>` — read, find, \
extract, calc, check, and consult when a head model is up; `ALL_TOOLS` lists those available now. \
Each takes a question and, optionally, the paths to read. Hand them the reading of the matter's \
documents: a sub-agent reads the files where they are and reports back with exact quotes, each \
checked against its file. Prefer them to reading long documents yourself.";

/// `meta.setdefault("claudeCode", {}).setdefault("options", {})`.
fn claude_options(meta: &mut Value) -> &mut Map<String, Value> {
    if !meta.is_object() {
        *meta = json!({});
    }
    let cc = meta.as_object_mut().unwrap().entry("claudeCode").or_insert_with(|| json!({}));
    if !cc.is_object() {
        *cc = json!({});
    }
    let options = cc.as_object_mut().unwrap().entry("options").or_insert_with(|| json!({}));
    if !options.is_object() {
        *options = json!({});
    }
    options.as_object_mut().unwrap()
}

fn list_of(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Array(a)) => a.clone(),
        Some(Value::String(s)) if !s.is_empty() => s.chars().map(|c| json!(c.to_string())).collect(),
        _ => Vec::new(),
    }
}

/// The session's `_meta` for Claude Code with sub-agents of the organization's own.
pub fn with_subagents(meta: Option<&Value>) -> Value {
    let mut meta = meta.filter(|m| py::truthy(Some(m))).cloned().unwrap_or(json!({}));
    let options = claude_options(&mut meta);
    let mut disallowed = list_of(options.get("disallowedTools"));
    if !disallowed.contains(&json!("Task")) {
        disallowed.push(json!("Task"));
    }
    options.insert("disallowedTools".into(), json!(disallowed));
    let mut allowed = list_of(options.get("allowedTools"));
    for t in subagent_mcp_tools() {
        if !allowed.contains(&json!(t)) {
            allowed.push(json!(t));
        }
    }
    options.insert("allowedTools".into(), json!(allowed));
    meta
}

pub const PEOPLES_FILES: [&str; 6] = ["/Users", "/Volumes", "/home", "/root", "/mnt", "/media"];
pub const GUARD_EVENTS: [&str; 4] = ["UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure"];

/// Claude Code's settings for a matter: its sandbox, and agent-guard as its hooks.
pub fn claude_settings(folder: &Path, confine: bool, guard: &[String], guard_log: Option<&Path>) -> Map<String, Value> {
    let mut settings = Map::new();
    if confine {
        settings.insert("sandbox".into(), json!({
            "enabled": true, "failIfUnavailable": true, "allowUnsandboxedCommands": false, "autoAllowBashIfSandboxed": false,
            "filesystem": {"denyRead": PEOPLES_FILES, "allowRead": [folder.to_string_lossy()]},
            "network": {"allowedDomains": [], "strictAllowlist": true},
        }));
    }
    if !guard.is_empty() {
        let mut command = guard.iter().map(|p| confine::shquote(p)).chain(std::iter::once("hook".to_string()))
            .collect::<Vec<_>>().join(" ");
        if let Some(log) = guard_log {
            command = format!("AGENT_GUARD_LOG={} {command}", confine::shquote(&log.to_string_lossy()));
        }
        let mut hooks = Map::new();
        for event in GUARD_EVENTS {
            hooks.insert(event.into(), json!([{"matcher": "*", "hooks": [{"type": "command", "command": command, "timeout": 10}]}]));
        }
        settings.insert("hooks".into(), Value::Object(hooks));
    }
    settings
}

/// The session's `_meta` with the matter's settings added to those it gives; Err is Python's message.
pub fn with_claude_settings(meta: Option<&Value>, settings: &Map<String, Value>, folder: &Path) -> Result<Value, String> {
    let mut meta = meta.filter(|m| py::truthy(Some(m))).cloned().unwrap_or(json!({}));
    if settings.is_empty() {
        return Ok(meta);
    }
    let options = claude_options(&mut meta);
    let given = options.get("settings").filter(|g| py::truthy(Some(g))).cloned().unwrap_or(json!({}));
    let given = match given {
        Value::String(file) => {
            let path = folder.join(&file);
            let text = match std::fs::read(&path) {
                Ok(b) => String::from_utf8(b).map(|s| py::newlines(&s))
                    .map_err(|_| "'utf-8' codec can't decode the file".to_string())?,
                Err(e) => return Err(acp::os_error(&e, Some(&path.to_string_lossy()))),
            };
            pyjson::loads(&text).ok_or_else(|| "Expecting value: the file is not JSON".to_string())?
        }
        other => other,
    };
    let Value::Object(mut merged) = given else {
        return Err("claudeCode.options.settings is neither an object nor a file of one".into());
    };
    for (key, value) in settings {
        if key == "hooks" {
            let mut hooks = merged.get("hooks").and_then(Value::as_object).cloned().unwrap_or_default();
            for (event, entries) in value.as_object().unwrap() {
                let mut list = list_of(hooks.get(event));
                list.extend(entries.as_array().cloned().unwrap_or_default());
                hooks.insert(event.clone(), json!(list));
            }
            merged.insert("hooks".into(), Value::Object(hooks));
        } else {
            merged.insert(key.clone(), value.clone());
        }
    }
    options.insert("settings".into(), Value::Object(merged));
    Ok(meta)
}

/// The interpreter a script's first line names, when it is a Python.
fn shebang_python(program: &str) -> Option<String> {
    use std::io::Read;
    let mut buf = vec![0u8; 512];
    let n = std::fs::File::open(program).and_then(|mut f| f.read(&mut buf)).ok()?;
    let head = String::from_utf8_lossy(&buf[..n]).into_owned();
    let first = py::strip(head.split('\n').next().unwrap_or("")).to_string();
    let rest = first.strip_prefix("#!")?;
    let parts = py::split_ws(rest);
    if parts.first().is_some_and(|p| base_name(p).starts_with("python")) {
        return Some(parts[0].to_string());
    }
    if parts.len() > 1 && base_name(parts[0]) == "env" && parts[1].starts_with("python") {
        return confine::which(parts[1]);
    }
    None
}

// ── the brief ─────────────────────────────────────────────────────────────────────────────────────

pub const BRIEF_SAID: usize = 1500;
pub const BRIEF_TASK: usize = 3000;
pub const BRIEF_STEPS: usize = 40;
pub const BRIEF_CHARS: usize = 24000;

fn ended(reason: &str) -> Option<&'static str> {
    Some(match reason {
        "end_turn" => "finished",
        "cancelled" => "stopped by the person",
        "refusal" => "refused by the agent",
        "max_tokens" => "stopped: its reply ran out of room",
        "max_turn_requests" => "stopped: it ran out of steps",
        _ => return None,
    })
}

pub const CARRY_ON: &str = "The person has handed you the last task above, which {previous} was doing and did not finish. \
Carry it on from where it stopped: the documents are as {previous} left them. Read again \
whatever you need, and finish the task as the person asked it.";

fn clip_end(text: &str, n: usize) -> String {
    let t = py::strip(text);
    if py::clen(t) <= n { t.to_string() } else { format!("…{}", py::lstrip(py::last(t, n))) }
}

fn clip_start(text: &str, n: usize) -> String {
    let t = py::strip(text);
    if py::clen(t) <= n { t.to_string() } else { format!("{}…", py::rstrip(py::take(t, n))) }
}

fn cap(who: &str) -> String {
    let mut c = who.chars();
    match c.next() {
        Some(f) => crate::unicode::upper(&f.to_string()) + c.as_str(),
        None => String::new(),
    }
}

struct Run {
    agent: String,
    taken_from: Option<String>,
    items: Vec<(&'static str, String)>,
    steps: HashMap<String, (String, Value)>,
    plan: Option<Vec<Value>>,
    end: Option<Value>,
}

struct Task {
    text: Option<String>,
    runs: Vec<Run>,
}

fn s_of(e: &Value, k: &str) -> String {
    e.get(k).map(py::py_str).unwrap_or_default()
}

/// `e.get(k) or default`, as text.
fn or_text(e: &Value, k: &str, default: &str) -> String {
    match e.get(k) {
        Some(v) if py::truthy(Some(v)) => py::py_str(v),
        _ => default.to_string(),
    }
}

/// What an agent is told of work on the matter that is not in its own context; None when there is none.
pub fn brief(events: &[Value], me: &str, carrying_on: bool) -> Option<String> {
    let mut tasks: Vec<Task> = Vec::new();
    let mut perms: HashMap<String, String> = HashMap::new();
    let mut files: Option<&Value> = None;
    let mut has_run = false;
    for e in events {
        let kind = e.get("kind").and_then(Value::as_str).unwrap_or("");
        match kind {
            "files" => {
                files = Some(e);
                continue;
            }
            "user" => {
                tasks.push(Task { text: Some(s_of(e, "text")), runs: Vec::new() });
                let agent = or_text(e, "agent", "the agent");
                tasks.last_mut().unwrap().runs.push(Run { agent, taken_from: None, items: Vec::new(), steps: HashMap::new(), plan: None, end: None });
                has_run = true;
                continue;
            }
            "handover" => {
                if tasks.is_empty() {
                    tasks.push(Task { text: None, runs: Vec::new() });
                }
                let to = or_text(e, "to", "another agent");
                let from = e.get("from").filter(|v| !v.is_null()).map(py::py_str);
                tasks.last_mut().unwrap().runs.push(Run { agent: to, taken_from: from, items: Vec::new(), steps: HashMap::new(), plan: None, end: None });
                has_run = true;
                continue;
            }
            _ => {}
        }
        if !has_run {
            continue;
        }
        let run = tasks.last_mut().unwrap().runs.last_mut().unwrap();
        match kind {
            "message" => {
                let t = s_of(e, "text");
                match run.items.last_mut() {
                    Some(("said", said)) => said.push_str(&t),
                    _ => run.items.push(("said", t)),
                }
            }
            "tool" => {
                let sid = match e.get("id") {
                    Some(v) if py::truthy(Some(v)) => py::dumps(v),
                    _ => py::dumps(&json!(format!("#{}", run.steps.len()))),
                };
                let title = match e.get("title") {
                    Some(v) if py::truthy(Some(v)) => py::py_str(v),
                    _ => or_text(e, "tool", "a step"),
                };
                let status = match e.get("status") {
                    Some(v) if py::truthy(Some(v)) => v.clone(),
                    _ => json!("pending"),
                };
                run.steps.insert(sid.clone(), (title, status));
                run.items.push(("step", sid));
            }
            "tool_update" => {
                let key = py::dumps(e.get("id").unwrap_or(&Value::Null));
                if let Some(step) = run.steps.get_mut(&key) {
                    if let Some(s) = e.get("status").filter(|v| py::truthy(Some(v))) {
                        step.1 = s.clone();
                    }
                    if let Some(t) = e.get("title").filter(|v| py::truthy(Some(v))) {
                        step.0 = py::py_str(t);
                    }
                }
            }
            "plan" => run.plan = Some(e.get("entries").and_then(Value::as_array).cloned().unwrap_or_default()),
            "permission" => {
                perms.insert(s_of(e, "id"), or_text(e, "title", "an action"));
            }
            "permission_done" => {
                let id = s_of(e, "id");
                let answer = match e.get("outcome").and_then(Value::as_str) {
                    Some("allowed") => Some("allowed it"),
                    Some("refused") => Some("refused it"),
                    Some("cancelled") => Some("did not answer before it stopped"),
                    _ => None,
                };
                if let (Some(title), Some(a)) = (perms.get(&id).filter(|_| e.get("id").is_some()), answer) {
                    run.items.push(("line", format!("It asked to: {title} — the person {a}.")));
                }
            }
            "notice" if !py::truthy(e.get("handing")) => run.items.push(("line", s_of(e, "text"))),
            "error" => run.items.push(("line", format!("Error: {}", s_of(e, "text")))),
            "turn_end" => run.end = e.get("stopReason").filter(|v| !v.is_null()).cloned(),
            _ => {}
        }
    }

    let render = |task: &Task, number: usize, whole: bool| -> Vec<String> {
        let mut out = Vec::new();
        match &task.text {
            None => out.push("The task you were doing when it was handed over:".to_string()),
            Some(text) => {
                let first = task.runs.first().map_or("the agent", |r| r.agent.as_str());
                out.push(format!("Task {number} — the person asked {}: {}", if first == me { "you" } else { first },
                                 clip_start(text, BRIEF_TASK)));
            }
        }
        for (i, r) in task.runs.iter().enumerate() {
            let who = if r.agent == me { "you".to_string() } else { r.agent.clone() };
            if let Some(from) = &r.taken_from {
                let gave = if from == me { "you" } else { from.as_str() };
                out.push(if who == "you" && r.items.is_empty() { format!("  Then the person handed the task to {who}.") }
                         else { format!("  {} took the task over from {gave}.", cap(&who)) });
            }
            if whole {
                let steps = r.items.iter().filter(|x| x.0 == "step").count();
                let mut skip = steps.saturating_sub(BRIEF_STEPS);
                if skip > 0 {
                    out.push(format!("  ({skip} earlier steps not listed)"));
                }
                for (kind, value) in &r.items {
                    match *kind {
                        "said" => {
                            if !py::strip(value).is_empty() {
                                out.push(format!("  {} said: {}", cap(&who), clip_end(value, BRIEF_SAID)));
                            }
                        }
                        "step" => {
                            if skip > 0 {
                                skip -= 1;
                                continue;
                            }
                            let (title, status) = &r.steps[value];
                            let shown = match status.as_str() {
                                Some("completed") => "done",
                                Some("failed") => "failed",
                                _ => "not finished",
                            };
                            out.push(format!("  · {title} — {shown}"));
                        }
                        _ => out.push(format!("  {value}")),
                    }
                }
                if let Some(plan) = r.plan.as_ref().filter(|p| !p.is_empty()) {
                    let entries: Vec<String> = plan.iter().map(|p| {
                        let mark = if p.get("status") == Some(&json!("completed")) { "✓ " } else { "• " };
                        format!("{mark}{}", p.get("content").map_or(String::new(), py::py_str))
                    }).collect();
                    out.push(format!("  Plan: {}", entries.join("; ")));
                }
            }
            if let Some(end) = &r.end {
                let handed = i + 1 < task.runs.len() && task.runs[i + 1].taken_from.is_some();
                let reason = py::py_str(end);
                let shown = if handed && end == &json!("cancelled") {
                    "stopped so that the person could hand the task over".to_string()
                } else {
                    end.as_str().and_then(ended).map_or(reason, str::to_string)
                };
                out.push(format!("  Ended: {shown}."));
            }
        }
        out
    };

    if !tasks.iter().any(|t| t.runs.iter().any(|r| !r.items.is_empty() || r.end.is_some())) {
        return None;
    }
    let head = "The work on this matter that you have not seen — done while another agent worked on it — as the \
                person saw it: what was asked, what each agent said and did, and how it ended. Their reasoning \
                and what they read are not part of it: read again whatever you need.";
    let mut tail = vec![if carrying_on { String::new() } else { "The person's new task follows; the work above is what came before it.".to_string() }];
    if let Some(f) = files {
        let list = |k: &str, label: &str| -> Vec<String> {
            f.get(k).and_then(Value::as_array).into_iter().flatten().map(|p| format!("{} ({label})", py::py_str(p))).collect()
        };
        let parts: Vec<String> = [list("changed", "changed"), list("added", "new"), list("removed", "removed")].concat();
        if !parts.is_empty() {
            tail.insert(0, format!("Documents changed since the session began: {}.", parts.join(", ")));
        }
    }
    let numbered: Vec<usize> = (0..tasks.len()).map(|i| tasks[..=i].iter().filter(|t| t.text.is_some()).count()).collect();
    let mut text = String::new();
    for shortened in 0..=tasks.len() {
        let mut lines = vec![head.to_string(), String::new()];
        for (i, t) in tasks.iter().enumerate() {
            lines.extend(render(t, numbered[i], i >= shortened));
            lines.push(String::new());
        }
        lines.extend(tail.iter().filter(|x| !x.is_empty()).cloned());
        text = py::strip(&lines.join("\n")).to_string();
        if py::clen(&text) <= BRIEF_CHARS {
            return Some(text);
        }
    }
    Some(format!("…{}", py::last(&text, BRIEF_CHARS)))
}

// ── the host ──────────────────────────────────────────────────────────────────────────────────────

/// An agent of the page with its own process and session, kept while the person works with another.
pub struct Seat {
    pub choice: AgentChoice,
    pub agent: Option<Rc<AcpAgent>>,
    pub session_id: Option<String>,
    pub models: Option<Value>,
    pub context: Option<Value>,
    pub seen: usize,
    pub codex_dir: Option<PathBuf>,
    pub claude_dir: Option<PathBuf>,
}

impl Seat {
    fn new(choice: AgentChoice) -> Seat {
        Seat { choice, agent: None, session_id: None, models: None, context: None, seen: 0, codex_dir: None, claude_dir: None }
    }

    pub fn alive(&self) -> bool {
        self.agent.as_ref().is_some_and(|a| a.running()) && self.session_id.as_ref().is_some_and(|s| !s.is_empty())
    }
}

struct State {
    events: Vec<Value>,
    listeners: Vec<mpsc::UnboundedSender<Value>>,
    permissions: HashMap<String, Option<oneshot::Sender<Value>>>,
    busy: bool,
    baseline: BTreeMap<String, String>,
    status: String,
    key_refused_told: bool,
    usage: Map<String, Value>,
    agents: Vec<AgentChoice>,
    current: AgentChoice,
    /// by name, in the order they were first sat in
    seats: Vec<(String, Seat)>,
    handover: Option<AgentChoice>,
    upstream: String,
    upstream_openai: String,
    requests: Vec<RequestUsage>,
}

impl State {
    fn seat(&mut self) -> &mut Seat {
        let name = self.current.name.clone();
        self.seat_of(&name, self.current.clone())
    }

    fn seat_of(&mut self, name: &str, choice: AgentChoice) -> &mut Seat {
        if let Some(i) = self.seats.iter().position(|(n, _)| n == name) {
            return &mut self.seats[i].1;
        }
        self.seats.push((name.to_string(), Seat::new(choice)));
        &mut self.seats.last_mut().unwrap().1
    }
}

pub struct AgentHost {
    pub cfg: HostConfig,
    pub proxy: Rc<Proxy>,
    st: RefCell<State>,
}

type Boxed<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

fn jr(status: u16, data: Value) -> Response {
    Response::full(status, vec![("Content-Type".into(), "application/json; charset=utf-8".into())], py::dumps_ascii(&data))
}

fn err(status: u16, message: &str) -> Response {
    jr(status, json!({"error": message}))
}

fn server_error() -> Response {
    Response::full(500, vec![("Content-Type".into(), "text/plain; charset=utf-8".into())],
                   "500 Internal Server Error\n\nServer got itself in trouble")
}

/// `os.path.relpath(path, start)`, lexically.
fn relpath(path: &str, start: &Path) -> String {
    let abs = |p: &str| -> Vec<String> {
        let full = if p.starts_with('/') { p.to_string() } else {
            format!("{}/{p}", std::env::current_dir().unwrap_or_default().to_string_lossy())
        };
        let mut parts: Vec<String> = Vec::new();
        for part in full.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                other => parts.push(other.to_string()),
            }
        }
        parts
    };
    let (a, b) = (abs(path), abs(&start.to_string_lossy()));
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut out: Vec<String> = vec!["..".to_string(); b.len() - common];
    out.extend(a[common..].iter().cloned());
    if out.is_empty() { ".".into() } else { out.join("/") }
}

/// Python's `round(x, 1)` of seconds.
fn round1(x: f64) -> f64 {
    py::round(x, 1)
}

impl AgentHost {
    pub fn new(cfg: HostConfig, proxy: Rc<Proxy>) -> Rc<AgentHost> {
        let agents = if cfg.agents.is_empty() { vec![AgentChoice::new("", cfg.agent_command.clone())] } else { cfg.agents.clone() };
        let current = agents.iter().find(|a| a.unavailable.is_none()).unwrap_or(&agents[0]).clone();
        let (upstream, upstream_openai) = {
            let r = proxy.route.borrow();
            (r.upstream.clone(), r.upstream_openai.clone())
        };
        let mut usage = Map::new();
        usage.insert("tasks".into(), json!(0));
        let st = State { events: Vec::new(), listeners: Vec::new(), permissions: HashMap::new(), busy: false,
                         baseline: snapshot(&cfg.folder), status: "starting".into(), key_refused_told: false, usage, agents,
                         current, seats: Vec::new(), handover: None, upstream, upstream_openai, requests: Vec::new() };
        let host = Rc::new(AgentHost { cfg, proxy: proxy.clone(), st: RefCell::new(st) });
        let w = Rc::downgrade(&host);
        *proxy.on_key_refused.borrow_mut() = Some(Box::new(move || {
            if let Some(h) = w.upgrade() {
                h.key_refused();
            }
        }));
        let w = Rc::downgrade(&host);
        *proxy.on_usage.borrow_mut() = Some(Box::new(move |model: &str, usage: &Map<String, Value>| {
            if let Some(h) = w.upgrade() {
                h.st.borrow_mut().requests.push(pricing::request_usage(usage, model));
            }
        }));
        host
    }

    // ── events ──

    pub fn emit(&self, event: Value) {
        let mut e = Map::new();
        e.insert("at".into(), json!(py::time()));
        for (k, v) in event.as_object().cloned().unwrap_or_default() {
            e.insert(k, v);
        }
        let e = Value::Object(e);
        let mut st = self.st.borrow_mut();
        st.events.push(e.clone());
        st.listeners.retain(|q| q.send(e.clone()).is_ok());
    }

    pub fn status(&self) -> String {
        self.st.borrow().status.clone()
    }

    fn set_status(&self, s: &str) {
        self.st.borrow_mut().status = s.into();
    }

    fn current(&self) -> AgentChoice {
        self.st.borrow().current.clone()
    }

    fn changes(&self) -> Map<String, Value> {
        let now = snapshot(&self.cfg.folder);
        let st = self.st.borrow();
        let base = &st.baseline;
        let mut m = Map::new();
        m.insert("changed".into(), json!(now.iter().filter(|(p, d)| base.get(*p).is_some_and(|b| b != *d)).map(|(p, _)| p).collect::<Vec<_>>()));
        m.insert("added".into(), json!(now.keys().filter(|p| !base.contains_key(*p)).collect::<Vec<_>>()));
        m.insert("removed".into(), json!(base.keys().filter(|p| !now.contains_key(*p)).collect::<Vec<_>>()));
        m.insert("files".into(), json!(now.keys().collect::<Vec<_>>()));
        m
    }

    // ── the agent ──

    /// Pass the folder's text documents through the scope's vault before the first task.
    pub fn warm_vault(&self) -> Result<usize, String> {
        fn visit(dir: &Path, texts: &mut Vec<String>) -> Result<(), String> {
            let rd = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
            let mut dirs = Vec::new();
            let mut files = Vec::new();
            for e in rd.filter_map(Result::ok) {
                let p = e.path();
                let name = e.file_name().to_string_lossy().into_owned();
                let is_link = e.file_type().is_ok_and(|t| t.is_symlink());
                if p.is_dir() {
                    if !name.starts_with('.') && !is_link {
                        dirs.push((name, p));
                    }
                } else {
                    files.push((name, p));
                }
            }
            files.sort();
            dirs.sort();
            for (name, p) in files {
                let suffix = name.rfind('.').filter(|i| *i > 0 && *i + 1 < name.len())
                    .map(|i| crate::unicode::lower(&name[i..])).unwrap_or_default();
                if WARM_SUFFIXES.contains(&suffix.as_str()) {
                    let size = std::fs::metadata(&p).map_err(|e| e.to_string())?.len();
                    if size <= WARM_MAX_BYTES {
                        texts.push(py::read_text(&p).map_err(|e| e.to_string())?);
                    }
                }
                if texts.len() >= WARM_MAX_FILES {
                    break;
                }
            }
            for (_, p) in dirs {
                visit(&p, texts)?;
            }
            Ok(())
        }
        let mut texts = Vec::new();
        visit(&self.cfg.folder, &mut texts)?;
        if !texts.is_empty() {
            self.proxy.engine(&self.cfg.scope).tokenize_all(&texts)?;
        }
        Ok(texts.len())
    }

    pub async fn start_agent(self: Rc<Self>) {
        match self.warm_vault() {
            Ok(n) => log::info(&format!("vault warmed from {n} document(s)")),
            Err(e) => log::error(&format!("the vault could not be warmed from the folder's documents: {e}")),
        }
        if self.st.borrow().agents.len() > 1 {
            self.emit_agents();
        }
        let current = self.current();
        self.launch(current).await;
    }

    fn emit_agents(&self) {
        if let Some(Value::Object(m)) = self.agent_choice() {
            let mut e = Map::new();
            e.insert("kind".into(), json!("agents"));
            e.extend(m);
            self.emit(Value::Object(e));
        }
    }

    /// The agents the page offers, None when there is one.
    pub fn agent_choice(&self) -> Option<Value> {
        let st = self.st.borrow();
        if st.agents.len() < 2 {
            return None;
        }
        let options: Vec<Value> = st.agents.iter().map(|a| {
            let mut o = json!({"value": a.name, "name": a.name});
            if let Some(u) = &a.unavailable {
                o["unavailable"] = json!(u);
            }
            o
        }).collect();
        Some(json!({"current": st.current.name, "options": options}))
    }

    /// Point the proxy at `choice`'s model; whether the agent is lent the key.
    pub fn route(&self, choice: &AgentChoice) -> bool {
        let own_server = choice.upstream.is_some();
        let (upstream, upstream_openai) = {
            let st = self.st.borrow();
            (st.upstream.clone(), st.upstream_openai.clone())
        };
        let server = self.subagent_server(choice).is_some();
        let mut r = self.proxy.route.borrow_mut();
        r.upstream = choice.upstream.clone().unwrap_or(upstream).trim_end_matches('/').to_string();
        r.upstream_openai = if own_server && choice.responses() {
            format!("{}/v1", choice.upstream.as_deref().unwrap().trim_end_matches('/'))
        } else {
            upstream_openai
        };
        r.upstream_key = if own_server || choice.responses() { None } else { self.cfg.api_key.clone() };
        r.drop_tool_namespaces = if server && choice.codex() { ["collaboration".to_string()].into() } else { Default::default() };
        self.cfg.api_key.as_ref().is_some_and(|k| !k.is_empty()) && !own_server && !choice.responses()
    }

    /// `choice` becomes the agent at work; true when its seat was alive and goes on.
    pub fn take_seat(&self, choice: &AgentChoice) -> bool {
        {
            let mut st = self.st.borrow_mut();
            if choice.name != st.current.name {
                let n = st.events.len();
                st.seat().seen = n;
                st.current = choice.clone();
            }
        }
        if self.st.borrow().agents.len() > 1 {
            self.emit_agents();
        }
        let (alive, models, info) = {
            let mut st = self.st.borrow_mut();
            let seat = st.seat();
            (seat.alive(), seat.models.clone(), seat.agent.as_ref().map(|a| a.agent_info.borrow().clone()))
        };
        if alive {
            self.route(choice);
            self.set_status("ready");
            self.emit_models(models.as_ref());
            self.emit(json!({"kind": "status", "status": "ready", "agent": agent_name(info.as_ref())}));
            return true;
        }
        self.set_status("starting");
        self.emit(json!({"kind": "models", "options": []}));
        self.emit(json!({"kind": "status", "status": "starting"}));
        false
    }

    fn emit_models(&self, models: Option<&Value>) {
        match models.filter(|m| py::truthy(Some(m))) {
            Some(Value::Object(m)) => {
                let mut e = Map::new();
                e.insert("kind".into(), json!("models"));
                e.extend(m.clone());
                self.emit(Value::Object(e));
            }
            _ => self.emit(json!({"kind": "models", "options": []})),
        }
    }

    fn fail(&self, text: String) {
        self.set_status("failed");
        self.emit(json!({"kind": "error", "text": text}));
    }

    /// Start `choice` and open its session over the folder: its seat.
    pub fn launch(self: Rc<Self>, choice: AgentChoice) -> Boxed<'static, ()> {
        Box::pin(async move {
            self.st.borrow_mut().seat_of(&choice.name, choice.clone()).seen = 0;
            if let Some(why) = &choice.unavailable {
                return self.fail(format!("{} cannot start here: {why}", choice.name));
            }
            if self.cfg.api_key.as_deref().unwrap_or("").is_empty() && !choice.codex() {
                let rerouted = rerouting_settings(&self.cfg.folder);
                if !rerouted.is_empty() {
                    return self.fail(format!("The agent was not started: a settings file would send its requests around the proxy ({}). \
                                              Remove the setting, or give the agent the organization's key.", rerouted.join("; ")));
                }
            }
            let lend = self.route(&choice);
            let server = self.subagent_server(&choice);
            let mut command = choice.command.clone();
            if let Some(hosts) = &self.cfg.subagents {
                if choice.matteragent() && !command.iter().any(|a| a == "--subagents") {
                    command.extend(["--subagents".to_string(), hosts.to_string_lossy().into_owned()]);
                }
            }
            let base_url = format!("http://127.0.0.1:{}", self.cfg.port);
            let mut extra: Vec<(String, String)>;
            // Claude Code confined whole, and what it starts with if its check fails
            let (mut confined, mut legacy): (Option<confine::ConfinedClaude>, Vec<(String, String)>) = (None, Vec::new());
            if choice.codex() {
                let home = self.cfg.codex_home.clone().unwrap_or_else(|| {
                    static SAFE: OnceLock<Regex> = OnceLock::new();
                    let t: Vec<char> = self.cfg.scope.chars().collect();
                    let name = pyre::lazy(&SAFE, r"[^A-Za-z0-9._-]", pyre::NONE).sub_with(&t, 0, |_| "_".into()).0;
                    py::expanduser("~/.deid/agent/codex").join(name)
                });
                let auth = match confine::matter_codex_home(&home, None) {
                    Ok(a) => a,
                    Err(e) => return self.fail(format!("Codex was not started: {}.", acp::os_error(&e, Some(&home.to_string_lossy())))),
                };
                extra = acp::codex_env(&base_url, &self.cfg.scope, self.cfg.codex_executable.as_deref(), Some(&home.to_string_lossy()));
                let codex_args = server.as_ref().map(|s| self.codex_server_args(s)).unwrap_or_default();
                if self.cfg.codex_confine || !codex_args.is_empty() {
                    self.drop_codex_dir(&choice.name);
                    let dir = make_temp_dir("deid-codex-");
                    self.st.borrow_mut().seat_of(&choice.name, choice.clone()).codex_dir = dir.clone();
                }
                let dir = self.st.borrow_mut().seat_of(&choice.name, choice.clone()).codex_dir.clone();
                let set = |extra: &mut Vec<(String, String)>, v: String| {
                    extra.retain(|(k, _)| k != "CODEX_PATH");
                    extra.push(("CODEX_PATH".into(), v));
                };
                if !self.cfg.codex_confine && !codex_args.is_empty() {
                    if let Some(d) = &dir {
                        match confine::codex_with(self.cfg.codex_executable.as_deref(), d, &codex_args) {
                            Ok(w) => set(&mut extra, w.to_string_lossy().into_owned()),
                            Err(e) => return self.fail(format!("The agent did not start: {}", e.0)),
                        }
                    }
                }
                if self.cfg.codex_confine {
                    let extra_read = server.as_ref().map(|s| self.subagent_paths(s)).unwrap_or_default();
                    let made = dir.as_ref().ok_or_else(|| confine::ConfinementUnavailable("no folder for its profile".into()))
                        .and_then(|d| confine::confined_codex(&self.cfg.folder.to_string_lossy(), self.cfg.codex_executable.as_deref(), d,
                                                              Some(&home.to_string_lossy()), &extra_read, &codex_args,
                                                              auth.as_ref().map(|a| a.to_string_lossy().into_owned()).as_deref(),
                                                              &self.agent_network(server.as_ref())));
                    match made {
                        Ok(w) => {
                            set(&mut extra, w.to_string_lossy().into_owned());
                            extra.extend(confine::proxy_env(self.cfg.port));
                            extra.extend(confine::tools_env(&home.join("tmp").to_string_lossy()));
                        }
                        Err(e) => {
                            self.drop_codex_dir(&choice.name);
                            return self.fail(format!("Codex was not started: {}.", e.0));
                        }
                    }
                }
            } else {
                extra = match &self.cfg.claude_executable {
                    Some(c) if !c.is_empty() => vec![("CLAUDE_CODE_EXECUTABLE".into(), c.clone())],
                    _ => Vec::new(),
                };
                if server.is_some() && choice.claude() {
                    extra.push(("MCP_TOOL_TIMEOUT".into(), ((self.cfg.subagents_timeout * 1000.0) as i64).to_string()));
                }
                if choice.claude() && self.whole_claude() {
                    legacy = extra.clone();
                    match self.confine_claude(&choice, server.as_ref()) {
                        Ok(c) => {
                            let s = |p: &Path| p.to_string_lossy().into_owned();
                            extra.retain(|(k, _)| k != "CLAUDE_CODE_EXECUTABLE");
                            extra.extend([("CLAUDE_CODE_EXECUTABLE".to_string(), s(&c.executable)),
                                          ("CLAUDE_CODE_SHELL".to_string(), s(&c.shell)),
                                          ("TMPDIR".to_string(), format!("{}/", s(&c.temp))),
                                          ("CLAUDE_CODE_TMPDIR".to_string(), s(&c.temp)),
                                          // the person's claude.ai connectors (mail, drives) are not the matter's
                                          ("ENABLE_CLAUDEAI_MCP_SERVERS".to_string(), "false".to_string())]);
                            extra.extend(confine::proxy_env(self.cfg.port));
                            extra.extend(confine::tools_env(&s(&c.temp)));
                            confined = Some(c);
                        }
                        Err(e) => {
                            self.drop_claude_dir(&choice.name);
                            return self.fail(format!("Claude Code was not started: {}.", e.0));
                        }
                    }
                }
            }
            let key = if lend { Some(self.cfg.agent_key.as_str()) } else { None };
            let mut env = acp::agent_env(&base_url, &self.cfg.scope, key, &extra);
            if let Some(c) = &confined {
                // tried first: what its profile would break shows here, and Claude Code then starts as before 0.1.38
                if let Err(why) = confine::check_claude(c, &self.cfg.folder, &env, !lend).await {
                    log::warning(&format!("Claude Code is not confined whole: {why}; only its shell commands are, in its own sandbox"));
                    self.emit(json!({"kind": "notice", "text": format!(
                        "Claude Code started with only its shell commands in a sandbox, not the whole of it: {why}.")}));
                    self.drop_claude_dir(&choice.name);
                    confined = None;
                    env = acp::agent_env(&base_url, &self.cfg.scope, key, &legacy);
                }
            }
            let whole = confined.is_some();
            let (w1, w2) = (Rc::downgrade(&self), Rc::downgrade(&self));
            let (n1, n2) = (choice.name.clone(), choice.name.clone());
            let agent = AcpAgent::new(command, env, self.cfg.folder.to_string_lossy().into_owned(),
                Rc::new(move |params: Value| {
                    if let Some(h) = w1.upgrade() {
                        h.on_update(&params, Some(&n1));
                    }
                }),
                Rc::new(move |params: Value| -> Boxed<'static, Value> {
                    let (w, n) = (w2.clone(), n2.clone());
                    Box::pin(async move {
                        match w.upgrade() {
                            Some(h) => h.on_permission(params, Some(n)).await,
                            None => json!({"outcome": "cancelled"}),
                        }
                    })
                }));
            self.st.borrow_mut().seat_of(&choice.name, choice.clone()).agent = Some(agent.clone());
            let mut meta = choice.meta.clone().filter(|m| py::truthy(Some(m))).or_else(|| self.cfg.session_meta.clone());
            if server.is_some() && choice.claude() {
                meta = Some(with_subagents(meta.as_ref()));
            }
            if choice.claude() {
                let mut settings = claude_settings(&self.cfg.folder, self.cfg.claude_confine && !whole, &self.cfg.guard,
                                                   self.cfg.guard_log.as_deref());
                if whole {
                    // its own command sandbox cannot start inside the profile it runs under: the launcher stands for it
                    settings.insert("sandbox".into(), json!({"enabled": false}));
                }
                match with_claude_settings(meta.as_ref(), &settings, &self.cfg.folder) {
                    Ok(m) => meta = Some(m),
                    Err(e) => return self.fail(format!("Claude Code was not started: its settings: {e}")),
                }
            }
            if server.is_some() && choice.codex() {
                let mut m = meta.clone().filter(|m| py::truthy(Some(m))).and_then(|m| m.as_object().cloned()).unwrap_or_default();
                m.insert("mcpStartupAwaitTimeoutMs".into(), json!(30000));
                meta = Some(Value::Object(m));
            }
            let started: Result<(Value, String), AcpError> = async {
                let info = agent.start().await?;
                let servers = match &server {
                    Some((cmd, args)) => vec![json!({"name": "matter", "command": cmd, "args": args, "env": []})],
                    None => Vec::new(),
                };
                let sid = agent.new_session(&self.cfg.folder.to_string_lossy(), meta.as_ref(), servers).await?;
                Ok((info, sid))
            }.await;
            match started {
                Ok((info, sid)) => {
                    let models = model_choice(&agent.session_setup.borrow());
                    {
                        let mut st = self.st.borrow_mut();
                        let seat = st.seat_of(&choice.name, choice.clone());
                        seat.session_id = Some(sid);
                        seat.models = models.clone();
                    }
                    if models.is_some() {
                        self.emit_models(models.as_ref());
                    }
                    self.set_status("ready");
                    self.emit(json!({"kind": "status", "status": "ready", "agent": agent_name(Some(&info))}));
                }
                Err(e) => {
                    log::error(&format!("the agent did not start: {e}"));
                    self.fail(format!("The agent did not start: {e}"));
                }
            }
        })
    }

    /// The sub-agents' MCP server for `choice`, as (command, args).
    pub fn subagent_server(&self, choice: &AgentChoice) -> Option<(String, Vec<String>)> {
        let hosts = self.cfg.subagents.as_ref()?;
        if self.cfg.subagents_command.is_empty() || choice.matteragent() {
            return None;
        }
        let family = if choice.codex() { "gpt" } else if choice.claude() { "claude" } else { "" };
        let mut command = self.cfg.subagents_command.clone();
        command.extend(["--folder".into(), self.cfg.folder.to_string_lossy().into_owned(), "--hosts".into(), hosts.to_string_lossy().into_owned()]);
        if !family.is_empty() {
            command.extend(["--author-family".into(), family.into()]);
        }
        Some((command[0].clone(), command[1..].to_vec()))
    }

    /// The matter server for Codex, defined whole where Codex starts (`-c`).
    pub fn codex_server_args(&self, server: &(String, Vec<String>)) -> Vec<String> {
        vec!["-c".into(), format!("mcp_servers.matter.command={}", py::dumps(&json!(server.0))),
             "-c".into(), format!("mcp_servers.matter.args={}", py::dumps(&json!(server.1))),
             "-c".into(), format!("mcp_servers.matter.tool_timeout_sec={}", self.cfg.subagents_timeout as i64),
             "-c".into(), "mcp_servers.matter.startup_timeout_sec=30".into(),
             "-c".into(), "mcp_servers.matter.default_tools_approval_mode=\"approve\"".into(),
             "-c".into(), format!("developer_instructions={}", py::dumps(&json!(CODEX_SUBAGENTS_NOTE)))]
    }

    /// What the sub-agents' MCP server reads besides the folder, for Codex's confinement.
    pub fn subagent_paths(&self, server: &(String, Vec<String>)) -> Vec<String> {
        let mut paths = vec![self.cfg.subagents.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()];
        let program = confine::which(&server.0).unwrap_or_else(|| server.0.clone());
        let real = py::realpath(Path::new(&program)).to_string_lossy().into_owned();
        paths.push(real.clone());
        paths.push(confine::dirname(&confine::dirname(&real)));
        if base_name(&program).starts_with("python") || shebang_python(&program).is_some() {
            let python = shebang_python(&program).unwrap_or_else(|| program.clone());
            let probe = "import os, sys\nprint(sys.prefix)\nprint(sys.base_prefix)\ntry:\n    import matteragent\n    print(os.path.dirname(matteragent.__path__[0]))\nexcept Exception:\n    pass\n";
            match run_with_timeout(&python, &["-c", probe], Duration::from_secs(20)) {
                Some(out) => paths.extend(out.lines().filter(|l| !py::strip(l).is_empty()).map(str::to_string)),
                None => log::warning(&format!("could not ask {python} where it is installed")),
            }
            let rp = py::realpath(Path::new(&python)).to_string_lossy().into_owned();
            paths.push(confine::dirname(&confine::dirname(&rp)));
        }
        let mut seen = Vec::new();
        paths.retain(|p| if seen.contains(p) { false } else { seen.push(p.clone()); true });
        paths
    }

    pub fn on_update(&self, params: &Value, choice: Option<&str>) {
        if let Some(name) = choice {
            if name != self.st.borrow().current.name {
                return;
            }
        }
        let u = params.get("update").filter(|u| py::truthy(Some(u))).cloned().unwrap_or(json!({}));
        let kind = u.get("sessionUpdate").and_then(Value::as_str).unwrap_or("");
        let content = u.get("content").filter(|c| c.is_object()).cloned().unwrap_or(json!({}));
        let get = |k: &str, d: Value| u.get(k).cloned().unwrap_or(d);
        match kind {
            "agent_message_chunk" | "agent_thought_chunk" if content.get("type") == Some(&json!("text")) => {
                self.emit(json!({"kind": if kind == "agent_message_chunk" { "message" } else { "thought" },
                                 "text": content.get("text").cloned().unwrap_or(json!(""))}));
            }
            "tool_call" => self.emit(json!({"kind": "tool", "id": get("toolCallId", Value::Null), "title": get("title", json!("")),
                                            "tool": get("kind", Value::Null), "status": get("status", json!("pending"))})),
            "tool_call_update" => self.emit(json!({"kind": "tool_update", "id": get("toolCallId", Value::Null),
                                                   "status": get("status", Value::Null), "title": get("title", Value::Null)})),
            "plan" => {
                let entries: Vec<Value> = u.get("entries").and_then(Value::as_array).into_iter().flatten().filter(|e| e.is_object())
                    .map(|e| json!({"content": e.get("content").cloned().unwrap_or(json!("")),
                                    "status": e.get("status").cloned().unwrap_or(json!(""))})).collect();
                self.emit(json!({"kind": "plan", "entries": entries}));
            }
            "usage_update" => {
                let is_int = |v: Option<&Value>| matches!(v, Some(Value::Number(n)) if n.is_i64() || n.is_u64()) || matches!(v, Some(Value::Bool(_)));
                if is_int(u.get("used")) && is_int(u.get("size")) {
                    let mut context = json!({"used": u["used"], "size": u["size"]});
                    if let Some(Value::Object(cost)) = u.get("cost") {
                        if cost.contains_key("amount") && cost.contains_key("currency") {
                            context["cost"] = json!({"amount": cost["amount"], "currency": cost["currency"]});
                        }
                    }
                    self.st.borrow_mut().seat().context = Some(context.clone());
                    let mut e = Map::new();
                    e.insert("kind".into(), json!("usage"));
                    e.extend(context.as_object().unwrap().clone());
                    self.emit(Value::Object(e));
                }
            }
            "config_option_update" => {
                let choice = model_choice(&json!({"configOptions": u.get("configOptions").cloned().unwrap_or(Value::Null)}));
                let changed = choice.is_some() && choice != self.st.borrow_mut().seat().models;
                if changed {
                    self.st.borrow_mut().seat().models = choice.clone();
                    self.emit_models(choice.as_ref());
                }
            }
            _ => {}
        }
    }

    pub async fn on_permission(self: Rc<Self>, params: Value, choice: Option<String>) -> Value {
        {
            let st = self.st.borrow();
            if choice.as_ref().is_some_and(|c| *c != st.current.name) || st.handover.is_some() {
                return json!({"outcome": "cancelled"});
            }
        }
        let call = params.get("toolCall").filter(|c| c.is_object()).cloned().unwrap_or(json!({}));
        let options: Vec<Value> = params.get("options").and_then(Value::as_array).cloned().unwrap_or_default();
        let paths: Vec<String> = call.get("locations").and_then(Value::as_array).into_iter().flatten()
            .filter_map(|l| l.get("path").filter(|p| py::truthy(Some(p))).map(py::py_str)).collect();
        let mut outside: Vec<String> = paths.iter().filter(|p| !inside(&self.cfg.folder, p)).cloned().collect();
        let raw = call.get("rawInput").filter(|r| r.is_object()).cloned().unwrap_or(json!({}));
        let mut command = raw.get("command").and_then(Value::as_str).map(str::to_string);
        if command.is_none() && call.get("kind") == Some(&json!("execute")) {
            command = Some(or_text(&call, "title", ""));
        }
        if let Some(c) = command.filter(|c| !c.is_empty()) {
            outside.extend(outside_paths(&c, &self.cfg.folder));
        }
        if !outside.is_empty() {
            let reject = options.iter().find(|o| o.get("kind").and_then(Value::as_str).unwrap_or("").starts_with("reject"));
            self.emit(json!({"kind": "notice", "text": "Refused without asking: the action names a path outside the matter folder."}));
            return match reject {
                Some(r) => json!({"outcome": "selected", "optionId": r.get("optionId").cloned().unwrap_or(Value::Null)}),
                None => json!({"outcome": "cancelled"}),
            };
        }
        let rid = py::token_hex(8);
        let (tx, rx) = oneshot::channel();
        self.st.borrow_mut().permissions.insert(rid.clone(), Some(tx));
        let shown: Vec<String> = paths.iter().map(|p| if p.starts_with('/') { relpath(p, &self.cfg.folder) } else { p.clone() }).collect();
        self.emit(json!({"kind": "permission", "id": rid, "title": call.get("title").cloned().unwrap_or(json!("An action")),
                         "tool": call.get("kind").cloned().unwrap_or(Value::Null), "paths": shown,
                         "options": options.iter().map(|o| json!({"optionId": o.get("optionId").cloned().unwrap_or(Value::Null),
                                                                  "name": o.get("name").cloned().unwrap_or(Value::Null),
                                                                  "kind": o.get("kind").cloned().unwrap_or(Value::Null)})).collect::<Vec<_>>()}));
        let answer = rx.await.unwrap_or(json!({"outcome": "cancelled"}));
        self.st.borrow_mut().permissions.remove(&rid);
        let option = options.iter().find(|o| answer.get("outcome") == Some(&json!("selected"))
                                          && o.get("optionId").unwrap_or(&Value::Null) == answer.get("optionId").unwrap_or(&Value::Null));
        let outcome = match option {
            None => "cancelled",
            Some(o) if o.get("kind").map_or("None".into(), py::py_str).starts_with("allow") => "allowed",
            Some(_) => "refused",
        };
        self.emit(json!({"kind": "permission_done", "id": rid, "outcome": outcome}));
        answer
    }

    /// The provider refused the organization's key: the person is told, a running turn stopped.
    pub fn key_refused(&self) {
        let (busy, agent, sid) = {
            let mut st = self.st.borrow_mut();
            if st.key_refused_told {
                return;
            }
            st.key_refused_told = true;
            let busy = st.busy;
            let seat = st.seat();
            (busy, seat.agent.clone(), seat.session_id.clone())
        };
        let text = if let (true, Some(agent), Some(sid)) = (busy, agent, sid.filter(|s| !s.is_empty())) {
            tokio::task::spawn_local(async move {
                let _ = agent.cancel(&sid).await;
            });
            "Anthropic refused the organization's key, so the turn was stopped."
        } else {
            "Anthropic refuses the organization's key."
        };
        self.emit(json!({"kind": "error", "text": format!("{text} The key is wrong or revoked: replace it, then open the agent again.")}));
    }

    /// A task for the agent at work — or, `carrying_from` another agent, the task under way, handed over.
    pub fn run_prompt(self: Rc<Self>, text: String, carrying_from: Option<String>, shown: Option<String>) -> Boxed<'static, ()> {
        Box::pin(async move {
            let (told, me, agent, sid) = {
                let mut st = self.st.borrow_mut();
                st.busy = true;
                st.key_refused_told = false;
                st.requests.clear();
                let me = st.current.name.clone();
                let seen = st.seat().seen;
                let told = brief(&st.events[seen.min(st.events.len())..], &me, carrying_from.is_some());
                let seat = st.seat();
                (told, me, seat.agent.clone(), seat.session_id.clone())
            };
            if carrying_from.is_none() {
                self.emit(json!({"kind": "user", "text": shown.clone().unwrap_or_else(|| text.clone()), "agent": me}));
            }
            if let Some(t) = &told {
                self.emit(json!({"kind": "brief", "agent": me, "text": t}));
            }
            {
                let mut st = self.st.borrow_mut();
                let n = st.events.len();
                st.seat_of(&me, AgentChoice::default()).seen = n;
            }
            let started = Instant::now();
            let before = snapshot(&self.cfg.folder);
            let mut blocks: Vec<String> = told.into_iter().collect();
            blocks.push(text);
            let result = match (agent, sid) {
                (Some(a), Some(s)) => a.prompt(&s, &blocks).await,
                (None, _) => Err(AcpError("'NoneType' object has no attribute 'prompt'".into())),
                (Some(a), None) => a.prompt("", &blocks).await,
            };
            match result {
                Ok(result) if result.is_object() => {
                    let usage = turn_usage(&result);
                    let seconds = round1(started.elapsed().as_secs_f64());
                    let requests = self.st.borrow().requests.clone();
                    let cost = (!requests.is_empty()).then(|| pricing::task_cost(&requests));
                    let mut e = json!({"kind": "turn_end", "stopReason": result.get("stopReason").cloned().unwrap_or(Value::Null),
                                       "usage": usage, "seconds": seconds});
                    if let Some(c) = &cost {
                        e["cost"] = c.clone();
                    }
                    self.emit(e);
                    self.count(result.get("stopReason").cloned().unwrap_or(Value::Null), &usage, seconds, cost.as_ref());
                }
                Ok(other) => {
                    let _ = turn_usage(&other);
                    self.emit(json!({"kind": "error", "text": format!("The turn failed: '{}' object has no attribute 'get'", acp::type_name(&other))}));
                }
                Err(e) => self.emit(json!({"kind": "error", "text": format!("The turn failed: {e}")})),
            }
            self.st.borrow_mut().busy = false;
            self.note_written(&before);
            let mut files = Map::new();
            files.insert("kind".into(), json!("files"));
            files.extend(self.changes());
            self.emit(Value::Object(files));
            let handover = {
                let mut st = self.st.borrow_mut();
                let n = st.events.len();
                st.seat_of(&me, AgentChoice::default()).seen = n;
                st.handover.clone()
            };
            if let Some(h) = handover {
                let host = self.clone();
                let current = self.current().name;
                tokio::task::spawn_local(async move { host.carry_on(h, current).await });
            }
        })
    }

    /// Who wrote what in the task just done: a note on each text document the agent added or changed.
    fn note_written(&self, before: &BTreeMap<String, String>) {
        let now = snapshot(&self.cfg.folder);
        let model = self.model_name();
        let label = self.current().label();
        for (path, digest) in &now {
            let how = match before.get(path) {
                None => "added",
                Some(d) if d != digest => "changed",
                _ => "",
            };
            if !how.is_empty() && review::writable(path) {
                let mut creator = json!({"type": "Software", "name": label});
                if !model.is_empty() {
                    creator["model"] = json!(model);
                }
                if review::store(&self.cfg.folder, path, &review::written(path, &creator, how, "")).is_err() {
                    log::warning("who wrote the documents of the task was not noted");
                    return;
                }
            }
        }
    }

    /// The task under way, stopped so that the person could hand it to `choice`, goes on with `choice`.
    pub fn carry_on(self: Rc<Self>, choice: AgentChoice, previous: String) -> Boxed<'static, ()> {
        Box::pin(async move {
            if !self.take_seat(&choice) {
                self.clone().launch(choice.clone()).await;
            }
            if self.status() != "ready" {
                self.st.borrow_mut().handover = None;
                return;
            }
            self.emit(json!({"kind": "handover", "from": previous, "to": choice.name}));
            self.st.borrow_mut().handover = None;
            self.run_prompt(CARRY_ON.replace("{previous}", &previous), Some(previous), None).await;
        })
    }

    /// Stop the turn under way: `session/cancel`, and the asks it waits on answered as cancelled.
    pub async fn stop_turn(&self) {
        let (agent, sid) = {
            let mut st = self.st.borrow_mut();
            let seat = st.seat();
            (seat.agent.clone(), seat.session_id.clone())
        };
        if let (Some(a), Some(s)) = (agent, sid.filter(|s| !s.is_empty())) {
            if a.cancel(&s).await.is_err() {
                log::warning("the cancel did not reach the agent");
            }
        }
        let waiting: Vec<oneshot::Sender<Value>> = self.st.borrow_mut().permissions.values_mut().filter_map(Option::take).collect();
        for tx in waiting {
            let _ = tx.send(json!({"outcome": "cancelled"}));
        }
    }

    async fn stop_if_stuck(self: Rc<Self>, seat_name: String, after: Duration) {
        tokio::time::sleep(after).await;
        let agent = {
            let mut st = self.st.borrow_mut();
            let go = st.busy && st.handover.is_some() && st.current.name == seat_name;
            if !go {
                return;
            }
            st.seat().agent.clone()
        };
        if let Some(a) = agent {
            self.emit(json!({"kind": "notice", "text": format!("{seat_name} did not stop when asked: it was stopped.")}));
            a.stop().await;
        }
    }

    /// Add a finished task to the session's totals and to the usage log.
    fn count(&self, stop_reason: Value, usage: &Map<String, Value>, seconds: f64, cost: Option<&Value>) {
        let (agents, current, model) = {
            let mut st = self.st.borrow_mut();
            let tasks = st.usage.get("tasks").and_then(Value::as_i64).unwrap_or(0) + 1;
            st.usage.insert("tasks".into(), json!(tasks));
            for (k, v) in usage {
                let sum = st.usage.get(k).and_then(Value::as_i64).unwrap_or(0) + v.as_i64().unwrap_or(0);
                st.usage.insert(k.clone(), json!(sum));
            }
            let model = st.seat().models.as_ref().and_then(|m| m.get("current").cloned()).unwrap_or(Value::Null);
            (st.agents.len(), st.current.name.clone(), model)
        };
        let Some(path) = &self.cfg.usage_log else { return };
        let mut line = Map::new();
        line.insert("at".into(), json!(py::utc_seconds()));
        line.insert("scope".into(), json!(self.cfg.scope));
        if agents > 1 {
            line.insert("agent".into(), json!(current));
        }
        line.insert("model".into(), model);
        line.insert("stopReason".into(), stop_reason);
        line.insert("seconds".into(), json!(seconds));
        line.insert("usage".into(), Value::Object(usage.clone()));
        if let Some(c) = cost {
            line.insert("cost".into(), c.clone());
        }
        if append_line(path, &Value::Object(line)).is_err() {
            log::error("the usage log could not be written");
        }
    }

    fn model_name(&self) -> String {
        let mut st = self.st.borrow_mut();
        if let Some(m) = st.requests.last().map(|r| r.model.clone()).filter(|m| !m.is_empty()) {
            return m;
        }
        let models = st.seat().models.clone().unwrap_or(json!({}));
        let current = models.get("current").cloned().unwrap_or(Value::Null);
        for o in models.get("options").and_then(Value::as_array).into_iter().flatten() {
            if o.get("value").unwrap_or(&Value::Null) == &current {
                return o.get("name").map(py::py_str).unwrap_or_default();
            }
        }
        if py::truthy(Some(&current)) { py::py_str(&current) } else { String::new() }
    }

    // ── HTTP ──

    pub async fn handle(self: Rc<Self>, req: Request) -> Response {
        let get = req.method == "GET" || req.method == "HEAD";
        let path = req.path.clone();
        if path == "/" && get {
            let mut html = PAGE.to_string();
            if self.cfg.style.is_some() {
                html = html.replacen("</head>", "<link rel=\"stylesheet\" href=\"/deid-agent/style.css\">\n</head>", 1);
            }
            return Response::full(200, vec![("Content-Type".into(), "text/html; charset=utf-8".into()),
                                            ("Cache-Control".into(), "no-store".into())], html);
        }
        if path == "/deid-agent/style.css" && get {
            let css = self.cfg.style.as_ref().and_then(|s| py::read_text_strict(s).ok().flatten());
            return match css {
                Some(css) => Response::full(200, vec![("Content-Type".into(), "text/css; charset=utf-8".into()),
                                                      ("Cache-Control".into(), "no-store".into())], css),
                None => err(404, "not found"),
            };
        }
        if let Some(name) = path.strip_prefix("/deid-agent/api/") {
            if !name.is_empty() && !name.contains('/') {
                let name = name.to_string();
                return self.api(&name, req).await;
            }
        }
        self.proxy.clone().handle(req).await
    }

    fn authorised(&self, req: &Request) -> bool {
        let given = req.header("x-deid-agent-token").unwrap_or("").as_bytes();
        let want = self.cfg.token.as_bytes();
        given.len() == want.len() && given.iter().zip(want).fold(0u8, |a, (x, y)| a | (x ^ y)) == 0
    }

    async fn api(self: Rc<Self>, name: &str, req: Request) -> Response {
        if !self.authorised(&req) {
            return err(401, "unauthorised");
        }
        let method = req.method.as_str();
        let body = || -> Option<Value> {
            let text = std::str::from_utf8(&req.body).ok()?;
            serde_json::from_str(text).ok()
        };
        /// `(body.get(k) or "").strip()`: None where Python raises (a value that is no string).
        fn stripped(b: &Value, k: &str) -> Option<String> {
            match b.get(k) {
                Some(v) if py::truthy(Some(v)) => v.as_str().map(|s| py::strip(s).to_string()),
                _ => Some(String::new()),
            }
        }
        match (name, method) {
            ("state", "GET") => {
                let mut st = self.st.borrow_mut();
                let auth = if self.cfg.api_key.as_deref().is_some_and(|k| !k.is_empty()) { "organization key" } else { "personal sign-in (development)" };
                let (models, context) = {
                    let seat = st.seat();
                    (seat.models.clone(), seat.context.clone())
                };
                let mut m = Map::new();
                m.insert("title".into(), json!(self.cfg.title));
                m.insert("folder".into(), json!(self.cfg.folder.to_string_lossy()));
                m.insert("scope".into(), json!(self.cfg.scope));
                m.insert("status".into(), json!(st.status));
                m.insert("busy".into(), json!(st.busy));
                m.insert("auth".into(), json!(auth));
                m.insert("models".into(), models.unwrap_or(Value::Null));
                let handover = st.handover.as_ref().map(|h| h.name.clone());
                let usage = st.usage.clone();
                drop(st);
                m.insert("agents".into(), self.agent_choice().unwrap_or(Value::Null));
                m.insert("handover".into(), handover.map_or(Value::Null, |h| json!(h)));
                m.insert("usage".into(), Value::Object(usage));
                m.insert("context".into(), context.unwrap_or(Value::Null));
                m.extend(self.changes());
                jr(200, Value::Object(m))
            }
            ("events", "GET") => self.events_stream(),
            ("prompt", "POST") => {
                let Some(b) = body().filter(Value::is_object) else { return server_error() };
                let Some(text) = stripped(&b, "text") else { return server_error() };
                if text.is_empty() {
                    return err(400, "empty task");
                }
                {
                    let st = self.st.borrow();
                    if st.busy || st.status != "ready" || st.handover.is_some() {
                        return err(409, "the agent is busy or not ready");
                    }
                }
                let host = self.clone();
                tokio::task::spawn_local(async move { host.run_prompt(text, None, None).await });
                jr(202, json!({"ok": true}))
            }
            ("permission", "POST") => {
                let Some(b) = body().filter(Value::is_object) else { return server_error() };
                let id = match b.get("id") {
                    None => String::new(),
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(_) | Value::Object(_)) => return server_error(),
                    Some(_) => return err(404, "no such request"),
                };
                let tx = self.st.borrow_mut().permissions.get_mut(&id).and_then(Option::take);
                let Some(tx) = tx else { return err(404, "no such request") };
                let option = b.get("optionId").filter(|o| py::truthy(Some(o)));
                let _ = tx.send(match option {
                    Some(o) => json!({"outcome": "selected", "optionId": o}),
                    None => json!({"outcome": "cancelled"}),
                });
                jr(200, json!({"ok": true}))
            }
            ("model", "POST") => {
                let Some(b) = body().filter(Value::is_object) else { return server_error() };
                let Some(value) = stripped(&b, "value") else { return server_error() };
                self.set_model(value).await
            }
            ("agent", "POST") => {
                let Some(b) = body().filter(Value::is_object) else { return server_error() };
                let Some(value) = stripped(&b, "value") else { return server_error() };
                self.set_agent(value, b.get("handover") == Some(&json!(true))).await
            }
            ("cancel", "POST") => {
                self.stop_turn().await;
                jr(200, json!({"ok": true}))
            }
            ("review", "GET") => self.review_view(&req.query("path").unwrap_or_default()),
            ("review", "POST") => {
                let Some(b) = body() else { return server_error() };
                let b = if b.is_object() { b } else { json!({}) };
                let s = |k: &str| py::strip(&or_text(&b, k, "")).to_string();
                self.review_start(s("path"), s("agent"))
            }
            ("review-suggest" | "review-decide" | "review-text" | "review-vote", "POST") => {
                let Some(b) = body() else { return server_error() };
                if !b.is_object() {
                    return err(400, "bad request");
                }
                match name {
                    "review-suggest" => self.review_suggest(&b),
                    "review-decide" => self.review_decide(&b),
                    "review-text" => self.review_text(&b),
                    _ => self.review_vote(&b),
                }
            }
            ("review-dataset", "GET") => match review::dataset(&self.cfg.folder, &self.proxy.engine(&self.cfg.scope)) {
                Ok(records) => {
                    let count = records.len();
                    jr(200, json!({"records": records, "count": count}))
                }
                Err(_) => server_error(),
            },
            ("file", "GET") => {
                let rel = req.query("path").unwrap_or_default();
                let target = if rel.starts_with('/') { PathBuf::from(&rel) } else { self.cfg.folder.join(&rel) };
                if !inside(&self.cfg.folder, &target.to_string_lossy()) || !target.is_file() {
                    return err(404, "no such file");
                }
                let data = std::fs::read(&target).unwrap_or_default();
                let data = &data[..data.len().min(MAX_FILE as usize)];
                jr(200, json!({"path": rel, "text": String::from_utf8_lossy(data)}))
            }
            _ => err(404, "not found"),
        }
    }

    fn events_stream(&self) -> Response {
        let (resp, out) = Response::stream(200, vec![("Content-Type".into(), "text/event-stream".into()),
                                                     ("Cache-Control".into(), "no-store".into())]);
        let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
        {
            let mut st = self.st.borrow_mut();
            for e in &st.events {
                let _ = tx.send(e.clone());
            }
            st.listeners.push(tx);
        }
        tokio::task::spawn_local(async move {
            loop {
                let chunk = match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                    Ok(Some(e)) => format!("data: {}\n\n", py::dumps(&e)),
                    Ok(None) => break,
                    Err(_) => ": keep-alive\n\n".to_string(),
                };
                if out.send(Bytes::from(chunk)).is_err() {
                    break;
                }
            }
        });
        resp
    }

    // ── review ──

    fn review_view(&self, path: &str) -> Response {
        let Some((source, text)) = review::read_text(&self.cfg.folder, path) else {
            return jr(404, json!({"error": "no text for this document", "source": review::text_source(path)}));
        };
        let reds = match self.proxy.engine(&self.cfg.scope).tokenize_all(&[text.clone()]) {
            Ok(r) => r,
            Err(_) => return server_error(),
        };
        let red = &reds[0];
        let deid = review::deid_spans(&text, &red.text, &red.surfaces, &red.mapping);
        let (label, busy, ready) = {
            let st = self.st.borrow();
            (st.current.label(), st.busy, st.status == "ready")
        };
        let mut m = Map::new();
        m.insert("path".into(), json!(path));
        m.insert("source".into(), json!(source));
        m.insert("text".into(), json!(text));
        m.insert("deid".into(), json!(deid));
        m.insert("norms".into(), json!(review::norm_mentions(&text)));
        m.extend(review::view(&self.cfg.folder, path, &text).as_object().unwrap().clone());
        m.insert("writable".into(), json!(review::writable(path)));
        m.insert("sha".into(), json!(review::sha(&text)));
        m.insert("person".into(), json!(self.cfg.person));
        m.insert("agent".into(), json!({"name": label, "busy": busy, "ready": ready}));
        jr(200, Value::Object(m))
    }

    fn review_suggest(&self, body: &Value) -> Response {
        let path = or_text(body, "path", "");
        let (start, end) = (int_value(body.get("start")), int_value(body.get("end")));
        let replacement = body.get("text").and_then(Value::as_str);
        let Some((_, text)) = review::read_text(&self.cfg.folder, &path) else { return err(404, "no text for this document") };
        let t: Vec<char> = text.chars().collect();
        let ok = match (start, end, replacement) {
            (Some(s), Some(e), Some(r)) => 0 <= s && s <= e && e <= t.len() as i64 && py::clen(r) <= review::MAX_REPLACEMENT
                && !(s == e && r.is_empty()) && t[s as usize..e as usize].iter().collect::<String>() != r,
            _ => false,
        };
        if !ok {
            return err(400, "no change to propose");
        }
        if py::truthy(body.get("sha")) && body["sha"] != json!(review::sha(&text)) {
            return err(409, "the document changed: open it again");
        }
        let ann = review::suggestion(&path, &t, start.unwrap() as usize, end.unwrap() as usize, replacement.unwrap(),
                                     &review::person(&self.cfg.person));
        if review::store(&self.cfg.folder, &path, &ann).is_err() {
            return server_error();
        }
        jr(200, json!({"ok": true, "id": ann["id"]}))
    }

    fn review_decide(&self, body: &Value) -> Response {
        let path = or_text(body, "path", "");
        let target = or_text(body, "target", "");
        let verdict = body.get("verdict").and_then(Value::as_str).unwrap_or("");
        let comment = review::text_of(body.get("comment"), review::MAX_NOTE);
        let correction = body.get("correction").and_then(Value::as_str)
            .map_or(String::new(), |c| py::take(py::strip(c), review::MAX_REPLACEMENT).to_string());
        if !review::VERDICTS.contains(&verdict) || !body.get("verdict").is_some_and(Value::is_string) {
            return err(400, "no such decision");
        }
        let ann = review::find(&self.cfg.folder, &path, &target);
        let Some(ann) = ann.filter(|a| !["decision", "revision", "written"].contains(&review::kind_of(a).as_str())) else {
            return err(404, "no such note");
        };
        let (mut applied, mut why) = (false, "");
        let replacement = review::body_of(&ann, "replacing", review::MAX_REPLACEMENT);
        if verdict == "accepted" && (!replacement.is_empty() || review::kind_of(&ann) == "edit") {
            if !review::writable(&path) {
                why = "a Word or PDF document is changed in its own program: the decision is kept";
            } else {
                let changed = review::read_text(&self.cfg.folder, &path).and_then(|r| review::apply(&r.1, &ann, &replacement));
                let Some(changed) = changed else { return err(409, "its words are no longer in the document") };
                if review::write_text(&self.cfg.folder, &path, &changed).is_err() {
                    return server_error();
                }
                applied = true;
            }
        }
        let d = review::decision(&path, &target, verdict, &review::person(&self.cfg.person), &comment, &correction, applied);
        if review::store(&self.cfg.folder, &path, &d).is_err() {
            return server_error();
        }
        let mut out = json!({"ok": true, "applied": applied});
        if !why.is_empty() {
            out["why"] = json!(why);
        }
        jr(200, out)
    }

    fn review_vote(&self, body: &Value) -> Response {
        let verdict = body.get("verdict").and_then(Value::as_str).unwrap_or("");
        let key = body.get("key").and_then(Value::as_str);
        let question = body.get("question").and_then(Value::as_str);
        let answer = body.get("answer").and_then(Value::as_str);
        let ok = (verdict == "up" || verdict == "down") && body.get("verdict").is_some_and(Value::is_string)
            && key.is_some_and(|k| !py::strip(k).is_empty() && py::clen(k) <= 200)
            && answer.is_some_and(|a| !py::strip(a).is_empty()) && question.is_some();
        if !ok {
            return err(400, "no such vote");
        }
        let correction = body.get("correction").and_then(Value::as_str)
            .map_or(String::new(), |c| py::take(py::strip(c), review::MAX_REPLACEMENT).to_string());
        let ann = review::vote(key.unwrap(), question.unwrap(), answer.unwrap(), &or_text(body, "agent", ""), &or_text(body, "model", ""),
                               verdict, &review::person(&self.cfg.person), &review::text_of(body.get("comment"), review::MAX_NOTE), &correction);
        if review::store_vote(&self.cfg.folder, &ann).is_err() {
            return server_error();
        }
        jr(200, json!({"ok": true, "id": ann["id"]}))
    }

    fn review_text(&self, body: &Value) -> Response {
        let path = or_text(body, "path", "");
        let base = or_text(body, "sha", "");
        let Some(text) = body.get("text").and_then(Value::as_str).filter(|t| t.len() <= review::MAX_TEXT) else {
            return err(400, "no text");
        };
        let me = review::person(&self.cfg.person);
        if base.is_empty() {
            if !review::new_path_ok(&self.cfg.folder, &path) {
                return err(400, "a new document is a .md or .txt file of the folder, not one that is there");
            }
            if review::write_text(&self.cfg.folder, &path, text).is_err() {
                return server_error();
            }
            let by = body.get("written_by").filter(|b| b.is_object());
            let creator = match by {
                Some(by) => {
                    let name = review::text_of(by.get("agent"), 80);
                    let mut c = json!({"type": "Software", "name": if name.is_empty() { "Agent".to_string() } else { name }});
                    if py::truthy(by.get("model")) {
                        c["model"] = json!(review::text_of(by.get("model"), 80));
                    }
                    c
                }
                None => me.clone(),
            };
            let _ = review::store(&self.cfg.folder, &path, &review::written(&path, &creator, "wrote", ""));
            if by.is_some() {
                let _ = review::store(&self.cfg.folder, &path, &review::written(&path, &me, "saved", ""));
            }
            return jr(200, json!({"ok": true, "sha": review::sha(text)}));
        }
        if !review::writable(&path) {
            return err(400, "a Word or PDF document is changed in its own program");
        }
        let Some((_, before)) = review::read_text(&self.cfg.folder, &path) else { return err(404, "no such document") };
        if review::sha(&before) != base {
            return err(409, "the document changed since it was opened: open it again");
        }
        if text != before {
            if review::write_text(&self.cfg.folder, &path, text).is_err() {
                return server_error();
            }
            let _ = review::store(&self.cfg.folder, &path, &review::revision(&path, &before, text, &me));
        }
        jr(200, json!({"ok": true, "sha": review::sha(text)}))
    }

    fn review_start(self: Rc<Self>, path: String, agent: String) -> Response {
        let Some((source, _)) = review::read_text(&self.cfg.folder, &path) else {
            return jr(404, json!({"error": "no text for this document", "source": review::text_source(&path)}));
        };
        let current = self.current();
        let mut reviewer = None;
        if !agent.is_empty() && agent != current.name && agent != current.label() {
            let found = self.st.borrow().agents.iter().find(|a| agent == a.name || agent == a.label()).cloned();
            let Some(r) = found else { return err(400, "there is no such agent") };
            if let Some(why) = &r.unavailable {
                return err(409, &format!("{} cannot start here: {why}", r.name));
            }
            reviewer = Some(r);
        }
        {
            let mut st = self.st.borrow_mut();
            if st.busy || st.status != "ready" || st.handover.is_some() {
                return err(409, "the agent is busy or not ready");
            }
            st.busy = true;
        }
        let run = py::uuid4_hex()[..12].to_string();
        let host = self.clone();
        let r = run.clone();
        tokio::task::spawn_local(async move { host.run_review(path, source, r, reviewer).await });
        jr(202, json!({"ok": true, "run": run}))
    }

    async fn at_work(self: &Rc<Self>, choice: AgentChoice) -> bool {
        if !self.take_seat(&choice) {
            self.clone().launch(choice).await;
        }
        self.status() == "ready"
    }

    async fn run_review(self: Rc<Self>, path: String, source: String, run: String, reviewer: Option<AgentChoice>) {
        let Some(reviewer) = reviewer else { return self.review(path, source, run).await };
        let back = self.current();
        if !self.at_work(reviewer.clone()).await {
            self.emit(json!({"kind": "review", "state": "failed", "path": path, "run": run, "agent": reviewer.label(),
                             "text": format!("{} did not start", reviewer.name)}));
        } else {
            self.clone().review(path, source, run).await;
        }
        self.st.borrow_mut().busy = true;
        if self.current().name != back.name {
            self.at_work(back).await;
        }
        self.st.borrow_mut().busy = false;
    }

    async fn review(self: Rc<Self>, path: String, source: String, run: String) {
        let agent = self.current().label();
        self.emit(json!({"kind": "review", "state": "started", "path": path, "run": run, "agent": agent}));
        let at = self.st.borrow().events.len();
        self.st.borrow_mut().busy = false;
        self.clone().run_prompt(review::prompt(&path, &source), None, Some(format!("Review {path}"))).await;
        let (answer, failed) = {
            let st = self.st.borrow();
            let since = &st.events[at.min(st.events.len())..];
            let answer: String = since.iter().filter(|e| e.get("kind") == Some(&json!("message"))).map(|e| s_of(e, "text")).collect();
            let failed = since.iter().find(|e| e.get("kind") == Some(&json!("error"))).map(|e| e.get("text").cloned().unwrap_or(Value::Null));
            (answer, failed)
        };
        let (notes, cut) = review::notes_of(&answer);
        let read = review::read_text(&self.cfg.folder, &path);
        let (Some(notes), Some((_, text))) = (notes, read.clone()) else {
            let said = py::split_ws(&answer).join(" ");
            let why = match failed.filter(|f| py::truthy(Some(f))) {
                Some(f) => f,
                None => json!(if read.is_none() { "the document's text is gone".to_string() }
                              else if (1..=400).contains(&py::clen(&said)) { said }
                              else { "the agent did not answer with notes".to_string() }),
            };
            self.emit(json!({"kind": "review", "state": "failed", "path": path, "run": run, "agent": agent, "text": why}));
            return;
        };
        let model = self.model_name();
        let mut creator = json!({"type": "Software", "name": agent});
        if !model.is_empty() {
            creator["model"] = json!(model);
        }
        let quotes: Vec<String> = notes.iter().map(|n| n["quote"].as_str().unwrap_or("").to_string()).collect();
        let found = review::find_quotes(&text, &quotes, Some(&self.proxy.engine(&self.cfg.scope)));
        let kept = match review::keep(&self.cfg.folder, &path, &text, &notes, &creator, &run, Some(&found)) {
            Ok(k) => k,
            Err(e) => {
                log::error(&format!("the review's notes were not kept: {e}"));
                return;
            }
        };
        let mut e = json!({"kind": "review", "state": "done", "path": path, "run": run, "agent": agent, "model": model,
                           "kept": kept["kept"].as_array().map_or(0, Vec::len), "refused": kept["refused"]});
        if cut {
            e["cut"] = json!(true);
        }
        self.emit(e);
    }

    async fn set_model(self: Rc<Self>, value: String) -> Response {
        let (models, agent, sid, busy) = {
            let mut st = self.st.borrow_mut();
            let busy = st.busy;
            let seat = st.seat();
            (seat.models.clone(), seat.agent.clone(), seat.session_id.clone(), busy)
        };
        let (Some(models), Some(agent), Some(sid)) = (models.filter(|m| py::truthy(Some(m))), agent, sid.filter(|s| !s.is_empty())) else {
            return err(409, "the agent offers no choice of model");
        };
        if busy {
            return err(409, "the agent is working: change the model between tasks");
        }
        let offered = models["options"].as_array().into_iter().flatten().any(|o| o["value"] == json!(value))
            || models.get("current") == Some(&json!(value));
        if !offered {
            return err(400, "the agent offers no such model");
        }
        let with_current = |m: &Value| {
            let mut m = m.clone();
            m["current"] = json!(value);
            m
        };
        let choice = if models["via"] == json!("config") {
            match agent.set_config_option(&sid, &models["id"], &value).await {
                Ok(options) => model_choice(&json!({"configOptions": options})).unwrap_or_else(|| with_current(&models)),
                Err(e) => return err(502, &format!("the model was not changed: {e}")),
            }
        } else {
            match agent.set_model(&sid, &value).await {
                Ok(()) => with_current(&models),
                Err(e) => return err(502, &format!("the model was not changed: {e}")),
            }
        };
        self.st.borrow_mut().seat().models = Some(choice.clone());
        self.emit_models(Some(&choice));
        jr(200, json!({"ok": true, "models": choice}))
    }

    async fn set_agent(self: Rc<Self>, value: String, handover: bool) -> Response {
        let (choice, n, current, hand, busy, status) = {
            let st = self.st.borrow();
            (st.agents.iter().find(|a| a.name == value).cloned(), st.agents.len(), st.current.clone(), st.handover.clone(), st.busy,
             st.status.clone())
        };
        let Some(choice) = choice.filter(|_| n >= 2) else { return err(400, "there is no such agent") };
        if let Some(why) = &choice.unavailable {
            return err(409, &format!("{} cannot start here: {why}", choice.name));
        }
        if let Some(h) = hand {
            return err(409, &format!("the task is being handed over to {}", h.name));
        }
        if busy {
            if !handover {
                return err(409, "the agent is working: switch between tasks, or hand the task over");
            }
            if choice.name == current.name {
                return jr(200, json!({"ok": true, "agents": self.agent_choice()}));
            }
            self.st.borrow_mut().handover = Some(choice.clone());
            self.emit(json!({"kind": "notice", "handing": true,
                             "text": format!("Handing the task over to {}: {} stops at its current step.", choice.name, current.name)}));
            self.stop_turn().await;
            let host = self.clone();
            let seat = self.current().name;
            tokio::task::spawn_local(async move { host.stop_if_stuck(seat, Duration::from_secs(30)).await });
            return jr(202, json!({"ok": true, "handover": choice.name, "agents": self.agent_choice()}));
        }
        if status == "starting" {
            return err(409, "the agent is starting");
        }
        if choice.name == current.name && status == "ready" {
            return jr(200, json!({"ok": true, "agents": self.agent_choice()}));
        }
        if !self.take_seat(&choice) {
            let host = self.clone();
            tokio::task::spawn_local(async move { host.launch(choice).await });
        }
        jr(202, json!({"ok": true, "agents": self.agent_choice()}))
    }

    fn drop_codex_dir(&self, name: &str) {
        let dir = self.st.borrow_mut().seats.iter_mut().find(|(n, _)| n == name).and_then(|(_, s)| s.codex_dir.take());
        if let Some(d) = dir {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    fn drop_claude_dir(&self, name: &str) {
        let dir = self.st.borrow_mut().seats.iter_mut().find(|(n, _)| n == name).and_then(|(_, s)| s.claude_dir.take());
        if let Some(d) = dir {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// Claude Code confined whole (`agents::confine`): asked for, the launcher at hand, and Seatbelt here.
    pub fn whole_claude(&self) -> bool {
        self.cfg.claude_confine && self.cfg.claude_shell.is_some() && confine::supported()
    }

    /// The network of a confined agent: deid-agent's port, and the sub-agents' servers when it has them.
    fn agent_network(&self, server: Option<&(String, Vec<String>)>) -> Vec<String> {
        let servers = match (&self.cfg.subagents, server) {
            (Some(hosts), Some(_)) => confine::server_addresses(hosts),
            _ => Vec::new(),
        };
        confine::network(self.cfg.port, &servers)
    }

    /// Claude Code laid out confined for `choice`, in a folder made anew for this start.
    fn confine_claude(&self, choice: &AgentChoice, server: Option<&(String, Vec<String>)>)
                      -> Result<confine::ConfinedClaude, confine::ConfinementUnavailable> {
        self.drop_claude_dir(&choice.name);
        let dir = make_temp_dir("deid-claude-").ok_or_else(|| confine::ConfinementUnavailable("no folder for its profile".into()))?;
        self.st.borrow_mut().seat_of(&choice.name, choice.clone()).claude_dir = Some(dir.clone());
        // agent-guard runs as Claude Code's hook: its program, the person's policy for it, this matter's journal
        let mut read: Vec<String> = self.cfg.guard.clone();
        if !self.cfg.guard.is_empty() {
            read.push(py::expanduser("~/.agent-guard/environment.json").to_string_lossy().into_owned());
        }
        read.extend(server.map(|s| self.subagent_paths(s)).unwrap_or_default());
        let write: Vec<String> = self.cfg.guard_log.iter().map(|l| l.to_string_lossy().into_owned()).collect();
        let launcher = self.cfg.claude_shell.as_deref().ok_or_else(|| confine::ConfinementUnavailable("no launcher".into()))?;
        confine::confined_claude("~", &self.cfg.folder.to_string_lossy(), self.cfg.claude_executable.as_deref(), launcher, &dir,
                                 &self.agent_network(server), &read, &write)
    }

    /// Every agent stopped and every confined Codex's folder removed: the server is going away.
    /// The agents stopped, all at once (each may take its 5 s), and Codex's folders dropped.
    pub async fn cleanup(&self) {
        let seats: Vec<(String, Option<Rc<AcpAgent>>)> = self.st.borrow().seats.iter().map(|(n, s)| (n.clone(), s.agent.clone())).collect();
        let stopping: Vec<_> = seats.iter().filter_map(|(_, a)| a.clone())
            .map(|a| tokio::task::spawn_local(async move { a.stop().await })).collect();
        for s in stopping {
            let _ = s.await;
        }
        for (name, _) in seats {
            self.drop_codex_dir(&name);
            self.drop_claude_dir(&name);
        }
    }
}

fn agent_name(info: Option<&Value>) -> Value {
    let ai = info.and_then(|i| i.get("agentInfo")).filter(|a| py::truthy(Some(a))).cloned().unwrap_or(json!({}));
    ai.get("name").cloned().unwrap_or(json!("agent"))
}

/// `isinstance(x, int)` (a bool is one) as its value.
fn int_value(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.as_i64(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// `tempfile.mkdtemp(prefix=…)`: a new folder only its owner can enter.
fn make_temp_dir(prefix: &str) -> Option<PathBuf> {
    use std::os::unix::fs::DirBuilderExt;
    let base = std::env::temp_dir();
    for _ in 0..100 {
        let name: String = py::random_bytes(8).iter().map(|b| b"abcdefghijklmnopqrstuvwxyz0123456789_"[*b as usize % 37] as char).collect();
        let p = base.join(format!("{prefix}{name}"));
        if std::fs::DirBuilder::new().mode(0o700).create(&p).is_ok() {
            return Some(p);
        }
    }
    None
}

/// Run a program and read its output, giving up after `timeout`.
fn run_with_timeout(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = std::process::Command::new(program).args(args).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null()).spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < timeout => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                return None;
            }
        }
    }
    let mut out = String::new();
    use std::io::Read;
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    Some(out)
}
