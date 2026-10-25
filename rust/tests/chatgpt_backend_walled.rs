// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A Codex deid-agent does not confine, walled in all the same by what runs deid-agent (`--agents-walled`: a
//! container's firewall), reaches ChatGPT's own backend through deid-agent, as a confined Codex does
//! (tests/chatgpt_backend.rs): its wrapper starts it with deid-agent's `/backend-api/` as its `chatgpt_base_url`, in a
//! folder another user may pass through (the container's `codex`, by sudo), deid-agent its model provider from the start
//! (its list of models, asked for before any session, comes here too). Without the walls an unconfined Codex is
//! started as it is. End to end with deid-agent itself and the scripted agent tests/fake_acp_agent.py as Codex.
//! Needs python3.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const TOKEN: &str = "walled-codex-test-token";
const CODEX: &str = "/opt/codex/bin/codex";

struct Agent {
    child: Child,
    port: u16,
    root: PathBuf,
}

impl Drop for Agent {
    fn drop(&mut self) {
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// deid-agent with the scripted agent as its Codex, unconfined; `walled` adds `--agents-walled`.
fn start(name: &str, walled: bool) -> Agent {
    let root = std::env::temp_dir().join(format!("deid-walled-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("matter")).unwrap();
    std::fs::create_dir_all(root.join("home")).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let agent = format!("python3 {} Codex codex-acp", Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fake_acp_agent.py").display());
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_deid-agent"));
    cmd.args(["--folder", root.join("matter").to_str().unwrap(), "--scope", "test-walled", "--port", &port.to_string()])
        .args(["--vault", root.join("agent/vault.sqlite").to_str().unwrap()])
        .args(["--audit", root.join("agent/audit.jsonl").to_str().unwrap(), "--usage-log", ""])
        .args(["--agent-command", &agent, "--agent", &format!("Codex={agent}")])
        .args(["--codex-unconfined", "--codex-executable", CODEX, "--codex-home", root.join("codex").to_str().unwrap()])
        .args(["--secrets-stdin", "--no-guard", "--dev-login"])
        .env("HOME", root.join("home")).env_remove("CODEX_HOME").env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    if walled {
        cmd.arg("--agents-walled");
    }
    let mut child = cmd.spawn().unwrap();
    writeln!(child.stdin.take().unwrap(), "{}", json!({"token": TOKEN})).unwrap();
    let agent = Agent { child, port, root };
    let until = Instant::now() + Duration::from_secs(30);
    while Instant::now() < until {
        if let Some((200, state)) = request(port, "GET", "/deid-agent/api/state", None) {
            if state["status"] == "ready" {
                return agent;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("deid-agent did not get ready");
}

fn request(port: u16, method: &str, path: &str, body: Option<&Value>) -> Option<(u16, Value)> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(30))).ok()?;
    let data = body.map(Value::to_string).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\nContent-Type: application/json\r\n\
Content-Length: {}\r\n\r\n{data}", data.len()).ok()?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).ok()?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n")?;
    let status = head.split(' ').nth(1)?.parse().ok()?;
    Some((status, serde_json::from_str(rest).unwrap_or(Value::String(rest.to_string()))))
}

/// What the scripted Codex says of how it was started: its answer to a prompt (its CODEX_PATH, and the wrapper's
/// text when CODEX_PATH is a file).
fn said(a: &Agent) -> String {
    let (tx, rx) = mpsc::channel();
    let mut s = TcpStream::connect(("127.0.0.1", a.port)).unwrap();
    write!(s, "GET /deid-agent/api/events HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\n\r\n").unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(s).lines().map_while(Result::ok) {
            if let Some(data) = line.strip_prefix("data: ") {
                if tx.send(serde_json::from_str::<Value>(data).unwrap()).is_err() {
                    break;
                }
            }
        }
    });
    let (status, _) = request(a.port, "POST", "/deid-agent/api/prompt", Some(&json!({"text": "Go"}))).unwrap();
    assert_eq!(status, 202);
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let e = rx.recv_timeout(until.saturating_duration_since(Instant::now())).expect("the agent's answer");
        if let Some(text) = e["text"].as_str().filter(|t| e["kind"] == "message" && t.contains("[provider ")) {
            return text.to_string();
        }
    }
}

/// CODEX_PATH, from `[provider NAME CONFIG PATH KEY MODE]` (the config has spaces of its own).
fn codex_path(said: &str) -> String {
    let provider = said.split("[provider ").nth(1).unwrap().split("] ").next().unwrap();
    provider.rsplitn(4, ' ').nth(2).unwrap().to_string()
}

#[test]
fn a_walled_in_codex_reaches_chatgpts_backend_through_deid_agent() {
    let a = start("on", true);
    let said = said(&a);
    let path = PathBuf::from(codex_path(&said));
    assert_ne!(path, Path::new(CODEX), "Codex starts through a wrapper: {said}");
    let wrapper = std::fs::read_to_string(&path).unwrap();
    assert!(wrapper.contains(&format!("chatgpt_base_url=\"http://127.0.0.1:{}/backend-api/\"", a.port)), "{wrapper}");
    // its list of models, asked for at its start, comes here too: deid-agent is its provider from the start
    assert!(wrapper.contains("'model_provider=\"deid\"'"), "{wrapper}");
    assert!(wrapper.contains(&format!("'model_providers.deid.base_url=\"http://127.0.0.1:{}/v1\"'", a.port)), "{wrapper}");
    assert!(wrapper.contains("'model_providers.deid.http_headers={\"x-deid-scope\" = \"test-walled\"}'"), "{wrapper}");
    assert!(wrapper.contains(CODEX), "the wrapper starts the Codex given: {wrapper}");
    // the container's codex is not the user deid-agent runs as: it passes through the folder and reads the wrapper
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(path.parent().unwrap()), 0o711);
    assert_eq!(mode(&path) & 0o005, 0o005);
}

#[test]
fn an_unconfined_codex_without_walls_is_started_as_it_is() {
    let a = start("off", false);
    let said = said(&a);
    assert_eq!(codex_path(&said), CODEX);
    assert!(!said.contains("chatgpt_base_url"), "{said}");
}
