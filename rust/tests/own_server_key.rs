// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The key of an agent's own model server (`upstream_keys` of `--secrets-stdin`), end to end with deid-agent itself and
//! the scripted agent tests/fake_acp_agent.py: the proxy holds the key and the agent gets a stand-in, as with the
//! organization's key; only the agent's requests reach its server with the key, as both headers an own server may read
//! (`Authorization: Bearer` and `x-api-key`); a key the server refuses is told as that server's. An own server without
//! a key gets the agent's requests too and no one else's, and no credential a request brings. Needs python3.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const TOKEN: &str = "own-server-test-token";
const SERVER_KEY: &str = "own-server-key-0123456789abcdef";

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// What the stand-in server was sent: the path and the two headers a key may come in.
#[derive(Clone, Debug)]
struct Seen {
    path: String,
    authorization: Option<String>,
    api_key: Option<String>,
}

/// A stand-in for the agent's own server: `keyed`, it takes `SERVER_KEY` in either header, else answers 401 as
/// llama-server does; not, it takes any request.
fn server(keyed: bool) -> (u16, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut first = String::new();
            r.read_line(&mut first).unwrap_or(0);
            let (mut length, mut authorization, mut api_key) = (0usize, None, None);
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                let (name, value) = line.split_once(':').map(|(n, v)| (n.trim().to_lowercase(), v.trim().to_string())).unwrap_or_default();
                match name.as_str() {
                    "content-length" => length = value.parse().unwrap_or(0),
                    "authorization" => authorization = Some(value),
                    "x-api-key" => api_key = Some(value),
                    _ => {}
                }
            }
            let mut body = vec![0u8; length];
            r.read_exact(&mut body).unwrap();
            let path = first.split_whitespace().nth(1).unwrap_or("").to_string();
            let ok = !keyed || authorization.as_deref() == Some(&format!("Bearer {SERVER_KEY}")) || api_key.as_deref() == Some(SERVER_KEY);
            s.lock().unwrap().push(Seen { path: path.clone(), authorization, api_key });
            let (status, answer) = match (ok, path.as_str()) {
                (false, _) => ("401 Unauthorized", json!({"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})),
                (true, "/v1/models") => ("200 OK", json!({"object": "list", "data": [{"id": "local-model", "meta": {"n_ctx": 32768}}]})),
                (true, _) => ("200 OK", json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "local-model",
                                               "content": [{"type": "text", "text": "Noted."}], "stop_reason": "end_turn",
                                               "usage": {"input_tokens": 10, "output_tokens": 2}})),
            };
            let answer = answer.to_string();
            let mut conn = conn;
            let _ = write!(conn, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                           answer.len());
        }
    });
    (port, seen)
}

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

fn command(name: &str, upstream: u16) -> (Command, PathBuf, u16) {
    let root = std::env::temp_dir().join(format!("deid-own-key-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("matter")).unwrap();
    std::fs::create_dir_all(root.join("home")).unwrap();
    let port = free_port();
    let agent = format!("python3 {} Local say-key", Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fake_acp_agent.py").display());
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_deid-agent"));
    cmd.args(["--folder", root.join("matter").to_str().unwrap(), "--scope", "test-own", "--port", &port.to_string()])
        .args(["--vault", root.join("agent/vault.sqlite").to_str().unwrap()])
        .args(["--audit", root.join("agent/audit.jsonl").to_str().unwrap(), "--usage-log", ""])
        .args(["--agent-command", &agent, "--agent", &format!("Local={agent}")])
        .args(["--agent-upstream", &format!("Local=http://127.0.0.1:{upstream}"), "--secrets-stdin", "--no-guard", "--dev-login"])
        .env("HOME", root.join("home")).env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
    (cmd, root, port)
}

/// deid-agent with `Local` on its own server at `upstream`, the server's key `key` (if any) in the secrets on stdin.
fn start(name: &str, upstream: u16, key: &str) -> Agent {
    let (mut cmd, root, port) = command(name, upstream);
    let mut child = cmd.spawn().unwrap();
    let secrets = if key.is_empty() { json!({"token": TOKEN}) } else { json!({"token": TOKEN, "upstream_keys": {"Local": key}}) };
    writeln!(child.stdin.take().unwrap(), "{secrets}").unwrap();
    let agent = Agent { child, port, root };
    let until = Instant::now() + Duration::from_secs(30);
    while Instant::now() < until {
        if let Some((200, state)) = request(port, "GET", "/deid-agent/api/state", &[], None) {
            if state["status"] == "ready" {
                return agent;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("deid-agent did not get ready");
}

fn request(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&Value>) -> Option<(u16, Value)> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(30))).ok()?;
    let data = body.map(Value::to_string).unwrap_or_default();
    let extra: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
    write!(s, "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\nContent-Type: application/json\r\n\
anthropic-version: 2023-06-01\r\n{extra}Content-Length: {}\r\n\r\n{data}", data.len()).ok()?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).ok()?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n")?;
    let status = head.split(' ').nth(1)?.parse().ok()?;
    Some((status, serde_json::from_str(rest).unwrap_or(Value::String(rest.to_string()))))
}

fn events(port: u16) -> mpsc::Receiver<Value> {
    let (tx, rx) = mpsc::channel();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "GET /deid-agent/api/events HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\n\r\n").unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(s).lines().map_while(Result::ok) {
            if let Some(data) = line.strip_prefix("data: ") {
                if tx.send(serde_json::from_str(data).unwrap()).is_err() {
                    break;
                }
            }
        }
    });
    rx
}

fn next(rx: &mpsc::Receiver<Value>, want: impl Fn(&Value) -> bool) -> Value {
    let until = Instant::now() + Duration::from_secs(30);
    let mut seen = Vec::new();
    loop {
        let e = rx.recv_timeout(until.saturating_duration_since(Instant::now())).unwrap_or_else(|_| panic!("no such event; seen: {seen:#?}"));
        if want(&e) {
            return e;
        }
        seen.push(e);
    }
}

/// The stand-in the proxy lent the agent: the scripted agent says it in its answer (`say-key`).
fn lent(a: &Agent) -> String {
    let rx = events(a.port);
    let (status, _) = request(a.port, "POST", "/deid-agent/api/prompt", &[], Some(&json!({"text": "Go"}))).unwrap();
    assert_eq!(status, 202);
    let said = next(&rx, |e| e["kind"] == "message" && e["text"].as_str().is_some_and(|t| t.contains("[key ")));
    let text = said["text"].as_str().unwrap();
    text.split("[key ").nth(1).unwrap().split(']').next().unwrap().to_string()
}

fn ask(a: &Agent, key: &str) -> u16 {
    let body = json!({"model": "local-model", "max_tokens": 10, "messages": [{"role": "user", "content": "Hello"}]});
    request(a.port, "POST", "/v1/messages", &[("x-api-key", key)], Some(&body)).unwrap().0
}

#[test]
fn an_own_servers_key_goes_only_with_the_agents_requests_and_in_both_headers() {
    let (upstream, seen) = server(true);
    let a = start("lend", upstream, SERVER_KEY);
    let stand_in = lent(&a);
    assert!(stand_in != SERVER_KEY && stand_in.starts_with("deid-agent-"), "the agent gets a stand-in, not the key: {stand_in}");

    assert_eq!(ask(&a, &stand_in), 200, "the agent's request reaches its server with the key");
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!((last.path.as_str(), last.authorization.as_deref(), last.api_key.as_deref()),
               ("/v1/messages", Some(format!("Bearer {SERVER_KEY}").as_str()), Some(SERVER_KEY)));

    let before = seen.lock().unwrap().len();
    assert_eq!(ask(&a, "the-persons-own"), 401, "anyone else's request is refused by the proxy");
    assert_eq!(seen.lock().unwrap().len(), before, "and never reaches the server");

    let (status, models) = request(a.port, "GET", "/v1/models", &[("x-api-key", &stand_in)], None).unwrap();
    assert_eq!((status, models["data"][0]["id"].as_str()), (200, Some("local-model")), "the agent asks its server for the window too");
}

/// Without a key the agent still gets a stand-in, and the server gets only its requests, with no credential:
/// not an idle Claude Code's background request with its own sign-in.
#[test]
fn a_keyless_own_server_gets_only_the_agents_requests_and_no_ones_credentials() {
    let (upstream, seen) = server(false);
    let a = start("keyless", upstream, "");
    let stand_in = lent(&a);
    assert!(stand_in.starts_with("deid-agent-"), "the agent gets a stand-in without a key too: {stand_in}");
    let body = json!({"model": "local-model", "max_tokens": 10, "messages": [{"role": "user", "content": "Hello"}]});
    let (status, _) = request(a.port, "POST", "/v1/messages", &[("x-api-key", &stand_in), ("Authorization", "Bearer someones-own-sign-in")],
                              Some(&body)).unwrap();
    assert_eq!(status, 200);
    let last = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!((last.authorization, last.api_key), (None, None), "no credential goes to the own server");
    let before = seen.lock().unwrap().len();
    let (status, _) = request(a.port, "POST", "/v1/messages/count_tokens", &[("Authorization", "Bearer someones-own-sign-in")],
                              Some(&body)).unwrap();
    assert_eq!(status, 401, "a request without the stand-in is refused");
    assert_eq!(seen.lock().unwrap().len(), before, "and never reaches the server");
}

#[test]
fn a_key_the_own_server_refuses_is_told_as_that_servers() {
    let (upstream, _) = server(true);
    let a = start("refused", upstream, "a-wrong-key");
    let rx = events(a.port);
    let stand_in = lent(&a);
    assert_eq!(ask(&a, &stand_in), 401);
    let told = next(&rx, |e| e["kind"] == "error");
    let text = told["text"].as_str().unwrap();
    assert!(text.starts_with("Local's model server refuse") && text.contains("its key") && !text.contains("Anthropic"), "{text}");
}

#[test]
fn a_key_for_an_agent_without_its_own_server_stops_the_start() {
    let (mut cmd, root, _) = command("nobody", free_port());
    let mut child = cmd.spawn().unwrap();
    writeln!(child.stdin.take().unwrap(), "{}", json!({"token": TOKEN, "upstream_keys": {"Nobody": "k"}})).unwrap();
    let out = child.wait_with_output().unwrap();
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && said.contains("upstream_keys names no agent with --agent-upstream: Nobody"), "{said}");
    let _ = std::fs::remove_dir_all(root);
}
