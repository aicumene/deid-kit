// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A client for the Agent Client Protocol (ACP, version 1): a coding agent as a child process,
//! JSON-RPC 2.0 over its stdin and stdout, one message a line (`deidkit/agents/acp.py`). The client
//! advertises neither file-system nor terminal capabilities; every permission request is decided by
//! `on_permission`, which the host hands to a person.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::rc::Rc;
use std::time::Duration;

use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;

use crate::{log, py};

pub const PROTOCOL_VERSION: i64 = 1;

pub type OnUpdate = Rc<dyn Fn(Value)>;
pub type OnPermission = Rc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Value>>>>;

/// An error of the agent, or of talking to it, worded as Python's `str(exc)` would have it.
#[derive(Clone, Debug)]
pub struct AcpError(pub String);

impl std::fmt::Display for AcpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `AcpError(error)`: "ACP error <code>: <message>", each as Python's str() prints it.
fn rpc_error(error: &Value) -> AcpError {
    let get = |k: &str| error.get(k).map_or("None".to_string(), py::py_str);
    AcpError(format!("ACP error {}: {}", get("code"), get("message")))
}

/// An OSError as Python prints it: `[Errno 2] No such file or directory: 'name'`.
pub fn os_error(e: &std::io::Error, filename: Option<&str>) -> String {
    match e.raw_os_error() {
        Some(n) => {
            let text = unsafe { std::ffi::CStr::from_ptr(libc::strerror(n)) }.to_string_lossy().into_owned();
            match filename {
                Some(f) => format!("[Errno {n}] {text}: {}", py::repr_str(f)),
                None => format!("[Errno {n}] {text}"),
            }
        }
        None => e.to_string(),
    }
}

type Pending = RefCell<HashMap<i64, oneshot::Sender<Result<Value, AcpError>>>>;

/// One agent process and its JSON-RPC connection.
pub struct AcpAgent {
    pub command: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: String,
    on_update: OnUpdate,
    on_permission: OnPermission,
    child: RefCell<Option<Child>>,
    stdin: Rc<tokio::sync::Mutex<Option<ChildStdin>>>,
    next_id: Cell<i64>,
    pending: Rc<Pending>,
    tasks: RefCell<Vec<tokio::task::JoinHandle<()>>>,
    pub agent_info: RefCell<Value>,
    /// The agent's answer to session/new.
    pub session_setup: RefCell<Value>,
}

impl AcpAgent {
    pub fn new(command: Vec<String>, env: Vec<(String, String)>, cwd: String, on_update: OnUpdate,
               on_permission: OnPermission) -> Rc<AcpAgent> {
        Rc::new(AcpAgent { command, env, cwd, on_update, on_permission, child: RefCell::new(None),
                           stdin: Rc::new(tokio::sync::Mutex::new(None)), next_id: Cell::new(0),
                           pending: Rc::new(RefCell::new(HashMap::new())), tasks: RefCell::new(Vec::new()),
                           agent_info: RefCell::new(json!({})), session_setup: RefCell::new(json!({})) })
    }

    /// The process, while it has not exited.
    pub fn running(&self) -> bool {
        self.child.borrow_mut().as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)))
    }

    pub async fn start(self: &Rc<Self>) -> Result<Value, AcpError> {
        let program = self.command.first().cloned().unwrap_or_default();
        let mut cmd = Command::new(&program);
        cmd.args(&self.command[1.min(self.command.len())..]).current_dir(&self.cwd).env_clear()
            .envs(self.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(false);
        let mut child = cmd.spawn().map_err(|e| {
            // asyncio names the program when it is not found, the folder when that is what is missing
            let missing_cwd = !std::path::Path::new(&self.cwd).is_dir();
            AcpError(os_error(&e, Some(if missing_cwd { &self.cwd } else { &program })))
        })?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        *self.stdin.lock().await = child.stdin.take();
        *self.child.borrow_mut() = Some(child);
        let me = self.clone();
        let reader = tokio::task::spawn_local(async move { me.read(stdout).await });
        let drain = tokio::task::spawn_local(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(_line)) = lines.next_line().await {}
        });
        self.tasks.borrow_mut().extend([reader, drain]);
        let info = self.request("initialize", json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false},
            "clientInfo": {"name": "deid-agent", "version": "0.1"},
        })).await?;
        *self.agent_info.borrow_mut() = info.clone();
        Ok(info)
    }

    /// A session over `cwd`; `meta` as the request's `_meta`, `mcp_servers` in ACP's stdio form.
    pub async fn new_session(&self, cwd: &str, meta: Option<&Value>, mcp_servers: Vec<Value>) -> Result<String, AcpError> {
        let mut params = json!({"cwd": cwd, "mcpServers": mcp_servers});
        if let Some(m) = meta.filter(|m| py::truthy(Some(m))) {
            params["_meta"] = m.clone();
        }
        let result = self.request("session/new", params).await?;
        *self.session_setup.borrow_mut() = if py::truthy(Some(&result)) { result.clone() } else { json!({}) };
        match &result {
            Value::Object(m) => match m.get("sessionId") {
                Some(Value::String(s)) => Ok(s.clone()),
                Some(other) => Ok(py::py_str(other)),
                None => Err(AcpError("'sessionId'".into())),
            },
            other => Err(AcpError(format!("'{}' object is not subscriptable", type_name(other)))),
        }
    }

    /// `session/set_config_option`; all the session's settings as the agent now has them.
    pub async fn set_config_option(&self, session_id: &str, config_id: &Value, value: &str) -> Result<Vec<Value>, AcpError> {
        let result = self.request("session/set_config_option",
                                  json!({"sessionId": session_id, "configId": config_id, "value": value})).await?;
        Ok(result.get("configOptions").and_then(Value::as_array).cloned().unwrap_or_default())
    }

    /// `session/set_model`, for agents that offer a `models` block.
    pub async fn set_model(&self, session_id: &str, model_id: &str) -> Result<(), AcpError> {
        self.request("session/set_model", json!({"sessionId": session_id, "modelId": model_id})).await.map(|_| ())
    }

    /// A prompt of several text blocks, in order.
    pub async fn prompt(&self, session_id: &str, blocks: &[String]) -> Result<Value, AcpError> {
        let prompt: Vec<Value> = blocks.iter().map(|t| json!({"type": "text", "text": t})).collect();
        self.request("session/prompt", json!({"sessionId": session_id, "prompt": prompt})).await
    }

    pub async fn cancel(&self, session_id: &str) -> Result<(), AcpError> {
        self.send(&json!({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}})).await
    }

    /// Terminate the process (killed after 5 s), and stop reading it.
    pub async fn stop(&self) {
        let child = self.child.borrow_mut().take();
        if let Some(mut c) = child {
            if matches!(c.try_wait(), Ok(None)) {
                if let Some(pid) = c.id() {
                    unsafe { libc::kill(pid as i32, libc::SIGTERM) };
                }
                if tokio::time::timeout(Duration::from_secs(5), c.wait()).await.is_err() {
                    let _ = c.kill().await;
                }
            }
            *self.child.borrow_mut() = Some(c);
        }
        for t in self.tasks.borrow_mut().drain(..) {
            t.abort();
        }
        self.fail_pending();
    }

    fn fail_pending(&self) {
        for (_, tx) in self.pending.borrow_mut().drain() {
            let _ = tx.send(Err(AcpError("ACP error -32000: the agent exited".into())));
        }
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value, AcpError> {
        let rid = self.next_id.get() + 1;
        self.next_id.set(rid);
        let (tx, rx) = oneshot::channel();
        self.pending.borrow_mut().insert(rid, tx);
        self.send(&json!({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})).await?;
        rx.await.unwrap_or_else(|_| Err(AcpError("ACP error -32000: the agent exited".into())))
    }

    async fn send(&self, message: &Value) -> Result<(), AcpError> {
        let line = py::dumps(message) + "\n";
        let mut guard = self.stdin.lock().await;
        let Some(stdin) = guard.as_mut() else { return Err(AcpError("[Errno 32] Broken pipe".into())) };
        let written = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        }.await;
        written.map_err(|e| AcpError(match e.kind() {
            std::io::ErrorKind::BrokenPipe => "[Errno 32] Broken pipe".into(),
            std::io::ErrorKind::ConnectionReset => "Connection lost".into(),
            _ => os_error(&e, None),
        }))
    }

    async fn read(self: Rc<Self>, stdout: tokio::process::ChildStdout) {
        let mut reader = BufReader::with_capacity(1 << 16, stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let msg: Value = match serde_json::from_slice(&line) {
                Ok(v) => v,
                Err(_) => {
                    log::warning("agent wrote a line that is not JSON-RPC");
                    continue;
                }
            };
            self.dispatch(msg).await;
        }
        self.fail_pending();
    }

    async fn dispatch(self: &Rc<Self>, msg: Value) {
        let Some(m) = msg.as_object() else { return };
        if m.contains_key("id") && (m.contains_key("result") || m.contains_key("error")) && !m.contains_key("method") {
            let id = match &m["id"] {
                Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
                Value::Bool(b) => Some(*b as i64),
                _ => None,
            };
            let tx = id.and_then(|i| self.pending.borrow_mut().remove(&i));
            if let Some(tx) = tx {
                let _ = tx.send(match m.get("error") {
                    Some(e) => Err(rpc_error(e)),
                    None => Ok(m.get("result").cloned().unwrap_or(Value::Null)),
                });
            }
            return;
        }
        let method = m.get("method").and_then(Value::as_str).unwrap_or("");
        let params = match m.get("params") {
            Some(p) if py::truthy(Some(p)) => p.clone(),
            _ => json!({}),
        };
        if method == "session/update" {
            (self.on_update)(params);
            return;
        }
        let Some(id) = m.get("id").cloned() else { return };
        if method == "session/request_permission" {
            let me = self.clone();
            let decided = (self.on_permission)(params);
            tokio::task::spawn_local(async move {
                let outcome = decided.await;
                let _ = me.send(&json!({"jsonrpc": "2.0", "id": id, "result": {"outcome": outcome}})).await;
            });
            return;
        }
        let shown = m.get("method").map_or("None".to_string(), py::py_str);
        let _ = self.send(&json!({"jsonrpc": "2.0", "id": id,
                                  "error": {"code": -32601, "message": format!("method not supported: {shown}")}})).await;
    }
}

/// Python's name for the type of a JSON value.
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The environment for a Claude agent behind the proxy: a minimal inherited base, the proxy as its only
/// endpoint, and the scope; `api_key` is the stand-in it sends.
pub fn agent_env(base_url: &str, scope: &str, api_key: Option<&str>, extra: &[(String, String)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();
    let set = |env: &mut Vec<(String, String)>, k: &str, v: &str| {
        match env.iter_mut().find(|(x, _)| x == k) {
            Some(slot) => slot.1 = v.to_string(),
            None => env.push((k.to_string(), v.to_string())),
        }
    };
    for k in ["HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LANG", "SHELL", "TERM"] {
        if let Ok(v) = std::env::var(k) {
            set(&mut env, k, &v);
        }
    }
    set(&mut env, "ANTHROPIC_BASE_URL", base_url);
    set(&mut env, "ANTHROPIC_CUSTOM_HEADERS", &format!("x-deid-scope: {scope}"));
    set(&mut env, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        set(&mut env, "ANTHROPIC_API_KEY", key);
        set(&mut env, "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", "1");
    }
    for (k, v) in extra {
        set(&mut env, k, v);
    }
    env
}

/// What Codex behind the proxy needs, through its ACP adapter: the proxy as its model provider.
pub fn codex_env(base_url: &str, scope: &str, codex_path: Option<&str>, codex_home: Option<&str>) -> Vec<(String, String)> {
    let mut provider = Map::new();
    provider.insert("name".into(), json!("OpenAI through deid-agent"));
    provider.insert("base_url".into(), json!(format!("{}/v1", base_url.trim_end_matches('/'))));
    provider.insert("requires_openai_auth".into(), json!(true));
    provider.insert("wire_api".into(), json!("responses"));
    provider.insert("http_headers".into(), json!({"x-deid-scope": scope}));
    let config = json!({"model_providers": {"deid": provider}, "features": {"enable_request_compression": false}});
    let mut env = vec![("MODEL_PROVIDER".to_string(), "deid".to_string()),
                       ("CODEX_CONFIG".to_string(), py::dumps_ascii(&config)),
                       ("INITIAL_AGENT_MODE".to_string(), "read-only".to_string())];
    if let Some(p) = codex_path.filter(|p| !p.is_empty()) {
        env.push(("CODEX_PATH".into(), p.into()));
    }
    if let Some(h) = codex_home.filter(|h| !h.is_empty()) {
        env.push(("CODEX_HOME".into(), h.into()));
    }
    env
}
