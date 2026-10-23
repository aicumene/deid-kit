// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The local HTTP proxy between a coding agent and its model (as `server.py`). Claude Code's Messages API
//! (`POST /v1/messages`, `/v1/messages/count_tokens`) goes to `upstream`; Codex's Responses API
//! (`POST /v1/responses`) to the ChatGPT backend or OpenAI. Both are de-identified on the way out and
//! re-identified on the way back; GET and HEAD pass as they are; a WebSocket upgrade is answered 426;
//! anything else is refused. Every crossing is audited (identifiers and counts, never content). A held
//! provider key is lent only to the agent's stand-in.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Duration;

use bytes::Bytes;
use serde_json::{json, Map, Value};

use crate::detect::PiiDetector;
use crate::gateway::content_hash;
use crate::http::{self, Headers, Request, Response};
use crate::odict::ODict;
use crate::proxy::anthropic::{self, PrepareError, StreamRestorer};
use crate::proxy::engine::ScopeEngine;
use crate::proxy::openai::{self, ResponsesRestorer};
use crate::proxy::sse;
use crate::pyre::{self, Regex, NONE};
use crate::seeds::SeedSource;
use crate::store::TokenStore;
use crate::{log, py};

const HOP: [&str; 13] = ["host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
    "proxy-authorization", "proxy-connection", "accept-encoding", "content-encoding", "x-deid-scope"];
/// The path of ChatGPT's backend: `chatgpt_base_url` of a confined Codex is deid-agent's own, under it.
pub const CHATGPT_BACKEND: &str = "/backend-api/";
const BACK_DROP: [&str; 5] =["content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"];

pub struct ProxyConfig {
    pub default_scope: String,
    pub upstream: String,
    pub upstream_openai: String,
    pub upstream_chatgpt: String,
    pub openai_local_tools: HashSet<String>,
    pub binary: String,
    pub local_tools: HashSet<String>,
    pub record: Option<PathBuf>,
    pub audit: Option<PathBuf>,
    pub detector: Option<Rc<dyn PiiDetector>>,
    pub seeds: Option<Rc<dyn SeedSource>>,
    /// `(folder, scope)` pairs, longest folder first.
    pub scope_paths: Vec<(String, String)>,
    pub require_scope: bool,
    /// The provider key, held here and lent only to the agent's stand-in (Anthropic only).
    pub upstream_key: Option<String>,
    pub agent_key: Option<String>,
    /// Tool namespaces taken out of an OpenAI request.
    pub drop_tool_namespaces: HashSet<String>,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        ProxyConfig { default_scope: "default".into(), upstream: "https://api.anthropic.com".into(),
                      upstream_openai: "https://api.openai.com/v1".into(), upstream_chatgpt: "https://chatgpt.com/backend-api/codex".into(),
                      openai_local_tools: openai::local_tools(), binary: "withhold".into(), local_tools: anthropic::local_tools(),
                      record: None, audit: None, detector: None, seeds: None, scope_paths: Vec::new(), require_scope: false,
                      upstream_key: None, agent_key: None, drop_tool_namespaces: HashSet::new() }
    }
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// One JSON line appended; a new file readable by its owner only.
pub fn append(path: &std::path::Path, obj: &Value) {
    use std::io::Write;
    let fresh = !path.exists();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = fh.write_all(format!("{}\n", py::dumps(obj)).as_bytes());
    }
    if fresh {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

fn claude_cwd() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"Primary working directory: *([^\n]+)", NONE)
}

fn codex_cwd() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"<cwd>([^<]+)</cwd>", NONE)
}

fn first_group(rx: &Regex, text: &str) -> Option<String> {
    let c: Vec<char> = text.chars().collect();
    rx.search_at(&c, 0).map(|m| py::strip(&pyre::text(&c, m.group(1).unwrap())).to_string())
}

/// The folder the agent works in, as the agent states it in the request.
pub fn working_directory(body: &Value) -> Option<String> {
    let texts: Vec<String> = match body.get("system") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter(|b| b.is_object()).map(|b| b.get("text").and_then(Value::as_str).unwrap_or("").to_string()).collect(),
        _ => Vec::new(),
    };
    for t in texts {
        if let Some(cwd) = first_group(claude_cwd(), &t) {
            return Some(cwd);
        }
    }
    for item in body.get("input").and_then(Value::as_array).into_iter().flatten() {
        if !item.is_object() {
            continue;
        }
        let parts: Vec<String> = match item.get("content") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a.iter().filter(|c| c.is_object()).map(|c| c.get("text").and_then(Value::as_str).unwrap_or("").to_string()).collect(),
            _ => Vec::new(),
        };
        for t in parts {
            if let Some(cwd) = first_group(codex_cwd(), &t) {
                return Some(cwd);
            }
        }
    }
    None
}

pub fn scope_for_folder(folder: Option<&str>, scope_paths: &[(String, String)]) -> Option<String> {
    let folder = folder.filter(|f| !f.is_empty())?;
    let real = py::realpath(&py::expanduser(folder)).to_string_lossy().into_owned();
    for (root, scope) in scope_paths {
        if real == *root || real.starts_with(&format!("{}/", root.trim_end_matches('/'))) {
            return Some(scope.clone());
        }
    }
    None
}

pub fn error(status: u16, message: &str) -> Response {
    let kind = if status < 500 { "invalid_request_error" } else { "api_error" };
    Response::json(status, &json!({"type": "error", "error": {"type": kind, "message": message}}))
}

/// The model's own words, kept by the store's restore table or in memory.
pub trait RestoreTable {
    fn get(&self, key: &str) -> Option<String>;
    fn put(&self, key: &str, value: &str);
}

#[derive(Default)]
pub struct MemoryRestore(RefCell<HashMap<String, String>>);

impl RestoreTable for MemoryRestore {
    fn get(&self, key: &str) -> Option<String> {
        self.0.borrow().get(key).cloned()
    }
    fn put(&self, key: &str, value: &str) {
        self.0.borrow_mut().insert(key.into(), value.into());
    }
}

pub struct StoreRestore(pub Rc<dyn TokenStore>);

impl RestoreTable for StoreRestore {
    fn get(&self, key: &str) -> Option<String> {
        self.0.restore_get(key)
    }
    fn put(&self, key: &str, value: &str) {
        self.0.restore_put(key, value)
    }
}

pub type UsageHook = Box<dyn Fn(&str, &Map<String, Value>)>;
pub type CrossingHook = Box<dyn Fn(&Value, &str)>;

/// Where requests go and with which key: set from the configuration, changed by deid-agent at each
/// switch of agents (`AgentHost.route`).
#[derive(Clone)]
pub struct Route {
    pub upstream: String,
    pub upstream_openai: String,
    pub upstream_key: Option<String>,
    pub drop_tool_namespaces: HashSet<String>,
}

pub struct Proxy {
    pub store: Rc<dyn TokenStore>,
    pub restore: Rc<dyn RestoreTable>,
    pub cfg: ProxyConfig,
    /// `cfg.upstream`, `cfg.upstream_openai`, `cfg.upstream_key` and `cfg.drop_tool_namespaces` as they
    /// are now: read from here, not from `cfg`.
    pub route: RefCell<Route>,
    engines: RefCell<HashMap<String, Rc<ScopeEngine>>>,
    /// Audit records when no file is set (`InMemoryAuditSink`).
    pub audited: RefCell<Vec<Value>>,
    /// Called when the provider refuses the held key (401).
    pub on_key_refused: RefCell<Option<Box<dyn Fn()>>>,
    /// Called after each answer of the Messages API with the model and the answer's usage.
    pub on_usage: RefCell<Option<UsageHook>>,
    /// Called for each request about to go to the model, with its audit line and the bytes sent.
    pub on_crossing: RefCell<Option<CrossingHook>>,
}

const CONNECT: Duration = Duration::from_secs(30);
const READ: Duration = Duration::from_secs(900);

fn dedupe(headers: Vec<(String, String)>) -> Headers {
    // `{k: v for k, v in headers.items()}`: the last value of a name, in the place of its first
    let mut out: Headers = Vec::new();
    for (k, v) in headers {
        match out.iter_mut().find(|(x, _)| x.eq_ignore_ascii_case(&k)) {
            Some(slot) => slot.1 = v,
            None => out.push((k, v)),
        }
    }
    out
}

fn back_headers(up: &http::Upstream) -> Headers {
    dedupe(up.headers.iter().filter(|(k, _)| !BACK_DROP.contains(&k.to_lowercase().as_str())).cloned().collect())
}

impl Proxy {
    pub fn new(store: Rc<dyn TokenStore>, restore: Rc<dyn RestoreTable>, cfg: ProxyConfig) -> Rc<Proxy> {
        let route = Route { upstream: cfg.upstream.clone(), upstream_openai: cfg.upstream_openai.clone(),
                            upstream_key: cfg.upstream_key.clone(), drop_tool_namespaces: cfg.drop_tool_namespaces.clone() };
        Rc::new(Proxy { store, restore, cfg, route: RefCell::new(route), engines: RefCell::new(HashMap::new()), audited: RefCell::new(Vec::new()),
                        on_key_refused: RefCell::new(None), on_usage: RefCell::new(None), on_crossing: RefCell::new(None) })
    }

    /// The request's scope: its header, else the listed folder the agent works in, else the default.
    pub fn scope(&self, req: &Request, body: &Value) -> Option<String> {
        if let Some(h) = req.header("x-deid-scope").filter(|h| !h.is_empty()) {
            return Some(h.to_string());
        }
        if let Some(found) = scope_for_folder(working_directory(body).as_deref(), &self.cfg.scope_paths) {
            return Some(found);
        }
        (!self.cfg.require_scope).then(|| self.cfg.default_scope.clone())
    }

    pub fn engine(&self, scope: &str) -> Rc<ScopeEngine> {
        self.engines.borrow_mut().entry(scope.to_string()).or_insert_with(|| {
            Rc::new(ScopeEngine::new(self.store.clone(), scope, self.cfg.seeds.clone(), self.cfg.detector.clone(), None))
        }).clone()
    }

    fn record(&self, obj: Value) {
        if let Some(path) = &self.cfg.record {
            let mut m = Map::new();
            m.insert("at".into(), json!(now()));
            for (k, v) in obj.as_object().cloned().unwrap_or_default() {
                m.insert(k, v);
            }
            append(path, &Value::Object(m));
        }
    }

    fn headers(&self, req: &Request) -> Headers {
        let mut out: Headers = dedupe(req.headers.iter().filter(|(k, _)| !HOP.contains(&k.to_lowercase().as_str())).cloned().collect());
        out.retain(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding"));
        out.push(("Accept-Encoding".into(), "identity".into()));
        if let Some(key) = self.route.borrow().upstream_key.clone() {
            let lend = self.from_agent(req);
            out.retain(|(k, _)| !k.eq_ignore_ascii_case("x-api-key") && !k.eq_ignore_ascii_case("authorization"));
            if lend {
                out.push(("x-api-key".into(), key));
            }
        }
        out
    }

    fn from_agent(&self, req: &Request) -> bool {
        let given = req.header("x-api-key").unwrap_or("").as_bytes();
        let Some(agent) = &self.cfg.agent_key else { return false };
        let want = agent.as_bytes();
        given.len() == want.len() && given.iter().zip(want).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
    }

    fn lend_key(&self, req: &Request) -> Option<Response> {
        if !self.from_agent(req) {
            if req.method == "GET" || req.method == "HEAD" {
                return None;
            }
            log::warning(&format!("refused {} {}: not the agent's key", req.method, req.path));
            return Some(error(401, "deid proxy: this proxy holds the organization's key and lends it only to its own agent"));
        }
        if req.path.ends_with("/responses") || !self.url(req).starts_with(self.route.borrow().upstream.trim_end_matches('/')) {
            return Some(error(403, "deid proxy: the organization's key is for Anthropic only"));
        }
        None
    }

    fn openai_base(&self, req: &Request) -> String {
        let chatgpt = req.header("chatgpt-account-id").is_some();
        if chatgpt { self.cfg.upstream_chatgpt.trim_end_matches('/').to_string() } else { self.route.borrow().upstream_openai.trim_end_matches('/').to_string() }
    }

    /// Where a request goes: Anthropic for Claude Code, OpenAI or ChatGPT for Codex — and ChatGPT's backend itself for
    /// Codex's own calls there (its plan, its limits, its workspace's policies), which a confined Codex makes here.
    pub fn url(&self, req: &Request) -> String {
        if req.path.starts_with(CHATGPT_BACKEND) {
            let base = &self.cfg.upstream_chatgpt;
            let origin = base.find(CHATGPT_BACKEND).map_or(base.trim_end_matches('/'), |i| &base[..i]);
            return origin.to_string() + &req.path_qs;
        }
        if req.path.ends_with("/responses") || req.header("chatgpt-account-id").is_some() || req.header("originator").is_some() {
            let path = req.path_qs.strip_prefix("/v1").unwrap_or(&req.path_qs);
            return self.openai_base(req) + path;
        }
        self.route.borrow().upstream.trim_end_matches('/').to_string() + &req.path_qs
    }

    pub async fn handle(self: Rc<Self>, req: Request) -> Response {
        if req.path == "/deid/health" {
            return Response::json(200, &json!({"ok": true, "service": "deid-proxy"}));
        }
        if req.header("upgrade").is_some_and(|u| u.eq_ignore_ascii_case("websocket")) {
            return Response::text(426, "deid proxy: HTTP only");
        }
        if self.route.borrow().upstream_key.is_some() {
            if let Some(refused) = self.lend_key(&req) {
                return refused;
            }
        }
        if req.method == "POST" && (req.path == "/v1/messages" || req.path == "/v1/messages/count_tokens") {
            return self.messages(req).await;
        }
        if req.method == "POST" && (req.path == "/v1/responses" || req.path == "/responses") {
            return self.responses(req).await;
        }
        if req.method == "GET" || req.method == "HEAD" {
            return self.pass(req).await;
        }
        log::warning(&format!("refused {} {}", req.method, req.path));
        error(403, &format!("deid proxy: {} {} is not forwarded, because the proxy cannot de-identify it", req.method, req.path))
    }

    async fn pass(&self, req: Request) -> Response {
        let url = self.url(&req);
        let mut up = match http::fetch(&req.method, &url, &self.headers(&req), Bytes::new(), CONNECT, READ).await {
            Ok(u) => u,
            Err(e) => return error(502, &format!("deid proxy: {e}")),
        };
        let body = up.read().await.unwrap_or_default();
        log::info(&format!("{} {} -> {}", req.method, req.path, up.status));
        Response::full(up.status, back_headers(&up), body)
    }

    fn body(&self, req: &Request) -> Result<Value, Response> {
        let encoding = req.header("content-encoding").unwrap_or("identity").to_lowercase();
        let raw: Vec<u8> = if encoding == "zstd" {
            match crate::zstd::decompress(&req.body, 256 << 20) {
                Ok(b) => b,
                Err(e) => return Err(error(400, &format!("deid proxy: the zstd-compressed request body could not be read ({e})"))),
            }
        } else if encoding.is_empty() || encoding == "identity" {
            req.body.to_vec()
        } else {
            return Err(error(415, &format!("deid proxy: {encoding}-compressed request bodies are not supported")));
        };
        serde_json::from_slice::<Value>(&raw).map_err(|_| error(400, "deid proxy: the request body is not JSON"))
    }

    fn audit(&self, scope: &str, path: &str, sent: &str, mapping: &ODict<String>, target: &str) {
        let mut kinds: Vec<String> = mapping.keys().map(|t| t.rsplit_once('_').map_or(t, |(k, _)| k).to_string()).collect();
        kinds.sort();
        kinds.dedup();
        let line = json!({"at": now(), "actor": scope, "action": path, "plane": "cloud", "sensitivity": "Internal",
                          "sha256": content_hash(sent), "target": target, "entity_types": kinds, "redacted_count": mapping.len()});
        if let Some(hook) = self.on_crossing.borrow().as_ref() {
            hook(&line, sent);
        }
        match &self.cfg.audit {
            Some(p) => append(p, &line),
            None => self.audited.borrow_mut().push(line),
        }
    }

    fn report_usage(&self, model: &str, usage: &Map<String, Value>) {
        if usage.is_empty() {
            return;
        }
        if let Some(hook) = self.on_usage.borrow().as_ref() {
            hook(model, usage);
        }
    }

    fn report_responses_usage(&self, model: &str, usage: &Map<String, Value>) {
        if usage.contains_key("input_tokens") || usage.contains_key("output_tokens") {
            let mut u = usage.clone();
            u.insert("api".into(), json!("responses"));
            self.report_usage(model, &u);
        }
    }

    async fn responses(self: Rc<Self>, req: Request) -> Response {
        let body = match self.body(&req) {
            Ok(b) => b,
            Err(r) => return r,
        };
        let Some(scope) = self.scope(&req, &body) else {
            return error(400, "deid proxy: this folder has no scope; list it under `paths` in a seed file, or send an x-deid-scope header");
        };
        let engine = self.engine(&scope);
        let restore = self.restore.clone();
        let drop = self.route.borrow().drop_tool_namespaces.clone();
        let prep = match openai::prepare_responses(&body, &engine, &self.headers(&req), &|k| restore.get(k), &self.cfg.binary,
                                                   &drop) {
            Ok(p) => p,
            Err(PrepareError::Refused(r)) => {
                log::warning(&format!("refused {}: {r}", req.path));
                return error(400, &format!("deid proxy: {r}"));
            }
            Err(PrepareError::Failed(f)) => {
                log::error(&format!("de-identification failed; nothing sent: {f}"));
                return error(500, "deid proxy: de-identification failed, so nothing was sent");
            }
        };
        let sent = py::dumps_with(&prep.body, false, true);
        let url = self.url(&req);
        self.audit(&scope, &req.path, &sent, &prep.mapping, &url);
        let meta: Map<String, Value> = prep.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("x-codex-turn-metadata"))
            .map(|(k, v)| (k.clone(), json!(v))).collect();
        self.record(json!({"dir": "out", "scope": scope, "path": req.path_qs, "body": prep.body, "turn_metadata": meta, "withheld": prep.withheld}));
        log::info(&format!("{} scope={scope} tokens={} withheld={} restored={}", req.path, prep.mapping.len(), prep.withheld.len(), prep.restored));
        let mut up = match http::fetch("POST", &url, &prep.headers, Bytes::from(sent), CONNECT, READ).await {
            Ok(u) => u,
            Err(e) => return error(502, &format!("deid proxy: {e}")),
        };
        log::info(&format!("upstream {} content-type={}", up.status, py::repr_str(up.header("content-type").unwrap_or(""))));
        let streamed = body.get("stream") == Some(&json!(true)) || up.header("content-type").unwrap_or("").contains("event-stream");
        if up.status >= 400 || !streamed {
            let mut data = up.read().await.unwrap_or_default();
            if up.status < 400 {
                if let Some(Value::Object(mut obj)) = serde_json::from_slice::<Value>(&data).ok() {
                    if let Some(Value::Array(items)) = obj.get("output").cloned() {
                        let model = obj.get("model").filter(|m| py::truthy(Some(m))).or_else(|| body.get("model")).map_or(String::new(), py::py_str);
                        self.report_responses_usage(&model, obj.get("usage").and_then(Value::as_object).unwrap_or(&Map::new()));
                        self.record(json!({"dir": "in", "scope": scope, "body": obj}));
                        let put = |k: &str, v: &str| restore.put(k, v);
                        let restored: Vec<Value> = items.iter().map(|i| openai::restore_item(i, &engine, &prep.mapping, Some(&put),
                                                                                              &self.cfg.openai_local_tools, Some(&prep.spellings))).collect();
                        obj.insert("output".into(), Value::Array(restored));
                        data = py::dumps(&Value::Object(obj)).into_bytes();
                    }
                }
            }
            return Response::full(up.status, back_headers(&up), data);
        }
        let (resp, tx) = Response::stream(up.status, back_headers(&up));
        let me = self.clone();
        let model = body.get("model").map_or(String::new(), py::py_str);
        tokio::task::spawn_local(async move {
            let restore = me.restore.clone();
            let mut restorer = ResponsesRestorer::new(&engine, prep.mapping.clone(), Box::new(move |k: &str, v: &str| restore.put(k, v)),
                                                      me.cfg.openai_local_tools.clone(), prep.spellings.clone());
            let mut collected = Map::new();
            me.relay(&mut up, &tx, &scope, Some(&mut collected), |n, d| restorer.event(n, d)).await;
            let m = collected.remove("model").filter(|v| py::truthy(Some(v))).map(|v| py::py_str(&v)).unwrap_or(model);
            me.report_responses_usage(&m, &collected);
        });
        resp
    }

    async fn messages(self: Rc<Self>, req: Request) -> Response {
        let counting = req.path.ends_with("/count_tokens");
        let body = match self.body(&req) {
            Ok(b) => b,
            Err(r) => return r,
        };
        let Some(scope) = self.scope(&req, &body) else {
            return error(400, "deid proxy: this folder has no scope; list it under `paths` in a seed file, or send an x-deid-scope header");
        };
        let engine = self.engine(&scope);
        let restore = self.restore.clone();
        let prep = match anthropic::prepare_request(&body, &engine, &|k| restore.get(k), &self.cfg.binary) {
            Ok(p) => p,
            Err(PrepareError::Refused(r)) => {
                log::warning(&format!("refused {}: {r}", req.path));
                return error(400, &format!("deid proxy: {r}"));
            }
            Err(PrepareError::Failed(f)) => {
                log::error(&format!("de-identification failed; nothing sent: {f}"));
                return error(500, "deid proxy: de-identification failed, so nothing was sent");
            }
        };
        let sent = py::dumps_with(&prep.body, false, true);
        let upstream = self.route.borrow().upstream.clone();
        self.audit(&scope, &req.path, &sent, &prep.mapping, &upstream);
        self.record(json!({"dir": "out", "scope": scope, "path": req.path_qs, "body": prep.body, "withheld": prep.withheld}));
        log::info(&format!("{} scope={scope} tokens={} withheld={} restored={}", req.path, prep.mapping.len(), prep.withheld.len(), prep.restored));
        let url = format!("{}{}", upstream.trim_end_matches('/'), req.path_qs);
        let mut up = match http::fetch("POST", &url, &self.headers(&req), Bytes::from(sent), CONNECT, READ).await {
            Ok(u) => u,
            Err(e) => return error(502, &format!("deid proxy: {e}")),
        };
        if self.route.borrow().upstream_key.is_some() && up.status == 401 {
            log::warning("the provider refused the held key");
            if let Some(hook) = self.on_key_refused.borrow().as_ref() {
                hook();
            }
        }
        let headers = back_headers(&up);
        let streamed = body.get("stream") == Some(&json!(true)) || up.header("content-type").unwrap_or("").contains("event-stream");
        if up.status >= 400 || !streamed {
            let mut data = up.read().await.unwrap_or_default();
            if up.status < 400 && !counting {
                if let Some(obj @ Value::Object(_)) = serde_json::from_slice::<Value>(&data).ok() {
                    if obj.get("type").and_then(Value::as_str) == Some("message") {
                        let model = obj.get("model").filter(|m| py::truthy(Some(m))).or_else(|| body.get("model")).map_or(String::new(), py::py_str);
                        self.report_usage(&model, obj.get("usage").and_then(Value::as_object).unwrap_or(&Map::new()));
                        self.record(json!({"dir": "in", "scope": scope, "body": obj}));
                        let put = |k: &str, v: &str| restore.put(k, v);
                        let restored = anthropic::restore_json_response(obj, &engine, &prep.mapping, &put, &self.cfg.local_tools, Some(&prep.spellings));
                        data = py::dumps(&restored).into_bytes();
                    }
                }
            }
            return Response::full(up.status, headers, data);
        }
        let (resp, tx) = Response::stream(up.status, headers);
        let me = self.clone();
        let model = body.get("model").map_or(String::new(), py::py_str);
        tokio::task::spawn_local(async move {
            let restore = me.restore.clone();
            let mut restorer = StreamRestorer::new(&engine, prep.mapping.clone(), Box::new(move |k: &str, v: &str| restore.put(k, v)),
                                                   me.cfg.local_tools.clone(), prep.spellings.clone());
            let mut collected = Map::new();
            let usage = if counting { None } else { Some(&mut collected) };
            me.relay(&mut up, &tx, &scope, usage, |n, d| restorer.event(n, d)).await;
            if !collected.is_empty() {
                let m = collected.remove("model").filter(|v| py::truthy(Some(v))).map(|v| py::py_str(&v)).unwrap_or(model);
                me.report_usage(&m, &collected);
            }
        });
        resp
    }

    /// Stream the upstream's events back through `restore`; with `usage`, collect the answer's model and
    /// usage on the way.
    async fn relay(&self, up: &mut http::Upstream, tx: &tokio::sync::mpsc::UnboundedSender<Bytes>, scope: &str,
                   mut usage: Option<&mut Map<String, Value>>,
                   mut restore: impl FnMut(&str, Value) -> Result<Vec<(String, Value)>, String>) {
        let mut reader = sse::Reader::default();
        let mut failed: Option<String> = None;
        'read: loop {
            let line = match up.line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            };
            let Ok(line) = String::from_utf8(line) else {
                failed = Some("UnicodeDecodeError".into());
                break;
            };
            for (name, data) in reader.line(&line) {
                if !self.event(&name, &data, tx, scope, &mut usage, &mut restore, &mut failed) {
                    break 'read;
                }
            }
        }
        if failed.is_none() {
            if let Some((name, data)) = reader.finish() {
                self.event(&name, &data, tx, scope, &mut usage, &mut restore, &mut failed);
            }
        }
        if let Some(why) = failed {
            log::error(&format!("re-identification failed mid-stream: {why}"));
            let _ = tx.send(Bytes::from(sse::encode("error", &json!({"type": "error", "error": {"type": "api_error",
                "message": "deid proxy: the answer could not be re-identified; please retry"}}))));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn event(&self, name: &str, data: &str, tx: &tokio::sync::mpsc::UnboundedSender<Bytes>, scope: &str,
             usage: &mut Option<&mut Map<String, Value>>, restore: &mut impl FnMut(&str, Value) -> Result<Vec<(String, Value)>, String>,
             failed: &mut Option<String>) -> bool {
        if name.is_empty() && data.starts_with(':') {
            return tx.send(Bytes::from(format!("{data}\n\n"))).is_ok();
        }
        self.record(json!({"dir": "in", "scope": scope, "event": name, "data": data}));
        let Some(obj) = py::loads(data) else {
            return tx.send(Bytes::from(sse::encode_text(name, data))).is_ok();
        };
        if let (Some(u), Value::Object(o)) = (usage.as_deref_mut(), &obj) {
            let t = o.get("type").and_then(Value::as_str).unwrap_or("");
            if t == "message_start" {
                if let Some(Value::Object(m)) = o.get("message") {
                    let model = m.get("model").filter(|v| py::truthy(Some(v))).cloned().or_else(|| u.get("model").cloned()).unwrap_or(json!(""));
                    u.insert("model".into(), model);
                    for (k, v) in m.get("usage").and_then(Value::as_object).cloned().unwrap_or_default() {
                        u.insert(k, v);
                    }
                }
            } else if t == "message_delta" {
                if let Some(Value::Object(us)) = o.get("usage") {
                    for (k, v) in us.iter().filter(|(_, v)| !v.is_null()) {
                        u.insert(k.clone(), v.clone());
                    }
                }
            } else if t == "response.completed" || t == "response.incomplete" {
                if let Some(Value::Object(r)) = o.get("response") {
                    let model = r.get("model").filter(|v| py::truthy(Some(v))).cloned().or_else(|| u.get("model").cloned()).unwrap_or(json!(""));
                    u.insert("model".into(), model);
                    for (k, v) in r.get("usage").and_then(Value::as_object).cloned().unwrap_or_default() {
                        u.insert(k, v);
                    }
                }
            }
        }
        if !obj.is_object() {
            *failed = Some("AttributeError: an event that is not an object".into());
            return false;
        }
        match restore(name, obj) {
            Ok(events) => {
                for (n, d) in events {
                    if tx.send(Bytes::from(sse::encode(&n, &d))).is_err() {
                        return false; // the agent closed the stream once it had what it needed
                    }
                }
                true
            }
            Err(e) => {
                *failed = Some(e);
                false
            }
        }
    }
}
