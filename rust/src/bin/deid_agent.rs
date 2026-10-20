// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! `deid-agent`: a coding agent over one folder, behind deid-proxy, with a page to drive it — the same
//! flags as Python's `deidkit.agents.cli`.
//!
//!     deid-agent --folder ~/matters/client-a --scope client-a --seeds ~/.deid/seeds/client-a.toml
//!
//! The page's address, with its token, is printed on start. A program that starts deid-agent and opens
//! the page itself passes `--secrets-stdin` and writes one JSON line, `{"token": "…", "api_key": "…"}`.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use serde_json::Value;

use deidkit::agents::confine;
use deidkit::agents::host::{shlex_split, AgentChoice, AgentHost, HostConfig};
use deidkit::cli::{self, flag, opt, Opt};
use deidkit::patterns::RegexDetector;
use deidkit::proxy::server::{Proxy, ProxyConfig, StoreRestore};
use deidkit::pyre;
use deidkit::seeds::{load_scope_paths, load_seed_files, InMemorySeedSource, SeedEntity};
use deidkit::store::SqliteTokenStore;
use deidkit::{http, log, py};

const PROG: &str = "deid-agent";
const DEFAULT_AGENT: &str = "npx -y @agentclientprotocol/claude-agent-acp@0.85.0";
const GENERIC_ACCOUNTS: [&str; 12] = ["user", "users", "admin", "administrator", "root", "home", "guest", "default", "public",
                                      "shared", "owner", "office"];

const fn none(name: &'static str, metavar: &'static str, help: &'static str) -> Opt {
    Opt { name, value: true, default: None, choices: &[], help, metavar, required: false }
}

const OPTS: &[Opt] = &[
    Opt { required: true, ..none("--folder", "", "the folder the agent works in") },
    Opt { required: true, ..none("--scope", "", "the folder's deid scope") },
    none("--title", "", "shown on the page (default: the folder's name)"),
    none("--seeds", "TOML", ""),
    none("--seeds-dir", "DIR", ""),
    opt("--vault", "~/.deid/agent/vault.sqlite", "token store; not the one a running deid-proxy uses"),
    opt("--audit", "~/.deid/agent/audit.jsonl", ""),
    none("--record", "", "write what crossed (tokens only) to this JSONL file"),
    opt("--usage-log", "~/.deid/agent/usage.jsonl", "one JSON line per task: scope, model, seconds, tokens ('' for none)"),
    opt("--port", "8790", ""),
    opt("--bind", "127.0.0.1", "the address the page and the proxy listen on: 0.0.0.0 in a container whose port is \
published to its host (the token still guards the page)"),
    none("--unix", "PATH", "also take the page and the proxy on this Unix socket (only its owner may connect), each \
connection passed on to the port: a container's published socket, which reaches the host without its network"),
    opt("--agent-command", DEFAULT_AGENT, "the ACP agent to start"),
    none("--agent", "NAME=COMMAND", "an agent the page switches to (repeat for each)"),
    none("--agent-upstream", "NAME=URL", "that agent's model requests go to this server of the organization's own"),
    none("--agent-unavailable", "NAME=REASON", "the page lists that agent but cannot start it on this machine"),
    none("--agent-meta", "NAME=JSON", "that agent's own session options, sent as the session's _meta"),
    none("--session-meta", "JSON", "the session's _meta for an agent that names none of its own"),
    none("--style", "CSS", "a stylesheet the page loads after its own"),
    none("--claude-executable", "", "the claude binary the Claude adapter should run"),
    none("--codex-executable", "", "the person's own codex binary, for an agent that is Codex (codex-acp)"),
    none("--codex-home", "DIR", "Codex's home for this matter (default ~/.deid/agent/codex/<scope>)"),
    flag("--codex-unconfined", "start Codex without confining it to the folder"),
    flag("--claude-unconfined", "run Claude Code's shell commands outside its sandbox"),
    none("--guard", "", "agent-guard, which decides each of Claude Code's tool calls before it runs"),
    flag("--no-guard", "start Claude Code without agent-guard"),
    none("--person", "", "who the person at this Mac is, in the notes and decisions they make on documents"),
    opt("--upstream", "https://api.anthropic.com", "where the agent's model requests go"),
    opt("--api-key-env", "ANTHROPIC_API_KEY", "environment variable holding the organization's key"),
    flag("--secrets-stdin", "read the page token and the key as one JSON line on stdin"),
    flag("--lifeline", "keep reading stdin (after the secrets); when it closes, whoever started deid-agent is gone \
(even killed): stop the agents and exit"),
    flag("--dev-login", "allow the machine's own Claude sign-in when no API key is set"),
    none("--subagents", "HOSTS", "sub-agents on the organization's own model servers (matteragent's hosts file)"),
    none("--subagents-command", "", "what starts the sub-agents' MCP server (default: matteragent-mcp on PATH)"),
    opt("--subagents-timeout", "900.0", "seconds a client waits for one sub-agent's report"),
];

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// The operating-system account the agent runs under, as its home folder names it; None when generic.
fn account_name() -> Option<String> {
    let home = py::expanduser("~");
    let name = home.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    (py::clen(&name) >= 3 && !GENERIC_ACCOUNTS.contains(&deidkit::unicode::casefold(&name).as_str())).then_some(name)
}

/// The one JSON line a starting program writes: `token` and `api_key`.
fn read_secrets() -> Value {
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    if py::strip(&line).is_empty() {
        return serde_json::json!({});
    }
    match serde_json::from_str::<Value>(&line) {
        Ok(v @ Value::Object(_)) => v,
        _ => die("deid-agent: --secrets-stdin expects one JSON line with token and api_key"),
    }
}

/// `--lifeline`: stdin stays open as long as the program that started deid-agent lives — the system closes it when
/// that program ends, even killed (the desktop app, LegalAiz 14.10: a killed app left deid-agent and its agents
/// running, holding the port). Read on a thread of its own, through the same buffered stdin the secrets came by.
fn watch_lifeline() -> tokio::sync::oneshot::Receiver<()> {
    use std::io::Read;
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut buf = [0u8; 256];
        let mut stdin = std::io::stdin();
        loop {
            match stdin.read(&mut buf) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = tx.send(());
    });
    rx
}

/// `--unix`: the page and the proxy on a Unix socket as well, each connection passed on to the port. Apple's container
/// publishes such a socket to its host through the machine's own channel (`container run --publish-socket`), not
/// its network — a published port needs macOS's Local Network permission for Apple's helper, which a person never
/// sees asked (Bernio, 20.10.2026), and deid-agent on 0.0.0.0. The socket is its owner's alone (0600).
async fn pass_on(path: String, to: std::net::SocketAddr) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::remove_file(&path);
    let listener = match tokio::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => die(&format!("deid-agent: --unix {path}: {}", deidkit::agents::acp::os_error(&e, None))),
    };
    if let Err(e) = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)) {
        die(&format!("deid-agent: --unix {path}: {e}"));
    }
    loop {
        let Ok((mut from, _)) = listener.accept().await else { continue };
        tokio::task::spawn_local(async move {
            if let Ok(mut port) = tokio::net::TcpStream::connect(to).await {
                let _ = tokio::io::copy_bidirectional(&mut from, &mut port).await;
            }
        });
    }
}

fn json_object(text: &str, option: &str) -> Value {
    match deidkit::pyjson::loads(text) {
        Some(v @ Value::Object(_)) => v,
        _ => die(&format!("deid-agent: {option} expects a JSON object, not {}", py::repr_str(text))),
    }
}

fn pairs(specs: &[String], option: &str) -> Vec<(String, String)> {
    specs.iter().map(|spec| {
        let (name, value) = spec.split_once('=').unwrap_or((spec, ""));
        let (name, value) = (py::strip(name), py::strip(value));
        if !spec.contains('=') || name.is_empty() || value.is_empty() {
            die(&format!("deid-agent: {option} expects NAME=VALUE, not {}", py::repr_str(spec)));
        }
        (name.to_string(), value.to_string())
    }).collect()
}

/// The agents of `--agent NAME=COMMAND`, in their order, with their upstreams, reasons and options.
fn agent_choices(args: &cli::Args) -> Vec<AgentChoice> {
    let mut choices: Vec<AgentChoice> = Vec::new();
    let find = |choices: &mut Vec<AgentChoice>, name: &str| choices.iter().position(|c| c.name == name);
    for (name, command) in pairs(&args.all("--agent"), "--agent") {
        let command = shlex_split(&command).unwrap_or_else(|_| die(&format!("deid-agent: --agent {name}: No closing quotation")));
        match find(&mut choices, &name) {
            Some(i) => choices[i] = AgentChoice::new(&name, command),
            None => choices.push(AgentChoice::new(&name, command)),
        }
    }
    for (name, url) in pairs(&args.all("--agent-upstream"), "--agent-upstream") {
        let Some(i) = find(&mut choices, &name) else { die(&format!("deid-agent: --agent-upstream names no agent: {name}")) };
        choices[i].upstream = Some(url.trim_end_matches('/').to_string());
    }
    for (name, reason) in pairs(&args.all("--agent-unavailable"), "--agent-unavailable") {
        match find(&mut choices, &name) {
            Some(i) => choices[i].unavailable = Some(reason),
            None => {
                let mut c = AgentChoice::new(&name, Vec::new());
                c.unavailable = Some(reason);
                choices.push(c);
            }
        }
    }
    for (name, meta) in pairs(&args.all("--agent-meta"), "--agent-meta") {
        let Some(i) = find(&mut choices, &name) else { die(&format!("deid-agent: --agent-meta names no agent: {name}")) };
        choices[i].meta = Some(json_object(&meta, "--agent-meta"));
    }
    choices
}

fn seed_files(files: &[String], directory: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = files.iter().map(|f| py::expanduser(f).to_string_lossy().into_owned()).collect();
    if let Some(dir) = directory {
        let mut found: Vec<String> = std::fs::read_dir(py::expanduser(dir)).into_iter().flatten().flatten()
            .map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".toml")))
            .map(|p| p.to_string_lossy().into_owned()).collect();
        found.sort();
        out.extend(found);
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = cli::parse(PROG, "deid-agent: a coding agent over one folder, behind deid-proxy, with a page to drive it.", OPTS, &argv);
    let _ = log::init("", "INFO");
    let port: u16 = cli::number(PROG, OPTS, "--port", &args.str("--port"), "int");
    let bind: std::net::IpAddr = args.str("--bind").parse()
        .unwrap_or_else(|_| die(&format!("deid-agent: --bind: not an address: {}", args.str("--bind"))));
    let timeout: f64 = cli::number(PROG, OPTS, "--subagents-timeout", &args.str("--subagents-timeout"), "float");
    let folder = py::realpath(&py::expanduser(&args.str("--folder")));
    if !folder.is_dir() {
        die(&format!("deid-agent: no such folder: {}", folder.display()));
    }
    let (token, api_key) = if args.on("--secrets-stdin") {
        let given = read_secrets();
        let s = |k: &str| given.get(k).filter(|v| py::truthy(Some(v))).map(py::py_str);
        (s("token").unwrap_or_default(), s("api_key"))
    } else {
        let token = std::env::var("DEID_AGENT_TOKEN").unwrap_or_default();
        std::env::remove_var("DEID_AGENT_TOKEN");
        (token, std::env::var(args.str("--api-key-env")).ok().filter(|k| !k.is_empty()))
    };
    if api_key.is_none() && !args.on("--dev-login") {
        die(&format!("deid-agent: no organization key (set {}, or pass it with --secrets-stdin), or pass --dev-login for development",
                     args.str("--api-key-env")));
    }
    let scope = args.str("--scope");
    let agent_command = shlex_split(&args.str("--agent-command")).unwrap_or_else(|_| die("deid-agent: --agent-command: No closing quotation"));
    let title = args.get("--title").filter(|t| !t.is_empty()).map(str::to_string)
        .unwrap_or_else(|| folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let mut cfg = HostConfig::new(folder.clone(), &scope, &title, agent_command);
    cfg.api_key = api_key.clone();
    cfg.dev_login = args.on("--dev-login");
    cfg.claude_executable = args.get("--claude-executable").map(str::to_string).or_else(|| confine::which("claude"));
    cfg.codex_executable = args.get("--codex-executable").map(str::to_string).or_else(|| confine::which("codex"));
    cfg.codex_confine = !args.on("--codex-unconfined");
    cfg.codex_home = args.get("--codex-home").filter(|h| !h.is_empty()).map(py::expanduser);
    cfg.port = port;
    if let Some(p) = args.get("--person") {
        cfg.person = p.to_string();
    }
    cfg.usage_log = args.get("--usage-log").filter(|u| !u.is_empty()).map(py::expanduser);
    cfg.agents = agent_choices(&args);
    cfg.session_meta = args.get("--session-meta").filter(|m| !m.is_empty()).map(|m| json_object(m, "--session-meta"));
    cfg.style = args.get("--style").filter(|s| !s.is_empty()).map(py::expanduser);
    cfg.subagents = args.get("--subagents").filter(|s| !s.is_empty()).map(|s| py::realpath(&py::expanduser(s)));
    let sub_cmd = args.get("--subagents-command").map(str::to_string).or_else(|| confine::which("matteragent-mcp")).unwrap_or_default();
    cfg.subagents_command = shlex_split(&sub_cmd).unwrap_or_else(|_| die("deid-agent: --subagents-command: No closing quotation"));
    cfg.subagents_timeout = timeout;
    cfg.claude_confine = !args.on("--claude-unconfined");
    // the launcher of a confined Claude Code's shell commands, beside this program (the app's agent runtime)
    cfg.claude_shell = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("agent-shell")))
        .filter(|l| l.is_file());
    if cfg.claude_confine && cfg.claude_shell.is_none() && confine::supported() {
        log::warning("agent-shell is not beside deid-agent: Claude Code runs with only its shell commands sandboxed");
    }
    // agent-guard: given, else $DEID_AGENT_GUARD, else the one beside this program (the app's agent runtime
    // carries both in bin/ — what the Python runtime's wrapper said in $DEID_AGENT_GUARD), else on PATH
    let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("agent-guard")))
        .filter(|g| g.is_file()).map(|g| g.to_string_lossy().into_owned());
    let guard = args.get("--guard").map(str::to_string)
        .or_else(|| std::env::var("DEID_AGENT_GUARD").ok().filter(|g| !g.is_empty()))
        .or(beside)
        .or_else(|| confine::which("agent-guard")).unwrap_or_default();
    if !guard.is_empty() && !args.on("--no-guard") {
        let g = py::expanduser(&guard);
        let executable = g.is_file() && std::ffi::CString::new(g.to_string_lossy().as_bytes())
            .is_ok_and(|c| unsafe { libc::access(c.as_ptr(), libc::X_OK) } == 0);
        if !executable {
            die(&format!("deid-agent: --guard: not an executable file: {}", g.display()));
        }
        cfg.guard = vec![g.to_string_lossy().into_owned()];
        let journals = py::expanduser(&args.str("--audit")).parent().map(Path::to_path_buf).unwrap_or_default().join("guard");
        {
            use std::os::unix::fs::DirBuilderExt;
            let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&journals);
        }
        let t: Vec<char> = scope.chars().collect();
        let safe = pyre::Regex::new(r"[^A-Za-z0-9._-]").unwrap().sub_with(&t, 0, |_| "_".into()).0;
        cfg.guard_log = Some(journals.join(format!("{safe}.jsonl")));
    } else if !args.on("--no-guard") {
        log::warning("agent-guard is not on PATH: Claude Code works without it (--guard, or --no-guard)");
    }
    if let Some(hosts) = &cfg.subagents {
        if !hosts.is_file() {
            die(&format!("deid-agent: --subagents: no such file: {}", hosts.display()));
        }
        let agents = if cfg.agents.is_empty() { vec![AgentChoice::new("", cfg.agent_command.clone())] } else { cfg.agents.clone() };
        if cfg.subagents_command.is_empty() && !agents.iter().all(AgentChoice::matteragent) {
            die("deid-agent: --subagents needs --subagents-command for Claude Code and Codex (matteragent-mcp is not on PATH)");
        }
    }
    if !cfg.agents.is_empty() && cfg.agents.iter().all(|a| a.unavailable.is_some()) {
        die(&format!("deid-agent: no agent can start here: {}",
                     cfg.agents.iter().map(|a| format!("{}: {}", a.name, a.unavailable.as_deref().unwrap_or(""))).collect::<Vec<_>>().join("; ")));
    }
    if !token.is_empty() {
        cfg.token = token.clone();
    }
    let files = seed_files(&args.all("--seeds"), args.get("--seeds-dir"));
    let store = match SqliteTokenStore::open(&py::expanduser(&args.str("--vault"))) {
        Ok(s) => Rc::new(s),
        Err(e) => die(&format!("deid-agent: {e}")),
    };
    let seeds = load_seed_files(&files, &scope, InMemorySeedSource::default()).unwrap_or_else(|e| die(&format!("deid-agent: {e}")));
    if let Some(account) = account_name() {
        seeds.add_entity(&scope, SeedEntity { r#type: "account".into(), name: account, identifier: String::new(),
                                              identifier_type: String::new(), jurisdiction_country: String::new(),
                                              role: String::new(), attributes: None, relationships: None, id: Value::Null });
    }
    let proxy_cfg = ProxyConfig {
        default_scope: scope.clone(), seeds: Some(Rc::new(seeds)), upstream: args.str("--upstream").trim_end_matches('/').to_string(),
        scope_paths: load_scope_paths(&files, &scope).unwrap_or_default(), detector: Some(Rc::new(RegexDetector::default())),
        record: args.get("--record").filter(|r| !r.is_empty()).map(py::expanduser),
        audit: Some(py::expanduser(&args.str("--audit"))),
        upstream_key: api_key.clone(), agent_key: api_key.as_ref().map(|_| cfg.agent_key.clone()),
        ..ProxyConfig::default()
    };
    let url = format!("http://127.0.0.1:{port}/{}", if token.is_empty() { format!("#t={}", cfg.token) } else { String::new() });
    let lifeline = args.on("--lifeline");
    let unix = args.get("--unix").filter(|p| !p.is_empty()).map(str::to_string);
    let unix_left = unix.clone();
    let inner: std::net::IpAddr = if bind.is_unspecified() { [127, 0, 0, 1].into() } else { bind };
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("the runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        let listener = match tokio::net::TcpListener::bind((bind, port)).await {
            Ok(l) => l,
            Err(e) => die(&format!("deid-agent: {bind}:{port}: {}", deidkit::agents::acp::os_error(&e, None))),
        };
        let proxy = Proxy::new(store.clone(), Rc::new(StoreRestore(store)), proxy_cfg);
        let host = AgentHost::new(cfg, proxy);
        println!("deid-agent: {url}");
        let h = host.clone();
        let handler: http::Handler = Rc::new(move |req| {
            let h = h.clone();
            Box::pin(async move { h.handle(req).await })
        });
        let tunnels = Rc::new(confine::SIGN_IN.iter().map(|h| format!("{h}:443")).collect::<Vec<_>>());
        tokio::task::spawn_local(http::serve_with_tunnels(listener, handler, tunnels));
        if let Some(path) = unix {
            tokio::task::spawn_local(pass_on(path, (inner, port).into()));
        }
        tokio::task::spawn_local(host.clone().start_agent());
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM");
        let gone = lifeline.then(watch_lifeline);
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
            _ = async { match gone { Some(rx) => { let _ = rx.await; } None => std::future::pending().await } } => {
                log::info("the program that started deid-agent is gone (its stdin closed): stopping the agents");
            }
        }
        host.cleanup().await;
    });
    if let Some(path) = unix_left {
        let _ = std::fs::remove_file(path);
    }
    let _ = PathBuf::new();
}
