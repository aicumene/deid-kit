// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! `deid-proxy`: run the de-identification proxy on this machine (the same flags as Python's).
//!
//!     deid-proxy --scope client-a --seeds ./deid.toml
//!     ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

use std::path::PathBuf;
use std::rc::Rc;

use deidkit::cli::{self, flag, opt, Opt};
use deidkit::patterns::RegexDetector;
use deidkit::proxy::server::{Proxy, ProxyConfig, StoreRestore};
use deidkit::seeds::{load_scope_paths, load_seed_files, InMemorySeedSource};
use deidkit::store::SqliteTokenStore;
use deidkit::{http, log, py};

const PROG: &str = "deid-proxy";

const OPTS: &[Opt] = &[
    opt("--host", "127.0.0.1", "address to listen on (default: loopback)"),
    opt("--port", "8787", ""),
    opt("--upstream", "https://api.anthropic.com", "Anthropic's API, for Claude Code"),
    opt("--upstream-openai", "https://api.openai.com/v1", "OpenAI's API, for Codex with an API key"),
    opt("--upstream-chatgpt", "https://chatgpt.com/backend-api/codex", "the ChatGPT backend, for Codex signed in with ChatGPT"),
    opt("--vault", "~/.deid/vault.sqlite", "token store (SQLite)"),
    opt("--scope", "default", "scope for requests without an x-deid-scope header"),
    Opt { default: None, metavar: "TOML", ..opt("--seeds", "", "known people and organisations (repeatable)") },
    Opt { default: None, metavar: "DIR", ..opt("--seeds-dir", "", "load every *.toml in DIR; each file names its scope") },
    opt("--audit", "~/.deid/audit.jsonl", ""),
    Opt { default: None, ..opt("--record", "", "write what crossed (tokens only) to this JSONL file") },
    Opt { choices: &["withhold", "refuse"], ..opt("--binary", "withhold", "images and PDFs: replace with a note (default) or refuse the request") },
    opt("--kinds", "EMAIL,IBAN,CARD,PHONE", "pattern kinds to detect beyond the known names"),
    flag("--glossary", "send a glossary line per token (not in this build)"),
    flag("--require-scope", "refuse requests from folders no seed file lists and without a scope header"),
];

/// The seed files to load: those named, then every `*.toml` in `directory` (sorted).
pub fn seed_files(files: &[String], directory: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = files.iter().map(|f| py::expanduser(f).to_string_lossy().into_owned()).collect();
    if let Some(dir) = directory {
        let mut found: Vec<String> = std::fs::read_dir(py::expanduser(dir)).into_iter().flatten().flatten()
            .map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "toml")).map(|p| p.to_string_lossy().into_owned()).collect();
        found.sort();
        out.extend(found);
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("check") {
        eprintln!("deid-proxy check: not in this build; use the Python deid-kit for it");
        std::process::exit(2);
    }
    let seeds_given: Vec<String> = argv.windows(2).filter(|w| w[0] == "--seeds").map(|w| w[1].clone())
        .chain(argv.iter().filter_map(|a| a.strip_prefix("--seeds=").map(str::to_string))).collect();
    let args = cli::parse(PROG, "deid-proxy: run the de-identification proxy on this machine.", OPTS, &argv);
    if args.on("--glossary") {
        eprintln!("deid-proxy: --glossary is not in this build");
        std::process::exit(2);
    }
    let _ = log::init("deidkit.proxy", "INFO");
    let host = args.str("--host");
    let port: u16 = cli::number(PROG, OPTS, "--port", &args.str("--port"), "int");
    let loopback = host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if !loopback {
        eprintln!("deid-proxy: listening on {host}, not on loopback: anyone who reaches it can read the answers in clear");
    }
    let store = match SqliteTokenStore::open(&py::expanduser(&args.str("--vault"))) {
        Ok(s) => Rc::new(s),
        Err(e) => {
            eprintln!("deid-proxy: {e}");
            std::process::exit(1);
        }
    };
    let files = seed_files(&seeds_given, args.get("--seeds-dir"));
    let scope = args.str("--scope");
    let seeds = load_seed_files(&files, &scope, InMemorySeedSource::default()).unwrap_or_else(|e| {
        eprintln!("deid-proxy: {e}");
        std::process::exit(1);
    });
    let kinds: Vec<String> = args.str("--kinds").split(',').map(|k| py::strip(k).to_uppercase()).filter(|k| !k.is_empty()).collect();
    let cfg = ProxyConfig {
        default_scope: scope.clone(), upstream: args.str("--upstream"), upstream_openai: args.str("--upstream-openai"),
        upstream_chatgpt: args.str("--upstream-chatgpt"), binary: args.str("--binary"),
        record: args.get("--record").map(|r| py::expanduser(r)),
        audit: Some(args.str("--audit")).filter(|a| !a.is_empty()).map(|a| py::expanduser(&a)),
        detector: if kinds.is_empty() { None } else { Some(Rc::new(RegexDetector { kinds })) },
        seeds: Some(Rc::new(seeds)), scope_paths: load_scope_paths(&files, &scope).unwrap_or_default(),
        require_scope: args.on("--require-scope"), ..ProxyConfig::default()
    };
    let upstreams = format!("{}, {}, {}", cfg.upstream, cfg.upstream_chatgpt, cfg.upstream_openai);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("the runtime");
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async move {
        let listener = match tokio::net::TcpListener::bind((host.as_str(), port)).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("deid-proxy: {host}:{port}: {e}");
                std::process::exit(1);
            }
        };
        eprintln!("deid-proxy: {host}:{port} -> {upstreams}");
        let proxy = Proxy::new(store.clone(), Rc::new(StoreRestore(store)), cfg);
        let handler: http::Handler = Rc::new(move |req| {
            let p = proxy.clone();
            Box::pin(async move { p.handle(req).await })
        });
        http::serve(listener, handler).await;
    });
    let _ = PathBuf::new();
}
