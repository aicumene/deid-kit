// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A refusal of the held key is the key of the route the request went by, not of the route at its answer: the chat
//! switches agents while Claude Code's requests are out (25.10.2026: to MatterAgent's own server, with its key, while
//! Claude Code's count_tokens went to Anthropic on its own sign-in, and Anthropic's 401s were told as MatterAgent's key
//! refused). The proxy and deid-agent's host in one process; Anthropic a stand-in that answers 401 only once the
//! route has switched.

use std::rc::Rc;

use bytes::Bytes;
use deidkit::agents::host::{AgentChoice, AgentHost, HostConfig};
use deidkit::http::{Payload, Request};
use deidkit::proxy::server::{MemoryRestore, Proxy, ProxyConfig};
use deidkit::store::InMemoryTokenStore;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const STAND_IN: &str = "deid-agent-stand-in";

/// The page's events so far, as text.
async fn told(host: &Rc<AgentHost>, token: &str) -> String {
    let req = Request { method: "GET".into(), path: "/deid-agent/api/events".into(), path_qs: "/deid-agent/api/events".into(),
                        headers: vec![("x-deid-agent-token".into(), token.into())], body: Bytes::new() };
    let Payload::Stream(mut rx) = host.clone().handle(req).await.payload else { panic!("the events are a stream") };
    let mut text = String::new();
    while let Ok(Some(chunk)) = tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await {
        text.push_str(&String::from_utf8_lossy(&chunk));
    }
    text
}

/// Claude Code's count_tokens out to Anthropic — with the organization's key held, or on its own sign-in — and the
/// chat switching to MatterAgent's own server (with its key) before Anthropic answers 401. What the page told, and the
/// key the request went with.
fn switched_while_out(org_key: Option<&str>) -> (u16, String, String) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        let far = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = far.local_addr().unwrap().port();
        let (heard_tx, heard_rx) = oneshot::channel::<String>();
        let (go_tx, go_rx) = oneshot::channel::<()>();
        tokio::task::spawn_local(async move {
            let (mut s, _) = far.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let n = s.read(&mut buf).await.unwrap();
            let head = String::from_utf8_lossy(&buf[..n]).to_lowercase();
            let key = head.lines().find_map(|l| l.strip_prefix("x-api-key:")).map(|k| k.trim().to_string()).unwrap_or_default();
            heard_tx.send(key).unwrap();
            go_rx.await.unwrap();
            let body = r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
            let reply = format!("HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                                body.len());
            s.write_all(reply.as_bytes()).await.unwrap();
        });
        let anthropic = format!("http://127.0.0.1:{port}");
        let cfg = ProxyConfig { default_scope: "test".into(), upstream: anthropic.clone(), upstream_key: org_key.map(str::to_string),
                                agent_key: org_key.map(|_| STAND_IN.to_string()), ..ProxyConfig::default() };
        let proxy = Proxy::new(Rc::new(InMemoryTokenStore::new()), Rc::new(MemoryRestore::default()), cfg);
        let claude = AgentChoice::new("Claude Code", vec!["claude-agent-acp".into()]);
        let mut matter = AgentChoice::new("MatterAgent", vec!["matteragent".into()]);
        matter.upstream = Some("http://127.0.0.1:9".into());
        matter.upstream_key = Some("own-server-key".into());
        let folder = std::env::temp_dir().join(format!("deid-key-route-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let mut hc = HostConfig::new(folder.clone(), "test", "Test", Vec::new());
        hc.api_key = org_key.map(str::to_string);
        hc.dev_login = org_key.is_none();
        hc.agents = vec![claude.clone(), matter.clone()];
        let token = hc.token.clone();
        let host = AgentHost::new(hc, proxy.clone());
        host.route(&claude);

        // Claude Code's request: the stand-in with the organization's key held, its own sign-in without
        let mut headers = vec![("anthropic-version".to_string(), "2023-06-01".to_string()), ("x-deid-scope".to_string(), "test".to_string())];
        headers.push(match org_key {
            Some(_) => ("x-api-key".into(), STAND_IN.into()),
            None => ("authorization".into(), "Bearer claude-code-own-sign-in".into()),
        });
        let body = json!({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "Hello"}]}).to_string();
        let req = Request { method: "POST".into(), path: "/v1/messages/count_tokens".into(), path_qs: "/v1/messages/count_tokens".into(),
                            headers, body: Bytes::from(body) };
        let p = proxy.clone();
        let out = tokio::task::spawn_local(async move { p.handle(req).await });
        let sent_key = heard_rx.await.unwrap();
        host.route(&matter);                               // the chat switches while the request is out
        go_tx.send(()).unwrap();
        let got = out.await.unwrap();
        let text = told(&host, &token).await;
        let _ = std::fs::remove_dir_all(&folder);
        (got.status, text, sent_key)
    })
}

#[test]
fn anthropics_refusal_of_claude_codes_own_sign_in_is_no_ones_key_refused() {
    let (status, told, sent) = switched_while_out(None);
    assert_eq!(status, 401, "the refusal goes back to Claude Code as it was");
    assert_eq!(sent, "", "no key held, none sent");
    assert!(!told.contains("refuses its key") && !told.contains("refused its key"), "{told}");
    assert!(!told.contains("\"kind\": \"error\""), "nothing was refused that deid-agent holds: {told}");
}

#[test]
fn a_refusal_of_the_organizations_key_is_anthropics_after_a_switch_too() {
    let (status, told, sent) = switched_while_out(Some("org-key"));
    assert_eq!(status, 401);
    assert_eq!(sent, "org-key", "the request went with the key of its route");
    assert!(told.contains("Anthropic refuses the organization's key"), "{told}");
    assert!(!told.contains("MatterAgent"), "not the key of the agent switched to: {told}");
}
