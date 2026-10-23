// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A confined Codex's calls to ChatGPT's own backend (its `chatgpt_base_url` is deid-agent's `/backend-api/`):
//! a GET reaches ChatGPT's backend at the same path, its answer comes back as it is; a POST is refused, as every
//! request the proxy cannot de-identify; the model's requests go where they went.

use std::rc::Rc;

use bytes::Bytes;
use deidkit::http::{Request, Response};
use deidkit::proxy::server::{MemoryRestore, Proxy, ProxyConfig};
use deidkit::store::InMemoryTokenStore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn request(method: &str, path_qs: &str) -> Request {
    let path = path_qs.split('?').next().unwrap().to_string();
    Request { method: method.into(), path, path_qs: path_qs.into(),
              headers: vec![("authorization".into(), "Bearer test".into()), ("chatgpt-account-id".into(), "acct-test".into())],
              body: Bytes::new() }
}

#[test]
fn codex_reaches_chatgpts_backend_through_the_proxy() {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        // ChatGPT's backend: answers with the request line it heard
        let far = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = far.local_addr().unwrap().port();
        tokio::task::spawn_local(async move {
            let (mut s, _) = far.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = s.read(&mut buf).await.unwrap();
            let heard = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string();
            let reply = format!("HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{heard}", heard.len());
            s.write_all(reply.as_bytes()).await.unwrap();
        });
        let cfg = ProxyConfig { upstream_chatgpt: format!("http://127.0.0.1:{port}/backend-api/codex"), ..ProxyConfig::default() };
        let proxy = Proxy::new(Rc::new(InMemoryTokenStore::new()), Rc::new(MemoryRestore::default()), cfg);

        assert_eq!(proxy.url(&request("GET", "/backend-api/wham/config/bundle")),
                   format!("http://127.0.0.1:{port}/backend-api/wham/config/bundle"));
        assert_eq!(proxy.url(&request("POST", "/v1/responses")), format!("http://127.0.0.1:{port}/backend-api/codex/responses"));
        assert_eq!(proxy.url(&request("GET", "/v1/models?client_version=1")),
                   format!("http://127.0.0.1:{port}/backend-api/codex/models?client_version=1"));

        let got: Response = proxy.clone().handle(request("GET", "/backend-api/wham/config/bundle?x=1")).await;
        assert_eq!(got.status, 200);
        let body = match got.payload { deidkit::http::Payload::Full(b) => b, _ => panic!("a whole answer") };
        assert_eq!(String::from_utf8_lossy(&body), "GET /backend-api/wham/config/bundle?x=1 HTTP/1.1");

        let refused = proxy.clone().handle(request("POST", "/backend-api/wham/usage/thread_usage/query")).await;
        assert_eq!(refused.status, 403);
    });
}
