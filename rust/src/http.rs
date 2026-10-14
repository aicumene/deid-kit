// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! HTTP/1.1 for the proxy and the agent host, on hyper: a server whose handlers run on the current thread,
//! one task a connection (as aiohttp runs them on one event loop — the vault is not for concurrent use),
//! responses whole or streamed through a channel; and a client to the model's server, over TLS where the
//! URL says so, whose body is read whole or line by line as it arrives.

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

/// Headers in their order, names as they came.
pub type Headers = Vec<(String, String)>;

pub fn header<'a>(headers: &'a Headers, name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
}

/// One request, its body read whole.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    /// The path and the query, as aiohttp's `path_qs`.
    pub path_qs: String,
    pub headers: Headers,
    pub body: Bytes,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    pub fn query(&self, name: &str) -> Option<String> {
        let q = self.path_qs.split_once('?')?.1;
        q.split('&').find_map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (percent_decode(k) == name).then(|| percent_decode(v))
        })
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A response's body: whole, or the chunks a task writes into a channel.
pub enum Payload {
    Full(Bytes),
    Stream(mpsc::UnboundedReceiver<Bytes>),
}

pub struct Response {
    pub status: u16,
    pub headers: Headers,
    pub payload: Payload,
}

impl Response {
    pub fn full(status: u16, headers: Headers, body: impl Into<Bytes>) -> Response {
        Response { status, headers, payload: Payload::Full(body.into()) }
    }

    /// `web.json_response(data, status=...)`.
    pub fn json(status: u16, data: &serde_json::Value) -> Response {
        Response::full(status, vec![("Content-Type".into(), "application/json; charset=utf-8".into())], crate::py::dumps(data))
    }

    pub fn text(status: u16, text: &str) -> Response {
        Response::full(status, vec![("Content-Type".into(), "text/plain; charset=utf-8".into())], text.to_string())
    }

    /// A streamed response and the sender its chunks go into; dropping the sender ends it.
    pub fn stream(status: u16, headers: Headers) -> (Response, mpsc::UnboundedSender<Bytes>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Response { status, headers, payload: Payload::Stream(rx) }, tx)
    }
}

/// The body hyper sends.
pub struct OutBody {
    payload: Option<Payload>,
}

impl Body for OutBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.payload.as_mut() {
            None => Poll::Ready(None),
            Some(Payload::Full(_)) => {
                let Some(Payload::Full(b)) = self.payload.take() else { unreachable!() };
                if b.is_empty() { Poll::Ready(None) } else { Poll::Ready(Some(Ok(Frame::data(b)))) }
            }
            Some(Payload::Stream(rx)) => match rx.poll_recv(cx) {
                Poll::Ready(Some(b)) => Poll::Ready(Some(Ok(Frame::data(b)))),
                Poll::Ready(None) => {
                    self.payload = None;
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            },
        }
    }

    fn is_end_stream(&self) -> bool {
        match &self.payload {
            None => true,
            Some(Payload::Full(b)) => b.is_empty(),
            Some(Payload::Stream(_)) => false,
        }
    }

    fn size_hint(&self) -> SizeHint {
        match &self.payload {
            Some(Payload::Full(b)) => SizeHint::with_exact(b.len() as u64),
            None => SizeHint::with_exact(0),
            _ => SizeHint::default(),
        }
    }
}

pub type Handler = Rc<dyn Fn(Request) -> Pin<Box<dyn Future<Output = Response>>>>;

/// The largest request body read (aiohttp's `client_max_size` there).
pub const MAX_BODY: usize = 64 * 1024 * 1024;

/// Serve `handler` on `listener` until the task is dropped; each connection a task of the current thread.
pub async fn serve(listener: TcpListener, handler: Handler) {
    loop {
        let Ok((tcp, _)) = listener.accept().await else { continue };
        let _ = tcp.set_nodelay(true);
        let handler = handler.clone();
        tokio::task::spawn_local(async move {
            let svc = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                let handler = handler.clone();
                async move { Ok::<_, Infallible>(answer(handler, req).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new().keep_alive(true)
                .serve_connection(TokioIo::new(tcp), svc).await;
        });
    }
}

async fn answer(handler: Handler, req: hyper::Request<Incoming>) -> hyper::Response<OutBody> {
    let (parts, body) = req.into_parts();
    let headers: Headers = parts.headers.iter().map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned())).collect();
    let path = parts.uri.path().to_string();
    let path_qs = parts.uri.path_and_query().map_or(path.clone(), |p| p.as_str().to_string());
    let limited = http_body_util::Limited::new(body, MAX_BODY);
    let body = match limited.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => return build(Response::text(413, "request body too large")),
    };
    let request = Request { method: parts.method.as_str().to_string(), path, path_qs, headers, body };
    build(handler(request).await)
}

fn build(r: Response) -> hyper::Response<OutBody> {
    let mut b = hyper::Response::builder().status(r.status);
    for (k, v) in &r.headers {
        if let (Ok(name), Ok(value)) = (http::HeaderName::from_bytes(k.as_bytes()), http::HeaderValue::from_str(v)) {
            b = b.header(name, value);
        }
    }
    b.body(OutBody { payload: Some(r.payload) }).unwrap_or_else(|_| hyper::Response::new(OutBody { payload: None }))
}

// ── the client ──────────────────────────────────────────────────────────────────────────────────

fn tls_config() -> Arc<rustls::ClientConfig> {
    static C: std::sync::OnceLock<Arc<rustls::ClientConfig>> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        Arc::new(rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions().expect("TLS versions").with_root_certificates(roots).with_no_client_auth())
    }).clone()
}

/// An answer of the model's server: its status and headers, and its body as it arrives.
pub struct Upstream {
    pub status: u16,
    pub headers: Headers,
    body: Incoming,
    pending: Vec<u8>,
    read_timeout: Duration,
}

impl Upstream {
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    /// The next chunk of the body, or None at its end.
    pub async fn chunk(&mut self) -> Result<Option<Bytes>, String> {
        loop {
            let frame = tokio::time::timeout(self.read_timeout, self.body.frame()).await
                .map_err(|_| "the model's server stopped answering".to_string())?;
            match frame {
                None => return Ok(None),
                Some(Err(e)) => return Err(e.to_string()),
                Some(Ok(f)) => {
                    if let Ok(data) = f.into_data() {
                        return Ok(Some(data));
                    }
                }
            }
        }
    }

    /// The whole body.
    pub async fn read(&mut self) -> Result<Vec<u8>, String> {
        let mut out = std::mem::take(&mut self.pending);
        while let Some(c) = self.chunk().await? {
            out.extend_from_slice(&c);
        }
        Ok(out)
    }

    /// The next line (with its end), or None at the body's end; the last line may have no end.
    pub async fn line(&mut self) -> Result<Option<Vec<u8>>, String> {
        loop {
            if let Some(i) = self.pending.iter().position(|b| *b == b'\n') {
                let rest = self.pending.split_off(i + 1);
                return Ok(Some(std::mem::replace(&mut self.pending, rest)));
            }
            match self.chunk().await? {
                Some(c) => self.pending.extend_from_slice(&c),
                None => {
                    if self.pending.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some(std::mem::take(&mut self.pending)));
                }
            }
        }
    }
}

struct Target {
    https: bool,
    host: String,
    port: u16,
    authority: String,
    path: String,
}

fn target(url: &str) -> Result<Target, String> {
    let uri: http::Uri = url.parse().map_err(|e| format!("{url}: {e}"))?;
    let https = uri.scheme_str() == Some("https");
    let host = uri.host().ok_or_else(|| format!("{url}: no host"))?.trim_matches(['[', ']']).to_string();
    let port = uri.port_u16().unwrap_or(if https { 443 } else { 80 });
    let authority = uri.authority().map_or(host.clone(), |a| a.as_str().to_string());
    let path = uri.path_and_query().map_or("/".to_string(), |p| p.as_str().to_string());
    Ok(Target { https, host, port, authority, path })
}

/// One request to `url` (a new connection, as each POST through the proxy was its own); the answer, its
/// body still to read. `connect` and `read` are the timeouts aiohttp's session had.
pub async fn fetch(method: &str, url: &str, headers: &Headers, body: Bytes, connect: Duration, read: Duration) -> Result<Upstream, String> {
    let t = target(url)?;
    let tcp = tokio::time::timeout(connect, TcpStream::connect((t.host.as_str(), t.port))).await
        .map_err(|_| format!("Cannot connect to host {}:{}", t.host, t.port))?
        .map_err(|e| format!("Cannot connect to host {}:{} [{e}]", t.host, t.port))?;
    let _ = tcp.set_nodelay(true);
    let mut req = hyper::Request::builder().method(method).uri(t.path.as_str());
    let mut has_host = false;
    for (k, v) in headers {
        has_host |= k.eq_ignore_ascii_case("host");
        if let (Ok(n), Ok(val)) = (http::HeaderName::from_bytes(k.as_bytes()), http::HeaderValue::from_str(v)) {
            req = req.header(n, val);
        }
    }
    if !has_host {
        req = req.header(http::header::HOST, t.authority.as_str());
    }
    if !body.is_empty() || method == "POST" {
        req = req.header(http::header::CONTENT_LENGTH, body.len());
    }
    let req = req.body(http_body_util::Full::new(body)).map_err(|e| e.to_string())?;
    let resp = if t.https {
        let name = rustls::pki_types::ServerName::try_from(t.host.clone()).map_err(|e| e.to_string())?;
        let tls = tokio_rustls::TlsConnector::from(tls_config()).connect(name, tcp).await.map_err(|e| e.to_string())?;
        let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tls)).await.map_err(|e| e.to_string())?;
        tokio::task::spawn_local(async move {
            let _ = conn.await;
        });
        send.send_request(req).await.map_err(|e| e.to_string())?
    } else {
        let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tcp)).await.map_err(|e| e.to_string())?;
        tokio::task::spawn_local(async move {
            let _ = conn.await;
        });
        send.send_request(req).await.map_err(|e| e.to_string())?
    };
    let (parts, body) = resp.into_parts();
    let headers = parts.headers.iter().map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned())).collect();
    Ok(Upstream { status: parts.status.as_u16(), headers, body, pending: Vec::new(), read_timeout: read })
}
