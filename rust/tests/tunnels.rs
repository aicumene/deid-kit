// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! deid-agent's tunnels (`http::serve_with_tunnels`): a CONNECT to a place it names opens a tunnel that passes
//! the bytes both ways; a CONNECT anywhere else is refused; the server's own routes are as they were.

use std::rc::Rc;

use deidkit::http::{self, Handler, Response};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn head(stream: &mut TcpStream) -> String {
    let mut got = Vec::new();
    let mut byte = [0u8; 1];
    while !got.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).await.unwrap() == 0 {
            break;
        }
        got.push(byte[0]);
    }
    String::from_utf8_lossy(&got).into_owned()
}

#[test]
fn a_connect_opens_a_tunnel_only_where_it_is_named() {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        // the far end: says back what it hears
        let echo = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let far = echo.local_addr().unwrap().port();
        tokio::task::spawn_local(async move {
            let (mut s, _) = echo.accept().await.unwrap();
            let mut buf = [0u8; 64];
            let n = s.read(&mut buf).await.unwrap();
            s.write_all(&buf[..n]).await.unwrap();
        });
        let handler: Handler = Rc::new(|_req| Box::pin(async { Response::text(200, "the page") }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::task::spawn_local(http::serve_with_tunnels(listener, handler, Rc::new(vec![format!("127.0.0.1:{far}")])));

        let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(format!("CONNECT 127.0.0.1:{far} HTTP/1.1\r\nHost: 127.0.0.1:{far}\r\n\r\n").as_bytes()).await.unwrap();
        assert!(head(&mut s).await.starts_with("HTTP/1.1 200"));
        s.write_all(b"through").await.unwrap();
        let mut back = [0u8; 7];
        s.read_exact(&mut back).await.unwrap();
        assert_eq!(&back, b"through");

        let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n").await.unwrap();
        assert!(head(&mut s).await.starts_with("HTTP/1.1 403"));

        let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(format!("CONNECT 127.0.0.1:{} HTTP/1.1\r\nHost: x\r\n\r\n", far + 1).as_bytes()).await.unwrap();
        assert!(head(&mut s).await.starts_with("HTTP/1.1 403"));

        let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").await.unwrap();
        let mut page = String::new();
        s.read_to_string(&mut page).await.unwrap();
        assert!(page.starts_with("HTTP/1.1 200") && page.ends_with("the page"));
    });
}
