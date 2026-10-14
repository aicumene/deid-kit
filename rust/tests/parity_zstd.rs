// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The Zstandard decoder reads back what the zstandard package wrote: every level, with and without a
//! checksum or a content size, streamed in several blocks. Needs `DEIDKIT_PARITY=<dir>` after
//! `python tools/vectors_zstd.py <dir>`.

mod common;

fn b64(s: &str) -> Vec<u8> {
    let t: Vec<u8> = s.bytes().filter(|b| *b != b'=').map(|b| match b {
        b'A'..=b'Z' => b - b'A', b'a'..=b'z' => b - b'a' + 26, b'0'..=b'9' => b - b'0' + 52, b'+' => 62, _ => 63,
    }).collect();
    let mut out = Vec::new();
    for c in t.chunks(4) {
        let v = c.iter().enumerate().fold(0u32, |acc, (i, x)| acc | (*x as u32) << (18 - 6 * i));
        out.push((v >> 16) as u8);
        if c.len() > 2 { out.push((v >> 8) as u8); }
        if c.len() > 3 { out.push(v as u8); }
    }
    out
}

#[test]
fn frames_of_the_zstandard_package_read_back() {
    let Some((_, v)) = common::vectors("zstd.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let mut bad = Vec::new();
    let cases = v.as_array().unwrap();
    for (i, c) in cases.iter().enumerate() {
        let plain = b64(c["plain"].as_str().unwrap());
        let frame = b64(c["frame"].as_str().unwrap());
        match deidkit::zstd::decompress(&frame, 256 << 20) {
            Ok(out) if out == plain => {}
            Ok(out) => bad.push(format!("case {i} level {} {}: {} bytes, want {}", c["level"], c["kw"], out.len(), plain.len())),
            Err(e) => bad.push(format!("case {i} level {} {} ({} bytes): {e}", c["level"], c["kw"], plain.len())),
        }
    }
    assert!(bad.is_empty(), "{} of {} frames:\n{}", bad.len(), cases.len(), bad.join("\n"));
}
