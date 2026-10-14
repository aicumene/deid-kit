// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! pyre answers as Python's re does: random patterns of every feature over random texts, and every
//! literal pattern of deid-kit over texts of its kind — finditer with every group, fullmatch, split, sub,
//! and whether the pattern compiles. Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_pyre.py <dir>`.

mod common;

use deidkit::pyre::{Flags, Regex};
use serde_json::{json, Value};

fn flags(f: u64) -> Flags {
    Flags { ignorecase: f & 2 != 0, multiline: f & 8 != 0, dotall: f & 16 != 0, verbose: f & 64 != 0, ascii: f & 256 != 0 }
}

#[test]
fn re_as_python_has_it() {
    let Some((_, v)) = common::vectors("pyre.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let mut bad: Vec<String> = Vec::new();
    let cases = v["cases"].as_array().unwrap();
    for c in cases {
        let p = c["p"].as_str().unwrap();
        let t: Vec<char> = c["t"].as_str().unwrap().chars().collect();
        let rx = Regex::with_flags(p, flags(c["f"].as_u64().unwrap()));
        let got = match &rx {
            Err(_) => json!({"error": true}),
            Ok(rx) => {
                let it: Vec<Value> = rx.finditer(&t).iter().map(|m| {
                    let groups: Vec<Value> = (1..=rx.groups).map(|g| match m.group(g) {
                        Some((a, b)) => json!([a, b]),
                        None => json!([-1, -1]),
                    }).collect();
                    json!([m.start, m.end, groups])
                }).collect();
                let fm = rx.fullmatch(&t).map(|m| json!([m.start, m.end])).unwrap_or(Value::Null);
                let split: Vec<Value> = rx.split(&t, 0).into_iter().map(|s| s.map_or(Value::Null, Value::String)).collect();
                let sub = rx.sub("<\\g<0>>", &t, 0).map(|(s, _)| Value::String(s)).unwrap_or(Value::Null);
                json!({"finditer": it, "fullmatch": fm, "split": split, "sub": sub})
            }
        };
        let mut want = c.clone();
        let w = want.as_object_mut().unwrap();
        w.remove("p");
        w.remove("f");
        w.remove("t");
        if got != want && bad.len() < 40 {
            let err = rx.as_ref().err().cloned().unwrap_or_default();
            bad.push(format!("{p:?} flags {} on {:?}\n  python: {want}\n  rust:   {got} {err}", c["f"], c["t"]));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
}
