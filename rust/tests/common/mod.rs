// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What the parity tests share: the vectors and a report of where Python and Rust differ.
#![allow(dead_code)]

use std::path::PathBuf;

use serde_json::{json, Value};

/// The vectors `name` written by `tools/vectors_*.py` into `DEIDKIT_PARITY`, or None (the test is skipped).
pub fn vectors(name: &str) -> Option<(PathBuf, Value)> {
    let dir = PathBuf::from(std::env::var_os("DEIDKIT_PARITY")?);
    let data = std::fs::read_to_string(dir.join(name)).ok()?;
    Some((dir, serde_json::from_str(&data).unwrap()))
}

pub fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

pub struct Diff(pub Vec<String>);

/// Where two values first differ, and what each has there.
pub fn first_difference(path: String, a: &Value, b: &Value) -> (String, Value, Value) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, va) in x {
                match y.get(k) {
                    Some(vb) if vb == va => {}
                    Some(vb) => return first_difference(format!("{path}.{k}"), va, vb),
                    None => return (format!("{path}.{k}"), va.clone(), Value::Null),
                }
            }
            (format!("{path} (keys)"), json!(x.keys().collect::<Vec<_>>()), json!(y.keys().collect::<Vec<_>>()))
        }
        (Value::Array(x), Value::Array(y)) => {
            for (i, (va, vb)) in x.iter().zip(y).enumerate() {
                if va != vb {
                    return first_difference(format!("{path}[{i}]"), va, vb);
                }
            }
            (format!("{path} (length)"), json!(x.len()), json!(y.len()))
        }
        (Value::String(x), Value::String(y)) => {
            let at = x.chars().zip(y.chars()).take_while(|(p, q)| p == q).count();
            let from = at.saturating_sub(60);
            let cut = |t: &str| t.chars().skip(from).take(160).collect::<String>();
            (format!("{path} (from char {from})"), json!(cut(x)), json!(cut(y)))
        }
        _ => (path, a.clone(), b.clone()),
    }
}

impl Diff {
    pub fn eq(&mut self, what: impl std::fmt::Display, expected: &Value, got: &Value) {
        if expected != got {
            let (at, e, g) = first_difference(String::new(), expected, got);
            self.0.push(format!("{what} at {at}\n  python: {e}\n  rust:   {g}"));
        }
    }
}

impl Diff {
    pub fn new() -> Diff {
        Diff(Vec::new())
    }

    pub fn done(self) {
        assert!(self.0.is_empty(), "{} differences:\n{}", self.0.len(), self.0.join("\n"));
    }
}
