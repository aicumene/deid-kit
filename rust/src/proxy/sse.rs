// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Server-sent events: read them from lines, write them back (as `sse.py`).

use serde_json::Value;

use crate::py;

/// Collects `(event name, data)` from lines as they come; `("", ":comment")` for a comment line.
#[derive(Default)]
pub struct Reader {
    name: String,
    data: Vec<String>,
}

impl Reader {
    /// One line (without its end); the events it completes.
    pub fn line(&mut self, raw: &str) -> Vec<(String, String)> {
        let line = raw.trim_end_matches(['\r', '\n']);
        let mut out = Vec::new();
        if line.is_empty() {
            if !self.data.is_empty() || !self.name.is_empty() {
                out.push((std::mem::take(&mut self.name), std::mem::take(&mut self.data).join("\n")));
            }
            self.name.clear();
            self.data.clear();
        } else if line.starts_with(':') {
            out.push((String::new(), line.to_string()));
        } else if let Some(rest) = line.strip_prefix("event:") {
            self.name = py::strip(rest).to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            self.data.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        }
        out
    }

    /// The end of the stream: the event still open, if any.
    pub fn finish(&mut self) -> Option<(String, String)> {
        (!self.data.is_empty() || !self.name.is_empty())
            .then(|| (std::mem::take(&mut self.name), std::mem::take(&mut self.data).join("\n")))
    }
}

/// One event as bytes: `event: name` and a `data:` line per line of the data (JSON, compact).
pub fn encode(name: &str, data: &Value) -> Vec<u8> {
    let body = match data {
        Value::String(s) => s.clone(),
        other => py::dumps_with(other, false, true),
    };
    encode_text(name, &body)
}

pub fn encode_text(name: &str, body: &str) -> Vec<u8> {
    let mut out = String::new();
    if !name.is_empty() {
        out.push_str(&format!("event: {name}\n"));
    }
    for line in body.split('\n') {
        out.push_str(&format!("data: {line}\n"));
    }
    out.push('\n');
    out.into_bytes()
}
