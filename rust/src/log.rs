// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Logs on stderr, as Python's `logging` writes them for the agent:
//! `2026-10-14 11:59:01,123 matteragent WARNING …`.

use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

static LEVEL: AtomicU8 = AtomicU8::new(20);
static NAME: OnceLock<String> = OnceLock::new();

/// `--log-level`: DEBUG, INFO, WARNING, ERROR or CRITICAL (or a number).
pub fn init(name: &str, level: &str) -> Result<(), String> {
    let n = match level.to_uppercase().as_str() {
        "DEBUG" => 10,
        "INFO" => 20,
        "WARNING" | "WARN" => 30,
        "ERROR" => 40,
        "CRITICAL" | "FATAL" => 50,
        other => other.parse().map_err(|_| format!("Unknown level: {}", crate::py::repr_str(level)))?,
    };
    LEVEL.store(n, Ordering::Relaxed);
    let _ = NAME.set(name.to_string());
    Ok(())
}

fn now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = t.as_secs() as i64;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let raw = secs as libc::time_t;
    unsafe { libc::localtime_r(&raw, &mut tm) };
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02},{:03}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
            tm.tm_min, tm.tm_sec, t.subsec_millis())
}

pub fn write(level: u8, label: &str, message: &str) {
    if level < LEVEL.load(Ordering::Relaxed) {
        return;
    }
    let name = NAME.get().map_or("deidkit", String::as_str);
    let _ = writeln!(std::io::stderr().lock(), "{} {name} {label} {message}", now());
}

pub fn info(message: &str) {
    write(20, "INFO", message);
}

pub fn warning(message: &str) {
    write(30, "WARNING", message);
}

pub fn error(message: &str) {
    write(40, "ERROR", message);
}
