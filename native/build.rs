// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The tables come from the interpreter the module is built for (maturin names it in PYO3_PYTHON): its
//! Unicode database is the one deidkit.namefold folds with, so the two folds answer from the same data.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let python = env::var("PYO3_PYTHON")
        .or_else(|_| env::var("PYTHON_SYS_EXECUTABLE"))
        .unwrap_or_else(|_| "python3".to_string());
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("out dir")).join("tables.rs");
    let status = Command::new(&python)
        .arg(here.join("gen_tables.py"))
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("{python} did not run the table generator: {e}"));
    assert!(status.success(), "{python} could not make the tables");
    println!("cargo:rerun-if-changed=gen_tables.py");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
}
