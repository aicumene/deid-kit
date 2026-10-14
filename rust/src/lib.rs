// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! deid-kit in Rust — the same de-identification as the Python package `deidkit`, for the desktop's
//! agent runtime without Python: the vault and its name fold, deid-proxy and deid-agent. Python's own
//! text rules (`unicode`, `py`) are taken from Python's Unicode data, so a vault key made here is the key
//! the Python vault made.

pub mod namefold;
pub mod py;
pub mod pyre;
pub mod textmatch;
pub mod unicode;
pub mod unicode_tables;
