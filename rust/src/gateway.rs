// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What a crossing produces and what it audits (as `gateway.py`): the de-identified text with the
//! mapping that reverses it, and a record of identifiers and counts only — never content.

use sha2::{Digest, Sha256};

use crate::odict::ODict;

#[derive(Clone, Debug, Default)]
pub struct Redaction {
    pub text: String,
    pub entity_types: Vec<String>,
    pub redacted_count: usize,
    /// token → real value, for re-hydrating the answer on the trusted plane.
    pub mapping: ODict<String>,
    pub glossary: Vec<String>,
    /// Per token, the raw surfaces this text used.
    pub surfaces: ODict<Vec<String>>,
    /// Per token, the vault's canonical value.
    pub canonical: ODict<String>,
}

impl Redaction {
    pub fn of(text: &str) -> Redaction {
        Redaction { text: text.to_string(), ..Default::default() }
    }
}

/// SHA-256 (hex) of the UTF-8 text — the hash a crossing is audited under.
pub fn content_hash(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}
