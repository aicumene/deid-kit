// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The residual PII detector the vault asks about text its seed graph does not cover (as `detect.py`),
//! and the document's language (as `lang.py`).

/// A label asked for and never tokenised: a span the detector read as a location vetoes a person.
pub const VETO_KIND: &str = "__LOCATION__";

/// Kinds whose detection is a pattern, not a judgement: admitted on sight.
pub const STRUCTURAL_KINDS: [&str; 7] = ["EMAIL", "PHONE", "IBAN", "CARD", "ID", "ACCOUNT", "IP"];

pub fn structural(kind: &str) -> bool {
    STRUCTURAL_KINDS.contains(&kind)
}

/// Residual PII detection: de-duplicated `(span_text, kind)` pairs, never a span holding a token.
pub trait PiiDetector {
    fn detect(&self, text: &str, language: &str) -> Vec<(String, String)>;
}

/// 'ru' if the text is predominantly Cyrillic (30 % of its letters), else 'en'.
pub fn detect_language(text: &str) -> &'static str {
    let (mut cyr, mut lat) = (0usize, 0usize);
    for ch in text.chars() {
        if ('Ѐ'..='ӿ').contains(&ch) {
            cyr += 1;
        } else if ch.is_ascii_alphabetic() {
            lat += 1;
        }
    }
    let letters = cyr + lat;
    if letters == 0 {
        return "en";
    }
    if cyr as f64 / letters as f64 >= 0.30 { "ru" } else { "en" }
}
