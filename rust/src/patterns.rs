// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A residual detector made of patterns only (as `patterns.py`): e-mail addresses, IBANs (ISO 13616
//! mod 97), payment cards (Luhn and a network's prefix and length), phone numbers in international
//! form. It recognises nothing by judgement — the detector for code, logs and command output.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::detect::PiiDetector;
use crate::namefold as nf;
use crate::pyre::{self, Regex, NONE};
use crate::unicode as u;

fn email() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, concat!(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+",
                           r"[A-Za-z]{2,24}(?![A-Za-z0-9-])"), NONE)
}

fn iban() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?![A-Za-z0-9])", NONE)
}

fn card() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])", NONE)
}

fn phone() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![\w+])\+\d(?:[ .()-]{0,2}\d){7,14}(?!\d)", NONE)
}

const ROLE_LOCAL_PARTS: [&str; 6] = ["noreply", "no-reply", "donotreply", "do-not-reply", "mailer-daemon", "postmaster"];

/// ISO 13616: the moved IBAN's base-36 digits, mod 97, are 1.
pub fn iban_ok(value: &str) -> bool {
    let s = u::upper(&value.replace(' ', ""));
    let n = py_len(&s);
    if !(15..=34).contains(&n) {
        return false;
    }
    let chars: Vec<char> = s.chars().collect();
    let moved: Vec<char> = chars[4..].iter().chain(&chars[..4]).copied().collect();
    let mut rem: u64 = 0;
    for c in moved {
        let Some(v) = base36(c) else { return false };
        for d in v.to_string().chars() {
            rem = (rem * 10 + d.to_digit(10).unwrap() as u64) % 97;
        }
    }
    rem == 1
}

/// `int(c, 36)` of one character.
fn base36(c: char) -> Option<u32> {
    if let Some(d) = u::digit_value(c) {
        return Some(d);
    }
    let l = c.to_ascii_lowercase();
    l.is_ascii_lowercase().then(|| l as u32 - 'a' as u32 + 10)
}

fn py_len(s: &str) -> usize {
    s.chars().count()
}

/// The longest prefix of a match that is a valid IBAN.
fn longest_iban(m: &[char]) -> Option<String> {
    let mut end = m.len();
    while end > 14 {
        let cut: String = m[..end].iter().collect();
        let stripped = crate::py::rstrip(&cut).to_string();
        let ok_cut = stripped == cut && !(end < m.len() && m[end] != ' ');
        if ok_cut && iban_ok(&stripped) {
            return Some(stripped);
        }
        end -= 1;
    }
    None
}

fn networks() -> &'static [(Regex, &'static [usize])] {
    static N: OnceLock<Vec<(Regex, &'static [usize])>> = OnceLock::new();
    N.get_or_init(|| vec![
        (Regex::new(r"^4").unwrap(), &[13, 16, 19][..]),
        (Regex::new(r"^(?:5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720))").unwrap(), &[16][..]),
        (Regex::new(r"^3[47]").unwrap(), &[15][..]),
        (Regex::new(r"^(?:6011|65|64[4-9])").unwrap(), &[16, 19][..]),
        (Regex::new(r"^35(?:2[89]|[3-8]\d)").unwrap(), &[16, 19][..]),
        (Regex::new(r"^3(?:0[0-5]|[68])").unwrap(), &[14][..]),
    ])
}

pub fn card_ok(value: &str) -> bool {
    let digits: String = value.chars().filter(|c| u::is_digit(*c)).collect();
    let dchars: Vec<char> = digits.chars().collect();
    luhn_ok(&digits) && networks().iter().any(|(p, lengths)| p.match_at(&dchars, 0).is_some() && lengths.contains(&dchars.len()))
}

pub fn luhn_ok(value: &str) -> bool {
    let digits: Vec<u32> = value.chars().filter(|c| u::is_digit(*c)).filter_map(u::digit_value).collect();
    if !(13..=19).contains(&digits.len()) || digits.iter().collect::<HashSet<_>>().len() == 1 {
        return false;
    }
    let mut total = 0;
    for (i, d) in digits.iter().rev().enumerate() {
        let mut d = *d;
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        total += d;
    }
    total % 10 == 0
}

/// The text with each backslash escape replaced by spaces of the same length, so offsets hold.
pub fn blank_escapes(text: &[char]) -> Vec<char> {
    if !text.contains(&'\\') {
        return text.to_vec();
    }
    let mut out = text.to_vec();
    let mut i = 0;
    while i + 1 < text.len() {
        if text[i] == '\\' {
            let next = text[i + 1];
            if "ntrfbv".contains(next) {
                out[i] = ' ';
                out[i + 1] = ' ';
                i += 2;
                continue;
            }
            if "uU".contains(next) && i + 5 < text.len() && text[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit()) {
                out[i..i + 6].iter_mut().for_each(|c| *c = ' ');
                i += 6;
                continue;
            }
            // an escaped character (\\, \") is skipped whole
            i += 2;
            continue;
        }
        i += 1;
    }
    out
}

fn email_is_personal(value: &str) -> bool {
    let local = value.rsplit_once('@').map_or("", |(l, _)| l);
    !ROLE_LOCAL_PARTS.contains(&u::lower(local).as_str())
}

/// A detector over patterns only. `kinds` narrows what it reports.
pub struct RegexDetector {
    pub kinds: Vec<String>,
}

impl Default for RegexDetector {
    fn default() -> Self {
        RegexDetector { kinds: ["EMAIL", "IBAN", "CARD", "PHONE"].iter().map(|s| s.to_string()).collect() }
    }
}

impl PiiDetector for RegexDetector {
    fn detect(&self, text: &str, _language: &str) -> Vec<(String, String)> {
        let raw: Vec<char> = text.chars().collect();
        let blank = blank_escapes(&raw);
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut found: Vec<(String, String)> = Vec::new();
        let mut add = |span: String, kind: &str| {
            if self.kinds.iter().any(|k| k == kind) && !crate::py::strip(&span).is_empty() && nf::token_in_text(&span).is_none()
                && seen.insert((nf::key(&span), kind.to_string())) {
                found.push((span, kind.to_string()));
            }
        };
        let at = |m: &pyre::Match| -> String { raw[m.start..m.end].iter().collect() };
        for m in email().finditer(&blank) {
            if email_is_personal(&at(&m)) {
                add(at(&m), "EMAIL");
            }
        }
        for m in iban().finditer(&blank) {
            if let Some(v) = longest_iban(&raw[m.start..m.end]) {
                add(v, "IBAN");
            }
        }
        for m in card().finditer(&blank) {
            if card_ok(&at(&m)) {
                add(at(&m), "CARD");
            }
        }
        for m in phone().finditer(&blank) {
            let n = at(&m).chars().filter(|c| u::is_digit(*c)).count();
            if (8..=15).contains(&n) {
                add(at(&m), "PHONE");
            }
        }
        found
    }
}
