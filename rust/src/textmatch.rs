// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The pipeline-invariant comparison form of a text (as `textmatch.py`): fold what a pipeline can change
//! without a human changing a word — line-break hyphens, compatibility forms, format characters,
//! punctuation shapes, whitespace, case — and nothing else.

use crate::unicode;

/// `_LINEBREAK_HYPHEN`'s hyphens: `[-‐‑‒–—−]`.
pub const HYPHENS: [char; 7] = ['-', '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}'];

/// Shapes a pipeline swaps freely, folded to the ASCII form on both sides.
pub fn punct_fold(c: char) -> Option<char> {
    Some(match c as u32 {
        0x2010..=0x2015 | 0x2212 => '-',
        0x2018 | 0x2019 | 0x201A | 0x201B | 0x2032 => '\'',
        0x201C | 0x201D | 0x201E | 0x201F | 0x2033 | 0x00AB | 0x00BB => '"',
        _ => return None,
    })
}

/// The characters `_LINEBREAK_HYPHEN.finditer` covers: a hyphen and the run of whitespace after it, when
/// the run holds a line break.
pub fn linebreak_hyphens(chars: &[char]) -> Vec<bool> {
    let n = chars.len();
    let mut skip = vec![false; n];
    let mut i = 0;
    while i < n {
        if HYPHENS.contains(&chars[i]) {
            let mut j = i + 1;
            let mut newline = false;
            while j < n && unicode::is_space(chars[j]) {
                newline |= chars[j] == '\n';
                j += 1;
            }
            if newline {
                skip[i..j].iter_mut().for_each(|s| *s = true);
                i = j;
                continue;
            }
        }
        i += 1;
    }
    skip
}

/// Pipeline-invariant comparison form, applied to BOTH sides of every comparison.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let skip = linebreak_hyphens(&chars);
    let kept: String = chars.iter().zip(&skip).filter(|(_, s)| !**s).map(|(c, _)| *c).collect();
    let mut s = unicode::normalize("NFKC", &kept);
    if !s.is_ascii() {
        s = s.chars().filter(|c| unicode::category(*c) != "Cf").collect();
    }
    let s: String = s.chars().map(|c| punct_fold(c).unwrap_or(c)).collect();
    unicode::casefold(&squash(&s))
}

/// `re.sub(r"\s+", " ", s).strip()`: a run of whitespace is one space, none at the ends.
pub fn squash(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if unicode::is_space(c) {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(c);
    }
    out
}
