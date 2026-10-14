// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Python's text rules over Python's own Unicode data (`unicode_tables.rs`, made by tools/gen_unicode.py
//! from the interpreter the desktop carried): `unicodedata.normalize`, `category`, `combining`, and the
//! `str` methods deid-kit asks — `isalpha`, `isupper`, `lower`, `capitalize`, `casefold` and the rest —
//! answering exactly as CPython 3.13 does, so a vault key made here is the key Python made.

use crate::unicode_tables as t;

fn in_ranges(table: &[(u32, u32)], c: u32) -> bool {
    table.binary_search_by(|&(a, b)| {
        if c < a {
            std::cmp::Ordering::Greater
        } else if c > b {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }).is_ok()
}

fn in_runs(table: &[(u32, u32, u8)], c: u32) -> Option<u8> {
    table.binary_search_by(|&(a, b, _)| {
        if c < a {
            std::cmp::Ordering::Greater
        } else if c > b {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }).ok().map(|i| table[i].2)
}

fn mapped(table: &'static [(u32, &'static str)], c: char) -> Option<&'static str> {
    table.binary_search_by_key(&(c as u32), |(k, _)| *k).ok().map(|i| table[i].1)
}

// ── unicodedata ─────────────────────────────────────────────────────────────────────────────────

/// `unicodedata.category(c)`.
pub fn category(c: char) -> &'static str {
    t::CATEGORY_NAMES[in_runs(&t::CATEGORY, c as u32).unwrap_or(0) as usize]
}

/// `unicodedata.combining(c)`.
pub fn combining(c: char) -> u8 {
    in_runs(&t::CCC, c as u32).unwrap_or(0)
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decompose_into(c: u32, compat: bool, out: &mut Vec<u32>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&c) {
        let s = c - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if s % T_COUNT != 0 {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    match t::DECOMPOSITION.binary_search_by_key(&c, |(k, _, _)| *k) {
        Ok(i) if compat || !t::DECOMPOSITION[i].1 => {
            for &p in t::DECOMPOSITION[i].2 {
                decompose_into(p, compat, out);
            }
        }
        _ => out.push(c),
    }
}

fn ccc(c: u32) -> u8 {
    in_runs(&t::CCC, c).unwrap_or(0)
}

/// Canonical ordering: a stable sort of every run of non-starters by combining class.
fn reorder(cps: &mut [u32]) {
    let mut i = 0;
    while i < cps.len() {
        if ccc(cps[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < cps.len() && ccc(cps[i]) != 0 {
            i += 1;
        }
        cps[start..i].sort_by_key(|c| ccc(*c));
    }
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a) && (a - S_BASE) % T_COUNT == 0 && (T_BASE + 1..T_BASE + T_COUNT).contains(&b) {
        return Some(a + (b - T_BASE));
    }
    t::COMPOSITION.binary_search_by_key(&(a, b), |(x, y, _)| (*x, *y)).ok().map(|i| t::COMPOSITION[i].2)
}

/// Canonical composition (UAX #15): each character joins the last starter unless something between
/// them blocks it.
fn compose(cps: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(cps.len());
    let mut starter: Option<usize> = None;
    // the combining class of the last character left between the starter and the next one
    let mut last_cc: u8 = 0;
    for &c in cps {
        let cc = ccc(c);
        if let Some(si) = starter {
            let between = out.len() > si + 1;
            let blocked = between && (last_cc == 0 || last_cc >= cc);
            if !blocked {
                if let Some(composite) = compose_pair(out[si], c) {
                    out[si] = composite;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
        }
        last_cc = cc;
        out.push(c);
    }
    out
}

/// `unicodedata.normalize(form, text)` for NFC, NFD, NFKC and NFKD.
pub fn normalize(form: &str, text: &str) -> String {
    let compat = form == "NFKC" || form == "NFKD";
    let mut cps = Vec::with_capacity(text.len());
    for c in text.chars() {
        decompose_into(c as u32, compat, &mut cps);
    }
    reorder(&mut cps);
    let cps = if form == "NFC" || form == "NFKC" { compose(&cps) } else { cps };
    cps.into_iter().filter_map(char::from_u32).collect()
}

// ── str predicates, per character ───────────────────────────────────────────────────────────────

pub fn is_space(c: char) -> bool {
    in_ranges(&t::SPACE, c as u32)
}

pub fn is_alpha(c: char) -> bool {
    matches!(category(c), "Lu" | "Ll" | "Lt" | "Lm" | "Lo")
}

pub fn is_decimal(c: char) -> bool {
    category(c) == "Nd"
}

pub fn is_digit(c: char) -> bool {
    in_ranges(&t::DIGIT, c as u32)
}

pub fn is_numeric(c: char) -> bool {
    in_ranges(&t::NUMERIC, c as u32)
}

pub fn is_alnum(c: char) -> bool {
    is_alpha(c) || is_decimal(c) || is_digit(c) || is_numeric(c)
}

pub fn is_lower(c: char) -> bool {
    in_ranges(&t::LOWERCASE, c as u32)
}

pub fn is_upper(c: char) -> bool {
    in_ranges(&t::UPPERCASE, c as u32)
}

pub fn is_title(c: char) -> bool {
    category(c) == "Lt"
}

/// Unicode's Cased: Lowercase, Uppercase or titlecase.
pub fn is_cased(c: char) -> bool {
    is_lower(c) || is_upper(c) || is_title(c)
}

/// Unicode's Case_Ignorable: marks, format characters, modifier letters and symbols, and the
/// apostrophes, full stops and colons a word may hold.
pub fn is_case_ignorable(c: char) -> bool {
    matches!(category(c), "Mn" | "Me" | "Cf" | "Lm" | "Sk")
        || matches!(c, '\'' | '.' | ':' | '^' | '`' | '\u{b7}' | '\u{2018}' | '\u{2019}' | '\u{2024}' | '\u{2027}' | '\u{fe13}'
                       | '\u{fe52}' | '\u{fe55}' | '\u{ff07}' | '\u{ff0e}' | '\u{ff1a}' | '\u{387}' | '\u{55f}' | '\u{5f4}')
}

/// `re`'s `\w` on str.
pub fn is_word(c: char) -> bool {
    in_ranges(&t::WORD, c as u32)
}

pub fn is_printable(c: char) -> bool {
    in_ranges(&t::PRINTABLE, c as u32)
}

/// The value of a decimal digit of any script, as `int()` reads it.
pub fn digit_value(c: char) -> Option<u32> {
    let c = c as u32;
    let i = t::DIGIT_ZEROS.partition_point(|z| *z <= c);
    let zero = *t::DIGIT_ZEROS.get(i.checked_sub(1)?)?;
    (c - zero < 10).then_some(c - zero)
}

// ── str methods ─────────────────────────────────────────────────────────────────────────────────

fn all_nonempty(s: &str, f: impl Fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(f)
}

pub fn isalpha(s: &str) -> bool {
    all_nonempty(s, is_alpha)
}

pub fn isalnum(s: &str) -> bool {
    all_nonempty(s, is_alnum)
}

pub fn isdigit(s: &str) -> bool {
    all_nonempty(s, is_digit)
}

pub fn isdecimal(s: &str) -> bool {
    all_nonempty(s, is_decimal)
}

pub fn isnumeric(s: &str) -> bool {
    all_nonempty(s, is_numeric)
}

pub fn isspace(s: &str) -> bool {
    all_nonempty(s, is_space)
}

pub fn isascii(s: &str) -> bool {
    s.is_ascii()
}

/// `str.isupper()`: every cased character upper case, and one at least.
pub fn isupper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if is_lower(c) || is_title(c) {
            return false;
        }
        cased |= is_upper(c);
    }
    cased
}

/// `str.islower()`.
pub fn islower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if is_upper(c) || is_title(c) {
            return false;
        }
        cased |= is_lower(c);
    }
    cased
}

/// Σ lower-cased where it stands: ς at the end of a word, σ elsewhere (CPython's handle_capital_sigma).
fn sigma(chars: &[char], i: usize) -> char {
    let mut j = i;
    let mut before = None;
    while j > 0 {
        j -= 1;
        if !is_case_ignorable(chars[j]) {
            before = Some(chars[j]);
            break;
        }
    }
    let mut final_sigma = before.is_some_and(is_cased);
    if final_sigma && i + 1 < chars.len() {
        let after = chars[i + 1..].iter().copied().find(|c| !is_case_ignorable(*c));
        final_sigma = after.is_none_or(|c| !is_cased(c));
    }
    if final_sigma { '\u{3c2}' } else { '\u{3c3}' }
}

fn lower_at(chars: &[char], i: usize, out: &mut String) {
    let c = chars[i];
    if c == '\u{3a3}' {
        out.push(sigma(chars, i));
    } else {
        match mapped(&t::LOWER, c) {
            Some(m) => out.push_str(m),
            None => out.push(c),
        }
    }
}

pub fn lower(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..chars.len() {
        lower_at(&chars, i, &mut out);
    }
    out
}

pub fn upper(s: &str) -> String {
    s.chars().map(|c| mapped(&t::UPPER, c).map_or_else(|| c.to_string(), str::to_string)).collect()
}

pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match mapped(&t::CASEFOLD, c) {
            Some(m) => out.push_str(m),
            None => out.push(c),
        }
    }
    out
}

fn title_of(c: char, out: &mut String) {
    match mapped(&t::TITLE, c) {
        Some(m) => out.push_str(m),
        None => out.push(c),
    }
}

/// `str.capitalize()`: the first character in title case, the rest lower case.
pub fn capitalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for i in 0..chars.len() {
        if i == 0 { title_of(chars[0], &mut out) } else { lower_at(&chars, i, &mut out) }
    }
    out
}

/// `str.title()`.
pub fn title(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut previous_cased = false;
    for i in 0..chars.len() {
        let c = chars[i];
        if previous_cased { lower_at(&chars, i, &mut out) } else { title_of(c, &mut out) }
        previous_cased = is_cased(c);
    }
    out
}
