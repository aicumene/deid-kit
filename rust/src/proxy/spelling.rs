// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! File names, paths and copied lines come back spelled the way they were written (as `spelling.py`).
//! Per request, the text around each token is remembered as it was before tokenising — a segment between
//! separators, and a run of name characters — and a tool's arguments get the remembered text back; prose
//! gets file names and code spans back; a name that may still grow is held back while the answer
//! streams.

use std::sync::OnceLock;

use crate::gateway::Redaction;
use crate::odict::ODict;
use crate::pyre::{self, Regex, NONE, S};
use crate::py;

fn re(cell: &'static OnceLock<Regex>, p: &str) -> &'static Regex {
    pyre::lazy(cell, p, NONE)
}

/// A token as the vault mints it.
pub fn token() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?<![0-9A-Za-z])[A-Z]+_\d+(?![0-9A-Za-z])")
}

fn run_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"[\w.\-+~#@%=&]+")
}

fn run_char() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"[\w.\-+~#@%=&]")
}

fn sep() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"\\(?:u[0-9A-Fa-f]{4}|.)|[/\n\r\t→]", S)
}

fn quote() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r#"(["'`])"#)
}

fn last_run() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"[\w.\-+~#@%=&]+$")
}

fn code() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?<!`)`([^`\n]+)`(?!`)")
}

fn ticks() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"`+")
}

const WRAP: &str = " \"'`()[]<>{},;:";
const EDGE: &str = ".-_";
const LONGEST: usize = 2000;

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn has_token(t: &[char]) -> bool {
    token().search_at(t, 0).is_some()
}

fn token_full(t: &[char]) -> bool {
    token().fullmatch(t).is_some()
}

fn strip_set<'a>(t: &'a [char], set: &str) -> &'a [char] {
    let mut a = 0;
    let mut b = t.len();
    while a < b && set.contains(t[a]) {
        a += 1;
    }
    while b > a && set.contains(t[b - 1]) {
        b -= 1;
    }
    &t[a..b]
}

fn lstrip_len(t: &[char], set: &str) -> usize {
    t.iter().take_while(|c| set.contains(**c)).count()
}

fn rstrip_len(t: &[char], set: &str) -> usize {
    t.iter().rev().take_while(|c| set.contains(**c)).count()
}

fn starts_at(hay: &[char], needle: &[char], at: usize) -> bool {
    at + needle.len() <= hay.len() && hay[at..at + needle.len()] == *needle
}

/// For each token that replaced words: its span in `red.text` and the span of the words it replaced.
fn align(original: &[char], red: &Redaction) -> Option<Vec<(usize, usize, usize, usize)>> {
    let text = chars(&red.text);
    let mut queues: std::collections::HashMap<String, Vec<Vec<char>>> = red.surfaces.iter()
        .map(|(t, v)| (t.to_string(), v.iter().rev().map(|s| chars(s)).collect())).collect();
    let mut spans = Vec::new();
    let (mut i, mut j) = (0, 0);
    for m in token().finditer(&text) {
        let gap = &text[i..m.start];
        if !starts_at(original, gap, j) {
            return None;
        }
        j += gap.len();
        let tok: String = text[m.start..m.end].iter().collect();
        let queue = queues.get_mut(&tok);
        match queue {
            Some(q) if q.last().is_some_and(|s| starts_at(original, s, j)) => {
                let surface = q.pop().unwrap();
                spans.push((m.start, m.end, j, j + surface.len()));
                j += surface.len();
            }
            _ if starts_at(original, &text[m.start..m.end], j) => j += m.end - m.start,
            _ => return None,
        }
        i = m.end;
    }
    (original[j.min(original.len())..] == text[i..]).then_some(spans)
}

/// Positions in the tokenised text, outside any token, as positions in the original.
struct Positions {
    ends: Vec<usize>,
    shift: Vec<isize>,
}

impl Positions {
    fn new(spans: &[(usize, usize, usize, usize)]) -> Positions {
        let mut shift = vec![0isize];
        for &(ts, te, os, oe) in spans {
            let last = *shift.last().unwrap();
            shift.push(last + (oe as isize - os as isize) - (te as isize - ts as isize));
        }
        Positions { ends: spans.iter().map(|s| s.1).collect(), shift }
    }

    fn at(&self, pos: usize) -> usize {
        (pos as isize + self.shift[self.ends.partition_point(|e| *e <= pos)]) as usize
    }
}

/// One request's table: the segment and the run around each token, as tokenised, mapped to the same text
/// as it was written.
pub fn spellings(texts: &[String], reds: &[Redaction]) -> ODict<String> {
    let mut table: ODict<String> = ODict::new();
    for (original, red) in texts.iter().zip(reds) {
        if red.surfaces.is_empty() || red.text == *original {
            continue;
        }
        let orig = chars(original);
        let Some(spans) = align(&orig, red).filter(|s| !s.is_empty()) else { continue };
        let text = chars(&red.text);
        let where_ = Positions::new(&spans);
        let seps: Vec<(usize, usize)> = sep().finditer(&text).iter().map(|m| (m.start, m.end)).collect();
        let starts: Vec<usize> = seps.iter().map(|s| s.0).collect();
        let slice = |a: usize, b: usize| -> String { text[a..b].iter().collect() };
        let orig_slice = |a: usize, b: usize| -> String { orig[where_.at(a)..where_.at(b)].iter().collect() };
        let remember = |table: &mut ODict<String>, a: usize, b: usize| {
            let key = slice(a, b);
            if !key.is_empty() && !table.contains_key(&key) && !token_full(strip_set(&text[a..b], EDGE)) {
                table.insert(&key, orig_slice(a, b));
            }
        };
        let remember_folder = |table: &mut ODict<String>, a: usize, b: usize| {
            let a2 = a + lstrip_len(&text[a..b], EDGE);
            let b2 = b - rstrip_len(&text[a2..b], EDGE);
            table.setdefault(&format!("/{}", slice(a2, b2)), orig_slice(a2, b2));
        };
        for &(ts, te, _, _) in &spans {
            let (mut a, mut b) = (ts, te);
            while a > 0 && run_char().fullmatch(&text[a - 1..a]).is_some() {
                a -= 1;
            }
            while b < text.len() && run_char().fullmatch(&text[b..b + 1]).is_some() {
                b += 1;
            }
            remember(&mut table, a, b);
            let k = starts.partition_point(|s| *s <= ts) as isize - 1;
            let s0 = if k >= 0 { seps[k as usize].1 } else { 0 };
            let s1 = if ((k + 1) as usize) < starts.len() { starts[(k + 1) as usize] } else { text.len() };
            let seg = &text[s0..s1];
            let lo = s0 + lstrip_len(seg, WRAP);
            let hi = s0 + seg.len() - rstrip_len(seg, WRAP);
            if !(hi > lo && hi - lo <= LONGEST) {
                continue;
            }
            if !token_full(strip_set(&text[lo..hi], EDGE)) {
                remember(&mut table, lo, hi);
            } else if (s0 > 0 && text[s0 - 1] == '/') || text.get(s1) == Some(&'/') {
                remember_folder(&mut table, lo, hi);
            }
        }
    }
    table
}

/// `json.dumps(written, ensure_ascii=False)[1:-1]`.
fn literal_of(written: &str) -> String {
    let d = py::dumps(&serde_json::Value::String(written.to_string()));
    d[1..d.len() - 1].to_string()
}

/// `text` with each remembered segment, or else run, put back as it was written (escaped for a string
/// literal when `literal`); with `prose`, only names and inline code spans.
pub fn respell(text: &str, table: &ODict<String>, literal: bool, prose: bool) -> String {
    let t = chars(text);
    if table.is_empty() || text.is_empty() || !has_token(&t) {
        return text.to_string();
    }
    let put = |w: &str| if literal { literal_of(w) } else { w.to_string() };
    if !prose {
        return respell_inner(&t, table, &put, true);
    }
    let mut out = String::new();
    let mut cursor = 0;
    for m in code().finditer(&t) {
        out.push_str(&respell_inner(&t[cursor..m.start], table, &put, false));
        let (a, b) = m.group(1).unwrap();
        out.push('`');
        out.push_str(&respell_inner(&t[a..b], table, &put, true));
        out.push('`');
        cursor = m.end;
    }
    out.push_str(&respell_inner(&t[cursor..], table, &put, false));
    out
}

fn respell_inner(text: &[char], table: &ODict<String>, put: &dyn Fn(&str) -> String, whole: bool) -> String {
    let s = |c: &[char]| -> String { c.iter().collect() };
    let run = |m: &[char]| -> String {
        let mut core: Vec<char> = m.to_vec();
        let mut tail = String::new();
        if !has_token(&core) {
            return s(m);
        }
        while !table.contains_key(&s(&core)) && core.last().is_some_and(|c| ".-".contains(*c)) {
            let c = core.pop().unwrap();
            tail.insert(0, c);
        }
        match table.get(&s(&core)) {
            Some(v) => put(v) + &tail,
            None => s(m),
        }
    };
    let sub_runs = |seg: &[char]| -> String {
        let mut out = String::new();
        let mut last = 0;
        for m in run_re().finditer(seg) {
            out.push_str(&s(&seg[last..m.start]));
            out.push_str(&run(&seg[m.start..m.end]));
            last = m.end;
        }
        out.push_str(&s(&seg[last..]));
        out
    };
    let remembered = |piece: &[char], folder: bool| -> Option<String> {
        let lo = lstrip_len(piece, WRAP);
        let hi = piece.len() - rstrip_len(&piece[lo..], WRAP);
        let key = &piece[lo..hi];
        if let Some(v) = table.get(&s(key)) {
            return Some(s(&piece[..lo]) + &put(v) + &s(&piece[hi..]));
        }
        if key.first().is_some_and(|c| "+-".contains(*c)) {
            let inner = &key[1 + lstrip_len(&key[1..], WRAP)..];
            if let Some(v) = table.get(&s(inner)) {
                return Some(s(&piece[..lo]) + &s(&key[..key.len() - inner.len()]) + &put(v) + &s(&piece[hi..]));
            }
        }
        let core = strip_set(key, EDGE);
        if folder && table.contains_key(&format!("/{}", s(core))) && token_full(core) {
            let at = lstrip_len(key, EDGE);
            return Some(s(&piece[..lo + at]) + &put(table.get(&format!("/{}", s(core))).unwrap()) + &s(&piece[lo + at + core.len()..]));
        }
        None
    };
    let folder_name = |r: &[char]| -> Option<String> {
        let core = strip_set(r, EDGE);
        let key = format!("/{}", s(core));
        if !token_full(core) || !table.contains_key(&key) {
            return None;
        }
        let at = lstrip_len(r, EDGE);
        Some(s(&r[..at]) + &put(table.get(&key).unwrap()) + &s(&r[at + core.len()..]))
    };
    let segment = |seg: &[char], after_slash: bool, before_slash: bool| -> String {
        if !has_token(seg) {
            return s(seg);
        }
        if whole {
            if let Some(got) = remembered(seg, after_slash || before_slash) {
                return got;
            }
        }
        let (mut head, mut tail) = (String::new(), String::new());
        let mut seg: Vec<char> = seg.to_vec();
        if after_slash {
            if let Some(first) = run_re().match_at(&seg, 0) {
                if let Some(name) = folder_name(&seg[first.start..first.end]) {
                    head = name;
                    seg = seg[first.end..].to_vec();
                }
            }
        }
        if before_slash {
            if let Some(last) = last_run().search_at(&seg, 0) {
                if let Some(name) = folder_name(&seg[last.start..last.end]) {
                    tail = name;
                    seg.truncate(last.start);
                }
            }
        }
        if !whole {
            return head + &sub_runs(&seg) + &tail;
        }
        let parts = quote().split(&seg, 0);
        let mut out = head;
        for p in parts.into_iter().flatten() {
            let pc = chars(&p);
            if !has_token(&pc) {
                out.push_str(&p);
            } else {
                out.push_str(&remembered(&pc, false).unwrap_or_else(|| sub_runs(&pc)));
            }
        }
        out + &tail
    };
    let mut out = String::new();
    let mut cursor = 0;
    let seps = sep().finditer(text);
    for i in 0..=seps.len() {
        let end = seps.get(i).map_or(text.len(), |m| m.start);
        let after = cursor > 0 && text[cursor - 1] == '/';
        let before = text.get(end) == Some(&'/');
        out.push_str(&segment(&text[cursor..end], after, before));
        if let Some(m) = seps.get(i) {
            out.push_str(&s(&text[m.start..m.end]));
            cursor = m.end;
        }
    }
    out
}

/// Where an inline code span opens on the last line of `text` and has not closed yet.
fn open_code_span(text: &[char]) -> Option<usize> {
    let line = text.iter().rposition(|c| *c == '\n').map_or(0, |p| p + 1);
    let mut opened: Option<usize> = None;
    for m in ticks().finditer(&text[line..]) {
        if m.end - m.start == 1 {
            opened = if opened.is_some() { None } else { Some(line + m.start) };
        } else {
            opened = None;
        }
    }
    opened
}

/// How many characters at the end of streaming prose to hold back: a name that holds a token and may
/// still grow, or an open code span that holds one.
pub fn holdback(text: &str, table: &ODict<String>) -> usize {
    if table.is_empty() {
        return 0;
    }
    let t = chars(text);
    let mut hold = 0;
    if let Some(last) = last_run().search_at(&t, 0) {
        if has_token(&t[last.start..last.end]) {
            hold = t.len() - last.start;
        }
    }
    if let Some(tick) = open_code_span(&t) {
        if token().search_at(&t, tick).is_some() {
            hold = hold.max(t.len() - tick);
        }
    }
    hold
}
