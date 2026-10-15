// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A matter document under review, as the desktop shows it (`deidkit/agents/review.py`): its text, and
//! over the text what is de-identified before a model sees it, the norms the text refers to, and the
//! agents' notes — each on a verbatim quote, looked up in the text by code; a kept note is one W3C Web
//! Annotation a file, `.legix/markup/<path>/<id>.json`. Positions are characters (code points).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::{json, Map, Value};

use crate::agents::cite;
use crate::odict::ODict;
use crate::proxy::engine::ScopeEngine;
use crate::pyre::{self, Regex};
use crate::{py, pyjson};

pub const TEXT_SUFFIXES: [&str; 4] = [".md", ".markdown", ".txt", ".text"];
pub const NOTES: &str = ".legix";
pub const MAX_TEXT: usize = 4 * 1024 * 1024;

pub const KINDS: [&str; 4] = ["risk", "question", "point", "clause"];
pub const ASPECTS: [&str; 12] = ["parties", "subject", "term", "termination", "payment", "liability", "deadlines", "notices",
                                 "data protection", "dispute resolution", "compliance", "other"];
pub const CLAUSE_TYPES: [&str; 10] = ["definition", "obligation", "right", "condition", "deadline", "payment", "notice",
                                      "proviso", "warranty", "other"];
pub const MAX_QUOTE: usize = 600;
pub const MAX_NOTE: usize = 1200;
pub const MAX_NOTES: usize = 80;
pub const MAX_REPLACEMENT: usize = 4000;
pub const VERDICTS: [&str; 5] = ["accepted", "rejected", "useful", "not useful", "wrong"];

fn motivation(kind: &str) -> &'static str {
    match kind {
        "risk" => "assessing",
        "question" => "questioning",
        "clause" => "classifying",
        _ => "commenting",
    }
}

const PROMPT: &str = "Review the document \"{path}\" for the lawyer. Its text is in the file \"{source}\" in this folder: \
read that file, all of it. Do not change any file.

Write down what a lawyer reviewing this document must see. For each point, copy the document's own words \
that it is about — exactly, character for character, at most 300 characters, enough to be found in the \
text — and say:
- \"kind\": \"risk\" (could hurt the client), \"question\" (must be asked or checked before this goes further), \
\"point\" (important to know), or \"clause\" (what a clause does — the document's structure: give one for each \
clause that matters);
- \"aspect\": one of {aspects};
- \"importance\": 1 (minor) to 5 (critical) for the client;
- \"clause_type\", for a clause: one of {clause_types};
- \"who\", \"what\", \"to_whom\", \"when\": for a clause, who must or may do what, for whom, and when — from the \
text only, empty where it does not say;
- \"norm\": the statute or regulation the point turns on, as it is usually cited (for example \"§ 573c BGB\", \
\"Housing Act 1988 s. 21\", \"Art. 28 GDPR\"), or empty;
- \"suggestion\": where you would fix the point by changing the words, the words to put in place of the quote, \
exactly as they should read — the lawyer accepts or rejects it; otherwise empty;
- \"note\": one or two sentences for the lawyer, in the language they write to you in — under 300 characters.

At most 20 points, the most important first. Answer with one JSON object and nothing else:
{\"notes\": [{\"quote\": \"...\", \"kind\": \"risk\", \"aspect\": \"termination\", \"importance\": 4, \"clause_type\": \"\", \
\"who\": \"\", \"what\": \"\", \"to_whom\": \"\", \"when\": \"\", \"norm\": \"\", \"suggestion\": \"\", \"note\": \"...\"}]}";

pub fn prompt(path: &str, source: &str) -> String {
    // str.format substitutes each field once, in one pass: a path holding "{source}" stays as it is
    let mut out = String::new();
    let mut rest = PROMPT;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open..];
        let field = ["{path}", "{source}", "{aspects}", "{clause_types}"].iter().find(|f| after.starts_with(**f));
        match field {
            Some(f) => {
                out.push_str(&match *f {
                    "{path}" => path.to_string(),
                    "{source}" => source.to_string(),
                    "{aspects}" => ASPECTS.join(", "),
                    _ => CLAUSE_TYPES.join(", "),
                });
                rest = &after[f.len()..];
            }
            None => {
                out.push('{');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

// ── characters ────────────────────────────────────────────────────────────────────────────────────

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn st(t: &[char]) -> String {
    t.iter().collect()
}

/// `haystack.find(needle, start, end)` in characters.
fn find_in(h: &[char], n: &[char], start: usize, end: usize) -> Option<usize> {
    let end = end.min(h.len());
    if start > end {
        return None;
    }
    if n.is_empty() {
        return Some(start);
    }
    if n.len() > end - start {
        return None;
    }
    (start..=end - n.len()).find(|&i| h[i] == n[0] && h[i..i + n.len()] == *n)
}

fn starts_at(h: &[char], n: &[char], at: usize) -> bool {
    at <= h.len() && h.len() - at >= n.len() && h[at..at + n.len()] == *n
}

/// Every place `needle` starts in `haystack`, overlapping ones too (`_all`).
fn all(h: &[char], n: &[char]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = find_in(h, n, 0, h.len());
    while let Some(i) = at {
        out.push(i);
        at = find_in(h, n, i + 1, h.len());
    }
    out
}

fn suffix_of(path: &str) -> String {
    // PurePath.suffix of the last part, lower-cased
    let name = path.rsplit('/').next().unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => crate::unicode::lower(&name[i..]),
        _ => String::new(),
    }
}

// ── the document's text ───────────────────────────────────────────────────────────────────────────

/// `folder / rel` if it is a file in `folder` (no way out by `..` or a link).
pub fn inside(folder: &Path, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\0') {
        return None;
    }
    let target = py::realpath(&folder.join(rel));
    if !target.starts_with(py::realpath(folder)) {
        return None;
    }
    target.is_file().then_some(target)
}

/// Where the text of the document `path` is, relative to the matter folder.
pub fn text_source(path: &str) -> String {
    if TEXT_SUFFIXES.contains(&suffix_of(path).as_str()) { path.to_string() } else { format!("{NOTES}/text/{path}.md") }
}

/// `(source, text)` of the document `path`, or None.
pub fn read_text(folder: &Path, path: &str) -> Option<(String, String)> {
    if path.split('/').next() == Some(NOTES) || inside(folder, path).is_none() {
        return None;
    }
    let source = text_source(path);
    let target = inside(folder, &source)?;
    if std::fs::metadata(&target).ok()?.len() as usize > MAX_TEXT {
        return None;
    }
    Some((source, py::read_text(&target).ok()?))
}

// ── what is de-identified ─────────────────────────────────────────────────────────────────────────

/// A token placed by [`walk`]: in the tokenised text it is `start..end`, the literal piece before it
/// `after..start`; in the original the piece begins at `piece`, the token's words are `at..to`.
struct Placed {
    after: usize,
    start: usize,
    end: usize,
    piece: usize,
    at: usize,
    to: usize,
    token: String,
}

/// The tokenised text walked against the original (review.py `_walk`): the literal piece before a token
/// matched in the original (found again further on when the walk lost step), and the token takes the
/// recorded surface that stands there, else what lies up to the next literal piece.
fn walk(o: &[char], t: &[char], surfaces: &ODict<Vec<String>>, mapping: &ODict<String>) -> Vec<Placed> {
    let mut out = Vec::new();
    let (mut at, mut last) = (0usize, 0usize);
    let tokens = crate::proxy::spelling::token().finditer(t);
    for (i, m) in tokens.iter().enumerate() {
        let token = st(&t[m.start..m.end]);
        if mapping.get(&token).is_none() && surfaces.get(&token).is_none() {
            continue;
        }
        let piece = &t[last.min(m.start)..m.start];
        let begins = if starts_at(o, piece, at) {
            at
        } else {
            let found = if piece.is_empty() { Some(at) } else { find_in(o, piece, at, o.len()) };
            match found {
                Some(f) => f,
                None => {
                    last = m.end;
                    continue;
                }
            }
        };
        at = begins + piece.len();
        let mut candidates: Vec<String> = Vec::new();
        for s in surfaces.get(&token).cloned().unwrap_or_default().into_iter().chain(mapping.get(&token).cloned()) {
            if !s.is_empty() && !candidates.contains(&s) {
                candidates.push(s);
            }
        }
        candidates.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        let mut surface: Option<usize> = candidates.iter().map(|s| chars(s)).find(|s| starts_at(o, s, at)).map(|s| s.len());
        if surface.is_none() {
            let nxt = tokens.get(i + 1).map_or(t.len(), |n| n.start);
            let follow: Vec<char> = t[m.end..nxt.max(m.end)].iter().take(40).copied().collect();
            let end = if follow.is_empty() { Some(o.len()) } else { find_in(o, &follow, at, o.len()) };
            match end {
                Some(e) if e > at => surface = Some(e - at),
                _ => {
                    last = m.end;
                    continue;
                }
            }
        }
        let len = surface.unwrap();
        out.push(Placed { after: last.min(m.start), start: m.start, end: m.end, piece: begins, at, to: at + len, token });
        at += len;
        last = m.end;
    }
    out
}

/// Where each token of `tokenised` stands in `original`: `{start, end, type, token}`.
pub fn deid_spans(original: &str, tokenised: &str, surfaces: &ODict<Vec<String>>, mapping: &ODict<String>) -> Vec<Value> {
    walk(&chars(original), &chars(tokenised), surfaces, mapping).into_iter().map(|p| {
        let kind = p.token.rsplit_once('_').map_or(p.token.as_str(), |(k, _)| k).to_string();
        json!({"start": p.at, "end": p.to, "type": kind, "token": p.token})
    }).collect()
}

/// The tokenised text in pieces with their places in the original: `(t0, t1, o0, o1, token)` — a literal
/// piece (the same characters) or a token and the words it stands for (review.py `_alignment`).
fn alignment(o: &[char], t: &[char], surfaces: &ODict<Vec<String>>, mapping: &ODict<String>) -> Vec<(usize, usize, usize, usize, bool)> {
    let mut out = Vec::new();
    let (mut tail_t, mut tail_o) = (0usize, 0usize);
    for p in walk(o, t, surfaces, mapping) {
        out.push((p.after, p.start, p.piece, p.piece + p.start - p.after, false));
        out.push((p.start, p.end, p.at, p.to, true));
        (tail_t, tail_o) = (p.end, p.to);
    }
    let tail = &t[tail_t..];
    let found = if starts_at(o, tail, tail_o) { Some(tail_o) } else { find_in(o, tail, tail_o, o.len()) };
    if let Some(f) = found {
        out.push((tail_t, t.len(), f, f + tail.len(), false));
    }
    out.retain(|s| s.0 < s.1);
    out
}

/// `a..b` of the tokenised text as a span of the original, a token taken whole; None — an end falls where
/// the walk placed nothing.
fn to_original(pieces: &[(usize, usize, usize, usize, bool)], a: usize, b: usize) -> Option<(usize, usize)> {
    let (mut start, mut end) = (None, None);
    for &(t0, t1, o0, o1, token) in pieces {
        if start.is_none() && t0 <= a && a < t1 {
            start = Some(if token { o0 } else { o0 + (a - t0) });
        }
        if t0 < b && b <= t1 {
            end = Some(if token { o1 } else { o0 + (b - t0) });
        }
    }
    match (start, end) {
        (Some(s), Some(e)) if s < e => Some((s, e)),
        _ => None,
    }
}

/// Where each quote is in `text` ([`Folded::find`]), or None. A quote not found as written is looked up
/// again in tokens with the scope's `engine`: the quote and the text tokenised, every spelling of a party is
/// its one token — the model saw "ORG_…" where the document says "HFL", and the proxy gave the quote back
/// "Harrowgate Freight Ltd". Found there, it is the document's own words at that place (review.py `find_quotes`).
pub fn find_quotes(text: &str, quotes: &[String], engine: Option<&ScopeEngine>) -> Vec<Option<(usize, usize)>> {
    let folded = Folded::of(text);
    let mut found: Vec<Option<(usize, usize)>> = quotes.iter().map(|q| if q.is_empty() { None } else { folded.find(q, None) }).collect();
    let missing: Vec<usize> = (0..quotes.len()).filter(|&i| !quotes[i].is_empty() && found[i].is_none()).collect();
    let Some(engine) = engine.filter(|_| !missing.is_empty()) else { return found };
    let mut texts = vec![text.to_string()];
    texts.extend(missing.iter().map(|&i| quotes[i].clone()));
    let Ok(reds) = engine.tokenize_all(&texts) else { return found };
    let doc = &reds[0];
    let pieces = alignment(&folded.text, &chars(&doc.text), &doc.surfaces, &doc.mapping);
    let tokenised = Folded::of(&doc.text);
    for (&i, red) in missing.iter().zip(&reds[1..]) {
        if red.text == quotes[i] {
            continue;
        }
        if let Some((a, b)) = tokenised.find(&red.text, None) {
            found[i] = to_original(&pieces, a, b);
        }
    }
    found
}

// ── norms ─────────────────────────────────────────────────────────────────────────────────────────

/// The norms `text` refers to: `{start, end, citation, jurisdiction}`, in the order of the text.
pub fn norm_mentions(text: &str) -> Vec<Value> {
    let mut spans = cite::extract_citation_spans(text);
    spans.sort_by(|a, b| (a.start, &a.citation).cmp(&(b.start, &b.citation)));
    spans.iter().map(|s| json!({"start": s.start, "end": s.end, "citation": s.citation,
                                "jurisdiction": cite::jurisdiction_of(&s.citation)})).collect()
}

// ── finding a quote ───────────────────────────────────────────────────────────────────────────────

fn fold_char(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' | '\u{2032}' | '`' | '\u{b4}' => '\'',
        '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' | '\u{2033}' | '\u{ab}' | '\u{bb}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
        other => other,
    }
}

const DROP: [char; 6] = ['\u{ad}', '\u{200b}', '\u{200c}', '\u{200d}', '\u{2060}', '\u{feff}'];
const MAX_GAP: usize = 2000;

fn escaped(c: char) -> bool {
    // `\\([!-/:-@\[-`{-~])`: Markdown's backslash before punctuation
    matches!(c, '!'..='/' | ':'..='@' | '['..='`' | '{'..='~')
}

/// `text` folded for comparison, and for each character of the result where it came from.
fn fold(text: &[char], unescape: bool) -> (Vec<char>, Vec<usize>) {
    let (mut out, mut back) = (Vec::new(), Vec::new());
    let mut space = false;
    let mut i = 0;
    while i < text.len() {
        let ch = text[i];
        if unescape && ch == '\\' && i + 1 < text.len() && escaped(text[i + 1]) {
            i += 1;
            continue;
        }
        if DROP.contains(&ch) {
            i += 1;
            continue;
        }
        if py::is_space(ch) {
            if !space && !out.is_empty() {
                out.push(' ');
                back.push(i);
            }
            space = true;
            i += 1;
            continue;
        }
        space = false;
        out.push(fold_char(ch));
        back.push(i);
        i += 1;
    }
    if out.last() == Some(&' ') {
        out.pop();
        back.pop();
    }
    (out, back)
}

fn ellipsis() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"\s*(?:\.\.\.|…|\[\.\.\.\]|\[…\])\s*", pyre::NONE)
}

/// A text folded once, for looking many quotes up in it.
pub struct Folded {
    pub text: Vec<char>,
    folded: Vec<char>,
    back: Vec<usize>,
}

fn nearest(hits: &[(usize, usize)], near: Option<i64>) -> (usize, usize) {
    match near {
        Some(n) => *hits.iter().min_by_key(|h| (h.0 as i64 - n).abs()).unwrap(),
        None => hits[0],
    }
}

impl Folded {
    pub fn of(text: &str) -> Folded {
        let t = chars(text);
        let (folded, back) = fold(&t, true);
        Folded { text: t, folded, back }
    }

    /// Where `quote` is in the text: exact first; then folded, an ellipsis splitting it into parts in
    /// order, each within a page of the one before. Of several places, the one nearest `near`.
    pub fn find(&self, quote: &str, near: Option<i64>) -> Option<(usize, usize)> {
        let q = chars(py::strip(quote));
        if q.is_empty() {
            return None;
        }
        // re.finditer(re.escape(quote), text): the places that do not overlap
        let mut exact = Vec::new();
        let mut at = 0;
        while let Some(i) = find_in(&self.text, &q, at, self.text.len()) {
            exact.push((i, i + q.len()));
            at = i + q.len();
        }
        if !exact.is_empty() {
            return Some(nearest(&exact, near));
        }
        let parts: Vec<Vec<char>> = ellipsis().split(&q, 0).into_iter().flatten()
            .map(|p| fold(&chars(&p), true).0).filter(|p| !p.is_empty()).collect();
        if parts.is_empty() || parts.iter().map(Vec::len).sum::<usize>() < 8 {
            return None;
        }
        let mut hits = Vec::new();
        'first: for first in all(&self.folded, &parts[0]) {
            let mut end = first + parts[0].len();
            for part in &parts[1..] {
                match find_in(&self.folded, part, end, end + MAX_GAP + part.len()) {
                    Some(nxt) => end = nxt + part.len(),
                    None => continue 'first,
                }
            }
            hits.push((self.back[first], self.back[end - 1] + 1));
        }
        (!hits.is_empty()).then(|| nearest(&hits, near))
    }
}

// ── clause addresses ──────────────────────────────────────────────────────────────────────────────

/// The address of the clause the place `at` is in: of its own line if it has one, else the nearest before.
pub fn clause_at(text: &[char], at: usize) -> Option<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = pyre::lazy(&R, r"(?<!\\)\{#([a-z0-9_-]+)\}[ \t]*$", pyre::M);
    let line_end = find_in(text, &['\n'], at.min(text.len()), text.len()).unwrap_or(text.len());
    re.finditer(&text[..line_end]).last().and_then(|m| m.group(1)).map(|g| st(&text[g.0..g.1]))
}

// ── the agent's answer ────────────────────────────────────────────────────────────────────────────

fn notes_object() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r#"\{\s*"notes"\s*:"#, pyre::NONE)
}

/// The whole notes of an answer cut short in the middle of its list; None — no list of notes begins in it.
fn cut_notes(answer: &[char]) -> Option<Vec<Value>> {
    let starts = notes_object().finditer(answer);
    let last = starts.last()?;
    let at = find_in(answer, &['['], last.end, answer.len())?;
    if !py::strip(&st(&answer[last.end..at])).is_empty() {
        return None;
    }
    let mut notes = Vec::new();
    let mut at = at + 1;
    loop {
        while at < answer.len() && " \t\r\n,".contains(answer[at]) {
            at += 1;
        }
        match pyjson::raw_decode(answer, at) {
            Ok((value, end)) => {
                notes.push(value);
                at = end;
            }
            Err(()) => return Some(notes),
        }
    }
}

/// The notes object in the agent's answer: the whole answer if it is one, else the last whole one that
/// starts with `{"notes":`.
fn json_object(answer: &str) -> Option<Map<String, Value>> {
    if let Some(Value::Object(m)) = pyjson::loads(answer) {
        return Some(m);
    }
    let t = chars(answer);
    let mut found = None;
    for m in notes_object().finditer(&t) {
        if let Ok((Value::Object(o), _)) = pyjson::raw_decode(&t, m.start) {
            found = Some(o);
        }
    }
    found
}

/// `" ".join(str(value).split())[:limit]` of a str, int or float (a bool is an int); "" for anything else.
pub fn text_of(value: Option<&Value>, limit: usize) -> String {
    match value {
        Some(v @ (Value::String(_) | Value::Number(_) | Value::Bool(_))) => {
            py::take(&py::split_ws(&py::py_str(v)).join(" "), limit).to_string()
        }
        _ => String::new(),
    }
}

/// `(notes, cut)`: the notes in an agent's answer, in our vocabulary, and whether it was cut short.
pub fn notes_of(answer: &str) -> (Option<Vec<Value>>, bool) {
    let data = json_object(answer);
    let notes = data.as_ref().and_then(|d| d.get("notes")).and_then(Value::as_array).cloned();
    match notes {
        Some(n) => (Some(normal_notes(&n)), false),
        None => {
            if data.is_some() {
                return (None, false);
            }
            match cut_notes(&chars(answer)) {
                Some(whole) if !whole.is_empty() => (Some(normal_notes(&whole)), true),
                _ => (None, false),
            }
        }
    }
}

/// `int(value)` clamped to 1–5; 3 where Python's int() would raise.
fn importance(value: Option<&Value>) -> i64 {
    value.and_then(|v| if v.is_null() { None } else { py::int(v) }).map_or(3, |i| i.clamp(1, 5))
}

fn normal_notes(raw: &[Value]) -> Vec<Value> {
    let mut out = Vec::new();
    for r in raw.iter().take(MAX_NOTES) {
        let Some(o) = r.as_object() else { continue };
        let quote = o.get("quote").and_then(Value::as_str).unwrap_or("");
        let kind = crate::unicode::lower(&text_of(o.get("kind"), 20));
        let aspect = crate::unicode::lower(&text_of(o.get("aspect"), 40));
        let clause_type = crate::unicode::lower(&text_of(o.get("clause_type"), 40));
        let mut note = Map::new();
        note.insert("quote".into(), json!(py::take(py::strip(quote), MAX_QUOTE)));
        note.insert("kind".into(), json!(if KINDS.contains(&kind.as_str()) { kind.as_str() } else { "point" }));
        note.insert("aspect".into(), json!(if ASPECTS.contains(&aspect.as_str()) { aspect.as_str() } else { "other" }));
        note.insert("importance".into(), json!(importance(o.get("importance"))));
        note.insert("clause_type".into(), json!(if CLAUSE_TYPES.contains(&clause_type.as_str()) { clause_type.as_str() }
                                                else if kind == "clause" { "other" } else { "" }));
        for k in ["who", "what", "to_whom", "when"] {
            note.insert(k.into(), json!(text_of(o.get(k), 200)));
        }
        note.insert("norm".into(), json!(text_of(o.get("norm"), 120)));
        note.insert("note".into(), json!(text_of(o.get("note"), MAX_NOTE)));
        note.insert("suggestion".into(), json!(o.get("suggestion").and_then(Value::as_str)
            .map_or(String::new(), |s| py::take(py::strip(s), MAX_REPLACEMENT).to_string())));
        out.push(Value::Object(note));
    }
    out
}

// ── the markup ────────────────────────────────────────────────────────────────────────────────────

fn quote_selectors(text: &[char], start: usize, end: usize) -> Vec<Value> {
    let mut selectors = vec![
        json!({"type": "TextQuoteSelector", "exact": st(&text[start..end]), "prefix": st(&text[start.saturating_sub(32)..start]),
               "suffix": st(&text[end..(end + 32).min(text.len())])}),
        json!({"type": "TextPositionSelector", "start": start, "end": end}),
    ];
    if let Some(clause) = clause_at(text, start) {
        selectors.push(json!({"type": "FragmentSelector", "value": clause}));
    }
    selectors
}

/// A kept note as a W3C Web Annotation.
pub fn annotation(note: &Value, path: &str, text: &[char], start: usize, end: usize, creator: &Value, run: &str) -> Value {
    let g = |k: &str| note.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut body = Vec::new();
    for (k, purpose) in [("note", "commenting"), ("clause_type", "classifying"), ("norm", "linking"), ("suggestion", "replacing")] {
        if !g(k).is_empty() {
            body.push(json!({"type": "TextualBody", "value": g(k), "purpose": purpose}));
        }
    }
    let mut bernio = Map::new();
    for k in ["kind", "aspect", "importance", "clause_type", "who", "what", "to_whom", "when", "norm"] {
        match note.get(k) {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s.is_empty() => {}
            Some(v) => {
                bernio.insert(k.into(), v.clone());
            }
        }
    }
    json!({
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": py::uuid4_hex(),
        "type": "Annotation",
        "motivation": motivation(&g("kind")),
        "creator": creator,
        "created": py::utc_seconds(),
        "generator": {"type": "Software", "name": "deid-agent review", "run": run},
        "target": {"source": path, "selector": quote_selectors(text, start, end)},
        "body": body,
        "bernio": bernio,
    })
}

pub fn markup_dir(folder: &Path, path: &str) -> PathBuf {
    folder.join(NOTES).join("markup").join(path)
}

fn write_json(target: &Path, value: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(target.parent().unwrap())?;
    let tmp = target.with_extension("json.tmp");
    std::fs::write(&tmp, py::dumps_indent(value, 2) + "\n")?;
    std::fs::rename(&tmp, target)
}

/// One annotation a file.
pub fn store(folder: &Path, path: &str, ann: &Value) -> std::io::Result<PathBuf> {
    let target = markup_dir(folder, path).join(format!("{}.json", py::py_str(&ann["id"])));
    write_json(&target, ann)?;
    Ok(target)
}

fn selector<'a>(ann: &'a Value, kind: &str) -> Option<&'a Map<String, Value>> {
    ann.get("target").and_then(Value::as_object)?.get("selector")?.as_array()?.iter()
        .filter_map(Value::as_object).find(|s| s.get("type").and_then(Value::as_str) == Some(kind))
}

fn sel_str(ann: &Value, kind: &str, key: &str) -> String {
    selector(ann, kind).and_then(|s| s.get(key)).and_then(Value::as_str).unwrap_or("").to_string()
}

/// A position as Python's `isinstance(x, int)` takes it (a bool is one).
fn int_of(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// `text[a:b]` with Python's slice rules (negative counts from the end, out of range clipped).
fn slice(text: &[char], a: i64, b: i64) -> &[char] {
    let n = text.len() as i64;
    let fix = |x: i64| if x < 0 { (x + n).max(0) } else { x.min(n) } as usize;
    let (a, b) = (fix(a), fix(b));
    if a >= b { &[] } else { &text[a..b] }
}

/// Where an annotation's words are: `status` "here" or "moved", and its start and end as Python has
/// them — a position read from a file is given back as it was written there (an int, or a bool).
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub status: &'static str,
    pub start: Value,
    pub end: Value,
}

impl Place {
    fn at(status: &'static str, start: usize, end: usize) -> Place {
        Place { status, start: json!(start), end: json!(end) }
    }

    /// The start as a number, for ordering (`True` is 1).
    pub fn start_key(&self) -> i64 {
        int_of(Some(&self.start)).unwrap_or(0)
    }
}

/// `text[:i]` / `text[i:]`: where a Python index cuts a text of `n` characters.
fn cut_at(i: i64, n: usize) -> usize {
    let n = n as i64;
    (if i < 0 { (i + n).max(0) } else { i.min(n) }) as usize
}

/// Where the annotation's words are in the text now, or None — they are no longer in it.
pub fn place(ann: &Value, folded: &Folded) -> Option<Place> {
    let exact = chars(&sel_str(ann, "TextQuoteSelector", "exact"));
    let pos = selector(ann, "TextPositionSelector");
    let start_v = pos.and_then(|p| p.get("start")).cloned();
    let end_v = pos.and_then(|p| p.get("end")).cloned();
    let start = int_of(start_v.as_ref());
    let end = int_of(end_v.as_ref());
    let text = &folded.text;
    if exact.is_empty() {
        let prefix = chars(&sel_str(ann, "TextQuoteSelector", "prefix"));
        let suffix = chars(&sel_str(ann, "TextQuoteSelector", "suffix"));
        if prefix.is_empty() && suffix.is_empty() {
            return None;
        }
        if let Some(s) = start {
            if slice(text, (s - prefix.len() as i64).max(0), s) == prefix.as_slice()
                && slice(text, s, s + suffix.len() as i64) == suffix.as_slice() {
                let v = start_v.unwrap();
                return Some(Place { status: "here", start: v.clone(), end: v });
            }
        }
        for (p, q) in [(prefix.len(), suffix.len()), (16, 16), (16, 0), (0, 16)] {
            let before: &[char] = if p > 0 { &prefix[prefix.len().saturating_sub(p)..] } else { &[] };
            let after: &[char] = &suffix[..q.min(suffix.len())];
            if before.is_empty() && after.is_empty() {
                continue;
            }
            let both: Vec<char> = before.iter().chain(after).copied().collect();
            let hits: Vec<(usize, usize)> = all(text, &both).into_iter().map(|m| (m + before.len(), 0)).collect();
            if !hits.is_empty() {
                let at = nearest(&hits, start).0;
                return Some(Place::at("moved", at, at));
            }
        }
        return None;
    }
    if let (Some(s), Some(e)) = (start, end) {
        if slice(text, s, e) == exact.as_slice() {
            return Some(Place { status: "here", start: start_v.unwrap(), end: end_v.unwrap() });
        }
    }
    folded.find(&st(&exact), start).map(|(a, b)| Place::at("moved", a, b))
}

fn base(motivation: &str, creator: &Value, target: Value, body: Vec<Value>, bernio: Value) -> Value {
    json!({"@context": "http://www.w3.org/ns/anno.jsonld", "id": py::uuid4_hex(), "type": "Annotation",
           "motivation": motivation, "creator": creator, "created": py::utc_micros(), "target": target, "body": body,
           "bernio": bernio})
}

pub fn person(name: &str) -> Value {
    json!({"type": "Person", "name": if name.is_empty() { "person" } else { name }})
}

/// A change a person proposes, as Word's tracked change.
pub fn suggestion(path: &str, text: &[char], start: usize, end: usize, replacement: &str, creator: &Value) -> Value {
    base("editing", creator, json!({"source": path, "selector": quote_selectors(text, start, end)}),
         vec![json!({"type": "TextualBody", "value": replacement, "purpose": "replacing"})], json!({"kind": "edit"}))
}

fn verdict_body(verdict: &str, comment: &str, correction: &str) -> Vec<Value> {
    let mut body = vec![json!({"type": "TextualBody", "value": verdict, "purpose": "assessing"})];
    if !comment.is_empty() {
        body.push(json!({"type": "TextualBody", "value": comment, "purpose": "commenting"}));
    }
    if !correction.is_empty() {
        body.push(json!({"type": "TextualBody", "value": correction, "purpose": "replacing"}));
    }
    body
}

/// What a person decided of a note or a proposed change.
pub fn decision(path: &str, target: &str, verdict: &str, creator: &Value, comment: &str, correction: &str, applied: bool) -> Value {
    let mut bernio = json!({"kind": "decision", "verdict": verdict});
    if applied {
        bernio["applied"] = json!(true);
    }
    base("assessing", creator, json!({"source": path, "annotation": target}), verdict_body(verdict, comment, correction), bernio)
}

fn words() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"\s+|\w+|[^\w\s]", pyre::NONE)
}

/// How `after` differs from `before`, word by word: `{at, old, new}`.
pub fn hunks(before: &str, after: &str, limit: usize) -> Vec<Value> {
    let split = |s: &str| -> Vec<String> {
        let t = chars(s);
        words().finditer(&t).iter().map(|m| st(&t[m.start..m.end])).collect()
    };
    let (a, b) = (split(before), split(after));
    let mut starts = Vec::with_capacity(a.len() + 1);
    let mut at = 0;
    for w in &a {
        starts.push(at);
        at += w.chars().count();
    }
    starts.push(at);
    let mut out = Vec::new();
    for (tag, i1, i2, j1, j2) in crate::difflib::opcodes(&a, &b) {
        if tag != "equal" {
            out.push(json!({"at": starts[i1], "old": a[i1..i2].concat(), "new": b[j1..j2].concat()}));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

/// The person's own writing, saved from the editor.
pub fn revision(path: &str, before: &str, after: &str, creator: &Value) -> Value {
    base("editing", creator, json!({"source": path}), vec![],
         json!({"kind": "revision", "before": sha(before), "after": sha(after), "hunks": hunks(before, after, 200)}))
}

/// Who wrote the document: an agent in a task, or a person.
pub fn written(path: &str, creator: &Value, how: &str, run: &str) -> Value {
    let mut bernio = json!({"kind": "written", "how": how});
    if !run.is_empty() {
        bernio["run"] = json!(run);
    }
    base("describing", creator, json!({"source": path}), vec![], bernio)
}

pub fn sha(text: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// A document whose text is the document itself (Markdown, plain text).
pub fn writable(path: &str) -> bool {
    TEXT_SUFFIXES.contains(&suffix_of(path).as_str()) && path.split('/').next() != Some(NOTES)
}

/// The `*.json` files of a folder, sorted as Python sorts paths.
fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(Result::ok).map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".json")))
        .collect()).unwrap_or_default();
    out.sort();
    out
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The document's annotations as they are in its files.
pub fn annotations(folder: &Path, path: &str) -> Vec<Value> {
    let root = markup_dir(folder, path);
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    for f in json_files(&root) {
        let Ok(Some(text)) = py::read_text_strict(&f) else { continue };
        let Some(Value::Object(mut ann)) = pyjson::loads(&text) else { continue };
        let ok = ann.get("target").and_then(Value::as_object)
            .is_some_and(|t| t.get("source").and_then(Value::as_str) == Some(path));
        if ok {
            if !ann.contains_key("id") {
                ann.insert("id".into(), json!(stem(&f)));
            }
            out.push(Value::Object(ann));
        }
    }
    out
}

fn who(creator: Option<&Value>) -> (String, String, bool) {
    match creator {
        Some(Value::Object(c)) => (text_of(c.get("name"), 80), text_of(c.get("model"), 80),
                                   c.get("type").and_then(Value::as_str) == Some("Person")),
        other => (text_of(other, 80), String::new(), false),
    }
}

/// The body of `purpose`: a replacement as written (cut to `limit`), anything else squashed.
pub fn body_of(ann: &Value, purpose: &str, limit: usize) -> String {
    for b in ann.get("body").and_then(Value::as_array).into_iter().flatten() {
        if b.get("purpose").and_then(Value::as_str) == Some(purpose) {
            if let Some(v) = b.get("value").and_then(Value::as_str) {
                return if purpose == "replacing" { py::take(v, limit).to_string() } else { text_of(Some(&json!(v)), limit) };
            }
        }
    }
    String::new()
}

pub fn kind_of(ann: &Value) -> String {
    ann.get("bernio").and_then(Value::as_object).and_then(|b| b.get("kind")).and_then(Value::as_str).unwrap_or("").to_string()
}

fn bernio(ann: &Value) -> Map<String, Value> {
    ann.get("bernio").and_then(Value::as_object).cloned().unwrap_or_default()
}

/// `(bernio.get("importance") if in (1, 2, 3, 4, 5) else 3)`, and its sort key: 1.0 and True are 1.
fn note_importance(b: &Map<String, Value>) -> (Value, f64) {
    match b.get("importance") {
        Some(Value::Bool(true)) => (json!(true), 1.0),
        Some(v @ Value::Number(n)) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && (1.0..=5.0).contains(&f) => (v.clone(), f),
            _ => (json!(3), 3.0),
        },
        _ => (json!(3), 3.0),
    }
}

/// `notes`, `revisions` and `written` of the document, each note placed in the text as it is now.
pub fn view(folder: &Path, path: &str, text: &str) -> Value {
    let folded = Folded::of(text);
    let anns = annotations(folder, path);
    let mut decided: HashMap<String, Vec<Value>> = HashMap::new();
    let (mut revisions, mut authors) = (Vec::new(), Vec::new());
    for ann in &anns {
        match kind_of(ann).as_str() {
            "decision" => {
                let target = ann["target"].get("annotation");
                let b = bernio(ann);
                let verdict = b.get("verdict").and_then(Value::as_str).unwrap_or("");
                if let (Some(Value::String(t)), true) = (target, VERDICTS.contains(&verdict)) {
                    decided.entry(t.clone()).or_default().push(json!({
                        "verdict": verdict, "by": who(ann.get("creator")).0, "at": text_of(ann.get("created"), 40),
                        "comment": body_of(ann, "commenting", MAX_NOTE), "correction": body_of(ann, "replacing", MAX_REPLACEMENT),
                        "applied": b.get("applied") == Some(&json!(true))}));
                }
            }
            "revision" => {
                let hs = bernio(ann).get("hunks").and_then(Value::as_array).map_or(0, Vec::len);
                let (n, m, p) = who(ann.get("creator"));
                revisions.push(json!({"id": text_of(ann.get("id"), 64), "by": {"name": n, "model": m, "person": p},
                                      "at": text_of(ann.get("created"), 40), "changes": hs}));
            }
            "written" => {
                let how = bernio(ann).get("how").and_then(Value::as_str).unwrap_or("").to_string();
                let how = if ["added", "changed", "wrote", "saved"].contains(&how.as_str()) { how } else { String::new() };
                let (n, m, p) = who(ann.get("creator"));
                authors.push(json!({"by": {"name": n, "model": m, "person": p}, "at": text_of(ann.get("created"), 40), "how": how}));
            }
            _ => {}
        }
    }
    let mut notes: Vec<(Value, (bool, i64, f64))> = Vec::new();
    for ann in &anns {
        let kind = kind_of(ann);
        if !(KINDS.contains(&kind.as_str()) || kind == "edit") {
            continue;
        }
        let where_ = place(ann, &folded);
        let b = bernio(ann);
        let key = ann.get("id").map(py::py_str).unwrap_or_else(|| "None".into());
        let mut decisions = decided.get(&key).cloned().unwrap_or_default();
        decisions.sort_by(|x, y| x["at"].as_str().cmp(&y["at"].as_str()));
        let state = decisions.iter().rev().filter_map(|d| d["verdict"].as_str())
            .find(|v| *v == "accepted" || *v == "rejected").unwrap_or("").to_string();
        let (n, m, p) = who(ann.get("creator"));
        let (imp, imp_key) = note_importance(&b);
        let aspect = match b.get("aspect").and_then(Value::as_str) {
            Some(a) if ASPECTS.contains(&a) => a.to_string(),
            _ => if kind == "edit" { String::new() } else { "other".into() },
        };
        let run = match ann.get("generator") {
            Some(Value::Object(g)) => text_of(g.get("run"), 64),
            _ => String::new(),
        };
        let mut note = Map::new();
        note.insert("id".into(), json!(text_of(ann.get("id"), 64)));
        note.insert("created".into(), json!(text_of(ann.get("created"), 40)));
        note.insert("creator".into(), json!({"name": n, "model": m}));
        note.insert("by_person".into(), json!(p));
        note.insert("run".into(), json!(run));
        note.insert("status".into(), json!(where_.as_ref().map_or("gone", |w| w.status)));
        note.insert("start".into(), where_.as_ref().map_or(Value::Null, |w| w.start.clone()));
        note.insert("end".into(), where_.as_ref().map_or(Value::Null, |w| w.end.clone()));
        note.insert("quote".into(), json!(text_of(selector(ann, "TextQuoteSelector").and_then(|s| s.get("exact")), MAX_QUOTE)));
        note.insert("clause".into(), json!(text_of(selector(ann, "FragmentSelector").and_then(|s| s.get("value")), 80)));
        note.insert("note".into(), json!(body_of(ann, "commenting", MAX_NOTE)));
        note.insert("kind".into(), json!(kind));
        note.insert("aspect".into(), json!(aspect));
        note.insert("importance".into(), imp);
        for k in ["clause_type", "who", "what", "to_whom", "when", "norm"] {
            note.insert(k.into(), json!(text_of(b.get(k), 200)));
        }
        note.insert("suggestion".into(), json!(body_of(ann, "replacing", MAX_REPLACEMENT)));
        note.insert("decisions".into(), json!(decisions));
        note.insert("state".into(), json!(state));
        let sort = (where_.is_none(), where_.as_ref().map_or(0, Place::start_key), -imp_key);
        notes.push((Value::Object(note), sort));
    }
    notes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    revisions.sort_by(|x, y| x["at"].as_str().cmp(&y["at"].as_str()));
    authors.sort_by(|x, y| x["at"].as_str().cmp(&y["at"].as_str()));
    json!({"notes": notes.into_iter().map(|n| n.0).collect::<Vec<_>>(), "revisions": revisions, "written": authors})
}

/// Keep the notes whose quotes are in the text; say of the others why not.
/// `found`: where each note's quote is ([`find_quotes`], which can look in tokens); else looked up as written.
pub fn keep(folder: &Path, path: &str, text: &str, notes: &[Value], creator: &Value, run: &str,
            found: Option<&[Option<(usize, usize)>]>) -> std::io::Result<Value> {
    let folded = Folded::of(text);
    let (mut kept, mut refused) = (Vec::new(), Vec::new());
    for (n, note) in notes.iter().enumerate() {
        let quote = note["quote"].as_str().unwrap_or("");
        let found = match found {
            Some(f) => f.get(n).copied().flatten(),
            None => if quote.is_empty() { None } else { folded.find(quote, None) },
        };
        let Some((a, b)) = found else {
            refused.push(json!({"quote": py::take(quote, 120), "why": if quote.is_empty() { "no quote" } else { "not in the document" }}));
            continue;
        };
        let ann = annotation(note, path, &folded.text, a, b, creator, run);
        store(folder, path, &ann)?;
        kept.push(ann["id"].clone());
    }
    Ok(json!({"kept": kept, "refused": refused}))
}

/// The document's annotation by its id.
pub fn find(folder: &Path, path: &str, ident: &str) -> Option<Value> {
    let ok = (8..=64).contains(&ident.len()) && ident.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !ok {
        return None;
    }
    annotations(folder, path).into_iter().find(|a| a.get("id").and_then(Value::as_str) == Some(ident))
}

/// The text with the annotation's words read as `replacement`; None — they are no longer in it.
pub fn apply(text: &str, ann: &Value, replacement: &str) -> Option<String> {
    let folded = Folded::of(text);
    let p = place(ann, &folded)?;
    let n = folded.text.len();
    let (a, b) = (cut_at(int_of(Some(&p.start)).unwrap_or(0), n), cut_at(int_of(Some(&p.end)).unwrap_or(0), n));
    Some(st(&folded.text[..a]) + replacement + &st(&folded.text[b..]))
}

/// The document's new text, whole or not at all.
pub fn write_text(folder: &Path, path: &str, text: &str) -> std::io::Result<()> {
    let target = folder.join(path);
    std::fs::create_dir_all(target.parent().unwrap())?;
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = target.with_file_name(format!(".{name}.{}.tmp", &py::uuid4_hex()[..8]));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &target)
}

/// A new document may be written here: Markdown or plain text, inside the folder, not in its notes,
/// not over a document that is there.
pub fn new_path_ok(folder: &Path, path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\0') || !writable(path) {
        return false;
    }
    // PurePath.parts: "." parts are dropped, so only ".." and hidden names can be found here
    if path.split('/').filter(|p| !p.is_empty() && *p != ".").any(|p| p == ".." || p.starts_with('.')) {
        return false;
    }
    if !py::realpath(&folder.join(path)).starts_with(py::realpath(folder)) {
        return false;
    }
    !folder.join(path).exists()
}

// ── votes on the chat's answers ───────────────────────────────────────────────────────────────────

pub const CHAT_FEEDBACK: &str = "chat-feedback";
pub const MAX_ANSWER: usize = 20000;
pub const MAX_QUESTION: usize = 4000;

/// A person's 👍 or 👎 on an answer of the agent in the chat; `key` names it, and a later vote replaces it.
#[allow(clippy::too_many_arguments)]
pub fn vote(key: &str, question: &str, answer: &str, agent: &str, model: &str, verdict: &str, creator: &Value,
            comment: &str, correction: &str) -> Value {
    let mut ann = base("assessing", creator, json!({"source": "chat"}), verdict_body(verdict, comment, correction), json!({
        "kind": "vote", "verdict": verdict, "question": py::take(question, MAX_QUESTION), "answer": py::take(answer, MAX_ANSWER),
        "agent": text_of(Some(&json!(agent)), 80), "model": text_of(Some(&json!(model)), 80)}));
    use sha2::{Digest, Sha256};
    ann["id"] = json!(hex::encode(Sha256::digest(key.as_bytes()))[..32]);
    ann
}

pub fn store_vote(folder: &Path, ann: &Value) -> std::io::Result<PathBuf> {
    let target = folder.join(NOTES).join(CHAT_FEEDBACK).join(format!("{}.json", py::py_str(&ann["id"])));
    write_json(&target, ann)?;
    Ok(target)
}

/// The votes on the chat's answers kept in this folder, as records.
pub fn votes(folder: &Path) -> Vec<Value> {
    let root = folder.join(NOTES).join(CHAT_FEEDBACK);
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    for f in json_files(&root) {
        let Ok(Some(text)) = py::read_text_strict(&f) else { continue };
        let Some(ann) = pyjson::loads(&text) else { continue };
        let b = if ann.is_object() { bernio(&ann) } else { Map::new() };
        let verdict = b.get("verdict").and_then(Value::as_str).unwrap_or("");
        if b.get("kind").and_then(Value::as_str) != Some("vote") || !(verdict == "up" || verdict == "down") {
            continue;
        }
        let s = |k: &str, n: usize| -> String {
            match b.get(k) {
                None | Some(Value::Null) => String::new(),
                Some(v) if !py::truthy(Some(v)) => String::new(),
                Some(v) => py::take(&py::py_str(v), n).to_string(),
            }
        };
        out.push(json!({"type": "answer_vote", "at": text_of(ann.get("created"), 40),
                        "agent": text_of(b.get("agent"), 80), "model": text_of(b.get("model"), 80),
                        "question": s("question", MAX_QUESTION), "answer": s("answer", MAX_ANSWER),
                        "verdict": verdict, "comment": body_of(&ann, "commenting", MAX_NOTE),
                        "correction": body_of(&ann, "replacing", MAX_REPLACEMENT)}));
    }
    out
}

// ── what people decided, for the harness and for learning ─────────────────────────────────────────

/// Every `*.json` under `root`, as Path.rglob finds them (links to folders not followed).
fn rglob_json(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    for e in rd.filter_map(Result::ok) {
        let p = e.path();
        let link_dir = e.file_type().is_ok_and(|t| t.is_symlink());
        if p.is_dir() && !link_dir {
            rglob_json(&p, out);
        }
        if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".json")) {
            out.push(p);
        }
    }
}

/// What people decided of the agents' work in this folder, one record each.
pub fn records(folder: &Path) -> Vec<Value> {
    let root = folder.join(NOTES).join("markup");
    let mut out = votes(folder);
    if !root.is_dir() {
        return out;
    }
    let mut files = Vec::new();
    rglob_json(&root, &mut files);
    let mut docs: Vec<String> = files.iter().filter_map(|f| f.parent()?.strip_prefix(&root).ok())
        .map(|p| { let s = p.to_string_lossy().into_owned(); if s.is_empty() { ".".into() } else { s } })
        .collect::<HashSet<_>>().into_iter().collect();
    docs.sort();
    for path in docs {
        let anns = annotations(folder, &path);
        let mut by_id: HashMap<String, Value> = HashMap::new();
        for a in &anns {
            by_id.insert(a.get("id").map(py::py_repr).unwrap_or_default(), a.clone());
        }
        let agent_wrote: Vec<&Value> = anns.iter().filter(|a| kind_of(a) == "written" && !who(a.get("creator")).2).collect();
        for ann in &anns {
            match kind_of(ann).as_str() {
                "decision" => {
                    let key = ann["target"].get("annotation").map(py::py_repr).unwrap_or_else(|| "None".into());
                    let Some(target) = by_id.get(&key) else { continue };
                    let (name, model, is_person) = who(target.get("creator"));
                    let tb = bernio(target);
                    let mut about = Map::new();
                    about.insert("by".into(), json!(if is_person { "person" } else { "agent" }));
                    if !is_person {
                        about.insert("agent".into(), json!(name));
                        about.insert("model".into(), json!(model));
                    }
                    about.insert("run".into(), json!(match target.get("generator") {
                        Some(Value::Object(g)) => text_of(g.get("run"), 64),
                        _ => String::new(),
                    }));
                    about.insert("kind".into(), json!(kind_of(target)));
                    about.insert("quote".into(), json!(text_of(selector(target, "TextQuoteSelector").and_then(|s| s.get("exact")), MAX_QUOTE)));
                    about.insert("note".into(), json!(body_of(target, "commenting", MAX_NOTE)));
                    about.insert("suggestion".into(), json!(body_of(target, "replacing", MAX_REPLACEMENT)));
                    for k in ["aspect", "importance", "clause_type", "who", "what", "to_whom", "when", "norm"] {
                        match tb.get(k) {
                            None | Some(Value::Null) => {}
                            Some(Value::String(s)) if s.is_empty() => {}
                            Some(v) => {
                                about.insert(k.into(), v.clone());
                            }
                        }
                    }
                    out.push(json!({"type": "decision", "document": path, "at": text_of(ann.get("created"), 40), "about": about,
                                    "verdict": bernio(ann).get("verdict").cloned().unwrap_or(Value::Null),
                                    "comment": body_of(ann, "commenting", MAX_NOTE),
                                    "correction": body_of(ann, "replacing", MAX_REPLACEMENT)}));
                }
                "revision" if !agent_wrote.is_empty() => {
                    let (name, model, _) = who(agent_wrote.last().unwrap().get("creator"));
                    let changes = match bernio(ann).get("hunks") {
                        Some(v) if py::truthy(Some(v)) => v.clone(),
                        _ => json!([]),
                    };
                    out.push(json!({"type": "revision", "document": path, "at": text_of(ann.get("created"), 40),
                                    "written_by": {"agent": name, "model": model}, "changes": changes}));
                }
                _ => {}
            }
        }
    }
    out
}

/// `records` with every text in them tokenized by the scope's engine.
pub fn dataset(folder: &Path, engine: &ScopeEngine) -> Result<Vec<Value>, String> {
    fn collect(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Object(m) => m.values().for_each(|x| collect(x, out)),
            Value::Array(a) => a.iter().for_each(|x| collect(x, out)),
            _ => {}
        }
    }
    fn rebuild(v: &Value, swap: &mut std::vec::IntoIter<String>) -> Value {
        match v {
            Value::String(_) => json!(swap.next().unwrap_or_default()),
            Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), rebuild(x, swap))).collect()),
            Value::Array(a) => Value::Array(a.iter().map(|x| rebuild(x, swap)).collect()),
            other => other.clone(),
        }
    }
    let mut out = Vec::new();
    for rec in records(folder) {
        let mut texts = Vec::new();
        collect(&rec, &mut texts);
        let reds = if texts.is_empty() { Vec::new() } else { engine.tokenize_all(&texts)? };
        let mut swap = reds.into_iter().map(|r| r.text).collect::<Vec<_>>().into_iter();
        out.push(rebuild(&rec, &mut swap));
    }
    Ok(out)
}
