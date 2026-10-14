// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Python's text rules, as the Python agent has them, so the Rust one counts, cuts, folds and writes
//! text the same way: lengths and limits in characters (not bytes), `str.splitlines`, `str.split()`,
//! `str.strip()`, `str.casefold()` (tables from Python itself, `pytables.rs`), and `json.dumps` with
//! `ensure_ascii=False` — its separators decide the sizes the window is estimated from.

use serde_json::Value;

use std::sync::OnceLock;

use crate::unicode;
use crate::unicode_tables::{DIGIT_ZEROS, WORD};

/// `str.isspace()` of one character.
pub fn is_space(c: char) -> bool {
    unicode::is_space(c)
}

/// `len(text)`: characters, not bytes.
pub fn clen(text: &str) -> usize {
    text.chars().count()
}

/// The byte offset of character `n` (the end, when the text is shorter).
pub fn byte_at(text: &str, n: usize) -> usize {
    text.char_indices().nth(n).map_or(text.len(), |(b, _)| b)
}

/// `text[:n]`.
pub fn take(text: &str, n: usize) -> &str {
    &text[..byte_at(text, n)]
}

/// `text[a:b]` (a and b in characters, clamped as Python does).
pub fn slice(text: &str, a: usize, b: usize) -> &str {
    let start = byte_at(text, a);
    let end = byte_at(text, b.max(a));
    &text[start..end]
}

/// `text[-n:]`.
pub fn last(text: &str, n: usize) -> &str {
    let total = clen(text);
    if n >= total {
        text
    } else {
        &text[byte_at(text, total - n)..]
    }
}

/// `str.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

pub fn lstrip(text: &str) -> &str {
    text.trim_start_matches(is_space)
}

pub fn rstrip(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

/// `str.strip(chars)`.
pub fn strip_chars<'a>(text: &'a str, chars: &str) -> &'a str {
    text.trim_matches(|c| chars.contains(c))
}

/// `str.split()`: runs of whitespace, none empty.
pub fn split_ws(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|p| !p.is_empty()).collect()
}

/// `" ".join(text.split())`.
pub fn squash(text: &str) -> String {
    split_ws(text).join(" ")
}

/// `str.splitlines()`: the line boundaries Python knows, `\r\n` as one, no empty line after the last one.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary = matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
            | '\u{2028}' | '\u{2029}');
        if !boundary {
            continue;
        }
        out.push(&text[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(_, '\n')) = chars.peek() {
                chars.next();
                next += 1;
            }
        }
        start = next;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `str.casefold()`.
pub fn casefold(text: &str) -> String {
    unicode::casefold(text)
}

fn ranges(pairs: impl Iterator<Item = (u32, u32)>) -> String {
    pairs.map(|(a, b)| format!("\\x{{{a:x}}}-\\x{{{b:x}}}")).collect()
}

/// The insides of a character class for Python's `\s`, `\w` and `\d`.
fn class_body(c: char) -> &'static str {
    static WORD_BODY: OnceLock<String> = OnceLock::new();
    static DIGIT_BODY: OnceLock<String> = OnceLock::new();
    match c {
        // `str.isspace()`: White_Space and U+001C–U+001F
        's' => r"\s\x1C-\x1F",
        'w' => WORD_BODY.get_or_init(|| ranges(WORD.iter().copied())),
        _ => DIGIT_BODY.get_or_init(|| ranges(DIGIT_ZEROS.iter().map(|z| (*z, z + 9)))),
    }
}

/// A Python regular expression in the Rust engine's spelling. Python's `\s`, `\w` and `\d` are its own
/// (`str.isspace()`, `str.isalnum()` and `_`, decimal digits of its Unicode), written out as classes; `\Z`
/// is the engine's `\z`. (`\b` stays the engine's: it differs only next to combining marks.)
pub fn re_source(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut class = 0usize;
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(e @ ('s' | 'w' | 'd')) if class > 0 => out.push_str(class_body(e)),
                Some(e @ ('s' | 'w' | 'd')) => out.push_str(&format!("[{}]", class_body(e))),
                Some(e @ ('S' | 'W' | 'D')) => out.push_str(&format!("[^{}]", class_body(e.to_ascii_lowercase()))),
                Some('Z') => out.push_str(r"\z"),
                Some(e) => {
                    out.push('\\');
                    out.push(e);
                }
                None => out.push('\\'),
            },
            '[' => {
                class += 1;
                out.push(c);
                // `[]` and `[^]` open with a literal bracket in Python
                if chars.peek() == Some(&'^') {
                    out.push(chars.next().unwrap());
                }
                if chars.peek() == Some(&']') {
                    chars.next();
                    out.push_str(r"\]");
                }
            }
            ']' if class > 0 => {
                class -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// `re.compile(pattern)` of a pattern of the agent's own (it is known to compile).
pub fn re(pattern: &str) -> regex::Regex {
    regex::Regex::new(&re_source(pattern)).unwrap_or_else(|e| panic!("a regex of the agent: {pattern}: {e}"))
}

/// `str.isdigit()` for what `int()` takes: ASCII digits only (a superscript two is a digit to
/// `isdigit` but not to `int`).
pub fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

// ── json.dumps(…, ensure_ascii=False) ───────────────────────────────────────────────────────────────

/// `json.dumps(value, ensure_ascii=False)`: `", "` and `": "` between items, keys in their order.
pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write(value, false, &mut out);
    out
}

/// `json.dumps(value, sort_keys=True, ensure_ascii=False)`.
pub fn dumps_sorted(value: &Value) -> String {
    let mut out = String::new();
    write(value, true, &mut out);
    out
}

fn write(value: &Value, sorted: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(item, sorted, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            if sorted {
                keys.sort();
            }
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                string(k, out);
                out.push_str(": ");
                write(&map[k.as_str()], sorted, out);
            }
            out.push('}');
        }
    }
}

/// A number as Python writes it: integers as they are, floats as `repr` (`1.0`, `0.5`, `1e+16`).
fn number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    float_repr(f)
}

pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == f.trunc() && f.abs() < 1e16 {
        return format!("{f:.1}");
    }
    // Shortest round-trip digits, then Python's choice between plain and exponent form.
    let shortest = format!("{f:e}");
    let (mantissa, exp) = shortest.split_once('e').unwrap_or((&shortest, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        let plain = format!("{f}");
        if plain.contains('.') { plain } else { format!("{plain}.0") }
    } else {
        let mantissa = mantissa.to_string();
        format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.loads(text, strict=False)`-ish: a JSON value, or None.
pub fn loads(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok()
}

// ── small helpers for JSON dicts as Python reads them ───────────────────────────────────────────────

/// `d.get(key)` of a JSON object (None for anything else).
pub fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|m| m.get(key))
}

/// `d.get(key, "")` as a string: the string, else "".
pub fn get_str<'a>(value: &'a Value, key: &str) -> &'a str {
    get(value, key).and_then(Value::as_str).unwrap_or("")
}

/// `int(value)` of a JSON value: None where Python would raise.
pub fn int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(b) => Some(*b as i64),
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Value::String(s) => parse_int(s),
        _ => None,
    }
}

/// The value of a decimal digit of any script, as `int()` reads it.
pub fn digit(c: char) -> Option<i64> {
    let c = c as u32;
    let i = DIGIT_ZEROS.partition_point(|z| *z <= c);
    let zero = *DIGIT_ZEROS.get(i.checked_sub(1)?)?;
    (c - zero < 10).then_some((c - zero) as i64)
}

/// `int(text)` of a string: whitespace around, a sign, decimal digits of any script with single
/// underscores between them; None where Python would raise (or the number is past an i64).
pub fn parse_int(text: &str) -> Option<i64> {
    let t = strip(text);
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let mut n: i64 = 0;
    for c in digits.chars().filter(|c| *c != '_') {
        n = n.checked_mul(10)?.checked_add(digit(c)?)?;
    }
    Some(sign * n)
}

/// Python's truthiness of a JSON value.
pub fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str(value)` of a JSON value as Python would print the object it was parsed to.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => number(n),
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// `repr(value)`.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => repr_str(s),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), py_repr(v)))
            .collect::<Vec<_>>().join(", ")),
        other => py_str(other),
    }
}

/// `repr(text)` of a string: single quotes unless it holds one and no double quote.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\u{7f}' => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lines_as_python_splits_them() {
        assert_eq!(splitlines("a\nb\r\nc\rd\u{2028}e\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n\n"), vec!["", ""]);
        assert_eq!(splitlines("x\u{0c}y"), vec!["x", "y"]);
    }

    #[test]
    fn json_as_python_writes_it() {
        let v = json!({"b": [1, "ü\n\"x\""], "a": {"z": true, "y": null}, "f": 1.5});
        assert_eq!(dumps(&v), r#"{"b": [1, "ü\n\"x\""], "a": {"z": true, "y": null}, "f": 1.5}"#);
        assert_eq!(dumps_sorted(&v), r#"{"a": {"y": null, "z": true}, "b": [1, "ü\n\"x\""], "f": 1.5}"#);
        assert_eq!(dumps(&json!("\u{1}")), "\"\\u0001\"");
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(0.00001), "1e-05");
        assert_eq!(float_repr(1.5e-7), "1.5e-07");
    }

    #[test]
    fn folding_and_spaces_as_python() {
        assert_eq!(casefold("Straße ΣΑΣ"), "strasse σασ");
        assert!(is_space('\u{1c}') && is_space('\u{a0}') && !is_space('x'));
        assert_eq!(squash("  a \u{a0} b\n"), "a b");
        assert_eq!(take("añb", 2), "añ");
        assert_eq!(last("añb", 2), "ñb");
        assert_eq!(slice("abcdef", 2, 4), "cd");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'b\"c"), "'a\\'b\"c'");
    }
}

/// `json.dumps(value)` with Python's defaults: `ensure_ascii=True` (every character past ASCII as `\uXXXX`,
/// a pair of them past the BMP) and the `", "` / `": "` separators.
pub fn dumps_ascii(value: &serde_json::Value) -> String {
    let text = dumps(value);
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if (c as u32) < 0x80 {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

// ── paths, as os.path.realpath(strict=False) resolves them ──────────────────────────────────────────

use std::path::{Component, Path, PathBuf};

/// `os.path.realpath(path)`: every link resolved, `..` taken after the real path so far; what does not
/// exist is kept as written. A link loop stops where it was found.
pub fn realpath(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(path)
    };
    let mut seen = std::collections::HashMap::new();
    let (resolved, _) = join_real(PathBuf::from("/"), &abs, &mut seen, 0);
    resolved
}

fn join_real(mut path: PathBuf, rest: &Path, seen: &mut std::collections::HashMap<PathBuf, Option<PathBuf>>,
             depth: usize) -> (PathBuf, bool) {
    let parts: Vec<Component> = rest.components().collect();
    let mut i = 0;
    while i < parts.len() {
        let part = parts[i];
        i += 1;
        match part {
            Component::RootDir => path = PathBuf::from("/"),
            Component::CurDir | Component::Prefix(_) => {}
            Component::ParentDir => {
                if path.file_name().is_some() {
                    path.pop();
                }
            }
            Component::Normal(name) => {
                let newpath = path.join(name);
                let is_link = std::fs::symlink_metadata(&newpath).is_ok_and(|m| m.file_type().is_symlink());
                if !is_link {
                    path = newpath;
                    continue;
                }
                if let Some(known) = seen.get(&newpath) {
                    match known {
                        Some(real) => {
                            path = real.clone();
                            continue;
                        }
                        None => {
                            // a loop: the rest as it is written
                            let mut out = newpath;
                            for p in &parts[i..] {
                                out.push(p.as_os_str());
                            }
                            return (out, false);
                        }
                    }
                }
                if depth > 64 {
                    return (newpath, false);
                }
                seen.insert(newpath.clone(), None);
                let target = match std::fs::read_link(&newpath) {
                    Ok(t) => t,
                    Err(_) => {
                        path = newpath;
                        continue;
                    }
                };
                let (real, ok) = join_real(path.clone(), &target, seen, depth + 1);
                if !ok {
                    let mut out = real;
                    for p in &parts[i..] {
                        out.push(p.as_os_str());
                    }
                    return (out, false);
                }
                seen.insert(newpath, Some(real.clone()));
                path = real;
            }
        }
    }
    (path, true)
}

/// `Path(text).expanduser()`.
pub fn expanduser(text: &str) -> PathBuf {
    if text == "~" || text.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let mut p = PathBuf::from(home);
            if text.len() > 2 {
                p.push(&text[2..]);
            }
            return p;
        }
    }
    PathBuf::from(text)
}


/// `json.dumps(value, ensure_ascii=False, sort_keys=sort, separators=(",", ":") if compact)`.
pub fn dumps_with(value: &serde_json::Value, sort: bool, compact: bool) -> String {
    fn walk(v: &serde_json::Value, sort: bool, compact: bool, out: &mut String) {
        use serde_json::Value;
        let (item, kv) = if compact { (",", ":") } else { (", ", ": ") };
        match v {
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    walk(x, sort, compact, out);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                let mut keys: Vec<&String> = m.keys().collect();
                if sort {
                    keys.sort();
                }
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    out.push_str(&dumps(&Value::String(k.clone())));
                    out.push_str(kv);
                    walk(&m[k], sort, compact, out);
                }
                out.push('}');
            }
            other => out.push_str(&dumps(other)),
        }
    }
    let mut out = String::new();
    walk(value, sort, compact, &mut out);
    out
}
