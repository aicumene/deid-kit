// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! deidkit.namefold's fold in Rust — `fold`, `confusable`, `strip_marks`, and `fold_haystack` and `key`
//! made of them — the same steps over the same data, for speed: in Python they walk the text one
//! character at a time, and on a long document that walk was three quarters of a request's time in the
//! proxy (measured 10.10.2026: 0.54 s for a 183K-character contract).
//!
//! THE SAME DATA. What the Python fold asks of the interpreter's Unicode database per code point —
//! NFKC, casefold, the categories, `\s`, the base NFD leaves — is in tables made by that interpreter
//! when the module is built (gen_tables.py). They carry its Unicode version, and the module refuses to
//! start for an interpreter with another (`Fold::new`). NFKC of a cluster of several code points (a
//! base with combining marks) is asked of the interpreter itself: composition is not a per-code-point
//! table. What deid-kit decides itself — the punctuation shapes, the homoglyphs, the letters kept —
//! the module is given by deidkit.namefold when it starts, so the two cannot drift; the one pattern it
//! implements in code (`_LINEBREAK_HYPHEN`) it checks against the one it is given.
//!
//! Offsets are in code points, as Python's string indices are.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[allow(clippy::all)]
mod tables {
    include!(concat!(env!("OUT_DIR"), "/tables.rs"));
}

/// `deidkit.textmatch._LINEBREAK_HYPHEN`, which [`skips`] implements.
const LINEBREAK_HYPHEN: &str = "[-\u{2010}\u{2011}\u{2012}\u{2013}\u{2014}\u{2212}]\\s*\\n\\s*";
const HYPHENS: [char; 7] = ['-', '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}'];
/// `deidkit.namefold._WS_RE`: one whitespace character, as `re` sees it — the SPACE table.
const WS: &str = "\\s";

// ── the tables ───────────────────────────────────────────────────────────────────────────────

fn in_ranges(table: &[(u32, u32)], c: u32) -> bool {
    table
        .binary_search_by(|&(a, b)| {
            if c < a {
                std::cmp::Ordering::Greater
            } else if c > b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn find<T: Copy>(table: &[(u32, T)], c: u32) -> Option<T> {
    table.binary_search_by_key(&c, |&(k, _)| k).ok().map(|i| table[i].1)
}

/// The same answers for ASCII, looked up once: most of a legal text is ASCII.
struct Ascii {
    space: [bool; 128],
    cf: [bool; 128],
    cluster_mark: [bool; 128],
    mark: [bool; 128],
    nfkc: [Option<&'static str>; 128],
    casefold: [Option<&'static str>; 128],
    base: [Option<char>; 128],
}

fn ascii() -> &'static Ascii {
    static ASCII: OnceLock<Ascii> = OnceLock::new();
    ASCII.get_or_init(|| {
        let mut a = Ascii {
            space: [false; 128],
            cf: [false; 128],
            cluster_mark: [false; 128],
            mark: [false; 128],
            nfkc: [None; 128],
            casefold: [None; 128],
            base: [None; 128],
        };
        for c in 0..128u32 {
            let i = c as usize;
            a.space[i] = in_ranges(tables::SPACE, c);
            a.cf[i] = in_ranges(tables::CF, c);
            a.cluster_mark[i] = in_ranges(tables::CLUSTER_MARK, c);
            a.mark[i] = in_ranges(tables::MARK, c);
            a.nfkc[i] = find(tables::NFKC, c);
            a.casefold[i] = find(tables::CASEFOLD, c);
            a.base[i] = find(tables::NFD_BASE, c);
        }
        a
    })
}

#[inline]
fn is_space(ch: char) -> bool {
    let c = ch as u32;
    if c < 128 { ascii().space[c as usize] } else { in_ranges(tables::SPACE, c) }
}

#[inline]
fn is_cf(ch: char) -> bool {
    let c = ch as u32;
    if c < 128 { ascii().cf[c as usize] } else { in_ranges(tables::CF, c) }
}

/// A character `_clusters` keeps with the one before it: `combining() != 0` or a mark category.
#[inline]
fn is_cluster_mark(ch: char) -> bool {
    let c = ch as u32;
    if c < 128 { ascii().cluster_mark[c as usize] } else { in_ranges(tables::CLUSTER_MARK, c) }
}

/// A character `strip_marks` drops: categories Mn, Mc, Me.
#[inline]
fn is_mark(ch: char) -> bool {
    let c = ch as u32;
    if c < 128 { ascii().mark[c as usize] } else { in_ranges(tables::MARK, c) }
}

/// NFKC of one code point, where it is not the code point itself.
#[inline]
fn nfkc1(ch: char) -> Option<&'static str> {
    let c = ch as u32;
    if c < 128 { ascii().nfkc[c as usize] } else { find(tables::NFKC, c) }
}

#[inline]
fn casefold(ch: char) -> Option<&'static str> {
    let c = ch as u32;
    if c < 128 { ascii().casefold[c as usize] } else { find(tables::CASEFOLD, c) }
}

/// The one base character NFD leaves without its combining marks, where it is another character.
#[inline]
fn nfd_base(ch: char) -> Option<char> {
    let c = ch as u32;
    if c < 128 { ascii().base[c as usize] } else { find(tables::NFD_BASE, c) }
}

// ── the steps ────────────────────────────────────────────────────────────────────────────────

/// The characters `_LINEBREAK_HYPHEN.finditer` covers: a hyphen and the run of whitespace after it,
/// when the run holds a line break (`\s*` gives back up to the last `\n` of the run, the second `\s*`
/// takes the rest of it).
fn skips(chars: &[char]) -> Vec<bool> {
    let n = chars.len();
    let mut skip = vec![false; n];
    let mut i = 0;
    while i < n {
        if HYPHENS.contains(&chars[i]) {
            let mut j = i + 1;
            let mut newline = false;
            while j < n && is_space(chars[j]) {
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

/// The folded text being written, with the raw span of every character.
struct Out {
    text: String,
    starts: Vec<usize>,
    ends: Vec<usize>,
    pending_ws: Option<(usize, usize)>,
}

impl Out {
    fn emit(&mut self, ch: char, s: usize, e: usize) {
        self.text.push(ch);
        self.starts.push(s);
        self.ends.push(e);
    }
}

/// The text without the marks `strip_marks` drops, with the offsets each kept character covers.
struct Stripped {
    text: Vec<char>,
    starts: Vec<usize>,
    ends: Vec<usize>,
}

fn strip(chars: &[char], offsets: Option<(&[usize], &[usize])>) -> PyResult<Option<Stripped>> {
    if !chars.iter().any(|&c| is_mark(c)) {
        return Ok(None);
    }
    if let Some((s, e)) = offsets {
        if s.len() < chars.len() || e.len() < chars.len() {
            return Err(PyIndexError::new_err("tuple index out of range"));
        }
    }
    let mut out = Stripped { text: Vec::with_capacity(chars.len()), starts: Vec::new(), ends: Vec::new() };
    for (i, &ch) in chars.iter().enumerate() {
        if is_mark(ch) && !out.text.is_empty() {
            if let Some((_, e)) = offsets {
                if let Some(last) = out.ends.last_mut() {
                    *last = (*last).max(e[i]);
                }
            }
            continue;
        }
        out.text.push(ch);
        if let Some((s, e)) = offsets {
            out.starts.push(s[i]);
            out.ends.push(e[i]);
        }
    }
    Ok(Some(out))
}

fn one(s: &str) -> PyResult<char> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyValueError::new_err(format!("not one character: {s:?}"))),
    }
}

/// The fold, set up with what deidkit.namefold decides itself.
#[pyclass(frozen, module = "deidkit_fold")]
struct Fold {
    punct: HashMap<char, char>,
    homoglyph: HashMap<char, char>,
    singleton: HashMap<char, char>,
    keep: HashSet<char>,
    /// `unicodedata.normalize`, for NFKC of a cluster of several code points.
    normalize: Py<PyAny>,
}

impl Fold {
    fn fold_chars(&self, py: Python<'_>, raw: &str) -> PyResult<Out> {
        let chars: Vec<char> = raw.chars().collect();
        let n = chars.len();
        let skip = skips(&chars);
        let mut out = Out { text: String::with_capacity(raw.len()), starts: Vec::with_capacity(n),
                            ends: Vec::with_capacity(n), pending_ws: None };
        let mut i = 0;
        while i < n {
            if skip[i] {
                i += 1;
                continue;
            }
            let s = i;
            i += 1;
            while i < n && !skip[i] && is_cluster_mark(chars[i]) {
                i += 1;
            }
            let e = i;
            if e - s == 1 {
                match nfkc1(chars[s]) {
                    Some(t) => t.chars().for_each(|ch| self.step(&mut out, ch, s, e)),
                    None => self.step(&mut out, chars[s], s, e),
                }
            } else {
                let piece: String = chars[s..e].iter().collect();
                let composed: String = self.normalize.bind(py).call1(("NFKC", piece))?.extract()?;
                composed.chars().for_each(|ch| self.step(&mut out, ch, s, e));
            }
        }
        // trailing whitespace is stripped, as in `normalize`
        Ok(out)
    }

    /// One character of a cluster's NFKC: format characters dropped, punctuation folded, whitespace
    /// collapsed into one space (none at the start, none at the end), casefolded.
    fn step(&self, out: &mut Out, ch: char, s: usize, e: usize) {
        if is_cf(ch) {
            return;
        }
        let ch = *self.punct.get(&ch).unwrap_or(&ch);
        if is_space(ch) {
            out.pending_ws = Some(match out.pending_ws {
                Some((ps, _)) => (ps, e),
                None => (s, e),
            });
            return;
        }
        if let Some((ps, pe)) = out.pending_ws.take() {
            if !out.starts.is_empty() {
                out.emit(' ', ps, pe);
            }
        }
        match casefold(ch) {
            Some(folded) => folded.chars().for_each(|c| out.emit(c, s, e)),
            None => out.emit(ch, s, e),
        }
    }

    fn confusable_char(&self, ch: char) -> char {
        if self.keep.contains(&ch) {
            return ch;
        }
        if let Some(&rep) = self.homoglyph.get(&ch).or_else(|| self.singleton.get(&ch)) {
            return rep;
        }
        match nfd_base(ch) {
            Some(base) => *self.homoglyph.get(&base).unwrap_or(&base),
            None => ch,
        }
    }
}

#[pymethods]
impl Fold {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, unidata_version: &str, linebreak_hyphen: &str, ws: &str,
           punct_fold: HashMap<u32, String>, homoglyph: HashMap<String, String>,
           singleton: HashMap<String, String>, keep: Vec<String>) -> PyResult<Self> {
        if unidata_version != tables::UNIDATA_VERSION {
            return Err(PyValueError::new_err(format!(
                "the tables are of Unicode {} (Python {}), this interpreter has {unidata_version}",
                tables::UNIDATA_VERSION, tables::PYTHON_VERSION)));
        }
        if linebreak_hyphen != LINEBREAK_HYPHEN || ws != WS {
            return Err(PyValueError::new_err("the patterns are not the ones this module implements"));
        }
        let pairs = |map: HashMap<String, String>| -> PyResult<HashMap<char, char>> {
            map.iter().map(|(k, v)| Ok((one(k)?, one(v)?))).collect()
        };
        let punct = punct_fold
            .iter()
            .map(|(k, v)| {
                let k = char::from_u32(*k).ok_or_else(|| PyValueError::new_err(format!("not a character: {k}")))?;
                Ok((k, one(v)?))
            })
            .collect::<PyResult<HashMap<char, char>>>()?;
        Ok(Fold {
            punct,
            homoglyph: pairs(homoglyph)?,
            singleton: pairs(singleton)?,
            keep: keep.iter().map(|k| one(k)).collect::<PyResult<_>>()?,
            normalize: py.import("unicodedata")?.getattr("normalize")?.unbind(),
        })
    }

    /// `namefold.fold`: (text, starts, ends).
    fn fold<'py>(&self, py: Python<'py>, raw: &str)
                 -> PyResult<(String, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        let out = self.fold_chars(py, raw)?;
        Ok((out.text, PyTuple::new(py, out.starts)?, PyTuple::new(py, out.ends)?))
    }

    /// `namefold.confusable`.
    fn confusable(&self, text: &str) -> String {
        text.chars().map(|c| self.confusable_char(c)).collect()
    }

    /// `namefold.strip_marks` with the folded offsets given apart: None when there is no mark to drop,
    /// else (text, starts, ends) — the offsets None when none were given.
    #[pyo3(signature = (text, starts=None, ends=None))]
    #[allow(clippy::type_complexity)]
    fn strip_marks<'py>(&self, py: Python<'py>, text: &str, starts: Option<Vec<usize>>, ends: Option<Vec<usize>>)
                        -> PyResult<Option<(String, Option<Bound<'py, PyTuple>>, Option<Bound<'py, PyTuple>>)>> {
        let chars: Vec<char> = text.chars().collect();
        let offsets = match (&starts, &ends) {
            (Some(s), Some(e)) => Some((s.as_slice(), e.as_slice())),
            _ => None,
        };
        let Some(out) = strip(&chars, offsets)? else { return Ok(None) };
        let text: String = out.text.iter().collect();
        if offsets.is_none() {
            return Ok(Some((text, None, None)));
        }
        Ok(Some((text, Some(PyTuple::new(py, out.starts)?), Some(PyTuple::new(py, out.ends)?))))
    }

    /// `namefold.fold_haystack`: (searchable text, the text of its Folded, starts, ends). Without a mark to
    /// drop, the Folded is the fold's own, its text the fold's text, as in Python.
    fn fold_haystack<'py>(&self, py: Python<'py>, raw: &str)
                          -> PyResult<(String, String, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        let folded = self.fold_chars(py, raw)?;
        let confused: Vec<char> = folded.text.chars().map(|c| self.confusable_char(c)).collect();
        match strip(&confused, Some((&folded.starts, &folded.ends)))? {
            None => Ok((confused.iter().collect(), folded.text,
                        PyTuple::new(py, folded.starts)?, PyTuple::new(py, folded.ends)?)),
            Some(out) => {
                let text: String = out.text.iter().collect();
                Ok((text.clone(), text, PyTuple::new(py, out.starts)?, PyTuple::new(py, out.ends)?))
            }
        }
    }

    /// `namefold.key` after `normalize`: the confusable fold, marks dropped.
    fn key_tail(&self, normalized: &str) -> PyResult<String> {
        let confused: Vec<char> = normalized.chars().map(|c| self.confusable_char(c)).collect();
        Ok(match strip(&confused, None)? {
            None => confused.iter().collect(),
            Some(out) => out.text.iter().collect(),
        })
    }
}

#[pymodule]
fn deidkit_fold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Fold>()?;
    m.add("UNIDATA_VERSION", tables::UNIDATA_VERSION)?;
    m.add("PYTHON_VERSION", tables::PYTHON_VERSION)?;
    m.add("LINEBREAK_HYPHEN", LINEBREAK_HYPHEN)?;
    Ok(())
}
