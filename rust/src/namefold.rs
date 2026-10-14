// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Name folding, offset-preserving matching, and the vault's admission predicate (as `namefold.py`).
//!
//! `fold` reproduces `textmatch::normalize` exactly and records, per folded character, the raw span it
//! came from (offsets in characters, as Python's indices are); `confusable` folds diacritics and
//! homoglyphs one to one; `strip_marks` drops the marks casefold creates; `key` is the vault's match key.
//! Then the word-boundary predicate, the identity skeletons, the surface variants enrolled at seed time,
//! and the gate a residual NER span passes before it becomes a token of its own.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use regex::Regex;

use crate::py;
use crate::textmatch::{self, linebreak_hyphens, punct_fold};
use crate::unicode as u;

// ── 1. the fold, with offsets ──────────────────────────────────────────────────────────────────

/// 1→1 replacements for letters whose NFD carries no combining mark.
fn singleton(c: char) -> Option<char> {
    Some(match c {
        'ł' => 'l', 'ø' => 'o', 'đ' => 'd', 'ð' => 'd', 'þ' => 't', 'ħ' => 'h', 'ı' => 'i', 'ŧ' => 't',
        _ => return None,
    })
}

/// Visually identical letters from other scripts, mapped onto their Latin twin.
fn homoglyph(c: char) -> Option<char> {
    Some(match c {
        'α' => 'a', 'β' => 'b', 'γ' => 'y', 'ε' => 'e', 'ζ' => 'z', 'η' => 'h', 'ι' => 'i', 'κ' => 'k',
        'μ' => 'm', 'ν' => 'v', 'ο' => 'o', 'ρ' => 'p', 'τ' => 't', 'υ' => 'y', 'χ' => 'x', 'ϲ' => 'c',
        'а' => 'a', 'с' => 'c', 'е' => 'e', 'о' => 'o', 'р' => 'p', 'х' => 'x', 'у' => 'y', 'ѕ' => 's',
        'і' => 'i', 'ј' => 'j', 'ԁ' => 'd', 'ѵ' => 'v', 'ԛ' => 'q', 'ԝ' => 'w',
        _ => return None,
    })
}

/// Cyrillic letters whose "diacritic" is a letter of its own: pinned through unchanged.
const CONFUSABLE_KEEP: char = 'й';

/// A folded string plus, per folded character, the raw half-open span it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Folded {
    pub text: String,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
    pub raw_len: usize,
}

impl Folded {
    /// Raw half-open span covering folded[start:end].
    pub fn raw_span(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        (start < end).then(|| (self.starts[start], self.ends[end - 1]))
    }
}

fn is_cluster_mark(c: char) -> bool {
    u::combining(c) != 0 || matches!(u::category(c), "Mn" | "Mc" | "Me")
}

fn is_mark(c: char) -> bool {
    matches!(u::category(c), "Mn" | "Mc" | "Me")
}

/// `textmatch.normalize(raw)`, with a raw span recorded for every folded character.
pub fn fold(raw: &str) -> Folded {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    let skip = linebreak_hyphens(&chars);
    let mut text = String::with_capacity(raw.len());
    let (mut starts, mut ends) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut pending_ws: Option<(usize, usize)> = None;
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
        let piece: String = chars[s..e].iter().collect();
        for ch in u::normalize("NFKC", &piece).chars() {
            if u::category(ch) == "Cf" {
                continue;
            }
            let ch = punct_fold(ch).unwrap_or(ch);
            if u::is_space(ch) {
                // collapse: remember the run, emit one space when a non-space arrives
                pending_ws = Some(match pending_ws {
                    Some((ps, _)) => (ps, e),
                    None => (s, e),
                });
                continue;
            }
            if let Some((ps, pe)) = pending_ws.take() {
                if !starts.is_empty() {
                    // leading whitespace is stripped, not emitted
                    text.push(' ');
                    starts.push(ps);
                    ends.push(pe);
                }
            }
            for out in u::casefold(&ch.to_string()).chars() {
                text.push(out);
                starts.push(s);
                ends.push(e);
            }
        }
    }
    // trailing whitespace: `normalize` strips it, so `pending_ws` is dropped.
    Folded { text, starts, ends, raw_len: n }
}

/// The base NFD leaves once its combining marks are gone, if it is one character.
fn nfd_base(c: char) -> Option<char> {
    let base: Vec<char> = u::normalize("NFD", &c.to_string()).chars().filter(|c| u::combining(*c) == 0).collect();
    (base.len() == 1).then(|| base[0])
}

fn nfd_base_cached(c: char) -> Option<char> {
    if c.is_ascii() {
        return Some(c);
    }
    static CACHE: OnceLock<std::sync::Mutex<HashMap<char, Option<char>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(v) = cache.lock().unwrap().get(&c) {
        return *v;
    }
    let v = nfd_base(c);
    cache.lock().unwrap().insert(c, v);
    v
}

/// Length-preserving DIACRITIC fold only (š→s, ć→c, ѐ→е, ł→l). Script is preserved.
pub fn diacritics(folded_text: &str) -> String {
    folded_text.chars().map(|ch| {
        if ch == CONFUSABLE_KEEP {
            return ch;
        }
        if let Some(rep) = singleton(ch) {
            return rep;
        }
        nfd_base_cached(ch).unwrap_or(ch)
    }).collect()
}

fn confusable_char(ch: char) -> char {
    if ch == CONFUSABLE_KEEP {
        return ch;
    }
    if let Some(rep) = homoglyph(ch).or_else(|| singleton(ch)) {
        return rep;
    }
    match nfd_base_cached(ch) {
        Some(base) => homoglyph(base).unwrap_or(base),
        None => ch,
    }
}

/// The MATCH-path fold, applied to an already-folded string: 1:1 and length-preserving.
pub fn confusable(folded_text: &str) -> String {
    folded_text.chars().map(confusable_char).collect()
}

/// Drop combining marks, absorbing each into the PRECEDING character's raw span.
pub fn strip_marks(text: &str, folded: Option<&Folded>) -> (String, Option<Folded>) {
    if !text.chars().any(is_mark) {
        return (text.to_string(), folded.cloned());
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    let mut kept = 0usize;
    for (i, &ch) in chars.iter().enumerate() {
        if is_mark(ch) && kept > 0 {
            if let Some(f) = folded {
                let last = ends.len() - 1;
                ends[last] = std::cmp::max(ends[last], f.ends[i]);
            }
            continue;
        }
        out.push(ch);
        kept += 1;
        if let Some(f) = folded {
            starts.push(f.starts[i]);
            ends.push(f.ends[i]);
        }
    }
    let f = folded.map(|f| Folded { text: out.clone(), starts, ends, raw_len: f.raw_len });
    (out, f)
}

/// The vault's match key for a needle: canonical fold, confusable fold, marks dropped.
pub fn key(value: &str) -> String {
    strip_marks(&confusable(&textmatch::normalize(value)), None).0
}

/// (searchable text, offsets) for a document, folded exactly as `key` folds a needle.
pub fn fold_haystack(raw: &str) -> (String, Folded) {
    let f = fold(raw);
    let (text, f2) = strip_marks(&confusable(&f.text), Some(&f));
    (text, f2.unwrap_or(f))
}

// ── 2. the boundary predicate ──────────────────────────────────────────────────────────────────

/// Word character for boundary purposes: any letter, digit, or combining mark.
pub fn is_word_char(ch: char) -> bool {
    u::is_alnum(ch) || u::category(ch).starts_with('M')
}

fn odd_backslashes_before(hay: &[char], i: isize) -> bool {
    let (mut n, mut i) = (0, i);
    while i >= 0 && hay[i as usize] == '\\' {
        n += 1;
        i -= 1;
    }
    n % 2 == 1
}

/// Does the text before `start` end in a backslash escape (`\n`, `\t`, ` `)?
pub fn after_escape(hay: &[char], start: usize) -> bool {
    if start >= 2 && "ntrfbv".contains(hay[start - 1]) && odd_backslashes_before(hay, start as isize - 2) {
        return true;
    }
    start >= 6 && "uU".contains(hay[start - 5]) && hay[start - 4..start].iter().all(|c| c.is_ascii_hexdigit())
        && odd_backslashes_before(hay, start as isize - 6)
}

/// True when hay[start:end] is a standalone occurrence of `needle` (offsets in characters).
pub fn boundary_ok(hay: &[char], start: usize, end: usize, needle: &[char]) -> bool {
    if let Some(&first) = needle.first() {
        if is_word_char(first) && start > 0 && is_word_char(hay[start - 1]) && !after_escape(hay, start) {
            return false;
        }
    }
    if let Some(&last) = needle.last() {
        if is_word_char(last) && end < hay.len() && is_word_char(hay[end]) {
            return false;
        }
    }
    true
}

/// Every standalone occurrence of `needle` in `hay` (both already folded), as character spans.
pub fn find_all(hay: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let hay_chars: Vec<char> = hay.chars().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    find_all_chars(&hay_chars, &needle_chars)
}

pub fn find_all_chars(hay: &[char], needle: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > hay.len() {
        return out;
    }
    let first = needle[0];
    for i in 0..=hay.len() - needle.len() {
        if hay[i] == first && hay[i..i + needle.len()] == *needle && boundary_ok(hay, i, i + needle.len(), needle) {
            out.push((i, i + needle.len()));
        }
    }
    out
}

// ── 3. identity: script-independent skeletons ──────────────────────────────────────────────────

fn cyr2lat(c: char) -> Option<&'static str> {
    Some(match c {
        'а' => "a", 'б' => "b", 'в' => "v", 'г' => "g", 'д' => "d", 'е' => "e", 'ё' => "e", 'ж' => "zh",
        'з' => "z", 'и' => "i", 'й' => "i", 'к' => "k", 'л' => "l", 'м' => "m", 'н' => "n", 'о' => "o",
        'п' => "p", 'р' => "r", 'с' => "s", 'т' => "t", 'у' => "u", 'ф' => "f", 'х' => "h", 'ц' => "c",
        'ч' => "ch", 'ш' => "sh", 'щ' => "sh", 'ъ' => "", 'ы' => "i", 'ь' => "", 'э' => "e", 'ю' => "iu",
        'я' => "ia", 'і' => "i", 'ї' => "i", 'є' => "e", 'ґ' => "g",
        _ => return None,
    })
}

const SKEL_RULES: [(&str, &str); 15] = [("kh", "h"), ("zh", "j"), ("shch", "sh"), ("tch", "ch"), ("ck", "k"),
    ("yo", "e"), ("ye", "e"), ("ya", "ia"), ("yu", "iu"), ("j", "i"), ("y", "i"), ("ij", "i"), ("ii", "i"),
    ("ie", "e"), ("ei", "i")];

pub fn translit(s: &str) -> String {
    s.chars().map(|c| cyr2lat(c).map_or_else(|| c.to_string(), str::to_string)).collect()
}

fn re(cell: &'static OnceLock<Regex>, src: &str) -> &'static Regex {
    cell.get_or_init(|| py::re(src))
}

/// Script- and romanisation-independent skeleton of ONE name part.
pub fn skeleton_part(part: &str) -> String {
    static NONALNUM: OnceLock<Regex> = OnceLock::new();
    let s = translit(&diacritics(&textmatch::normalize(part)));
    let mut s = re(&NONALNUM, "[^0-9a-zA-Zа-яёA-ZА-ЯЁ]+").replace_all(&s, "").into_owned();
    for (a, b) in SKEL_RULES {
        s = s.replace(a, b);
    }
    // collapse doubled letters ("hallberg" → "halberg")
    let mut out = String::with_capacity(s.len());
    let mut last = None;
    for ch in s.chars() {
        if last != Some(ch) {
            out.push(ch);
        }
        last = Some(ch);
    }
    out
}

/// Name split into parts, punctuation-insensitive, order preserved.
pub fn name_parts(name: &str) -> Vec<String> {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    re(&SPLIT, r#"[\s,./\\()"'’]+"#).split(py::strip(name)).filter(|p| !p.is_empty()).map(str::to_string).collect()
}

pub fn skeleton(name: &str) -> Vec<String> {
    let mut parts: Vec<String> = name_parts(name).iter().map(|x| skeleton_part(x)).filter(|p| !p.is_empty()).collect();
    parts.sort();
    parts
}

/// Two skeleton parts denote the same name part, allowing an inflectional tail.
pub fn stem_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (lo, hi) = if py::clen(a) <= py::clen(b) { (a, b) } else { (b, a) };
    py::clen(lo) >= 4 && hi.starts_with(lo) && py::clen(hi) - py::clen(lo) <= 3
}

// ── 4. surface variants enrolled at seed time ──────────────────────────────────────────────────

pub const LEGAL_FORMS: [&[&str]; 40] = [
    &["ltd", "ltd.", "limited", "co ltd", "co. ltd", "co.ltd"], &["llc", "l.l.c.", "l.l.c", "lc"],
    &["inc", "inc.", "incorporated"], &["corp", "corp.", "corporation"], &["plc", "p.l.c."],
    &["gmbh"], &["ug"], &["ag"], &["kg"], &["se"],
    &["gbr"], &["ohg"], &["kgaa"], &["partg", "partgmbb"], &["llp", "l.l.p."], &["lp", "l.p."],
    &["bv", "b.v.", "b v"], &["nv", "n.v."],
    &["sa", "s.a."], &["sl", "s.l."], &["srl", "s.r.l."], &["spa", "s.p.a."],
    &["dmcc"], &["fzco"], &["fze"], &["pjsc"], &["jsc"], &["pte"], &["oy"], &["ab"],
    &["as", "a/s"], &["doo", "d.o.o.", "d.o.o"], &["ad"], &["sirketi"], &["anonim"],
    &["ооо"], &["оао"], &["зао"], &["пао"], &["ао"],
];

const STRUCTURE_WORDS: &[&str] = &["centre", "center", "tower", "building", "house", "street", "road", "avenue", "plaza",
    "square", "floor", "district", "region", "city", "port", "airport", "hotel", "island",
    "county", "province", "bay", "harbour", "harbor", "station", "park", "bridge",
    "улица", "дом", "здание", "центр", "башня", "площадь", "район", "город", "порт",
    "область", "край", "проспект", "набережная", "остров",
    "straße", "strasse", "str", "weg", "platz", "allee", "ufer", "damm", "ring", "gasse",
    "chaussee", "wohnung", "etage", "stock", "og", "eg", "haus", "gebäude", "stadt"];

const ROLE_STEMS: &[&str] = &["истец", "истц", "ответчик", "ответчиц", "заявител", "заинтересован", "третье лицо",
    "должник", "взыскател", "потерпевш", "свидетел", "представител", "доверител",
    "claimant", "respondent", "plaintiff", "defendant", "applicant", "appellant",
    "petitioner", "witness", "trustee", "liquidator", "administrator",
    "kläger", "beklagt", "antragstell", "antragsgegn", "schuldner", "gläubiger", "zeug",
    "vermieter", "mieter", "geschäftsführ", "gesellschaft", "insolvenzverwalt", "betreuer"];

const GENERIC_WORDS: &[&str] = &["client", "husband", "wife", "spouse", "company", "entity", "management", "shipping",
    "corporation", "holding", "group", "draft", "roadmap", "overview", "shareholding",
    "shareholdings", "director", "manager", "owner", "shareholder", "unknown", "n/a",
    "клиент", "муж", "жена", "супруг", "супруга", "компания", "общество", "подача",
    "уплатить", "предварительное", "давности"];

const HONORIFICS: &[&str] = &["herr", "herrn", "frau", "fräulein", "dr", "prof", "professor", "mr", "mrs", "ms", "miss",
    "mme", "mlle", "sr", "sra", "sig", "sig.ra", "sir", "dame", "lord", "lady", "hon",
    "господин", "госпожа", "г-н", "г-жа"];

const PATRONYMIC_TAILS: &[&str] = &["ович", "евич", "ьич", "овна", "евна", "ична", "инична",
    "ovich", "evich", "yevich", "ayevich", "ovna", "evna", "ichna"];

const COMMON_WORDS: &[&str] = &[
    "forwarded", "message", "original", "subject", "from", "sent", "to", "cc", "bcc",
    "attachment", "attached", "reply", "regards", "best", "dear", "kind", "sincerely",
    "translated", "machine", "translation", "page", "note", "notes", "summary", "memo",
    "confidential", "privileged", "draft", "final", "version", "appendix", "annex",
    "br", "fyi", "asap", "re", "fwd", "ps", "nb",
    "table", "figure", "section", "article", "clause", "schedule", "exhibit",
    "container", "ship", "vessel", "cargo", "bulk", "carrier", "fleet", "charter",
    "trading", "trade", "trades", "invoice", "payment", "account", "bank", "loan",
    "purchase", "sale", "seller", "buyer", "lease", "asset", "assets", "share", "shares",
    "capital", "profit", "loss", "value", "price", "total", "amount", "balance",
    "registration", "register", "certificate", "licence", "license", "agreement",
    "contract", "letter", "notice", "claim", "court", "case", "matter", "law", "legal",
    "verifying", "keeping", "checking", "confirming", "pending", "unconfirmed",
    "yes", "no", "and", "or", "the", "of", "in", "on", "at", "for", "with", "without",
    "debts", "debt", "assets", "annual", "year", "years", "other", "profit", "loss",
    "taxes", "tax", "rules", "code", "cash", "equity", "reserves", "income", "expenses",
    "white", "black", "green", "blue", "red", "golden", "silver", "royal", "grand",
    "global", "international", "national", "general", "united", "new", "old", "great",
    "first", "second", "third", "north", "south", "east", "west", "central", "prosperity",
    "ocean", "sea", "river", "lake", "rock", "star", "sun", "moon", "sky", "wave",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "ihr", "ihre", "ihrer", "ihren", "ihrem", "sie", "wir", "unser", "unsere", "sehr", "geehrte",
    "geehrter", "mit", "freundlichen", "grüßen", "grüssen", "anlage", "schreiben", "betreff",
    "kündigung", "wohnung", "recht", "rechte", "mietvertrag", "mietverhältnis", "miete",
    "eigenbedarf", "widerspruch", "härte", "frist", "termin", "urteil", "beschluss", "antrag",
    "klage", "vertrag", "vereinbarung", "gesellschafterin", "gesellschafter", "hinweis",
    "insert", "title", "rate", "vat", "address", "date", "name", "placeholder", "tbd", "tbc",
];

const JOINERS: &[&str] = &["of", "and", "the", "und", "von", "van", "der", "den", "de", "du", "la", "le", "y"];

struct Folds {
    form_of: HashMap<String, usize>,
    common: HashSet<String>,
    generic: HashSet<String>,
    structure: HashSet<String>,
    role_stems: Vec<String>,
    honorifics: HashSet<String>,
    patronymic_tails: Vec<String>,
}

fn folds() -> &'static Folds {
    static F: OnceLock<Folds> = OnceLock::new();
    F.get_or_init(|| {
        let mut form_of = HashMap::new();
        for (i, grp) in LEGAL_FORMS.iter().enumerate() {
            for w in *grp {
                form_of.insert(key(w), i);
            }
        }
        let set = |words: &[&str]| words.iter().map(|w| key(w)).collect::<HashSet<_>>();
        Folds { form_of, common: set(COMMON_WORDS), generic: set(GENERIC_WORDS), structure: set(STRUCTURE_WORDS),
                role_stems: ROLE_STEMS.iter().map(|w| key(w)).collect(), honorifics: set(HONORIFICS),
                patronymic_tails: PATRONYMIC_TAILS.iter().map(|w| key(w)).collect() }
    })
}

pub fn is_honorific(part: &str) -> bool {
    folds().honorifics.contains(&key(part.trim_end_matches('.')))
}

/// True when `part` is ordinary vocabulary rather than a coined name.
pub fn is_common_word(part: &str) -> bool {
    let k = key(part);
    let f = folds();
    !k.is_empty() && (f.common.contains(&k) || f.generic.contains(&k) || f.structure.contains(&k)
                      || f.role_stems.iter().any(|s| k.starts_with(s.as_str())))
}

pub fn is_generic_folded(k: &str) -> bool {
    folds().generic.contains(k)
}

pub fn is_structure_folded(k: &str) -> bool {
    folds().structure.contains(k)
}

pub fn legal_form_index(part: &str) -> Option<usize> {
    folds().form_of.get(&key(part)).copied()
}

pub fn has_legal_form(name: &str) -> bool {
    name_parts(name).iter().any(|p| legal_form_index(p).is_some())
}

/// Name parts with legal-form words removed.
pub fn strip_legal_form(name: &str) -> Vec<String> {
    name_parts(name).into_iter().filter(|p| legal_form_index(p).is_none()).collect()
}

pub fn is_patronymic(part: &str) -> bool {
    let p = key(part);
    py::clen(&p) >= 6 && folds().patronymic_tails.iter().any(|t| p.ends_with(t.as_str()))
}

pub fn is_initial(part: &str) -> bool {
    let p = py::strip(part).trim_end_matches('.');
    py::clen(p) == 1 && u::isalpha(p)
}

fn wordish() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"[^\W_]+(?:\.[^\W_]+)*\.?")
}

fn spaces2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"\s{2,}")
}

fn spaces1() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"\s+")
}

pub fn dedupe(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for x in values {
        let v = py::strip(&spaces2().replace_all(x, " ")).to_string();
        let k = key(&v);
        if !k.is_empty() && seen.insert(k) {
            out.push(v);
        }
    }
    out
}

/// (start, end, word) of each wordish run, in characters.
fn spans(name: &str) -> Vec<(usize, usize, String)> {
    wordish().find_iter(name).map(|m| (py::clen(&name[..m.start()]), py::clen(&name[..m.end()]), m.as_str().to_string())).collect()
}

fn cut(name: &str, s: usize, e: usize, with: &str) -> String {
    format!("{}{with}{}", py::slice(name, 0, s), py::slice(name, e, py::clen(name)))
}

/// Extra surfaces of one organisation name that a document plausibly uses.
pub fn org_variants(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if py::strip(name).is_empty() {
        return out;
    }
    for (s, e, word) in spans(name) {
        let Some(g) = legal_form_index(&word) else { continue };
        for syn in LEGAL_FORMS[g] {
            out.push(py::strip(&cut(name, s, e, syn)).to_string());
        }
        let bare = cut(name, s, e, "");
        let bare = py::strip(py::strip_chars(py::strip(&bare), ",;.-")).to_string();
        if py::clen(&bare) >= 4 {
            out.push(bare);
        }
    }
    let base = match out.last() {
        Some(last) if py::clen(last) < py::clen(name) => last.clone(),
        _ => name.to_string(),
    };
    let mut heads: Vec<String> = Vec::new();
    for src in dedupe(&[name.to_string(), base.clone()]) {
        out.extend(plural_variants(&src));
        out.extend(shortened_variants(&src));
        heads.extend(initialism_variants(&src));
    }
    out.extend(heads.iter().cloned());
    let mut sources = vec![name.to_string(), base];
    sources.extend(heads);
    for src in dedupe(&sources) {
        let compact = spaces1().replace_all(&src, "").into_owned();
        if py::clen(&compact) >= 6 && compact != src {
            out.push(compact);
        }
    }
    let k = key(name);
    dedupe(&out).into_iter().filter(|v| key(v) != k).collect()
}

fn plural_variants(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let subs: Vec<(usize, usize, String)> = spans(name).into_iter().filter(|(_, _, w)| legal_form_index(w).is_none()).collect();
    if subs.len() < 2 {
        return out;
    }
    for (s, e, w) in subs {
        if py::clen(&w) < 4 || !u::isalpha(&w) || !w.is_ascii() {
            continue;
        }
        let alt = if u::casefold(&w).ends_with('s') {
            py::slice(&w, 0, py::clen(&w) - 1).to_string()
        } else {
            format!("{w}{}", if u::isupper(&w) { "S" } else { "s" })
        };
        if py::clen(&alt) >= 3 {
            out.push(py::strip(&cut(name, s, e, &alt)).to_string());
        }
    }
    out
}

fn shortened_variants(name: &str) -> Vec<String> {
    let subs: Vec<(usize, usize, String)> = spans(name).into_iter().filter(|(_, _, w)| legal_form_index(w).is_none()).collect();
    if subs.len() < 3 {
        return Vec::new();
    }
    let (s, e, _) = subs[subs.len() - 1].clone();
    let short = py::strip(&cut(name, s, e, "")).to_string();
    if py::clen(&short) >= 6 { vec![spaces2().replace_all(&short, " ").into_owned()] } else { Vec::new() }
}

fn initialism_variants(name: &str) -> Vec<String> {
    let subs: Vec<(usize, usize, String)> = spans(name).into_iter().filter(|(_, _, w)| legal_form_index(w).is_none()).collect();
    let mut out = Vec::new();
    for n in [2usize, 3] {
        if subs.len() <= n {
            continue;
        }
        let head = &subs[..n];
        if head.iter().any(|(_, _, w)| py::clen(w) < 3 || !u::isupper(py::take(w, 1)) || !u::isalpha(w)) {
            continue;
        }
        let letters: Vec<String> = head.iter().map(|(_, _, w)| u::upper(py::take(w, 1))).collect();
        let tail = py::slice(name, head[n - 1].1, py::clen(name)).to_string();
        for joiner in [format!("{}.", letters.join(".")), letters.join("."), letters.join("")] {
            out.push(py::strip(&format!("{joiner}{tail}")).to_string());
        }
    }
    out
}

const RU_NOUN_RULES: [(&str, &[&str]); 13] = [
    ("ов", &["ова", "ову", "овым", "ове", "овой", "овы", "овых", "овыми"]),
    ("ев", &["ева", "еву", "евым", "еве", "евой", "евы", "евых", "евыми"]),
    ("ёв", &["ёва", "ёву", "ёвым", "ёве", "ёвой"]),
    ("ин", &["ина", "ину", "иным", "ине", "иной", "ины", "иных"]),
    ("ын", &["ына", "ыну", "ыным", "ыне", "ыной"]),
    ("ский", &["ского", "скому", "ским", "ском", "ская", "ской", "скую"]),
    ("цкий", &["цкого", "цкому", "цким", "цком", "цкая", "цкой", "цкую"]),
    ("ый", &["ого", "ому", "ым", "ом", "ая", "ой", "ую"]),
    ("ий", &["ия", "ию", "ием", "ии"]),
    ("й", &["я", "ю", "ем", "е"]),
    ("а", &["ы", "е", "у", "ой", "ою"]),
    ("я", &["и", "е", "ю", "ей"]),
    ("ь", &["я", "ю", "ем", "е"]),
];

fn first_upper(s: &str) -> String {
    format!("{}{}", u::upper(py::take(s, 1)), py::slice(s, 1, py::clen(s)))
}

/// Declined surfaces of one Cyrillic name part (bounded, deterministic, never a word).
pub fn russian_inflections(part: &str) -> Vec<String> {
    static CYRILLIC: OnceLock<Regex> = OnceLock::new();
    let p = py::strip(part);
    if !re(&CYRILLIC, "^[а-яёА-ЯЁ][а-яёА-ЯЁ-]+$").is_match(p) || py::clen(p) < 5 || is_patronymic(p) || is_common_word(p) {
        return Vec::new();
    }
    let low = u::casefold(p);
    let upper = u::isupper(p);
    let cased = |s: &str| if upper { u::upper(s) } else { first_upper(s) };
    let mut out = Vec::new();
    let mut matched = false;
    for (tail, repls) in RU_NOUN_RULES {
        if !low.ends_with(tail) {
            continue;
        }
        let stem = py::slice(&low, 0, py::clen(&low) - py::clen(tail)).to_string();
        if py::clen(&stem) < 3 {
            continue;
        }
        out.extend(repls.iter().map(|r| cased(&format!("{stem}{r}"))));
        matched = true;
        break;
    }
    if !matched {
        return Vec::new();
    }
    let k = key(p);
    dedupe(&out).into_iter().filter(|v| key(v) != k && !is_common_word(v)).collect()
}

fn lat2cyr_letter(c: char) -> &'static str {
    match c {
        'a' => "а", 'b' => "б", 'c' => "к", 'd' => "д", 'e' => "е", 'f' => "ф", 'g' => "г", 'h' => "х",
        'i' => "и", 'j' => "й", 'k' => "к", 'l' => "л", 'm' => "м", 'n' => "н", 'o' => "о", 'p' => "п",
        'q' => "к", 'r' => "р", 's' => "с", 't' => "т", 'u' => "у", 'v' => "в", 'w' => "в", 'x' => "кс",
        'y' => "у", 'z' => "з", _ => "",
    }
}

pub fn acronym_cyrillic(part: &str) -> Option<String> {
    let p = py::strip(part);
    if !(3..=6).contains(&py::clen(p)) || !p.is_ascii() || !u::isalpha(p) || !u::isupper(p) {
        return None;
    }
    Some(u::upper(&u::casefold(p).chars().map(lat2cyr_letter).collect::<String>()))
}

const CONTRACT_RULES: [(&str, &str); 11] = [("iy", "y"), ("iy", "i"), ("ii", "i"), ("yi", "i"), ("ye", "e"), ("yo", "e"),
    ("kh", "h"), ("ij", "i"), ("j", "y"), ("ts", "c"), ("ck", "k")];
const FINAL_RULES: [(&str, &str); 5] = [("y", "iy"), ("y", "i"), ("i", "iy"), ("iy", "y"), ("ii", "y")];
const INITIAL_RULES: [(&str, &str); 7] = [("e", "ye"), ("ya", "ia"), ("ia", "ya"), ("yu", "iu"), ("iu", "yu"), ("i", "y"), ("y", "i")];

/// Alternative romanisations of one Latin name part (bounded, deterministic).
pub fn translit_variants(part: &str) -> Vec<String> {
    let low = u::casefold(part);
    if !low.is_ascii() || !u::isalpha(&low) || py::clen(&low) < 4 {
        return Vec::new();
    }
    let up = u::isupper(py::take(part, 1));
    let cased = |s: &str| if up { u::capitalize(s) } else { u::lower(s) };
    let mut out = Vec::new();
    for (a, b) in CONTRACT_RULES {
        if low.contains(a) {
            let v = low.replace(a, b);
            if py::clen(&v) >= 4 && v != low {
                out.push(cased(&v));
            }
        }
    }
    for (a, b) in FINAL_RULES {
        if low.ends_with(a) {
            let v = format!("{}{b}", &low[..low.len() - a.len()]);
            if py::clen(&v) >= 4 && v != low {
                out.push(cased(&v));
            }
        }
    }
    for (a, b) in INITIAL_RULES {
        if low.starts_with(a) {
            let v = format!("{b}{}", &low[a.len()..]);
            if py::clen(&v) >= 4 && v != low {
                out.push(cased(&v));
            }
        }
    }
    dedupe(&out).into_iter().take(8).collect()
}

const LAT2CYR: [(&str, &str); 38] = [
    ("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("ch", "ч"), ("sh", "ш"),
    ("ts", "ц"), ("yu", "ю"), ("ya", "я"), ("ye", "е"), ("yo", "ё"), ("iy", "ий"),
    ("y", "й"), ("a", "а"), ("b", "б"), ("v", "в"), ("g", "г"), ("d", "д"), ("e", "е"),
    ("z", "з"), ("i", "и"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"),
    ("p", "п"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("f", "ф"), ("h", "х"),
    ("c", "к"), ("j", "й"), ("w", "в"), ("x", "кс"), ("q", "к"),
];

/// The Cyrillic spelling of a Latin name part, or None when it is not worth guessing.
pub fn cyrillic_form(part: &str) -> Option<String> {
    let mut low = u::casefold(part);
    if !low.is_ascii() || !u::isalpha(&low) || py::clen(&low) < 5 {
        return None;
    }
    // A final "-y" is "-ий" after a consonant and "-й" after a vowel.
    let b = low.as_bytes();
    if low.ends_with('y') && low.len() > 3 && !b"aeiou".contains(&b[b.len() - 2]) {
        low = format!("{}iy", &low[..low.len() - 1]);
    }
    let mut i = 0;
    let mut out = String::new();
    'outer: while i < low.len() {
        for (a, c) in &LAT2CYR {
            if low[i..].starts_with(a) {
                out.push_str(c);
                i += a.len();
                continue 'outer;
            }
        }
        return None;
    }
    if u::isupper(py::take(part, 1)) {
        Some(u::capitalize(&out))
    } else if py::clen(&out) >= 5 {
        Some(out)
    } else {
        None
    }
}

/// Extra whole-name surfaces of one person: surname-first order and initialised forms.
pub fn person_variants(name: &str) -> Vec<String> {
    let parts: Vec<String> = name_parts(name).into_iter().filter(|p| !is_initial(p)).collect();
    if parts.len() < 2 {
        return Vec::new();
    }
    let mut out = vec![parts.iter().rev().cloned().collect::<Vec<_>>().join(" ")];
    let (given, surname) = (&parts[0], &parts[parts.len() - 1]);
    let g0 = py::take(given, 1);
    if py::clen(surname) >= 4 {
        out.extend([format!("{g0}. {surname}"), format!("{surname} {g0}."), format!("{g0}.{surname}")]);
    }
    if parts.len() == 3 {
        out.extend([format!("{} {}", parts[0], parts[2]), format!("{} {}", parts[2], parts[0])]);
    }
    let mut respelled = Vec::new();
    let bparts = name_parts(name);
    for (i, p) in bparts.iter().enumerate() {
        if is_patronymic(p) {
            continue;
        }
        for alt in translit_variants(p) {
            let mut v = bparts.clone();
            v[i] = alt;
            respelled.push(v.join(" "));
        }
    }
    let cyr: Vec<String> = parts.iter().filter_map(|p| cyrillic_form(p)).collect();
    if cyr.len() == parts.len() && parts.len() >= 2 {
        let c0 = py::take(&cyr[0], 1).to_string();
        let cl = cyr[cyr.len() - 1].clone();
        respelled.extend([cyr.join(" "), cyr.iter().rev().cloned().collect::<Vec<_>>().join(" "),
                          format!("{c0}. {cl}"), format!("{cl} {c0}."), format!("{c0}.{cl}")]);
    }
    out.extend(respelled);
    let k = key(name);
    dedupe(&out).into_iter().filter(|v| key(v) != k).collect()
}

const NO_GENITIVE_S: [&str; 5] = ["s", "ß", "x", "z", "ce"];

/// Other ways German text writes a name the vault holds: without its umlauts, with them, and for a
/// person the genitive. Matching keys only.
pub fn spelling_variants(surface: &str, genitive: bool) -> Vec<String> {
    static UMLAUT_IN: OnceLock<Regex> = OnceLock::new();
    let s = py::strip(surface).to_string();
    if s.is_empty() {
        return Vec::new();
    }
    let mut forms: Vec<String> = vec![s.clone()];
    let add = |f: String, forms: &mut Vec<String>| {
        if !forms.contains(&f) {
            forms.push(f);
        }
    };
    if s.chars().any(|c| "äöüÄÖÜ".contains(c)) {
        let out: String = s.chars().map(|c| match c {
            'ä' => "ae".into(), 'ö' => "oe".into(), 'ü' => "ue".into(), 'Ä' => "Ae".into(), 'Ö' => "Oe".into(), 'Ü' => "Ue".into(),
            c => c.to_string(),
        }).collect();
        add(out, &mut forms);
    }
    let rx = re(&UMLAUT_IN, "([AaOoUu])([Ee])");
    if rx.is_match(&s) {
        let back = rx.replace_all(&s, |m: &regex::Captures| match &m[1] {
            "a" => "ä", "o" => "ö", "u" => "ü", "A" => "Ä", "O" => "Ö", _ => "Ü",
        }.to_string()).into_owned();
        add(back, &mut forms);
    }
    if genitive {
        for f in forms.clone() {
            let last = f.split_whitespace().last().unwrap_or("").to_string();
            let tail = py::last(&last, 1);
            if !tail.is_empty() && u::isalpha(tail) && !NO_GENITIVE_S.iter().any(|t| u::casefold(&last).ends_with(t)) {
                add(format!("{f}s"), &mut forms);
            }
        }
    }
    forms.sort();
    let k = key(&s);
    dedupe(&forms).into_iter().filter(|f| key(f) != k).collect()
}

/// A person's name with the surname first: "Brenner, Ada" and "Brenner Ada" for "Ada Brenner".
pub fn surname_first(surface: &str) -> Vec<String> {
    let parts: Vec<String> = name_parts(surface).into_iter().filter(|p| !is_honorific(p)).collect();
    if parts.len() < 2 {
        return Vec::new();
    }
    let last = &parts[parts.len() - 1];
    let mut givens = vec![parts[..parts.len() - 1].join(" ")];
    if givens[0] != parts[0] {
        givens.push(parts[0].clone());
    }
    let mut out = Vec::new();
    for given in givens {
        out.push(format!("{last}, {given}"));
        out.push(format!("{last} {given}"));
    }
    out
}

fn definition_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, concat!(r"(?i)\(\s*(?:(?:together|each|jointly|collectively)\s+)?",
                   r"(?:(?:hereinafter|hereafter|henceforth)\s+(?:(?:referred\s+to|called|known)\s+)?(?:as\s+)?)?",
                   r"(?:(?:nachfolgend|im\s+folgenden|im\s+weiteren)\s+(?:(?:kurz|auch)\s+)?)?",
                   r"(?:(?:the|a|an|der|die|das)\s+)?",
                   r#"["“„«‘']([^"”“„«»‘’'()\n]{1,40})["”“»’']\s*\)"#))
}

/// The short names `text` defines, as (term, character offset of the opening bracket).
pub fn defined_terms(text: &str) -> Vec<(String, usize)> {
    definition_re().captures_iter(text).filter_map(|m| {
        let term = py::strip(&m[1]).to_string();
        (!term.is_empty()).then(|| (term, py::clen(&text[..m.get(0).unwrap().start()])))
    }).collect()
}

/// Whether a defined `term` is a short form of `name`: its initials or words of it.
pub fn defines_alias(term: &str, name: &str) -> bool {
    static COMPACT: OnceLock<Regex> = OnceLock::new();
    let words: Vec<String> = name_parts(name).into_iter().filter(|p| p != "&" && !is_honorific(p)).collect();
    if term.is_empty() || words.is_empty() {
        return false;
    }
    let compact = re(&COMPACT, r"[\s.&]").replace_all(term, "").into_owned();
    if u::isalpha(&compact) && u::isupper(&compact) && (2..=6).contains(&py::clen(&compact)) {
        let significant: Vec<&String> = words.iter().filter(|w| !JOINERS.contains(&key(w).as_str()) && legal_form_index(w).is_none()).collect();
        let initials = |ws: &[&String]| u::upper(&ws.iter().map(|w| py::take(w, 1).to_string()).collect::<String>());
        let all: Vec<&String> = words.iter().collect();
        return [significant, all].iter().any(|ws| !ws.is_empty() && compact == initials(ws));
    }
    let parts: Vec<String> = name_parts(term).into_iter().filter(|p| !is_honorific(p)).collect();
    if parts.is_empty() || parts.iter().all(|p| legal_form_index(p).is_some() || JOINERS.contains(&key(p).as_str())) {
        return false;
    }
    let have: Vec<String> = words.iter().map(|w| key(w)).collect();
    let want: Vec<String> = parts.iter().map(|p| key(p)).collect();
    if want.len() > have.len() {
        return false;
    }
    (0..=have.len() - want.len()).any(|i| have[i..i + want.len()] == want[..])
}

/// Distinctive single parts of a person name, as (surface, role).
pub fn person_fragments(name: &str) -> Vec<(String, &'static str)> {
    let parts: Vec<String> = name_parts(name).into_iter()
        .filter(|p| !is_initial(p) && !is_honorific(p))
        .filter(|p| py::clen(p) >= 4 && u::isalpha(py::take(p, 1))).collect();
    let mut out = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        if is_patronymic(p) || is_generic_folded(&key(p)) {
            continue;
        }
        let role = if i == parts.len() - 1 { "surname" } else { "given" };
        out.push((p.clone(), role));
        for alt in translit_variants(p) {
            out.push((alt, role));
        }
        if let Some(cyr) = cyrillic_form(p) {
            out.push((cyr, role));
        }
    }
    out
}

// ── 5. the admission predicate for residual (NER) detections ──────────────────────────────────

/// `_TOKEN_SHAPE`: `^[A-Za-z]+_\d+$`.
pub fn token_shaped(v: &str) -> bool {
    let Some((head, digits)) = v.split_once('_') else { return false };
    !head.is_empty() && head.bytes().all(|b| b.is_ascii_alphabetic())
        && !digits.is_empty() && digits.chars().all(|c| u::is_decimal(c))
        || v.strip_suffix('\n').is_some_and(|w| w != v && token_shaped(w))
}

/// `TOKEN_IN_TEXT.search`: `(?<![0-9A-Za-z])[A-Za-z]+_\d+(?![0-9A-Za-z])`, its lookarounds done by hand;
/// the first match as a character span.
pub fn token_in_text(text: &str) -> Option<(usize, usize)> {
    token_in_text_all(text).into_iter().next()
}

/// Every match of `TOKEN_IN_TEXT` (`finditer`), as character spans.
pub fn token_in_text_all(text: &str) -> Vec<(usize, usize)> {
    let c: Vec<char> = text.chars().collect();
    let alnum = |ch: char| ch.is_ascii_alphanumeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if !c[i].is_ascii_alphabetic() || (i > 0 && alnum(c[i - 1])) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < c.len() && c[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j < c.len() && c[j] == '_' {
            let mut k = j + 1;
            while k < c.len() && u::is_decimal(c[k]) {
                k += 1;
            }
            // the longest digit run whose next character is not an ASCII letter or digit
            let end = (j + 2..=k).rev().find(|&e| e == c.len() || !alnum(c[e]));
            if let Some(e) = end {
                out.push((i, e));
                i = e;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn balanced(s: &str) -> bool {
    let mut stack = Vec::new();
    for ch in s.chars() {
        match ch {
            '(' => stack.push(')'),
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            ')' | ']' | '}' => {
                if stack.pop() != Some(ch) {
                    return false;
                }
            }
            _ => {}
        }
    }
    stack.is_empty()
}

/// A model may write an ASCII letter or digit inside a token as its lookalike: mapped back, 1:1.
pub fn deconfuse_char(c: char) -> char {
    match c {
        'А' => 'A', 'В' => 'B', 'Е' => 'E', 'К' => 'K', 'М' => 'M', 'Н' => 'H', 'О' => 'O', 'Р' => 'P',
        'С' => 'C', 'Т' => 'T', 'У' => 'Y', 'Х' => 'X', 'І' => 'I', 'Ј' => 'J', 'Ѕ' => 'S', 'Ԛ' => 'Q',
        'Ԝ' => 'W', 'Г' => 'F', 'Д' => 'D', 'Л' => 'L', 'Ф' => 'F',
        'а' => 'a', 'в' => 'b', 'е' => 'e', 'к' => 'k', 'м' => 'm', 'о' => 'o', 'р' => 'p', 'с' => 'c',
        'т' => 't', 'у' => 'y', 'х' => 'x', 'і' => 'i', 'ј' => 'j', 'ѕ' => 's',
        'Α' => 'A', 'Β' => 'B', 'Ε' => 'E', 'Ζ' => 'Z', 'Η' => 'H', 'Ι' => 'I', 'Κ' => 'K', 'Μ' => 'M',
        'Ν' => 'N', 'Ο' => 'O', 'Ρ' => 'P', 'Τ' => 'T', 'Υ' => 'Y', 'Χ' => 'X',
        'α' => 'a', 'ε' => 'e', 'ι' => 'i', 'κ' => 'k', 'ν' => 'v', 'ο' => 'o', 'ρ' => 'p', 'τ' => 't',
        'υ' => 'y', 'χ' => 'x',
        '＿' => '_',
        c @ '\u{ff10}'..='\u{ff19}' => char::from_u32(c as u32 - 0xff10 + '0' as u32).unwrap(),
        c @ '\u{ff21}'..='\u{ff3a}' => char::from_u32(c as u32 - 0xff21 + 'A' as u32).unwrap(),
        c @ '\u{ff41}'..='\u{ff5a}' => char::from_u32(c as u32 - 0xff41 + 'a' as u32).unwrap(),
        c => c,
    }
}

/// 1:1 map of ASCII-lookalike characters onto ASCII. Same length, same offsets.
pub fn deconfuse_ascii(text: &str) -> String {
    text.chars().map(deconfuse_char).collect()
}

/// The leading word of an organisation name when that word identifies the company alone.
pub fn distinctive_head(name: &str) -> Option<String> {
    let parts: Vec<String> = strip_legal_form(name).into_iter().filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    let head = py::strip_chars(&parts[0], ".,;:-").to_string();
    if py::clen(&head) < 4 || !u::isalpha(py::take(&head, 1)) || is_common_word(&head) {
        return None;
    }
    if !head.chars().all(|ch| u::is_alpha(ch) || "-'’".contains(ch)) {
        return None;
    }
    Some(head)
}

fn script_class(ch: char) -> Option<&'static str> {
    if !u::is_alpha(ch) {
        return None;
    }
    if ch.is_ascii() {
        Some("latin")
    } else if ('Ͱ'..='Ͽ').contains(&ch) || ('ἀ'..='῿').contains(&ch) {
        Some("greek")
    } else if ('Ѐ'..='ӿ').contains(&ch) {
        Some("cyrillic")
    } else {
        None
    }
}

/// The dominant alphabetic script of `text`: latin, cyrillic, greek or None.
pub fn script_of(text: &str) -> Option<&'static str> {
    let mut counts = [("latin", 0), ("cyrillic", 0), ("greek", 0)];
    for ch in text.chars() {
        if let Some(s) = script_class(ch) {
            counts.iter_mut().find(|(n, _)| *n == s).unwrap().1 += 1;
        }
    }
    // `max(counts, key=...)`: the first of the largest
    let mut top = counts[0];
    for c in &counts[1..] {
        if c.1 > top.1 {
            top = *c;
        }
    }
    (top.1 > 0).then_some(top.0)
}

/// True when one WORD draws its letters from more than one alphabet.
pub fn mixed_script(word: &str) -> bool {
    let mut seen: Vec<&str> = Vec::new();
    for ch in word.chars() {
        if let Some(s) = script_class(ch) {
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
    }
    seen.len() > 1
}

/// Should a residual PERSON span become a token of its own? (ok, reason).
pub fn admit_person(surface: &str) -> (bool, String) {
    let v = py::strip(surface);
    let no = |why: &str| (false, why.to_string());
    if v.is_empty() || py::clen(v) < 3 || py::clen(v) > 80 {
        return no("length");
    }
    if token_shaped(v) {
        return no("token-shaped");
    }
    if token_in_text(v).is_some() {
        return no("contains a token");
    }
    if !balanced(v) || v.chars().any(|c| "()[]{}/\\|<>@".contains(c)) {
        return no("malformed span");
    }
    if v.chars().any(u::is_digit) || v.contains('~') {
        return no("contains digits");
    }
    if surface.contains('\n') || surface.contains('\r') || surface.contains('\t') || surface.contains("  ") {
        return no("line break or column gap in span");
    }
    let parts: Vec<String> = name_parts(v).into_iter().filter(|p| !is_honorific(p)).collect();
    if parts.len() < 2 {
        return no("single word");
    }
    if parts.len() > 4 {
        return no("too many parts for a name");
    }
    let substantive: Vec<&String> = parts.iter().filter(|p| !is_initial(p)).collect();
    if !substantive.is_empty() && substantive.iter().all(|p| is_common_word(p)) {
        return no("every part is an ordinary word");
    }
    for p in &parts {
        let low = key(p);
        if is_generic_folded(&low) || is_structure_folded(&low) {
            return (false, format!("generic/structure word: {p}"));
        }
        if folds().role_stems.iter().any(|s| low.starts_with(s.as_str())) {
            return (false, format!("procedural role noun: {p}"));
        }
        if !u::isupper(py::take(p, 1)) && !is_initial(p) {
            return (false, format!("uncapitalised part: {p}"));
        }
        if !p.chars().all(|ch| u::is_alpha(ch) || "-'’.".contains(ch)) {
            return (false, format!("non-name characters: {p}"));
        }
        if mixed_script(p) {
            return (false, format!("mixed script within one word: {p}"));
        }
    }
    (true, "ok".into())
}

/// Role references that are already de-identified ("the husband") — never tokenised.
pub fn is_generic(value: &str) -> bool {
    let v = u::lower(py::strip(value));
    v.is_empty() || v.starts_with("the ") || v.starts_with("a ") || v.starts_with("an ") || v.starts_with("unknown")
        || py::clen(&v) < 2
}
