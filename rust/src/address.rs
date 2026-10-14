// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Street addresses as tokens, for text that crosses to a cloud model (as `address.py`): the street and
//! house number and the postal code become `ADDRESS_n`, derived from the scope's salt; the city and the
//! country stay, because they are context, not identity.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::odict::ODict;
use crate::pyre::{self, Regex, NONE};
use crate::{namefold as nf, py};

const DE_SUFFIX: &str = r"(?:straße|strasse|str\.|weg|platz|allee|ufer|damm|ring|gasse|chaussee|pfad|steig|markt|brücke|hof|kamp|stieg|twiete|kai|park|berg|feld|tor|graben|wall)";
const DE_NUMBER: &str = r"\d{1,4}(?:\s?[a-zA-Z](?![\wäöüß]))?(?:[-–/]\d{1,4}(?:[a-zA-Z](?![\wäöüß]))?)?(?!\d)";

fn patterns() -> &'static [(&'static str, Regex)] {
    static P: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    P.get_or_init(|| {
        let c = |p: String| Regex::with_flags(&p, NONE).unwrap_or_else(|e| panic!("{p}: {e}"));
        vec![
            ("de_street", c(format!(r"(?<![\wÄÖÜäöüß])[A-ZÄÖÜ][\wÄÖÜäöüß\-]*{DE_SUFFIX}(?:\s+|[-_]){DE_NUMBER}"))),
            ("de_street_words", c(format!(concat!(r"(?<![\wÄÖÜäöüß])(?:(?:Am|An der|An den|Auf dem|Auf der|Im|In der|In den|Zum|Zur|Hinter der|Unter den)\s+",
                                                  r"[A-ZÄÖÜ][\wÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][\wÄÖÜäöüß\-]+)?|Große[nrs]?\s+[A-ZÄÖÜ][\wÄÖÜäöüß\-]+)\s+{}"), DE_NUMBER))),
            ("en_street", c(concat!(r"(?<![\w])\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'’\-]+\s+){1,3}",
                                    r"(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|",
                                    r"Terrace|Crescent|Close|Way|Gardens|Row|Mews|Walk|Hill|Boulevard|Blvd\.?|Parade|Grove|Wharf)\b").to_string())),
            ("en_unit", c(r"(?<![\w])(?:Flat|Apartment|Apt\.?|Unit|Suite)\s+\d{1,4}[A-Za-z]?\b".to_string())),
            ("uk_postcode", c(r"(?<![\w])[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}(?![\w])".to_string())),
            ("postcode_city", c(concat!(r"(?<![\w\d.,§])(\d{5})(?=[ \t]+(?!(?:Euro|EUR|Dollar|USD|GBP|CHF|Pfund|Franken|Stück|Anteil|Aktie|",
                                        r"Tag|Monat|Jahr|Seite|Quadratmeter|Mal)\w*)[A-ZÄÖÜ][a-zäöüß]{2,})").to_string())),
            ("postcode_prefixed", c(r"(?<![\w])(?:A|AT|CH)-\d{4}(?![\d])".to_string())),
        ]
    })
}

/// Non-overlapping `(start, end, pattern)` character spans, longest first where two overlap.
pub fn find_addresses(text: &[char]) -> Vec<(usize, usize, &'static str)> {
    let mut spans = Vec::new();
    for (name, pat) in patterns() {
        for m in pat.finditer(text) {
            let (s, e) = if pat.groups > 0 { m.group(1).unwrap_or((usize::MAX, usize::MAX)) } else { (m.start, m.end) };
            spans.push((s, e, *name));
        }
    }
    spans.sort_by_key(|&(s, e, _)| (s, std::cmp::Reverse(e.wrapping_sub(s))));
    let mut out: Vec<(usize, usize, &'static str)> = Vec::new();
    for (s, e, name) in spans {
        if out.last().is_some_and(|last| s < last.1) {
            continue;
        }
        out.push((s, e, name));
    }
    out
}

pub struct AddressRedaction {
    pub text: String,
    /// token → address as written
    pub mapping: ODict<String>,
    /// (pattern, surface)
    pub found: Vec<(&'static str, String)>,
}

/// The fold one address is known by across the pieces of a request.
pub fn address_key(surface: &str) -> String {
    crate::unicode::casefold(&py::split_ws(surface).join(" "))
}

/// Replace every street address in `text` with a stable `ADDRESS_n` token. `derive(surface, taken)` is the
/// vault's derivation bound to the scope salt; `known` (folded surface → token) is shared by every piece
/// of one request.
pub fn tokenize_addresses(text: &str, derive: &dyn Fn(&str, &HashSet<String>) -> String, taken: &HashSet<String>,
                          known: &mut ODict<String>) -> AddressRedaction {
    let mut taken = taken.clone();
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut red = AddressRedaction { text: text.to_string(), mapping: ODict::new(), found: Vec::new() };
    let mut pos = 0;
    for (s, e, name) in find_addresses(&chars) {
        let surface: String = chars[s..e].iter().collect();
        let norm = address_key(&surface);
        let tok = match known.get(&norm) {
            Some(t) => t.clone(),
            None => {
                let t = derive(&surface, &taken);
                taken.insert(t.clone());
                known.insert(&norm, t.clone());
                t
            }
        };
        red.mapping.setdefault(&tok, surface.clone());
        red.found.push((name, surface));
        out.extend(&chars[pos..s]);
        out.push_str(&tok);
        pos = e;
    }
    out.extend(&chars[pos..]);
    red.text = out;
    red
}

/// "MARTIN KESSLER" → "MK", "Dr. Ada Brenner" → "AB"; None for a one-part name.
pub fn initials_of(name: &str) -> Option<String> {
    let parts: Vec<String> = nf::name_parts(name).into_iter()
        .filter(|p| crate::unicode::isalpha(py::take(p, 1)) && !nf::is_honorific(p) && !nf::is_initial(p)).collect();
    if !(2..=4).contains(&parts.len()) {
        return None;
    }
    Some(parts.iter().map(|p| crate::unicode::upper(py::take(p, 1))).collect())
}

/// Replace the initials of the scope's known people ("MK", "M.K.", "M. K.") with their token. Returns
/// the text and `{token: name}` for the tokens it used.
pub fn tokenize_initials(text: &str, people: &ODict<String>) -> (String, ODict<String>) {
    let mut used = ODict::new();
    let mut out = text.to_string();
    for (token, name) in people.iter() {
        let Some(ini) = initials_of(name) else { continue };
        let body: Vec<String> = ini.chars().map(|c| pyre::escape(&c.to_string())).collect();
        let pat = format!(r"(?<![\w.]){}\.?(?![\w])", body.join(r"\.?\s?"));
        let Ok(rx) = Regex::new(&pat) else { continue };
        let chars: Vec<char> = out.chars().collect();
        let (new, n) = rx.sub_with(&chars, 0, |_| token.to_string());
        if n > 0 {
            out = new;
            used.insert(token, name.clone());
        }
    }
    (out, used)
}
