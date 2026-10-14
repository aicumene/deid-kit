// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! References to norms in a document's text: where each is, and its canonical form ("§ 823 BGB",
//! "Art. 6 DSGVO", "8 U.S.C. § 1427", "Cap. 113 s. 105", "ГК РФ ст. 450") — `deidkit/agents/cite.py`,
//! itself the server's extraction (LegalTech `apps/api/app/knowledge/cite.py`, `cite_de.py`) without its
//! database. The same patterns, run by a regex engine with Python's semantics ([`crate::pyre`]); the
//! reasons for each are in the Python module's comments, kept there.
//!
//! Only what the review page uses is here: the spans and the jurisdiction of a citation.

use std::cmp::Ordering;
use std::sync::OnceLock;

use crate::pyre::{self, Regex};
use crate::unicode;

const PARAGRAPH_CODES: &[&str] = &[
    "BGB", "ZPO", "EGZPO", "FamFG", "GVG", "EGGVG", "ZVG", "EGZVG", "GBO", "GBV", "RPflG",
    "BeurkG", "AnfG", "IntFamRVG", "LPartG", "VersAusglG", "UhVorschG", "AUG", "SachenRBerG",
    "SchuldRAnpG", "ErbbauRG", "WEG", "HeizkostenV", "BetrKV", "WoBindG", "WoFG",
    "RVG", "GKG", "GNotKG", "JVEG", "JVKostG", "BRAO", "BORA", "BNotO", "StBerG", "WPO", "PAO",
    "RDG",
    "StGB", "EGStGB", "StPO", "EGStPO", "OWiG", "JGG", "StVollzG", "StrEG", "BtMG", "KCanG",
    "WaffG", "SchwarzArbG",
    "HGB", "EGHGB", "AktG", "GmbHG", "EGGmbHG", "GenG", "PartGG", "UmwG", "InsO", "EGInsO",
    "StaRUG", "PublG", "DrittelbG", "MitbestG", "BörsG", "DepotG", "ScheckG", "GüKG", "BinSchG",
    "UWG", "GWB", "UrhG", "VGG", "MarkenG", "PatG", "GebrMG", "DesignG",
    "BetrVG", "SprAuG", "BPersVG", "KSchG", "TVG", "ArbGG", "TzBfG", "ArbZG", "ArbSchG", "ASiG",
    "BUrlG", "MuSchG", "BEEG", "PflegeZG", "FPfZG", "AGG", "EntgTranspG", "EntgFG", "NachwG",
    "AÜG", "AEntG", "MiLoG", "BBiG", "BBG", "BeamtStG", "SGG",
    "VwGO", "VwVfG", "VwZG", "VwVG", "BVerfGG", "IFG", "UIG", "VIG", "ROG", "BauGB", "BauNVO",
    "BImSchG", "KrWG", "WHG", "BNatSchG", "UVPG", "UmweltHG", "GenTG", "EnWG",
    "AO", "EGAO", "FGO", "EStG", "KStG", "GewStG", "UStG", "UStDV", "ErbStG", "GrEStG", "GrStG",
    "BewG", "SolZG", "AStG", "InvStG", "UmwStG",
    "StVG", "StVO", "StVZO", "FeV", "FZV", "PflVG", "PBefG", "LuftVG", "HaftPflG",
    "VVG", "VAG", "KWG", "WpHG", "WpÜG", "ZAG", "KAGB", "GwG",
    "BDSG", "TDDDG", "TTDSG", "DDG", "TMG", "TKG",
    "GewO", "HandwO", "GastG", "ApoG", "BÄO", "PsychThG", "AMG", "MPDG", "IfSG", "TFG", "TPG",
    "LFGB", "ProdHaftG", "ProdSG", "TierSchG", "JuSchG", "KHG", "KHEntgG",
    "AufenthG", "AsylG", "StAG", "AZRG", "BeschV", "PStG", "PassG", "PAuswG", "AdVermiG",
    "AdWirkG", "FreizügG/EU",
    "ZPOEG", "HGBEG", "GVGEG", "ZVGEG", "StPOEG", "StGBEG", "InsOEG", "AOEG", "HwO",
];

const ARTICLE_CODES: &[&str] = &[
    "GG", "EGBGB", "BGBEG",
    "AEUV", "EUV", "GRCh", "EMRK", "CISG",
    "DSGVO", "DS-GVO",
    "EuGVVO", "EuGüVO", "EuPartVO", "EuUntVO", "EuErbVO", "EuInsVO", "EuVTVO", "EuMahnverfVO",
    "EuZVO", "EuBVO", "HUP", "HKÜ",
];

const CODE_ALIASES: &[(&str, &str)] = &[
    ("ZPOEG", "EGZPO"), ("BGBEG", "EGBGB"), ("HGBEG", "EGHGB"), ("GVGEG", "EGGVG"), ("ZVGEG", "EGZVG"),
    ("StPOEG", "EGStPO"), ("StGBEG", "EGStGB"), ("InsOEG", "EGInsO"), ("AOEG", "EGAO"), ("HwO", "HandwO"),
];

const SGB: &str = r"SGB\s+(?:XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)";
const EU_NAMED: &str = r"(?:Brüssel|Rom)[\s-]I{1,3}[ab]?-VO";
const RU_NAMED: &str = concat!(r"(?:ГГУ|ГТУ|ГПК\s+(?:ФРГ|Германии)|",
                               r"[Гг]ерманск\w{1,4}\s+[Гг]ражданск\w{1,4}\s+[Уу]ложени\w{1,2}|",
                               r"[Гг]ерманск\w{1,4}\s+[Тт]оргов\w{1,4}\s+[Уу]ложени\w{1,2})");
const INVISIBLE: [char; 6] = ['\u{200b}', '\u{200c}', '\u{200d}', '\u{2060}', '\u{feff}', '\u{ad}'];

/// `sorted(items, key=len, reverse=True)`: longest first (in characters), equal lengths in their order.
fn longest_first(items: &[&'static str]) -> Vec<&'static str> {
    let mut out = items.to_vec();
    out.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()));
    out
}

fn alternation(codes: &[&'static str], fragments: &[&str]) -> String {
    let mut parts: Vec<String> = fragments.iter().map(|f| f.to_string()).collect();
    parts.extend(longest_first(codes).iter().map(|c| pyre::escape(c)));
    parts.join("|")
}

fn normalize_code(raw: &str) -> String {
    static LEAD: OnceLock<Regex> = OnceLock::new();
    let squashed = raw.split_whitespace_py();
    let t: Vec<char> = squashed.chars().collect();
    let code = pyre::lazy(&LEAD, r"^(Brüssel|Rom)[\s-]", pyre::NONE).sub(r"\1 ", &t, 0).map(|r| r.0).unwrap_or(squashed);
    if code == "ГГУ" || code.contains("ражданск") {
        return "BGB".into();
    }
    if code == "ГТУ" || code.contains("оргов") {
        return "HGB".into();
    }
    if code.starts_with("ГПК") {
        return "ZPO".into();
    }
    CODE_ALIASES.iter().find(|(k, _)| *k == code).map_or(code.clone(), |(_, v)| v.to_string())
}

trait SplitPy {
    fn split_whitespace_py(&self) -> String;
}

impl SplitPy for str {
    /// `" ".join(s.split())`
    fn split_whitespace_py(&self) -> String {
        crate::py::split_ws(self).join(" ")
    }
}

const SUB_WORDS: &[&str] = &[
    "Absatz", "Abs", "Unterabsatz", "Unterabs", "UAbs", "Satz", "S", "Halbsatz", "Halbs", "Hs",
    "Teilsatz", "Teils", "Nummer", "Nr", "Ziffer", "Ziff", "Spiegelstrich", "Alt", "Var", "Fall",
    "ff", "f",
];
const LETTERED: &str = r"(?:Doppelbuchstabe|Doppelbuchst|Buchstabe|Buchst|sublit|lit)\.?\s*[a-z]{1,2}(?![a-z])\)?";
const SECTION: &str = r"\d+(?!\d)(?:[a-z](?![a-z]))?";
const SEP: &str = r"(?:\s*[,;]\s*|\s+(?:und|sowie|bis)\s+|\s*[-–—]\s*)";
const PARA_MARK: &str = r"§§?|Paragrap[fh]en|Paragrap[fh]|Par\.";
const ART_MARK: &str = r"Artt?\.?|Artikel";

fn token() -> String {
    format!(r"(?:{LETTERED}|a\.E\.|\(\d+[a-z]?\)|{SECTION}\.?|[IVX]{{1,4}}(?![IVX])|(?:{})\.?|[a-z]\))",
            longest_first(SUB_WORDS).join("|"))
}

fn list() -> String {
    let item = format!(r"{t}(?:\s*{t})*", t = token());
    format!(r"{item}(?:{SEP}(?:§§?\s*)?{item}){{0,60}}")
}

struct Patterns {
    para: Regex,
    art: Regex,
    egbgb: Regex,
    schedule: Regex,
    has_sub: Regex,
    marker: Regex,
    number: Regex,
    sep: Regex,
    ru_art_first: Regex,
    ru_code_first: Regex,
    cpr: Regex,
    mca_sec_first: Regex,
    mca_act_first: Regex,
    usc: Regex,
    cfr_us: Regex,
    cy_cap_first: Regex,
    cy_sec_first: Regex,
    non_digit: Regex,
    digit: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| {
        let re = |p: &str| Regex::new(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        let rei = |p: &str| Regex::with_flags(p, pyre::I).unwrap_or_else(|e| panic!("{p}: {e}"));
        let list = list();
        let token = token();
        let mca = r"(?:MCA\s*1973|Matrimonial Causes Act\s*1973)";
        let cy_cap = r"Cap\.?\s*113";
        let cy_suffix = r"[0-9]+[A-Za-zΑ-Ωα-ω]?";
        Patterns {
            para: re(&format!(r"({PARA_MARK})\s*({list})\s*\b({})\b(?!-\w)", alternation(PARAGRAPH_CODES, &[SGB, RU_NAMED]))),
            art: re(&format!(r"({ART_MARK})\s*({list})\s*\b({})\b(?!-\w)", alternation(ARTICLE_CODES, &[EU_NAMED]))),
            egbgb: re(&format!(r"(?:{ART_MARK})\s*(\d+[a-z]?)\s*§\s*({SECTION})(?:\s*{token})*\s*\b(EGBGB|BGBEG)\b")),
            schedule: re(r"\bNr\.?\s*(\d{4})\s*(VV\s*RVG|KV\s*GKG)\b"),
            has_sub: re(concat!(r"\b(?:Absatz|Abs|Unterabsatz|Unterabs|UAbs|Satz|Halbsatz|Halbs|Hs|",
                                r"Teilsatz|Teils|Nummer|Nr|Ziffer|Ziff|Spiegelstrich|Alt|Var|Fall|",
                                r"Buchstabe|Buchst|sublit|lit)\b")),
            marker: re(&format!(r"^\s*(?:{PARA_MARK}|{ART_MARK})\s*")),
            number: re(&format!(r"^\s*({SECTION})")),
            sep: re(SEP),
            ru_art_first: re(r"ст\.?\s*(\d+)\s+(СК РФ|ГК РФ)"),
            ru_code_first: re(r"(СК РФ|ГК РФ)\s+ст\.?\s*(\d+)"),
            cpr: re(r"\bCPR\s+\d+(?:\.\d+)?\b"),
            mca_sec_first: rei(&format!(r"(?:section|s)\.?\s*(\d+)(?:\(\d+[A-Za-z]?\))?\s*(?:of\s+the\s+)?{mca}")),
            mca_act_first: rei(&format!(r"{mca}[,\s]+(?:section|s)\.?\s*(\d+)")),
            usc: re(r"\b(\d+)\s*U\.?\s*S\.?\s*C\.?\s*(?:§+\s*|sec(?:tion)?s?\.?\s*)?(\d+)"),
            cfr_us: rei(r"\b(\d+)\s*C\.?\s*F\.?\s*R\.?\s*(?:§+\s*|part\s*|sec(?:tion)?s?\.?\s*)?(\d+\.\d+)"),
            cy_cap_first: rei(&format!(r"{cy_cap}[,\s]+(?:section|sec\.?|s)\.?\s*({cy_suffix})")),
            cy_sec_first: rei(&format!(r"(?:section|sec\.?|s)\.?\s*({cy_suffix})\s+of\s+(?:the\s+)?{cy_cap}")),
            non_digit: re(r"\D"),
            digit: re(r"\d"),
        }
    })
}

fn group(t: &[char], m: &pyre::Match, i: usize) -> String {
    m.group(i).map(|s| pyre::text(t, s)).unwrap_or_default()
}

/// `int(a) <= int(b)` of two strings of decimal digits (of any script), however long.
fn le_int(a: &str, b: &str) -> bool {
    let value = |s: &str| -> Vec<i64> {
        let d: Vec<i64> = s.chars().filter_map(crate::py::digit).collect();
        let lead = d.iter().take_while(|x| **x == 0).count();
        d[lead..].to_vec()
    };
    let (x, y) = (value(a), value(b));
    match x.len().cmp(&y.len()) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => x <= y,
    }
}

/// The section numbers a cluster cites (cite.py `_sections`: the comma rule).
fn sections(marker: &str, body: &str) -> Vec<String> {
    let p = patterns();
    let bt: Vec<char> = body.chars().collect();
    let mut parts: Vec<String> = p.sep.split(&bt, 0).into_iter().map(Option::unwrap_or_default).collect();
    let plural = marker.starts_with("§§") || marker.starts_with("Artt");
    let has_marker = |s: &str| p.marker.match_at(&s.chars().collect::<Vec<_>>(), 0).is_some();
    let has_sub = |s: &str| p.has_sub.is_match(s);
    let repeats_marker = parts.iter().skip(1).any(|x| has_marker(x));
    if !plural && !repeats_marker && has_sub(body) {
        parts.truncate(1);
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen_sub = false;
    for (i, part) in parts.iter().enumerate() {
        let pt: Vec<char> = part.chars().collect();
        let stripped = p.marker.sub_with(&pt, 1, |_| String::new()).0;
        let st: Vec<char> = stripped.chars().collect();
        let Some(m) = p.number.match_at(&st, 0) else {
            seen_sub = seen_sub || has_sub(part);
            continue;
        };
        let number = group(&st, &m, 1);
        let repeats_own_marker = has_marker(part);
        let after: String = st[m.end..].iter().collect();
        let carries_own_sub = has_sub(&after);
        if i > 0 && seen_sub && !repeats_own_marker && !carries_own_sub && !out.is_empty() {
            let digits = |s: &str| p.non_digit.replace_all(s, |_, _| String::new());
            if le_int(&digits(&number), &digits(out.last().unwrap())) {
                seen_sub = true;
                continue;
            }
        }
        let rest = crate::py::lstrip(&after);
        if p.digit.match_at(&rest.chars().collect::<Vec<_>>(), 0).is_some() {
            continue;
        }
        out.push(number);
        seen_sub = seen_sub || has_sub(part);
    }
    out
}

/// NFKC by cluster (a base and its combining marks), the invisible characters dropped, with for each
/// character of the result where it came from.
fn normalized_with_map(text: &str) -> (Vec<char>, Vec<usize>) {
    let t: Vec<char> = text.chars().collect();
    let (mut out, mut back) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < t.len() {
        let mut j = i + 1;
        while j < t.len() && unicode::combining(t[j]) != 0 {
            j += 1;
        }
        let piece: String = t[i..j].iter().collect();
        for c in unicode::normalize("NFKC", &piece).chars().filter(|c| !INVISIBLE.contains(c)) {
            out.push(c);
            back.push(i);
        }
        i = j;
    }
    (out, back)
}

/// German citations with their places in the original text, sorted by where they start.
fn de_spans(text: &str) -> Vec<(usize, usize, String)> {
    let p = patterns();
    let (norm, back) = normalized_with_map(text);
    let span = |m: &pyre::Match| -> (usize, usize) {
        let (a, b) = (m.start, m.end);
        if back.is_empty() || a >= back.len() {
            return (a, b);
        }
        let last = b.min(back.len()) as isize - 1;
        let last = if last < 0 { back.len() - 1 } else { last as usize };
        (back[a], back[last] + 1)
    };
    let mut found: Vec<(usize, usize, String)> = Vec::new();
    for m in p.egbgb.finditer(&norm) {
        let (a, b) = span(&m);
        found.push((a, b, format!("Art. {} § {} EGBGB", group(&norm, &m, 1), group(&norm, &m, 2))));
    }
    for m in p.schedule.finditer(&norm) {
        let (a, b) = span(&m);
        found.push((a, b, format!("Nr. {} {}", group(&norm, &m, 1), group(&norm, &m, 2).split_whitespace_py())));
    }
    for (pattern, art) in [(&p.para, false), (&p.art, true)] {
        for m in pattern.finditer(&norm) {
            let code = normalize_code(&group(&norm, &m, 3));
            let (a, b) = span(&m);
            for section in sections(&group(&norm, &m, 1), &group(&norm, &m, 2)) {
                found.push((a, b, if art { format!("Art. {section} {code}") } else { format!("§ {section} {code}") }));
            }
        }
    }
    found.sort_by_key(|f| f.0);
    found
}

/// One mention of a norm: its place in the text (characters) and its canonical form.
#[derive(Clone, Debug, PartialEq)]
pub struct CitationSpan {
    pub start: usize,
    pub end: usize,
    pub citation: String,
}

/// Every mention, family by family as the server lists them (German last); duplicates kept.
pub fn extract_citation_spans(text: &str) -> Vec<CitationSpan> {
    let p = patterns();
    let t: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut add = |m: &pyre::Match, citation: String| found.push(CitationSpan { start: m.start, end: m.end, citation });
    for m in p.ru_art_first.finditer(&t) {
        add(&m, format!("{} ст. {}", group(&t, &m, 2), group(&t, &m, 1)));
    }
    for m in p.ru_code_first.finditer(&t) {
        add(&m, format!("{} ст. {}", group(&t, &m, 1), group(&t, &m, 2)));
    }
    for m in p.cpr.finditer(&t) {
        add(&m, group(&t, &m, 0));
    }
    for m in p.mca_sec_first.finditer(&t) {
        add(&m, format!("MCA 1973 s. {}", group(&t, &m, 1)));
    }
    for m in p.mca_act_first.finditer(&t) {
        add(&m, format!("MCA 1973 s. {}", group(&t, &m, 1)));
    }
    for m in p.usc.finditer(&t) {
        add(&m, format!("{} U.S.C. § {}", group(&t, &m, 1), group(&t, &m, 2)));
    }
    for m in p.cfr_us.finditer(&t) {
        add(&m, format!("{} CFR § {}", group(&t, &m, 1), group(&t, &m, 2)));
    }
    for m in p.cy_cap_first.finditer(&t) {
        add(&m, format!("Cap. 113 s. {}", group(&t, &m, 1)));
    }
    for m in p.cy_sec_first.finditer(&t) {
        add(&m, format!("Cap. 113 s. {}", group(&t, &m, 1)));
    }
    for (start, end, citation) in de_spans(text) {
        found.push(CitationSpan { start, end, citation });
    }
    found
}

/// The jurisdiction a canonical citation belongs to, by its form.
pub fn jurisdiction_of(c: &str) -> &'static str {
    if c.contains(" U.S.C. ") || c.contains(" CFR ") {
        "US"
    } else if c.starts_with("CPR ") || c.starts_with("MCA 1973") {
        "GB"
    } else if c.starts_with("Cap. 113") {
        "CY"
    } else if c.starts_with("СК РФ") || c.starts_with("ГК РФ") {
        "RU"
    } else if ["§ ", "§§ ", "Art. ", "Nr. "].iter().any(|p| c.starts_with(p)) {
        "DE"
    } else {
        ""
    }
}
