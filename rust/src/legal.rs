// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The residual detector for legal documents (31.10.2026): what the pattern detector finds, plus the parties, people
//! and identifiers a legal text announces by its own conventions — a company by its legal form, a person by a title,
//! a signature line, a party clause or a role beside the name, an identifier by its label and format. A matter
//! without name files had only e-mails, IBANs, cards and phones hidden; its people and companies reached the model.
//!
//! Each rule reports a span and a kind, and the vault decides as for any detector: a person still passes its
//! admission gate (two name parts, no ordinary words, no digits), so a cue that catches a phrase is refused there.
//! Places, dates, amounts, courts and statutes are not reported: the model needs them. Links are read as well as
//! the text: `Tamsin%20Okafor` in a URL is found as `Tamsin Okafor` (the fold matches the encoded form).

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::detect::PiiDetector;
use crate::namefold as nf;
use crate::patterns::RegexDetector;

/// What the pattern detector finds, and the legal conventions above.
pub struct LegalDetector {
    patterns: RegexDetector,
}

impl Default for LegalDetector {
    fn default() -> Self {
        LegalDetector { patterns: RegexDetector::default() }
    }
}

/// A word of a name: a capital, then letters, marks, apostrophes or hyphens.
const W: &str = r"\p{Lu}[\p{L}\p{M}'’\-]*";
/// A word of a name that is not all capitals (a role noun's name, after a comma, is in title case).
const TW: &str = r"\p{Lu}\p{Ll}[\p{L}\p{M}'’\-]*";
/// A name's particle between its words ("Pieter van Dijk", "Sanne de Groot").
const P: &str = r"(?:(?:van|von|de|der|den|da|di|du|le|la|ten|ter|bin|al|del|dos|das|y)[ \t]+)?";
const HON: &str = r"(?:Mr|Mrs|Ms|Miss|Mx|MR|MRS|Dr|DR|Prof|Sir|Dame|Lord|Lady|Judge|Captain|Capt|Herr|Herrn|Frau|Mme|Mlle|M\.|Monsieur|Madame|Señor|Señora|Sr|Sra|Dott|Avv)";
const ROLE: &str = r"(?i:plaintiff|respondent|applicant|appellant|petitioner|debtor|creditor|individual|colega|collègue|kollege|kollegin|avocat|abogado|rechtsanwalt|rechtsanwältin|colleague|manager|director|associate|partner|accountant|adviser|advisor|counsel|surveyor|secretary|chair|chairman|chairwoman|client|claimant|defendant|witness|employee|employer|wife|husband|son|daughter|mother|father|brother|sister|officer|engineer|agent|representative|solicitor|barrister|attorney|clerk|broker|master|supervisor|consultant|contractor|tenant|landlord|guarantor|trustee|executor|beneficiary|nominee|founder|shareholder|auditor|notary|paralegal|assistant|deputy)";
/// Legal forms: a name before one is a company. The short ones (two letters) are ambiguous in capitals.
const FORMS: &str = r"(?:Pty\.?[ \t]Ltd\.?|GmbH[ \t]&[ \t]Co\.[ \t]KG|S\.à[ \t]r\.l\.|S\.A\.S\.|S\.A\.|S\.r\.l\.|S\.p\.A\.|B\.V\.|N\.V\.|L\.L\.C\.|L\.L\.P\.|sp\.[ \t]z[ \t]o\.o\.|S\.L\.|s\.r\.o\.|a\.s\.|d\.o\.o\.|Limited|LIMITED|Ltd\.?|LTD\.?|PLC|plc|LLLP|LLP|LLC|Inc\.?|INC\.?|Incorporated|Corp\.?|Corporation|gGmbH|GmbH|KGaA|OHG|SARL|SAS|SpA|Srl|Oyj|Oy|ASA|Abp|A/S|ApS|OÜ|Lda\.?|Kft\.?|U\.A\.|W\.A\.|B\.A\.|Solicitors|Rechtsanwälte|AG|KG|SE|SA|AS|AB|BV|NV|LP)";
const SHORT_FORMS: [&str; 10] = ["AG", "KG", "SE", "SA", "AS", "AB", "BV", "NV", "LP", "Oy"];
/// Words a capitalised run picks up before a company's name: sentence starts, roles, function words.
const LEAD: [&str; 52] = ["The", "THE", "A", "An", "And", "AND", "Between", "BETWEEN", "With", "For", "From", "To", "By", "Our",
    "Your", "Their", "Its", "His", "Her", "Messrs", "Dear", "Re", "Signed", "Owners", "Charterers", "Brokers", "Seller", "Buyer",
    "Client", "Landlord", "Tenant", "Employer", "Employee", "Claimant", "Defendant", "Of", "In", "At", "On", "Der", "Die", "Das",
    "Zwischen", "Und", "Mit", "Von", "Le", "La", "Les", "Entre", "Et", "El"];
const RU_LEAD: [&str; 8] = ["Уважаемый", "Уважаемая", "Уважаемые", "Господин", "Госпожа", "Генеральному", "Директору", "Адвокат"];

/// The words defined terms are made of ("Order Form", "Supplier Personnel", "Pension Contributions"): a contextual
/// person needs one word that is none of them. Real contracts showed why (gov-msc-schedules: 31 such "people").
const DEFINED: &[&str] = &["party", "parties", "form", "order", "terms", "term", "claim", "claims", "agreement", "supplier",
    "authority", "representative", "personnel", "data", "laws", "law", "insurance", "costs", "cost", "plan", "planning", "employees",
    "employee", "requirements", "requirement", "addendum", "contribution", "contributions", "rate", "date", "year", "transfer",
    "security", "adviser", "advisor", "company", "system", "management", "information", "risk", "appetite", "behaviours",
    "behaviors", "process", "project", "specific", "expected", "eligible", "incurred", "approved", "replacement", "monitored",
    "relevant", "initial", "previous", "key", "independent", "pollution", "liability", "pension", "cash", "flows", "flow", "word",
    "annex", "schedule", "clause", "version", "standard", "covered", "disclosing", "receiving", "effective", "service", "services",
    "fees", "fee", "customer", "provider", "vendor", "licensee", "licensor", "buyer", "seller", "purchaser", "landlord", "tenant",
    "lender", "borrower", "guarantor", "contractor", "sub-contractor", "sub-contractors", "subcontractor", "consultant", "client",
    "user", "users", "document", "documents", "notice", "notices", "deliverables", "milestone", "milestones", "statement", "work",
    "works", "material", "materials", "confidential", "personal", "protection", "processing", "processor", "controller", "subject",
    "breach", "default", "event", "events", "change", "changes", "control", "period", "price", "payment", "payments", "charges",
    "charge", "invoice", "account", "accounts", "group", "member", "members", "parent", "subsidiary", "affiliate", "affiliates",
    "intellectual", "property", "rights", "right", "product", "products", "software", "licence", "license", "site", "premises",
    "goods", "equipment", "assets", "asset", "resolution", "dispute", "procedure", "policy", "policies", "regulations", "act",
    "code", "directive", "guidance", "standards", "framework", "contract", "contracts", "call", "off", "lot", "tender", "award",
    "crown", "government", "department", "public", "sector", "body", "office", "officer", "director", "manager", "chief",
    "executive", "board", "committee", "council", "court", "tribunal", "panel", "expert", "auditor", "trustee", "trustees",
    "scheme", "fund", "funds", "employer", "staff", "team", "unit", "appendix", "exhibit", "part", "section", "recital",
    "definitions", "general", "special", "additional", "other", "new", "existing", "former", "current", "final", "interim",
    "principal", "senior", "deputy", "assistant", "holding", "holdings", "trading", "operating", "commercially", "sensitive",
    "objective", "objectives", "outcome", "outcomes", "service", "level", "levels", "credits", "credit", "start", "end", "plaintiff",
    "defendant", "claimant", "respondent", "witness", "statement", "truth",
    // places are never people: "United Kingdom", "European Union", "Russian Federation"
    "united", "kingdom", "states", "union", "republic", "federation", "emirates", "nations", "european", "islands"];

/// Could `w` be part of a person's name: not contract vocabulary, not a noun by its ending, no short acronym or numeral.
fn name_like(w: &str) -> bool {
    let low = w.to_lowercase();
    let low = low.strip_suffix("'s").or_else(|| low.strip_suffix("’s")).unwrap_or(&low).trim_end_matches(['’', '\'']).to_string();
    let letters = w.chars().filter(|c| c.is_alphabetic()).count();
    if letters < 2 || (letters <= 4 && !w.chars().any(char::is_lowercase)) {
        return false;
    }
    if DEFINED.contains(&low.as_str()) || nf::is_common_word(w) {
        return false;
    }
    const NOUNS: [&str; 22] = ["tion", "sion", "ment", "ments", "ance", "ence", "ity", "ness", "ship", "ure", "ies", "ings", "ers", "ors",
                               "able", "ible", "ive", "ory", "ary", "ism", "ware", "ture"];
    !(letters > 6 && NOUNS.iter().any(|s| low.ends_with(s)))
}

/// Inside a case's name ("Data Protection Commissioner v Facebook Ireland Ltd", "… / Facebook Ireland Ltd y …"): the
/// parties of a cited judgment are part of the law the model reasons with, and stay.
fn cited(text: &str, start: usize, end: usize) -> bool {
    const SEP: [&str; 15] = [" v ", " v. ", " vs ", " vs. ", " / ", "/", " c. ", " contre ", " κατά ", " przeciwko ", " gegen ", " contro ",
                             " tegen ", " contra ", " против "];
    let before = &text[text.floor_char_boundary(start.saturating_sub(160))..start];
    let after = &text[end..text.ceil_char_boundary((end + 12).min(text.len()))];
    let next_to = SEP.iter().any(|s| before.ends_with(s)) || SEP.iter().any(|s| after.starts_with(s))
        || [" y ", " and ", " und ", " et ", " i ", " και ", " e ", " en ", " и "].iter().any(|s| after.starts_with(s))
            // a bare "/" is in every case number ("C-311/18"): the spaced separators only
            && SEP.iter().filter(|s| **s != "/").any(|s| before.contains(s));
    static CASE: OnceLock<Regex> = OnceLock::new();
    let case = CASE.get_or_init(|| Regex::new(r"(?i:\bcase\b|asunto|affaire|rechtssache|causa|zaak|sprawie|C-\d+/\d+|\[\d{4}\])").unwrap());
    next_to && case.is_match(before)
}

macro_rules! rx {
    ($name:ident, $src:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new(&$src).expect(stringify!($name)))
        }
    };
}

rx!(org_form, format!(r"(?:{W}|\d+)(?:[ \t]+(?:{W}|&|and|of|de|du|des|van|von|der|den|di|da|y|und|et|\d+)){{0,6}},?[ \t]+{FORMS}(?:[^\p{{L}}\p{{N}}]|$)"));
rx!(org_ru, r"\b(?:ООО|АО|ПАО|ЗАО|ОАО|НАО|АНО|НКО|ТОО|ЧУП)[ \t]*[«“\x22]([^»”\x22\n]{2,80})[»”\x22]");
rx!(honorific, format!(r"\b{HON}\.?(?:[ \t]+(?:Justice|Dr\.?|Prof\.?))?[ \t]+({W}(?:[ \t]+{W}){{0,2}})"));
rx!(role_before, format!(r"\b{ROLE}s?[,:]?[ \t]+(?:{HON}\.?[ \t]+)?({TW}(?:[ \t]+{P}{W}){{1,2}})"));
rx!(role_after, format!(r"({TW}(?:[ \t]+{P}{W}){{1,2}}),[ \t]+(?:a|an|the|our|its|their|his|her)[ \t]+(?:[\p{{L}}\-]+[ \t]+){{0,2}}{ROLE}\b"));
rx!(cue_name, format!(r"(?:\bSigned[ \t]+by|\bSigned:|\bBy:|\bName:|\bAttention:|\bAttn:|\bc/o|\bI,|(?i:statement|affidavit|declaration)[ \t]+(?i:of))[ \t]+(?:the[ \t]+)?({W}(?:[ \t]+{P}{W}){{1,3}})"));
rx!(esquire, format!(r"({W}(?:[ \t]+{W}){{1,3}}),?[ \t]+Esq\.?"));
rx!(numbered_party, format!(r"(?m)^\(\d+\)[ \t]+({W}(?:[ \t]+{P}{W}){{1,3}})(?:[ \t]+of\b|,)"));
rx!(mail_header, format!(r"(?m)^(?:From|To|Cc|Bcc):[ \t]*({W}(?:[ \t]+{W}){{1,3}})[ \t]*(?:<|$)"));
rx!(valediction, format!(r"(?m)^(?:Yours[ \t]+(?:sincerely|faithfully|truly)|Very[ \t]+truly[ \t]+yours|Kind[ \t]+regards|Best[ \t]+regards|Regards|Mit[ \t]+freundlichen[ \t]+Grüßen),?[ \t]*\n(?:[ \t]*\n)*[ \t]*({W}(?:[ \t]+{W}){{1,3}})[ \t]*$"));
rx!(attendance, r"(?m)^(?:Present|In attendance|Attendees|Apologies|Attending)[ \t]*:[ \t]*(.+)$");
rx!(list_name, format!(r"^\s*(?:{HON}\.?[ \t]+)?({W}(?:[ \t]+{W}){{1,3}})\s*(?:\(.*\))?\s*$"));
rx!(ru_fio, r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?[ \t]+[А-ЯЁ][а-яё]+[ \t]+[А-ЯЁ][а-яё]+(?:вич|вн|ичн|ьич)[а-яё]*");
rx!(ru_name_patronymic, r"[А-ЯЁ][а-яё]+[ \t]+[А-ЯЁ][а-яё]+(?:вич|вн|ичн|ьич)[а-яё]*");
rx!(ru_initials, r"[А-ЯЁ][а-яё]+[ \t]+[А-ЯЁ]\.[ \t]?[А-ЯЁ]\.|[А-ЯЁ]\.[ \t]?[А-ЯЁ]\.[ \t]?[А-ЯЁ][а-яё]+");
rx!(query_value, r#"[?&](?:name|client|customer|party|person|q|query|search|company)=([^&#\s"']+)"#);
rx!(vessel, r#"\b(?:MV|M/V|M\.V\.|MT|M/T|mv)[ \t]+["“']?(\p{Lu}[\p{L}\-]+(?:[ \t]+\p{Lu}[\p{L}\-]+){0,2})"#);
rx!(us_phone, r"\(\d{3}\)[ \t]?\d{3}-\d{4}");
rx!(company_number, r"(?i:company[ \t]+(?:number|no\.?|registration[ \t]+number)|registered[ \t]+(?:number|no\.?)|companies[ \t]+house[ \t]+number)[ \t]*:?[ \t]*(?:is[ \t]+)?([A-Z]{2}\d{6}|\d{8})\b");
rx!(vat_labelled, r"(?:\bVAT(?:[ \t]+(?:registration|reg\.?))?(?:[ \t]+(?:number|no\.?))?|USt-?IdNr\.?|USt-ID|Umsatzsteuer-Identifikationsnummer|N°[ \t]TVA|BTW-nummer|Partita[ \t]IVA)[ \t]*(?:is[ \t]+|des[ \t]+\p{L}+:[ \t]*)?:?[ \t]*((?:[A-Z]{2}[ \t]?)?\d[\d \t]{6,13}\d)");
rx!(vat_format, r"\b(?:DE\d{9}|ATU\d{8}|FR[0-9A-Z]{2}\d{9}|NL\d{9}B\d{2}|BE0?\d{9}|IT\d{11}|ES[A-Z0-9]\d{7}[A-Z0-9]|GB\d{9}(?:\d{3})?|IE\d{7}[A-W][A-I]?|PL\d{10}|SE\d{12}|DK\d{8}|FI\d{8}|PT\d{9}|CZ\d{8,10}|LU\d{8})\b");
rx!(register_de, r"\bHR[AB][ \t]?\d{3,6}(?:[ \t]?[A-Z]{1,2}\b)?");
rx!(imo, r"\bIMO(?:[ \t]+(?:No\.?|number))?[ \t]*:?[ \t]*(\d{7})\b");
rx!(ni_number, r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z][ \t]?\d{2}[ \t]?\d{2}[ \t]?\d{2}[ \t]?[A-D]\b");
rx!(ssn, r"\b(\d{3})-(\d{2})-(\d{4})\b");
rx!(ein, r"(?i:\bEIN\b|employer[ \t]+identification[ \t]+number|\btax[ \t]+id(?:entification)?(?:[ \t]+number)?)\D{0,12}(\d{2}-\d{7})\b");
rx!(ru_ids, r"(?:ИНН|ОГРНИП|ОГРН|КПП|СНИЛС)[ \t]*:?[ \t]*(\d[\d \-]{7,16}\d)");
rx!(ru_passport, r"паспорт[а-я]*[ \t]+(?:серии[ \t]+)?(\d{2}[ \t]?\d{2})[ \t]+(?:№[ \t]*)?(\d{6})");
rx!(passport, r"(?i:passport)(?:[ \t]+(?:number|no\.?))?[ \t]*:?[ \t]*([A-Z0-9]{6,9})\b");
rx!(case_number, r"(?:\bClaim[ \t]+No\.?|\bCase[ \t]+No\.?|\bCase[ \t]+Number|\bClaim[ \t]+Number|\bDocket[ \t]+No\.?|\bIndex[ \t]+No\.?|\bAz\.|\bAktenzeichen:?|\bдел[оау][ \t]+№)[ \t]*:?[ \t]*([A-ZА-Я0-9][A-Za-zА-Яа-я0-9/\-\.:]{3,30}[A-Za-zА-Яа-я0-9])");
rx!(register_label, r"(?i:\bKvK\b|Kamer[ \t]+van[ \t]+Koophandel|Chamber[ \t]+of[ \t]+Commerce|\bRCS\b|\bSIREN\b|\bSIRET\b|registro[ \t]+mercantil|\bCIF\b|\bNIF\b|\bDNI\b|\bNIE\b)[^\n.;]{0,40}?(\d[\d \.]{4,16}\d|[A-Z]\d{7,8}[A-Z]?|\d{8}[A-Z])\b");
rx!(bar_number, r"(?i:State[ \t]+Bar[ \t]+No\.?|Bar[ \t]+No\.?|Bar[ \t]+Number|\bSBN\b)[ \t]*:?[ \t]*(\d{4,8})\b");
rx!(ru_account, r"(?:р/с|р/сч|к/с|л/с|расч[её]тный[ \t]+сч[её]т)[ \t]*:?[ \t]*№?[ \t]*(\d{20})\b");

/// The name words of a span, without the words a run picks up before a company's or a person's name.
fn trim_lead(span: &str) -> String {
    let mut words: Vec<&str> = span.split([' ', '\t']).filter(|w| !w.is_empty()).collect();
    while words.len() > 2 && (LEAD.contains(&words[0]) || RU_LEAD.contains(&words[0])) {
        words.remove(0);
    }
    words.join(" ")
}

/// A space written for a link: `%20`, and `+` between letters.
fn url_spaces(text: &str) -> Option<String> {
    if !text.contains("%20") && !text.contains('+') {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let letter = |j: usize| chars.get(j).is_some_and(|c| c.is_alphanumeric() || *c == '.');
        if chars[i] == '%' && chars.get(i + 1) == Some(&'2') && chars.get(i + 2) == Some(&'0') && i > 0 && letter(i - 1) && letter(i + 3) {
            out.push(' ');
            i += 3;
        } else if chars[i] == '+' && i > 0 && chars[i - 1].is_alphabetic() && chars.get(i + 1).is_some_and(|c| c.is_alphabetic()) {
            out.push(' ');
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    (out != text).then_some(out)
}

/// An IMO number's check digit: the first six digits weighted 7 to 2, the sum's last digit.
fn imo_ok(n: &str) -> bool {
    let d: Vec<u32> = n.chars().filter_map(|c| c.to_digit(10)).collect();
    d.len() == 7 && d[..6].iter().zip((2..=7).rev()).map(|(a, w)| a * w).sum::<u32>() % 10 == d[6]
}

/// A US Social Security number as issued: no area 000, 666 or 900+, no group 00, no serial 0000.
fn ssn_ok(area: &str, group: &str, serial: &str) -> bool {
    let a: u32 = area.parse().unwrap_or(0);
    a != 0 && a != 666 && a < 900 && group != "00" && serial != "0000"
}

/// The UK National Insurance prefixes never issued.
fn ni_ok(v: &str) -> bool {
    let p: String = v.chars().filter(|c| c.is_ascii_alphabetic()).take(2).collect();
    !["BG", "GB", "NK", "KN", "TN", "NT", "ZZ"].contains(&p.as_str())
}

/// The companies, people and identifiers of one text (`text` as given, or as its links decode), each with the rule
/// that found it.
fn legal(text: &str, add: &mut dyn FnMut(&'static str, String, &str)) {
    for m in org_form().find_iter(text) {
        let raw = m.as_str().trim_end_matches(|c: char| !c.is_alphanumeric() && c != '.');
        // a full stop is the form's own only in a dotted form ("S.A.", "B.V."); after "Ltd" or "Limited" it ends the sentence
        let raw = match raw.strip_suffix('.') {
            Some(r) if !r.rsplit([' ', '\t']).next().unwrap_or("").contains('.') => r,
            _ => raw,
        };
        if cited(text, m.start(), m.start() + raw.len()) {
            continue;
        }
        let mut span = trim_lead(raw);
        // a heading's capitals run into a title-case name ("AS AMENDED and Solberg Wind AS"): the name starts at the first
        // title-case word
        let caps = |w: &str| w.chars().filter(|c| c.is_alphabetic()).count() > 1 && !w.chars().any(char::is_lowercase);
        let all: Vec<&str> = span.split(' ').collect();
        if all.first().is_some_and(|w| caps(w)) {
            if let Some(at) = all[..all.len() - 1].iter().position(|w| w.chars().next().is_some_and(char::is_uppercase) && !caps(w)) {
                span = all[at..].join(" ");
            }
        }
        let words: Vec<&str> = span.split(' ').collect();
        let (form, name) = words.split_last().map(|(f, n)| (*f, n.to_vec())).unwrap_or(("", Vec::new()));
        if name.is_empty() {
            continue;
        }
        // "SUCH AS", "SHARE PURCHASE AGREEMENT AS": a two-letter form needs a name that is not ordinary words
        if SHORT_FORMS.contains(&form.trim_end_matches('.')) && name.iter().all(|w| nf::is_common_word(w) || w.chars().all(|c| c.is_ascii_digit())) {
            continue;
        }
        add("legal form", span, "ORG");
    }
    for c in org_ru().captures_iter(text) {
        add("Russian legal form", c[1].trim().to_string(), "ORG");
    }
    for c in honorific().captures_iter(text) {
        let whole = c.get(0).unwrap();
        if cited(text, whole.start(), whole.end()) {
            continue;
        }
        let title = whole.as_str().split([' ', '\t']).next().unwrap_or("").trim_end_matches('.');
        add("title", format!("{title} {}", &c[1]), "PERSON");
    }
    // a name found by what stands around it, not by a title: it must have a word that is not contract vocabulary
    let rules: [(&'static str, &'static Regex); 7] = [("role before", role_before()), ("role after", role_after()), ("cue", cue_name()),
        ("Esq.", esquire()), ("numbered party", numbered_party()), ("mail header", mail_header()), ("valediction", valediction())];
    for (rule, re) in rules {
        for c in re.captures_iter(text) {
            let g = c.get(1).unwrap();
            let name = trim_lead(g.as_str());
            if name.split(' ').any(name_like) && !cited(text, g.start(), g.end()) {
                add(rule, name, "PERSON");
            }
        }
    }
    for c in attendance().captures_iter(text) {
        for item in c[1].split([',', ';']).flat_map(|s| s.split(" and ")) {
            if let Some(n) = list_name().captures(item) {
                if n[1].split(' ').any(name_like) {
                    add("attendance", n[1].to_string(), "PERSON");
                }
            }
        }
    }
    for (rule, re) in [("Russian full name", ru_fio()), ("Russian initials", ru_initials()), ("Russian name and patronymic", ru_name_patronymic())] {
        for m in re.find_iter(text) {
            add(rule, trim_lead(m.as_str()), "PERSON");
        }
    }
    for c in query_value().captures_iter(text) {
        let v = c[1].replace('+', " ").replace("%20", " ");
        if v.contains(' ') && v.split(' ').all(|w| w.chars().next().is_some_and(char::is_uppercase)) && v.split(' ').any(name_like) {
            add("link", v, "PERSON");
        }
    }
    for c in vessel().captures_iter(text) {
        add("vessel", c[1].to_string(), "VESSEL");
    }
    for m in us_phone().find_iter(text) {
        add("US phone", m.as_str().to_string(), "PHONE");
    }
    let labelled: [(&'static str, &'static Regex); 10] = [("company number", company_number()), ("VAT number", vat_labelled()),
        ("IMO number", imo()), ("EIN", ein()), ("Russian registration", ru_ids()), ("passport", passport()), ("case number", case_number()),
        ("register number", register_label()), ("bar number", bar_number()), ("Russian account", ru_account())];
    for (rule, re) in labelled {
        for c in re.captures_iter(text) {
            let v = c[1].trim().to_string();
            if (rule == "IMO number" && !imo_ok(&v)) || (rule == "passport" && !v.chars().any(|c| c.is_ascii_digit())) {
                continue;
            }
            add(rule, v, if rule == "Russian account" { "ACCOUNT" } else { "ID" });
        }
    }
    for (rule, re) in [("VAT format", vat_format()), ("Handelsregister", register_de())] {
        for m in re.find_iter(text) {
            add(rule, m.as_str().to_string(), "ID");
        }
    }
    for m in ni_number().find_iter(text) {
        if ni_ok(m.as_str()) {
            add("National Insurance number", m.as_str().to_string(), "ID");
        }
    }
    for c in ssn().captures_iter(text) {
        if ssn_ok(&c[1], &c[2], &c[3]) {
            add("SSN", c[0].to_string(), "ID");
        }
    }
    for c in ru_passport().captures_iter(text) {
        add("Russian passport", format!("{} {}", &c[1], &c[2]), "ID");
    }
}

/// What the legal rules find in `text`, with the rule that found each: `(rule, span, kind)`, to see why a span is
/// hidden.
pub fn explain(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut add = |rule: &'static str, span: String, kind: &str| out.push((rule.to_string(), span, kind.to_string()));
    legal(text, &mut add);
    if let Some(decoded) = url_spaces(text) {
        legal(&decoded, &mut add);
    }
    out
}

impl PiiDetector for LegalDetector {
    fn detect(&self, text: &str, language: &str) -> Vec<(String, String)> {
        let mut found = self.patterns.detect(text, language);
        let mut seen: HashSet<(String, String)> = found.iter().map(|(s, k)| (nf::key(s), k.clone())).collect();
        let mut add = |_rule: &'static str, span: String, kind: &str| {
            let span = span.trim().to_string();
            if span.chars().count() >= 3 && nf::token_in_text(&span).is_none() && seen.insert((nf::key(&span), kind.to_string())) {
                found.push((span, kind.to_string()));
            }
        };
        legal(text, &mut add);
        if let Some(decoded) = url_spaces(text) {
            legal(&decoded, &mut add);
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<(String, String)> {
        LegalDetector::default().detect(text, "en")
    }

    fn has(found: &[(String, String)], span: &str, kind: &str) -> bool {
        found.iter().any(|(s, k)| s == span && k == kind)
    }

    #[test]
    fn a_company_is_found_by_its_legal_form_and_its_name_alone_is_kept() {
        let f = kinds("Signed for and on behalf of Harrowgate Freight Ltd. The Seller holds shares in Kestrel Haulage Limited.");
        assert!(has(&f, "Harrowgate Freight Ltd", "ORG") && has(&f, "Kestrel Haulage Limited", "ORG"), "{f:?}");
        let f = kinds("THE SHARE PURCHASE AGREEMENT AS AMENDED and Solberg Wind AS");
        assert!(!f.iter().any(|(s, _)| s.contains("AGREEMENT")), "two-letter forms after ordinary words are no company: {f:?}");
        assert!(has(&f, "Solberg Wind AS", "ORG"), "{f:?}");
    }

    #[test]
    fn people_are_found_by_their_titles_signatures_and_roles() {
        let f = kinds("Dear Ms Kerr,\nthe site manager, Mr Owen Jeffreys, said so. Priya Raman, a director of the Seller, agreed.\n\
Signed by Daniel Okonkwo.\nOur work will be led by Marcus Bell, Esq.");
        for (s, k) in [("Ms Kerr", "PERSON"), ("Mr Owen Jeffreys", "PERSON"), ("Priya Raman", "PERSON"), ("Daniel Okonkwo", "PERSON"),
                       ("Marcus Bell", "PERSON")] {
            assert!(has(&f, s, k), "{s}: {f:?}");
        }
        let f = kinds("Present: Agnieszka Novak (Chair), Björn Lindqvist, Henrik Sjöberg");
        assert!(has(&f, "Agnieszka Novak", "PERSON") && has(&f, "Henrik Sjöberg", "PERSON"), "{f:?}");
    }

    #[test]
    fn russian_names_and_companies_by_their_own_forms() {
        let f = kinds("Генеральному директору ООО «Северный Путь» Иванову Сергею Петровичу. Уважаемый Сергей Петрович!");
        assert!(has(&f, "Северный Путь", "ORG"), "{f:?}");
        assert!(has(&f, "Иванову Сергею Петровичу", "PERSON") && has(&f, "Сергей Петрович", "PERSON"), "{f:?}");
    }

    #[test]
    fn identifiers_by_their_labels_and_check_digits() {
        let f = kinds("company number 08123456; VAT registration number is GB 123 4567 89; HRB 214365; USt-IdNr.: DE298765431; \
IMO 9871232 and IMO 9871231; NI number JT 48 26 91 B and QQ 12 34 56 C; SSN 512-44-7890 and 000-12-3456; EIN 94-3456781; ИНН 5027113466; \
паспорт 4510 123456; Claim No. KB-2025-001234");
        for v in ["08123456", "GB 123 4567 89", "HRB 214365", "DE298765431", "9871232", "JT 48 26 91 B", "512-44-7890", "94-3456781",
                  "5027113466", "4510 123456", "KB-2025-001234"] {
            assert!(has(&f, v, "ID"), "{v}: {f:?}");
        }
        assert!(!has(&f, "9871231", "ID") && !has(&f, "000-12-3456", "ID") && !has(&f, "QQ 12 34 56 C", "ID"),
                "a bad check digit, an area never issued, a prefix never issued: {f:?}");
    }

    #[test]
    fn a_name_in_a_link_is_found_as_written_out() {
        let f = kinds("https://portal.example-law.com/clients?name=Tamsin%20Okafor&matter=x and q=Brightwater+Maritime+Ltd");
        assert!(has(&f, "Tamsin Okafor", "PERSON") && has(&f, "Brightwater Maritime Ltd", "ORG"), "{f:?}");
    }

    #[test]
    fn defined_terms_are_no_people_and_a_cited_case_keeps_its_parties() {
        let f = kinds("Signed by the Authority Representative. The Supplier's Independent Security Adviser, a member of the \
Supplier Personnel, approved the Order Form. Name: Pension Contributions");
        assert!(f.iter().all(|(_, k)| k != "PERSON"), "defined terms: {f:?}");
        let f = kinds("judgment in Case C-311/18, Data Protection Commissioner v Facebook Ireland Ltd and Maximillian Schrems, and \
en el asunto C-311/18, Data Protection Commissioner / Facebook Ireland Ltd y Maximillian Schrems");
        assert!(f.is_empty(), "a cited judgment's parties stay: {f:?}");
    }

    #[test]
    fn names_with_particles_french_titles_and_more_identifiers() {
        let f = kinds("(3) Pieter van Dijk, residing at Herengracht 210, as Guarantor. Its managing director, Sanne de Groot, \
signs. Représentée par M. Antoine Garnier, gérant. Our client: Nadia Rahimi. By: Samuel Okoro (State Bar No. 287654)");
        for s in ["Pieter van Dijk", "Sanne de Groot", "M Antoine Garnier", "Nadia Rahimi", "Samuel Okoro"] {
            assert!(has(&f, s, "PERSON"), "{s}: {f:?}");
        }
        let f = kinds("registered with the Dutch Chamber of Commerce (KvK) under number 64123987; immatriculée au RCS de Lyon sous \
le numéro 512 345 678; State Bar No. 287654; Case No. 3:25-cv-04512; р/с 40802810762000012345");
        for v in ["64123987", "512 345 678", "287654", "3:25-cv-04512"] {
            assert!(has(&f, v, "ID"), "{v}: {f:?}");
        }
        assert!(has(&f, "40802810762000012345", "ACCOUNT"), "{f:?}");
        assert!(explain("Signed by Daniel Okonkwo.").iter().any(|(r, s, _)| r == "cue" && s == "Daniel Okonkwo"));
    }

    #[test]
    fn statutes_courts_places_and_dates_are_left_alone() {
        let f = kinds("In the High Court of Justice, under the Companies Act 2006 and the Landlord and Tenant Act 1954, \
in London on 3 March 2025, the Buyer paid £2,400,000.");
        assert!(f.iter().all(|(_, k)| k != "ORG" && k != "ID"), "{f:?}");
    }
}
