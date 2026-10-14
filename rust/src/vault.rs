// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The anonymization vault — reversible, per-scope pseudonymization (as `vault.py`).
//!
//! Each real referent gets a stable token (`PERSON_48170392`) derived from the scope's salt; the vault is
//! seeded from the scope's own graph, a residual detector covers the rest, matching goes through the
//! canonical fold with offsets back to the raw text and a word boundary, a name part is an alias (a part
//! two people share gets a token of its own), and nothing is deleted. Places and dates are deliberately
//! not tokenised: we hide who, not where or when.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::detect::{self, PiiDetector, VETO_KIND};
use crate::gateway::Redaction;
use crate::odict::ODict;
use crate::pyre::{self, Regex, I, NONE};
use crate::seeds::{SeedEntity, SeedSource};
use crate::store::{AliasRef, TokenRef, TokenStore};
use crate::{log, namefold as nf, py, unicode as u};

const MIN_KEY_LEN: usize = 3;

fn entity_kind(t: &str) -> &'static str {
    match t {
        "individual" => "PERSON",
        "company" => "ORG",
        "vessel" => "VESSEL",
        "account" => "ACCOUNT",
        "property" => "PROPERTY",
        _ => "ENTITY",
    }
}

const ORG_ROLE_MARKERS: &[&str] = &["company", "entity", "corporation", "corp", "holding", "vessel owner", "shipowner",
    "ship owner", "lessor", "lessee", "bank", "employer", "firm", "llc", "ltd", "limited",
    "insurer", "charterer", "operator", "registry", "counterparty / company", "subsidiary"];
const PERSON_ROLE_MARKERS: &[&str] = &["individual", "person", "director", "shareholder", "beneficial owner", "manager",
    "subject", "applicant", "respondent", "claimant", "spouse", "husband", "wife", "client",
    "employee", "accountant", "witness", "officer", "signatory", "heir", "son", "daughter"];

/// Words of address and articles that are never a match key on their own.
const FRAME_KEYS: &[&str] = &["herr", "herrn", "frau", "fraeulein", "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam",
    "der", "die", "das", "dem", "den", "des", "the", "firma", "gmbh", "ltd", "limited", "ag", "kg"];

fn frame_key(k: &str) -> bool {
    FRAME_KEYS.contains(&k)
}

fn normalize(value: &str) -> String {
    nf::key(value)
}

fn klen(k: &str) -> usize {
    py::clen(k)
}

fn next_index(tokens: &HashSet<String>, kind: &str) -> i64 {
    let prefix = format!("{kind}_");
    let mut mx = 0i64;
    for t in tokens {
        if let Some(rest) = t.strip_prefix(&prefix) {
            if let Some(n) = py::parse_int(rest) {
                mx = mx.max(n);
            }
        }
    }
    mx + 1
}

/// (token kind, is_individual) for a party, from its role.
fn party_kind(role: &str, name: &str) -> (&'static str, bool) {
    let r = u::casefold(role);
    if ORG_ROLE_MARKERS.iter().any(|m| r.contains(m)) {
        return ("ORG", false);
    }
    if PERSON_ROLE_MARKERS.iter().any(|m| r.contains(m)) {
        return ("PERSON", true);
    }
    if nf::has_legal_form(name) {
        return ("ORG", false);
    }
    ("PERSON", false)
}

fn attr_shared(row: &TokenRef) -> bool {
    py::truthy(row.borrow().attr("shared_by"))
}

// ── the index: surface → token ─────────────────────────────────────────────────────────────────

/// The scope's match index, loaded once per call.
pub struct Index {
    pub salt: Option<Vec<u8>>,
    pub rows: Vec<TokenRef>,
    pub by_token: HashMap<String, TokenRef>,
    pub tokens_in_use: HashSet<String>,
    pub aliases: Vec<AliasRef>,
    pub keys: ODict<String>,
    /// Every key that already has an alias row.
    pub alias_keys: HashSet<String>,
    pub legacy_keys: HashSet<String>,
    people: Option<Vec<TokenRef>>,
    pub norms: HashSet<String>,
    /// Short names the documents define for a party ("TW"), matched as written.
    pub defined: ODict<String>,
    /// Other spellings of the names held — keys for matching only.
    pub variants: ODict<String>,
    /// The variants that put a person's surname first.
    pub inverted: HashSet<String>,
}

impl Index {
    pub fn new(rows: Vec<TokenRef>, aliases: Vec<AliasRef>, salt: Option<Vec<u8>>) -> Index {
        let mut by_token = HashMap::new();
        for r in &rows {
            by_token.insert(r.borrow().token.clone(), r.clone());
        }
        let tokens_in_use: HashSet<String> = rows.iter().map(|r| r.borrow().token.clone()).collect();
        let alias_keys: HashSet<String> = aliases.iter().map(|a| a.borrow().normalized.clone()).collect();
        let norms: HashSet<String> = rows.iter().map(|r| r.borrow().normalized.clone()).collect();
        let mut keys = ODict::new();
        let mut legacy_keys = HashSet::new();
        for r in &rows {
            let r = r.borrow();
            if r.status == "active" && !py::strip(&r.real_value).is_empty() {
                let k = normalize(&r.real_value);
                if klen(&k) >= MIN_KEY_LEN && !keys.contains_key(&k) {
                    keys.insert(&k, r.token.clone());
                    legacy_keys.insert(k);
                }
            }
        }
        let mut defined = ODict::new();
        for a in &aliases {
            let a = a.borrow();
            let active = by_token.get(&a.token).is_some_and(|row: &TokenRef| row.borrow().status == "active");
            if !active {
                continue;
            }
            if a.source == "defined" {
                defined.insert(&a.surface, a.token.clone());
                continue;
            }
            keys.insert(&a.normalized, a.token.clone());
            legacy_keys.remove(&a.normalized);
        }
        let mut variants: ODict<String> = ODict::new();
        let mut inverted = HashSet::new();
        let mut held: Vec<(String, String)> = rows.iter().map(|r| (r.borrow().real_value.clone(), r.borrow().token.clone())).collect();
        held.extend(aliases.iter().filter(|a| a.borrow().source != "defined").map(|a| (a.borrow().surface.clone(), a.borrow().token.clone())));
        for (surface, token) in &held {
            let Some(row) = by_token.get(token) else { continue };
            let (status, kind) = (row.borrow().status.clone(), row.borrow().kind.clone());
            if status != "active" || !matches!(kind.as_str(), "PERSON" | "ORG" | "ENTITY") {
                continue;
            }
            for v in nf::spelling_variants(surface, kind == "PERSON") {
                let k = normalize(&v);
                if klen(&k) >= MIN_KEY_LEN && !keys.contains_key(&k) && !frame_key(&k) {
                    variants.setdefault(&k, token.clone());
                }
            }
            if kind != "PERSON" {
                continue;
            }
            let mut forms = vec![surface.clone()];
            forms.extend(nf::spelling_variants(surface, false));
            for form in forms {
                for v in nf::surname_first(&form) {
                    let k = normalize(&v);
                    if klen(&k) >= MIN_KEY_LEN && !keys.contains_key(&k) && !frame_key(&k)
                        && *variants.setdefault(&k, token.clone()) == *token {
                        inverted.insert(k);
                    }
                }
            }
        }
        Index { salt, rows, by_token, tokens_in_use, aliases, keys, alias_keys, legacy_keys, people: None, norms, defined, variants, inverted }
    }

    pub fn token_for(&self, surface: &str) -> Option<String> {
        self.keys.get(&normalize(surface)).cloned()
    }

    /// What text is matched on: the keys and the spellings of the names they hold.
    pub fn match_keys(&self) -> ODict<String> {
        if self.variants.is_empty() { self.keys.clone() } else { self.variants.merged(&self.keys) }
    }

    fn row(&self, token: &str) -> Option<TokenRef> {
        self.by_token.get(token).cloned()
    }

    /// Follow `canonical_token` to the row that carries the referent today.
    pub fn canonical(&self, token: Option<&str>) -> Option<String> {
        let mut seen = HashSet::new();
        let mut token = token.map(str::to_string);
        while let Some(t) = token.clone().filter(|t| !t.is_empty() && !seen.contains(t)) {
            seen.insert(t.clone());
            let Some(row) = self.by_token.get(&t) else { return Some(t) };
            let row = row.borrow();
            if row.status == "active" || row.canonical_token.as_deref().is_none_or(str::is_empty) {
                return Some(t);
            }
            token = row.canonical_token.clone();
        }
        token
    }

    /// Two tokens denote one referent.
    pub fn same_referent(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let (ca, cb) = (self.canonical(Some(a)), self.canonical(Some(b)));
        if ca == cb {
            return true;
        }
        for (x, y) in [(a, b), (b, a)] {
            let shared = self.by_token.get(x).and_then(|r| r.borrow().attr("shared_by").cloned());
            if let Some(Value::Array(list)) = shared.filter(|s| py::truthy(Some(s))) {
                let cy = self.canonical(Some(y));
                if list.iter().any(|t| self.canonical(t.as_str()) == cy) {
                    return true;
                }
            }
        }
        false
    }

    pub fn active(&self) -> Vec<TokenRef> {
        self.rows.iter().filter(|r| r.borrow().status == "active").cloned().collect()
    }

    /// One representative row per distinct person the vault currently knows.
    pub fn people(&mut self) -> Vec<TokenRef> {
        if self.people.is_none() {
            let mut cands: Vec<TokenRef> = self.active().into_iter().filter(|r| {
                let r = r.borrow();
                r.kind == "PERSON" && !py::truthy(r.attr("shared_by")) && nf::name_parts(&r.real_value).len() >= 2
                    && nf::admit_person(&r.real_value).0
            }).collect();
            cands.sort_by_key(|r| {
                let r = r.borrow();
                (if py::truthy(r.attr("entity_type")) { 0 } else { 1 }, -(nf::skeleton(&r.real_value).len() as i64), token_index(&r.token))
            });
            let mut reps: Vec<TokenRef> = Vec::new();
            for r in cands {
                let sk = nf::skeleton(&r.borrow().real_value);
                if reps.iter().any(|x| same_referent_skel(&nf::skeleton(&x.borrow().real_value), &sk)) {
                    continue;
                }
                reps.push(r);
            }
            self.people = Some(reps);
        }
        self.people.clone().unwrap()
    }

    pub fn invalidate(&mut self) {
        self.people = None;
    }
}

fn index_of(store: &dyn TokenStore, scope: &str, salt: Option<Vec<u8>>) -> Index {
    let salt = salt.or_else(|| store.load_salt(scope));
    Index::new(store.load_tokens(scope), store.load_aliases(scope), salt)
}

/// The scope's salt, made at the first call (on the writing path). Not committed here.
pub fn ensure_salt(store: &dyn TokenStore, scope: &str) -> Vec<u8> {
    if let Some(s) = store.load_salt(scope) {
        return s;
    }
    let mut salt = vec![0u8; 32];
    use std::io::Read;
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut salt)).expect("random bytes for a salt");
    store.add_salt(scope, salt.clone());
    salt
}

const TOKEN_SPAN: u64 = 90_000_000;
const TOKEN_FLOOR: u64 = 10_000_000;

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(msg);
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    outer.finalize().into()
}

/// `HMAC(salt, kind ‖ normalized value)` → `PERSON_48170392`; a collision moves on deterministically.
pub fn derive_token(salt: &[u8], kind: &str, surface: &str, taken: &HashSet<String>) -> String {
    let base = format!("{kind}\0{}", normalize(surface)).into_bytes();
    for attempt in 0..64 {
        let mut msg = base.clone();
        if attempt > 0 {
            msg.push(0);
            msg.extend(attempt.to_string().as_bytes());
        }
        let digest = hmac_sha256(salt, &msg);
        let n = u64::from_be_bytes([0, 0, 0, digest[0], digest[1], digest[2], digest[3], digest[4]]) % TOKEN_SPAN + TOKEN_FLOOR;
        let tok = format!("{kind}_{n}");
        if !taken.contains(&tok) {
            return tok;
        }
    }
    panic!("vault: could not derive a free token for {kind}");
}

fn mint_token(idx: &mut Index, kind: &str, surface: Option<&str>) -> String {
    let tok = match (&idx.salt, surface) {
        (Some(salt), Some(s)) if !salt.is_empty() => derive_token(salt, kind, s, &idx.tokens_in_use),
        _ => format!("{kind}_{}", next_index(&idx.tokens_in_use, kind)),
    };
    idx.tokens_in_use.insert(tok.clone());
    tok
}

fn take500(s: &str) -> String {
    py::take(s, 500).to_string()
}

/// Enrol one surface → token. Refuses to move a key that already resolves elsewhere.
fn add_alias(store: &dyn TokenStore, idx: &mut Index, scope: &str, surface: &str, token: &str, source: &str) -> bool {
    let k = normalize(surface);
    if klen(&k) < MIN_KEY_LEN || nf::token_in_text(surface).is_some() {
        return false;
    }
    let owner = idx.keys.get(&k).cloned();
    if let Some(o) = &owner {
        if o != token && !idx.legacy_keys.contains(&k) {
            return false;
        }
        if o == token && !idx.legacy_keys.contains(&k) {
            return true;
        }
    }
    let stored = take500(&k);
    if idx.alias_keys.contains(&stored) {
        // a row for this key exists: re-point it instead of inserting a duplicate
        for a in &idx.aliases {
            let mut a = a.borrow_mut();
            if a.normalized == stored {
                a.token = token.to_string();
                a.surface = py::strip(surface).to_string();
                a.source = source.to_string();
                break;
            }
        }
        idx.keys.insert(&k, token.to_string());
        idx.legacy_keys.remove(&k);
        return true;
    }
    let row = store.add_alias(scope, token, py::strip(surface), &stored, source);
    idx.aliases.push(row);
    idx.alias_keys.insert(stored);
    idx.keys.insert(&k, token.to_string());
    idx.legacy_keys.remove(&k);
    true
}

/// Enrol a short name a document defines for a party ("TW"), matched as written.
fn add_defined_alias(store: &dyn TokenStore, idx: &mut Index, scope: &str, surface: &str, token: &str) -> bool {
    let surface = py::strip(surface).to_string();
    if surface.is_empty() || nf::token_in_text(&surface).is_some() || idx.defined.contains_key(&surface)
        || idx.keys.contains_key(&normalize(&surface)) {
        return false;
    }
    let stored = take500(&format!("={surface}"));
    if idx.alias_keys.contains(&stored) {
        return false;
    }
    let row = store.add_alias(scope, token, &surface, &stored, "defined");
    idx.aliases.push(row);
    idx.alias_keys.insert(stored);
    idx.defined.insert(&surface, token.to_string());
    true
}

fn new_token(store: &dyn TokenStore, idx: &mut Index, scope: &str, surface: &str, kind: &str,
             attributes: Option<Map<String, Value>>, source: &str) -> Option<String> {
    if nf::token_in_text(surface).is_some() {
        log::warning(&format!("vault: refusing a token-bearing value for a new {kind} row"));
        return None;
    }
    let tok = mint_token(idx, kind, Some(surface));
    let mut norm = take500(&normalize(surface));
    if idx.norms.contains(&norm) {
        let room = 500usize.saturating_sub(py::clen(&tok) + 1);
        norm = format!("{}#{tok}", py::take(&norm, room));
    }
    idx.norms.insert(norm.clone());
    let attributes = attributes.filter(|a| !a.is_empty());
    let row = store.add_token(scope, &tok, kind, py::strip(surface), &norm, "active", attributes);
    idx.rows.push(row.clone());
    idx.by_token.insert(tok.clone(), row);
    idx.invalidate();
    add_alias(store, idx, scope, surface, &tok, source);
    Some(tok)
}

// ── identity resolution ────────────────────────────────────────────────────────────────────────

fn token_index(token: &str) -> i64 {
    py::parse_int(token.rsplit('_').next().unwrap_or("")).unwrap_or(0)
}

/// How many parts of `cand` line up with distinct parts of `canon` (skeletons).
fn referent_score(canon: &[String], cand: &[String]) -> usize {
    let mut used: HashSet<usize> = HashSet::new();
    let mut score = 0;
    let mut ordered: Vec<&String> = cand.iter().collect();
    ordered.sort_by_key(|x| klen(x) <= 1);
    for c in ordered {
        for (i, k) in canon.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            let hit = if klen(c) <= 1 { k.starts_with(c.as_str()) } else { nf::stem_match(k, c) };
            if hit {
                used.insert(i);
                score += 1;
                break;
            }
        }
    }
    score
}

fn same_referent_skel(a: &[String], b: &[String]) -> bool {
    referent_score(a, b).max(referent_score(b, a)) >= 2
}

/// (token, ambiguous) for a person surface not found verbatim in the index.
fn resolve_person(idx: &mut Index, surface: &str) -> (Option<String>, bool) {
    let cand = nf::skeleton(surface);
    if cand.is_empty() {
        return (None, false);
    }
    let hits: Vec<String> = idx.people().iter().filter(|r| referent_score(&nf::skeleton(&r.borrow().real_value), &cand) >= 2)
        .map(|r| r.borrow().token.clone()).collect();
    match hits.len() {
        1 => (Some(hits[0].clone()), false),
        0 => (None, false),
        _ => (None, true),
    }
}

/// Active PERSON tokens whose name contains `fragment` as a part.
fn fragment_owners(idx: &mut Index, fragment: &str) -> Vec<String> {
    let f = nf::skeleton_part(fragment);
    if f.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = idx.people().iter().filter(|r| nf::skeleton(&r.borrow().real_value).iter().any(|p| nf::stem_match(p, &f)))
        .map(|r| r.borrow().token.clone()).collect();
    out.sort();
    out.dedup();
    out.sort_by_key(|t| token_index(t));
    out
}

/// key → (surface, tokens proposed for it)
type Pending = ODict<(String, Vec<String>)>;

fn propose_variant(pending: &mut Pending, surface: &str, token: &str) {
    let slot = pending.setdefault(&normalize(surface), (surface.to_string(), Vec::new()));
    if !slot.1.iter().any(|t| t == token) {
        slot.1.push(token.to_string());
    }
}

/// Enrol generated surfaces, but only the unambiguous ones; a surface two referents claim gets a token of
/// its own, glossed as shared.
fn flush_variants(store: &dyn TokenStore, idx: &mut Index, scope: &str, pending: &Pending) {
    let mut items: Vec<(String, (String, Vec<String>))> = pending.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (surface, tokens)) in items {
        if tokens.len() == 1 {
            add_alias(store, idx, scope, &surface, &tokens[0], "variant");
            continue;
        }
        if idx.keys.get(&k).is_none() {
            let mut kinds: Vec<String> = tokens.iter().filter_map(|t| idx.by_token.get(t).map(|r| r.borrow().kind.clone())).collect();
            kinds.sort();
            kinds.dedup();
            if kinds.len() == 1 && matches!(kinds[0].as_str(), "ORG" | "VESSEL" | "ACCOUNT") {
                let mut shared = tokens.clone();
                shared.sort_by(|a, b| token_index(a).cmp(&token_index(b)).then_with(|| a.cmp(b)));
                let mut attrs = Map::new();
                attrs.insert("fragment".into(), json!("name"));
                attrs.insert("shared_by".into(), json!(shared));
                new_token(store, idx, scope, &surface, &kinds[0], Some(attrs), "variant");
            }
            continue;
        }
        for a in &idx.aliases.clone() {
            let mut a = a.borrow_mut();
            if a.normalized == k && a.source == "variant" {
                // neutralised in place, not deleted
                a.source = "retracted".into();
                a.normalized = take500(&format!("~retracted:{k}"));
                idx.alias_keys.remove(&k);
                idx.alias_keys.insert(a.normalized.clone());
                idx.keys.remove(&k);
            }
        }
    }
}

/// Get-or-create the token for one canonical surface, and enrol its variant surfaces.
#[allow(clippy::too_many_arguments)]
fn register_surface(store: &dyn TokenStore, idx: &mut Index, scope: &str, surface: &str, kind: &str,
                    attributes: Option<Map<String, Value>>, source: &str, variants: bool, pending: Option<&mut Pending>) -> Option<String> {
    let surface = py::strip(surface).to_string();
    if nf::is_generic(&surface) || klen(&normalize(&surface)) < MIN_KEY_LEN {
        return None;
    }
    let mut tok = idx.token_for(&surface);
    match &tok {
        None => tok = new_token(store, idx, scope, &surface, kind, attributes, source),
        Some(t) => {
            if let (Some(row), Some(attrs)) = (idx.row(t), attributes.filter(|a| !a.is_empty())) {
                let mut row = row.borrow_mut();
                let mut merged = row.attributes.clone().unwrap_or_default();
                for (k, v) in attrs {
                    merged.entry(k).or_insert(v);
                }
                if row.kind != kind {
                    merged.entry("also_kind").or_insert(json!(kind));
                }
                row.attributes = Some(merged);
            }
        }
    }
    let tok = tok?;
    if variants {
        let mut gen = if kind == "PERSON" {
            nf::person_variants(&surface)
        } else {
            let mut g = nf::org_variants(&surface);
            if let Some(head) = nf::distinctive_head(&surface) {
                g.push(head.clone());
                if let Some(cyr) = nf::cyrillic_form(&head).or_else(|| nf::acronym_cyrillic(&head)) {
                    g.push(cyr);
                }
            }
            g
        };
        match pending {
            None => {
                for v in gen.drain(..) {
                    add_alias(store, idx, scope, &v, &tok, "variant");
                }
            }
            Some(p) => {
                for v in gen.drain(..) {
                    propose_variant(p, &v, &tok);
                }
            }
        }
    }
    Some(tok)
}

/// Enrol the name parts of known individuals: one owner → an alias; shared → a token of its own.
fn register_fragments(store: &dyn TokenStore, idx: &mut Index, scope: &str, people: &[(String, String)]) {
    struct Slot {
        surface: String,
        role: &'static str,
        tokens: Vec<String>,
        skel: String,
    }
    let mut proposals: ODict<Slot> = ODict::new();
    for (surface, token) in people {
        for (part, role) in nf::person_fragments(surface) {
            let k = normalize(&part);
            if klen(&k) < 4 {
                continue;
            }
            let slot = proposals.setdefault(&k, Slot { surface: part.clone(), role, tokens: Vec::new(), skel: nf::skeleton_part(&part) });
            if !slot.tokens.contains(token) {
                slot.tokens.push(token.clone());
            }
        }
    }
    let mut shared_tokens: HashMap<(String, Vec<String>), String> = HashMap::new();
    let also_inflections = |store: &dyn TokenStore, idx: &mut Index, surface: &str, token: &str| {
        for infl in nf::russian_inflections(surface) {
            add_alias(store, idx, scope, &infl, token, "fragment");
        }
        if let Some(cyr) = nf::cyrillic_form(surface) {
            for infl in nf::russian_inflections(&cyr) {
                add_alias(store, idx, scope, &infl, token, "fragment");
            }
        }
    };
    let mut keys: Vec<String> = proposals.keys().map(str::to_string).collect();
    keys.sort();
    for k in keys {
        let slot = proposals.get(&k).unwrap();
        let mut tokens: Vec<String> = slot.tokens.clone();
        tokens.extend(fragment_owners(idx, &slot.surface));
        tokens.sort();
        tokens.dedup();
        let owner_now = idx.keys.get(&k).cloned();
        let existing = owner_now.as_ref().and_then(|o| idx.row(o));
        let group = (slot.skel.clone(), tokens.clone());
        if tokens.len() == 1 {
            add_alias(store, idx, scope, &slot.surface, &tokens[0], "fragment");
            also_inflections(store, idx, &slot.surface, &tokens[0]);
        } else if let Some(grouped) = shared_tokens.get(&group).cloned() {
            add_alias(store, idx, scope, &slot.surface, &grouped, "fragment");
            also_inflections(store, idx, &slot.surface, &grouped);
        } else if let Some(ex) = existing.filter(|e| {
            let e = e.borrow();
            e.kind == "PERSON" && normalize(&e.real_value) == k && !py::truthy(e.attr("entity_type"))
        }) {
            let ex_token = ex.borrow().token.clone();
            {
                let mut e = ex.borrow_mut();
                let mut attrs = e.attributes.clone().unwrap_or_default();
                attrs.insert("fragment".into(), json!(slot.role));
                attrs.insert("shared_by".into(), json!(tokens.iter().filter(|t| **t != ex_token).collect::<Vec<_>>()));
                e.attributes = Some(attrs);
            }
            idx.invalidate();
            shared_tokens.insert(group, ex_token.clone());
            also_inflections(store, idx, &slot.surface, &ex_token);
        } else if owner_now.is_none() || idx.legacy_keys.contains(&k) {
            let mut attrs = Map::new();
            attrs.insert("fragment".into(), json!(slot.role));
            attrs.insert("shared_by".into(), json!(tokens));
            if let Some(minted) = new_token(store, idx, scope, &slot.surface, "PERSON", Some(attrs), "fragment") {
                shared_tokens.insert(group, minted.clone());
                also_inflections(store, idx, &slot.surface, &minted);
            }
        }
    }
}

// ── seeding from the scope's graph ─────────────────────────────────────────────────────────────

fn entity_attributes(e: &SeedEntity) -> Map<String, Value> {
    let mut attrs = Map::new();
    attrs.insert("entity_type".into(), json!(e.r#type));
    if !e.jurisdiction_country.is_empty() {
        attrs.insert("jurisdiction".into(), json!(e.jurisdiction_country));
    }
    let roles: Vec<String> = e.role.split(',').map(|r| py::strip(r).to_string()).filter(|r| !r.is_empty()).collect();
    if !roles.is_empty() {
        attrs.insert("roles".into(), json!(roles.into_iter().take(4).collect::<Vec<_>>()));
    }
    let mut rels = Vec::new();
    for rel in e.relationships.iter().flatten().take(6) {
        if !rel.is_object() || !py::truthy(rel.get("target")) {
            continue;
        }
        rels.push(json!({"type": rel.get("type").filter(|t| py::truthy(Some(t))).map_or("linked".to_string(), py::py_str),
                         "target_name": py::py_str(&rel["target"]),
                         "pct": rel.get("shareholding_pct").cloned().unwrap_or(Value::Null)}));
    }
    if !rels.is_empty() {
        attrs.insert("relations".into(), Value::Array(rels));
    }
    if e.r#type == "vessel" {
        attrs.insert("vessel_type".into(), json!("vessel"));
    }
    attrs
}

fn attrs(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// Pre-populate the vault from the scope's known entities and parties.
pub fn seed_from_scope(store: &dyn TokenStore, scope: &str, seeds: Option<&dyn SeedSource>) -> Result<(), String> {
    let mut idx = index_of(store, scope, None);
    let before = idx.rows.len() + idx.aliases.len();
    let mut people: Vec<(String, String)> = Vec::new();
    let mut pending: Pending = ODict::new();
    let ents = seeds.map(|s| s.entities(scope)).unwrap_or_default();
    for e in &ents {
        let kind = entity_kind(&e.r#type);
        let mut name = py::strip(&e.name).to_string();
        if !name.is_empty() && kind == "PROPERTY" && py::strip(&e.identifier).is_empty() {
            log::info(&format!("vault: property entity {} has no identifier; not enrolled", py::py_str(&e.id)));
            name.clear();
        }
        if !name.is_empty() {
            let tok = register_surface(store, &mut idx, scope, &name, kind, Some(entity_attributes(e)), "entity", true, Some(&mut pending));
            if let (Some(t), "PERSON") = (tok, kind) {
                people.push((name.clone(), t));
            }
        }
        if !py::strip(&e.identifier).is_empty() {
            register_surface(store, &mut idx, scope, &e.identifier, if kind == "ACCOUNT" { "ACCOUNT" } else { "ID" },
                             Some(attrs(&[("identifier_of_name", json!(name)), ("identifier_type", json!(e.identifier_type))])),
                             "entity", false, None);
        }
        for field in ["imo", "mmsi", "callsign", "iban", "registration_number"] {
            let v = e.attributes.as_ref().and_then(|a| a.get(field)).filter(|v| py::truthy(Some(v))).map_or(String::new(), py::py_str);
            let v = py::strip(&v).to_string();
            if klen(&v) >= 4 {
                register_surface(store, &mut idx, scope, &v, "ID",
                                 Some(attrs(&[("identifier_of_name", json!(name)), ("identifier_type", json!(field))])), "entity", false, None);
            }
        }
    }
    let parties = seeds.and_then(|s| s.parties(scope));
    for p in parties.iter().flatten() {
        if !p.is_object() {
            continue;
        }
        let name = py::strip(&p.get("name").filter(|v| py::truthy(Some(v))).map_or(String::new(), py::py_str)).to_string();
        if name.is_empty() || nf::is_generic(&name) {
            continue;
        }
        let role = p.get("role").filter(|v| py::truthy(Some(v))).map_or(String::new(), py::py_str);
        let (kind, individual) = party_kind(&role, &name);
        let mut tok = idx.token_for(&name);
        if tok.is_none() && individual {
            let (resolved, ambiguous) = resolve_person(&mut idx, &name);
            if let Some(r) = resolved {
                add_alias(store, &mut idx, scope, &name, &r, "intake");
                for v in nf::person_variants(&name) {
                    propose_variant(&mut pending, &v, &r);
                }
                tok = Some(r);
            } else if ambiguous {
                tok = new_token(store, &mut idx, scope, &name, "PERSON", Some(attrs(&[("source", json!("intake"))])), "intake");
            }
        }
        if tok.is_none() && !individual {
            tok = merge_org(store, &mut idx, scope, &name, Some(&mut pending));
        }
        if tok.is_none() {
            tok = register_surface(store, &mut idx, scope, &name, kind, Some(attrs(&[("roles", json!([role]))])), "intake",
                                   individual || kind == "ORG", Some(&mut pending));
        }
        if let (Some(t), true) = (tok, individual) {
            people.push((name, t));
        }
    }
    flush_variants(store, &mut idx, scope, &pending);
    register_fragments(store, &mut idx, scope, &people);
    if idx.rows.len() + idx.aliases.len() != before {
        store.commit()?;
    }
    Ok(())
}

/// Resolve a short organisation surface onto an existing organisation, or None.
fn merge_org(store: &dyn TokenStore, idx: &mut Index, scope: &str, name: &str, pending: Option<&mut Pending>) -> Option<String> {
    let parts: Vec<String> = nf::strip_legal_form(name).iter().map(|p| nf::skeleton_part(p)).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() || nf::has_legal_form(name) {
        return None;
    }
    let mut hits: Vec<String> = Vec::new();
    for r in idx.active() {
        let r = r.borrow();
        if !matches!(r.kind.as_str(), "ORG" | "VESSEL" | "ACCOUNT") {
            continue;
        }
        let rparts: Vec<String> = nf::strip_legal_form(&r.real_value).iter().map(|p| nf::skeleton_part(p)).filter(|p| !p.is_empty()).collect();
        if (rparts.len() > parts.len() && rparts[..parts.len()] == parts[..])
            || (rparts == parts && normalize(&r.real_value) != normalize(name)) {
            hits.push(r.token.clone());
        }
    }
    hits.sort();
    hits.dedup();
    if hits.len() != 1 {
        return None;
    }
    let owner = hits[0].clone();
    add_alias(store, idx, scope, name, &owner, "intake");
    match pending {
        None => {
            for v in nf::org_variants(name) {
                add_alias(store, idx, scope, &v, &owner, "variant");
            }
        }
        Some(p) => {
            for v in nf::org_variants(name) {
                propose_variant(p, &v, &owner);
            }
        }
    }
    Some(owner)
}

// ── residual PII ───────────────────────────────────────────────────────────────────────────────

const ROLE_MAILBOXES: [&str; 23] = ["info", "office", "admin", "sales", "support", "contact", "mail", "accounts",
    "accounting", "finance", "hr", "legal", "noreply", "no-reply", "service", "help",
    "team", "billing", "invoice", "reception", "secretariat", "post", "general"];

/// "jose.munoz@corvida.example" → "Jose Munoz".
fn person_from_email(address: &str) -> Option<String> {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let local = address.split('@').next().unwrap_or("");
    let chars: Vec<char> = local.chars().collect();
    let parts: Vec<String> = pyre::lazy(&SPLIT, r"[._-]+", NONE).split(&chars, 0).into_iter().flatten().filter(|p| !p.is_empty()).collect();
    if parts.len() != 2 || parts.iter().any(|p| !u::isalpha(p) || klen(p) < 3) {
        return None;
    }
    if parts.iter().any(|p| ROLE_MAILBOXES.contains(&u::casefold(p).as_str())) {
        return None;
    }
    Some(parts.iter().map(|p| u::capitalize(p)).collect::<Vec<_>>().join(" "))
}

/// Strip the sign-off/framing word a NER span dragged in with the name.
fn trim_frame_words(surface: &str) -> String {
    let mut parts = nf::name_parts(surface);
    while parts.len() > 1 && nf::is_common_word(&parts[0]) && klen(&parts[0]) <= 4 {
        parts.remove(0);
    }
    while parts.len() > 1 && nf::is_common_word(&parts[parts.len() - 1]) && klen(&parts[parts.len() - 1]) <= 4 {
        parts.pop();
    }
    let trimmed = parts.join(" ");
    if klen(&trimmed) >= 4 { trimmed } else { surface.to_string() }
}

/// Should a residual detection be enrolled at all?
fn admit(surface: &str, kind: &str, idx: Option<&Index>, language: Option<&str>, vetoed: Option<&HashSet<String>>) -> (bool, String) {
    if detect::structural(kind) {
        if py::strip(surface).is_empty() {
            return (false, "empty".into());
        }
        if nf::token_in_text(surface).is_some() {
            return (false, "contains a token".into());
        }
        return (true, "structural".into());
    }
    if kind == "PROPERTY" {
        return (false, "prose description, not a designator".into());
    }
    if kind != "PERSON" {
        if py::strip(surface).is_empty() {
            return (false, "empty".into());
        }
        return if nf::token_in_text(surface).is_some() { (false, "contains a token".into()) } else { (true, "non-person".into()) };
    }
    let (ok, reason) = nf::admit_person(surface);
    if !ok {
        return (ok, reason);
    }
    if vetoed.is_some_and(|v| v.contains(&normalize(surface))) {
        return (false, "Presidio also labelled this span a location".into());
    }
    if let Some(lang) = language {
        let want = match lang { "ru" => Some("cyrillic"), "en" => Some("latin"), _ => None };
        if let (Some(w), Some(got)) = (want, nf::script_of(surface)) {
            if got != w {
                return (false, format!("span is {got}, NER language is {lang}"));
            }
        }
    }
    if let Some(idx) = idx {
        if let Some(owner) = covering_referent(idx, surface) {
            return (false, format!("span contains the enrolled surface of {owner}"));
        }
    }
    (true, "ok".into())
}

/// An enrolled ORG/VESSEL/ACCOUNT that this span is part of, in either direction.
fn covering_referent(idx: &Index, surface: &str) -> Option<String> {
    let (hay, _) = nf::fold_haystack(surface);
    if klen(&hay) < 4 {
        return None;
    }
    for (k, tok) in idx.keys.iter() {
        let Some(row) = idx.by_token.get(tok) else { continue };
        if !matches!(row.borrow().kind.as_str(), "ORG" | "VESSEL" | "ACCOUNT") || k == hay {
            continue;
        }
        if klen(k) >= 5 && !nf::find_all(&hay, k).is_empty() {
            return Some(tok.clone());
        }
        if klen(k) > klen(&hay) && !nf::find_all(k, &hay).is_empty() {
            return Some(tok.clone());
        }
    }
    None
}

/// Resolve each residual detection onto an existing referent, or enrol a new one.
fn absorb(store: &dyn TokenStore, scope: &str, detected: &[(String, String)], language: Option<&str>) -> Result<(), String> {
    let mut idx = index_of(store, scope, None);
    let before = idx.rows.len() + idx.aliases.len();
    let mut pending: Pending = ODict::new();
    let vetoed: HashSet<String> = detected.iter().filter(|(_, k)| k == VETO_KIND).map(|(v, _)| normalize(v)).collect();
    let mut found_people: Vec<(String, String)> = Vec::new();
    let mut ordered: Vec<&(String, String)> = detected.iter().collect();
    ordered.sort_by_key(|(s, _)| std::cmp::Reverse(klen(s)));
    for (surface, kind) in ordered {
        let surface = py::strip(surface).to_string();
        if kind == VETO_KIND {
            continue;
        }
        let (ok, _) = admit(&surface, kind, Some(&idx), language, Some(&vetoed));
        if !ok || idx.token_for(&surface).is_some() {
            continue;
        }
        if kind == "PERSON" {
            let (resolved, ambiguous) = resolve_person(&mut idx, &surface);
            if let Some(r) = resolved {
                add_alias(store, &mut idx, scope, &surface, &r, "residual");
                found_people.push((surface.clone(), r));
                continue;
            }
            if ambiguous {
                let owners = fragment_owners(&mut idx, &surface);
                new_token(store, &mut idx, scope, &surface, "PERSON", Some(attrs(&[("shared_by", json!(owners))])), "residual");
                continue;
            }
            let owners = if nf::name_parts(&surface).len() == 1 { fragment_owners(&mut idx, &surface) } else { Vec::new() };
            if owners.len() == 1 {
                add_alias(store, &mut idx, scope, &surface, &owners[0], "residual");
                continue;
            }
            if owners.len() > 1 {
                new_token(store, &mut idx, scope, &surface, "PERSON",
                          Some(attrs(&[("shared_by", json!(owners)), ("fragment", json!("name part"))])), "residual");
                continue;
            }
        }
        let value = if kind == "PERSON" { trim_frame_words(&surface) } else { surface.clone() };
        let tok = register_surface(store, &mut idx, scope, &value, kind, Some(attrs(&[("source", json!("residual"))])), "residual",
                                   matches!(kind.as_str(), "PERSON" | "ORG" | "VESSEL"), Some(&mut pending));
        if let Some(t) = &tok {
            if value != surface {
                add_alias(store, &mut idx, scope, &surface, t, "residual");
            }
            if kind == "PERSON" {
                found_people.push((surface.clone(), t.clone()));
            }
        }
        if kind == "EMAIL" && tok.is_some() {
            if let Some(derived) = person_from_email(&surface) {
                if admit(&derived, "PERSON", None, None, None).0 && idx.token_for(&derived).is_none() {
                    let (resolved, ambiguous) = resolve_person(&mut idx, &derived);
                    if let Some(r) = resolved {
                        add_alias(store, &mut idx, scope, &derived, &r, "residual");
                        found_people.push((derived, r));
                    } else if !ambiguous {
                        if let Some(d) = register_surface(store, &mut idx, scope, &derived, "PERSON", Some(attrs(&[("source", json!("mailbox"))])),
                                                          "residual", true, Some(&mut pending)) {
                            found_people.push((derived, d));
                        }
                    }
                }
            }
        }
    }
    flush_variants(store, &mut idx, scope, &pending);
    if !found_people.is_empty() {
        register_fragments(store, &mut idx, scope, &found_people);
    }
    if idx.rows.len() + idx.aliases.len() != before {
        store.commit()?;
    }
    Ok(())
}

// ── recall passes over the outbound text ───────────────────────────────────────────────────────

fn id_literal() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![0-9A-Za-z])(\d{9}|\d{7})(?![0-9A-Za-z])", NONE)
}

fn callsign() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![0-9A-Za-z])(?=[A-Z0-9]{4,7}(?![0-9A-Za-z]))(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])([A-Z0-9]{4,7})", NONE)
}

fn id_context() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"\b(imo|mmsi|call\s*sign|callsign|позывн|ммси|номер\s+имо|flag|флаг)\b", I)
}

fn domain() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"(?<![0-9A-Za-z@.])((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})(?![0-9A-Za-z])", NONE)
}

/// Does this number carry the structure of an IMO (check digit) or an MMSI (MID in 2–7)?
fn identifier_shaped(digits: &str) -> bool {
    let d: Vec<u32> = digits.chars().filter_map(u::digit_value).collect();
    if d.len() == 7 && digits.chars().count() == 7 {
        return d[..6].iter().zip((2..=7).rev()).map(|(x, w)| x * w).sum::<u32>() % 10 == d[6];
    }
    d.len() == 9 && digits.chars().count() == 9 && "234567".contains(digits.chars().next().unwrap())
}

fn identifier_owner(idx: &Index, token: &str) -> Option<String> {
    let row = idx.by_token.get(token)?;
    let v = row.borrow().attr("identifier_of_name").filter(|v| py::truthy(Some(v))).map(py::py_str).unwrap_or_default();
    (!v.is_empty()).then_some(v)
}

/// The text cut into record-sized pieces: table cells, and windows of three consecutive lines.
fn identifier_segments(text: &str) -> Vec<String> {
    let lines = py::splitlines(text);
    let mut out = Vec::new();
    for i in 0..lines.len() {
        let window = lines[i..(i + 3).min(lines.len())].join("\n");
        if klen(&window) <= 400 {
            out.push(window);
        } else {
            out.extend(window.split('|').filter(|c| klen(c) <= 400).map(str::to_string));
        }
    }
    out
}

/// Enrol identifier literals that sit beside an already-tokenised identifier of exactly one hull.
fn recall_identifiers(store: &dyn TokenStore, idx: &mut Index, scope: &str, text: &str) -> bool {
    let id_keys: Vec<(String, String)> = idx.keys.iter()
        .filter(|(k, tok)| klen(k) >= 4 && idx.by_token.get(*tok).is_some_and(|r| r.borrow().kind == "ID"))
        .map(|(k, t)| (k.to_string(), t.clone())).collect();
    if id_keys.is_empty() {
        return false;
    }
    let mut grew = false;
    for seg in identifier_segments(text) {
        if !seg.chars().any(u::is_digit) {
            continue;
        }
        let (hay, _) = nf::fold_haystack(&seg);
        let mut owners: Vec<String> = Vec::new();
        for (k, tok) in &id_keys {
            if hay.contains(k.as_str()) && !nf::find_all(&hay, k).is_empty() {
                if let Some(owner) = identifier_owner(idx, tok) {
                    if !owners.contains(&owner) {
                        owners.push(owner);
                    }
                }
            }
        }
        let schars: Vec<char> = seg.chars().collect();
        if owners.len() != 1 || id_context().search_at(&schars, 0).is_none() {
            continue;
        }
        let owner = owners[0].clone();
        let mut cands: Vec<String> = id_literal().finditer(&schars).iter()
            .map(|m| pyre::text(&schars, m.group(1).unwrap())).filter(|c| identifier_shaped(c)).collect();
        cands.extend(callsign().finditer(&schars).iter().map(|m| pyre::text(&schars, m.group(1).unwrap())).filter(|c| !nf::is_common_word(c)));
        for c in cands {
            if idx.token_for(&c).is_some() || klen(&c) < 4 {
                continue;
            }
            let tok = register_surface(store, idx, scope, &c, "ID",
                                       Some(attrs(&[("identifier_of_name", json!(owner)), ("identifier_type", json!("secondary"))])),
                                       "residual", false, None);
            grew = grew || tok.is_some();
        }
    }
    grew
}

/// Enrol a domain whose registrable label is an enrolled organisation.
fn recall_domains(store: &dyn TokenStore, idx: &mut Index, scope: &str, text: &str) -> bool {
    static NONALNUM: OnceLock<Regex> = OnceLock::new();
    let mut grew = false;
    let mut hosts: HashMap<String, String> = HashMap::new();
    for row in idx.active() {
        let row = row.borrow();
        if row.kind == "EMAIL" && row.real_value.contains('@') {
            hosts.insert(u::casefold(row.real_value.split_once('@').unwrap().1), row.token.clone());
        }
    }
    let chars: Vec<char> = text.chars().collect();
    for m in domain().finditer(&chars) {
        let dom = pyre::text(&chars, m.group(1).unwrap());
        if idx.token_for(&dom).is_some() {
            continue;
        }
        let labels: Vec<&str> = dom.split('.').collect();
        if labels.len() < 2 || klen(labels[0]) < 3 {
            continue;
        }
        let label = if ["www", "mail", "web"].contains(&u::casefold(labels[0]).as_str()) { labels[labels.len() - 2] } else { labels[0] };
        let lchars: Vec<char> = label.chars().collect();
        let bare = pyre::lazy(&NONALNUM, r"[^0-9A-Za-z]", NONE).sub_with(&lchars, 0, |_| String::new()).0;
        let mut owner = idx.token_for(label).or_else(|| idx.token_for(&bare));
        if owner.is_none() {
            let last_two = labels[labels.len() - 2..].join(".");
            let email_tok = hosts.get(&u::casefold(&last_two)).or_else(|| hosts.get(&u::casefold(&dom)));
            if email_tok.is_none() {
                continue;
            }
            let sp = nf::skeleton_part(label);
            let owner_row = idx.active().into_iter().find(|r| {
                let r = r.borrow();
                matches!(r.kind.as_str(), "ORG" | "VESSEL") && !sp.is_empty()
                    && nf::strip_legal_form(&r.real_value).iter().any(|p| nf::skeleton_part(p) == sp)
            });
            let Some(owner_row) = owner_row else { continue };
            owner = Some(owner_row.borrow().token.clone());
        }
        let owner = owner.unwrap();
        let ok_kind = idx.by_token.get(&owner).is_some_and(|r| matches!(r.borrow().kind.as_str(), "ORG" | "VESSEL" | "ACCOUNT"));
        if !ok_kind {
            continue;
        }
        if add_alias(store, idx, scope, &dom, &owner, "residual") {
            grew = true;
        }
        if klen(label) >= 5 && idx.token_for(label).is_none() {
            grew = add_alias(store, idx, scope, label, &owner, "residual") || grew;
        }
    }
    grew
}

// ── the crossing ───────────────────────────────────────────────────────────────────────────────

const DEFINITION_REACH: usize = 240;

/// Enrol the short names `text` defines for a party it has just named ("TW").
fn learn_definitions(store: &dyn TokenStore, idx: &mut Index, scope: &str, text: &str) -> bool {
    let mut added = false;
    let chars: Vec<char> = text.chars().collect();
    for (term, at) in nf::defined_terms(text) {
        if idx.defined.contains_key(&term) {
            continue;
        }
        let line_start = chars[..at].iter().rposition(|c| *c == '\n').map_or(0, |p| p + 1);
        let start = line_start.max(at.saturating_sub(DEFINITION_REACH));
        let piece: String = chars[start..at].iter().collect();
        let (before, _) = apply_index(&piece, &idx.match_keys(), Some(idx));
        let named: Vec<String> = nf::token_in_text_all(&before).into_iter()
            .map(|(a, b)| py::slice(&before, a, b).to_string()).filter(|t| idx.by_token.contains_key(t)).collect();
        let Some(last) = named.last() else { continue };
        let Some(token) = idx.canonical(Some(last)) else { continue };
        let Some(row) = idx.row(&token) else { continue };
        let ok = {
            let r = row.borrow();
            r.status == "active" && matches!(r.kind.as_str(), "PERSON" | "ORG" | "ENTITY")
        };
        if !ok {
            continue;
        }
        let mut names = vec![row.borrow().real_value.clone()];
        names.extend(idx.aliases.iter().filter(|a| a.borrow().token == token && a.borrow().source != "defined").map(|a| a.borrow().surface.clone()));
        if names.iter().any(|n| nf::defines_alias(&term, n)) {
            added |= add_defined_alias(store, idx, scope, &term, &token);
        }
    }
    added
}

/// Replace every enrolled surface in `text` with its token, in one pass; and per token the surfaces it
/// replaced, in the order the text has them.
pub fn apply_index(text: &str, keys: &ODict<String>, idx: Option<&Index>) -> (String, ODict<Vec<String>>) {
    let empty = ODict::new();
    let defined = idx.map_or(&empty, |i| &i.defined);
    if py::strip(text).is_empty() || (keys.is_empty() && defined.is_empty()) {
        return (text.to_string(), ODict::new());
    }
    let (hay, folded) = nf::fold_haystack(text);
    let hay_chars: Vec<char> = hay.chars().collect();
    let raw: Vec<char> = text.chars().collect();
    let mut taken = vec![false; hay_chars.len()];
    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    let no_inverted = HashSet::new();
    let inverted = idx.map_or(&no_inverted, |i| &i.inverted);
    let mut order: Vec<(&str, &String)> = keys.iter().collect();
    order.sort_by_key(|(k, _)| (if inverted.contains(*k) { 1 } else if k.contains(' ') { 0 } else { 2 }, std::cmp::Reverse(klen(k))));
    for (k, tok) in order {
        let kc: Vec<char> = k.chars().collect();
        for (s, e) in nf::find_all_chars(&hay_chars, &kc) {
            if taken[s..e].iter().any(|t| *t) {
                continue;
            }
            taken[s..e].iter_mut().for_each(|t| *t = true);
            hits.push((s, e, tok.clone()));
        }
    }
    let mut defs: Vec<(&str, &String)> = defined.iter().collect();
    defs.sort_by_key(|(s, _)| std::cmp::Reverse(klen(s)));
    for (surface, tok) in defs {
        let kc: Vec<char> = normalize(surface).chars().collect();
        for (s, e) in nf::find_all_chars(&hay_chars, &kc) {
            if taken[s..e].iter().any(|t| *t) {
                continue;
            }
            let (rs, re_) = folded.raw_span(s, e).unwrap();
            if raw[rs..re_].iter().collect::<String>() != surface {
                continue;
            }
            taken[s..e].iter_mut().for_each(|t| *t = true);
            hits.push((s, e, tok.clone()));
        }
    }
    if hits.is_empty() {
        return (text.to_string(), ODict::new());
    }
    let one_referent = |a: &str, b: &str| a == b || idx.is_some_and(|i| i.same_referent(a, b));
    let more_specific = |a: &str, b: &str| -> String {
        let Some(i) = idx else { return a.to_string() };
        let shared = |t: &str| i.by_token.get(t).is_some_and(attr_shared);
        for (x, y) in [(a, b), (b, a)] {
            if i.by_token.contains_key(x) && shared(x) && !(i.by_token.contains_key(y) && shared(y)) {
                return y.to_string();
            }
        }
        a.to_string()
    };
    hits.sort();
    let mut out: Vec<String> = Vec::new();
    let mut found: ODict<Vec<String>> = ODict::new();
    let mut cursor = 0;
    let mut last_token: Option<String> = None;
    for (s, e, token) in hits {
        let (rs, re_) = folded.raw_span(s, e).unwrap();
        if rs < cursor {
            continue;
        }
        let gap: String = raw[cursor..rs].iter().collect();
        if let Some(lt) = last_token.clone().filter(|lt| py::strip(&gap).is_empty() && one_referent(&token, lt)) {
            let keep = more_specific(&lt, &token);
            let mut surfaces = found.remove(&lt).unwrap_or_default();
            let merged = format!("{}{gap}{}", surfaces.pop().unwrap_or_default(), raw[rs..re_].iter().collect::<String>());
            if !surfaces.is_empty() {
                found.insert(&lt, surfaces);
            }
            found.setdefault(&keep, Vec::new()).push(merged);
            if keep != lt {
                let n = out.len();
                out[n - 1] = keep.clone();
            }
            cursor = re_;
            last_token = Some(keep);
            continue;
        }
        out.push(gap);
        out.push(token.clone());
        found.setdefault(&token, Vec::new()).push(raw[rs..re_].iter().collect());
        cursor = re_;
        last_token = Some(token);
    }
    out.push(raw[cursor..].iter().collect());
    (out.concat(), found)
}

/// Replace real values in `text` with stable per-scope tokens (reversible).
pub fn tokenize(store: &dyn TokenStore, scope: &str, text: &str, seeds: Option<&dyn SeedSource>, detector: Option<&dyn PiiDetector>,
                language: Option<&str>, seed: bool) -> Result<Redaction, String> {
    if py::strip(text).is_empty() {
        return Ok(Redaction::of(text));
    }
    let salt = store.load_salt(scope);
    if seed {
        seed_from_scope(store, scope, seeds)?;
    }
    let lang = language.unwrap_or_else(|| detect::detect_language(text));
    let detected = match detector {
        Some(d) => d.detect(text, lang),
        None => {
            log::warning("vault: Presidio unavailable, residual PII pass skipped (no detector configured)");
            Vec::new()
        }
    };
    if !detected.is_empty() {
        absorb(store, scope, &detected, Some(lang))?;
    }
    let mut idx = index_of(store, scope, salt.clone());
    let a = recall_identifiers(store, &mut idx, scope, text);
    let b = recall_domains(store, &mut idx, scope, text);
    if a | b {
        store.commit()?;
        idx = index_of(store, scope, salt.clone());
    }
    if learn_definitions(store, &mut idx, scope, text) {
        store.commit()?;
        idx = index_of(store, scope, salt);
    }
    let (out, found) = apply_index(text, &idx.match_keys(), Some(&idx));
    let mut red = Redaction::of(&out);
    let mut kinds: ODict<usize> = ODict::new();
    for (token, surfaces) in found.iter() {
        let Some(row) = idx.by_token.get(token) else { continue };
        let row = row.borrow();
        let distinct: HashSet<String> = surfaces.iter().map(|s| normalize(s)).collect();
        red.mapping.insert(token, if distinct.len() == 1 { surfaces[0].clone() } else { row.real_value.clone() });
        red.surfaces.insert(token, surfaces.clone());
        red.canonical.insert(token, row.real_value.clone());
        *kinds.setdefault(&row.kind, 0) += surfaces.len();
    }
    let mut types: Vec<String> = kinds.keys().map(str::to_string).collect();
    types.sort();
    red.entity_types = types;
    red.redacted_count = kinds.iter().map(|(_, n)| *n).sum();
    Ok(red)
}

/// One re-hydration mapping for a call that tokenised several pieces of text.
pub fn merge_mappings(reds: &[Redaction]) -> ODict<String> {
    let mut surfaces: ODict<Vec<String>> = ODict::new();
    let mut canonical: ODict<String> = ODict::new();
    let mut fallback: ODict<String> = ODict::new();
    for red in reds {
        for (tok, vals) in red.surfaces.iter() {
            surfaces.setdefault(tok, Vec::new()).extend(vals.iter().cloned());
        }
        for (k, v) in red.canonical.iter() {
            canonical.insert(k, v.clone());
        }
        for (k, v) in red.mapping.iter() {
            fallback.insert(k, v.clone());
        }
    }
    let mut out = fallback;
    for (tok, vals) in surfaces.iter() {
        let distinct: HashSet<String> = vals.iter().map(|v| normalize(v)).collect();
        let v = if distinct.len() == 1 { vals[0].clone() } else { canonical.get(tok).cloned().unwrap_or_else(|| vals[0].clone()) };
        out.insert(tok, v);
    }
    out
}

fn token_pattern(tokens: &[String]) -> Regex {
    let mut sorted: Vec<&String> = tokens.iter().collect();
    sorted.sort_by_key(|t| std::cmp::Reverse(klen(t)));
    let alts: Vec<String> = sorted.iter().map(|t| pyre::escape(t)).collect();
    Regex::with_flags(&format!(r"(?<![0-9A-Za-z])(?:{})(?![0-9A-Za-z])", alts.join("|")), I).expect("the token pattern")
}

/// A token as re's IGNORECASE matched it, back to the token's own spelling: casefolded, and the letters
/// sre counts as an ASCII letter's case (ı, İ, ſ, K) read as that letter.
fn token_key(matched: &str) -> String {
    let folded: String = matched.chars().map(|c| match c {
        'ı' | 'İ' => 'i',
        'ſ' => 's',
        '\u{212a}' => 'k',
        c => c,
    }).collect();
    u::casefold(&folded)
}

/// Restore real values from tokens: case-insensitive, lookalike letters read as ASCII, longest token
/// first, only the tokens of `mapping` (or, without it, every row through its canonical token).
pub fn detokenize(store: &dyn TokenStore, scope: &str, text: &str, mapping: Option<&ODict<String>>) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let built;
    let mapping = match mapping {
        Some(m) => m,
        None => {
            let rows = store.load_tokens(scope);
            let idx = Index::new(rows.clone(), Vec::new(), None);
            let mut m = ODict::new();
            for r in &rows {
                let tok = r.borrow().token.clone();
                let canon = idx.canonical(Some(&tok)).and_then(|c| idx.by_token.get(&c).cloned()).unwrap_or_else(|| r.clone());
                m.insert(&tok, canon.borrow().real_value.clone());
            }
            built = m;
            &built
        }
    };
    if mapping.is_empty() {
        return text.to_string();
    }
    let mut lookup: HashMap<String, String> = HashMap::new();
    for (t, v) in mapping.iter() {
        lookup.insert(u::casefold(t), v.clone());
    }
    let raw: Vec<char> = text.chars().collect();
    let probe: Vec<char> = raw.iter().map(|c| nf::deconfuse_char(*c)).collect();
    let tokens: Vec<String> = mapping.keys().map(str::to_string).collect();
    let pat = token_pattern(&tokens);
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for m in pat.finditer(&probe) {
        let matched: String = probe[m.start..m.end].iter().collect();
        let Some(value) = lookup.get(&u::casefold(&matched)).or_else(|| lookup.get(&token_key(&matched))) else { continue };
        out.extend(&raw[cursor..m.start]);
        out.push_str(value);
        cursor = m.end;
    }
    out.extend(&raw[cursor..]);
    out
}

/// Enrolled real values still present in `text` — a check of the result, with no enrolment.
pub fn residual_surfaces(store: &dyn TokenStore, scope: &str, text: &str) -> Vec<String> {
    if py::strip(text).is_empty() {
        return Vec::new();
    }
    let idx = index_of(store, scope, None);
    let (_, found) = apply_index(text, &idx.match_keys(), Some(&idx));
    let mut out: Vec<String> = found.iter().flat_map(|(_, s)| s.iter().cloned()).collect();
    out.sort();
    out.dedup();
    out
}

/// `{token: real name}` of the scope's active natural persons.
pub fn active_people(store: &dyn TokenStore, scope: &str) -> ODict<String> {
    let mut out = ODict::new();
    for r in store.load_tokens(scope) {
        let r = r.borrow();
        if r.kind == "PERSON" && r.status == "active" && !r.real_value.is_empty() {
            out.insert(&r.token, r.real_value.clone());
        }
    }
    out
}
