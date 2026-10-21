// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What crosses to the model, shown to the person, who answers for it (Rust only; deidkit's Python has
//! no twin): why each passage of a document is hidden, the names that look like names but are not
//! hidden, the names the person hides themselves, a record of each request that went to the model, the
//! person's approval before a task or a review sends anything, and the matter as a graph of its
//! documents, people, organisations and events. Positions are characters (code points), as in review.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::gateway::{content_hash, Redaction};
use crate::namefold as nf;
use crate::odict::ODict;
use crate::py;
use crate::store::TokenRef;

// ── why a passage is hidden ───────────────────────────────────────────────────────────────────────

/// A name the person hid themselves, kept in the vault's folder (`entities/`) and read again at start.
#[derive(Clone, Debug, PartialEq)]
pub struct Added {
    /// As the seed files say it: individual, company, vessel, account, other.
    pub kind: String,
    pub name: String,
    pub role: String,
    pub by: String,
    pub at: String,
}

/// The seed type for what the page calls it; None — not a kind the person can hide.
pub fn entity_type(given: &str) -> Option<&'static str> {
    match py::strip(given).to_lowercase().as_str() {
        "person" | "individual" => Some("individual"),
        "organization" | "organisation" | "company" => Some("company"),
        "vessel" => Some("vessel"),
        "account" => Some("account"),
        "other" | "" => Some("other"),
        _ => None,
    }
}

fn token_type(token: &str) -> &str {
    token.rsplit_once('_').map_or(token, |(k, _)| k)
}

/// Why the vault hides `token`: `{"why": code, "detail": ...}`. The codes: added (by the person), list
/// (the matter's list of names), identifier (of a listed name), party, mailbox (a name read from an e-mail
/// address), pattern (an e-mail address, IBAN, card, phone number…), name part (of a hidden name),
/// address (a street address), vault (learned earlier in this matter).
pub fn why(row: Option<&TokenRef>, token: &str, added: Option<&Added>) -> Value {
    if let Some(a) = added {
        return json!({"why": "added", "by": a.by, "at": a.at});
    }
    let kind = token_type(token);
    if kind == "ADDRESS" {
        return json!({"why": "address"});
    }
    let Some(row) = row else { return json!({"why": "vault"}) };
    let r = row.borrow();
    let text = |k: &str| r.attr(k).filter(|v| py::truthy(Some(v))).map(py::py_str);
    let roles = r.attr("roles").and_then(Value::as_array)
        .map(|a| a.iter().map(py::py_str).collect::<Vec<_>>().join(", ")).filter(|s| !s.is_empty());
    if r.attr("entity_type").is_some() {
        return json!({"why": "list", "detail": roles.unwrap_or_default()});
    }
    if r.attr("identifier_of_name").is_some() {
        return json!({"why": "identifier", "detail": text("identifier_type").unwrap_or_default()});
    }
    match text("source").as_deref() {
        Some("intake") => return json!({"why": "party", "detail": roles.unwrap_or_default()}),
        Some("mailbox") => return json!({"why": "mailbox"}),
        Some("residual") => return json!({"why": "pattern", "detail": kind.to_lowercase()}),
        _ => {}
    }
    if roles.is_some() {
        return json!({"why": "party", "detail": roles.unwrap_or_default()});
    }
    if r.attr("fragment").is_some() || r.attr("shared_by").is_some() {
        return json!({"why": "name part"});
    }
    json!({"why": "vault"})
}

/// The added names by their tokens now (a name's token is the vault's: the same every time).
pub fn added_tokens<'a>(added: &'a [Added], token_for: &dyn Fn(&str) -> Option<String>) -> HashMap<String, &'a Added> {
    added.iter().filter_map(|a| token_for(&a.name).map(|t| (t, a))).collect()
}

/// What the person sees of a document's de-identification: the text the model is sent, each token with
/// why it is hidden and how often, the counts by type, the tokens the walk could not place, and the
/// likely names that are not hidden.
pub fn hidden(text: &str, red: &Redaction, spans: &[Value], rows: &HashMap<String, TokenRef>,
              added: &HashMap<String, &Added>) -> Value {
    let mut order: ODict<usize> = ODict::new();
    let mut counts: ODict<usize> = ODict::new();
    for s in spans {
        let token = s["token"].as_str().unwrap_or("");
        let n = order.get(token).copied().unwrap_or(0);
        order.insert(token, n + 1);
        let kind = s["type"].as_str().unwrap_or("").to_string();
        let c = counts.get(&kind).copied().unwrap_or(0);
        counts.insert(&kind, c + 1);
    }
    let mut tokens = Vec::new();
    for (token, n) in order.iter() {
        let mut t = json!({"token": token, "type": token_type(token), "name": red.mapping.get(token).cloned().unwrap_or_default(),
                           "count": n, "spellings": red.surfaces.get(token).cloned().unwrap_or_default()});
        let w = why(rows.get(token), token, added.get(token).copied());
        t.as_object_mut().unwrap().extend(w.as_object().unwrap().clone());
        tokens.push(t);
    }
    let unplaced: Vec<String> = red.mapping.keys().filter(|t| !order.contains_key(t)).map(str::to_string).collect();
    let ranges: Vec<(usize, usize)> = spans.iter().filter_map(|s| Some((s["start"].as_u64()? as usize, s["end"].as_u64()? as usize))).collect();
    json!({"model_text": red.text, "tokens": tokens, "counts": Value::Object(counts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()),
           "unplaced": unplaced, "candidates": candidates(text, &ranges)})
}

// ── likely names that are not hidden ──────────────────────────────────────────────────────────────

pub const MAX_CANDIDATES: usize = 60;
const PARTICLES: [&str; 16] = ["von", "van", "der", "den", "de", "du", "da", "di", "del", "della", "la", "le", "bin", "al", "ter", "zu"];
const MAX_RUN: usize = 7;

fn word_char(c: char) -> bool {
    c.is_alphabetic() || "-'’.&".contains(c)
}

/// The words of `t` as `(start, end)`, a word's trailing full stop left out unless it is an initial.
fn words(t: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if !t[i].is_alphabetic() {
            i += 1;
            continue;
        }
        let s = i;
        while i < t.len() && word_char(t[i]) {
            i += 1;
        }
        let mut e = i;
        while e > s + 1 && "-'’.&".contains(t[e - 1]) && !(e - s == 2 && t[e - 1] == '.') {
            e -= 1;
        }
        out.push((s, e));
    }
    out
}

fn capitalised(w: &[char]) -> bool {
    w.first().is_some_and(|c| c.is_uppercase())
}

/// Runs of capitalised words (particles and initials between them), split where anything but one space
/// stands between two words.
fn runs(t: &[char]) -> Vec<Vec<(usize, usize)>> {
    let mut out: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut cur: Vec<(usize, usize)> = Vec::new();
    let mut last_end = 0usize;
    for (s, e) in words(t) {
        let w = &t[s..e];
        let lower: String = w.iter().collect::<String>().to_lowercase();
        let joined = !cur.is_empty() && s == last_end + 1 && t[last_end] == ' ';
        let fits = capitalised(w) || (joined && PARTICLES.contains(&lower.as_str()));
        if fits && (joined || cur.is_empty()) && cur.len() < MAX_RUN {
            cur.push((s, e));
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if capitalised(w) {
                cur.push((s, e));
            }
        }
        last_end = e;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    for r in &mut out {
        while r.last().is_some_and(|&(s, e)| !capitalised(&t[s..e])) {
            r.pop();
        }
    }
    out
}

/// Words that open a phrase rather than a name: articles and the words of a letter's greeting.
const OPENERS: [&str; 18] = ["the", "a", "an", "this", "that", "der", "die", "das", "ein", "eine", "le", "la", "les", "el", "il",
                             "dear", "sehr", "geehrte"];

/// `(kind, start, end)` of the name a run holds, by the vault's own rules for a person and its legal
/// forms for an organisation, once the words that open a phrase are dropped.
fn name_in(t: &[char], run: &[(usize, usize)]) -> Option<(&'static str, usize, usize)> {
    let word = |i: usize| t[run[i].0..run[i].1].iter().collect::<String>();
    let mut from = 0;
    while from < run.len() && (OPENERS.contains(&word(from).to_lowercase().as_str()) || nf::is_honorific(&word(from))) {
        from += 1;
    }
    if run.len() - from < 2 {
        return None;
    }
    let (s, e) = (run[from].0, run[run.len() - 1].1);
    let surface: String = t[s..e].iter().collect();
    if nf::has_legal_form(&surface) && nf::strip_legal_form(&surface).iter().any(|p| !nf::is_common_word(p)) {
        return Some(("organization", s, e));
    }
    nf::admit_person(&surface).0.then_some(("person", s, e))
}

/// Likely names of people and organisations in `text` that no span in `hidden` covers: each once, with
/// where it stands, the first found first.
pub fn candidates(text: &str, hidden: &[(usize, usize)]) -> Vec<Value> {
    let t: Vec<char> = text.chars().collect();
    let covered = |s: usize, e: usize| hidden.iter().any(|&(a, b)| s < b && a < e);
    let mut found: ODict<(String, Vec<Value>)> = ODict::new();
    for run in runs(&t) {
        if run.len() < 2 {
            continue;
        }
        let Some((kind, s, e)) = name_in(&t, &run) else { continue };
        if covered(s, e) {
            continue;
        }
        let surface: String = t[s..e].iter().collect();
        let key = nf::key(&surface);
        if key.is_empty() {
            continue;
        }
        if !found.contains_key(&key) {
            if found.len() >= MAX_CANDIDATES {
                continue;
            }
            found.insert(&key, (kind.to_string(), Vec::new()));
        }
        let mut entry = found.get(&key).cloned().unwrap();
        entry.1.push(json!([s, e]));
        found.insert(&key, entry);
    }
    found.iter().map(|(_, (kind, at))| {
        let (s, e) = (at[0][0].as_u64().unwrap() as usize, at[0][1].as_u64().unwrap() as usize);
        json!({"text": t[s..e].iter().collect::<String>(), "kind": kind, "at": at})
    }).collect()
}

// ── the names the person hides ────────────────────────────────────────────────────────────────────

/// `<vault's folder>/entities/<scope>-<hash>.toml`: the scope in safe letters, the hash keeps two scopes
/// that read alike apart.
pub fn added_file(vault: &Path, scope: &str) -> PathBuf {
    let safe: String = scope.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(60).collect();
    let hash = hex::encode(Sha256::digest(scope.as_bytes()));
    vault.parent().unwrap_or(Path::new(".")).join("entities").join(format!("{safe}-{}.toml", &hash[..8]))
}

/// The names in the file; none when there is no file or it does not read.
pub fn load_added(path: &Path) -> Vec<Added> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let Ok(table) = toml::from_str::<toml::Table>(&text) else { return Vec::new() };
    let s = |e: &toml::Value, k: &str| e.get(k).and_then(toml::Value::as_str).unwrap_or("").to_string();
    table.get("entity").and_then(toml::Value::as_array).into_iter().flatten()
        .map(|e| Added { kind: s(e, "type"), name: s(e, "name"), role: s(e, "role"), by: s(e, "added_by"), at: s(e, "added_at") })
        .filter(|a| !a.name.is_empty() && !a.kind.is_empty()).collect()
}

/// The whole list written again: a seed file (`scope`, `[[entity]]`) that only its owner reads.
pub fn save_added(path: &Path, scope: &str, list: &[Added]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    let mut table = toml::Table::new();
    table.insert("scope".into(), toml::Value::String(scope.into()));
    let entities = list.iter().map(|a| {
        let mut e = toml::Table::new();
        e.insert("type".into(), toml::Value::String(a.kind.clone()));
        e.insert("name".into(), toml::Value::String(a.name.clone()));
        if !a.role.is_empty() {
            e.insert("role".into(), toml::Value::String(a.role.clone()));
        }
        e.insert("added_by".into(), toml::Value::String(a.by.clone()));
        e.insert("added_at".into(), toml::Value::String(a.at.clone()));
        toml::Value::Table(e)
    }).collect();
    table.insert("entity".into(), toml::Value::Array(entities));
    let body = toml::to_string(&table).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("toml.tmp");
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
    f.write_all(format!("# The names the people on this matter hid themselves (deid-agent's page).\n{body}").as_bytes())?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

// ── what went to the model ────────────────────────────────────────────────────────────────────────

/// Bodies kept for the person to read, at most this many bytes in all; the oldest go first, their
/// records stay.
pub const KEEP_BODIES: usize = 64 * 1024 * 1024;
pub const KEEP_RECORDS: usize = 5000;

/// Each request that went to the model in this session: its audit line (when, where, the SHA-256 of the
/// bytes, the types hidden), the task or review it served, and the bytes themselves while they fit.
#[derive(Default)]
pub struct Crossings {
    items: VecDeque<(Value, Option<String>)>,
    kept: usize,
    next: u64,
}

impl Crossings {
    pub fn add(&mut self, audit: &Value, sent: &str, run: &str) -> Value {
        self.next += 1;
        let meta = json!({"id": self.next, "at": audit.get("at").cloned().unwrap_or(Value::Null), "run": run,
                          "action": audit.get("action").cloned().unwrap_or(Value::Null),
                          "to": audit.get("target").cloned().unwrap_or(Value::Null),
                          "sha256": audit.get("sha256").cloned().unwrap_or(Value::Null), "bytes": sent.len(),
                          "tokens": audit.get("redacted_count").cloned().unwrap_or(json!(0)),
                          "types": audit.get("entity_types").cloned().unwrap_or(json!([]))});
        self.kept += sent.len();
        self.items.push_back((meta.clone(), Some(sent.to_string())));
        for item in self.items.iter_mut() {
            if self.kept <= KEEP_BODIES {
                break;
            }
            if let Some(b) = item.1.take() {
                self.kept -= b.len();
            }
        }
        while self.items.len() > KEEP_RECORDS {
            if let Some((_, Some(b))) = self.items.pop_front() {
                self.kept -= b.len();
            }
        }
        meta
    }

    /// The records, the oldest first; with `run`, that task's or review's alone.
    pub fn list(&self, run: Option<&str>) -> Vec<Value> {
        self.items.iter().filter(|(m, _)| run.is_none_or(|r| m["run"] == json!(r)))
            .map(|(m, b)| {
                let mut m = m.clone();
                m["kept"] = json!(b.is_some());
                m
            }).collect()
    }

    pub fn get(&self, id: u64) -> Option<(&Value, Option<&str>)> {
        self.items.iter().find(|(m, _)| m["id"] == json!(id)).map(|(m, b)| (m, b.as_deref()))
    }
}

/// The record of one request with its bytes, and a check of them: the bytes are the ones the audit line
/// hashes, and no value the vault hides is among them (`found`: what is, in clear).
pub fn crossing(meta: &Value, sent: &str, residual: &[String]) -> Value {
    let mut out = meta.clone();
    out["sent"] = json!(sent);
    out["check"] = json!({"sha256_matches": meta["sha256"] == json!(content_hash(sent)), "found": residual,
                          "clean": residual.is_empty()});
    out
}

// ── the person's approval ─────────────────────────────────────────────────────────────────────────

pub const TRANSFERS: &str = "transfers";

/// One decision on a transfer: who let what go to which model, or refused it. Kept as
/// `.legix/transfers/<id>.json` in the matter folder.
pub fn transfer(id: &str, what: &str, run: &str, person: &str, verdict: &str, summary: &Value) -> Value {
    let mut out = json!({"id": id, "what": what, "run": run, "verdict": verdict, "by": person, "at": py::utc_micros()});
    if let Some(m) = summary.as_object() {
        for (k, v) in m {
            out.as_object_mut().unwrap().entry(k.clone()).or_insert(v.clone());
        }
    }
    out
}

pub fn transfers_dir(folder: &Path) -> PathBuf {
    folder.join(crate::agents::review::NOTES).join(TRANSFERS)
}

pub fn store_transfer(folder: &Path, record: &Value) -> std::io::Result<PathBuf> {
    let id = record["id"].as_str().filter(|i| !i.is_empty() && i.chars().all(|c| c.is_ascii_alphanumeric()))
        .ok_or_else(|| std::io::Error::other("a transfer needs an id of letters and digits"))?;
    let target = transfers_dir(folder).join(format!("{id}.json"));
    std::fs::create_dir_all(target.parent().unwrap())?;
    let tmp = target.with_extension("json.tmp");
    std::fs::write(&tmp, py::dumps_indent(record, 2) + "\n")?;
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}

/// The decisions kept in the matter folder, the oldest first.
pub fn transfers(folder: &Path) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(transfers_dir(folder)).into_iter().flatten().flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_str::<Value>(&std::fs::read_to_string(e.path()).ok()?).ok())
        .filter(Value::is_object).collect();
    out.sort_by(|a, b| a["at"].as_str().cmp(&b["at"].as_str()).then(a["id"].as_str().cmp(&b["id"].as_str())));
    out
}

/// What the person is asked to let go, of a text: the counts of what is hidden by type and the likely
/// names left in clear.
pub fn summary_of(text: &str, red: &Redaction, spans: &[Value]) -> Value {
    let mut counts: ODict<usize> = ODict::new();
    for s in spans {
        let kind = s["type"].as_str().unwrap_or("").to_string();
        let c = counts.get(&kind).copied().unwrap_or(0);
        counts.insert(&kind, c + 1);
    }
    let ranges: Vec<(usize, usize)> = spans.iter().filter_map(|s| Some((s["start"].as_u64()? as usize, s["end"].as_u64()? as usize))).collect();
    let clear: Vec<Value> = candidates(text, &ranges).into_iter().map(|c| c["text"].clone()).collect();
    json!({"hidden": Value::Object(counts.iter().map(|(k, v)| (k.to_string(), json!(v))).collect()), "tokens": red.mapping.len(),
           "left_in_clear": clear})
}

// ── the matter as a graph ─────────────────────────────────────────────────────────────────────────

/// One document as the graph takes it: the tokens in it (with how often), its likely names in clear,
/// and its review's clauses — who must or may do what, for whom, and when — each with the tokens its
/// who and for whom name.
pub struct GraphDoc {
    pub path: String,
    pub tokens: Vec<(String, usize)>,
    pub candidates: Vec<Value>,
    pub events: Vec<Value>,
}

/// Kinds of token that stand for a party: the others (e-mail addresses, numbers, street addresses) are
/// counted on the document, not drawn.
fn party_kind(token: &str, row: Option<&TokenRef>) -> Option<&'static str> {
    match token_type(token) {
        "PERSON" => Some("person"),
        "ORG" => Some("organization"),
        "VESSEL" => Some("vessel"),
        "ENTITY" => Some(match row.and_then(|r| r.borrow().attr("entity_type").map(py::py_str)).as_deref() {
            Some("company") => "organization",
            _ => "other",
        }),
        _ => None,
    }
}

/// The token a name part stands in for: its one owner, else itself.
fn owner_of(token: &str, rows: &HashMap<String, TokenRef>) -> String {
    let owners = rows.get(token).and_then(|r| r.borrow().attr("shared_by").and_then(Value::as_array).cloned()).unwrap_or_default();
    match owners.as_slice() {
        [one] if one.as_str().is_some_and(|o| rows.contains_key(o)) => one.as_str().unwrap().to_string(),
        _ => token.to_string(),
    }
}

/// Nodes (documents, people and organisations — hidden, or in clear —, events) and edges (a document
/// mentions a party; an event is in a document; a party is who acts in an event, or for whom).
pub fn graph(docs: &[GraphDoc], rows: &HashMap<String, TokenRef>, names: &HashMap<String, String>,
             added: &HashMap<String, &Added>) -> Value {
    let mut nodes: ODict<Value> = ODict::new();
    let mut edges: Vec<Value> = Vec::new();
    let mut mentions: HashMap<String, usize> = HashMap::new();
    for d in docs {
        let did = format!("d:{}", d.path);
        let mut others = 0usize;
        let mut weights: ODict<usize> = ODict::new();
        for (token, n) in &d.tokens {
            let owner = owner_of(token, rows);
            let Some(kind) = party_kind(&owner, rows.get(&owner)) else {
                others += n;
                continue;
            };
            let id = format!("t:{owner}");
            if !nodes.contains_key(&id) {
                let mut node = json!({"id": id, "kind": kind, "hidden": true, "token": owner,
                                      "label": names.get(&owner).cloned().unwrap_or_default()});
                node.as_object_mut().unwrap().extend(why(rows.get(&owner), &owner, added.get(&owner).copied()).as_object().unwrap().clone());
                nodes.insert(&id, node);
            }
            let w = weights.get(&id).copied().unwrap_or(0);
            weights.insert(&id, w + n);
        }
        for c in &d.candidates {
            let label = c["text"].as_str().unwrap_or("");
            let id = format!("c:{}", nf::key(label));
            if !nodes.contains_key(&id) {
                nodes.insert(&id, json!({"id": id, "kind": c["kind"], "hidden": false, "label": label}));
            }
            let n = c["at"].as_array().map_or(1, Vec::len);
            let w = weights.get(&id).copied().unwrap_or(0);
            weights.insert(&id, w + n);
        }
        nodes.insert(&did, json!({"id": did, "kind": "document", "label": d.path, "identifiers": others,
                                  "clear": d.candidates.len()}));
        for (id, w) in weights.iter() {
            *mentions.entry(id.to_string()).or_default() += 1;
            edges.push(json!({"from": did, "to": id, "kind": "mentions", "weight": w}));
        }
        for (i, e) in d.events.iter().enumerate() {
            let eid = format!("e:{}#{}", d.path, e["id"].as_str().map_or(i.to_string(), str::to_string));
            let mut node = e.clone();
            let label = [e["what"].as_str(), e["quote"].as_str()].into_iter().flatten().find(|s| !s.is_empty()).unwrap_or("").to_string();
            node["id"] = json!(eid);
            node["kind"] = json!("event");
            node["label"] = json!(py::take(&label, 120));
            node["document"] = json!(d.path);
            for field in ["who_tokens", "to_whom_tokens"] {
                node.as_object_mut().unwrap().remove(field);
            }
            nodes.insert(&eid, node);
            edges.push(json!({"from": eid, "to": did, "kind": "in"}));
            for (field, role) in [("who_tokens", "who"), ("to_whom_tokens", "to_whom")] {
                let mut seen = Vec::new();
                for t in e[field].as_array().into_iter().flatten().filter_map(Value::as_str) {
                    let owner = owner_of(t, rows);
                    let id = format!("t:{owner}");
                    if nodes.contains_key(&id) && !seen.contains(&id) {
                        edges.push(json!({"from": id, "to": eid, "kind": role}));
                        seen.push(id);
                    }
                }
            }
        }
    }
    let nodes: Vec<Value> = nodes.iter().map(|(id, n)| {
        let mut n = n.clone();
        if n["kind"] != "document" && n["kind"] != "event" {
            n["documents"] = json!(mentions.get(id).copied().unwrap_or(0));
        }
        n
    }).collect();
    json!({"nodes": nodes, "edges": edges})
}

/// The tokens of a document's spans with how often each stands there, the first found first.
pub fn token_counts(spans: &[Value]) -> Vec<(String, usize)> {
    let mut out: ODict<usize> = ODict::new();
    for s in spans {
        let t = s["token"].as_str().unwrap_or("");
        let n = out.get(t).copied().unwrap_or(0);
        out.insert(t, n + 1);
    }
    out.iter().map(|(t, n)| (t.to_string(), *n)).collect()
}

/// A map of the token rows of a scope by token.
pub fn rows_by_token(rows: Vec<TokenRef>) -> HashMap<String, TokenRef> {
    rows.into_iter().map(|r| {
        let t = r.borrow().token.clone();
        (t, r)
    }).collect()
}

/// `{"token": real value}` of a scope's rows, following a retired duplicate to the row that carries it.
pub fn names_of(rows: &HashMap<String, TokenRef>) -> HashMap<String, String> {
    rows.iter().map(|(t, r)| (t.clone(), r.borrow().real_value.clone())).collect()
}

/// The codes of [`why`], in the order a page's legend lists them.
pub const WHY_CODES: [&str; 9] = ["added", "list", "party", "identifier", "mailbox", "pattern", "name part", "address", "vault"];
