// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Anthropic's Messages API through deid-kit (as `anthropic.py`): the request out in tokens, the answer
//! back in names. Every string from the user's side is tokenised; the model's reasoning, the provider's
//! own blocks, tool definitions and the attribution block stay as they are; images and non-text
//! documents become a note or refuse the request. Text is de-tokenised as it streams, holding back what
//! could be the start of a token; a local tool's arguments are collected and restored whole. The model's
//! own words are remembered and sent back byte for byte when they return as history.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::namefold as nf;
use crate::odict::ODict;
use crate::proxy::engine::ScopeEngine;
use crate::proxy::spelling;
use crate::py;

pub const SUBAGENT_KINDS: [&str; 6] = ["read", "find", "extract", "calc", "check", "consult"];

pub fn subagent_tools() -> HashSet<String> {
    let mut s: HashSet<String> = ["Subagent", "subagent"].iter().map(|x| x.to_string()).collect();
    s.extend(SUBAGENT_KINDS.iter().map(|k| format!("mcp__matter__{k}")));
    s
}

/// Tools that act on the user's machine: their arguments get the real values back.
pub fn local_tools() -> HashSet<String> {
    let mut s: HashSet<String> = ["Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead", "Bash", "BashOutput",
        "KillShell", "KillBash", "Monitor", "Glob", "Grep", "LS", "TodoWrite", "AskUserQuestion", "ExitPlanMode"]
        .iter().map(|x| x.to_string()).collect();
    s.extend(subagent_tools());
    s
}

const UNTOUCHED: [&str; 12] = ["thinking", "redacted_thinking", "tool_reference", "server_tool_use", "web_search_tool_result",
    "web_fetch_tool_result", "code_execution_tool_result", "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result", "mcp_tool_use", "mcp_tool_result", "container_upload"];
const ATTRIBUTION_PREFIX: &str = "x-anthropic-billing-header";

/// The request cannot be de-identified; nothing is sent.
#[derive(Debug)]
pub struct Refused(pub String);

pub fn key(prefix: &str, value: &str) -> String {
    format!("{prefix}{}", hex::encode(Sha256::digest(value.as_bytes())))
}

pub fn canonical(v: &Value) -> String {
    py::dumps_with(v, true, true)
}

/// The strings of `text` (keys and values, one a line) when it is JSON — one value or several in a row, an
/// object or a list among them —, so that an escaped letter reads as itself; else `text` as it is.
pub fn readable(text: &str) -> String {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(m) => m.iter().for_each(|(k, x)| {
                out.push(k.clone());
                walk(x, out)
            }),
            _ => {}
        }
    }
    let values: Result<Vec<Value>, _> = serde_json::Deserializer::from_str(text).into_iter::<Value>().collect();
    match values {
        Ok(vs) if vs.iter().any(|v| v.is_object() || v.is_array()) => {
            let mut out = Vec::new();
            vs.iter().for_each(|v| walk(v, &mut out));
            out.join("\n")
        }
        _ => text.to_string(),
    }
}

/// The model's own words go back as it wrote them (the same bytes every turn), unless they carry what this
/// scope now hides: a name the model wrote in clear before it was enrolled would otherwise cross in clear on
/// every turn after. Returns the restores kept and the paths of those dropped; a dropped path goes as
/// tokenised now.
pub fn safe_restores(restores: Vec<(Path, String)>, engine: &ScopeEngine) -> (Vec<(Path, String)>, Vec<Path>) {
    let texts: Vec<String> = restores.iter().map(|(_, o)| readable(o)).collect();
    if texts.is_empty() || !engine.shows(&texts.join("\n\n")) {
        return (restores, Vec::new());
    }
    let (mut kept, mut dropped) = (Vec::new(), Vec::new());
    for ((p, original), text) in restores.into_iter().zip(texts) {
        if engine.shows(&text) {
            dropped.push(p);
        } else {
            kept.push((p, original));
        }
    }
    (kept, dropped)
}

/// Where a string sits in a JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Seg {
    Key(String),
    Idx(usize),
}

pub type Path = Vec<Seg>;

pub fn at<'a>(v: &'a mut Value, path: &[Seg]) -> &'a mut Value {
    let mut cur = v;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => cur.as_object_mut().expect("a slot's object").entry(k.clone()).or_insert(Value::Null),
            Seg::Idx(i) => cur.get_mut(*i).expect("a slot's index"),
        };
    }
    cur
}

/// The value at `path`, if every step of it is there.
pub fn try_get<'a>(v: &'a Value, path: &[Seg]) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => cur.get(k.as_str())?,
            Seg::Idx(i) => cur.get(*i)?,
        };
    }
    Some(cur)
}

/// The strings of the slots, as Python reads them: a missing key or a value that is no string fails the
/// request; None is no text.
pub fn slot_texts(body: &Value, slots: &[Path]) -> Result<Vec<String>, String> {
    slots.iter().map(|p| match try_get(body, p) {
        None => Err("KeyError".to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Null) => Ok(String::new()),
        Some(_) => Err("AttributeError: not a string".to_string()),
    }).collect()
}

pub fn get<'a>(v: &'a Value, path: &[Seg]) -> &'a Value {
    let mut cur = v;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => &cur[k.as_str()],
            Seg::Idx(i) => &cur[*i],
        };
    }
    cur
}

fn child(path: &[Seg], seg: Seg) -> Path {
    let mut p = path.to_vec();
    p.push(seg);
    p
}

/// Every string value inside a JSON value (keys untouched).
pub fn strings(v: &Value, path: &[Seg], slots: &mut Vec<Path>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                if x.is_string() {
                    slots.push(child(path, Seg::Key(k.clone())));
                } else {
                    strings(x, &child(path, Seg::Key(k.clone())), slots);
                }
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                if x.is_string() {
                    slots.push(child(path, Seg::Idx(i)));
                } else {
                    strings(x, &child(path, Seg::Idx(i)), slots);
                }
            }
        }
        _ => {}
    }
}

fn withhold(block: &Value, what: &str, binary: &str, withheld: &mut Vec<String>) -> Result<Value, Refused> {
    if binary == "refuse" {
        return Err(Refused(format!("the request carries a {what} block; deid-kit de-identifies text only. Convert it to text on this machine first.")));
    }
    withheld.push(what.into());
    let mut note = json!({"type": "text", "text": format!("[{what} withheld by the de-identification proxy: only text is sent to the model]")});
    if let Some(cc) = block.get("cache_control") {
        note["cache_control"] = cc.clone();
    }
    Ok(note)
}

fn block(body: &mut Value, path: &[Seg], slots: &mut Vec<Path>, binary: &str, withheld: &mut Vec<String>, where_: &str) -> Result<(), Refused> {
    let b = get(body, path).clone();
    if !b.is_object() {
        return Err(Refused(format!("unexpected content in {where_}")));
    }
    let t = b.get("type").cloned().unwrap_or(Value::Null);
    let ts = t.as_str().unwrap_or("");
    if ts == "text" && !t.is_null() {
        slots.push(child(path, Seg::Key("text".into())));
    } else if UNTOUCHED.contains(&ts) && t.is_string() {
    } else if ts == "document" && b.get("source").filter(|s| py::truthy(Some(s))).and_then(|s| s.get("type")).and_then(Value::as_str) == Some("text") {
        slots.push(child(&child(path, Seg::Key("source".into())), Seg::Key("data".into())));
        for k in ["title", "context"] {
            if b.get(k).is_some_and(Value::is_string) {
                slots.push(child(path, Seg::Key(k.into())));
            }
        }
    } else if ts == "image" || ts == "document" {
        *at(body, path) = withhold(&b, ts, binary, withheld)?;
    } else if ts == "search_result" {
        for k in ["title", "source"] {
            if b.get(k).is_some_and(Value::is_string) {
                slots.push(child(path, Seg::Key(k.into())));
            }
        }
        let n = b.get("content").filter(|c| py::truthy(Some(c))).and_then(Value::as_array).map_or(0, Vec::len);
        for j in 0..n {
            block(body, &child(&child(path, Seg::Key("content".into())), Seg::Idx(j)), slots, binary, withheld, where_)?;
        }
    } else {
        return Err(Refused(format!("unknown content block type {} in {where_}", py::py_repr(&t))));
    }
    Ok(())
}

pub struct Prepared {
    pub body: Value,
    pub mapping: ODict<String>,
    pub withheld: Vec<String>,
    pub restored: usize,
    /// File and folder names as they were written.
    pub spellings: ODict<String>,
}

pub enum PrepareError {
    Refused(String),
    Failed(String),
}

/// The request as it may cross: tokenised, with the mapping to read the answer back.
pub fn prepare_request(body: &Value, engine: &ScopeEngine, restore_get: &dyn Fn(&str) -> Option<String>, binary: &str)
    -> Result<Prepared, PrepareError> {
    let refused = |r: Refused| PrepareError::Refused(r.0);
    let mut body = body.clone();
    let mut slots: Vec<Path> = Vec::new();
    let mut withheld: Vec<String> = Vec::new();
    let mut restores: Vec<(Path, String)> = Vec::new();
    match body.get("system").cloned() {
        Some(Value::String(_)) => slots.push(vec![Seg::Key("system".into())]),
        Some(Value::Array(system)) => {
            for (i, b) in system.iter().enumerate() {
                if i == 0 && b.is_object() && b.get("text").map_or(String::new(), py::py_str).starts_with(ATTRIBUTION_PREFIX) {
                    continue;
                }
                block(&mut body, &[Seg::Key("system".into()), Seg::Idx(i)], &mut slots, binary, &mut withheld, "system").map_err(refused)?;
            }
        }
        _ => {}
    }
    let n_msgs = body.get("messages").filter(|m| py::truthy(Some(m))).and_then(Value::as_array).map_or(0, Vec::len);
    for m in 0..n_msgs {
        let msg = body["messages"][m].clone();
        let mpath = vec![Seg::Key("messages".into()), Seg::Idx(m)];
        let content = msg.get("content").cloned();
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("").to_string();
        match content {
            Some(Value::String(_)) => {
                slots.push(child(&mpath, Seg::Key("content".into())));
                continue;
            }
            Some(Value::Array(blocks)) => {
                for (i, b) in blocks.iter().enumerate() {
                    let bpath = child(&child(&mpath, Seg::Key("content".into())), Seg::Idx(i));
                    let t = b.get("type").and_then(Value::as_str).unwrap_or("");
                    if b.is_object() && t == "tool_use" {
                        let input = b.get("input").cloned().unwrap_or(json!({}));
                        strings(&input, &child(&bpath, Seg::Key("input".into())), &mut slots);
                        if role == "assistant" {
                            if let Some(original) = restore_get(&key("u:", &canonical(&input))) {
                                restores.push((child(&bpath, Seg::Key("input".into())), original));
                            }
                        }
                    } else if b.is_object() && t == "tool_result" {
                        match b.get("content") {
                            Some(Value::String(_)) => slots.push(child(&bpath, Seg::Key("content".into()))),
                            Some(Value::Array(c)) => {
                                for j in 0..c.len() {
                                    block(&mut body, &child(&child(&bpath, Seg::Key("content".into())), Seg::Idx(j)), &mut slots, binary,
                                          &mut withheld, &format!("message {m} tool_result")).map_err(refused)?;
                                }
                            }
                            _ => {}
                        }
                    } else {
                        block(&mut body, &bpath, &mut slots, binary, &mut withheld, &format!("message {m}")).map_err(refused)?;
                        if role == "assistant" && t == "text" {
                            let text = b.get("text").map_or(String::new(), py::py_str);
                            if let Some(original) = restore_get(&key("t:", &text)) {
                                restores.push((child(&bpath, Seg::Key("text".into())), original));
                            }
                        }
                    }
                }
            }
            _ => return Err(PrepareError::Refused(format!("message {m} has no content list"))),
        }
    }
    let texts = slot_texts(&body, &slots).map_err(PrepareError::Failed)?;
    let reds = engine.tokenize_all(&texts).map_err(PrepareError::Failed)?;
    for (p, red) in slots.iter().zip(&reds) {
        *at(&mut body, p) = Value::String(red.text.clone());
    }
    let (restores, _) = safe_restores(restores, engine);
    let restored = restores.len();
    for (p, original) in restores {
        let last_is_input = matches!(p.last(), Some(Seg::Key(k)) if k == "input");
        *at(&mut body, &p) = if last_is_input { py::loads(&original).unwrap_or(Value::String(original)) } else { Value::String(original) };
    }
    let sent = py::dumps(&body);
    let mapping = engine.mapping_for(&reds, &sent);
    let spellings = spelling::spellings(&texts, &reds);
    Ok(Prepared { body, mapping, withheld, restored, spellings })
}

pub fn detokenize_obj(v: &Value, engine: &ScopeEngine, mapping: &ODict<String>, spellings: Option<&ODict<String>>) -> Value {
    match v {
        Value::String(s) => Value::String(engine.detokenize(s, mapping, spellings, false, false)),
        Value::Array(a) => Value::Array(a.iter().map(|x| detokenize_obj(x, engine, mapping, spellings)).collect()),
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), detokenize_obj(x, engine, mapping, spellings))).collect()),
        other => other.clone(),
    }
}

/// A non-streamed answer, back in names.
pub fn restore_json_response(mut data: Value, engine: &ScopeEngine, mapping: &ODict<String>, restore_put: &dyn Fn(&str, &str),
                             local: &HashSet<String>, spellings: Option<&ODict<String>>) -> Value {
    if let Some(Value::Array(blocks)) = data.get_mut("content") {
        for b in blocks.iter_mut() {
            let t = b.get("type").and_then(Value::as_str).unwrap_or("").to_string();
            if t == "text" {
                let original = b.get("text").map_or(String::new(), py::py_str);
                let restored = engine.detokenize(&original, mapping, spellings, false, true);
                b["text"] = json!(restored);
                restore_put(&key("t:", &restored), &original);
            } else if t == "tool_use" && b.get("name").and_then(Value::as_str).is_some_and(|n| local.contains(n)) {
                let original = b.get("input").cloned().unwrap_or(json!({}));
                let restored = detokenize_obj(&original, engine, mapping, spellings);
                restore_put(&key("u:", &canonical(&restored)), &py::dumps(&original));
                b["input"] = restored;
            }
        }
    }
    data
}

#[derive(Default)]
struct Block {
    kind: String,
    local: bool,
    orig: Vec<char>,
    emitted_orig: usize,
    emitted: String,
    parts: Vec<String>,
}

/// The tail of streaming text that may still change: part of a token, or a name or code span that holds
/// one and is not finished.
pub struct Holdback {
    prefixes: HashSet<String>,
    longest: usize,
}

impl Holdback {
    pub fn new(mapping: &ODict<String>) -> Holdback {
        let mut prefixes = HashSet::new();
        let mut longest = 0;
        for t in mapping.keys() {
            let cf: Vec<char> = crate::unicode::casefold(t).chars().collect();
            for i in 1..=cf.len() {
                prefixes.insert(cf[..i].iter().collect());
            }
            longest = longest.max(py::clen(t));
        }
        Holdback { prefixes, longest }
    }

    pub fn of(&self, text: &[char], spellings: Option<&ODict<String>>) -> usize {
        let hold = spellings.map_or(0, |s| spelling::holdback(&text.iter().collect::<String>(), s));
        for n in (1..=text.len().min(self.longest)).rev() {
            let tail: String = text[text.len() - n..].iter().map(|c| nf::deconfuse_char(*c)).collect();
            if self.prefixes.contains(&crate::unicode::casefold(&tail)) {
                return n.max(hold);
            }
        }
        hold
    }
}

/// Rewrites a Messages API event stream from tokens back to names, event by event.
pub struct StreamRestorer<'e> {
    engine: &'e ScopeEngine,
    mapping: ODict<String>,
    restore_put: Box<dyn Fn(&str, &str) + 'e>,
    local: HashSet<String>,
    spellings: ODict<String>,
    hold: Holdback,
    blocks: HashMap<String, Block>,
}

fn delta_event(index: &Value, delta: Value) -> (String, Value) {
    ("content_block_delta".into(), json!({"type": "content_block_delta", "index": index, "delta": delta}))
}

impl<'e> StreamRestorer<'e> {
    pub fn new(engine: &'e ScopeEngine, mapping: ODict<String>, restore_put: Box<dyn Fn(&str, &str) + 'e>, local: HashSet<String>,
               spellings: ODict<String>) -> StreamRestorer<'e> {
        let hold = Holdback::new(&mapping);
        StreamRestorer { engine, mapping, restore_put, local, spellings, hold, blocks: HashMap::new() }
    }

    fn text(&self, b: &mut Block, final_: bool) -> Result<String, String> {
        let end = if final_ { b.orig.len() } else { b.orig.len() - self.hold.of(&b.orig, Some(&self.spellings)) };
        if end <= b.emitted_orig {
            return Ok(String::new());
        }
        let head: String = b.orig[..end].iter().collect();
        let full = self.engine.detokenize(&head, &self.mapping, Some(&self.spellings), false, true);
        if !full.starts_with(&b.emitted) {
            return Err("de-tokenised prefix moved".into());
        }
        let out = full[b.emitted.len()..].to_string();
        b.emitted = full;
        b.emitted_orig = end;
        Ok(out)
    }

    pub fn event(&mut self, name: &str, data: Value) -> Result<Vec<(String, Value)>, String> {
        let t = data.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        let index_key = |d: &Value| d.get("index").map_or("None".to_string(), py::py_str);
        match t.as_str() {
            "content_block_start" => {
                let cb = data.get("content_block").filter(|c| py::truthy(Some(c))).cloned().unwrap_or(json!({}));
                let idx = data.get("index").ok_or("KeyError: 'index'")?.clone();
                let mut b = Block { kind: cb.get("type").map_or(String::new(), py::py_str), ..Default::default() };
                let text = cb.get("text").filter(|x| py::truthy(Some(x))).map(py::py_str);
                if b.kind == "text" && text.is_some() {
                    b.orig = text.unwrap().chars().collect();
                    let mut d = data.clone();
                    let mut cb2 = cb.clone();
                    cb2["text"] = json!("");
                    d["content_block"] = cb2;
                    let first = self.text(&mut b, false)?;
                    self.blocks.insert(py::py_str(&idx), b);
                    let mut out = vec![(name.to_string(), d)];
                    if !first.is_empty() {
                        out.push(delta_event(&idx, json!({"type": "text_delta", "text": first})));
                    }
                    return Ok(out);
                }
                if b.kind == "tool_use" && cb.get("name").and_then(Value::as_str).is_some_and(|n| self.local.contains(n)) {
                    b.local = true;
                }
                self.blocks.insert(py::py_str(&idx), b);
                Ok(vec![(name.to_string(), data)])
            }
            "content_block_delta" => {
                let k = index_key(&data);
                let d = data.get("delta").filter(|x| py::truthy(Some(x))).cloned().unwrap_or(json!({}));
                let dt = d.get("type").and_then(Value::as_str).unwrap_or("");
                let Some(mut b) = self.blocks.remove(&k) else { return Ok(vec![(name.to_string(), data)]) };
                let result = if b.kind == "text" && dt == "text_delta" {
                    b.orig.extend(d.get("text").map_or(String::new(), py::py_str).chars());
                    let out = self.text(&mut b, false)?;
                    if out.is_empty() {
                        Vec::new()
                    } else {
                        let mut d2 = d.clone();
                        d2["text"] = json!(out);
                        let mut data2 = data.clone();
                        data2["delta"] = d2;
                        vec![(name.to_string(), data2)]
                    }
                } else if b.local && dt == "input_json_delta" {
                    b.parts.push(d.get("partial_json").map_or(String::new(), py::py_str));
                    Vec::new()
                } else {
                    vec![(name.to_string(), data)]
                };
                self.blocks.insert(k, b);
                Ok(result)
            }
            "content_block_stop" => {
                let k = index_key(&data);
                let idx = data.get("index").cloned().unwrap_or(Value::Null);
                let mut out = Vec::new();
                if let Some(mut b) = self.blocks.remove(&k) {
                    if b.kind == "text" {
                        let rest = self.text(&mut b, true)?;
                        if !rest.is_empty() {
                            out.push(delta_event(&idx, json!({"type": "text_delta", "text": rest})));
                        }
                        (self.restore_put)(&key("t:", &b.emitted), &b.orig.iter().collect::<String>());
                    } else if b.local {
                        let raw = b.parts.concat();
                        let original = if py::strip(&raw).is_empty() { Some(json!({})) } else { py::loads(&raw) };
                        let new = match original {
                            None => raw,
                            Some(o) => {
                                let restored = detokenize_obj(&o, self.engine, &self.mapping, Some(&self.spellings));
                                (self.restore_put)(&key("u:", &canonical(&restored)), &py::dumps(&o));
                                py::dumps(&restored)
                            }
                        };
                        out.push(delta_event(&idx, json!({"type": "input_json_delta", "partial_json": new})));
                    }
                    self.blocks.insert(k, b);
                }
                out.push((name.to_string(), data));
                Ok(out)
            }
            _ => Ok(vec![(name.to_string(), data)]),
        }
    }
}
