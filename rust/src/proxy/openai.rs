// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! OpenAI's Responses API through deid-kit, as Codex uses it (as `openai.py`). Out: every string from the
//! user's side is tokenised — instructions, messages, earlier function-call arguments and patches, tool
//! outputs, request metadata and the turn-metadata header. Back: text and patch deltas with a held-back
//! tail, and each completed item, where a local tool's arguments get real values; every emitted event is
//! renumbered. The model's own words come back byte for byte as history.

use std::collections::HashSet;
use std::sync::OnceLock;

use serde_json::{json, Map, Value};

use crate::odict::ODict;
use crate::proxy::anthropic::{at, detokenize_obj, key, safe_restores, slot_texts, strings, subagent_tools, Holdback, Path, PrepareError, Seg};
use crate::proxy::engine::ScopeEngine;
use crate::proxy::spelling;
use crate::pyre::{self, Regex, NONE};
use crate::py;

/// Tools that act on the user's machine.
pub fn local_tools() -> HashSet<String> {
    let mut s: HashSet<String> = ["exec_command", "write_stdin", "shell", "local_shell", "apply_patch", "update_plan", "view_image",
        "read_file", "list_dir", "grep_files", "write_file", "edit_file", "glob_files", "recall"].iter().map(|x| x.to_string()).collect();
    s.extend(subagent_tools());
    s
}

const UNTOUCHED: [&str; 9] = ["reasoning", "compaction", "compaction_summary", "additional_tools", "web_search_call", "item_reference",
    "image_generation_call", "tool_search_call", "tool_search_output"];
pub const CODE_TOOL: &str = "exec";
const TEXT_PARTS: [&str; 4] = ["input_text", "output_text", "text", "summary_text"];
const BINARY_PARTS: [&str; 5] = ["input_image", "input_file", "input_audio", "image", "file"];

fn script_call() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    pyre::lazy(&R, r"\btools\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)", NONE)
}

fn compact(v: &Value) -> String {
    py::dumps_with(v, false, true)
}

pub struct PreparedResponses {
    pub body: Value,
    pub mapping: ODict<String>,
    pub headers: Vec<(String, String)>,
    pub withheld: Vec<String>,
    pub restored: usize,
    pub spellings: ODict<String>,
}

fn refused(msg: String) -> PrepareError {
    PrepareError::Refused(msg)
}

fn note(part: &Value, binary: &str, withheld: &mut Vec<String>) -> Result<Value, PrepareError> {
    let what = part.get("type").map_or("file".to_string(), py::py_str);
    if binary == "refuse" {
        return Err(refused(format!("the request carries an {what} part; deid-kit de-identifies text only. Convert it to text on this machine first.")));
    }
    withheld.push(what.clone());
    Ok(json!({"type": "input_text", "text": format!("[{what} withheld by the de-identification proxy: only text is sent to the model]")}))
}

fn child(p: &[Seg], s: Seg) -> Path {
    let mut v = p.to_vec();
    v.push(s);
    v
}

#[allow(clippy::too_many_arguments)]
fn parts(root: &mut Value, path: &[Seg], slots: &mut Vec<Path>, binary: &str, withheld: &mut Vec<String>, where_: &str,
         restores: Option<&mut Vec<(Path, String)>>, restore_get: &dyn Fn(&str) -> Option<String>) -> Result<(), PrepareError> {
    let list = crate::proxy::anthropic::get(root, path).as_array().cloned().unwrap_or_default();
    let mut restores = restores;
    for (i, part) in list.iter().enumerate() {
        let p = child(path, Seg::Idx(i));
        if part.is_string() {
            slots.push(p);
            continue;
        }
        let t = part.get("type").and_then(Value::as_str).unwrap_or("");
        if TEXT_PARTS.contains(&t) {
            slots.push(child(&p, Seg::Key("text".into())));
            if let Some(r) = restores.as_deref_mut() {
                if t == "output_text" {
                    if let Some(original) = restore_get(&key("t:", &part.get("text").map_or(String::new(), py::py_str))) {
                        r.push((child(&p, Seg::Key("text".into())), original));
                    }
                }
            }
        } else if t == "refusal" {
            slots.push(child(&p, Seg::Key("refusal".into())));
        } else if BINARY_PARTS.contains(&t) {
            *at(root, &p) = note(part, binary, withheld)?;
        } else {
            return Err(refused(format!("unknown content part {} in {where_}", py::py_repr(part.get("type").unwrap_or(&Value::Null)))));
        }
    }
    Ok(())
}

fn output(root: &mut Value, path: &[Seg], slots: &mut Vec<Path>, binary: &str, withheld: &mut Vec<String>, where_: &str) -> Result<(), PrepareError> {
    let p = child(path, Seg::Key("output".into()));
    match crate::proxy::anthropic::try_get(root, &p) {
        Some(Value::String(_)) => slots.push(p),
        Some(Value::Array(_)) => parts(root, &p, slots, binary, withheld, where_, None, &|_| None)?,
        None | Some(Value::Null) => {}
        Some(_) => return Err(refused(format!("unexpected output in {where_}"))),
    }
    Ok(())
}

/// Take the tools of the namespaces `names` out of a request; how many were taken out.
pub fn drop_namespaces(body: &mut Value, names: &HashSet<String>) -> usize {
    if names.is_empty() {
        return 0;
    }
    let mut dropped = 0;
    let mut keep = |tools: &mut Vec<Value>| {
        let before = tools.len();
        tools.retain(|t| !(t.is_object() && t.get("type").and_then(Value::as_str) == Some("namespace")
            && t.get("name").and_then(Value::as_str).is_some_and(|n| names.contains(n))));
        dropped += before - tools.len();
    };
    if let Some(Value::Array(tools)) = body.get_mut("tools") {
        keep(tools);
    }
    if let Some(Value::Array(items)) = body.get_mut("input") {
        for item in items.iter_mut() {
            if item.get("type").and_then(Value::as_str) == Some("additional_tools") {
                if let Some(Value::Array(tools)) = item.get_mut("tools") {
                    keep(tools);
                }
            }
        }
    }
    dropped
}

/// The request as it may cross, the headers that carry metadata, and the mapping back.
pub fn prepare_responses(body: &Value, engine: &ScopeEngine, headers: &[(String, String)], restore_get: &dyn Fn(&str) -> Option<String>,
                         binary: &str, drop: &HashSet<String>) -> Result<PreparedResponses, PrepareError> {
    let mut b = body.clone();
    drop_namespaces(&mut b, drop);
    let hmap: Map<String, Value> = headers.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let mut root = json!({"b": b, "h": hmap});
    let mut slots: Vec<Path> = Vec::new();
    let mut withheld: Vec<String> = Vec::new();
    let mut restores: Vec<(Path, String)> = Vec::new();
    let bp = vec![Seg::Key("b".into())];
    if root["b"].get("instructions").is_some_and(Value::is_string) {
        slots.push(child(&bp, Seg::Key("instructions".into())));
    }
    if let Some(meta) = root["b"].get("client_metadata").filter(|m| m.is_object()).cloned() {
        strings(&meta, &child(&bp, Seg::Key("client_metadata".into())), &mut slots);
    }
    for (name, _) in headers {
        if name.to_lowercase() == "x-codex-turn-metadata" {
            slots.push(vec![Seg::Key("h".into()), Seg::Key(name.clone())]);
        }
    }
    let items: Vec<Value> = match root["b"].get("input").cloned() {
        Some(Value::String(_)) => {
            slots.push(child(&bp, Seg::Key("input".into())));
            Vec::new()
        }
        Some(Value::Array(a)) => a,
        _ => Vec::new(),
    };
    for (n, item) in items.iter().enumerate() {
        let ip = child(&child(&bp, Seg::Key("input".into())), Seg::Idx(n));
        if !item.is_object() {
            return Err(refused(format!("unexpected input item {n}")));
        }
        let t = match item.get("type") {
            Some(v) => py::py_str(v),
            None => if item.get("role").is_some() { "message".into() } else { "None".into() },
        };
        let where_ = format!("input item {n} ({t})");
        match t.as_str() {
            "message" => match item.get("content") {
                Some(Value::String(_)) => slots.push(child(&ip, Seg::Key("content".into()))),
                Some(Value::Array(_)) => {
                    let assistant = item.get("role").and_then(Value::as_str) == Some("assistant");
                    parts(&mut root, &child(&ip, Seg::Key("content".into())), &mut slots, binary, &mut withheld, &where_,
                          if assistant { Some(&mut restores) } else { None }, restore_get)?;
                }
                _ => {}
            },
            "function_call" => {
                let args = item.get("arguments").map_or(String::new(), py::py_str);
                if let Some(original) = restore_get(&key("f:", &args)) {
                    restores.push((child(&ip, Seg::Key("arguments".into())), original));
                } else {
                    let parsed = if py::strip(&args).is_empty() { Some(json!({})) } else { py::loads(&args) };
                    match parsed {
                        None => slots.push(child(&ip, Seg::Key("arguments".into()))),
                        Some(parsed) => {
                            strings(&parsed, &child(&ip, Seg::Key("arguments".into())), &mut slots);
                            *at(&mut root, &child(&ip, Seg::Key("arguments".into()))) = parsed;
                        }
                    }
                }
            }
            "custom_tool_call" => {
                let input = item.get("input").map_or(String::new(), py::py_str);
                if let Some(original) = restore_get(&key("c:", &input)) {
                    restores.push((child(&ip, Seg::Key("input".into())), original));
                } else {
                    slots.push(child(&ip, Seg::Key("input".into())));
                }
            }
            "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" | "mcp_tool_call_output" => {
                output(&mut root, &ip, &mut slots, binary, &mut withheld, &where_)?;
            }
            "local_shell_call" => {
                let action = item.get("action").cloned().unwrap_or(json!({}));
                strings(&action, &child(&ip, Seg::Key("action".into())), &mut slots);
            }
            t if UNTOUCHED.contains(&t) => {}
            _ => return Err(refused(format!("unknown input item type {}", py::py_repr(item.get("type").unwrap_or(&Value::Null))))),
        }
    }
    let mut texts = slot_texts(&root, &slots).map_err(PrepareError::Failed)?;
    let mut reds = engine.tokenize_all(&texts).map_err(PrepareError::Failed)?;
    for (p, red) in slots.iter().zip(&reds) {
        *at(&mut root, p) = Value::String(red.text.clone());
    }
    let (restores, dropped) = safe_restores(restores, engine);
    // a call's arguments or input restored whole was not tokenised above: it is now, as one string
    let mut late: Vec<Path> = Vec::new();
    for p in dropped.into_iter().filter(|p| !slots.contains(p)) {
        let v = crate::proxy::anthropic::get(&root, &p).clone();
        if v.is_string() { late.push(p) } else { strings(&v, &p, &mut late) }
    }
    if !late.is_empty() {
        let more = slot_texts(&root, &late).map_err(PrepareError::Failed)?;
        let red_more = engine.tokenize_all(&more).map_err(PrepareError::Failed)?;
        for (p, red) in late.iter().zip(&red_more) {
            *at(&mut root, p) = Value::String(red.text.clone());
        }
        texts.extend(more);
        reds.extend(red_more);
    }
    if let Some(Value::Array(items)) = root["b"].get_mut("input") {
        for item in items.iter_mut() {
            if item.get("type").and_then(Value::as_str) == Some("function_call") && !item.get("arguments").is_some_and(Value::is_string) {
                let c = compact(item.get("arguments").unwrap_or(&Value::Null));
                item["arguments"] = json!(c);
            }
        }
    }
    let restored = restores.len();
    for (p, original) in restores {
        *at(&mut root, &p) = Value::String(original);
    }
    let body = root["b"].take();
    let headers: Vec<(String, String)> = root["h"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), py::py_str(v))).collect();
    let hjson: Map<String, Value> = headers.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let sent = py::dumps(&body) + &py::dumps(&Value::Object(hjson));
    let mapping = engine.mapping_for(&reds, &sent);
    let spellings = spelling::spellings(&texts, &reds);
    Ok(PreparedResponses { body, mapping, headers, withheld, restored, spellings })
}

/// A call's tool as the tool list names it: `mcp__<server>__<tool>` for a tool in a namespace.
pub fn tool_name(item: &Value) -> String {
    let name = item.get("name").filter(|v| py::truthy(Some(v))).map_or(String::new(), py::py_str);
    match item.get("namespace").and_then(Value::as_str) {
        Some(space) if !space.is_empty() && !name.starts_with(space) => format!("{space}__{name}"),
        _ => name,
    }
}

/// Does a code-mode script call only tools that act on this machine?
pub fn script_is_local(script: &str, local: &HashSet<String>) -> bool {
    let c: Vec<char> = script.chars().collect();
    let calls: Vec<String> = script_call().finditer(&c).iter().map(|m| pyre::text(&c, m.group(1).unwrap())).collect();
    !calls.is_empty() && calls.iter().all(|x| local.contains(x))
}

/// The mapping with each value escaped for a string literal.
pub fn literal_mapping(mapping: &ODict<String>) -> ODict<String> {
    let mut out = ODict::new();
    for (t, v) in mapping.iter() {
        let d = py::dumps(&json!(v));
        out.insert(t, d[1..d.len() - 1].to_string());
    }
    out
}

/// A completed output item, back in names (and remembered as the model wrote it).
pub fn restore_item(item: &Value, engine: &ScopeEngine, mapping: &ODict<String>, restore_put: Option<&dyn Fn(&str, &str)>,
                    local: &HashSet<String>, spellings: Option<&ODict<String>>) -> Value {
    let mut item = item.clone();
    let t = item.get("type").and_then(Value::as_str).unwrap_or("").to_string();
    if t == "message" {
        if let Some(Value::Array(parts)) = item.get_mut("content") {
            for part in parts.iter_mut() {
                if part.is_object() && part.get("type").and_then(Value::as_str) == Some("output_text") {
                    let original = part.get("text").map_or(String::new(), py::py_str);
                    let restored = engine.detokenize(&original, mapping, spellings, false, true);
                    part["text"] = json!(restored);
                    if let Some(put) = restore_put {
                        put(&key("t:", &restored), &original);
                    }
                }
            }
        }
    } else if t == "function_call" && local.contains(&tool_name(&item)) {
        let original = item.get("arguments").map_or(String::new(), py::py_str);
        let restored = match py::loads(&original) {
            Some(v) => compact(&detokenize_obj(&v, engine, mapping, spellings)),
            None => engine.detokenize(&original, mapping, spellings, false, false),
        };
        item["arguments"] = json!(restored);
        if let Some(put) = restore_put {
            put(&key("f:", &restored), &original);
        }
    } else if t == "custom_tool_call" {
        let name = item.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        let input = item.get("input").map_or(String::new(), py::py_str);
        if local.contains(&name) || (name == CODE_TOOL && script_is_local(&input, local)) {
            let code = name == CODE_TOOL;
            let values = if code { literal_mapping(mapping) } else { mapping.clone() };
            let restored = engine.detokenize(&input, &values, spellings, code, false);
            item["input"] = json!(restored);
            if let Some(put) = restore_put {
                put(&key("c:", &restored), &input);
            }
        }
    }
    item
}

#[derive(Default, Clone)]
struct Stream {
    orig: Vec<char>,
    emitted_orig: usize,
    emitted: String,
    prose: bool,
    shape: Map<String, Value>,
    delta_type: String,
    item_id: Value,
}

/// Rewrites a Responses API event stream from tokens back to names, event by event.
pub struct ResponsesRestorer<'e> {
    engine: &'e ScopeEngine,
    mapping: ODict<String>,
    restore_put: Box<dyn Fn(&str, &str) + 'e>,
    local: HashSet<String>,
    spellings: ODict<String>,
    hold: Holdback,
    streams: ODict<Stream>,
    local_items: HashSet<String>,
    seq: i64,
}

fn stream_key(delta_type: &str, item_id: &Value, content_index: &Value) -> String {
    py::dumps(&json!([delta_type, item_id, content_index]))
}

impl<'e> ResponsesRestorer<'e> {
    pub fn new(engine: &'e ScopeEngine, mapping: ODict<String>, restore_put: Box<dyn Fn(&str, &str) + 'e>, local: HashSet<String>,
               spellings: ODict<String>) -> ResponsesRestorer<'e> {
        let hold = Holdback::new(&mapping);
        ResponsesRestorer { engine, mapping, restore_put, local, spellings, hold, streams: ODict::new(), local_items: HashSet::new(), seq: 0 }
    }

    fn advance(&self, s: &mut Stream, final_: bool) -> Result<String, String> {
        let spell = if s.prose { Some(&self.spellings) } else { None };
        let end = if final_ { s.orig.len() } else { s.orig.len() - self.hold.of(&s.orig, spell) };
        if end <= s.emitted_orig {
            return Ok(String::new());
        }
        let head: String = s.orig[..end].iter().collect();
        let full = self.engine.detokenize(&head, &self.mapping, spell, false, s.prose);
        if !full.starts_with(&s.emitted) {
            return Err("de-tokenised prefix moved".into());
        }
        let out = full[s.emitted.len()..].to_string();
        s.emitted = full;
        s.emitted_orig = end;
        Ok(out)
    }

    fn numbered(&mut self, mut data: Value) -> Value {
        if data.get("sequence_number").is_some() {
            data["sequence_number"] = json!(self.seq);
        }
        self.seq += 1;
        data
    }

    pub fn event(&mut self, name: &str, data: Value) -> Result<Vec<(String, Value)>, String> {
        let out = self.event_inner(name, data)?;
        Ok(out.into_iter().map(|(n, d)| (n, self.numbered(d))).collect())
    }

    fn put(&self) -> &dyn Fn(&str, &str) {
        &*self.restore_put
    }

    fn event_inner(&mut self, name: &str, data: Value) -> Result<Vec<(String, Value)>, String> {
        let t = data.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        let item_key = |d: &Value| d.get("item_id").filter(|v| py::truthy(Some(v))).map(py::py_str)
            .unwrap_or_else(|| py::py_str(d.get("output_index").unwrap_or(&Value::Null)));
        match t.as_str() {
            "response.output_item.added" => {
                let item = data.get("item").filter(|v| py::truthy(Some(v))).cloned().unwrap_or(json!({}));
                let it = item.get("type").and_then(Value::as_str).unwrap_or("");
                if (it == "function_call" || it == "custom_tool_call")
                    && (self.local.contains(&tool_name(&item)) || item.get("name").and_then(Value::as_str) == Some(CODE_TOOL)) {
                    let id = item.get("id").filter(|v| py::truthy(Some(v))).map(py::py_str)
                        .unwrap_or_else(|| py::py_str(data.get("output_index").unwrap_or(&Value::Null)));
                    self.local_items.insert(id);
                }
                Ok(vec![(name.into(), data)])
            }
            "response.output_text.delta" | "response.custom_tool_call_input.delta" => {
                if t == "response.custom_tool_call_input.delta" && !self.local_items.contains(&item_key(&data)) {
                    return Ok(vec![(name.into(), data)]);
                }
                let k = stream_key(&t, data.get("item_id").unwrap_or(&Value::Null), data.get("content_index").unwrap_or(&Value::Null));
                if !self.streams.contains_key(&k) {
                    self.streams.insert(&k, Stream { prose: t == "response.output_text.delta", ..Default::default() });
                }
                let mut s = self.streams.get(&k).cloned().unwrap();
                s.shape = data.as_object().cloned().unwrap_or_default();
                s.shape.remove("delta");
                s.delta_type = t.clone();
                s.item_id = data.get("item_id").cloned().unwrap_or(Value::Null);
                s.orig.extend(data.get("delta").map_or(String::new(), py::py_str).chars());
                let out = self.advance(&mut s, false);
                self.streams_put(&k, s);
                let out = out?;
                if out.is_empty() {
                    return Ok(Vec::new());
                }
                let mut d = data.clone();
                d["delta"] = json!(out);
                Ok(vec![(name.into(), d)])
            }
            "response.output_text.done" | "response.custom_tool_call_input.done" => {
                let field = if t == "response.output_text.done" { "text" } else { "input" };
                let delta_type = t.replace(".done", ".delta");
                let k = stream_key(&delta_type, data.get("item_id").unwrap_or(&Value::Null), data.get("content_index").unwrap_or(&Value::Null));
                let s = self.streams.remove(&k);
                let had = s.is_some();
                let mut out = Vec::new();
                if let Some(mut s) = s {
                    let rest = self.advance(&mut s, true)?;
                    if !rest.is_empty() {
                        let mut delta = data.as_object().cloned().unwrap_or_default();
                        delta.remove("text");
                        delta.remove("input");
                        delta.insert("type".into(), json!(delta_type));
                        delta.insert("delta".into(), json!(rest));
                        let n = if name.is_empty() { String::new() } else { name.replace(".done", ".delta") };
                        out.push((n, Value::Object(delta)));
                    }
                }
                let mut data = data;
                if let Some(Value::String(v)) = data.get(field).cloned() {
                    if t == "response.output_text.done" || had {
                        let restored = self.engine.detokenize(&v, &self.mapping, Some(&self.spellings), false, field == "text");
                        data[field] = json!(restored);
                    }
                }
                out.push((name.into(), data));
                Ok(out)
            }
            "response.content_part.done" => {
                let mut data = data;
                let part = data.get("part").filter(|v| py::truthy(Some(v))).cloned().unwrap_or(json!({}));
                if part.get("type").and_then(Value::as_str) == Some("output_text") {
                    let mut part = part.clone();
                    let text = part.get("text").map_or(String::new(), py::py_str);
                    part["text"] = json!(self.engine.detokenize(&text, &self.mapping, Some(&self.spellings), false, true));
                    data["part"] = part;
                }
                Ok(vec![(name.into(), data)])
            }
            "response.output_item.done" => {
                let item = data.get("item").filter(|v| py::truthy(Some(v))).cloned().unwrap_or(json!({}));
                let item_id = item.get("id").cloned().unwrap_or(Value::Null);
                let mut out = self.flush(|s| s.item_id == item_id)?;
                let restored = restore_item(&item, self.engine, &self.mapping, Some(self.put()), &self.local, Some(&self.spellings));
                let mut data = data;
                data["item"] = restored;
                out.push((name.into(), data));
                Ok(out)
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                let mut out = self.flush(|_| true)?;
                let mut data = data;
                if t == "response.completed" {
                    if let Some(Value::Object(resp)) = data.get("response").cloned() {
                        let mut resp = resp;
                        if let Some(Value::Array(items)) = resp.get("output").cloned() {
                            let restored: Vec<Value> = items.iter()
                                .map(|i| restore_item(i, self.engine, &self.mapping, None, &self.local, Some(&self.spellings))).collect();
                            resp.insert("output".into(), Value::Array(restored));
                        }
                        data["response"] = Value::Object(resp);
                    }
                }
                out.push((name.into(), data));
                Ok(out)
            }
            _ => Ok(vec![(name.into(), data)]),
        }
    }

    /// The stream back in its place (a dict keeps the place of a key it has).
    fn streams_put(&mut self, k: &str, s: Stream) {
        *self.streams.get_mut(k).unwrap() = s;
    }

    /// The text still held back in the streams `which` selects, as one last delta each.
    fn flush(&mut self, which: impl Fn(&Stream) -> bool) -> Result<Vec<(String, Value)>, String> {
        let keys: Vec<String> = self.streams.iter().filter(|(_, s)| which(s)).map(|(k, _)| k.to_string()).collect();
        let mut out = Vec::new();
        for k in keys {
            let mut s = self.streams.remove(&k).unwrap();
            let rest = self.advance(&mut s, true)?;
            if !rest.is_empty() {
                let mut d = s.shape.clone();
                d.insert("delta".into(), json!(rest));
                out.push((s.delta_type.clone(), Value::Object(d)));
            }
        }
        Ok(out)
    }
}
