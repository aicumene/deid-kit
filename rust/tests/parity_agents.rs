// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! deid-agent's helpers answer as Python's do: the norms a text cites, a quote found as models write it,
//! a clause's address, the notes in an agent's answer, where each token stands, the person's changes word
//! by word, a note placed again and applied, the brief an agent is told, the model choice, a turn's usage
//! and a task's cost, shell words, the paths a command names outside the folder, Claude Code's settings,
//! Codex's Seatbelt profile, and the markup of a document read back.
//! Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_agents.py <dir>`.

mod common;

use std::path::Path;

use common::Diff;
use deidkit::agents::{cite, confine, host, pricing, review};
use deidkit::odict::ODict;
use serde_json::{json, Map, Value};

fn s(v: &Value) -> &str {
    v.as_str().unwrap()
}

fn odict<V: Clone>(v: &Value, f: impl Fn(&Value) -> V) -> ODict<V> {
    let mut d = ODict::new();
    for (k, x) in v.as_object().unwrap() {
        d.insert(k, f(x));
    }
    d
}

fn report(d: Diff) {
    if d.0.len() > 40 {
        let n = d.0.len();
        let mut d = d;
        d.0.truncate(40);
        d.0.push(format!("… and {} more", n - 40));
        d.done();
    } else {
        d.done();
    }
}

#[test]
fn the_agent_helpers_are_pythons() {
    let Some((_, v)) = common::vectors("agents.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    // the vectors were made with this HOME; paths with ~ in them need the same
    std::env::set_var("HOME", s(&v["home"]));
    let mut d = Diff::new();

    for c in v["cite"].as_array().unwrap() {
        let t = s(&c["text"]);
        let spans: Vec<Value> = cite::extract_citation_spans(t).into_iter().map(|x| json!([x.start, x.end, x.citation])).collect();
        d.eq(format!("cite {t:?}"), &c["spans"], &json!(spans));
        d.eq(format!("norms {t:?}"), &c["norms"], &json!(review::norm_mentions(t)));
    }
    for c in v["find"].as_array().unwrap() {
        let f = review::Folded::of(s(&c["text"]));
        let got = f.find(s(&c["quote"]), c["near"].as_i64()).map(|(a, b)| json!([a, b])).unwrap_or(Value::Null);
        d.eq(format!("find {:?} in {:?}", c["quote"], c["text"]), &c["found"], &got);
    }
    for c in v["clause"].as_array().unwrap() {
        let t: Vec<char> = s(&c["text"]).chars().collect();
        d.eq(format!("clause_at {:?} {}", c["text"], c["at"]), &c["clause"], &json!(review::clause_at(&t, c["at"].as_u64().unwrap() as usize)));
    }
    for c in v["notes"].as_array().unwrap() {
        let (notes, cut) = review::notes_of(s(&c["answer"]));
        d.eq(format!("notes_of {:?}", c["answer"]), &json!({"notes": c["notes"], "cut": c["cut"]}), &json!({"notes": notes, "cut": cut}));
    }
    for c in v["deid"].as_array().unwrap() {
        let surfaces = odict(&c["surfaces"], |x| x.as_array().unwrap().iter().map(|y| s(y).to_string()).collect::<Vec<_>>());
        let mapping = odict(&c["mapping"], |x| s(x).to_string());
        let got = review::deid_spans(s(&c["original"]), s(&c["tokenised"]), &surfaces, &mapping);
        d.eq(format!("deid_spans {:?}", c["original"]), &c["spans"], &json!(got));
    }
    for c in v["hunks"].as_array().unwrap() {
        let limit = if c["hunks"].as_array().unwrap().len() <= 2 { 2 } else { 200 };
        let got = review::hunks(s(&c["before"]), s(&c["after"]), 200);
        let mut want = c["hunks"].clone();
        if limit == 2 {
            // made with limit 2 or 200: compare as many as Python gave
            let n = want.as_array().unwrap().len();
            d.eq(format!("hunks {:?} -> {:?}", c["before"], c["after"]), &want, &json!(got.iter().take(n).cloned().collect::<Vec<_>>()));
            want = Value::Null;
        }
        if !want.is_null() {
            d.eq(format!("hunks {:?} -> {:?}", c["before"], c["after"]), &want, &json!(got));
        }
    }
    for c in v["place"].as_array().unwrap() {
        let t = s(&c["text"]);
        let got = review::place(&c["ann"], &review::Folded::of(t)).map(|p| json!([p.status, p.start, p.end])).unwrap_or(Value::Null);
        d.eq(format!("place {} in {t:?}", c["ann"]), &c["place"], &got);
        d.eq(format!("apply {} in {t:?}", c["ann"]), &c["apply"], &json!(review::apply(t, &c["ann"], s(&c["replacement"]))));
    }
    for c in v["brief"].as_array().unwrap() {
        let events: Vec<Value> = c["events"].as_array().unwrap().clone();
        let got = host::brief(&events, s(&c["me"]), c["carrying_on"].as_bool().unwrap());
        d.eq(format!("brief of {} events for {}", events.len(), c["me"]), &c["brief"], &json!(got));
    }
    for c in v["model_choice"].as_array().unwrap() {
        d.eq(format!("model_choice {}", c["setup"]), &c["choice"], &json!(host::model_choice(&c["setup"])));
    }
    for c in v["turn_usage"].as_array().unwrap() {
        d.eq(format!("turn_usage {}", c["result"]), &c["usage"], &Value::Object(host::turn_usage(&c["result"])));
    }
    for c in v["pricing"].as_array().unwrap() {
        let reqs: Vec<pricing::RequestUsage> = c["usages"].as_array().unwrap().iter()
            .map(|u| pricing::request_usage(u[1].as_object().unwrap(), s(&u[0]))).collect();
        let got: Vec<Value> = reqs.iter().map(|r| json!([r.model, r.input, r.write_5m, r.write_1h, r.read, r.output])).collect();
        d.eq(format!("request_usage {}", c["usages"]), &c["requests"], &json!(got));
        d.eq(format!("task_cost {}", c["usages"]), &c["cost"], &pricing::task_cost(&reqs));
    }
    for c in v["shlex"].as_array().unwrap() {
        d.eq(format!("shlex {:?}", c["text"]), &c["words"], &json!(host::shlex_split(s(&c["text"])).ok()));
    }
    for c in v["outside"].as_array().unwrap() {
        d.eq(format!("outside_paths {:?}", c["command"]), &c["found"], &json!(host::outside_paths(s(&c["command"]), Path::new(s(&c["folder"])))));
    }
    for c in v["inside"].as_array().unwrap() {
        d.eq(format!("inside {:?}", c["path"]), &c["inside"], &json!(host::inside(Path::new(s(&c["folder"])), s(&c["path"]))));
    }
    let st = &v["settings"];
    let folder = Path::new(s(&st["folder"]));
    let mut kinds = host::subagent_mcp_tools();
    kinds.sort();
    d.eq("subagent tools", &st["kinds"], &json!(kinds));
    for c in st["cases"].as_array().unwrap() {
        let guard: Vec<String> = c["guard"].as_array().unwrap().iter().map(|g| s(g).to_string()).collect();
        let log = Path::new(s(&st["log"]));
        let settings = host::claude_settings(folder, c["confine"].as_bool().unwrap(), &guard, (!guard.is_empty()).then_some(log));
        d.eq(format!("claude_settings {c}"), &c["settings"], &Value::Object(settings.clone()));
        let meta = (!c["meta"].is_null()).then_some(&c["meta"]);
        let merged = match host::with_claude_settings(meta, &settings, folder) {
            Ok(m) => m,
            Err(e) if e.starts_with("[Errno") => json!({"error": "FileNotFoundError"}),
            Err(_) => json!({"error": "ValueError"}),
        };
        d.eq(format!("with_claude_settings {}", c["meta"]), &c["merged"], &merged);
        d.eq(format!("with_subagents {}", c["meta"]), &c["subagents"], &host::with_subagents(meta));
    }
    for c in v["profile"].as_array().unwrap() {
        let extra: Vec<String> = c["extra"].as_array().unwrap().iter().map(|e| s(e).to_string()).collect();
        let got = confine::profile(s(&c["folder"]), s(&c["codex"]), s(&c["home"]), &extra, c["auth"].as_str());
        d.eq(format!("profile {c}"), &c["profile"], &json!(got));
    }
    for c in v["prompt"].as_array().unwrap() {
        d.eq(format!("prompt {:?}", c["path"]), &c["prompt"], &json!(review::prompt(s(&c["path"]), s(&c["source"]))));
    }
    for c in v["text_source"].as_array().unwrap() {
        let p = s(&c["path"]);
        d.eq(format!("text_source {p:?}"), &json!([c["source"], c["writable"]]), &json!([review::text_source(p), review::writable(p)]));
    }
    for c in v["find_quotes"].as_array().unwrap() {
        let seeds = deidkit::seeds::InMemorySeedSource::default();
        for e in c["seeds"].as_array().unwrap() {
            seeds.add_entity("q", deidkit::seeds::SeedEntity {
                r#type: s(&e[0]).into(), name: s(&e[1]).into(), identifier: String::new(), identifier_type: String::new(),
                jurisdiction_country: String::new(), role: String::new(), attributes: None, relationships: None, id: Value::Null });
        }
        let engine = deidkit::proxy::engine::ScopeEngine::new(
            std::rc::Rc::new(deidkit::store::InMemoryTokenStore::new()), "q", Some(std::rc::Rc::new(seeds)),
            Some(std::rc::Rc::new(deidkit::patterns::RegexDetector::default())), None);
        let quotes: Vec<String> = c["quotes"].as_array().unwrap().iter().map(|q| s(q).to_string()).collect();
        let got: Vec<Value> = review::find_quotes(s(&c["text"]), &quotes, Some(&engine)).into_iter()
            .map(|f| f.map_or(Value::Null, |(a, b)| json!([a, b]))).collect();
        d.eq(format!("find_quotes {:?} in {:?}", c["quotes"], c["text"]), &c["found"], &json!(got));
    }
    for c in v["view"].as_array().unwrap() {
        let folder = Path::new(s(&c["folder"]));
        d.eq(format!("view {}", c["folder"]), &c["view"], &review::view(folder, s(&c["path"]), s(&c["text"])));
        d.eq(format!("records {}", c["folder"]), &c["records"], &json!(review::records(folder)));
    }
    let _ = Map::<String, Value>::new();
    report(d);
}
