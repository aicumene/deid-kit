// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The proxy's de-identification gives what deidkit.proxy's does: Messages and Responses requests out
//! (tokens, withheld parts, refusals, the model's own words restored), event streams back (held-back
//! tails, local tools restored whole, renumbered events), whole answers, the spellings table and the
//! restore table. Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_proxy.py <dir>`.

mod common;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use common::Diff;
use deidkit::odict::ODict;
use deidkit::patterns::RegexDetector;
use deidkit::proxy::anthropic::{self as a, PrepareError, StreamRestorer};
use deidkit::proxy::engine::ScopeEngine;
use deidkit::proxy::openai::{self as o, ResponsesRestorer};
use deidkit::seeds::{InMemorySeedSource, SeedEntity};
use deidkit::store::{InMemoryTokenStore, TokenStore};
use serde_json::{json, Value};

fn pairs(d: &ODict<String>) -> Value {
    Value::Array(d.iter().map(|(k, v)| json!([k, v])).collect())
}

fn events(list: Vec<(String, Value)>) -> Value {
    Value::Array(list.into_iter().map(|(n, d)| json!([n, d])).collect())
}

#[test]
fn the_proxy_is_pythons() {
    let Some((_, v)) = common::vectors("proxy.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let mut d = Diff::new();
    let store = Rc::new(InMemoryTokenStore::new());
    store.add_salt("s", v["salt"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8).collect());
    let seeds = InMemorySeedSource::default();
    for e in v["entities"].as_array().unwrap() {
        seeds.add_entity("s", SeedEntity { r#type: e["type"].as_str().unwrap().into(), name: e["name"].as_str().unwrap().into(), ..Default::default() });
    }
    let engine = ScopeEngine::new(store.clone(), "s", Some(Rc::new(seeds)), Some(Rc::new(RegexDetector::default())), None);
    let restore: Rc<RefCell<ODict<String>>> = Rc::new(RefCell::new(ODict::new()));
    let r1 = restore.clone();
    let get = move |k: &str| r1.borrow().get(k).cloned();
    let put_into = |r: Rc<RefCell<ODict<String>>>| move |k: &str, val: &str| r.borrow_mut().insert(k, val.to_string());
    let msgs = v["messages"].as_array().unwrap();
    let check_msg = |d: &mut Diff, i: usize, m: &Value| {
        match a::prepare_request(&m["in"], &engine, &get, "withhold") {
            Ok(p) => {
                let got = json!({"in": m["in"], "body": p.body, "mapping": pairs(&p.mapping), "withheld": p.withheld, "restored": p.restored,
                                 "spellings": pairs(&p.spellings)});
                d.eq(format!("messages {i}"), m, &got);
            }
            Err(PrepareError::Refused(r)) => d.eq(format!("messages {i}"), m, &json!({"in": m["in"], "refused": r})),
            Err(PrepareError::Failed(f)) => d.eq(format!("messages {i}"), m, &json!({"in": m["in"], "failed": f})),
        }
    };
    for (i, m) in msgs.iter().take(4).enumerate() {
        check_msg(&mut d, i, m);
    }
    let local = a::local_tools();
    let stream = &v["streams"][0];
    let p = match a::prepare_request(&stream["body"], &engine, &get, "withhold") { Ok(p) => p, Err(_) => panic!("the stream's request") };
    let mut restorer = StreamRestorer::new(&engine, p.mapping.clone(), Box::new(put_into(restore.clone())), local.clone(), p.spellings.clone());
    for (i, e) in stream["events"].as_array().unwrap().iter().enumerate() {
        let got = restorer.event(e[0].as_str().unwrap(), e[1].clone()).unwrap();
        d.eq(format!("stream event {i} {}", e[0]), &e[2], &events(got));
    }
    let whole = a::restore_json_response(v["whole"]["in"].clone(), &engine, &p.mapping, &put_into(restore.clone()), &local, Some(&p.spellings));
    d.eq("whole answer", &v["whole"]["out"], &whole);
    check_msg(&mut d, 4, &msgs[4]);
    let headers: Vec<(String, String)> = v["responses"][0]["headers"].as_array().unwrap().iter()
        .map(|h| (h[0].as_str().unwrap().to_string(), h[1].as_str().unwrap().to_string())).collect();
    let drop: HashSet<String> = ["collaboration".to_string()].into();
    for (i, r) in v["responses"].as_array().unwrap().iter().enumerate() {
        match o::prepare_responses(&r["in"], &engine, &headers, &get, "withhold", &drop) {
            Ok(p) => {
                let oh = Value::Array(p.headers.iter().map(|(k, v)| json!([k, v])).collect());
                let got = json!({"in": r["in"], "headers": r["headers"], "body": p.body, "mapping": pairs(&p.mapping), "out_headers": oh,
                                 "withheld": p.withheld, "restored": p.restored, "spellings": pairs(&p.spellings)});
                d.eq(format!("responses {i}"), r, &got);
            }
            Err(PrepareError::Refused(x)) => d.eq(format!("responses {i}"), r, &json!({"in": r["in"], "headers": r["headers"], "refused": x})),
            Err(PrepareError::Failed(f)) => d.eq(format!("responses {i}"), r, &json!({"failed": f})),
        }
    }
    let rs = &v["rstreams"][0];
    let p = match o::prepare_responses(&rs["body"], &engine, &headers, &get, "withhold", &HashSet::new()) { Ok(p) => p, Err(_) => panic!("the stream's request") };
    let mut restorer = ResponsesRestorer::new(&engine, p.mapping.clone(), Box::new(put_into(restore.clone())), o::local_tools(), p.spellings.clone());
    for (i, e) in rs["events"].as_array().unwrap().iter().enumerate() {
        let got = restorer.event(e[0].as_str().unwrap(), e[1].clone()).unwrap();
        d.eq(format!("responses event {i} {}", e[0]), &e[2], &events(got));
    }
    d.eq("restore table", &v["restore"], &pairs(&restore.borrow()));
    d.done();
}
