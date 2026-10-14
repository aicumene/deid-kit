// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The vault gives what deidkit's does, step by step on the same scopes with the same salt: seeding,
//! tokenising a corpus with the regex detector, merging mappings, detokenising answers, the residual
//! check, every row and alias after each step; and the detectors and address tokens apart.
//! Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_vault.py <dir>`.

mod common;

use std::collections::HashSet;

use common::Diff;
use deidkit::odict::ODict;
use deidkit::patterns::{self, RegexDetector};
use deidkit::seeds::{InMemorySeedSource, SeedEntity, SeedSource};
use deidkit::store::{InMemoryTokenStore, TokenStore};
use deidkit::{address, detect::PiiDetector, vault};
use serde_json::{json, Value};

fn dump(store: &InMemoryTokenStore, scope: &str) -> Value {
    let tokens: Vec<Value> = store.load_tokens(scope).iter().map(|r| {
        let r = r.borrow();
        json!([r.token, r.kind, r.real_value, r.normalized, r.status, r.canonical_token, r.attributes])
    }).collect();
    let aliases: Vec<Value> = store.load_aliases(scope).iter().map(|a| {
        let a = a.borrow();
        json!([a.token, a.surface, a.normalized, a.source])
    }).collect();
    json!({"tokens": tokens, "aliases": aliases})
}

fn pairs<V: serde::Serialize>(d: &ODict<V>) -> Value {
    Value::Array(d.iter().map(|(k, v)| json!([k, v])).collect())
}

#[test]
fn the_vault_is_pythons() {
    let Some((_, v)) = common::vectors("vault.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let mut d = Diff::new();
    let salt: Vec<u8> = v["salt"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as u8).collect();
    let store = InMemoryTokenStore::new();
    let seeds = InMemorySeedSource::default();
    for (scope, ents) in v["entities"].as_object().unwrap() {
        for e in ents.as_array().unwrap() {
            let s = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            seeds.add_entity(scope, SeedEntity { r#type: s("type"), name: s("name"), identifier: s("identifier"),
                identifier_type: s("identifier_type"), jurisdiction_country: s("jurisdiction_country"), role: s("role"),
                attributes: e.get("attributes").and_then(Value::as_object).cloned(), relationships: None, id: Value::Null });
        }
    }
    for (scope, parties) in v["parties"].as_object().unwrap() {
        seeds.set_parties(scope, parties.as_array().cloned());
    }
    let det = RegexDetector::default();
    let mut reds: std::collections::HashMap<String, Vec<deidkit::gateway::Redaction>> = Default::default();
    for (i, step) in v["steps"].as_array().unwrap().iter().enumerate() {
        let scope = step["scope"].as_str().unwrap();
        match step["op"].as_str().unwrap() {
            "seed" => {
                store.add_salt(scope, salt.clone());
                vault::seed_from_scope(&store, scope, Some(&seeds as &dyn SeedSource)).unwrap();
            }
            "tokenize" => {
                let text = step["text"].as_str().unwrap();
                let red = vault::tokenize(&store, scope, text, None, Some(&det as &dyn PiiDetector), None, false).unwrap();
                let got = json!({"text": red.text, "entity_types": red.entity_types, "redacted_count": red.redacted_count,
                                 "mapping": pairs(&red.mapping), "surfaces": pairs(&red.surfaces), "canonical": pairs(&red.canonical)});
                d.eq(format!("step {i} tokenize {text:?}"), &step["red"], &got);
                let list = reds.entry(scope.to_string()).or_default();
                list.push(red);
                let merged = vault::merge_mappings(list);
                d.eq(format!("step {i} merged"), &step["merged"], &pairs(&merged));
                let back = vault::detokenize(&store, scope, step["answer"].as_str().unwrap(), Some(&merged));
                d.eq(format!("step {i} detokenize"), &step["back"], &json!(back));
                d.eq(format!("step {i} residual"), &step["residual"], &json!(vault::residual_surfaces(&store, scope, text)));
            }
            "detokenize_all" => {
                let tokens: Vec<String> = store.load_tokens(scope).iter().map(|r| r.borrow().token.clone()).collect();
                let text = format!("PERSON_10000001 and {}", tokens.join(" "));
                d.eq(format!("step {i} detokenize all"), &step["out"], &json!(vault::detokenize(&store, scope, &text, None)));
                continue;
            }
            _ => unreachable!(),
        }
        d.eq(format!("step {i} {} state", step["op"]), &step["state"], &dump(&store, scope));
    }
    for x in v["detect"].as_array().unwrap() {
        let got: Vec<Value> = det.detect(x[0].as_str().unwrap(), "en").into_iter().map(|(s, k)| json!([s, k])).collect();
        d.eq(format!("detect {:?}", x[0]), &x[1], &Value::Array(got));
    }
    for x in v["checks"]["iban"].as_array().unwrap() {
        d.eq(format!("iban {}", x[0]), &x[1], &json!(patterns::iban_ok(x[0].as_str().unwrap())));
    }
    for x in v["checks"]["card"].as_array().unwrap() {
        d.eq(format!("card {}", x[0]), &x[1], &json!(patterns::card_ok(x[0].as_str().unwrap())));
    }
    let mut known: ODict<String> = ODict::new();
    for x in v["addresses"].as_array().unwrap() {
        let taken: HashSet<String> = known.iter().map(|(_, t)| t.clone()).collect();
        let derive = |s: &str, taken: &HashSet<String>| vault::derive_token(&salt, "ADDRESS", s, taken);
        let r = address::tokenize_addresses(x[0].as_str().unwrap(), &derive, &taken, &mut known);
        let found: Vec<Value> = r.found.iter().map(|(n, s)| json!([n, s])).collect();
        d.eq(format!("address {:?}", x[0]), &json!([x[0], x[1], x[2], x[3]]), &json!([x[0], r.text, found, pairs(&r.mapping)]));
    }
    for x in v["initials"].as_array().unwrap() {
        d.eq(format!("initials {}", x[0]), &x[1], &json!(address::initials_of(x[0].as_str().unwrap())));
    }
    d.done();
}
