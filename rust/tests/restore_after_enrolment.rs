// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The model's own words go back as it wrote them, unless they carry a name hidden since: a name the model
//! wrote in clear before the person hid it must not cross in clear again on the turns after.

use std::rc::Rc;

use deidkit::odict::ODict;
use deidkit::patterns::RegexDetector;
use deidkit::proxy::anthropic::{self as a, key};
use deidkit::proxy::engine::ScopeEngine;
use deidkit::proxy::openai as o;
use deidkit::seeds::{InMemorySeedSource, SeedEntity};
use deidkit::store::{InMemoryTokenStore, TokenStore};
use deidkit::vault;
use serde_json::{json, Value};

const PERSON: &str = "Jane Holloway";
const FIRM: &str = "Brightwater Maritime Ltd";

struct Matter {
    store: Rc<InMemoryTokenStore>,
    seeds: Rc<InMemorySeedSource>,
    engine: ScopeEngine,
}

fn matter() -> Matter {
    let store = Rc::new(InMemoryTokenStore::new());
    let seeds = Rc::new(InMemorySeedSource::default());
    seeds.add_entity("s", SeedEntity { r#type: "ORGANIZATION".into(), name: FIRM.into(), ..Default::default() });
    let engine = ScopeEngine::new(store.clone(), "s", Some(seeds.clone()), Some(Rc::new(RegexDetector::default())), None);
    Matter { store, seeds, engine }
}

impl Matter {
    fn token(&self, name: &str) -> String {
        self.engine.tokenize_all(&[name.to_string()]).unwrap()[0].text.clone()
    }

    /// The person hides `name` while the matter runs.
    fn hide(&self, name: &str) {
        self.seeds.add_entity("s", SeedEntity { r#type: "PERSON".into(), name: name.into(), ..Default::default() });
        vault::seed_from_scope(&*self.store, "s", Some(&*self.seeds)).unwrap();
        self.store.commit().unwrap();
    }
}

fn messages(text: &str, control: &str) -> Value {
    json!({"model": "m", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Who do we write to?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": text}, {"type": "text", "text": control}]},
        {"role": "user", "content": [{"type": "text", "text": "Go on."}]},
    ]})
}

#[test]
fn a_name_hidden_after_the_model_wrote_it_is_not_restored_in_clear() {
    let m = matter();
    let firm = m.token(FIRM);
    // what the model wrote (in tokens, the person's name in clear: it was not hidden then), and what the
    // chat shows and sends back; the control carries a mark the fresh tokens would not have
    let wrote = format!("I will write to {PERSON} at {firm}.");
    let shown = format!("I will write to {PERSON} at {FIRM}.");
    let control_wrote = format!("Sent to {firm}. \u{2713}");
    let control_shown = format!("Sent to {FIRM}.");
    let mut table = ODict::new();
    table.insert(&key("t:", &shown), wrote.clone());
    table.insert(&key("t:", &control_shown), control_wrote.clone());
    let get = |k: &str| table.get(k).cloned();
    let body = messages(&shown, &control_shown);

    let before = a::prepare_request(&body, &m.engine, &get, "").ok().unwrap();
    let sent = serde_json::to_string(&before.body).unwrap();
    assert_eq!(before.restored, 2, "both turns go back as the model wrote them");
    assert!(sent.contains(&serde_json::to_string(&wrote).unwrap()), "{sent}");

    m.hide(PERSON);
    let after = a::prepare_request(&body, &m.engine, &get, "").ok().unwrap();
    let sent = serde_json::to_string(&after.body).unwrap();
    assert!(!sent.contains(PERSON), "the hidden name crossed in clear: {sent}");
    assert!(sent.contains(&m.token(PERSON)), "{sent}");
    assert_eq!(after.restored, 1, "only the turn without the name is restored");
    assert!(sent.contains(&serde_json::to_string(&control_wrote).unwrap()), "the control lost its bytes: {sent}");
    assert!(after.mapping.get(&m.token(PERSON)).is_some_and(|v| v == PERSON), "the answer can name the person");
}

#[test]
fn a_tool_call_written_with_the_name_is_tokenised_once_it_is_hidden() {
    let m = matter();
    let args = format!("{{\"to\":\"{PERSON}\",\"subject\":\"Charter\"}}");
    let input = format!("{PERSON}, Charter");
    let mut table = ODict::new();
    table.insert(&key("f:", &args), args.clone());
    table.insert(&key("c:", &input), input.clone());
    let get = |k: &str| table.get(k).cloned();
    let body = json!({"model": "m", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Write the letter."}]},
        {"type": "function_call", "name": "write", "call_id": "c1", "arguments": args},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
        {"type": "custom_tool_call", "name": "note", "call_id": "c2", "input": input},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": "done"},
    ]});

    let before = o::prepare_responses(&body, &m.engine, &[], &get, "", &Default::default()).ok().unwrap();
    assert_eq!(before.restored, 2);
    assert!(serde_json::to_string(&before.body).unwrap().contains(PERSON), "control: the name was not hidden yet");

    m.hide(PERSON);
    let after = o::prepare_responses(&body, &m.engine, &[], &get, "", &Default::default()).ok().unwrap();
    let sent = serde_json::to_string(&after.body).unwrap();
    assert!(!sent.contains(PERSON), "the hidden name crossed in clear: {sent}");
    assert_eq!(after.restored, 0);
    let call = &after.body["input"][1]["arguments"];
    let parsed: Value = serde_json::from_str(call.as_str().unwrap()).expect("the arguments are still JSON");
    assert_eq!(parsed["to"], json!(m.token(PERSON)));
    assert_eq!(parsed["subject"], json!("Charter"));
    assert!(after.mapping.get(&m.token(PERSON)).is_some_and(|v| v == PERSON));
}
