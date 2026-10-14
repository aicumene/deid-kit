// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What the vault knows about a scope before it reads a word of text: its entities and parties (as
//! `seeds.py`), and the TOML seed files kept beside the files (as `seedfile.py`).

use std::cell::RefCell;
use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::py;

/// A known referent of a scope.
#[derive(Clone, Debug, Default)]
pub struct SeedEntity {
    pub r#type: String,
    pub name: String,
    pub identifier: String,
    pub identifier_type: String,
    pub jurisdiction_country: String,
    pub role: String,
    pub attributes: Option<Map<String, Value>>,
    pub relationships: Option<Vec<Value>>,
    pub id: Value,
}

/// Where the vault reads a scope's entities and parties from.
pub trait SeedSource {
    fn entities(&self, scope: &str) -> Vec<SeedEntity>;
    /// The scope's parties as `[{"name", "role"}]`, or None.
    fn parties(&self, scope: &str) -> Option<Vec<Value>>;
}

#[derive(Default)]
pub struct InMemorySeedSource {
    pub entities: RefCell<HashMap<String, Vec<SeedEntity>>>,
    pub parties: RefCell<HashMap<String, Option<Vec<Value>>>>,
}

impl InMemorySeedSource {
    pub fn add_entity(&self, scope: &str, e: SeedEntity) {
        self.entities.borrow_mut().entry(scope.to_string()).or_default().push(e);
    }

    pub fn set_parties(&self, scope: &str, parties: Option<Vec<Value>>) {
        self.parties.borrow_mut().insert(scope.to_string(), parties);
    }
}

impl SeedSource for InMemorySeedSource {
    fn entities(&self, scope: &str) -> Vec<SeedEntity> {
        self.entities.borrow().get(scope).cloned().unwrap_or_default()
    }

    fn parties(&self, scope: &str) -> Option<Vec<Value>> {
        self.parties.borrow().get(scope).cloned().flatten()
    }
}

fn toml_json(v: &toml::Value) -> Value {
    match v {
        toml::Value::String(s) => json!(s),
        toml::Value::Integer(i) => json!(i),
        toml::Value::Float(f) => json!(f),
        toml::Value::Boolean(b) => json!(b),
        toml::Value::Datetime(d) => json!(d.to_string()),
        toml::Value::Array(a) => Value::Array(a.iter().map(toml_json).collect()),
        toml::Value::Table(t) => Value::Object(t.iter().map(|(k, v)| (k.clone(), toml_json(v))).collect()),
    }
}

fn read_toml(path: &str) -> Result<Map<String, Value>, String> {
    let text = std::fs::read_to_string(py::expanduser(path)).map_err(|e| format!("{path}: {e}"))?;
    let table: toml::Table = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok(table.iter().map(|(k, v)| (k.clone(), toml_json(v))).collect())
}

fn scope_of(data: &Map<String, Value>, default_scope: &str) -> String {
    data.get("scope").filter(|v| py::truthy(Some(v))).map_or(default_scope.to_string(), py::py_str)
}

/// Read seed files into one in-memory seed source, keyed by scope.
pub fn load_seed_files(paths: &[String], default_scope: &str, into: InMemorySeedSource) -> Result<InMemorySeedSource, String> {
    for path in paths {
        let data = read_toml(path)?;
        let scope = scope_of(&data, default_scope);
        for e in data.get("entity").and_then(Value::as_array).into_iter().flatten() {
            let get = |k: &str| e.get(k).map_or(String::new(), py::py_str);
            if !py::truthy(e.get("name")) || !py::truthy(e.get("type")) {
                return Err(format!("{path}: every [[entity]] needs a type and a name"));
            }
            into.add_entity(&scope, SeedEntity {
                r#type: get("type"), name: get("name"), identifier: get("identifier"), identifier_type: get("identifier_type"),
                jurisdiction_country: get("jurisdiction_country"), role: get("role"),
                attributes: e.get("attributes").and_then(Value::as_object).cloned(), relationships: None, id: Value::Null,
            });
        }
        if let Some(parties) = data.get("party").and_then(Value::as_array).filter(|p| !p.is_empty()) {
            let mut list = Vec::new();
            for p in parties {
                let name = p.get("name").ok_or_else(|| format!("{path}: 'name'"))?;
                list.push(json!({"name": py::py_str(name), "role": p.get("role").map_or(String::new(), py::py_str)}));
            }
            into.set_parties(&scope, Some(list));
        }
    }
    Ok(into)
}

/// `(folder, scope)` pairs from the seed files' `paths` lists, folders resolved, longest first.
pub fn load_scope_paths(paths: &[String], default_scope: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for path in paths {
        let data = read_toml(path)?;
        let scope = scope_of(&data, default_scope);
        for folder in data.get("paths").and_then(Value::as_array).into_iter().flatten() {
            let real = py::realpath(&py::expanduser(&py::py_str(folder)));
            out.push((real.to_string_lossy().into_owned(), scope.clone()));
        }
    }
    out.sort_by_key(|(f, _)| std::cmp::Reverse(py::clen(f)));
    Ok(out)
}
