// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A map in insertion order, as Python's dict keeps it: setting a key that is there keeps its place,
//! removing one and setting it again puts it last. The vault's outcomes depend on that order (which
//! key is tried first among equals, the order of a request's mapping).

use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct ODict<V> {
    keys: Vec<Option<String>>,
    vals: Vec<Option<V>>,
    at: HashMap<String, usize>,
    live: usize,
}

impl<V> ODict<V> {
    pub fn new() -> Self {
        ODict { keys: Vec::new(), vals: Vec::new(), at: HashMap::new(), live: 0 }
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn get(&self, k: &str) -> Option<&V> {
        self.at.get(k).and_then(|&i| self.vals[i].as_ref())
    }

    pub fn get_mut(&mut self, k: &str) -> Option<&mut V> {
        match self.at.get(k) {
            Some(&i) => self.vals[i].as_mut(),
            None => None,
        }
    }

    pub fn contains_key(&self, k: &str) -> bool {
        self.at.contains_key(k)
    }

    /// `d[k] = v`.
    pub fn insert(&mut self, k: &str, v: V) {
        match self.at.get(k) {
            Some(&i) => self.vals[i] = Some(v),
            None => {
                self.at.insert(k.to_string(), self.keys.len());
                self.keys.push(Some(k.to_string()));
                self.vals.push(Some(v));
                self.live += 1;
            }
        }
    }

    /// `d.pop(k, None)`.
    pub fn remove(&mut self, k: &str) -> Option<V> {
        let i = self.at.remove(k)?;
        self.keys[i] = None;
        self.live -= 1;
        let v = self.vals[i].take();
        if self.keys.len() > 64 && self.live * 2 < self.keys.len() {
            self.compact();
        }
        v
    }

    fn compact(&mut self) {
        let keys = std::mem::take(&mut self.keys);
        let vals = std::mem::take(&mut self.vals);
        self.at.clear();
        for (k, v) in keys.into_iter().zip(vals) {
            if let (Some(k), Some(v)) = (k, v) {
                self.at.insert(k.clone(), self.keys.len());
                self.keys.push(Some(k));
                self.vals.push(Some(v));
            }
        }
    }

    /// `d.setdefault(k, v)`: the value now under `k`.
    pub fn setdefault(&mut self, k: &str, v: V) -> &mut V {
        if !self.at.contains_key(k) {
            self.insert(k, v);
        }
        self.get_mut(k).unwrap()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &V)> {
        self.keys.iter().zip(&self.vals).filter_map(|(k, v)| Some((k.as_deref()?, v.as_ref()?)))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|(k, _)| k)
    }
}

impl<V: Clone> ODict<V> {
    /// `{**self, **other}`.
    pub fn merged(&self, other: &ODict<V>) -> ODict<V> {
        let mut out = self.clone();
        for (k, v) in other.iter() {
            out.insert(k, v.clone());
        }
        out
    }
}
