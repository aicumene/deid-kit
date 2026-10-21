// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! One scope's de-identification for the proxy: many texts in, one consistent request out (as
//! `engine.py`). Every piece is tokenised against the same vault until it stops learning; a piece is
//! cached by its SHA-256 and the vault's size, so the history is sent as the same bytes every turn; one
//! mapping per request; street addresses cross as tokens.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

use sha2::{Digest, Sha256};

use crate::address;
use crate::detect::PiiDetector;
use crate::gateway::Redaction;
use crate::namefold as nf;
use crate::odict::ODict;
use crate::proxy::spelling::{self, respell};
use crate::py;
use crate::seeds::SeedSource;
use crate::store::TokenStore;
use crate::vault;

const CACHE_SIZE: usize = 50_000;

#[derive(Debug)]
pub struct VaultDidNotSettle(pub String);

pub struct ScopeEngine {
    pub store: Rc<dyn TokenStore>,
    pub scope: String,
    pub seeds: Option<Rc<dyn SeedSource>>,
    pub detector: Option<Rc<dyn PiiDetector>>,
    pub language: Option<String>,
    cache: RefCell<ODict<Redaction>>,
    ready: Cell<bool>,
    salt: RefCell<Option<Vec<u8>>>,
    /// Street address (folded) → its token, for the engine's lifetime.
    addresses: RefCell<ODict<String>>,
}

impl ScopeEngine {
    pub fn new(store: Rc<dyn TokenStore>, scope: &str, seeds: Option<Rc<dyn SeedSource>>, detector: Option<Rc<dyn PiiDetector>>,
               language: Option<String>) -> ScopeEngine {
        ScopeEngine { store, scope: scope.into(), seeds, detector, language, cache: RefCell::new(ODict::new()),
                      ready: Cell::new(false), salt: RefCell::new(None), addresses: RefCell::new(ODict::new()) }
    }

    fn prepare(&self) -> Result<(), String> {
        if !self.ready.get() {
            *self.salt.borrow_mut() = Some(vault::ensure_salt(&*self.store, &self.scope));
            vault::seed_from_scope(&*self.store, &self.scope, self.seeds.as_deref())?;
            self.store.commit()?;
            self.ready.set(true);
        }
        Ok(())
    }

    /// `red` with its street addresses replaced by their tokens, the surfaces recorded in text order.
    fn with_addresses(&self, red: Redaction) -> Redaction {
        let Some(salt) = self.salt.borrow().clone().filter(|s| !s.is_empty()) else { return red };
        let derive = |surface: &str, taken: &HashSet<String>| vault::derive_token(&salt, "ADDRESS", surface, taken);
        let taken: HashSet<String> = self.addresses.borrow().values().cloned().collect();
        let found = address::tokenize_addresses(&red.text, &derive, &taken, &mut self.addresses.borrow_mut());
        if found.found.is_empty() {
            return red;
        }
        let mut out = red;
        for (_, surface) in &found.found {
            let tok = self.addresses.borrow().get(&address::address_key(surface)).cloned().unwrap();
            out.surfaces.setdefault(&tok, Vec::new()).push(surface.clone());
            out.mapping.setdefault(&tok, surface.clone());
            out.canonical.setdefault(&tok, surface.clone());
        }
        out.text = found.text;
        let mut kinds: Vec<String> = out.entity_types.clone();
        kinds.push("ADDRESS".into());
        kinds.sort();
        kinds.dedup();
        out.entity_types = kinds;
        out.redacted_count += found.found.len();
        out
    }

    fn version(&self) -> (usize, usize) {
        (self.store.load_tokens(&self.scope).len(), self.store.load_aliases(&self.scope).len())
    }

    fn one(&self, text: &str) -> Result<Redaction, String> {
        if py::strip(text).is_empty() {
            return Ok(Redaction::of(text));
        }
        let digest = hex::encode(Sha256::digest(text.as_bytes()));
        let (a, b) = self.version();
        let key = format!("{digest}:{a}:{b}");
        let hit = self.cache.borrow_mut().remove(&key);
        if let Some(hit) = hit {
            self.cache.borrow_mut().insert(&key, hit.clone());
            return Ok(hit);
        }
        let red = vault::tokenize(&*self.store, &self.scope, text, None, self.detector.as_deref(), self.language.as_deref(), false)?;
        let red = self.with_addresses(red);
        let (a, b) = self.version();
        let mut cache = self.cache.borrow_mut();
        cache.insert(&format!("{digest}:{a}:{b}"), red.clone());
        while cache.len() > CACHE_SIZE {
            let first = cache.first_key().unwrap();
            cache.remove(&first);
        }
        Ok(red)
    }

    /// Tokenise every piece of one request against one settled vault.
    pub fn tokenize_all(&self, texts: &[String]) -> Result<Vec<Redaction>, String> {
        self.prepare()?;
        for _ in 0..4 {
            let before = self.version();
            let reds = texts.iter().map(|t| self.one(t)).collect::<Result<Vec<_>, _>>()?;
            if self.version() == before {
                return Ok(reds);
            }
        }
        Err(format!("VaultDidNotSettle: scope {}: the vault kept learning", py::repr_str(&self.scope)))
    }

    /// The request's mapping: the merged per-piece mappings, plus every other token of the scope that
    /// appears in what is sent.
    pub fn mapping_for(&self, reds: &[Redaction], sent: &str) -> ODict<String> {
        let mut mapping = vault::merge_mappings(reds);
        let probe: Vec<char> = nf::deconfuse_ascii(sent).chars().collect();
        let mut seen: Vec<String> = spelling::token().finditer(&probe).iter()
            .map(|m| probe[m.start..m.end].iter().collect::<String>()).filter(|t| !mapping.contains_key(t)).collect();
        seen.sort();
        seen.dedup();
        if seen.is_empty() {
            return mapping;
        }
        let rows: std::collections::HashMap<String, _> = self.store.load_tokens(&self.scope).into_iter()
            .map(|r| {
                let t = r.borrow().token.clone();
                (t, r)
            }).collect();
        for tok in seen {
            let Some(mut row) = rows.get(&tok).cloned() else { continue };
            let mut visited = HashSet::new();
            loop {
                let next = row.borrow().canonical_token.clone().filter(|c| !c.is_empty());
                match next.and_then(|c| rows.get(&c).cloned()) {
                    Some(n) if !std::rc::Rc::ptr_eq(&n, &row) && visited.insert(n.borrow().token.clone()) => row = n,
                    _ => break,
                }
            }
            mapping.insert(&tok, row.borrow().real_value.clone());
        }
        mapping
    }

    /// Tokens back to names; with `spellings`, a file or folder name seen in the request comes back as
    /// written.
    pub fn detokenize(&self, text: &str, mapping: &ODict<String>, spellings: Option<&ODict<String>>, literal: bool, prose: bool) -> String {
        let text = match spellings.filter(|s| !s.is_empty()) {
            Some(table) => respell(text, table, literal, prose),
            None => text.to_string(),
        };
        vault::detokenize(&*self.store, &self.scope, &text, Some(mapping))
    }

    /// Enrolled real values still present in `text` (read-only).
    pub fn residual(&self, text: &str) -> Vec<String> {
        vault::residual_surfaces(&*self.store, &self.scope, text)
    }

    /// Whether `text` still carries something a request of this scope would hide: an enrolled value or a
    /// street address (read-only).
    pub fn shows(&self, text: &str) -> bool {
        !self.residual(text).is_empty() || !address::find_addresses(&text.chars().collect::<Vec<_>>()).is_empty()
    }
}
