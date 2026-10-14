// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Where the vault keeps its tokens, aliases and per-scope salts (as `store.py` and `sqlite_store.py`).
//! Loads return the LIVE rows: the vault edits a loaded row in place, and `commit()` saves the added
//! rows and the edits. Nothing is ever deleted: a defective token is retired. The SQLite store reads its
//! file once and rewrites each scope's rows at every commit, in the same schema as the Python store, so
//! a vault file either of them wrote is read by the other.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use rusqlite::{params, Connection};
use serde_json::{Map, Value};

use crate::py;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenRow {
    pub scope_id: String,
    pub token: String,
    pub kind: String,
    pub real_value: String,
    /// The old dedup key (a fold of the value). Nothing reads it.
    pub normalized: String,
    pub status: String,
    /// For a retired duplicate: the token that now carries this referent.
    pub canonical_token: Option<String>,
    /// Derived, non-identifying facts for the glossary.
    pub attributes: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AliasRow {
    pub scope_id: String,
    pub token: String,
    pub surface: String,
    /// The match key (`namefold::key`), unique per scope.
    pub normalized: String,
    /// entity | intake | fragment | variant | residual | legacy | retracted | defined
    pub source: String,
}

pub type TokenRef = Rc<RefCell<TokenRow>>;
pub type AliasRef = Rc<RefCell<AliasRow>>;

impl TokenRow {
    /// `row.attributes or {}`, read.
    pub fn attr(&self, key: &str) -> Option<&Value> {
        self.attributes.as_ref().and_then(|a| a.get(key))
    }
}

/// What the vault reads and writes.
pub trait TokenStore {
    fn load_tokens(&self, scope: &str) -> Vec<TokenRef>;
    fn load_aliases(&self, scope: &str) -> Vec<AliasRef>;
    fn load_salt(&self, scope: &str) -> Option<Vec<u8>>;
    #[allow(clippy::too_many_arguments)]
    fn add_token(&self, scope: &str, token: &str, kind: &str, real_value: &str, normalized: &str, status: &str,
                 attributes: Option<Map<String, Value>>) -> TokenRef;
    fn add_alias(&self, scope: &str, token: &str, surface: &str, normalized: &str, source: &str) -> AliasRef;
    fn add_salt(&self, scope: &str, salt: Vec<u8>);
    fn commit(&self) -> Result<(), String>;
    /// The proxy's restore table: the model's own words, keyed by a hash of what was handed over.
    fn restore_get(&self, _key: &str) -> Option<String> {
        None
    }
    fn restore_put(&self, _key: &str, _value: &str) {}
}

/// A token store in process memory.
#[derive(Default)]
pub struct InMemoryTokenStore {
    pub tokens: RefCell<HashMap<String, Vec<TokenRef>>>,
    pub aliases: RefCell<HashMap<String, Vec<AliasRef>>>,
    pub salts: RefCell<HashMap<String, Vec<u8>>>,
    /// The scopes in the order they were first written (the order commit rewrites them in).
    order: RefCell<Vec<String>>,
    /// How many times the vault committed.
    pub commits: Cell<usize>,
}

impl InMemoryTokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn touch(&self, scope: &str) {
        if !self.order.borrow().iter().any(|s| s == scope) {
            self.order.borrow_mut().push(scope.to_string());
        }
    }
}

impl TokenStore for InMemoryTokenStore {
    fn load_tokens(&self, scope: &str) -> Vec<TokenRef> {
        self.tokens.borrow().get(scope).cloned().unwrap_or_default()
    }

    fn load_aliases(&self, scope: &str) -> Vec<AliasRef> {
        self.aliases.borrow().get(scope).cloned().unwrap_or_default()
    }

    fn load_salt(&self, scope: &str) -> Option<Vec<u8>> {
        self.salts.borrow().get(scope).cloned()
    }

    fn add_token(&self, scope: &str, token: &str, kind: &str, real_value: &str, normalized: &str, status: &str,
                 attributes: Option<Map<String, Value>>) -> TokenRef {
        let row = Rc::new(RefCell::new(TokenRow { scope_id: scope.into(), token: token.into(), kind: kind.into(),
            real_value: real_value.into(), normalized: normalized.into(), status: status.into(), canonical_token: None, attributes }));
        self.touch(scope);
        self.tokens.borrow_mut().entry(scope.to_string()).or_default().push(row.clone());
        row
    }

    fn add_alias(&self, scope: &str, token: &str, surface: &str, normalized: &str, source: &str) -> AliasRef {
        let row = Rc::new(RefCell::new(AliasRow { scope_id: scope.into(), token: token.into(), surface: surface.into(),
            normalized: normalized.into(), source: source.into() }));
        self.touch(scope);
        self.aliases.borrow_mut().entry(scope.to_string()).or_default().push(row.clone());
        row
    }

    fn add_salt(&self, scope: &str, salt: Vec<u8>) {
        self.salts.borrow_mut().insert(scope.to_string(), salt);
    }

    fn commit(&self) -> Result<(), String> {
        self.commits.set(self.commits.get() + 1);
        Ok(())
    }
}

const SCHEMA: &str = "
create table if not exists tokens (
    scope text not null, seq integer not null, token text not null, kind text not null,
    real_value text not null, normalized text not null, status text not null,
    canonical_token text, attributes text,
    primary key (scope, seq));
create table if not exists aliases (
    scope text not null, seq integer not null, token text not null, surface text not null,
    normalized text not null, source text not null,
    primary key (scope, seq));
create table if not exists salts (scope text primary key, salt blob not null);
create table if not exists restore (key text primary key, value text not null, at real not null);
";

/// A token store persisted to one SQLite file, for one process on one machine. Created readable by its
/// owner only: it holds the real values and the salts.
pub struct SqliteTokenStore {
    pub path: PathBuf,
    mem: InMemoryTokenStore,
    db: RefCell<Connection>,
}

impl SqliteTokenStore {
    pub fn open(path: &Path) -> Result<SqliteTokenStore, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let fresh = !path.exists();
        let db = Connection::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if fresh {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        db.execute_batch(SCHEMA).map_err(|e| e.to_string())?;
        let store = SqliteTokenStore { path: path.to_path_buf(), mem: InMemoryTokenStore::new(), db: RefCell::new(db) };
        store.load().map_err(|e| e.to_string())?;
        Ok(store)
    }

    fn load(&self) -> rusqlite::Result<()> {
        let db = self.db.borrow();
        let mut q = db.prepare("select scope, token, kind, real_value, normalized, status, canonical_token, attributes \
                                from tokens order by scope, seq")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, TokenRow {
            scope_id: r.get(0)?, token: r.get(1)?, kind: r.get(2)?, real_value: r.get(3)?, normalized: r.get(4)?,
            status: r.get(5)?, canonical_token: r.get(6)?,
            attributes: r.get::<_, Option<String>>(7)?.filter(|a| !a.is_empty())
                .and_then(|a| serde_json::from_str::<Value>(&a).ok()).and_then(|v| v.as_object().cloned()),
        })))?;
        for row in rows {
            let (scope, row) = row?;
            self.mem.touch(&scope);
            self.mem.tokens.borrow_mut().entry(scope).or_default().push(Rc::new(RefCell::new(row)));
        }
        let mut q = db.prepare("select scope, token, surface, normalized, source from aliases order by scope, seq")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, AliasRow {
            scope_id: r.get(0)?, token: r.get(1)?, surface: r.get(2)?, normalized: r.get(3)?, source: r.get(4)?,
        })))?;
        for row in rows {
            let (scope, row) = row?;
            self.mem.touch(&scope);
            self.mem.aliases.borrow_mut().entry(scope).or_default().push(Rc::new(RefCell::new(row)));
        }
        let mut q = db.prepare("select scope, salt from salts")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
        for row in rows {
            let (scope, salt) = row?;
            self.mem.salts.borrow_mut().insert(scope, salt);
        }
        Ok(())
    }

    pub fn commits(&self) -> usize {
        self.mem.commits.get()
    }
}

impl TokenStore for SqliteTokenStore {
    fn load_tokens(&self, scope: &str) -> Vec<TokenRef> {
        self.mem.load_tokens(scope)
    }

    fn load_aliases(&self, scope: &str) -> Vec<AliasRef> {
        self.mem.load_aliases(scope)
    }

    fn load_salt(&self, scope: &str) -> Option<Vec<u8>> {
        self.mem.load_salt(scope)
    }

    fn add_token(&self, scope: &str, token: &str, kind: &str, real_value: &str, normalized: &str, status: &str,
                 attributes: Option<Map<String, Value>>) -> TokenRef {
        self.mem.add_token(scope, token, kind, real_value, normalized, status, attributes)
    }

    fn add_alias(&self, scope: &str, token: &str, surface: &str, normalized: &str, source: &str) -> AliasRef {
        self.mem.add_alias(scope, token, surface, normalized, source)
    }

    fn add_salt(&self, scope: &str, salt: Vec<u8>) {
        self.mem.add_salt(scope, salt)
    }

    fn commit(&self) -> Result<(), String> {
        self.mem.commit()?;
        let mut db = self.db.borrow_mut();
        let tx = db.transaction().map_err(|e| e.to_string())?;
        let scopes = self.mem.order.borrow().clone();
        for scope in &scopes {
            if let Some(rows) = self.mem.tokens.borrow().get(scope) {
                tx.execute("delete from tokens where scope = ?", params![scope]).map_err(|e| e.to_string())?;
                let mut ins = tx.prepare("insert into tokens values (?, ?, ?, ?, ?, ?, ?, ?, ?)").map_err(|e| e.to_string())?;
                for (i, r) in rows.iter().enumerate() {
                    let r = r.borrow();
                    let attrs = r.attributes.as_ref().filter(|a| !a.is_empty()).map(|a| py::dumps_ascii(&Value::Object(a.clone())));
                    ins.execute(params![scope, i as i64, r.token, r.kind, r.real_value, r.normalized, r.status, r.canonical_token, attrs])
                        .map_err(|e| e.to_string())?;
                }
            }
            if let Some(rows) = self.mem.aliases.borrow().get(scope) {
                tx.execute("delete from aliases where scope = ?", params![scope]).map_err(|e| e.to_string())?;
                let mut ins = tx.prepare("insert into aliases values (?, ?, ?, ?, ?, ?)").map_err(|e| e.to_string())?;
                for (i, r) in rows.iter().enumerate() {
                    let r = r.borrow();
                    ins.execute(params![scope, i as i64, r.token, r.surface, r.normalized, r.source]).map_err(|e| e.to_string())?;
                }
            }
        }
        for (scope, salt) in self.mem.salts.borrow().iter() {
            tx.execute("insert or replace into salts values (?, ?)", params![scope, salt]).map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())
    }

    fn restore_get(&self, key: &str) -> Option<String> {
        self.db.borrow().query_row("select value from restore where key = ?", params![key], |r| r.get(0)).ok()
    }

    fn restore_put(&self, key: &str, value: &str) {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        let _ = self.db.borrow().execute("insert or replace into restore values (?, ?, ?)", params![key, value, now]);
    }
}
