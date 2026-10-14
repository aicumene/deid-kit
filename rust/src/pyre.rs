// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Python's `re`, for the patterns deid-kit is written in: a backtracking matcher with Python's syntax
//! and answers — lookbehind and lookahead, atomic groups and possessive repeats, named groups and back
//! references, scoped flags, `$` before a final newline, `\w`/`\d`/`\s`/`\b` as Python's str patterns
//! have them, IGNORECASE as sre compares (the simple lower case and re's extra equivalences), and
//! `search`, `finditer`, `sub` and `split` with Python's rules for empty matches. Offsets are in
//! characters, as Python's indices are.

use std::sync::Arc;

use crate::unicode as u;
use crate::unicode_tables as t;

// ── flags ───────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flags {
    pub ignorecase: bool,
    pub multiline: bool,
    pub dotall: bool,
    pub verbose: bool,
    pub ascii: bool,
}

pub const I: Flags = Flags { ignorecase: true, multiline: false, dotall: false, verbose: false, ascii: false };
pub const M: Flags = Flags { ignorecase: false, multiline: true, dotall: false, verbose: false, ascii: false };
pub const S: Flags = Flags { ignorecase: false, multiline: false, dotall: true, verbose: false, ascii: false };
pub const NONE: Flags = Flags { ignorecase: false, multiline: false, dotall: false, verbose: false, ascii: false };

impl std::ops::BitOr for Flags {
    type Output = Flags;
    fn bitor(self, o: Flags) -> Flags {
        Flags { ignorecase: self.ignorecase || o.ignorecase, multiline: self.multiline || o.multiline,
                dotall: self.dotall || o.dotall, verbose: self.verbose || o.verbose, ascii: self.ascii || o.ascii }
    }
}

// ── case, as sre compares ───────────────────────────────────────────────────────────────────────

fn lower(c: char) -> char {
    let cp = c as u32;
    if cp < 128 {
        return c.to_ascii_lowercase();
    }
    t::SIMPLE_LOWER.binary_search_by_key(&cp, |(a, _)| *a).ok()
        .and_then(|i| char::from_u32(t::SIMPLE_LOWER[i].1)).unwrap_or(c)
}

fn upper(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_uppercase();
    }
    let up = u::upper(&c.to_string());
    let mut it = up.chars();
    match (it.next(), it.next()) {
        (Some(x), None) => x,
        _ => c,
    }
}

fn extras(c: char) -> &'static [u32] {
    t::EXTRA_CASES.binary_search_by_key(&(c as u32), |(a, _)| *a).ok().map_or(&[], |i| t::EXTRA_CASES[i].1)
}

// ── the pattern ─────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cat {
    Digit,
    Word,
    Space,
}

#[derive(Clone, Debug)]
enum Item {
    Char(char),
    Range(char, char),
    Cat(Cat, bool),
}

#[derive(Debug)]
struct Set {
    items: Vec<Item>,
    negated: bool,
    ascii: bool,
}

fn cat_has(cat: Cat, c: char, ascii: bool) -> bool {
    if ascii {
        return match cat {
            Cat::Digit => c.is_ascii_digit(),
            Cat::Word => c.is_ascii_alphanumeric() || c == '_',
            Cat::Space => matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'),
        };
    }
    match cat {
        Cat::Digit => u::is_decimal(c),
        Cat::Word => u::is_word(c),
        Cat::Space => u::is_space(c),
    }
}

impl Set {
    fn plain(&self, c: char) -> bool {
        self.items.iter().any(|it| match *it {
            Item::Char(x) => x == c,
            Item::Range(a, b) => a <= c && c <= b,
            Item::Cat(cat, neg) => cat_has(cat, c, self.ascii) != neg,
        })
    }

    fn has(&self, c: char, icase: bool) -> bool {
        let hit = if !icase {
            self.plain(c)
        } else {
            let l = lower(c);
            self.items.iter().any(|it| match *it {
                Item::Char(x) => {
                    let lx = lower(x);
                    lx == l || extras(lx).contains(&(l as u32))
                }
                Item::Range(a, b) => {
                    let inr = |x: char| a <= x && x <= b;
                    inr(l) || inr(upper(l)) || extras(l).iter().filter_map(|x| char::from_u32(*x)).any(inr)
                }
                Item::Cat(cat, neg) => cat_has(cat, l, self.ascii) != neg,
            })
        };
        hit != self.negated
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Greedy,
    Lazy,
    Possessive,
}

#[derive(Clone, Debug)]
enum Node {
    Lit(char, bool),
    Any(bool),
    Set(Arc<Set>, bool),
    Bol(bool),
    Eol(bool),
    BeginText,
    EndText,
    Boundary { ascii: bool, neg: bool },
    Group(Option<usize>, Box<Node>),
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Rep { node: Box<Node>, min: usize, max: usize, kind: Kind },
    Look { behind: bool, neg: bool, node: Box<Node>, width: usize },
    Backref(usize, bool),
    Atomic(Box<Node>),
}

const INF: usize = usize::MAX;

impl Node {
    /// Characters it consumes, when that is fixed (a lookbehind needs it).
    fn width(&self) -> Option<usize> {
        match self {
            Node::Lit(..) | Node::Any(_) | Node::Set(..) => Some(1),
            Node::Bol(_) | Node::Eol(_) | Node::BeginText | Node::EndText | Node::Boundary { .. } | Node::Look { .. } => Some(0),
            Node::Group(_, n) | Node::Atomic(n) => n.width(),
            Node::Cat(ns) => ns.iter().map(Node::width).sum(),
            Node::Alt(ns) => {
                let w = ns.first().and_then(Node::width)?;
                ns.iter().all(|n| n.width() == Some(w)).then_some(w)
            }
            Node::Rep { node, min, max, .. } => (min == max).then(|| node.width().map(|w| w * min)).flatten(),
            Node::Backref(..) => None,
        }
    }

    fn single(&self) -> bool {
        matches!(self, Node::Lit(..) | Node::Any(_) | Node::Set(..))
    }
}

// ── parsing, as sre_parse ───────────────────────────────────────────────────────────────────────

struct Parser<'p> {
    p: Vec<char>,
    i: usize,
    groups: usize,
    names: Vec<(String, usize)>,
    open: Vec<usize>,
    _src: &'p str,
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn skip_verbose(&mut self, f: Flags) {
        if !f.verbose {
            return;
        }
        loop {
            match self.peek() {
                Some(c) if u::is_space(c) => self.i += 1,
                Some('#') => {
                    while let Some(c) = self.peek() {
                        self.i += 1;
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn alt(&mut self, f: Flags) -> Result<Node, String> {
        let mut branches = vec![self.seq(f)?];
        while self.eat('|') {
            branches.push(self.seq(f)?);
        }
        Ok(if branches.len() == 1 { branches.pop().unwrap() } else { Node::Alt(branches) })
    }

    fn seq(&mut self, f: Flags) -> Result<Node, String> {
        let mut items: Vec<Node> = Vec::new();
        loop {
            self.skip_verbose(f);
            let Some(c) = self.peek() else { break };
            if c == '|' || c == ')' {
                break;
            }
            if "*+?{".contains(c) {
                // a repeat of the item before it, as sre_parse reads it
                let here = self.i;
                self.i += 1;
                let (min, max) = match c {
                    '*' => (0, INF),
                    '+' => (1, INF),
                    '?' => (0, 1),
                    _ => {
                        if self.peek() == Some('}') {
                            items.push(Node::Lit('{', f.ignorecase));
                            continue;
                        }
                        let lo = self.number();
                        let hi = if self.eat(',') { self.number() } else { Some(lo.unwrap_or(usize::MAX)).filter(|_| lo.is_some()) };
                        let comma = self.p[here..self.i].contains(&',');
                        if !self.eat('}') || (lo.is_none() && !comma) {
                            self.i = here + 1;
                            items.push(Node::Lit('{', f.ignorecase));
                            continue;
                        }
                        let min = lo.unwrap_or(0);
                        let max = if comma { hi.unwrap_or(INF) } else { min };
                        if max < min {
                            return err(format!("min repeat greater than max repeat at position {}", here + 1));
                        }
                        (min, max)
                    }
                };
                let Some(item) = items.pop() else { return err(format!("nothing to repeat at position {here}")) };
                if matches!(item, Node::Bol(_) | Node::Eol(_) | Node::BeginText | Node::EndText | Node::Boundary { .. }) {
                    return err(format!("nothing to repeat at position {here}"));
                }
                if matches!(item, Node::Rep { .. }) {
                    return err(format!("multiple repeat at position {here}"));
                }
                let item = match item {
                    Node::Group(None, inner) => *inner,
                    other => other,
                };
                let kind = if self.eat('?') { Kind::Lazy } else if self.eat('+') { Kind::Possessive } else { Kind::Greedy };
                items.push(Node::Rep { node: Box::new(item), min, max, kind });
                continue;
            }
            if let Some(atom) = self.atom(f)? {
                items.push(atom);
            }
        }
        Ok(if items.len() == 1 { items.pop().unwrap() } else { Node::Cat(items) })
    }

    fn number(&mut self) -> Option<usize> {
        let s = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        (self.i > s).then(|| self.p[s..self.i].iter().collect::<String>().parse().unwrap_or(usize::MAX))
    }

    fn escape_char(&mut self, in_class: bool) -> Result<Escaped, String> {
        let Some(c) = self.peek() else { return err("bad escape (end of pattern)") };
        self.i += 1;
        let hex = |p: &mut Parser, n: usize| -> Result<char, String> {
            let s: String = p.p.get(p.i..p.i + n).map(|x| x.iter().collect()).unwrap_or_default();
            if s.chars().count() != n || !s.chars().all(|c| c.is_ascii_hexdigit()) {
                return err(format!("incomplete escape \\{c}{s}"));
            }
            p.i += n;
            char::from_u32(u32::from_str_radix(&s, 16).unwrap()).ok_or_else(|| "bad escape: not a character".to_string())
        };
        Ok(match c {
            'a' => Escaped::Char('\x07'),
            'f' => Escaped::Char('\x0c'),
            'n' => Escaped::Char('\n'),
            'r' => Escaped::Char('\r'),
            't' => Escaped::Char('\t'),
            'v' => Escaped::Char('\x0b'),
            '\\' => Escaped::Char('\\'),
            'x' => Escaped::Char(hex(self, 2)?),
            'u' => Escaped::Char(hex(self, 4)?),
            'U' => Escaped::Char(hex(self, 8)?),
            'd' => Escaped::Cat(Cat::Digit, false),
            'D' => Escaped::Cat(Cat::Digit, true),
            'w' => Escaped::Cat(Cat::Word, false),
            'W' => Escaped::Cat(Cat::Word, true),
            's' => Escaped::Cat(Cat::Space, false),
            'S' => Escaped::Cat(Cat::Space, true),
            'b' if in_class => Escaped::Char('\x08'),
            'b' => Escaped::Boundary(false),
            'B' if !in_class => Escaped::Boundary(true),
            'A' if !in_class => Escaped::BeginText,
            'Z' if !in_class => Escaped::EndText,
            '0' => {
                let mut v = 0u32;
                let mut n = 0;
                while n < 2 && self.peek().is_some_and(|d| ('0'..='7').contains(&d)) {
                    v = v * 8 + self.peek().unwrap().to_digit(8).unwrap();
                    self.i += 1;
                    n += 1;
                }
                Escaped::Char(char::from_u32(v).unwrap())
            }
            '1'..='9' => {
                // three octal digits are a character; otherwise a group reference
                let rest: Vec<char> = self.p[self.i..].iter().take(2).copied().collect();
                if ('1'..='7').contains(&c) && rest.len() == 2 && rest.iter().all(|d| ('0'..='7').contains(d)) {
                    self.i += 2;
                    let v = c.to_digit(8).unwrap() * 64 + rest[0].to_digit(8).unwrap() * 8 + rest[1].to_digit(8).unwrap();
                    if v > 0o377 {
                        return err(format!("octal escape value \\{c}{}{} outside of range 0-0o377", rest[0], rest[1]));
                    }
                    Escaped::Char(char::from_u32(v).unwrap())
                } else if in_class {
                    return err(format!("bad escape \\{c}"));
                } else {
                    let mut n = c.to_digit(10).unwrap() as usize;
                    if self.peek().is_some_and(|d| d.is_ascii_digit()) {
                        n = n * 10 + self.peek().unwrap().to_digit(10).unwrap() as usize;
                        self.i += 1;
                    }
                    Escaped::Backref(n)
                }
            }
            c if c.is_ascii_alphanumeric() => return err(format!("bad escape \\{c}")),
            c => Escaped::Char(c),
        })
    }

    fn class(&mut self, f: Flags) -> Result<Node, String> {
        let negated = self.eat('^');
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.peek() else { return err("unterminated character set") };
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let lo = if c == '\\' {
                match self.escape_char(true)? {
                    Escaped::Char(x) => x,
                    Escaped::Cat(cat, neg) => {
                        items.push(Item::Cat(cat, neg));
                        continue;
                    }
                    _ => return err("bad escape in set"),
                }
            } else {
                c
            };
            if self.peek() == Some('-') && self.p.get(self.i + 1).is_some_and(|n| *n != ']') {
                self.i += 1;
                let hc = self.peek().unwrap();
                self.i += 1;
                let hi = if hc == '\\' {
                    match self.escape_char(true)? {
                        Escaped::Char(x) => x,
                        _ => return err(format!("bad character range {lo}-\\")),
                    }
                } else {
                    hc
                };
                if hi < lo {
                    return err(format!("bad character range {lo}-{hi}"));
                }
                items.push(Item::Range(lo, hi));
            } else {
                items.push(Item::Char(lo));
            }
        }
        Ok(Node::Set(Arc::new(Set { items, negated, ascii: f.ascii }), f.ignorecase))
    }

    fn flags_of(&mut self, mut f: Flags) -> Result<(Flags, bool), String> {
        let mut on = true;
        loop {
            match self.peek() {
                Some('i') => f.ignorecase = on,
                Some('m') => f.multiline = on,
                Some('s') => f.dotall = on,
                Some('x') => f.verbose = on,
                Some('a') => f.ascii = on,
                Some('u') | Some('L') => {}
                Some('-') if on => on = false,
                Some(':') => {
                    self.i += 1;
                    return Ok((f, true));
                }
                Some(')') => {
                    self.i += 1;
                    return Ok((f, false));
                }
                _ => return err("unknown flag"),
            }
            self.i += 1;
        }
    }

    fn atom(&mut self, f: Flags) -> Result<Option<Node>, String> {
        let c = self.peek().unwrap();
        self.i += 1;
        Ok(Some(match c {
            '.' => Node::Any(f.dotall),
            '^' => Node::Bol(f.multiline),
            '$' => Node::Eol(f.multiline),
            '[' => self.class(f)?,
            '\\' => match self.escape_char(false)? {
                Escaped::Char(x) => Node::Lit(x, f.ignorecase),
                Escaped::Cat(cat, neg) => Node::Set(Arc::new(Set { items: vec![Item::Cat(cat, neg)], negated: false, ascii: f.ascii }), false),
                Escaped::Boundary(neg) => Node::Boundary { ascii: f.ascii, neg },
                Escaped::BeginText => Node::BeginText,
                Escaped::EndText => Node::EndText,
                Escaped::Backref(n) => {
                    if n > self.groups || self.open.contains(&n) {
                        return err(format!("invalid group reference {n}"));
                    }
                    Node::Backref(n, f.ignorecase)
                }
            },
            '(' => return self.group(f),
            ')' => return err("unbalanced parenthesis"),
            c => Node::Lit(c, f.ignorecase),
        }))
    }

    fn group(&mut self, f: Flags) -> Result<Option<Node>, String> {
        let mut capture = true;
        let mut name: Option<String> = None;
        let mut look: Option<(bool, bool)> = None;
        let mut atomic = false;
        let mut inner = f;
        if self.eat('?') {
            capture = false;
            match self.peek() {
                Some(':') => self.i += 1,
                Some('P') => {
                    self.i += 1;
                    if self.eat('<') {
                        let s = self.i;
                        while self.peek().is_some_and(|c| c != '>') {
                            self.i += 1;
                        }
                        let n: String = self.p[s..self.i].iter().collect();
                        if !self.eat('>') || n.is_empty() {
                            return err("missing >, unterminated name");
                        }
                        if self.names.iter().any(|(x, _)| *x == n) {
                            return err(format!("redefinition of group name {}", crate::py::repr_str(&n)));
                        }
                        name = Some(n);
                        capture = true;
                    } else if self.eat('=') {
                        let s = self.i;
                        while self.peek().is_some_and(|c| c != ')') {
                            self.i += 1;
                        }
                        let n: String = self.p[s..self.i].iter().collect();
                        self.eat(')');
                        let idx = self.names.iter().find(|(x, _)| *x == n).map(|(_, i)| *i)
                            .ok_or_else(|| format!("unknown group name {}", crate::py::repr_str(&n)))?;
                        return Ok(Some(Node::Backref(idx, f.ignorecase)));
                    } else {
                        return err("unknown extension ?P");
                    }
                }
                Some('#') => {
                    while self.peek().is_some_and(|c| c != ')') {
                        self.i += 1;
                    }
                    if !self.eat(')') {
                        return err("missing ), unterminated comment");
                    }
                    return Ok(None);
                }
                Some('=') => { self.i += 1; look = Some((false, false)); }
                Some('!') => { self.i += 1; look = Some((false, true)); }
                Some('<') => {
                    self.i += 1;
                    if self.eat('=') { look = Some((true, false)); }
                    else if self.eat('!') { look = Some((true, true)); }
                    else { return err("unknown extension ?<"); }
                }
                Some('>') => { self.i += 1; atomic = true; }
                Some(c) if "imsxauL-".contains(c) => {
                    let at = self.i;
                    let (nf, scoped) = self.flags_of(f)?;
                    if !scoped {
                        return err(format!("global flags not at the start of the expression at position {}", at - 2));
                    }
                    inner = nf;
                }
                _ => return err("unknown extension"),
            }
        }
        let idx = if capture {
            self.groups += 1;
            let idx = self.groups;
            if let Some(n) = name {
                self.names.push((n, idx));
            }
            self.open.push(idx);
            Some(idx)
        } else {
            None
        };
        let node = self.alt(inner)?;
        if !self.eat(')') {
            return err("missing ), unterminated subpattern");
        }
        if let Some(i) = idx {
            self.open.retain(|o| *o != i);
        }
        Ok(Some(if let Some((behind, neg)) = look {
            let width = if behind {
                node.width().ok_or_else(|| "look-behind requires fixed-width pattern".to_string())?
            } else {
                0
            };
            Node::Look { behind, neg, node: Box::new(node), width }
        } else if atomic {
            Node::Atomic(Box::new(node))
        } else {
            Node::Group(idx, Box::new(node))
        }))
    }
}

enum Escaped {
    Char(char),
    Cat(Cat, bool),
    Boundary(bool),
    BeginText,
    EndText,
    Backref(usize),
}

/// A compiled pattern.
#[derive(Debug)]
pub struct Regex {
    root: Node,
    pub groups: usize,
    pub names: Vec<(String, usize)>,
    pub pattern: String,
    pub flags: Flags,
    /// A character every match starts with (the search skips to it).
    first: Option<char>,
}

/// One match: the whole span and each group's, in characters.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    pub start: usize,
    pub end: usize,
    pub spans: Vec<Option<(usize, usize)>>,
}

impl Match {
    /// `m.span(i)` of a group, or None when it did not take part.
    pub fn group(&self, i: usize) -> Option<(usize, usize)> {
        if i == 0 { Some((self.start, self.end)) } else { self.spans.get(i).copied().flatten() }
    }
}

fn first_char(n: &Node) -> Option<char> {
    match n {
        Node::Lit(c, false) => Some(*c),
        Node::Cat(ns) => ns.first().and_then(first_char),
        Node::Group(_, n) | Node::Atomic(n) => first_char(n),
        Node::Rep { node, min, .. } if *min > 0 => first_char(node),
        _ => None,
    }
}

impl Regex {
    pub fn new(pattern: &str) -> Result<Regex, String> {
        Regex::with_flags(pattern, NONE)
    }

    pub fn with_flags(pattern: &str, flags: Flags) -> Result<Regex, String> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut flags = flags;
        // leading `(?flags)` groups set them for the whole pattern
        let mut at = 0;
        let mut lead = Parser { p: chars.clone(), i: 0, groups: 0, names: Vec::new(), open: Vec::new(), _src: pattern };
        while chars.get(at) == Some(&'(') && chars.get(at + 1) == Some(&'?') && chars.get(at + 2).is_some_and(|c| "imsxauL".contains(*c)) {
            lead.i = at + 2;
            match lead.flags_of(flags) {
                Ok((nf, false)) => {
                    flags = nf;
                    at = lead.i;
                }
                _ => break,
            }
        }
        let mut p = Parser { p: chars, i: at, groups: 0, names: Vec::new(), open: Vec::new(), _src: pattern };
        let root = p.alt(flags)?;
        if p.i < p.p.len() {
            return err(format!("unbalanced parenthesis at position {}", p.i));
        }
        Ok(Regex { first: first_char(&root), root, groups: p.groups, names: p.names, pattern: pattern.into(), flags })
    }

    pub fn group_index(&self, name: &str) -> Option<usize> {
        self.names.iter().find(|(n, _)| n == name).map(|(_, i)| *i)
    }

    fn run(&self, t: &[char], start: usize, anchored: bool, full: bool, must_advance: bool) -> Option<Match> {
        let mut m = Matcher { t, caps: vec![None; self.groups + 1] };
        let try_at = |m: &mut Matcher, i: usize| -> Option<Match> {
            m.caps.iter_mut().for_each(|c| *c = None);
            let mut end = None;
            let ok = m.m(&self.root, i, &mut |_, j| {
                if (full && j != t.len()) || (must_advance && i == start && j == start) {
                    return false;
                }
                end = Some(j);
                true
            });
            ok.then(|| Match { start: i, end: end.unwrap(), spans: m.caps.clone() })
        };
        if anchored {
            return try_at(&mut m, start);
        }
        let mut i = start;
        while i <= t.len() {
            if let Some(c) = self.first {
                match t[i.min(t.len())..].iter().position(|x| *x == c) {
                    Some(off) => i += off,
                    None => return None,
                }
            }
            if let Some(found) = try_at(&mut m, i) {
                return Some(found);
            }
            i += 1;
        }
        None
    }

    /// `pattern.search(text, pos)`.
    pub fn search_at(&self, t: &[char], pos: usize) -> Option<Match> {
        self.run(t, pos, false, false, false)
    }

    /// `pattern.match(text, pos)`.
    pub fn match_at(&self, t: &[char], pos: usize) -> Option<Match> {
        self.run(t, pos, true, false, false)
    }

    /// `pattern.fullmatch(text)`.
    pub fn fullmatch(&self, t: &[char]) -> Option<Match> {
        self.run(t, 0, true, true, false)
    }

    /// `pattern.finditer(text)`.
    pub fn finditer(&self, t: &[char]) -> Vec<Match> {
        let mut out = Vec::new();
        let (mut pos, mut must_advance) = (0, false);
        while pos <= t.len() {
            let Some(m) = self.run(t, pos, false, false, must_advance) else { break };
            must_advance = m.end == m.start;
            pos = m.end;
            out.push(m);
        }
        out
    }

    /// `pattern.sub(f, text, count)`: each match replaced by what `f` makes of it; the text and how many.
    pub fn sub_with(&self, t: &[char], count: usize, mut f: impl FnMut(&Match) -> String) -> (String, usize) {
        let mut out = String::with_capacity(t.len());
        let (mut last, mut n) = (0, 0);
        for m in self.finditer(t) {
            if count != 0 && n == count {
                break;
            }
            out.extend(&t[last..m.start]);
            out.push_str(&f(&m));
            last = m.end;
            n += 1;
        }
        out.extend(&t[last..]);
        (out, n)
    }

    /// `pattern.sub(template, text, count)`, the template read as Python reads it (`\1`, `\g<name>`, escapes).
    pub fn sub(&self, template: &str, t: &[char], count: usize) -> Result<(String, usize), String> {
        let parts = self.template(template)?;
        Ok(self.sub_with(t, count, |m| {
            let mut s = String::new();
            for p in &parts {
                match p {
                    Tpl::Text(x) => s.push_str(x),
                    Tpl::Group(g) => {
                        if let Some((a, b)) = m.group(*g) {
                            s.extend(&t[a..b]);
                        }
                    }
                }
            }
            s
        }))
    }

    fn template(&self, template: &str) -> Result<Vec<Tpl>, String> {
        let c: Vec<char> = template.chars().collect();
        let mut out: Vec<Tpl> = Vec::new();
        let mut text = String::new();
        let mut i = 0;
        let flush = |text: &mut String, out: &mut Vec<Tpl>| {
            if !text.is_empty() {
                out.push(Tpl::Text(std::mem::take(text)));
            }
        };
        while i < c.len() {
            if c[i] != '\\' {
                text.push(c[i]);
                i += 1;
                continue;
            }
            i += 1;
            let Some(&e) = c.get(i) else { return err("bad escape (end of pattern)") };
            i += 1;
            match e {
                'g' => {
                    if c.get(i) != Some(&'<') {
                        return err("missing <");
                    }
                    let s = i + 1;
                    let close = c[s..].iter().position(|x| *x == '>').ok_or("missing group name")? + s;
                    let name: String = c[s..close].iter().collect();
                    i = close + 1;
                    let g = match name.parse::<usize>() {
                        Ok(n) if n <= self.groups => n,
                        Ok(n) => return err(format!("invalid group reference {n}")),
                        Err(_) => self.group_index(&name).ok_or_else(|| format!("unknown group name {}", crate::py::repr_str(&name)))?,
                    };
                    flush(&mut text, &mut out);
                    out.push(Tpl::Group(g));
                }
                '0'..='9' => {
                    let mut n = e.to_digit(10).unwrap() as usize;
                    if e != '0' && c.get(i).is_some_and(|d| d.is_ascii_digit()) {
                        n = n * 10 + c[i].to_digit(10).unwrap() as usize;
                        i += 1;
                    }
                    if e == '0' || n > self.groups {
                        if e == '0' {
                            text.push('\0');
                            continue;
                        }
                        return err(format!("invalid group reference {n}"));
                    }
                    flush(&mut text, &mut out);
                    out.push(Tpl::Group(n));
                }
                'n' => text.push('\n'),
                't' => text.push('\t'),
                'r' => text.push('\r'),
                'f' => text.push('\x0c'),
                'v' => text.push('\x0b'),
                'a' => text.push('\x07'),
                'b' => text.push('\x08'),
                '\\' => text.push('\\'),
                e if e.is_ascii_alphabetic() => return err(format!("bad escape \\{e}")),
                e => {
                    text.push('\\');
                    text.push(e);
                }
            }
        }
        flush(&mut text, &mut out);
        Ok(out)
    }

    /// `pattern.split(text, maxsplit)`: the pieces, with each group's text (None where it did not take part).
    pub fn split(&self, t: &[char], maxsplit: usize) -> Vec<Option<String>> {
        let mut out = Vec::new();
        let mut last = 0;
        for (n, m) in self.finditer(t).into_iter().enumerate() {
            if maxsplit != 0 && n == maxsplit {
                break;
            }
            out.push(Some(t[last..m.start].iter().collect()));
            for g in 1..=self.groups {
                out.push(m.group(g).map(|(a, b)| t[a..b].iter().collect()));
            }
            last = m.end;
        }
        out.push(Some(t[last..].iter().collect()));
        out
    }

    // ── on str, for callers that do not keep the characters ──

    pub fn is_match(&self, s: &str) -> bool {
        self.search_at(&s.chars().collect::<Vec<_>>(), 0).is_some()
    }

    /// `pattern.sub(f, s)` on a str.
    pub fn replace_all(&self, s: &str, f: impl FnMut(&Match, &[char]) -> String) -> String {
        let t: Vec<char> = s.chars().collect();
        let mut f = f;
        self.sub_with(&t, 0, |m| f(m, &t)).0
    }
}

enum Tpl {
    Text(String),
    Group(usize),
}

/// The text of a span.
pub fn text(t: &[char], span: (usize, usize)) -> String {
    t[span.0..span.1].iter().collect()
}

// ── matching ────────────────────────────────────────────────────────────────────────────────────

type K<'k, 't> = &'k mut dyn FnMut(&mut Matcher<'t>, usize) -> bool;

struct Matcher<'t> {
    t: &'t [char],
    caps: Vec<Option<(usize, usize)>>,
}

impl<'t> Matcher<'t> {
    fn one(&self, n: &Node, i: usize) -> bool {
        let Some(&c) = self.t.get(i) else { return false };
        match n {
            Node::Lit(x, false) => c == *x,
            Node::Lit(x, true) => {
                let (l, lx) = (lower(c), lower(*x));
                l == lx || extras(lx).contains(&(l as u32))
            }
            Node::Any(dotall) => *dotall || c != '\n',
            Node::Set(s, icase) => s.has(c, *icase),
            _ => false,
        }
    }

    fn word_at(&self, i: usize, ascii: bool) -> bool {
        self.t.get(i).is_some_and(|c| cat_has(Cat::Word, *c, ascii))
    }

    fn m(&mut self, n: &Node, i: usize, k: K<'_, 't>) -> bool {
        let len = self.t.len();
        match n {
            Node::Lit(..) | Node::Any(_) | Node::Set(..) => self.one(n, i) && k(self, i + 1),
            Node::Bol(multi) => (i == 0 || (*multi && self.t[i - 1] == '\n')) && k(self, i),
            Node::Eol(multi) => {
                let ok = i == len || self.t[i] == '\n' && (*multi || i == len - 1);
                ok && k(self, i)
            }
            Node::BeginText => i == 0 && k(self, i),
            Node::EndText => i == len && k(self, i),
            Node::Boundary { ascii, neg } => {
                if len == 0 {
                    return false;
                }
                let before = i > 0 && self.word_at(i - 1, *ascii);
                let after = self.word_at(i, *ascii);
                ((before != after) != *neg) && k(self, i)
            }
            Node::Group(None, node) => self.m(node, i, k),
            Node::Group(Some(g), node) => {
                let g = *g;
                self.m(node, i, &mut |s: &mut Matcher<'t>, j| {
                    let prev = s.caps[g];
                    s.caps[g] = Some((i, j));
                    if k(s, j) {
                        return true;
                    }
                    s.caps[g] = prev;
                    false
                })
            }
            Node::Cat(nodes) => self.seq(nodes, i, k),
            Node::Alt(branches) => {
                for b in branches {
                    let saved = self.caps.clone();
                    if self.m(b, i, k) {
                        return true;
                    }
                    self.caps = saved;
                }
                false
            }
            Node::Rep { node, min, max, kind } => self.rep(node, *min, *max, *kind, i, k),
            Node::Look { behind, neg, node, width } => {
                let saved = self.caps.clone();
                let hit = if *behind {
                    i >= *width && self.m(node, i - width, &mut |_, j| j == i)
                } else {
                    self.m(node, i, &mut |_, _| true)
                };
                if hit == *neg {
                    self.caps = saved;
                    return false;
                }
                if *neg {
                    self.caps = saved.clone();
                }
                if k(self, i) {
                    return true;
                }
                self.caps = saved;
                false
            }
            Node::Atomic(node) => {
                let saved = self.caps.clone();
                let mut end = None;
                if !self.m(node, i, &mut |_, j| {
                    end = Some(j);
                    true
                }) {
                    return false;
                }
                if k(self, end.unwrap()) {
                    return true;
                }
                self.caps = saved;
                false
            }
            Node::Backref(g, icase) => {
                let Some((a, b)) = self.caps.get(*g).copied().flatten() else { return false };
                let w = b - a;
                if i + w > len {
                    return false;
                }
                let same = (0..w).all(|x| {
                    let (p, q) = (self.t[a + x], self.t[i + x]);
                    p == q || (*icase && lower(p) == lower(q))
                });
                same && k(self, i + w)
            }
        }
    }

    fn seq(&mut self, nodes: &[Node], i: usize, k: K<'_, 't>) -> bool {
        match nodes {
            [] => k(self, i),
            [only] => self.m(only, i, k),
            [first, rest @ ..] => self.m(first, i, &mut |s: &mut Matcher<'t>, j| s.seq(rest, j, k)),
        }
    }

    fn rep(&mut self, node: &Node, min: usize, max: usize, kind: Kind, i: usize, k: K<'_, 't>) -> bool {
        if node.single() {
            // one character a time: count how far it can go, then try the ends in the repeat's order
            let mut n = 0;
            if kind == Kind::Lazy {
                while n < min {
                    if !self.one(node, i + n) {
                        return false;
                    }
                    n += 1;
                }
                loop {
                    if k(self, i + n) {
                        return true;
                    }
                    if n >= max || !self.one(node, i + n) {
                        return false;
                    }
                    n += 1;
                }
            }
            while n < max && self.one(node, i + n) {
                n += 1;
            }
            if n < min {
                return false;
            }
            if kind == Kind::Possessive {
                return k(self, i + n);
            }
            loop {
                if k(self, i + n) {
                    return true;
                }
                if n == min {
                    return false;
                }
                n -= 1;
            }
        }
        if kind == Kind::Possessive {
            // sre's POSSESSIVE_REPEAT: the minimum whatever they match, then more while they advance; none
            // is given back
            let saved = self.caps.clone();
            let mut pos = i;
            let mut count = 0;
            while count < max {
                let mut end = None;
                let before = pos;
                if !self.m(node, pos, &mut |_, j| {
                    end = Some(j);
                    true
                }) {
                    break;
                }
                pos = end.unwrap();
                count += 1;
                if count >= min && pos == before {
                    break;
                }
            }
            if count < min {
                self.caps = saved;
                return false;
            }
            if k(self, pos) {
                return true;
            }
            self.caps = saved;
            return false;
        }
        self.rep_from(node, min, max, kind, 0, i, None, k)
    }

    /// sre's MAX_UNTIL and MIN_UNTIL: `count` iterations done, the next starting at `i`; `last` is where
    /// the last iteration beyond the minimum began — one more is not tried from the same place.
    #[allow(clippy::too_many_arguments)]
    fn rep_from(&mut self, node: &Node, min: usize, max: usize, kind: Kind, count: usize, i: usize, last: Option<usize>,
                k: K<'_, 't>) -> bool {
        if count < min {
            return self.m(node, i, &mut |s: &mut Matcher<'t>, j| s.rep_from(node, min, max, kind, count + 1, j, last, k));
        }
        let can_more = count < max && last != Some(i);
        if kind == Kind::Lazy {
            if k(self, i) {
                return true;
            }
            return can_more && self.m(node, i, &mut |s: &mut Matcher<'t>, j| s.rep_from(node, min, max, kind, count + 1, j, Some(i), k));
        }
        if can_more && self.m(node, i, &mut |s: &mut Matcher<'t>, j| s.rep_from(node, min, max, kind, count + 1, j, Some(i), k)) {
            return true;
        }
        k(self, i)
    }
}

/// `re.escape(text)`: Python 3.13 escapes these characters only.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A pattern compiled once, for the patterns of the code itself (known to compile).
pub fn lazy(cell: &'static std::sync::OnceLock<Regex>, pattern: &str, flags: Flags) -> &'static Regex {
    cell.get_or_init(|| Regex::with_flags(pattern, flags).unwrap_or_else(|e| panic!("{pattern}: {e}")))
}
