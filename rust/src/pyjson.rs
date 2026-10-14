// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! `json.JSONDecoder(strict=False)` as Python reads text: `raw_decode` at a place (no white space
//! skipped before the value), and `loads` (white space around it, nothing else). Control characters
//! may stand raw in a string; a key given twice keeps its first place and its last value, as a dict
//! built from the pairs does.
//!
//! Where a JSON value cannot hold what Python makes: NaN and the infinities read as null; a lone
//! surrogate as U+FFFD; an integer past 64 bits as a float.

use serde_json::{Map, Number, Value};

/// `json.loads(text, strict=False)`.
pub fn loads(text: &str) -> Option<Value> {
    let t: Vec<char> = text.chars().collect();
    let start = skip_ws(&t, 0);
    let (value, end) = raw_decode(&t, start).ok()?;
    (skip_ws(&t, end) == t.len()).then_some(value)
}

fn skip_ws(t: &[char], mut i: usize) -> usize {
    while i < t.len() && matches!(t[i], ' ' | '\t' | '\n' | '\r') {
        i += 1;
    }
    i
}

/// `decoder.raw_decode(text, at)`: the value that begins at `at` and where it ends.
pub fn raw_decode(t: &[char], at: usize) -> Result<(Value, usize), ()> {
    Parser { t }.value(at)
}

struct Parser<'a> {
    t: &'a [char],
}

impl Parser<'_> {
    fn at(&self, i: usize) -> Option<char> {
        self.t.get(i).copied()
    }

    fn lit(&self, i: usize, word: &str) -> bool {
        word.chars().enumerate().all(|(k, c)| self.at(i + k) == Some(c))
    }

    fn value(&self, i: usize) -> Result<(Value, usize), ()> {
        match self.at(i).ok_or(())? {
            '"' => self.string(i + 1).map(|(s, e)| (Value::String(s), e)),
            '{' => self.object(i + 1),
            '[' => self.array(i + 1),
            'n' if self.lit(i, "null") => Ok((Value::Null, i + 4)),
            't' if self.lit(i, "true") => Ok((Value::Bool(true), i + 4)),
            'f' if self.lit(i, "false") => Ok((Value::Bool(false), i + 5)),
            'N' if self.lit(i, "NaN") => Ok((Value::Null, i + 3)),
            'I' if self.lit(i, "Infinity") => Ok((Value::Null, i + 8)),
            '-' if self.lit(i, "-Infinity") => Ok((Value::Null, i + 9)),
            _ => self.number(i),
        }
    }

    fn number(&self, start: usize) -> Result<(Value, usize), ()> {
        let digit = |i: usize| self.at(i).is_some_and(|c| c.is_ascii_digit());
        let mut i = start;
        if self.at(i) == Some('-') {
            i += 1;
        }
        match self.at(i) {
            Some('0') => i += 1,
            Some('1'..='9') => {
                while digit(i) {
                    i += 1;
                }
            }
            _ => return Err(()),
        }
        let mut float = false;
        if self.at(i) == Some('.') && digit(i + 1) {
            float = true;
            i += 1;
            while digit(i) {
                i += 1;
            }
        }
        if matches!(self.at(i), Some('e' | 'E')) {
            let mut j = i + 1;
            if matches!(self.at(j), Some('+' | '-')) {
                j += 1;
            }
            if digit(j) {
                float = true;
                while digit(j) {
                    j += 1;
                }
                i = j;
            }
        }
        let text: String = self.t[start..i].iter().collect();
        let number = if float {
            let f: f64 = text.parse().map_err(|_| ())?;
            Number::from_f64(f).map_or(Value::Null, Value::Number)
        } else if let Ok(n) = text.parse::<i64>() {
            Value::Number(n.into())
        } else if let Ok(n) = text.parse::<u64>() {
            Value::Number(n.into())
        } else {
            let f: f64 = text.parse().map_err(|_| ())?;
            Number::from_f64(f).map_or(Value::Null, Value::Number)
        };
        Ok((number, i))
    }

    fn hex4(&self, i: usize) -> Result<u32, ()> {
        let mut n = 0;
        for k in 0..4 {
            n = n * 16 + self.at(i + k).and_then(|c| c.to_digit(16)).ok_or(())?;
        }
        Ok(n)
    }

    /// A string whose opening quote is just before `i`; strict=False: control characters allowed.
    fn string(&self, mut i: usize) -> Result<(String, usize), ()> {
        let mut out = String::new();
        loop {
            let c = self.at(i).ok_or(())?;
            match c {
                '"' => return Ok((out, i + 1)),
                '\\' => {
                    let e = self.at(i + 1).ok_or(())?;
                    i += 2;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let u = self.hex4(i)?;
                            i += 4;
                            if (0xd800..0xdc00).contains(&u) && self.at(i) == Some('\\') && self.at(i + 1) == Some('u') {
                                if let Ok(lo) = self.hex4(i + 2) {
                                    if (0xdc00..0xe000).contains(&lo) {
                                        out.push(char::from_u32(0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00)).unwrap_or('\u{fffd}'));
                                        i += 6;
                                        continue;
                                    }
                                }
                            }
                            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(()),
                    }
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            }
        }
    }

    fn object(&self, i: usize) -> Result<(Value, usize), ()> {
        let mut map = Map::new();
        let mut i = skip_ws(self.t, i);
        if self.at(i) == Some('}') {
            return Ok((Value::Object(map), i + 1));
        }
        loop {
            if self.at(i) != Some('"') {
                return Err(());
            }
            let (key, e) = self.string(i + 1)?;
            i = skip_ws(self.t, e);
            if self.at(i) != Some(':') {
                return Err(());
            }
            i = skip_ws(self.t, i + 1);
            let (value, e) = self.value(i)?;
            map.insert(key, value);
            i = skip_ws(self.t, e);
            match self.at(i) {
                Some('}') => return Ok((Value::Object(map), i + 1)),
                Some(',') => i = skip_ws(self.t, i + 1),
                _ => return Err(()),
            }
        }
    }

    fn array(&self, i: usize) -> Result<(Value, usize), ()> {
        let mut items = Vec::new();
        let mut i = skip_ws(self.t, i);
        if self.at(i) == Some(']') {
            return Ok((Value::Array(items), i + 1));
        }
        loop {
            let (value, e) = self.value(i)?;
            items.push(value);
            i = skip_ws(self.t, e);
            match self.at(i) {
                Some(']') => return Ok((Value::Array(items), i + 1)),
                Some(',') => i = skip_ws(self.t, i + 1),
                _ => return Err(()),
            }
        }
    }
}
