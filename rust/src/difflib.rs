// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! `difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()` over sequences of words.

use std::collections::HashMap;
use std::hash::Hash;

fn longest<T: Eq + Hash>(a: &[T], b: &[T], b2j: &HashMap<&T, Vec<usize>>, alo: usize, ahi: usize, blo: usize, bhi: usize)
                         -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for i in alo..ahi {
        let mut newj2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(&a[i]) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                newj2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = newj2len;
    }
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize] {
        bestsize += 1;
    }
    (besti, bestj, bestsize)
}

/// `get_matching_blocks()`: the matching runs in order, adjacent ones joined, ending with `(len a, len b, 0)`.
pub fn matching_blocks<T: Eq + Hash>(a: &[T], b: &[T]) -> Vec<(usize, usize, usize)> {
    let mut b2j: HashMap<&T, Vec<usize>> = HashMap::new();
    for (j, x) in b.iter().enumerate() {
        b2j.entry(x).or_default().push(j);
    }
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort();
    let (mut i1, mut j1, mut k1) = (0, 0, 0);
    let mut out = Vec::new();
    for (i2, j2, k2) in blocks {
        if i1 + k1 == i2 && j1 + k1 == j2 {
            k1 += k2;
        } else {
            if k1 > 0 {
                out.push((i1, j1, k1));
            }
            (i1, j1, k1) = (i2, j2, k2);
        }
    }
    if k1 > 0 {
        out.push((i1, j1, k1));
    }
    out.push((a.len(), b.len(), 0));
    out
}

/// `get_opcodes()`: `(tag, i1, i2, j1, j2)`, tag one of replace, delete, insert, equal.
pub fn opcodes<T: Eq + Hash>(a: &[T], b: &[T]) -> Vec<(&'static str, usize, usize, usize, usize)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    for (ai, bj, size) in matching_blocks(a, b) {
        let tag = if i < ai && j < bj { "replace" } else if i < ai { "delete" } else if j < bj { "insert" } else { "" };
        if !tag.is_empty() {
            out.push((tag, i, ai, j, bj));
        }
        (i, j) = (ai + size, bj + size);
        if size > 0 {
            out.push(("equal", ai, i, bj, j));
        }
    }
    out
}
