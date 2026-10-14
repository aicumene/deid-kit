// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! A Zstandard decoder (RFC 8878), for request bodies Codex compresses: frames and skippable frames;
//! raw, RLE and compressed blocks; Huffman-coded literals in one or four streams, with the tree given or
//! repeated; FSE-coded sequences with predefined, RLE, given or repeated tables and repeat offsets. No
//! dictionaries (Codex sends none); the content checksum is not verified.

const MAGIC: u32 = 0xFD2F_B528;

#[derive(Debug)]
pub struct Corrupt(pub String);

impl std::fmt::Display for Corrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn bad<T>(why: &str) -> Result<T, Corrupt> {
    Err(Corrupt(why.into()))
}

// ── bit readers ─────────────────────────────────────────────────────────────────────────────────

/// Reads forward, least significant bit first (FSE table descriptions).
struct Forward<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Forward<'_> {
    fn bits(&mut self, n: usize) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            let p = self.pos + i;
            let bit = self.data.get(p / 8).map_or(0, |b| (b >> (p % 8)) & 1);
            v |= (bit as u32) << i;
        }
        self.pos += n;
        v
    }

    fn peek(&self, n: usize) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            let p = self.pos + i;
            let bit = self.data.get(p / 8).map_or(0, |b| (b >> (p % 8)) & 1);
            v |= (bit as u32) << i;
        }
        v
    }
}

/// Reads backward from the marker bit of the last byte (Huffman streams, FSE streams).
struct Backward<'a> {
    data: &'a [u8],
    /// Bits left above position 0; negative once read past the start.
    pos: isize,
}

impl<'a> Backward<'a> {
    fn new(data: &'a [u8]) -> Result<Backward<'a>, Corrupt> {
        let Some(&last) = data.last() else { return bad("an empty bitstream") };
        if last == 0 {
            return bad("a bitstream without its end mark");
        }
        let marker = 7 - last.leading_zeros() as isize;
        Ok(Backward { data, pos: (data.len() as isize - 1) * 8 + marker })
    }

    fn bit(&self, p: isize) -> u64 {
        if p < 0 {
            return 0;
        }
        let p = p as usize;
        ((self.data[p / 8] >> (p % 8)) & 1) as u64
    }

    fn peek(&self, n: usize) -> u64 {
        let mut v = 0u64;
        for i in 0..n as isize {
            v = (v << 1) | self.bit(self.pos - 1 - i);
        }
        v
    }

    fn bits(&mut self, n: usize) -> u64 {
        let v = self.peek(n);
        self.pos -= n as isize;
        v
    }

    fn overflowed(&self) -> bool {
        self.pos < 0
    }
}

// ── FSE ─────────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct Entry {
    symbol: u16,
    bits: u8,
    base: u16,
}

#[derive(Clone)]
struct Fse {
    log: usize,
    table: Vec<Entry>,
}

/// The normalised counts of an FSE table description, and its accuracy log; the bytes it took.
fn read_counts(data: &[u8], max_symbol: usize, max_log: usize) -> Result<(Vec<i32>, usize, usize), Corrupt> {
    let mut r = Forward { data, pos: 0 };
    let log = r.bits(4) as usize + 5;
    if log > max_log {
        return bad("an FSE table too precise");
    }
    let mut remaining: i32 = (1 << log) + 1;
    let mut threshold: i32 = 1 << log;
    let mut nb = log + 1;
    let mut counts: Vec<i32> = Vec::new();
    let mut previous0 = false;
    while remaining > 1 && counts.len() <= max_symbol {
        if previous0 {
            let mut zeros = 0;
            loop {
                let rep = r.bits(2);
                zeros += rep;
                if rep != 3 {
                    break;
                }
            }
            for _ in 0..zeros {
                counts.push(0);
            }
            if counts.len() > max_symbol + 1 {
                return bad("an FSE table past its symbols");
            }
        }
        let max = 2 * threshold - 1 - remaining;
        let low = r.peek(nb - 1) as i32 & (threshold - 1);
        let mut count;
        if low < max {
            count = low;
            r.pos += nb - 1;
        } else {
            count = r.peek(nb) as i32 & (2 * threshold - 1);
            if count >= threshold {
                count -= max;
            }
            r.pos += nb;
        }
        count -= 1;
        remaining -= count.abs();
        counts.push(count);
        previous0 = count == 0;
        while remaining < threshold {
            nb -= 1;
            threshold >>= 1;
        }
    }
    if remaining != 1 || counts.len() > max_symbol + 1 {
        return bad("a broken FSE table description");
    }
    Ok((counts, log, r.pos.div_ceil(8)))
}

fn build_fse(counts: &[i32], log: usize) -> Fse {
    let size = 1usize << log;
    let mut table = vec![Entry::default(); size];
    let mut high = size - 1;
    let mut next: Vec<u32> = vec![0; counts.len()];
    for (s, &c) in counts.iter().enumerate() {
        if c == -1 {
            table[high].symbol = s as u16;
            high = high.wrapping_sub(1);
            next[s] = 1;
        } else {
            next[s] = c.max(0) as u32;
        }
    }
    let step = (size >> 1) + (size >> 3) + 3;
    let mask = size - 1;
    let mut pos = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        for _ in 0..c.max(0) {
            table[pos].symbol = s as u16;
            loop {
                pos = (pos + step) & mask;
                if pos <= high || high == usize::MAX {
                    break;
                }
            }
        }
    }
    for e in table.iter_mut() {
        let s = e.symbol as usize;
        let state = next[s];
        next[s] += 1;
        let bits = log as u32 - (31 - state.leading_zeros());
        e.bits = bits as u8;
        e.base = ((state << bits) as usize - size) as u16;
    }
    Fse { log, table }
}

fn rle_fse(symbol: u8) -> Fse {
    Fse { log: 0, table: vec![Entry { symbol: symbol as u16, bits: 0, base: 0 }] }
}

struct State<'t> {
    fse: &'t Fse,
    at: usize,
}

impl<'t> State<'t> {
    fn new(fse: &'t Fse, r: &mut Backward) -> State<'t> {
        State { fse, at: r.bits(fse.log) as usize }
    }

    fn symbol(&self) -> u16 {
        self.fse.table[self.at].symbol
    }

    fn update(&mut self, r: &mut Backward) {
        let e = self.fse.table[self.at];
        self.at = e.base as usize + r.bits(e.bits as usize) as usize;
    }
}

// ── Huffman ─────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Huffman {
    max_bits: usize,
    /// (symbol, bits) by the next `max_bits` bits.
    table: Vec<(u8, u8)>,
}

fn read_huffman(data: &[u8]) -> Result<(Huffman, usize), Corrupt> {
    let Some(&header) = data.first() else { return bad("no Huffman tree") };
    let mut weights: Vec<u8> = Vec::new();
    let used;
    if header < 128 {
        let size = header as usize;
        let body = data.get(1..1 + size).ok_or(Corrupt("a cut Huffman tree".into()))?;
        let (counts, log, taken) = read_counts(body, 255, 6)?;
        let fse = build_fse(&counts, log);
        let mut r = Backward::new(&body[taken..])?;
        let mut s1 = State::new(&fse, &mut r);
        let mut s2 = State::new(&fse, &mut r);
        loop {
            weights.push(s1.symbol() as u8);
            s1.update(&mut r);
            if r.overflowed() {
                weights.push(s2.symbol() as u8);
                break;
            }
            weights.push(s2.symbol() as u8);
            s2.update(&mut r);
            if r.overflowed() {
                weights.push(s1.symbol() as u8);
                break;
            }
            if weights.len() > 255 {
                return bad("too many Huffman weights");
            }
        }
        used = 1 + size;
    } else {
        let n = header as usize - 127;
        let bytes = n.div_ceil(2);
        let body = data.get(1..1 + bytes).ok_or(Corrupt("a cut Huffman tree".into()))?;
        for i in 0..n {
            let b = body[i / 2];
            weights.push(if i % 2 == 0 { b >> 4 } else { b & 15 });
        }
        used = 1 + bytes;
    }
    let total: u32 = weights.iter().filter(|w| **w > 0).map(|w| 1u32 << (w - 1)).sum();
    if total == 0 {
        return bad("an empty Huffman tree");
    }
    let max_bits = (32 - total.leading_zeros()) as usize;
    let rest = (1u32 << max_bits) - total;
    if !rest.is_power_of_two() {
        return bad("a Huffman tree that does not add up");
    }
    weights.push(rest.trailing_zeros() as u8 + 1);
    if max_bits > 11 {
        return bad("a Huffman tree too deep");
    }
    let size = 1usize << max_bits;
    let mut table = vec![(0u8, 0u8); size];
    let mut pos = 0usize;
    for w in 1..=max_bits as u8 {
        for (s, &sw) in weights.iter().enumerate() {
            if sw == w {
                let span = 1usize << (w - 1);
                let bits = (max_bits as u8 + 1) - w;
                for e in table.iter_mut().skip(pos).take(span) {
                    *e = (s as u8, bits);
                }
                pos += span;
            }
        }
    }
    if pos != size {
        return bad("a Huffman table that does not fill");
    }
    Ok((Huffman { max_bits, table }, used))
}

fn huffman_stream(h: &Huffman, data: &[u8], count: usize, out: &mut Vec<u8>) -> Result<(), Corrupt> {
    let mut r = Backward::new(data)?;
    for _ in 0..count {
        let (s, bits) = h.table[r.peek(h.max_bits) as usize];
        r.pos -= bits as isize;
        out.push(s);
    }
    if r.pos != 0 {
        return bad("a Huffman stream of another length");
    }
    Ok(())
}

// ── sequences ───────────────────────────────────────────────────────────────────────────────────

const LL_BASE: [(u32, u8); 36] = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
    (13, 0), (14, 0), (15, 0), (16, 1), (18, 1), (20, 1), (22, 1), (24, 2), (28, 2), (32, 3), (40, 3), (48, 4), (64, 6), (128, 7), (256, 8),
    (512, 9), (1024, 10), (2048, 11), (4096, 12), (8192, 13), (16384, 14), (32768, 15), (65536, 16)];

const ML_BASE: [(u32, u8); 53] = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0),
    (31, 0), (32, 0), (33, 0), (34, 0), (35, 1), (37, 1), (39, 1), (41, 1), (43, 2), (47, 2), (51, 3), (59, 3), (67, 4), (83, 4), (99, 5),
    (131, 7), (259, 8), (515, 9), (1027, 10), (2051, 11), (4099, 12), (8195, 13), (16387, 14), (32771, 15), (65539, 16)];

const LL_DEFAULT: [i32; 36] = [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1];
const ML_DEFAULT: [i32; 53] = [1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1];
const OF_DEFAULT: [i32; 29] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1];

/// What a frame carries from one block to the next.
struct Carry {
    huffman: Option<Huffman>,
    ll: Option<Fse>,
    of: Option<Fse>,
    ml: Option<Fse>,
    reps: [usize; 3],
}

fn table_of(mode: u8, data: &[u8], at: &mut usize, default: &[i32], log: usize, max_symbol: usize, max_log: usize,
            previous: &Option<Fse>) -> Result<Fse, Corrupt> {
    Ok(match mode {
        0 => build_fse(default, log),
        1 => {
            let s = *data.get(*at).ok_or(Corrupt("a cut RLE table".into()))?;
            *at += 1;
            if s as usize > max_symbol {
                return bad("an RLE symbol past its code");
            }
            rle_fse(s)
        }
        2 => {
            let (counts, log, taken) = read_counts(&data[*at..], max_symbol, max_log)?;
            *at += taken;
            build_fse(&counts, log)
        }
        _ => previous.clone().ok_or(Corrupt("a repeated table with none before".into()))?,
    })
}

fn literals(data: &[u8], carry: &mut Carry) -> Result<(Vec<u8>, usize), Corrupt> {
    let b0 = *data.first().ok_or(Corrupt("no literals section".into()))?;
    let kind = b0 & 3;
    let format = (b0 >> 2) & 3;
    let byte = |i: usize| -> Result<u32, Corrupt> { data.get(i).map(|b| *b as u32).ok_or(Corrupt("a cut literals header".into())) };
    if kind <= 1 {
        let (size, head) = match format {
            0 | 2 => ((b0 >> 3) as usize, 1),
            1 => (((b0 as u32 >> 4) + (byte(1)? << 4)) as usize, 2),
            _ => (((b0 as u32 >> 4) + (byte(1)? << 4) + (byte(2)? << 12)) as usize, 3),
        };
        if kind == 0 {
            let lit = data.get(head..head + size).ok_or(Corrupt("cut raw literals".into()))?;
            return Ok((lit.to_vec(), head + size));
        }
        let b = *data.get(head).ok_or(Corrupt("cut RLE literals".into()))?;
        return Ok((vec![b; size], head + 1));
    }
    let (regen, comp, head, streams) = match format {
        0 | 1 => {
            let v = (b0 as u32) | (byte(1)? << 8) | (byte(2)? << 16);
            ((v >> 4) & 0x3FF, (v >> 14) & 0x3FF, 3, if format == 0 { 1 } else { 4 })
        }
        2 => {
            let v = (b0 as u32) | (byte(1)? << 8) | (byte(2)? << 16) | (byte(3)? << 24);
            ((v >> 4) & 0x3FFF, (v >> 18) & 0x3FFF, 4, 4)
        }
        _ => {
            let v = (b0 as u64) | ((byte(1)? as u64) << 8) | ((byte(2)? as u64) << 16) | ((byte(3)? as u64) << 24) | ((byte(4)? as u64) << 32);
            (((v >> 4) & 0x3FFFF) as u32, ((v >> 22) & 0x3FFFF) as u32, 5, 4)
        }
    };
    let (regen, comp) = (regen as usize, comp as usize);
    let body = data.get(head..head + comp).ok_or(Corrupt("cut compressed literals".into()))?;
    let mut at = 0;
    if kind == 2 {
        let (h, used) = read_huffman(body)?;
        carry.huffman = Some(h);
        at = used;
    }
    let h = carry.huffman.clone().ok_or(Corrupt("treeless literals with no tree before".into()))?;
    let mut out = Vec::with_capacity(regen);
    if streams == 1 {
        huffman_stream(&h, &body[at..], regen, &mut out)?;
    } else {
        let jump = body.get(at..at + 6).ok_or(Corrupt("a cut jump table".into()))?;
        let sizes = [u16::from_le_bytes([jump[0], jump[1]]) as usize, u16::from_le_bytes([jump[2], jump[3]]) as usize,
                     u16::from_le_bytes([jump[4], jump[5]]) as usize];
        let mut p = at + 6;
        let per = regen.div_ceil(4);
        for (i, size) in sizes.iter().enumerate() {
            let s = body.get(p..p + size).ok_or(Corrupt("a cut literals stream".into()))?;
            huffman_stream(&h, s, per, &mut out)?;
            p += size;
            let _ = i;
        }
        let last = regen.checked_sub(3 * per).ok_or(Corrupt("literals of an impossible size".into()))?;
        huffman_stream(&h, &body[p..], last, &mut out)?;
    }
    Ok((out, head + comp))
}

fn block(data: &[u8], out: &mut Vec<u8>, carry: &mut Carry, limit: usize) -> Result<(), Corrupt> {
    let (lits, used) = literals(data, carry)?;
    let mut at = used;
    let b0 = *data.get(at).ok_or(Corrupt("no sequences section".into()))? as usize;
    at += 1;
    let count = if b0 < 128 {
        b0
    } else if b0 < 255 {
        let b1 = *data.get(at).ok_or(Corrupt("a cut sequence count".into()))? as usize;
        at += 1;
        ((b0 - 128) << 8) + b1
    } else {
        let b = data.get(at..at + 2).ok_or(Corrupt("a cut sequence count".into()))?;
        at += 2;
        b[0] as usize + ((b[1] as usize) << 8) + 0x7F00
    };
    if count == 0 {
        out.extend_from_slice(&lits);
        return if out.len() > limit { bad("the content is larger than allowed") } else { Ok(()) };
    }
    let modes = *data.get(at).ok_or(Corrupt("no table modes".into()))?;
    at += 1;
    let ll = table_of(modes >> 6, data, &mut at, &LL_DEFAULT, 6, 35, 9, &carry.ll)?;
    let of = table_of((modes >> 4) & 3, data, &mut at, &OF_DEFAULT, 5, 31, 8, &carry.of)?;
    let ml = table_of((modes >> 2) & 3, data, &mut at, &ML_DEFAULT, 6, 52, 9, &carry.ml)?;
    let mut r = Backward::new(&data[at..])?;
    {
        let mut sl = State::new(&ll, &mut r);
        let mut so = State::new(&of, &mut r);
        let mut sm = State::new(&ml, &mut r);
        let mut lit_at = 0usize;
        for i in 0..count {
            let of_code = so.symbol() as usize;
            let ll_code = sl.symbol() as usize;
            let ml_code = sm.symbol() as usize;
            if of_code > 31 || ll_code > 35 || ml_code > 52 {
                return bad("a sequence code out of range");
            }
            let offset_value = (1usize << of_code) + r.bits(of_code) as usize;
            let (mb, mx) = ML_BASE[ml_code];
            let match_len = mb as usize + r.bits(mx as usize) as usize;
            let (lb, lx) = LL_BASE[ll_code];
            let lit_len = lb as usize + r.bits(lx as usize) as usize;
            let reps = &mut carry.reps;
            let offset = if offset_value > 3 {
                let o = offset_value - 3;
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = o;
                o
            } else {
                let idx = if lit_len == 0 { offset_value } else { offset_value - 1 };
                let o = match idx {
                    0 => reps[0],
                    3 => reps[0].wrapping_sub(1),
                    k => reps[k],
                };
                if idx != 0 {
                    if idx != 1 {
                        reps[2] = reps[1];
                    }
                    reps[1] = reps[0];
                    reps[0] = o;
                }
                o
            };
            let lits_here = lits.get(lit_at..lit_at + lit_len).ok_or(Corrupt("a sequence past its literals".into()))?;
            out.extend_from_slice(lits_here);
            lit_at += lit_len;
            if offset == 0 || offset > out.len() {
                return bad("a match before the start");
            }
            if out.len() + match_len > limit {
                return bad("the content is larger than allowed");
            }
            let start = out.len() - offset;
            for k in 0..match_len {
                let b = out[start + k];
                out.push(b);
            }
            if i + 1 < count {
                sl.update(&mut r);
                sm.update(&mut r);
                so.update(&mut r);
            }
        }
        out.extend_from_slice(&lits[lit_at..]);
    }
    if r.pos != 0 {
        return bad("a sequences stream of another length");
    }
    carry.ll = Some(ll);
    carry.of = Some(of);
    carry.ml = Some(ml);
    if out.len() > limit { bad("the content is larger than allowed") } else { Ok(()) }
}

/// Every frame of `data`, decompressed, at most `limit` bytes.
pub fn decompress(data: &[u8], limit: usize) -> Result<Vec<u8>, Corrupt> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let magic = u32::from_le_bytes(data.get(at..at + 4).ok_or(Corrupt("a cut frame".into()))?.try_into().unwrap());
        at += 4;
        if (0x184D_2A50..=0x184D_2A5F).contains(&magic) {
            let size = u32::from_le_bytes(data.get(at..at + 4).ok_or(Corrupt("a cut skippable frame".into()))?.try_into().unwrap()) as usize;
            at += 4 + size;
            continue;
        }
        if magic != MAGIC {
            return bad("not a Zstandard frame");
        }
        let fhd = *data.get(at).ok_or(Corrupt("a cut frame header".into()))?;
        at += 1;
        let fcs_flag = fhd >> 6;
        let single = fhd & 0x20 != 0;
        let checksum = fhd & 4 != 0;
        let dict_flag = fhd & 3;
        if fhd & 8 != 0 {
            return bad("a reserved frame flag");
        }
        if !single {
            at += 1;
        }
        let dict_bytes = [0, 1, 2, 4][dict_flag as usize];
        if dict_bytes > 0 {
            let id = data.get(at..at + dict_bytes).ok_or(Corrupt("a cut dictionary id".into()))?;
            if id.iter().any(|b| *b != 0) {
                return bad("a frame that needs a dictionary");
            }
            at += dict_bytes;
        }
        let fcs_bytes = match fcs_flag {
            0 => usize::from(single),
            1 => 2,
            2 => 4,
            _ => 8,
        };
        at += fcs_bytes;
        let mut carry = Carry { huffman: None, ll: None, of: None, ml: None, reps: [1, 4, 8] };
        loop {
            let h = data.get(at..at + 3).ok_or(Corrupt("a cut block header".into()))?;
            let header = h[0] as u32 | (h[1] as u32) << 8 | (h[2] as u32) << 16;
            at += 3;
            let last = header & 1 != 0;
            let kind = (header >> 1) & 3;
            let size = (header >> 3) as usize;
            match kind {
                0 => {
                    let raw = data.get(at..at + size).ok_or(Corrupt("a cut raw block".into()))?;
                    out.extend_from_slice(raw);
                    at += size;
                }
                1 => {
                    let b = *data.get(at).ok_or(Corrupt("a cut RLE block".into()))?;
                    out.extend(std::iter::repeat_n(b, size));
                    at += 1;
                }
                2 => {
                    let body = data.get(at..at + size).ok_or(Corrupt("a cut compressed block".into()))?;
                    block(body, &mut out, &mut carry, limit)?;
                    at += size;
                }
                _ => return bad("a reserved block type"),
            }
            if out.len() > limit {
                return bad("the content is larger than allowed");
            }
            if last {
                break;
            }
        }
        if checksum {
            at += 4;
        }
    }
    Ok(out)
}
