// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What a task cost at Anthropic's API prices — with the prompt cache, and as it would without it
//! (`deidkit/agents/pricing.py`). Prices: Anthropic's first-party list prices, $ per million tokens,
//! as of 25 September 2026; a model not in the table is counted in tokens only.

use serde_json::{json, Map, Value};

use crate::py;

/// model id prefix → (input, output, cache read), $ per million tokens
pub const PRICES: [(&str, (f64, f64, f64)); 12] = [
    ("claude-fable-5-1", (10.0, 50.0, 0.25)),
    ("claude-mythos-5-1", (10.0, 50.0, 0.25)),
    ("claude-fable-5", (10.0, 50.0, 1.00)),
    ("claude-opus-5-5", (4.0, 20.0, 0.20)),
    ("claude-opus-5", (5.0, 25.0, 0.50)),
    ("claude-opus-4-8", (5.0, 25.0, 0.50)),
    ("claude-opus-4-7", (5.0, 25.0, 0.50)),
    ("claude-opus-4-6", (5.0, 25.0, 0.50)),
    ("claude-sonnet-5-5", (2.0, 10.0, 0.20)),
    ("claude-sonnet-5", (2.0, 10.0, 0.20)),
    ("claude-sonnet-4-6", (3.0, 15.0, 0.30)),
    ("claude-haiku-4-5", (1.0, 5.0, 0.10)),
];
pub const WRITE_5M: f64 = 1.25;
pub const WRITE_1H: f64 = 2.0;

/// The prices of `model`: its id, or the longest known id it begins with (dated or variant ids).
pub fn price(model: &str) -> Option<(f64, f64, f64)> {
    PRICES.iter()
        .filter(|(p, _)| model == *p || model.starts_with(&format!("{p}-")) || model.starts_with(&format!("{p}[")))
        .fold(None, |best: Option<&(&str, (f64, f64, f64))>, x| match best {
            Some(b) if b.0.len() >= x.0.len() => Some(b),
            _ => Some(x),
        })
        .map(|(_, p)| *p)
}

/// One answer of the model, as the provider counted it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestUsage {
    pub model: String,
    pub input: i64,
    pub write_5m: i64,
    pub write_1h: i64,
    pub read: i64,
    pub output: i64,
}

/// A count as Python takes it: an int (not a bool) above zero, else 0.
fn n(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(x)) if x.is_i64() || x.is_u64() => x.as_i64().filter(|v| *v > 0).unwrap_or(0),
        _ => 0,
    }
}

impl RequestUsage {
    /// From the Messages API's `usage`; the split of the cache writes by lifetime comes in `cache_creation`.
    pub fn from_anthropic(usage: &Map<String, Value>, model: &str) -> RequestUsage {
        let split = usage.get("cache_creation").and_then(Value::as_object);
        let written = n(usage.get("cache_creation_input_tokens"));
        let long = split.map_or(0, |s| n(s.get("ephemeral_1h_input_tokens")));
        let mut short = split.map_or(written, |s| n(s.get("ephemeral_5m_input_tokens")));
        if split.is_some() && short + long < written {
            short = written - long;
        }
        RequestUsage { model: model.into(), input: n(usage.get("input_tokens")), write_5m: short, write_1h: long,
                       read: n(usage.get("cache_read_input_tokens")), output: n(usage.get("output_tokens")) }
    }

    /// From the Responses API's `usage` (Codex): its input includes the cached tokens.
    pub fn from_openai(usage: &Map<String, Value>, model: &str) -> RequestUsage {
        let details = usage.get("input_tokens_details").and_then(Value::as_object);
        let total = n(usage.get("input_tokens"));
        let cached = details.map_or(0, |d| n(d.get("cached_tokens"))).min(total);
        RequestUsage { model: model.into(), input: total - cached, read: cached, output: n(usage.get("output_tokens")),
                       ..Default::default() }
    }
}

/// An answer's usage as the proxy reported it: the Responses API's (marked `api: responses`) or the Messages API's.
pub fn request_usage(usage: &Map<String, Value>, model: &str) -> RequestUsage {
    if usage.get("api").and_then(Value::as_str) == Some("responses") {
        RequestUsage::from_openai(usage, model)
    } else {
        RequestUsage::from_anthropic(usage, model)
    }
}

/// `round(x, 4)` as Python has it: the correctly rounded decimal, ties to even.
pub fn round4(x: f64) -> f64 {
    py::round(x, 4)
}

/// The task's tokens and, when every model is priced, its cost with the cache and without it.
pub fn task_cost(requests: &[RequestUsage]) -> Value {
    let sum = |f: fn(&RequestUsage) -> i64| requests.iter().map(f).sum::<i64>();
    let mut models: Vec<String> = requests.iter().filter(|r| !r.model.is_empty()).map(|r| r.model.clone()).collect();
    models.sort();
    models.dedup();
    let mut out = json!({"requests": requests.len(), "models": models,
                         "input": sum(|r| r.input), "cache_write_5m": sum(|r| r.write_5m),
                         "cache_write_1h": sum(|r| r.write_1h), "cache_read": sum(|r| r.read), "output": sum(|r| r.output),
                         "usd": null, "usd_without_cache": null, "saved_usd": null});
    let priced: Vec<_> = requests.iter().map(|r| (r, price(&r.model))).collect();
    if requests.is_empty() || priced.iter().any(|(_, p)| p.is_none()) {
        return out;
    }
    let (mut with_cache, mut without) = (0.0f64, 0.0f64);
    for (r, p) in priced {
        let (pin, pout, pread) = p.unwrap();
        with_cache += (r.input as f64 * pin + r.write_5m as f64 * pin * WRITE_5M + r.write_1h as f64 * pin * WRITE_1H
            + r.read as f64 * pread + r.output as f64 * pout) / 1e6;
        without += ((r.input + r.write_5m + r.write_1h + r.read) as f64 * pin + r.output as f64 * pout) / 1e6;
    }
    out["usd"] = json!(round4(with_cache));
    out["usd_without_cache"] = json!(round4(without));
    out["saved_usd"] = json!(round4(without - with_cache));
    out
}
