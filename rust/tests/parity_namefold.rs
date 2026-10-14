// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The name fold answers as deidkit.namefold's Python fold does: the fold with its offsets, the confusable
//! and diacritic folds, the key, matching with word boundaries, the variants, the admission gate.
//! Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_namefold.py <dir>`.

mod common;

use common::Diff;
use deidkit::{namefold as nf, textmatch as tm};
use serde_json::{json, Value};

#[test]
fn the_fold_is_pythons() {
    let Some((_, v)) = common::vectors("namefold.json") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let mut d = Diff::new();
    let texts: Vec<String> = common::strings(&v["texts"]);
    for (i, t) in texts.iter().enumerate() {
        d.eq(format!("normalize {t:?}"), &v["normalize"][i], &json!(tm::normalize(t)));
        let f = nf::fold(t);
        d.eq(format!("fold {t:?}"), &v["fold"][i], &json!([f.text, f.starts, f.ends]));
        d.eq(format!("confusable {t:?}"), &v["confusable"][i], &json!(nf::confusable(&f.text)));
        d.eq(format!("diacritics {t:?}"), &v["diacritics"][i], &json!(nf::diacritics(&f.text)));
        d.eq(format!("key {t:?}"), &v["key"][i], &json!(nf::key(t)));
        let (h, hf) = nf::fold_haystack(t);
        d.eq(format!("fold_haystack {t:?}"), &v["fold_haystack"][i], &json!([h, hf.text, hf.starts, hf.ends]));
        d.eq(format!("deconfuse {t:?}"), &v["deconfuse"][i], &json!(nf::deconfuse_ascii(t)));
        let spans: Vec<Value> = nf::token_in_text_all(t).into_iter().map(|(a, b)| json!([a, b])).collect();
        d.eq(format!("token_in_text {t:?}"), &v["token_in_text"][i], &Value::Array(spans));
        let terms: Vec<Value> = nf::defined_terms(t).into_iter().map(|(a, b)| json!([a, b])).collect();
        d.eq(format!("defined_terms {t:?}"), &v["defined_terms"][i], &Value::Array(terms));
    }
    for x in v["find_all"].as_array().unwrap() {
        let (h, n) = (x[0].as_str().unwrap(), x[1].as_str().unwrap());
        let got: Vec<Value> = nf::find_all(&nf::fold_haystack(h).0, &nf::key(n)).into_iter().map(|(a, b)| json!([a, b])).collect();
        d.eq(format!("find_all {h:?} {n:?}"), &x[2], &Value::Array(got));
    }
    let names: Vec<String> = common::strings(&v["names_for"]);
    let per = &v["per_name"];
    for (i, n) in names.iter().enumerate() {
        let n = n.as_str();
        let checks: Vec<(&str, Value)> = vec![
            ("name_parts", json!(nf::name_parts(n))),
            ("skeleton", json!(nf::skeleton(n))),
            ("org_variants", json!(nf::org_variants(n))),
            ("person_variants", json!(nf::person_variants(n))),
            ("surname_first", json!(nf::surname_first(n))),
            ("person_fragments", json!(nf::person_fragments(n).into_iter().map(|(s, r)| json!([s, r])).collect::<Vec<_>>())),
            ("distinctive_head", json!(nf::distinctive_head(n))),
            ("admit_person", { let (ok, why) = nf::admit_person(n); json!([ok, why]) }),
            ("script_of", json!(nf::script_of(n))),
            ("mixed_script", json!(nf::mixed_script(n))),
            ("is_generic", json!(nf::is_generic(n))),
            ("is_common_word", json!(nf::is_common_word(n))),
            ("is_honorific", json!(nf::is_honorific(n))),
            ("is_patronymic", json!(nf::is_patronymic(n))),
            ("is_initial", json!(nf::is_initial(n))),
            ("legal_form_index", json!(nf::legal_form_index(n))),
            ("has_legal_form", json!(nf::has_legal_form(n))),
            ("strip_legal_form", json!(nf::strip_legal_form(n))),
            ("russian_inflections", json!(nf::russian_inflections(n))),
            ("acronym_cyrillic", json!(nf::acronym_cyrillic(n))),
            ("translit_variants", json!(nf::translit_variants(n))),
            ("cyrillic_form", json!(nf::cyrillic_form(n))),
            ("skeleton_part", json!(nf::skeleton_part(n))),
            ("translit", json!(nf::translit(n))),
            ("spelling_variants", json!(nf::spelling_variants(n, false))),
            ("spelling_variants_gen", json!(nf::spelling_variants(n, true))),
        ];
        for (name, got) in checks {
            d.eq(format!("{name} {n:?}"), &per[name][i], &got);
        }
    }
    for x in v["stem_match"].as_array().unwrap() {
        d.eq(format!("stem_match {x}"), &x[2], &json!(nf::stem_match(x[0].as_str().unwrap(), x[1].as_str().unwrap())));
    }
    for x in v["defines_alias"].as_array().unwrap() {
        d.eq(format!("defines_alias {x}"), &x[2], &json!(nf::defines_alias(x[0].as_str().unwrap(), x[1].as_str().unwrap())));
    }
    d.done();
}
