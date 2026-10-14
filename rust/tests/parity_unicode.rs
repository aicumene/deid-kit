// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! The Unicode layer answers as Python 3.13's does: every code point's normalisations, category,
//! combining class and str predicates, and 30,000 random mixed strings through every normal form and
//! case method. Needs `DEIDKIT_PARITY=<dir>` after `python tools/vectors_unicode.py <dir>`.

use deidkit::unicode as u;
use serde_json::Value;

const CATS: [&str; 30] = ["Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
                          "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co"];

#[test]
fn every_code_point_and_string_as_python() {
    let Some(dir) = std::env::var_os("DEIDKIT_PARITY") else {
        eprintln!("DEIDKIT_PARITY not set: skipped");
        return;
    };
    let v: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(&dir).join("unicode.json")).unwrap()).unwrap();
    let cps: Vec<u32> = (0..0x110000u32).filter(|c| !(0xD800..=0xDFFF).contains(c)).collect();
    let mut bad: Vec<String> = Vec::new();
    for form in ["NFC", "NFD", "NFKC", "NFKD"] {
        let expected: std::collections::HashMap<u32, &str> = v["single"][form].as_array().unwrap().iter()
            .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_str().unwrap())).collect();
        for &cp in &cps {
            let c = char::from_u32(cp).unwrap();
            let want = expected.get(&cp).map_or_else(|| c.to_string(), |s| s.to_string());
            let got = u::normalize(form, &c.to_string());
            if got != want && bad.len() < 40 {
                bad.push(format!("{form} U+{cp:04X}: python {want:?} rust {got:?}"));
            }
        }
    }
    let cats: Vec<char> = v["category"].as_str().unwrap().chars().collect();
    for (i, &cp) in cps.iter().enumerate() {
        let c = char::from_u32(cp).unwrap();
        let want = CATS[(cats[i] as u32 - 0x40) as usize];
        if u::category(c) != want && bad.len() < 80 {
            bad.push(format!("category U+{cp:04X}: python {want} rust {}", u::category(c)));
        }
    }
    let ccc: std::collections::HashMap<u32, u8> = v["combining"].as_array().unwrap().iter()
        .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_u64().unwrap() as u8)).collect();
    for &cp in &cps {
        let c = char::from_u32(cp).unwrap();
        if u::combining(c) != ccc.get(&cp).copied().unwrap_or(0) && bad.len() < 120 {
            bad.push(format!("combining U+{cp:04X}"));
        }
    }
    let preds: [(&str, fn(&str) -> bool); 9] = [("isalpha", u::isalpha), ("isalnum", u::isalnum), ("isdigit", u::isdigit),
        ("isdecimal", u::isdecimal), ("isnumeric", u::isnumeric), ("isspace", u::isspace), ("isupper", u::isupper),
        ("islower", u::islower), ("isprintable", |s| s.chars().all(u::is_printable))];
    for (name, f) in preds {
        let bits: Vec<char> = v["per_char"][name].as_str().unwrap().chars().collect();
        for (i, &cp) in cps.iter().enumerate() {
            let c = char::from_u32(cp).unwrap().to_string();
            if f(&c) != (bits[i] == '1') && bad.len() < 160 {
                bad.push(format!("{name} U+{cp:04X}: python {}", bits[i]));
            }
        }
    }
    let strings: Vec<&str> = v["strings"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    for form in ["NFC", "NFD", "NFKC", "NFKD"] {
        for (i, s) in strings.iter().enumerate() {
            let want = v["normal"][form][i].as_str().unwrap();
            let got = u::normalize(form, s);
            if got != want && bad.len() < 200 {
                bad.push(format!("{form} {s:?}: python {want:?} rust {got:?}"));
            }
        }
    }
    let texts: [(&str, fn(&str) -> String); 5] = [("lower", u::lower), ("upper", u::upper), ("casefold", u::casefold),
                                                   ("capitalize", u::capitalize), ("title", u::title)];
    for (name, f) in texts {
        for (i, s) in strings.iter().enumerate() {
            let want = v["methods"][name][i].as_str().unwrap();
            if f(s) != want && bad.len() < 240 {
                bad.push(format!("{name} {s:?}: python {want:?} rust {:?}", f(s)));
            }
        }
    }
    for (name, f) in preds {
        for (i, s) in strings.iter().enumerate() {
            let want = v["methods"][name][i].as_bool().unwrap();
            let got = if name == "isprintable" { s.chars().all(u::is_printable) } else { f(s) };
            if got != want && bad.len() < 280 {
                bad.push(format!("{name} {s:?}: python {want}"));
            }
        }
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.join("\n"));
}
