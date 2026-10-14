// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Command lines read as Python's `argparse` reads them: `--name value` and `--name=value`, a unique
//! prefix for a long name, `-h`/`--help`, and the errors on stderr with exit status 2.

use std::collections::HashMap;

pub struct Opt {
    pub name: &'static str,
    /// Takes a value (else it is a switch).
    pub value: bool,
    pub default: Option<&'static str>,
    pub choices: &'static [&'static str],
    pub help: &'static str,
    pub metavar: &'static str,
    pub required: bool,
}

pub const fn opt(name: &'static str, default: &'static str, help: &'static str) -> Opt {
    Opt { name, value: true, default: Some(default), choices: &[], help, metavar: "", required: false }
}

pub const fn flag(name: &'static str, help: &'static str) -> Opt {
    Opt { name, value: false, default: None, choices: &[], help, metavar: "", required: false }
}

pub struct Args {
    values: HashMap<&'static str, String>,
    flags: Vec<&'static str>,
}

impl Args {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    pub fn str(&self, name: &str) -> String {
        self.get(name).unwrap_or("").to_string()
    }

    pub fn on(&self, name: &str) -> bool {
        self.flags.contains(&name)
    }
}

fn dest(name: &str) -> String {
    name.trim_start_matches('-').replace('-', "_").to_uppercase()
}

fn usage(prog: &str, opts: &[Opt]) -> String {
    let parts: Vec<String> = opts.iter().map(|o| {
        let shown = if !o.value {
            o.name.to_string()
        } else if !o.choices.is_empty() {
            format!("{} {{{}}}", o.name, o.choices.join(","))
        } else {
            format!("{} {}", o.name, if o.metavar.is_empty() { dest(o.name) } else { o.metavar.to_string() })
        };
        if o.required { shown } else { format!("[{shown}]") }
    }).collect();
    format!("usage: {prog} [-h] {}", parts.join(" "))
}

fn fail(prog: &str, opts: &[Opt], message: &str) -> ! {
    eprintln!("{}\n{prog}: error: {message}", usage(prog, opts));
    std::process::exit(2);
}

pub fn parse(prog: &str, description: &str, opts: &[Opt], argv: &[String]) -> Args {
    let mut args = Args { values: HashMap::new(), flags: Vec::new() };
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a == "-h" || a == "--help" {
            println!("{}\n\n{description}\n\noptions:\n  -h, --help            show this help message and exit", usage(prog, opts));
            for o in opts {
                let head = if o.value { format!("{} {}", o.name, if o.metavar.is_empty() { dest(o.name) } else { o.metavar.to_string() }) }
                           else { o.name.to_string() };
                println!("  {head:<22}{}", o.help);
            }
            std::process::exit(0);
        }
        if !a.starts_with("--") {
            fail(prog, opts, &format!("unrecognized arguments: {}", argv[i..].join(" ")));
        }
        let (given, inline) = match a.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (a.as_str(), None),
        };
        let exact = opts.iter().find(|o| o.name == given);
        let found = exact.or_else(|| {
            let matches: Vec<&Opt> = opts.iter().filter(|o| o.name.starts_with(given)).collect();
            match matches.len() {
                1 => Some(matches[0]),
                0 => None,
                _ => fail(prog, opts, &format!("ambiguous option: {given} could match {}",
                                               matches.iter().map(|o| o.name).collect::<Vec<_>>().join(", "))),
            }
        });
        let Some(o) = found else { fail(prog, opts, &format!("unrecognized arguments: {a}")) };
        if !o.value {
            if inline.is_some() {
                fail(prog, opts, &format!("argument {}: ignored explicit argument {}", o.name, crate::py::repr_str(inline.as_deref().unwrap())));
            }
            args.flags.push(o.name);
            i += 1;
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => {
                i += 1;
                match argv.get(i) {
                    Some(v) if !(v.starts_with('-') && v.len() > 1 && v.chars().nth(1).is_some_and(|c| !c.is_ascii_digit())) => v.clone(),
                    _ => fail(prog, opts, &format!("argument {}: expected one argument", o.name)),
                }
            }
        };
        if !o.choices.is_empty() && !o.choices.contains(&value.as_str()) {
            fail(prog, opts, &format!("argument {}: invalid choice: {} (choose from {})", o.name, crate::py::repr_str(&value),
                                      o.choices.iter().map(|c| crate::py::repr_str(c)).collect::<Vec<_>>().join(", ")));
        }
        args.values.insert(o.name, value);
        i += 1;
    }
    for o in opts {
        if o.required && !args.values.contains_key(o.name) {
            fail(prog, opts, &format!("the following arguments are required: {}", o.name));
        }
        if let (Some(d), false) = (o.default, args.values.contains_key(o.name)) {
            args.values.insert(o.name, d.to_string());
        }
    }
    args
}

/// A value that must be a number, or the error argparse gives.
pub fn number<T: std::str::FromStr>(prog: &str, opts: &[Opt], name: &str, value: &str, kind: &str) -> T {
    crate::py::strip(value).parse::<T>().unwrap_or_else(|_| {
        fail(prog, opts, &format!("argument {name}: invalid {kind} value: {}", crate::py::repr_str(value)))
    })
}

pub fn error(prog: &str, opts: &[Opt], message: &str) -> ! {
    fail(prog, opts, message)
}
