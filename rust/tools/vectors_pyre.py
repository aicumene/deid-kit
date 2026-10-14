# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what Python's re answers, for the Rust pyre to answer the same.

    python tools/vectors_pyre.py OUT_DIR   → OUT_DIR/pyre.json

Random patterns from a grammar of every feature deid-kit uses, over random texts; and every literal
pattern in deid-kit's code over texts of the kind it reads.
"""
import ast
import json
import random
import re
import signal
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "deidkit"
ALPHA = list("abcAB \n_1-.Äſ") + ["K"]


def rand_pattern(rnd, depth=0):
    def atom():
        r = rnd.random()
        if r < 0.30:
            return re.escape(rnd.choice("abcAB _1-.Ä")) if rnd.random() < 0.9 else rnd.choice(["\\n", "\\t", "\\x41", "\\u00c4", "\\\\"])
        if r < 0.45:
            return rnd.choice(["[ab]", "[^a]", "[a-c]", "[A-Za-z]", "[\\d.]", "[^\\W_]", "[-a]", "[a-]", "[]a]", "[^]]", "[\\s\\S]", "[ä-ö]", "[\\w-]"])
        if r < 0.58:
            return rnd.choice(["\\d", "\\w", "\\s", "\\W", "\\D", "\\S", "."])
        if r < 0.66:
            return rnd.choice(["^", "$", "\\b", "\\B", "\\A", "\\Z"])
        if depth > 2:
            return "a"
        inner = rand_pattern(rnd, depth + 1)
        kind = rnd.choice(["(", "(?:", "(?P<g%d>" % rnd.randint(0, 99), "(?=", "(?!", "(?<=", "(?<!", "(?>", "(?i:", "(?-i:"])
        if kind in ("(?<=", "(?<!"):
            inner = rnd.choice(["a", "ab", "[ab]", "\\d", "a|b", "(?:ab|cd)", "\\.", "\\w\\w"])
        return kind + inner + ")"

    parts = []
    for _ in range(rnd.randint(1, 4)):
        a = atom()
        if rnd.random() < 0.35:
            a += rnd.choice(["*", "+", "?", "{2}", "{1,3}", "{,2}", "{2,}", "{0,1}"]) + rnd.choice(["", "", "?", "+"])
        parts.append(a)
    p = "".join(parts)
    if rnd.random() < 0.15:
        p += "|" + rand_pattern(rnd, depth + 1)
    if rnd.random() < 0.08:
        p += "\\1"
    return p


def literal_patterns():
    out = []
    for f in SRC.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "re" and node.args:
                try:
                    p = ast.literal_eval(node.args[0])
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(p, str):
                    continue
                flags = 0
                for kw in node.keywords:
                    if kw.arg == "flags":
                        try:
                            flags = eval(ast.unparse(kw.value), {"re": re})
                        except Exception:  # noqa: BLE001
                            pass
                if node.func.attr in ("compile", "search", "match", "findall", "finditer", "split", "fullmatch") and len(node.args) > 1:
                    try:
                        flags = eval(ast.unparse(node.args[-1]), {"re": re})
                    except Exception:  # noqa: BLE001
                        pass
                if not isinstance(flags, int):
                    flags = 0
                out.append((p, int(flags)))
    return sorted(set(out))


DOCS = [
    "Dear Ms Voss,\\n\\nBrightwater Maritime Ltd asks — lena.voss@example.org, noreply@x.io, +44 20 7946 0958.",
    "IBAN DE89 3704 0044 0532 0130 00 EUR and GB82WEST12345698765432; card 4111 1111 1111 1111, 5500-0000-0000-0004.",
    "Musterweg 12, 2. OG links — 20095 Hamburg; Am Alten Hafen 3; Große Bleichen 12; 221B Baker Street; Flat 3, 12 Park Lane; SW1A 2AA; A-1010 Wien.",
    "TOBIAS WREN (\"TW\") and KESTREL VENTURES LLP (hereinafter “KV”); MK signed; M.K. and M. K. too. 12000 Euro, 28. August 2026.",
    "PERSON_1 and ORG_12 met PERSON_3x; PERSON_١٢ and _4. F1: the rent was reviewed); F3 (wrong date), F7",
    "## 2. Rent\n\n2.1 The rent is EUR 18,000 a year.\n**Re: Notice**\n  12→line text\n\tfoo | bar",
    "Иван Петрович Сидоров, ООО «Ромашка», ИНН 7701234567; Straße, STRASSE; ΣΑΣ σας; ﬁne; İstanbul.",
    "", "\n", "a\nb\n", "x" * 50 + "@" + "y" * 30,
]


class Slow(Exception):
    pass


def _slow(*_):
    raise Slow


def run(pattern, flags, text):
    """Python's answers, or None for a pattern that backtracks for too long (skipped on both sides)."""
    signal.signal(signal.SIGALRM, _slow)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        return _run(pattern, flags, text)
    except Slow:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _run(pattern, flags, text):
    try:
        rx = re.compile(pattern, flags)
    except re.error:
        return {"error": True}
    except (OverflowError, RecursionError):
        return None
    out = {"finditer": [[m.start(), m.end(), [list(m.span(g)) for g in range(1, rx.groups + 1)]] for m in rx.finditer(text)]}
    fm = rx.fullmatch(text)
    out["fullmatch"] = None if fm is None else [fm.start(), fm.end()]
    out["split"] = rx.split(text)
    try:
        out["sub"] = rx.sub("<\\g<0>>", text)
    except re.error:
        out["sub"] = None
    return out


def main(out_dir, seed=1410, count=4000, longest=10):
    rnd = random.Random(seed)
    cases = []
    for _ in range(count):
        p = rand_pattern(rnd)
        flags = rnd.choice([0, 0, 0, re.I, re.M, re.S, re.I | re.M])
        for _ in range(6):
            text = "".join(rnd.choice(ALPHA) for _ in range(rnd.randint(0, longest)))
            r = run(p, flags, text)
            if r is not None:
                cases.append({"p": p, "f": flags, "t": text, **r})
    lits = literal_patterns()
    for p, flags in lits:
        for text in DOCS:
            r = run(p, flags, text)
            if r is not None:
                cases.append({"p": p, "f": flags, "t": text, **r})
    Path(out_dir, "pyre.json").write_text(json.dumps({"cases": cases, "literal_patterns": len(lits)}, ensure_ascii=False), encoding="utf-8")
    print(len(cases), "cases,", len(lits), "patterns from deid-kit")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
