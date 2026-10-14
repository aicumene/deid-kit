# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what Python's unicodedata and str methods give, for the Rust ones to give the same.

    python tools/vectors_unicode.py OUT_DIR   → OUT_DIR/unicode.json
"""
import json
import random
import sys
import unicodedata

FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def main(out: str) -> None:
    rnd = random.Random(1410)
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    single = {f: [[cp, unicodedata.normalize(f, chr(cp))] for cp in cps if unicodedata.normalize(f, chr(cp)) != chr(cp)]
              for f in FORMS}
    marks = [cp for cp in cps if unicodedata.combining(chr(cp))]
    pool = ([chr(c) for c in range(0x20, 0x250)] + [chr(c) for c in range(0x370, 0x500)] + [chr(c) for c in marks[:400]]
            + [chr(c) for c in range(0x1100, 0x1200)] + [chr(c) for c in range(0xAC00, 0xAC80)] + list("ΣΣσςİıﬁ ẞß.'’:")
            + [chr(c) for c in range(0x1E00, 0x1F00)] + [chr(rnd.choice(cps)) for _ in range(300)])
    strings = ["".join(rnd.choice(pool) for _ in range(rnd.randint(1, 12))) for _ in range(30000)]
    strings += ["ΣΑΣ", "ΟΔΟΣ ΣΑΣ.", "Σ", "aΣ", "aΣb", "aΣ'", "a'Σ", "ΌΣΟΣ", "ﬁne", "İstanbul", "ǅ", "ǆemal",
                "Straße", "TARVELO DENİZCİLİK", "", " ", "ᾼ", "ǰ", "ŉ"]
    methods = {}
    for name in ("lower", "upper", "casefold", "capitalize", "title", "isupper", "islower", "isalpha", "isalnum",
                 "isdigit", "isdecimal", "isnumeric", "isspace", "isprintable"):
        methods[name] = [getattr(s, name)() for s in strings]
    per_char = {name: "".join("1" if getattr(chr(cp), name)() else "0" for cp in cps)
                for name in ("isalpha", "isalnum", "isdigit", "isdecimal", "isnumeric", "isspace", "isupper", "islower", "isprintable")}
    data = {"single": single, "strings": strings, "normal": {f: [unicodedata.normalize(f, s) for s in strings] for f in FORMS},
            "methods": methods, "per_char": per_char,
            "category": "".join(chr(0x40 + ["Cn", "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd",
                                             "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf",
                                             "Cs", "Co"].index(unicodedata.category(chr(cp)))) for cp in cps),
            "combining": [[cp, unicodedata.combining(chr(cp))] for cp in cps if unicodedata.combining(chr(cp))]}
    with open(f"{out}/unicode.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1])
