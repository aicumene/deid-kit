# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what deidkit.namefold and deidkit.textmatch give, for the Rust ones to give the same.

    python tools/vectors_namefold.py OUT_DIR   → OUT_DIR/namefold.json   (DEIDKIT_NATIVE=0: the Python fold)
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ["DEIDKIT_NATIVE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deidkit import namefold as nf, textmatch as tm  # noqa: E402

NAMES = [
    "Ada Brenner", "Herrn Paul Winter", "Dr. Anna-Lena Müller", "MUELLER, JOERG", "Jörg Müller", "Albrecht Weiß",
    "Quorvane Shipping Ltd", "QUORVANE SHIPPING LIMITED", "S.B. Management", "Silver Bay Shipping Ltd",
    "Red River Trading B.V.", "Arven Holdings Ltd", "ARVEN HOLDING LIMITED", "NRTK Asia M6 Limited",
    "Tarvelo Denizcilik Ticaret Anonim Sirketi", "TARVELO DENİZCİLİK", "Northwind Investments Ltd.",
    "Harbour Point Holdings Ltd", "Wren Example GmbH", "Kestrel Ventures LLP", "Bank of Cyprus",
    "Иван Петрович Сидоров", "Сидоров", "Сидорова", "Петровский", "Ковалёв", "Мария Иванова", "Игорь", "НРТК",
    "Yevgeniy Kovalev", "Evgeny Kovalyov", "Henry Zielinski", "Dmitriy Khrushchev", "Aleksandr Ivanov", "Yuliya",
    "José Muñoz", "Jose Munoz", "Łukasz Żółć", "Ørsted", "Sea ﬁnder", "SEAFINDER", "SEA FINDER",
    "Agίoy Nikoλάoy", "ΗΕ 123456", "Νίκος Παπαδόπουλος", "Forwarded Message", "Machine Translated",
    "C.  Verifying", "Marlow Centre", "Argentina", "Container Ship", "PERSON_1@example.org", "PERSON_12",
    "the husband", "an unknown person", "A", "Mr Wren", "Tobias Wren", "TW", "K.V.", "the Company",
    "Insert Title", "Red River BV", "ООО Ромашка", "Ромашка", "Ms O'Brien-Smith", "Jean-Luc Picard",
    "van der Berg", "Hans van der Berg", "Brenner, Ada", "physi-\ncally", "Brightwater Maritime Ltd",
    "SEA RUNNER Container Ship", "Marta (Hallberg)", "X Æ A-12", "ǅemal Bijedić", "Straße", "STRASSE",
    "Oleksandr Kovalenko", "Aleksei", "Sergey Sergeyevich", "Ivanovich", "Petrovna", "ABC", "ABCDEFG",
    "  spaced   name  ", "", "Σίσυφος", "ΣΙΣΥΦΟΣ", "İstanbul Liman A.Ş.", "Firma­name", "Naïve Café",
]
HAYS = [
    "Certificate of incorporation of the Corporation.", "(Hallberg), and Hallbergs", "Ltd.followed",
    "Dear Ms Voss,\\n\\nBrightwater Maritime Ltd asks", "x\\\\nBrightwater", "\\u00a0Brightwater Maritime",
    "physi-\n  cally speaking — “quoted” ‘single’ «guillemets» ﬁne № ５",
    "TARVELO DENİZCİLİK and Tarvelo Denizcilik", "José Muñoz met Jose Munoz", "ΗΕ 123456 vs HE 123456",
    "PERSON_1 and PERSON_12x and ORG_3. and _4 and A_٣٤ and B_12٣a", "Line one\r\nLine two three",
    "Ünïcödé​text with soft­hyphen", "A  B\t\tC", "", "   ", "abc—\n\ndef", "x-\ny- z",
]
TERMS = ["TOBIAS WREN of 1 Street (\"TW\") and KESTREL VENTURES LLP (hereinafter “KV”) and (the \"Purchaser\")",
         "Herr Paul Winter (nachfolgend „Vermieter“) und (im Folgenden kurz 'Mieter')", "(together the 'Sellers')",
         "(the Company)", "(\"\")", "(hereinafter referred to as \"Kestrel\")"]


def main(out: str) -> None:
    rnd = random.Random(14)
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,-'’\"()\n\t_0123456789") + list(
        "äöüßÄÖÜéèñçøłİıΣσςαβγабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙ́̈­​ ﬁ№－—–")
    randoms = ["".join(rnd.choice(pool) for _ in range(rnd.randint(0, 24))) for _ in range(3000)]
    texts = NAMES + HAYS + TERMS + randoms
    v = {"texts": texts}
    v["normalize"] = [tm.normalize(t) for t in texts]
    v["fold"] = [[f.text, list(f.starts), list(f.ends)] for f in map(nf.fold, texts)]
    v["confusable"] = [nf.confusable(nf.fold(t).text) for t in texts]
    v["diacritics"] = [nf.diacritics(nf.fold(t).text) for t in texts]
    v["key"] = [nf.key(t) for t in texts]
    v["fold_haystack"] = [[h, f.text, list(f.starts), list(f.ends)] for h, f in map(nf.fold_haystack, texts)]
    v["deconfuse"] = [nf.deconfuse_ascii(t) for t in texts]
    v["token_in_text"] = [[[m.start(), m.end()] for m in nf.TOKEN_IN_TEXT.finditer(t)] for t in texts]
    v["defined_terms"] = [nf.defined_terms(t) for t in texts]
    pairs = [(h, n) for h in HAYS + NAMES[:30] for n in NAMES[:40]] + [(t, t[2:6]) for t in randoms[:500]]
    v["find_all"] = [[h, n, [list(p) for p in nf.find_all(nf.fold_haystack(h)[0], nf.key(n))]] for h, n in pairs]
    per = {}
    for name in ("name_parts", "skeleton", "org_variants", "person_variants", "surname_first", "person_fragments",
                 "distinctive_head", "admit_person", "script_of", "mixed_script", "is_generic", "is_common_word",
                 "is_honorific", "is_patronymic", "is_initial", "legal_form_index", "has_legal_form", "strip_legal_form",
                 "russian_inflections", "acronym_cyrillic", "translit_variants", "cyrillic_form", "skeleton_part",
                 "translit"):
        f = getattr(nf, name)
        res = []
        for t in NAMES + randoms[:600]:
            r = f(t)
            res.append(list(r) if isinstance(r, tuple) else r)
        per[name] = res
    per["spelling_variants"] = [nf.spelling_variants(t) for t in NAMES + randoms[:600]]
    per["spelling_variants_gen"] = [nf.spelling_variants(t, genitive=True) for t in NAMES + randoms[:600]]
    per["person_fragments"] = [[list(x) for x in r] for r in per["person_fragments"]]
    v["per_name"] = per
    v["names_for"] = NAMES + randoms[:600]
    v["stem_match"] = [[a, b, nf.stem_match(a, b)] for a in ["sidorov", "sidorova", "petr", "petrov", "abc", "abcdefgh"]
                       for b in ["sidorov", "sidorovym", "petrovskii", "abcd", "abcdefghijk", "sido"]]
    v["defines_alias"] = [[t, n, nf.defines_alias(t, n)] for t in ["TW", "T.W.", "KV", "K.V.", "Kestrel", "Mr Wren", "Wren",
                                                                 "the Company", "Ltd", "of", "BOC", "BC", "Tobias Wren", ""]
                          for n in ["TOBIAS WREN", "KESTREL VENTURES LLP", "Bank of Cyprus", "Ms Ada Brenner", "&", ""]]
    Path(out, "namefold.json").write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
