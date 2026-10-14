# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what deidkit's vault, detectors and address tokens give, for the Rust ones to give the same.

    python tools/vectors_vault.py OUT_DIR   → OUT_DIR/vault.json   (DEIDKIT_NATIVE=0)

Scopes with a fixed salt, seeded from entities and parties; a corpus tokenised in order with the regex
detector; answers detokenised; every row and alias of the store dumped after each step.
"""
import asyncio
import json
import os
import random
import sys
from pathlib import Path

os.environ["DEIDKIT_NATIVE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deidkit import address, patterns, vault  # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity  # noqa: E402
from deidkit.store import InMemoryTokenStore  # noqa: E402

SALT = bytes(range(32))

ENTITIES = {
    "case-a": [
        {"type": "individual", "name": "Ada Brenner", "role": "Director, Shareholder", "jurisdiction_country": "GB"},
        {"type": "individual", "name": "Henry Zielinski", "role": "Beneficial Owner"},
        {"type": "individual", "name": "Marek Zielinski"},
        {"type": "individual", "name": "Иван Петрович Сидоров", "role": "Director"},
        {"type": "individual", "name": "Jörg Müller"},
        {"type": "company", "name": "Quorvane Shipping Ltd", "jurisdiction_country": "CY", "identifier": "HE 123456",
         "identifier_type": "registration"},
        {"type": "company", "name": "Red River Trading Ltd"},
        {"type": "company", "name": "Red River Trading B.V."},
        {"type": "company", "name": "Silver Bay Shipping Ltd"},
        {"type": "company", "name": "Harbour Point Holdings Ltd"},
        {"type": "vessel", "name": "SEA RUNNER", "attributes": {"imo": "9074729", "mmsi": "210123456", "callsign": "5BXY2"}},
        {"type": "account", "name": "Operating account", "identifier": "DE89370400440532013000"},
        {"type": "property", "name": "Suburban house in Dayton, Ohio"},
        {"type": "company", "name": "TARVELO DENİZCİLİK TİCARET ANONİM ŞİRKETİ"},
    ],
    "case-b": [
        {"type": "individual", "name": "Paul Winter"},
        {"type": "company", "name": "Wren Example GmbH"},
    ],
}
PARTIES = {
    "case-a": [{"name": "Tobias Wren", "role": "Director/Individual"}, {"name": "Kestrel Ventures LLP", "role": "Counterparty / Company"},
               {"name": "Tarvelo", "role": "Company"}, {"name": "Quorvane Shipping", "role": "Company"},
               {"name": "Jonas Peter Maier", "role": "Shareholder"}, {"name": "the husband", "role": "Spouse"},
               {"name": "Nils Ostrander", "role": "Witness"}, "not a dict", {"name": "", "role": "x"}],
    "case-b": None,
}

TEXTS = [
    ("case-a", "Ada Brenner and Henry Zielinski signed. A. Brenner, Brenner Ada and BRENNER, ADA too; Zielinski (which one?)."),
    ("case-a", "Quorvane Shipping Ltd (\"QSL\") — QUORVANE SHIPPING LIMITED, Quorvane, quorvane shipping; HE 123456 and ΗΕ 123456."),
    ("case-a", "(2) TOBIAS WREN of the address set out in Schedule 1 (\"TW\") and KESTREL VENTURES LLP (hereinafter “KV”). TW agrees; KV too; tw is a word."),
    ("case-a", "Red River Trading went bust; Red River Trading Ltd and Red River Trading B.V. differ. Silver Bay and S.B. Shipping Ltd, sbshipping.example."),
    ("case-a", "SEA RUNNER IMO: 9074729 MMSI: 210123456 (210987654) Flag: Cyprus Call sign: 5BXY2 alt 5BXZ3\nDEBTS 1234567 CYPRUS"),
    ("case-a", "Иван Петрович Сидоров, Сидорова, Сидорову, И. Сидоров; Ivan Sidorov; Jörg Müller, Joerg Mueller, MUELLER, Müllers Wohnung."),
    ("case-a", "Contact lena.voss@example.org or jose.munoz@corvida.example, noreply@x.io; José Muñoz wrote. IBAN DE89 3704 0044 0532 0130 00 EUR, card 4111 1111 1111 1111, +44 20 7946 0958."),
    ("case-a", "Dear Ms Voss,\\n\\nBrightwater Maritime Ltd asks Lena Voss and Nils Ostrander; Nils and Ostrander; Jonas Peter Mayer."),
    ("case-a", "TARVELO DENIZCILIK and Tarvelo Denizcilik; Tarvelo alone; Harbour Point and Harbour Point Holdings Limited at Musterweg 12, 2. OG links, 20095 Hamburg."),
    ("case-a", "PERSON_1 and ORG_2 are tokens already; a value PERSON_1@example.org; Ada Brenner's notes, Ada Brenners notes."),
    ("case-a", "Henry Zielinski and Marek Zielinski — the Zielinskis; H. Zielinski; M. Zielinski; Henri Zielinski."),
    ("case-a", "physi-\ncally Ada Brenner, Sea ﬁnder, SEA RUNNER Container Ship and the Suburban house in Dayton, Ohio."),
    ("case-b", "Herrn Paul Winter, Wren Example GmbH, Wren Example, Paul Winters Wohnung, Winter, Paul. paul.winter@wren.example"),
    ("case-b", "Am Alten Hafen 3, 22303 Hamburg; 221B Baker Street; Flat 3, 12 Park Lane; SW1A 2AA; A-1010 Wien; 12000 Euro; 28. August 2026."),
    ("case-a", "Ada Brenner"),
    ("case-a", "   "),
]

ANSWERS = [
    "PERSON_{a} said; person_{a} too; _PERSON_{a}_ and PERSON_{a}x and PERSОN_{a} (Cyrillic O) and ORG_{b}; İD and ID_{c}.",
]


def dump(store, scope):
    return {"tokens": [[r.token, r.kind, r.real_value, r.normalized, r.status, r.canonical_token, r.attributes]
                       for r in store.token_rows(scope)],
            "aliases": [[a.token, a.surface, a.normalized, a.source] for a in store.alias_rows(scope)]}


def red_json(red):
    return {"text": red.text, "entity_types": red.entity_types, "redacted_count": red.redacted_count,
            "mapping": [[k, v] for k, v in red.mapping.items()], "surfaces": [[k, v] for k, v in red.surfaces.items()],
            "canonical": [[k, v] for k, v in red.canonical.items()]}


async def scenario(randoms):
    store = InMemoryTokenStore()
    seeds = InMemorySeedSource()
    for scope, ents in ENTITIES.items():
        for e in ents:
            seeds.add_entity(scope, SeedEntity(**e))
    for scope, parties in PARTIES.items():
        seeds.set_parties(scope, parties)
    det = patterns.RegexDetector()
    steps = []
    for scope in ENTITIES:
        store.add_salt(scope, SALT)
        await vault.seed_from_scope(store, scope, seeds)
        steps.append({"op": "seed", "scope": scope, "state": dump(store, scope)})
    reds = {"case-a": [], "case-b": []}
    for scope, text in TEXTS + randoms:
        red = await vault.tokenize(store, scope, text, detector=det, seed=False)
        reds[scope].append(red)
        mapping = vault.merge_mappings(reds[scope])
        toks = [t for t in mapping]
        answer = red.text + " | " + " ".join(toks) + " | " + " ".join(t.lower() for t in toks[:3])
        back = await vault.detokenize(store, scope, answer, mapping=mapping)
        steps.append({"op": "tokenize", "scope": scope, "text": text, "red": red_json(red),
                      "merged": [[k, v] for k, v in mapping.items()], "answer": answer, "back": back,
                      "residual": await vault.residual_surfaces(store, scope, text), "state": dump(store, scope)})
    full = await vault.detokenize(store, "case-a", "PERSON_10000001 and " + " ".join(r.token for r in store.token_rows("case-a")))
    steps.append({"op": "detokenize_all", "scope": "case-a", "out": full})
    return steps


def main(out):
    rnd = random.Random(7)
    names = [e["name"] for es in ENTITIES.values() for e in es] + ["Tobias Wren", "Kestrel Ventures LLP", "Lena Voss", "Nils"]
    glue = [" and ", ", ", " — ", " (\"", "\") ", "\n", " of ", ". ", " met ", "'s ", " vs. "]
    randoms = []
    for _ in range(150):
        parts = []
        for _ in range(rnd.randint(1, 6)):
            n = rnd.choice(names)
            r = rnd.random()
            if r < 0.2:
                n = n.upper()
            elif r < 0.3:
                n = n.lower()
            elif r < 0.4:
                ws = n.split()
                n = " ".join(reversed(ws))
            parts.append(n)
            parts.append(rnd.choice(glue))
        randoms.append((rnd.choice(["case-a", "case-b"]), "".join(parts)))
    steps = asyncio.run(scenario(randoms))
    det = patterns.RegexDetector()
    docs = [t for _, t in TEXTS] + [t for _, t in randoms[:40]] + [
        "x\\nlena.voss@example.org and \\u00a0anna@b.co, a@b.c", "GB82WEST12345698765432 GB82 WEST 1234 5698 7654 32X",
        "5500-0000-0000-0004 378282246310005 4111111111111111111 30569309025904 6011111111111117", "+1 (555) 123-4567 +12"]
    detect = [[d, [list(x) for x in det.detect(d, "en")]] for d in docs]
    checks = {"iban": [[v, patterns.iban_ok(v)] for v in ["DE89 3704 0044 0532 0130 00", "GB82WEST12345698765432", "GB82WEST12345698765433", "XX", "de89370400440532013000"]],
              "card": [[v, patterns.card_ok(v)] for v in ["4111 1111 1111 1111", "4111-1111-1111-1112", "5500000000000004", "1111111111111111", "378282246310005"]]}
    known = {}
    salt = SALT
    addr = []
    for d in docs:
        r = address.tokenize_addresses(d, derive=lambda s, taken: vault._derive_token(salt, "ADDRESS", s, taken), taken=set(known.values()), known=known)
        addr.append([d, r.text, [list(x) for x in r.found], [[k, v] for k, v in r.mapping.items()]])
    initials = [[n, address.initials_of(n)] for n in names + ["Dr. Ada Brenner", "MARTIN KESSLER", "A. B. C", "One"]]
    Path(out, "vault.json").write_text(json.dumps({"salt": list(SALT), "entities": ENTITIES, "parties": PARTIES, "steps": steps,
                                                   "detect": detect, "checks": checks, "addresses": addr, "initials": initials},
                                                  ensure_ascii=False), encoding="utf-8")
    print(len(steps), "steps")


if __name__ == "__main__":
    main(sys.argv[1])
