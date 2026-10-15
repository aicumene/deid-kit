# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Dev only: what Python's deid-agent helpers make of many inputs, for rust/tests/parity_agents.rs.

    RT=…/agent-runtime/python/bin/python3.13       # aiohttp is imported by host.py
    $RT rust/tools/vectors_agents.py OUT_DIR [seed]

Writes OUT_DIR/agents.json and a fixture tree OUT_DIR/agents-tree (paths in the vectors point into it,
so the Rust test reads the same files and folders).
"""
import ast
import asyncio
import json
import os
import random
import shutil
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DEID = HERE.parents[2]
sys.path.insert(0, str(DEID / "src"))

from deidkit.agents import cite, confine, host, pricing, review  # noqa: E402
from deidkit.patterns import RegexDetector  # noqa: E402
from deidkit.proxy.engine import ScopeEngine  # noqa: E402
from deidkit.seeds import InMemorySeedSource, SeedEntity  # noqa: E402
from deidkit.store import InMemoryTokenStore  # noqa: E402


class Slow(Exception):
    pass


def _alarm(*_):
    raise Slow()


signal.signal(signal.SIGALRM, _alarm)


def timed(f, *a, limit=0.5):
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        return f(*a)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def jsonable(v):
    return json.loads(json.dumps(v))


def strings_of(path):
    out = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and 3 <= len(node.value) <= 4000:
            out.append(node.value)
    return out


CITE_BITS = ["§", "§§", "Art.", "Artt.", "Artikel", "Abs.", "Absatz", "1", "2", "2a", "5", "133", "157", "280", "281", "823",
             ",", ", ", " und ", " sowie ", " bis ", "-", "–", "BGB", "SGB V", "SGB XII", "Satz", "S.", "Nr.", "ff.", "f.",
             "EGBGB", "Brüssel Ia-VO", "Rom-II-VO", " ", " ", " ", " ", "​", "­", "ГГУ", "ст.", " ст. ",
             "ГК РФ", "СК РФ", "Германского Гражданского Уложения", "8 U.S.C.", "§1427", "U.S.C. §", "8 CFR § 214.2(o)",
             "Cap. 113", "s. 4Α", "s.105", "section", " of ", "MCA 1973", "section 25 of the Matrimonial Causes Act 1973",
             "CPR 3.1", "ü", "Nr. 3100 VV RVG", "KV GKG", "1210", "lit. f", "Buchst. a", "(1)", "III", "ZPO", "DSGVO",
             "DS-GVO", "BGB-Vorschriften", "Paragraph", "Par.", "HwO", "BGBEG", "\n", "a.E.", "Hs.", "Alt.", "x", "dieses Vertrages"]

WORDS = ["the", "Tenant", "may", "break", "lease", "on", "six", "months'", "written", "notice", "—", "“quoted”", "‘single’",
         "\\*", "\\_", "soft­hyphen", "zero​width", "...", "…", "[...]", "\n", "  ", "\t", "Ä", "ß", "İ", "{#sec_1}",
         "clause", "4.2", "§ 573c", "a-b", "x–y", "x−y", "`tick`", "«guillemets»", "end."]


def doc(rng, n):
    out = []
    for _ in range(n):
        w = rng.choice(WORDS)
        out.append(w)
        out.append(rng.choice([" ", " ", " ", "\n", "", " {#c%d}\n" % rng.randrange(9)]))
    return "".join(out)


def curl(q):
    return q.replace("'", "’").replace('"', "“")


def quote_from(rng, text):
    if not text:
        return "x"
    a = rng.randrange(len(text))
    b = min(len(text), a + rng.randrange(1, 120))
    q = text[a:b]
    k = rng.randrange(6)
    if k == 1:
        q = curl(q)
    elif k == 2:
        q = " ".join(q.split())
    elif k == 3 and len(q) > 20:
        m = len(q) // 2
        q = q[:m - 3] + rng.choice([" ... ", "…", " [...] ", " […] "]) + q[m + 3:]
    elif k == 4:
        q = q.replace("\\", "")
    elif k == 5:
        q = q + rng.choice(["", " never", "zz"])
    return q


def notes_answer(rng):
    notes = []
    for i in range(rng.randrange(0, 5)):
        n = {"quote": rng.choice(["six months' notice", "  padded  ", "", 5, None, "x" * 700]),
             "kind": rng.choice(["Risk", "question", "POINT", "clause", "other", 3, None]),
             "aspect": rng.choice(["Termination", "payment", "data protection", "nonsense", None]),
             "importance": rng.choice([1, 5, 9, -2, "4", " 3 ", "4.0", 2.7, True, False, None, [1], "４"]),
             "clause_type": rng.choice(["Right", "obligation", "", None, "weird"]),
             "who": rng.choice(["the  Tenant", "", 7, None]), "what": "break\nthe lease", "to_whom": "", "when": None,
             "norm": rng.choice(["§ 573c BGB", "", "x" * 150]),
             "suggestion": rng.choice(["  New words\n  here  ", "", None, 3, "y" * 4100]),
             "note": rng.choice(["A note.", "  spaced\tout  ", "z" * 1300, None])}
        if rng.random() < 0.2:
            n = rng.choice(["not a dict", 5, None, []])
        notes.append(n)
    body = json.dumps({"notes": notes}, ensure_ascii=rng.random() < 0.5)
    k = rng.randrange(9)
    if k == 0:
        return body
    if k == 1:
        return "I read it. " + body + " Done."
    if k == 2:
        return "```json\n" + body + "\n```"
    if k == 3:
        return body[:rng.randrange(len(body) + 1)]
    if k == 4:
        return "first " + body + " then " + json.dumps({"notes": notes[:1]})
    if k == 5:
        return body.replace("\\n", "\n")
    if k == 6:
        return "no notes at all"
    if k == 7:
        return json.dumps({"other": 1})
    return '{"notes": "not a list"} ' + body


AGENTS = ["Claude Code", "Codex", "MatterAgent"]


def events(rng):
    out = []
    n = rng.randrange(0, 40)
    for _ in range(n):
        k = rng.choice(["user", "handover", "message", "message", "tool", "tool_update", "plan", "permission",
                        "permission_done", "notice", "error", "turn_end", "files", "thought", "brief", "status"])
        a = rng.choice(AGENTS)
        if k == "user":
            e = {"kind": k, "text": rng.choice(["Draft a reply", "  spaced task  ", "T" * 3100]), "agent": rng.choice([a, "", None])}
        elif k == "handover":
            e = {"kind": k, "from": rng.choice([a, None]), "to": rng.choice([a, "", None])}
        elif k == "message":
            e = {"kind": k, "text": rng.choice(["Working on it. ", "", "  ", "M" * 1600, "done."])}
        elif k == "tool":
            e = {"kind": k, "id": rng.choice(["t1", "t2", None, 3, ""]), "title": rng.choice(["Write draft.md", "", None]),
                 "tool": rng.choice(["edit", None]), "status": rng.choice(["pending", "completed", None, "failed", ""])}
        elif k == "tool_update":
            e = {"kind": k, "id": rng.choice(["t1", "t2", 3, None]), "status": rng.choice(["completed", "failed", None]),
                 "title": rng.choice(["Renamed", None])}
        elif k == "plan":
            e = {"kind": k, "entries": [{"content": rng.choice(["Read", "Write", 5]), "status": rng.choice(["completed", "pending", ""])}
                                        for _ in range(rng.randrange(3))]}
        elif k == "permission":
            e = {"kind": k, "id": rng.choice(["p1", "p2"]), "title": rng.choice(["Write draft.md", "", None])}
        elif k == "permission_done":
            e = {"kind": k, "id": rng.choice(["p1", "p2", "p3"]), "outcome": rng.choice(["allowed", "refused", "cancelled", "odd"])}
        elif k == "notice":
            e = {"kind": k, "text": "Refused without asking.", **({"handing": True} if rng.random() < 0.3 else {})}
        elif k == "error":
            e = {"kind": k, "text": "The turn failed: boom"}
        elif k == "turn_end":
            e = {"kind": k, "stopReason": rng.choice(["end_turn", "cancelled", "refusal", "max_tokens", "max_turn_requests", "odd", None, 5])}
        elif k == "files":
            e = {"kind": k, "changed": rng.choice([[], ["a.md"]]), "added": rng.choice([[], ["draft.md", "b.md"]]),
                 "removed": rng.choice([[], ["old.md"]]), "files": []}
        else:
            e = {"kind": k, "text": "x"}
        out.append({"at": 1.5, **e})
    if rng.random() < 0.1:                       # many steps
        out.append({"at": 1, "kind": "user", "text": "Many", "agent": "Codex"})
        out += [{"at": 1, "kind": "tool", "id": f"s{i}", "title": f"Step {i}", "status": "completed"} for i in range(60)]
    if rng.random() < 0.1:                       # past the brief's length
        for i in range(30):
            out.append({"at": 1, "kind": "user", "text": f"Task {i} " + "w" * 900, "agent": rng.choice(AGENTS)})
            out.append({"at": 1, "kind": "message", "text": "said " + "v" * 1400})
            out.append({"at": 1, "kind": "turn_end", "stopReason": "end_turn"})
    return out


def setup_choice(rng):
    k = rng.randrange(6)
    opt = lambda: {"value": rng.choice(["default", "fast", "", None]), "name": rng.choice(["Default", "", None]),
                   "description": rng.choice(["d", None])}
    if k == 0:
        return {"configOptions": [{"id": "model", "category": "model", "type": "select", "currentValue": "default",
                                   "options": [opt() for _ in range(rng.randrange(4))]}]}
    if k == 1:
        return {"configOptions": [{"id": "x", "category": "model", "currentValue": "a",
                                   "options": [{"group": "g", "options": [opt(), opt()]}, opt()]}]}
    if k == 2:
        return {"configOptions": [{"id": "model", "type": "boolean"}, {"id": "mode"}]}
    if k == 3:
        return {"models": {"availableModels": [{"modelId": rng.choice(["m1", "", None]), "name": rng.choice(["M", None])}
                                               for _ in range(3)], "currentModelId": "m1"}}
    if k == 4:
        return {"models": {"availableModels": []}}
    return {}


def results(rng):
    u = {k: rng.choice([0, 5, -1, True, 2.5, "7", None]) for k in host.USAGE_FIELDS + ("costCents",)}
    return rng.choice([{"stopReason": "end_turn", "usage": u}, {"_meta": {"usage": u}}, {"usage": "x", "_meta": {"usage": u}},
                       None, [], {"usage": None}])


def usages(rng):
    def one():
        api = rng.random() < 0.3
        u = {"input_tokens": rng.choice([0, 100, 1234567, -3, True, 2.0]), "output_tokens": rng.choice([0, 10, 999])}
        if api:
            u["api"] = "responses"
            u["input_tokens_details"] = rng.choice([{"cached_tokens": rng.choice([0, 50, 10 ** 9])}, None, "x"])
        else:
            u["cache_read_input_tokens"] = rng.choice([0, 50000, None])
            u["cache_creation_input_tokens"] = rng.choice([0, 3000, 7])
            if rng.random() < 0.5:
                u["cache_creation"] = {"ephemeral_5m_input_tokens": rng.choice([0, 1000, 5]),
                                       "ephemeral_1h_input_tokens": rng.choice([0, 2000, 9])}
        model = rng.choice(["claude-opus-5-5", "claude-sonnet-5-5-20260901", "claude-fable-5-1[1m]", "claude-haiku-4-5", "gpt-5",
                            "", "claude-opus-5", "claude-opus-5-1"])
        return model, u
    return [one() for _ in range(rng.randrange(0, 5))]


def shlex_case(rng):
    alphabet = ["a", "b", " ", " ", "\t", "\n", '"', "'", "\\", "$HOME", "~/x", "/etc", ";", "|", "(", ")", "&", "é", "#", "2>", ">"]
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14)))


def tree(out: Path):
    root = out / "agents-tree"
    if root.exists():
        shutil.rmtree(root)
    for m in ("A-1", "B-2"):
        (root / "matters" / m / "documents").mkdir(parents=True)
        (root / "matters" / m / "documents" / "01-letter.md").write_text("A letter.\n")
    (root / "matters" / "A-1" / "drafts").mkdir()
    (root / "matters" / "A-1" / "link-out").symlink_to(root / "matters" / "B-2")
    (root / "matters" / "A-1" / "dangling").symlink_to(root / "nowhere")
    (root / "codex" / "bin").mkdir(parents=True)
    (root / "codex" / "bin" / "codex").write_text("#!/bin/sh\n")
    (root / "codex-home").mkdir()
    (root / "auth.json").write_text("{}")
    return root


def view_folder(rng, out: Path, n: int, text: str):
    folder = out / f"agents-view-{n}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    path = rng.choice(["doc.md", "sub/lease.md", "deal.docx"])
    (folder / path).parent.mkdir(parents=True, exist_ok=True)
    (folder / path).write_text(text)
    anns = []
    for i in range(rng.randrange(1, 9)):
        a = rng.randrange(len(text) + 1)
        b = min(len(text), a + rng.randrange(0, 40))
        kind = rng.choice(["risk", "question", "point", "clause", "edit", "decision", "revision", "written", "vote", ""])
        sels = [{"type": "TextQuoteSelector", "exact": rng.choice([text[a:b], text[a:b] + "x", "", curl(text[a:b])]),
                 "prefix": text[max(0, a - 32):a], "suffix": text[b:b + 32]},
                {"type": "TextPositionSelector", "start": rng.choice([a, a + 3, -2, True, "5"]), "end": rng.choice([b, b + 3])}]
        if rng.random() < 0.3:
            sels.append({"type": "FragmentSelector", "value": "sec_%d" % i})
        ann = {"@context": "http://www.w3.org/ns/anno.jsonld", "id": "%032x" % rng.getrandbits(128), "type": "Annotation",
               "creator": rng.choice([{"type": "Person", "name": "ann"}, {"type": "Software", "name": "Claude Code", "model": "m"}, "str"]),
               "created": "2026-10-%02dT10:00:%02d.000000Z" % (rng.randrange(1, 28), rng.randrange(60)),
               "target": {"source": path, "selector": sels}, "body": [], "bernio": {"kind": kind}}
        if rng.random() < 0.5:
            ann["generator"] = {"type": "Software", "run": "r%d" % i}
        for purpose in ("commenting", "replacing", "classifying"):
            if rng.random() < 0.5:
                ann["body"].append({"type": "TextualBody", "value": rng.choice(["a  note\n", "words", 5]), "purpose": purpose})
        b2 = ann["bernio"]
        if kind in ("risk", "question", "point", "clause"):
            b2.update({"aspect": rng.choice(["termination", "nope", None]), "importance": rng.choice([1, 4, 5, 9, True, 2.0, "3"]),
                       "clause_type": rng.choice(["right", ""]), "norm": rng.choice(["§ 1 BGB", ""])})
        if kind == "decision":
            target = rng.choice(anns).get("id", "deadbeef") if anns else "deadbeef"
            ann["target"] = {"source": path, "annotation": target}
            b2.update({"verdict": rng.choice(list(review.VERDICTS) + ["meh"]), **({"applied": True} if rng.random() < 0.5 else {})})
        if kind == "revision":
            b2["hunks"] = rng.choice([[{"at": 0, "old": "a", "new": "b"}], [], None])
        if kind == "written":
            b2["how"] = rng.choice(["added", "wrote", "saved", "odd"])
        if rng.random() < 0.1:
            del ann["id"]
        anns.append(ann)
        name = ann.get("id", "%032x" % rng.getrandbits(128)) + ".json"
        target = folder / ".legix" / "markup" / path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ann, ensure_ascii=False, indent=2))
    md = folder / ".legix" / "markup" / path
    (md / "broken.json").write_text("{not json")
    (md / "other.json").write_text(json.dumps({"id": "x", "target": {"source": "another.md"}}))
    fb = folder / ".legix" / "chat-feedback"
    fb.mkdir(parents=True)
    for i in range(rng.randrange(3)):
        v = review.vote(key=f"k{i}", question="Q?", answer="A " * rng.randrange(1, 4), agent="Codex", model="gpt",
                        verdict=rng.choice(["up", "down"]), creator=review.person("ann"), comment="c", correction="")
        v["created"] = "2026-10-01T00:00:0%d.000000Z" % i
        review.store_vote(folder, v)
    return {"folder": str(folder), "path": path, "text": text,
            "view": jsonable(review.view(folder, path, text)), "records": jsonable(review.records(folder))}


PARTIES = [("company", "Harrowgate Freight Ltd", "HFL"), ("individual", "Tobias Wren", "TW"),
           ("company", "Meridian Holdings Ltd", "Meridian"), ("individual", "Lena Voss", "LV")]
CLAUSES = ["{a} shall deliver the goods to {b} by 1 May.", "{a} may appoint three directors and {b} may appoint two.",
           "Notices to {a} go to its registered office.", "A quorum is three directors, one appointed by {b}.",
           "The shares are held as to 6,500 by {a} and 3,500 by {b}.", "Nothing here names anyone at all.",
           "{a} and {b} agree that clause 4 applies.", "Payment is due within 30 days of the invoice."]


def spelling(rng, party, defined):
    kind, name, short = party
    return rng.choice([name, name.upper(), short if defined else name, name.title()])


def quote_case(rng):
    parties = rng.sample(PARTIES, rng.randrange(1, 4))
    seeds = [[k, n] for k, n, _ in parties if rng.random() < 0.85]
    lines = ["This agreement is between " + " and ".join(f'{n.upper()} ("{s}")' for _, n, s in parties) + "."]
    for _ in range(rng.randrange(2, 9)):
        a, b = rng.choice(parties), rng.choice(parties)
        lines.append(rng.choice(CLAUSES).format(a=spelling(rng, a, True), b=spelling(rng, b, True)))
    text = rng.choice(["\n\n", "\n", " "]).join(lines) + "\n"
    quotes = []
    for _ in range(rng.randrange(1, 7)):
        k = rng.randrange(6)
        if k == 0:                      # a clause of it, the parties spelled anew
            a, b = rng.choice(parties), rng.choice(parties)
            quotes.append(rng.choice(CLAUSES).format(a=spelling(rng, a, False), b=spelling(rng, b, False)))
        elif k == 1:                    # a line of it with every short name spelled out
            line = rng.choice(lines[1:])
            for _, n, sh in parties:
                line = line.replace(sh, rng.choice([n, n.upper()]))
            quotes.append(line)
        elif k == 2:                    # with an ellipsis
            line = rng.choice(lines[1:])
            cut = len(line) // 2
            q = line[:max(8, cut - 4)] + " ... " + line[cut + 4:]
            for _, n, sh in parties:
                q = q.replace(sh, n)
            quotes.append(q)
        elif k == 3:
            quotes.append(rng.choice(lines))          # as written
        elif k == 4:
            quotes.append(rng.choice(["", "Words it never had.", "HARROWGATE FREIGHT LTD shall collect the goods."]))
        else:
            line = rng.choice(lines)
            quotes.append(line[rng.randrange(len(line) // 2):])
    return seeds, text, quotes


def main():
    out = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    root = tree(out)
    v: dict = {"home": os.path.expanduser("~")}

    # citations
    texts = [s for f in ("test_agent_cite_de.py", "test_agent_cite_us.py", "test_agent_cite_cy.py", "test_agent_review.py")
             for s in strings_of(DEID / "tests" / f)]
    for _ in range(3000):
        texts.append("".join(rng.choice(CITE_BITS) for _ in range(rng.randrange(1, 16))))
    cases = []
    for t in texts:
        try:
            spans = timed(cite.extract_citation_spans, t)
        except Slow:
            continue
        cases.append({"text": t, "spans": [[s.start, s.end, s.citation] for s in spans],
                      "norms": review.norm_mentions(t)})
    v["cite"] = cases

    # quotes, clauses, notes
    finds, clauses = [], []
    for _ in range(1500):
        t = doc(rng, rng.randrange(0, 60))
        f = review.Folded.of(t)
        for _ in range(3):
            q = quote_from(rng, t)
            near = rng.choice([None, rng.randrange(len(t) + 1)])
            finds.append({"text": t, "quote": q, "near": near, "found": f.find(q, near=near)})
        at = rng.randrange(len(t) + 1)
        clauses.append({"text": t, "at": at, "clause": review.clause_at(t, at)})
    v["find"], v["clause"] = finds, clauses
    v["notes"] = []
    for _ in range(1500):
        a = notes_answer(rng)
        notes, cut = review.notes_of(a)
        v["notes"].append({"answer": a, "notes": notes, "cut": cut})

    # where tokens stand
    seeds = InMemorySeedSource()
    for name, kind in (("Ada Brenner", "individual"), ("Harrowgate Freight Ltd", "company"), ("Tobias Wren", "individual")):
        seeds.add_entity("m", SeedEntity(kind, name))
    engine = ScopeEngine(InMemoryTokenStore(), "m", seeds=seeds, detector=RegexDetector())
    pieces = ["Ada Brenner", "Harrowgate Freight Ltd", "HFL", "Tobias Wren", "T. Wren", "Mr Wren", "ada@example.org",
              "+44 117 496 0321", "14 Harbour Lane", "Bristol BS1 4QA", "GB82 WEST 1234 5698 7654 32", "FORM_12", "said",
              "the", "lease", "\n", "Brenner's", "ADA BRENNER", "Ada  Brenner"]
    spans = []
    for _ in range(400):
        t = " ".join(rng.choice(pieces) for _ in range(rng.randrange(1, 18)))
        red = asyncio.run(engine.tokenize_all([t]))[0]
        tok = red.text
        if rng.random() < 0.2 and tok:
            i = rng.randrange(len(tok))
            tok = tok[:i] + rng.choice([" ", "", "x"]) + tok[i + 1:]
        surfaces, mapping = jsonable(red.surfaces or {}), jsonable(red.mapping or {})
        spans.append({"original": t, "tokenised": tok, "surfaces": surfaces, "mapping": mapping,
                      "spans": review.deid_spans(t, tok, surfaces, mapping)})
    v["deid"] = spans

    # hunks
    hk = []
    for _ in range(600):
        a = doc(rng, rng.randrange(0, 40))
        words = a.split(" ")
        for _ in range(rng.randrange(0, 5)):
            if words:
                i = rng.randrange(len(words))
                words[i] = rng.choice(["", "changed", words[i] + "s", "two words"])
        b = " ".join(words)
        hk.append({"before": a, "after": b, "hunks": review.hunks(a, b, rng.choice([200, 2]))})
    v["hunks"] = hk

    # place and apply
    pl = []
    for _ in range(1500):
        t = doc(rng, rng.randrange(1, 40))
        a = rng.randrange(len(t) + 1)
        b = min(len(t), a + rng.randrange(0, 30))
        t2 = rng.choice([t, "Inserted words. " + t, t.replace(" ", "  ", 2), t[: len(t) // 2]])
        sels = [{"type": "TextQuoteSelector", "exact": rng.choice([t[a:b], "", curl(t[a:b])]),
                 "prefix": rng.choice([t[max(0, a - 32):a], "", t[max(0, a - 5):a]]), "suffix": rng.choice([t[b:b + 32], "", t[b:b + 4]])},
                {"type": "TextPositionSelector", "start": rng.choice([a, a + 1, -3, True, None, "2"]), "end": rng.choice([b, b - 1, None])}]
        ann = {"target": {"selector": sels}}
        repl = rng.choice(["", "NEW", "line\nbreak"])
        pl.append({"text": t2, "ann": ann, "place": review.place(ann, review.Folded.of(t2)),
                   "apply": review.apply(t2, ann, repl), "replacement": repl})
    v["place"] = pl

    # the brief
    br = []
    for _ in range(800):
        ev = events(rng)
        me = rng.choice(AGENTS)
        carrying = rng.random() < 0.3
        br.append({"events": ev, "me": me, "carrying_on": carrying, "brief": host.brief(ev, me, carrying_on=carrying)})
    v["brief"] = br

    v["model_choice"] = [{"setup": s, "choice": host.model_choice(s)} for s in (setup_choice(rng) for _ in range(300))]
    tu = []
    for _ in range(300):
        r = results(rng)
        tu.append({"result": r, "usage": host.turn_usage(r)})
    v["turn_usage"] = tu

    # pricing
    pr = []
    for _ in range(500):
        us = usages(rng)
        reqs = [pricing.request_usage(u, m) for m, u in us]
        pr.append({"usages": us, "requests": [[r.model, r.input, r.write_5m, r.write_1h, r.read, r.output] for r in reqs],
                   "cost": pricing.task_cost(reqs)})
    v["pricing"] = pr

    # shell words and paths
    sh = []
    for _ in range(3000):
        s = shlex_case(rng)
        try:
            import shlex
            sh.append({"text": s, "words": shlex.split(s)})
        except ValueError:
            sh.append({"text": s, "words": None})
    v["shlex"] = sh
    folder = root / "matters" / "A-1"
    cmds = ['grep -rni "clause 4" {root}/ 2>/dev/null | head -20', "cat ../B-2/documents/01-letter.md", "ls ~/", "cd $HOME && ls",
            "cp drafts/a.md {tmp}/a.md", 'grep -rni "clause 4" {folder}/documents/ 2>/dev/null; echo "--- exit: $?"',
            'find . -type f -not -path "*/.git/*" | head -100', 'grep -n "/api/v1" documents/*.md',
            "/usr/bin/wc -l documents/*.md > /dev/null", "sed -n 1,40p documents/01-letter.md", "cat link-out/documents/01-letter.md",
            "cat dangling", "--file={root}/matters/B-2", "x=../B-2", "echo hi >{root}/out", "2>>{root}/err", "ls ${{HOME}}/Library",
            "cat 'unbalanced", "ls documents/../../B-2", "ls /dev/null /dev/tty", "cat /etc/hosts", "ls /", "(cd .. && ls)",
            "ls ~nobody/x", "cat ./documents/../documents/01-letter.md", "ls /opt/homebrew/bin", "ls /private/tmp"]
    v["outside"] = [{"command": c.format(root=root / "matters", tmp=root, folder=folder), "folder": str(folder),
                     "found": host.outside_paths(c.format(root=root / "matters", tmp=root, folder=folder), folder)} for c in cmds]
    ins = ["documents/01-letter.md", str(folder / "x"), str(folder), "/etc/hosts", "../B-2", "link-out/x", "dangling", "",
           str(root / "matters" / "A-1x"), "a\0b"]
    v["inside"] = [{"folder": str(folder), "path": p, "inside": host.inside(folder, p)} for p in ins]

    # settings, meta, profile, prompt
    (folder / "settings.json").write_text(json.dumps({"permissions": {"allow": ["Read"]}, "hooks": {"PreToolUse": [{"matcher": "x"}]}}))
    metas = [None, {}, {"claudeCode": {"options": {"disallowedTools": ["WebFetch", "Task"], "allowedTools": ["Read"]}}},
             {"claudeCode": {"options": {"settings": "settings.json"}}}, {"claudeCode": {"options": {"settings": {"env": {"A": 1}}}}},
             {"claudeCode": {"options": {"settings": 5}}}, {"claudeCode": {"options": {"settings": "missing.json"}}}, {"other": 1}]
    st = []
    for m in metas:
        for confine_on in (True, False):
            for guard in ([], ["/opt/agent guard"]):
                s = host.claude_settings(folder, confine=confine_on, guard=guard, guard_log=(root / "g.jsonl") if guard else None)
                try:
                    merged = host.with_claude_settings(m, s, folder)
                except (OSError, ValueError) as exc:
                    merged = {"error": type(exc).__name__}
                st.append({"meta": m, "confine": confine_on, "guard": guard, "settings": s, "merged": merged,
                           "subagents": host.with_subagents(m)})
    v["settings"] = {"folder": str(folder), "log": str(root / "g.jsonl"), "cases": st, "kinds": sorted(host.SUBAGENT_MCP_TOOLS)}
    prof = []
    for codex in (root / "codex" / "bin" / "codex", Path("/usr/bin/true")):
        for extra in ([], [str(root / "codex"), "/", str(root / "nowhere"), str(root / "auth.json")]):
            for auth in (None, root / "auth.json"):
                prof.append({"folder": str(folder), "codex": str(codex), "home": str(root / "codex-home"), "extra": extra,
                             "auth": str(auth) if auth else None,
                             "profile": confine.profile(folder, codex, root / "codex-home", extra, auth)})
    v["profile"] = prof
    v["prompt"] = [{"path": p, "source": s, "prompt": review.prompt(p, s)} for p, s in
                   (("lease.md", "lease.md"), ("deal.docx", ".legix/text/deal.docx.md"), ("{source}.md", "{path}"))]
    v["text_source"] = [{"path": p, "source": review.text_source(p), "writable": review.writable(p)} for p in
                        ("a.md", "A.MD", "b.docx", ".legix/x.md", "c.txt", "d.", ".md", "dir/e.markdown", "f.text", "g")]

    # quotes found as written, or in tokens
    fq = []
    for _ in range(400):
        seeds_l, text, quotes = quote_case(rng)
        src = InMemorySeedSource()
        for kind, name in seeds_l:
            src.add_entity("q", SeedEntity(kind, name))
        engine = ScopeEngine(InMemoryTokenStore(), "q", seeds=src, detector=RegexDetector())
        found = asyncio.run(review.find_quotes(text, quotes, engine))
        fq.append({"seeds": seeds_l, "text": text, "quotes": quotes, "found": [list(f) if f else None for f in found]})
    v["find_quotes"] = fq
    print("find_quotes: %d quotes, %d found in tokens" % (
        sum(len(c["quotes"]) for c in fq),
        sum(1 for c in fq for q, f in zip(c["quotes"], c["found"]) if f and review.Folded.of(c["text"]).find(q) is None)))

    # the markup of a document
    v["view"] = [view_folder(rng, out, i, doc(rng, rng.randrange(5, 50))) for i in range(60)]

    (out / "agents.json").write_text(json.dumps(v, ensure_ascii=False))
    print(f"{len(v['cite'])} cite, {len(finds)} finds, {len(v['notes'])} notes, {len(spans)} deid, {len(br)} briefs, "
          f"{len(v['view'])} views")


if __name__ == "__main__":
    main()
