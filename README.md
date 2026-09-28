# deid-kit

Reversible, per-scope de-identification of text on its way to a cloud model: a salted token
vault with name variants and an out-of-band glossary, street-address and initials tokens, a
privacy gateway that checks eligibility and audits every crossing, and a probe that measures
whether a de-identified excerpt can still be traced back to its source. Python package `deidkit`.

The core rules:

- **Who is hidden, where and when are not.** People, organisations, vessels, accounts and
  identifiers become tokens; places and dates stay readable, because they are what cross-border
  questions and deadline arithmetic run on. Street addresses are the exception — a home is
  personal data — and become `ADDRESS_…` tokens while the city and the country stay.
- **A token is a function of its referent.** With a per-scope salt, a new referent's token is
  `HMAC-SHA256(salt, kind ‖ key(value))` shaped `KIND_DIGITS` (`PERSON_48170392`): the same person
  gets the same token however often the vault is rebuilt, a different scope gives him an
  unrelated one, and without the salt the token confirms no guess. A scope without a salt keeps
  counter tokens (`PERSON_1`); issued tokens never move.
- **Nothing is deleted.** A defective token is retired — dropped from the match index, kept for
  the return leg — so every answer ever sent with it can still be re-hydrated.
- **The glossary describes, never identifies, and fails closed.** One line per token that
  actually crossed ("individual (natural person), male, jurisdiction BE, shareholder in ORG_2
  (≥50%)"); a line that would contain an enrolled real value, or name a token the request did not
  send, is dropped.

## What the vault does with a text

1. **Seeds** the scope's known entities and parties (roles decide who is split into name parts).
2. Runs the **residual detector** on the original text; every span passes a structural
   admission gate (no single words, no ordinary vocabulary, no layout artefacts, no mixed-script
   words, no span in another script than the detector ran in, no span inside a known company or
   vessel, no Presidio location).
3. **Recalls** identifiers written beside an already tokenised identifier of the same hull, and
   domains whose label is an enrolled company.
4. **Replaces** every enrolled surface in one pass over a fold that keeps offsets (whitespace,
   case, NFKC, punctuation shapes, diacritics, Greek/Cyrillic homoglyphs, Turkish dotted I),
   longest first, with a word boundary, merging adjacent spans of one referent.

Surfaces of one referent are aliases of one token: romanisations ("Henry" / "Henri"), name
order and initials, Cyrillic spellings and their Russian case endings, legal-form synonyms
("Ltd" / "Limited"), initialisms ("S.B. Shipping"), compact forms ("SEAFINDER"), plurals,
German umlaut spellings and the genitive. A surface shared by two referents (the surname of two
brothers) gets a token of its own, glossed as shared, instead of a guess.

## Layout

```
src/deidkit/
  vault.py           tokenize / detokenize, seed_from_scope, reconcile_scope, merge_mappings,
                     with_glossary, ensure_salt, residual_surfaces, graph_index / graph_tokenize /
                     graph_residual_surfaces, token_attributes, active_people
  namefold.py        the offset-keeping fold, the boundary predicate, identity skeletons, surface
                     variants, the admission predicate for residual detections
  glossary.py        build_glossary, render, legal_form_of
  address.py         find_addresses, tokenize_addresses, initials_of, tokenize_initials
  gateway.py         PrivacyGateway, Redaction, AuditSink / CrossingRecord / InMemoryAuditSink
  reid.py            probe_retrieval (PassageIndex, InMemoryPassageIndex), probe_judge
  store.py           TokenStore, TokenRow / AliasRow, InMemoryTokenStore
  seeds.py           SeedSource, SeedEntity, InMemorySeedSource
  detect.py          PiiDetector, PresidioDetector, the label → kind map
  presidio.py        the optional Presidio engine (extra `presidio`)
  model.py           ModelRouter, ModelRequest / ModelMessage, PrivacyViolation, QUERY / DOCUMENT
  classification.py  Sensitivity, max_crossable, may_cross_to_cloud
  tracing.py         optional span hooks
  textmatch.py       normalize — the comparison form every fold builds on
  lang.py            detect_language (Cyrillic share)
  jsonutil.py        extract_json — tolerant JSON from a model answer
  sqlite_store.py    SQLiteTokenStore — the vault in one SQLite file, for one process
  patterns.py        RegexDetector — e-mail, IBAN, card and phone patterns, no name recognition
  seedfile.py        load_seed_files — known people and organisations from a TOML file
  proxy/             deid-proxy — the local proxy between Claude Code and Anthropic's API
tests/               pytest suite, no network, no database, invented names only
```

## Install and test

```
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

No runtime dependencies (standard library only). Optional extra `proxy` (`aiohttp`) for
`deid-proxy`; optional extra `presidio` for
`PresidioDetector` and the gateway's default redactor (both import it lazily; a spaCy model such
as `en_core_web_sm` is installed separately). Test extra: `pytest`, `pytest-asyncio`, `aiohttp`.
Python 3.11 or newer. llm-kit is not required; when it is installed, `Sensitivity`,
`PrivacyViolation` and the request types are llm-kit's own objects (see `model.py`).

## Plugging in a deployment

The package has no database, detector, audit log or model of its own. A deployment supplies
them through small interfaces:

| Interface | What the deployment provides | In-package default |
|---|---|---|
| `store.TokenStore` | `load_tokens`, `load_aliases`, `load_salt` (async; fresh lists of live rows), `add_token`, `add_alias`, `add_salt` (return the row), `commit`. The vault edits loaded rows in place (`status`, `canonical_token`, `attributes`, an alias's `token` / `surface` / `source` / `normalized`); `commit` must persist those edits too. An ORM model with the same attribute names is a row as it stands. | `InMemoryTokenStore` |
| `seeds.SeedSource` | `entities(scope_id)` (objects with the `SeedEntity` attributes) and `parties(scope_id)` (`[{"name", "role"}]` or `None`) — what is known about a scope before its text is read. | `InMemorySeedSource` |
| `detect.PiiDetector` | `detect(text, language)` → `[(span, kind)]`, fail-open. | `PresidioDetector` over any engine with Presidio's `analyze`; without a detector the residual pass is skipped (logged) |
| `tokenize(..., language=)` | the text's language when it is known (`"en"`, `"ru"`; any other code gets only the structural recognisers). | `lang.detect_language` |
| `gateway.AuditSink` | `record(context, CrossingRecord)` — an append-only table, a log stream. `context` is whatever the caller passed to `cross_to_cloud` (e.g. a database session). | `InMemoryAuditSink` |
| `PrivacyGateway(redact_fn=...)` | the redactor for payloads that did not go through the vault. | Presidio Analyzer + Anonymizer; refuses the crossing when Presidio is missing |
| `model.ModelRouter` | `embed(texts, *, sensitivity, kind)` and `generate(request)` (a reply with `.text`) — for the probe. llm-kit's `LLMRouter` is one. | none |
| `reid.PassageIndex` | `nearest(vector, *, k, exclude_chunk_id)` and `nearest_in_scope(...)` over the embedded chunks of every scope (cosine distance). | `InMemoryPassageIndex` |
| `tracing.set_span_factory(f)`, `tracing.set_attribute_setter(g)` | span hooks, e.g. an OpenTelemetry facade. Process-wide. | no-op |

A minimal round trip:

```python
from deidkit import vault
from deidkit.seeds import InMemorySeedSource, SeedEntity
from deidkit.store import InMemoryTokenStore

store, seeds = InMemoryTokenStore(), InMemorySeedSource()
seeds.add_entity("project-7", SeedEntity("individual", "Ada Brenner", role="Director"))
await vault.ensure_salt(store, "project-7")              # once per scope, on a writing path

red = await vault.tokenize(store, "project-7", "Brenner signed on 3 May.", seeds=seeds)
payload = vault.with_glossary(red).text                  # what may cross: glossary + text
answer = await vault.detokenize(store, "project-7", model_answer, mapping=red.mapping)
```

`mapping` is the set of tokens this request sent; de-tokenisation reverses only those, so a token
the model invents stays a token. Several texts sent in one request are combined with
`vault.merge_mappings`, so one person is named one way.

Runtime text — the glossary lines and header, the probe's judge prompt, log messages — is kept
word for word from the code's first deployment, including its wording ("in this matter",
"case file").

## Coding agents

deid-kit can sit between a coding agent (Claude Code, Codex) and its model, so that the agent
works on the real files while the model receives tokens. For Claude Code this works today:
`deid-proxy` (extra `proxy`) runs on your machine in front of Anthropic's API. Point Claude Code at
it with `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`, and list the names you know in a TOML file.
[docs/coding-agents.md](docs/coding-agents.md) has the quick start and a measured run. It also
specifies what is not built yet: the proxy for Codex, Claude Code hooks, a tokenized working
copy, and a pre-commit scan.

## Tests

`pytest` runs 267 tests, with no network and no database. Most are the original deployment's
tests of this code re-expressed with invented names against the in-memory store, seed source and
a scripted detector. The rest test the interfaces themselves, the SQLite store, the pattern
detector and the proxy; the proxy is tested end to end against a scripted upstream on the
loopback interface. Names are Latin
script; Cyrillic behaviour is exercised through code points and through spellings the module
generates from a Latin name.

## How deid-kit compares

Text de-identification tools differ in what they are for. deid-kit is built for one job: sending a
document's text to a model outside the organisation and reading the answer back at home.

| | centres on | how an identifier is replaced | back to the original | across one set of documents |
|---|---|---|---|---|
| **deid-kit** | text on its way to a cloud model, and the answer coming back | a salted, deterministic token per referent (`PERSON_3`, `ORG_1`, `ADDRESS_…`); spellings, transliterations and inflected forms of a name fold into the same token; a glossary of non-identifying facts (legal form, role, gender from morphology) travels with the tokens | tokens are reversed at home from the vault, which never leaves | one token per referent within a scope, stable over time |
| Microsoft Presidio [1] | detecting and anonymizing PII in text | detection by NER and pattern recognizers; operators replace, redact, mask, hash or encrypt | through the encrypt/decrypt operator | up to the caller |
| scrubadub [2] | removing personal information from free text | placeholders by type, such as `{{NAME}}` | the placeholder is the result | per call |
| Philter [3] | removing protected health information from clinical notes | pattern and NLP filters tuned for recall on clinical text | removal is the result | per note |
| ARX [4] | anonymizing structured data sets | generalization and suppression under privacy models (k-anonymity, l-diversity, t-closeness, differential privacy) | the data set is released as anonymized | the whole table |

What deid-kit adds to that picture:

- The gateway is fail-closed: it refuses a payload classed above internal, de-identifies what it
  lets through (and refuses when no redactor is available), and records every crossing — the
  sensitivity, the entity types, the count and a SHA-256 of the text, never the content — through
  an audit hook.
- The re-identification probe asks whether an excerpt can still be traced to its scope, by
  retrieval and by a second model acting as a judge, so a de-identification can be measured and
  not only assumed.
- Detection is pluggable: Presidio [1] with spaCy models is one detector behind an interface.

## References

1. Microsoft Presidio. https://github.com/microsoft/presidio
2. scrubadub. https://github.com/LeapBeyond/scrubadub
3. B. Norgeot et al. Protected Health Information filter (Philter): accurately and securely
   de-identifying free-text clinical notes. npj Digital Medicine 3, 57 (2020).
4. F. Prasser et al. ARX Data Anonymization Tool. https://arx.deidentifier.org
5. L. Sweeney. k-anonymity: a model for protecting privacy. International Journal of Uncertainty,
   Fuzziness and Knowledge-Based Systems 10(5), 557–570 (2002).
6. H. Krawczyk, M. Bellare, R. Canetti. HMAC: Keyed-Hashing for Message Authentication. RFC 2104
   (1997). (The token derivation.)
7. spaCy. https://github.com/explosion/spaCy

## Author

deid-kit is written by Dr. Alexandra Bernadotte and developed under the AICumene name.
Copyright 2026 Alexandra Bernadotte (see `NOTICE`). To cite it, use `CITATION.cff` (GitHub shows it
as "Cite this repository").

## License

deid-kit is licensed under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`, see `LICENSE`). A service that runs a modified deid-kit offers its users the
modified source, as the AGPL requires.

Use under other terms, without the obligations of the AGPL, is available under a separate
commercial license from the copyright holder. For a commercial license, contact the copyright
holder through https://github.com/aicumene.

Contributions are accepted under a contributor license agreement; see `CONTRIBUTING.md`.
