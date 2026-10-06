# deid-kit with coding agents: Claude Code and Codex

A coding agent sends the model everything it reads: your prompt, the files it opens, the output
of the commands it runs, the results of its tools. When the folder it works in holds personal
data (client documents, e-mails, test fixtures, logs, database exports), that data leaves the
machine on every turn. This page describes how to put deid-kit between the agent and the model,
so that the agent works on the real files while the model receives tokens.

**Status.** The proxy is in the package: `deid-proxy`, with the extra `proxy`. One process
serves Claude Code (Anthropic's Messages API) and Codex (OpenAI's Responses API). The hook
scripts, the working-copy commands and the scanner are specified on this page but are not part of
the package yet.

## Quick start

To have a coding agent set this up for a project, point it at
[agent-setup.md](agent-setup.md): step-by-step instructions written for an agent, with the
config for both agents, a check with invented names, and a block for the project's `CLAUDE.md` or
`AGENTS.md`.

```sh
pip install -e '.[proxy]'                       # from a checkout of this repository
deid-proxy --scope client-a --seeds ~/private/deid.toml
```

`deid.toml` lists the people and organisations you already know (see `src/deidkit/seedfile.py`
for the format). It holds real names, so keep it outside the repository you work in:

```toml
scope = "client-a"
paths = ["~/matters/client-a"]      # the client's project folders

[[entity]]
type = "individual"
name = "Ada Brenner"

[[entity]]
type = "company"
name = "Harrowgate Freight Ltd"
```

An agent working inside a listed folder gets that scope with no setting of its own.

Then, in the shell where you start Claude Code:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_CUSTOM_HEADERS="x-deid-scope: client-a"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

For Codex, add a provider in `~/.codex/config.toml` (the full block is in the Codex section
below). It has no scope header, because the proxy takes the scope from the folder. The same
setting then covers the terminal, the ChatGPT app and the editors.

```toml
model_provider = "deid"

[model_providers.deid]
name = "OpenAI through deid-kit"
base_url = "http://127.0.0.1:8787/v1"
requires_openai_auth = true                     # ChatGPT sign-in or an API key

[features]
enable_request_compression = false
```

- **Where things are kept.** The vault goes to `~/.deid/vault.sqlite` and the audit to
  `~/.deid/audit.jsonl`; both files are readable by their owner only.
- **Recording.** `--record FILE` also writes what crossed, in tokens.
  `deid-proxy check --record FILE --scope client-a` then reports whether any known value crossed
  as a word of its own. It gives counts only, and its exit status is 1 when a value crossed.
- **What is detected.** Besides the known names, the proxy detects only patterns: e-mail
  addresses, IBANs, payment cards and international phone numbers (`deidkit.patterns`). It runs
  no name recognition, because such a model reads identifiers in code as people. A person whose
  e-mail address appears is enrolled from the address as well.
- **Short names a document defines.** A party named once and then called by a short form —
  `TOBIAS WREN of … ("TW")`, `KESTREL VENTURES LLP ("Kestrel")` — has the short form enrolled as
  its token when the definition is seen, and matched as written, case and all, from then on. Only
  the party's initials or words of its name count: `(the "Company")` stays in clear.
  `deid-agent` passes the folder's text documents through the vault before it starts the agent,
  so a task that already uses the short form is covered too.
- **Street addresses.** The street and house number (also joined by a dash, as in a file name
  `01-Lease-Musterweg-12.md`), a flat and a postal code cross as `ADDRESS_…`, one token per
  address in every request; the city and the country stay (`deidkit.address`).
- **The machine's account.** `deid-agent` enrols the account its home folder is named after, so
  the working directory and every absolute path cross as `/Users/ACCOUNT_…/…` and come back as
  on disk. A generic name such as `admin` is left alone.
- **Other folders.** `deid-agent` refuses, without asking, an action that names a path outside the
  matter folder — among its locations, or in the command it would run (`~/…`, `../…`,
  `$HOME/…`, `/tmp/…`). Another matter's names are not in this scope's vault and would cross in
  clear. Devices such as `/dev/null` and the system's own programs are allowed.
- **What a task cost.** `deid-agent` shows the agent's window as the agent reports it (ACP's
  `usage_update`: tokens in the window now, and its size) and, when a task ends, its tokens as the
  agent gives them on the answer to the task (ACP's draft `usage`: read, from the cache, written)
  with the seconds it took. Each task adds a line to `~/.deid/agent/usage.jsonl` (`--usage-log`,
  `''` for none): when, the scope, the model, how it ended, the seconds and the tokens — what a
  matter can be billed against. The line names the scope, never the folder, whose name may be a
  client's; the file is readable by its owner only. An agent that reports nothing gets the
  seconds alone.
- **Not applied yet.** Initials of a known person that no document defines, which the vault can
  tokenize (`deidkit.address.tokenize_initials`), are not yet applied by the proxy.

**Measured on 28 September 2026.** Both agents worked on the same folder of invented names, a
letter with two people, a company, two e-mail addresses, a phone number and an IBAN. Each agent
read the letter, wrote a summary file and answered.

| | Claude Code 2.1.263, claude.ai subscription | Codex 0.155 (in the ChatGPT app), ChatGPT sign-in |
|---|---|---|
| what crossed | none of the values | none of the values |
| the summary file and the answer | real values | real values |
| history | the second turn read 36,563 tokens from the prompt cache | the model's own items restored in every later request |

How it was checked: every string sent to the model was decoded (JSON inside text included) and
searched for each value as a plain substring. That check matters, because the first Codex run
leaked. In Codex's code mode a command's output comes back as JSON inside a text part, so line
breaks are the two characters `\` and `n`. The letter `n` stuck to the next word, the word
boundary failed, and a company name and a first name that each followed a line break crossed in
clear. The vault now treats a backslash escape as a boundary, as does the pattern detector, and
a test holds the case. A search that respects word boundaries is blind to exactly this, so the
check has to be a plain substring search.

## The idea

```
 your machine                                                 model provider
┌───────────────────────────────────────────────┐            ┌──────────────┐
│ files, commands, prompt (real values)         │            │              │
│                                               │   tokens   │              │
│ Claude Code / Codex ──▶ deid-kit: tokenize ───┼───────────▶│ Claude / GPT │
│                     ◀── deid-kit: detokenize ◀┼────────────┤              │
│                                               │   tokens   │              │
│ vault: salt, tokens, aliases                  │            │              │
│ audit: hashes and counts, never content       │            │              │
└───────────────────────────────────────────────┘            └──────────────┘
```

- The agent, the files and the vault stay on your machine, or on a server you trust. The model
  sees `PERSON_48170392`, `ORG_20514477`, `EMAIL_…`, `ADDRESS_…`.
- What the model sends back (its answer, and the edits and commands it asks for) gets its
  tokens replaced with the real values before the agent acts on it. Only the tokens that the
  same request sent are reversed (`vault.detokenize(..., mapping=...)`), so a token the model
  invents stays a token.
- A token in prose comes back as the document's own spelling if the request used only one,
  otherwise as the vault's canonical name. In the arguments of a local tool, a file or folder
  name, a quoted path in a command, or a line copied from a file comes back exactly as it was
  written: `03-Reply-to-Harrowgate-Freight.md` crosses as `03-Reply-to-ORG_1-Freight.md`, and a
  call to read it opens the file on disk, not `03-Reply-to-Harrowgate Freight Ltd-Freight.md`
  (`deidkit.proxy.spelling`).
- Tokens are derived from a per-scope salt, so a person has the same token in every turn and
  every session. The model sees a consistent conversation, and the provider's prompt cache keeps
  working because the history is re-sent byte for byte.
- Places and dates stay readable and street addresses become tokens, as everywhere in deid-kit
  (see the README).

## Four ways to use it

| | where deid-kit runs | what it covers | what it needs |
|---|---|---|---|
| **A. Model proxy** | a local HTTP proxy in front of the model API | everything the model receives: the prompt, files, command output, tool results, memory files, subagents | the agent pointed at the proxy |
| **B. Tool hooks** (Claude Code) | a hook script on each tool call | tool results and tool arguments; the prompt is checked but not rewritten | Claude Code hooks; the network path stays as it is |
| **C. Tokenized working copy** | a command that writes a tokenized copy of a folder and brings changes back | what the agent reads in the copy | nothing from the agent; works for agents that run in the cloud |
| **D. Pre-commit scan** | a git hook | real values going the other way, into the repository | git |

A (or B) and D together cover both directions.

## A. Model proxy

The agent is pointed at `http://127.0.0.1:8787` instead of the provider. The proxy tokenizes
every request, forwards it with the agent's own credentials, and detokenizes the answer on the
way back. The agent's credentials pass through the proxy; the proxy does not store them.

### What the proxy does with a request

1. **Scope.** It reads the scope from a request header (`x-deid-scope: client-a`). Without one it
   uses the folder the agent works in, as the agent states it in the request (Claude Code:
   "Primary working directory"; Codex: `<cwd>`), if a seed file lists that folder under `paths`.
   Otherwise it falls back to its default scope, or refuses with `--require-scope`. A scope is
   one salt and one token namespace. Use one scope per client, matter or project.
2. **Tokenize.** It tokenizes every piece of text that came from your side:
   - in Anthropic's Messages API: `system`; the `text` blocks of each message; the content of
     `tool_result` blocks; the string values in `tool_use` inputs; the text of earlier assistant
     turns, which reached your side detokenized;
   - in OpenAI's Responses API: `instructions`; the text of `input` messages; function-call
     arguments and outputs; custom tool-call input and output (the patches of `apply_patch`).

   Each piece is tokenized once and cached by its SHA-256. Later turns cost nothing extra for
   the history, and the history goes out byte-identical. When the vault learns a new name, the
   cached pieces that contain it are tokenized again, which costs one prompt-cache miss.
3. **Merge.** It combines the mappings of all the pieces with `vault.merge_mappings`, so each
   person is written one way in the reply.
4. **Withhold what it cannot read.** Images, PDFs and other binary blocks are replaced with a
   short note telling the model they were withheld; with `--binary refuse` the request is refused
   instead. Convert documents to text on the machine first. A block type the proxy does not
   know refuses the request. The model's reasoning (`thinking` and `redacted_thinking` blocks,
   OpenAI `reasoning` items) passes through untouched. It was produced from tokenized input,
   and it is signed or encrypted, so it must go back to the provider unchanged.
5. **Audit.** It records the crossing with `PrivacyGateway.cross_to_cloud(..., pre_redacted=...)`:
   sensitivity, entity types, the count and a SHA-256 of what crossed, never the content.
6. **Forward.** It forwards the agent's headers unchanged.

The proxy changes string values only. It never changes fields, the order of blocks, or cache
markers.

If any step fails, the proxy answers with an error and forwards nothing.

### What it does with the answer

- **Text.** Tokens are replaced as the answer streams. The proxy holds back a short tail that
  could be the start of a token split across two stream events.
- **Tool calls.** The proxy collects a call's arguments until the call is complete, replaces the
  tokens in the string values and emits the arguments in one piece. It does this only for tools
  that act on the machine: file reads and edits, the shell. Tools that send data elsewhere, such
  as web fetch or MCP servers that call other services, get the tokens as the model wrote them,
  so a real name never ends up in a URL. The agent's sandbox limits what shell commands can
  reach on the network.
- **Reasoning.** Passed through untouched. It shows tokens to the person reading it.
- **The model's own words.** The proxy remembers each block of the answer as the model wrote
  it, keyed by a hash of the version it handed to the agent. When the agent sends that block
  back as history, the proxy sends the original bytes, even where the model wrote a token in
  another case.

The history therefore goes out exactly as it went out before. That matters beyond caching:
providers that keep the model's reasoning across turns check that the earlier turns have not
changed. Two things can still change an earlier turn. The vault may learn a new name that
occurs there, and a cache entry may be lost. The provider then refuses the old reasoning, and
the agent drops it and carries on (see the Claude Code notes below).

### Claude Code

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_CUSTOM_HEADERS="x-deid-scope: client-a"     # Claude Code 2.1.227 or later
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

and in `settings.json`: `"skipWebFetchPreflight": true`.

- **Credentials.** The agent's own credential passes through: an API key
  (`ANTHROPIC_API_KEY`, `apiKeyHelper`), a gateway token (`ANTHROPIC_AUTH_TOKEN`), or a claude.ai
  sign-in. A claude.ai sign-in works through a custom base URL when no gateway credential
  variable is set. The upstream then requires an OAuth capability carried in `anthropic-beta`,
  which is one more reason to forward that header verbatim.
- **Endpoints.** Claude Code posts inference to `/v1/messages?beta=true` (streamed) and, when
  available, to `/v1/messages/count_tokens`. The proxy tokenizes both. It may also call
  `GET /v1/models` for model discovery, and it sends a `HEAD /api/hello` warming probe. Background
  calls (titles, summaries, compaction) and subagents go to the same base URL, so they are covered
  too.
- **Headers.** Forward `anthropic-version` and `anthropic-beta` unchanged; the second is an open
  list that changes between releases. Return `content-type`, `retry-after`, `x-should-retry` and
  the `anthropic-ratelimit-unified-*` headers, and pass error bodies through unmodified, because
  Claude Code's recovery reads the upstream's wording.
- **Body.** Leave the first `system` block as it is. It is Claude Code's attribution block, which
  the API strips only when it arrives unchanged in first place. Keep `cache_control` markers and
  block-form content as they are.
- **Streaming.** Stream, never buffer whole responses. While the proxy holds a tool call's
  arguments, it keeps forwarding `ping` events. Claude Code aborts a stream that has been
  silent for 300 seconds, and it stops reading at an event whose `content_block_start` never
  arrived.
- **Reasoning across turns.** The API's preserved-thinking check rejects a thinking block
  "bound to a different conversation" when `system`, `tools` or earlier `messages` differ from
  the request that produced it. When that happens, Claude Code removes the earlier thinking
  blocks, retries, and keeps them out from then on.
- **Other traffic.** Telemetry, error reports, feature flags, updates and `/feedback` go to their
  own hosts, not through the base URL. `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` turns them
  off, together with the fast-mode availability check. The WebFetch domain safety check sends
  the host name to `api.anthropic.com` unless `skipWebFetchPreflight` is set. MCP connectors from
  claude.ai run through Anthropic's MCP proxy; their results enter the conversation and are
  tokenized on the next request.
- **What changes.** When the base URL is not Anthropic's, MCP tool search is off by default.
  `ENABLE_TOOL_SEARCH=true` turns it back on, provided the proxy passes `tool_reference` blocks.
  Remote Control is unavailable.
- **Bedrock, Vertex, Foundry.** Claude Code can also reach these through a gateway, for example
  `ANTHROPIC_BEDROCK_BASE_URL` with `CLAUDE_CODE_SKIP_BEDROCK_AUTH=1`, and the Vertex and
  Foundry equivalents. The proxy then has to speak that provider's wire format.

### Codex

In `~/.codex/config.toml`. Codex ignores provider settings in a project's `.codex/config.toml`.

```toml
model_provider = "deid"
check_for_update_on_startup = false

[model_providers.deid]
name = "OpenAI through deid-kit"
base_url = "http://127.0.0.1:8787/v1"
requires_openai_auth = true                     # ChatGPT sign-in or an API key

[analytics]
enabled = false

[feedback]
enabled = false

[otel]
exporter = "none"
trace_exporter = "none"
metrics_exporter = "none"

[features]
apps = false
remote_plugin = false
```

- **Profiles.** A profile file (`codex -p <name>`) works in the terminal only; the ChatGPT app
  and the editor integrations run `codex app-server`, which accepts no profile. Put the provider
  in `~/.codex/config.toml` without a scope header, and let the proxy take the scope from the
  folder.
- **Credentials.** With `requires_openai_auth = true` you sign in with ChatGPT or use an API
  key; the Codex documentation describes this setting for LLM proxies. Leave `env_key` unset on
  this provider. The proxy receives `Authorization: Bearer …`, plus `ChatGPT-Account-ID` with a
  ChatGPT sign-in. With an API key it forwards to `https://api.openai.com/v1`. With a ChatGPT
  sign-in it forwards to `https://chatgpt.com/backend-api/codex`, which is where Codex itself
  connects; this is not documented for proxies. Token refresh goes to `auth.openai.com`
  directly and carries no content.
- **Transport.** A custom provider speaks HTTP with SSE only, because `supports_websockets` is
  false by default. Codex sends `POST /v1/responses`, streamed, with the full history in every
  request (`store: false`, reasoning returned as `encrypted_content`). `wire_api = "chat"` no
  longer exists. If you point the built-in provider at the proxy with `openai_base_url` instead,
  it tries a WebSocket first; the proxy answers that handshake with 426, and Codex falls back to
  HTTP. Request bodies may be zstd-compressed (`Content-Encoding: zstd`). The proxy decompresses
  them, or `features.enable_request_compression = false` turns compression off.
- **What to tokenize.** Besides `instructions` and the `input` items, `client_metadata` and the
  `x-codex-turn-metadata` header carry the repository's absolute path and its git remote URL, so
  the proxy tokenizes them too. Newer models receive the base instructions and the tool list as
  `input` items (a developer message and an `additional_tools` item) instead of in
  `instructions`.
- **Code mode.** Codex 0.155 with its default model gives the model one custom tool, `exec`,
  whose input is a short script calling other tools:
  `text(await tools.apply_patch("*** Begin Patch …"))`. The script gets real values only when
  every tool it calls is local, and each value is escaped for the string literal it lands in.
  A script that calls an MCP tool keeps the tokens.
- **No content type.** The ChatGPT backend streams without a `content-type` header, so the proxy
  decides streaming by the request's own `stream` flag.
- **What to detokenize.** Codex builds its history and runs tools from the complete item in
  `response.output_item.done`; the deltas are only for display. The proxy therefore replaces
  tokens in each completed item (message text, the `arguments` of a `function_call`, the `input`
  of a `custom_tool_call`) as well as in the text deltas. It keeps each item's
  `output_item.added` before that item's deltas, and it keeps the stream open until
  `response.completed`.
  - Local tools get real values: `exec_command` (`cmd`, `workdir`), `write_stdin`, and the
    `apply_patch` patch.
  - MCP tools keep the tokens, because Codex calls MCP servers directly.
  - Hosted web search runs at the provider with the tokens the model wrote.
- **Reasoning.** `reasoning` items (a plain summary plus `encrypted_content`) and `compaction`
  items pass through untouched.
- **Other traffic.** Codex also sends:
  - usage and health metrics, and analytics events;
  - `/feedback` uploads, which can attach local session transcripts that contain the real
    names;
  - the update check;
  - apps and remote plugins, which run as connectors through `chatgpt.com`;
  - cloud-managed requirements;
  - sign-in.

  The settings above turn off all of it except the last two.
- **Keeping a folder away from Codex's shell.** Permission profiles (beta, Codex 0.122 or later)
  can deny reads. Set `default_permissions = "<name>"`, then under
  `[permissions.<name>.filesystem]` mark a path `"deny"`. They cover spawned commands only, not
  MCP servers, web search or the model traffic, and they cannot be combined with `sandbox_mode`.
  The older `sandbox_mode` restricts writes and the network, not reads.
- **Where the proxy cannot sit.** Codex cloud tasks, `codex cloud`, the GitHub, Slack and Linear
  triggers, and the IDE's `/cloud` command run the agent in OpenAI's containers. Use the
  tokenized working copy (C) for those. Local chats in the IDE extension run the bundled CLI,
  read the same `config.toml`, and so go through the proxy.

## B. Tool hooks (Claude Code)

Claude Code's hooks can change what a tool receives and what the model sees of its result. Its
documentation names this use: redaction at `PreToolUse` for outbound tool inputs and at
`PostToolUse` for inbound tool results. Hooks work with any sign-in and also run inside
subagents.

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "deid-hook result" }] }
    ],
    "PreToolUse": [
      { "matcher": "Read|Edit|Write|Bash|Grep|Glob",
        "hooks": [{ "type": "command", "command": "deid-hook arguments" }] }
    ],
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "deid-hook prompt" }] }
    ]
  }
}
```

- **Results.** `deid-hook result` tokenizes the tool's result and returns it as
  `hookSpecificOutput.updatedToolOutput`. The model never sees the original. The hook keeps,
  per session, the tokens it has handed out.
- **Arguments.** `deid-hook arguments` replaces tokens with real values in the arguments of the
  local tools, using only the tokens handed out in this session, and returns them as
  `hookSpecificOutput.updatedInput`. The edit lands in the real file and the command runs on
  real paths. Web tools and MCP tools are not matched, so they keep the tokens.
- **Prompt.** A `UserPromptSubmit` hook cannot replace the prompt. `deid-hook prompt` checks it
  with `vault.residual_surfaces`. When it finds an enrolled name, it blocks the prompt
  (`"decision": "block"`) and shows you the tokens to write instead. It also blocks `@file`
  mentions of the folders that hold personal data, because a mentioned file goes into the prompt
  without passing through a tool.
- **Speed.** A hook is one process per tool call. The scripts talk to a small local service
  that keeps the vault loaded.

**Check before you rely on B.** Claude Code does not document one thing: whether the history it
sends back carries a tool call's arguments as the model wrote them (tokens) or as the hook
rewrote them (real values). If it is the second, real values reach the model on the next turn.
Once per Claude Code version, run a session through the proxy in record-only mode (recording,
no tokenization) and look for the rewritten arguments in the second request.

**Codex.** Codex has the same hook events; they have been stable since 0.124. In 0.158, though, a
hook can rewrite a tool's input but not its output: `updatedMCPToolOutput` is parsed but not
supported yet. A Codex hook therefore cannot tokenize what the model reads, so use the proxy for
Codex.

What hooks leave uncovered, compared with the proxy:

- files Claude Code loads by itself at start (`CLAUDE.md` and other memory files). Keep personal
  data out of them;
- pasted images;
- the model's own text answers, which reach you with tokens in them. `deid show` turns a pasted
  answer back into names locally.

To keep a folder away from the model entirely, rather than tokenized:

- add `Read` deny rules, such as `Read(./clients/**)`. They cover the file tools, Bash file
  commands Claude Code recognizes (`cat`, `head`, `tail`, `sed`), redirections and `@file`
  mentions (the last on a best-effort basis);
- add `sandbox.filesystem.denyRead` for everything else a shell command could start, such as a
  script that opens files itself.

## C. Tokenized working copy

For agents whose traffic you cannot route: Codex cloud tasks, Claude Code on the web, and chat
interfaces.

```sh
deid mirror ./matter ./matter.deid --scope client-a   # tokenized copy; the vault stays here
# run any agent in ./matter.deid, locally or remotely
deid apply ./matter.deid ./matter --scope client-a    # changes come back with real values
```

- Text files are tokenized, file and folder names included. Other files are converted to text
  on the machine, or left out.
- `apply` replaces tokens only in the lines the agent changed. When the original changed in the
  meantime, it merges three ways, as git does, and stops on a conflict.
- The agent cannot run code against real data or real services from inside the copy.

## D. Pre-commit scan

A commit is the other direction: real values leaving inside the repository, as a test
fixture, a snapshot, or a log pasted into a test. The scan reads the staged files and looks for
enrolled values with `vault.residual_surfaces`, which is read-only and enrolls nothing. It
reports counts per file, never the values, and fails the commit.

```sh
# .git/hooks/pre-commit
deid scan --staged --scope client-a
```

This covers every agent and every person who commits.

## Scope, seeds and detectors

- **Scope.** One scope per client, matter or project. It owns the salt (`vault.ensure_salt`),
  the tokens and the aliases. Two scopes give one person unrelated tokens.
- **Seeds.** Enroll the people and organisations you already know before the first request:
  `SeedEntity(type, name, role=...)` and the parties list. They can come from a `deid.yaml`
  beside the files, or from your case system through a `SeedSource`. Known names are matched
  deterministically, with their spellings, transliterations and inflected forms.
- **Detectors.** For prose (letters, contracts, notes, e-mails), use Presidio's NER in the
  text's language together with the pattern recognisers. For source code and command output,
  use the known names and the pattern recognisers only (e-mail, phone, IBAN, card and ID
  numbers). Name recognition on code turns identifiers into people. `PresidioDetector` already
  runs only the pattern recognisers for a language it has no NER model for; a wrapper makes the
  choice explicit:

  ```python
  class PatternsOnly:
      """E-mail, phone, IBAN, card and ID numbers; no name recognition."""
      def __init__(self, inner):
          self.inner = inner
      def detect(self, text, language):
          return self.inner.detect(text, "und")   # no NER model for "und": patterns only
  ```

  The proxy and the hooks know which file a tool result came from, because the tool call names
  the path, so they choose the detector per result.
- **Store.** Keep the token store on the machine, behind `TokenStore`: SQLite, or the database
  you already run. The store and the salt turn every answer back into names. Back them up and
  guard them like the data itself.
- **Glossary.** Optional here. With `vault.with_glossary`, each crossing carries one
  non-identifying line per token (legal form, role, jurisdiction). The proxy appends it to the
  newest user turn, never to the system prompt, so the cached prefix stays byte-identical.

## What crosses and what stays

**Crosses:** tokens, glossary lines when enabled, places, dates, amounts, code, and anything
the vault did not recognise.

**Stays:** the real values, the salt, the token store and the audit log.

## Limits

- Only enrolled or detected values are hidden. A name the detector misses crosses in clear.
  Measure before trusting it with real data (next section).
- Tokens hide who. Dates, amounts, places and the shape of a document can still point to a
  case. The probe measures this.
- Text only. Images, screenshots and PDFs are withheld (or refused), not tokenized.
- A name written with `\uXXXX` escapes inside JSON (`Jos\u00e9`) does not match the name it
  stands for, so it is not recognized yet.
- A newly enrolled name changes how earlier turns read, which costs one prompt-cache miss.
- The model reasons about tokens, so it cannot see the spelling, the length or the initials of a
  name.
- Keys and passwords are not deid-kit's job. Keep them away from the agent with its sandbox and
  a secret scanner.
- Traffic outside the model API, listed above for each agent, does not pass through the proxy.

## Checking that it works

1. **Record.** Run the proxy in record mode on a sample. It writes what crossed, in tokens, to a
   local file.
2. **Canaries.** Enroll a few invented names, put them into files the agent will read, and
   search the record for them. None may appear.
3. **Independent check.** Have a different model, running locally, list the personal data it
   finds in the record. Count the findings per 1,000 crossings.
4. **Traceability.** Run `reid.probe_retrieval` and `reid.probe_judge` on sampled crossings.

## License

Using deid-kit on your own machines, as described here, is covered by the AGPL-3.0. If you
modify deid-kit and run it as a service for other people, offer them the modified source. A
product or hosted service that uses deid-kit without the AGPL's obligations needs the commercial
license; contact the copyright holder through https://github.com/aicumene.

## Sources

Claude Code documentation, as of September 2026:

- Gateway compatibility guide: https://code.claude.com/docs/en/llm-gateway-protocol
- Connecting a gateway: https://code.claude.com/docs/en/llm-gateway-connect
- Environment variables: https://code.claude.com/docs/en/env-vars
- Hooks: https://code.claude.com/docs/en/hooks
- Permissions: https://code.claude.com/docs/en/permissions
- Sandboxing: https://code.claude.com/docs/en/sandboxing
- Network configuration: https://code.claude.com/docs/en/network-config

Codex, as of 28 September 2026 (developers.openai.com/codex now redirects to learn.chatgpt.com):

- Configuration reference: https://learn.chatgpt.com/docs/config-file/config-reference
- Advanced configuration: https://learn.chatgpt.com/docs/config-file/config-advanced
- Authentication: https://learn.chatgpt.com/docs/auth
- Hooks: https://learn.chatgpt.com/docs/hooks
- MCP: https://learn.chatgpt.com/docs/extend/mcp
- Permissions: https://learn.chatgpt.com/docs/permissions
- Approvals and security: https://learn.chatgpt.com/docs/agent-approvals-security
- Wire details not in the documentation are from the source at tag `rust-v0.158.0`:
  https://github.com/openai/codex/tree/rust-v0.158.0/codex-rs
