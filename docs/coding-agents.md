# deid-kit with coding agents: Claude Code and Codex

A coding agent sends the model everything it reads: your prompt, the files it opens, the output
of the commands it runs, the results of its tools. When the folder it works in holds personal
data (client documents, e-mails, test fixtures, logs, database exports), that data leaves the
machine on every turn. This page describes how to put deid-kit between the agent and the model,
so that the agent works on the real files while the model receives tokens.

**Status.** The proxy for Claude Code (Anthropic's Messages API) is in the package:
`deid-proxy`, with the extra `proxy`. The proxy for Codex, the hook scripts, the working-copy
commands and the scanner are specified on this page but are not part of the package yet.

## Quick start: Claude Code

```sh
pip install -e '.[proxy]'                       # from a checkout of this repository
deid-proxy --scope client-a --seeds ~/private/deid.toml
```

`deid.toml` lists the people and organisations you already know (see `src/deidkit/seedfile.py`
for the format). It holds real names, so keep it outside the repository you work in:

```toml
[[entity]]
type = "individual"
name = "Ada Brenner"

[[entity]]
type = "company"
name = "Harrowgate Freight Ltd"
```

Then, in the shell where you start Claude Code:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_CUSTOM_HEADERS="x-deid-scope: client-a"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

- **Where things are kept.** The vault goes to `~/.deid/vault.sqlite` and the audit to
  `~/.deid/audit.jsonl`; both files are readable by their owner only.
- **Recording.** `--record FILE` also writes what crossed, in tokens, so that you can check it.
- **What is detected.** Besides the known names, the proxy detects only patterns: e-mail
  addresses, IBANs, payment cards and international phone numbers (`deidkit.patterns`). It runs
  no name recognition, because such a model reads identifiers in code as people. A person whose
  e-mail address appears is enrolled from the address as well.
- **Not applied yet.** Street addresses and initials, which the vault can tokenize, are not yet
  applied by the proxy.

**Measured on 28 September 2026.** The test used Claude Code 2.1.263 signed in with a claude.ai
subscription and a folder of invented names. Claude Code read a letter, wrote a summary file and
answered.
- **What crossed:** none of the names, e-mail addresses, the phone number or the IBAN, checked
  word by word and through the vault. The model saw `PERSON_85844766`, `EMAIL_53250088`,
  `IBAN_97199306`.
- **What stayed on the machine:** the summary file and the answer, both with the real values.
- **Across turns:** the second turn read 36,563 tokens from the prompt cache. The model's
  reasoning blocks went back and forth without a rejection.

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

1. **Scope.** It reads the scope from a request header (`x-deid-scope: client-a`) and falls back
   to its default scope. A scope is one salt and one token namespace. Use one scope per client,
   matter or project.
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
http_headers = { "x-deid-scope" = "client-a" }

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
