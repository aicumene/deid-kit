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
- **What the cache did.** The proxy reads the usage of every answer it relays — the model, the input
  the cache did not cover, the cache writes by lifetime and the cache reads, the output — and
  `deid-agent` adds them up per task: the task's cost at Anthropic's list prices
  (`deidkit.agents.pricing`, as of 25 September 2026), with the cache and as it would be without it.
  A write costs 1.25× the input price for 5 minutes and 2× for an hour; a read costs 0.1× on most
  models, 0.05× on Claude Opus 5.5 and 0.025× on Claude Fable 5.1. MEASURED 07.10.2026: Claude Code
  writes its tools and instructions (about 30K tokens) to the 1-hour cache when a session starts, so
  the first question costs about twice what it would without the cache, and every later one in the
  hour reads it at a twentieth of the price on Claude Opus 5.5. The page and the usage log show both
  figures; on a personal sign-in nothing is billed per token, and they say what the same work costs on
  the organization's key.
- **An agent's own options.** `--agent-meta NAME=JSON` (or `--session-meta JSON` for a single agent)
  is sent as the session's `_meta`; the Claude adapter reads `claudeCode.options`, e.g.
  `{"tools": ["Read", "Write", "Edit", "Bash"], "strictMcpConfig": true}`: only the tools a matter
  needs, and no MCP servers of the person's own (claude.ai connectors included).
- **Claude Code held by the kernel, whole (macOS, 16.10.2026).** deid-agent starts Claude Code under a
  Seatbelt profile of its own (`agents::confine`), as it does Codex; its tools (Read, Write, Edit, Grep)
  and everything it starts run there:
  - the matter folder open, `/Users`, `/Volumes`, the shared temp folders and the folder's neighbours
    closed; Claude Code's install, its settings (read only) and this session's transcript, lock and
    temp folder open; the person's sign-in in the login keychain, which Claude Code reads with
    `/usr/bin/security`;
  - closed for writing, so nothing written now runs later outside any sandbox: `~/.claude.json`,
    `~/.claude/settings.json`, and in the matter folder `.git`, `.claude`, `.codex` and `.mcp.json`;
  - the network: deid-agent alone, on its port. Its https goes through deid-agent (`HTTPS_PROXY`),
    which opens a tunnel only to the sign-in (`platform.claude.com`, `auth.openai.com`) and refuses
    every other host in its log; no name can even be looked up. The person's claude.ai connectors
    are off (`ENABLE_CLAUDEAI_MCP_SERVERS=false`).

  Its shell commands go further. macOS starts no second sandbox inside a first, so Claude Code's own
  command sandbox cannot run there (deid-agent turns it off in the flag-tier settings). Claude Code's
  shell is instead `agent-shell`, beside deid-agent, copied into the session as `bin/zsh`: the one
  program Claude Code's profile lets start outside it, and it puts each command at once into the
  stricter profile beside it — the matter folder and the session's temp folder, no keychain, no
  network at all, not even deid-agent. git runs without the person's `~/.gitconfig`, and Apple's
  `/usr/bin/git` keeps its cache in the session's temp folder (`xcrun_db`). The person is still asked
  before each command. `--claude-unconfined` leaves Claude Code as it is.

  Before Claude Code starts, deid-agent tries its profile with the environment it will have: Claude
  Code starts in it (`--version`), reads the person's sign-in in it when it reads it outside
  (`auth status`), and its shell commands start through the launcher. What an older macOS, a newer
  Claude Code or an install the profile does not reach would break shows there; Claude Code then
  starts as below, and the page says why. deid-agent makes Claude Code's folders under `~/.claude`
  first (inside its profile it may write in them, not make them), and takes macOS's quarantine off
  its copy of the launcher: copied out of an app from a downloaded disk image, macOS kills it at its
  start. Without `agent-shell` beside deid-agent, or off macOS, or when the check fails, only the
  shell commands are held, by Claude Code's own sandbox:
  - no network at all (`network.allowedDomains: []`, `strictAllowlist`);
  - `/Users`, `/Volumes` and other homes closed to reading, the matter folder open again;
  - writes to the folder and a temp folder only;
  - no command outside the sandbox (`allowUnsandboxedCommands: false`), and no start without it
    (`failIfUnavailable`).

  Measured on Claude Code 2.1.293 and macOS 26.6: tests/confine_claude.rs tries each boundary under
  the kernel, and a run through deid-agent read, ran `ls` and `git status` and edited a file.
- **agent-guard on every call.** With `--guard PATH` (by default `$DEID_AGENT_GUARD`, which the
  desktop's runtime sets to its own, or `agent-guard` on PATH), deid-agent hooks agent-guard to
  Claude Code's request and to each tool call, before and after. It decides a call before it runs:
  allow, ask (the person's card), or deny. It journals the call in `<the audit's folder>/guard/<scope>.jsonl`,
  sealed line to line, under the person's home, where the sandboxed commands cannot read (Claude Code's
  own profile opens this matter's journal and the files beside it named after it). On allow it
  prints nothing, so Claude Code's own permission flow still asks the person. `--no-guard` leaves it
  out. Codex is confined by deid-agent's own Seatbelt profile too (its network as Claude Code's), and
  matteragent has no shell.
- **Codex and ChatGPT's own backend.** Codex signed in with ChatGPT reads its plan, its limits and its
  workspace's policies at `chatgpt.com/backend-api/` before its first session; refused there, the
  session fails (ACP -32603, "failed to load workspace requirements"; measured 23.10.2026 with 0.163).
  A Codex whose network reaches deid-agent alone therefore starts with deid-agent's `/backend-api/` as
  its `chatgpt_base_url`. deid-agent passes a GET or HEAD there to ChatGPT's backend at the same path
  and refuses anything else, like every request it cannot de-identify; the model's requests go where
  they went. Such a Codex also starts with deid-agent as its model provider (`model_provider = "deid"`
  and the provider above, as `-c` settings), not only each session: before any session it asks for
  its list of models with the provider it started with, and signed in with ChatGPT that was
  `chatgpt.com/backend-api/codex/models`, straight. Refused, Codex falls back to the list it carries,
  so the session still starts; through deid-agent it gets the list ChatGPT gives (measured 24.10.2026
  with 0.159.3). A confined Codex gets this. So does an unconfined one with `--agents-walled`: the agents
  are walled in by what runs deid-agent (a container's firewall, as in bernio-agents-container),
  though deid-agent does not confine them. deid-agent then writes Codex's wrapper in a folder another
  user may pass through, since a container runs Codex as a user of its own. An unconfined Codex
  without the flag is started as it is. When an agent does not start, the page names the hosts
  deid-agent refused it a tunnel to.
- **More than one agent.** `--agent NAME=COMMAND`, repeated, lists the agents the page offers,
  instead of `--agent-command`; the first that can start works first. `--agent-upstream NAME=URL`
  sends that agent's model requests to a server of the organization's own (the Messages API) and
  gives it no key, since the organization's key is for Anthropic. A server that takes a key of its own
  (llama-server's `--api-key`, Strata) gets it from the secrets line: `--secrets-stdin` with
  `"upstream_keys": {"NAME": "…"}`. The proxy holds the key and the agent gets a stand-in, as with the
  organization's key. Only the agent's requests go out with it, as `Authorization: Bearer` and
  `x-api-key`, and a refusal is told as that server's. With or without a key, an own server gets only
  the agent's requests (those with its stand-in), and no credential a request brings: an idle Claude
  Code's background requests, with its own sign-in, are refused. `--agent-unavailable NAME=REASON`
  lists an agent that cannot start on this machine, and why, without offering it. The usage line
  names the agent.
- **Switching keeps the work.** Each agent the person has worked with keeps its process and session
  while another works. Picked again between tasks, it goes on in the same session — its context as
  it was, and its model's prompt cache with it, since the context is only ever added to — and the
  work done meanwhile by the others comes as a brief at the head of its next prompt: only what it has
  not seen, in the same words for the same work (no times, counts or costs in it), so a brief does
  not unsettle the cache. During a task the person can hand the task over (`POST
  /deid-agent/api/agent` with `"handover": true`; the page asks first): the step under way stops —
  `session/cancel`, the asks it waits on answered as cancelled, an agent that does not stop within
  30 s stopped — and the one picked carries the task on in its own session, told what was done.
  Without `handover`, as the desktop's chat asks, a switch during a task is refused.
- **Contexts stay apart.** A brief is made from the record the person sees on the page — what was
  asked, what each agent said, its steps and how each ended, the person's answers to its asks, the
  documents changed — never from an agent's context: not its model's reasoning, not what it read.
  The agent told reads again what it needs, under its own permissions. The brief goes in that
  agent's prompt, so through the proxy like all it sends: its model sees tokens in it, not names.
  The page shows each brief as it was given.
- **Sub-agents on the organization's own servers.** `--subagents HOSTS` (matteragent's hosts file)
  hands the narrow jobs of a task — read, find, extract, calc, check, and consult where a head model
  answers — to sub-agents next to the folder, on models the organization runs; every quote in their
  reports is checked against its file. Claude Code and Codex get them as the MCP server `matter`
  (`--subagents-command`, default `matteragent-mcp` on the PATH), and their own cloud sub-agents are
  taken away: Claude Code's `Task` tool is disallowed, and the proxy drops Codex's `collaboration`
  tools from its requests (no Codex setting removes them: measured 09.10.2026 with 0.161).
  matteragent gets its Subagent tool. The server's tools run without asking — they only read, inside
  the folder — and the proxy restores real names in their arguments, so a sub-agent reads the
  documents as they are and only its report, tokenized, goes to the cloud model. Codex keeps MCP
  tools off its model's list (deferred, in code mode), so it is told of them in its instructions;
  confined Codex may read the hosts file and the server's installation besides the folder.
  `--subagents-timeout` (900 s) is how long Claude Code and Codex wait for one report. Without
  `--subagents` nothing changes.
- **Reviewing a document.** `GET /deid-agent/api/review?path=` gives a document of the folder with
  its layers, as the desktop's review tab shows them (`deidkit/agents/review.py`): its text (a Word
  document's is `.legix/text/<path>.md`, which the app writes), where each token of the scope's engine
  stands in it (`deid`: what a model sees as a token, found by walking the tokenized text against the
  original), the norms it refers to (`norms`: the server's citation extractor, copied without its
  knowledge base, so a mention is found, not checked), and the agents' notes. `POST` with `{"path"}`
  has the agent at work review it: it is asked for one JSON object of notes, each on a quote of the
  document's own words, with its kind (risk, question, point, or clause — what a clause does: type,
  who, what, for whom, when), aspect, importance 1–5, the norm it turns on and a note. The proxy
  tokenizes the task and the text as always. Each quote is looked up in the text by code — typographic
  quotes, dashes, white space, Markdown's escapes and an ellipsis between parts are allowed for —
  and a note whose quote is not there is not kept; the `review` event says how many were kept and why
  the others were not. A kept note is a W3C Web Annotation, one file each in
  `.legix/markup/<path>/<id>.json`, with the agent and the model that answered as its creator. The
  matter's notes folder is versioned and synced by leGix with the documents, and the host does not
  count it as the agent's work. Nothing in a note file is trusted when it is read: its words are
  looked up in the text again, so a note shows where they are now, or that they are gone.
- **The person's side of the review.** The person proposes changes as Word's tracked changes
  (`POST /deid-agent/api/review-suggest`: these words, or none, to read so), and decides of each change
  and each note of an agent (`review-decide`): a change accepted or rejected, a note useful, not
  useful or wrong, with what they say and what they would have written instead. An agent's note may
  carry its own wording for the quote, which the person accepts like a change. Accepted, a change is
  made in the document when its text is the document itself (Markdown, plain text); a Word document
  is changed in Word, and the decision is kept. The person also writes in the editor
  (`review-text`): the whole text, over the text it was opened with (its SHA-256: a document changed
  meanwhile is refused), or a new .md or .txt document — a draft the agent wrote in the chat, saved
  with who wrote it. Every one of these is the person's annotation in the same notes folder, and
  after each task deid-agent notes which text documents the agent added or changed, with the model.
  `GET review-dataset` makes records of it all — each decision with the agent, model and note or
  change it was about, and each revision a person made over a document an agent wrote — every text
  in them tokenized by the scope's engine, the person a role and not a name: data for the harness
  and for learning (RL) that can leave the Mac.
- **What crosses, shown to the person, who answers for it** (21 October 2026; Rust only, no Python
  twin: `rust/src/agents/disclosure.rs`).
  - *Why each passage is hidden.* The review's `hidden` gives the text exactly as the model is sent it
    (`model_text`), each token with its real name, how often it stands there, and why the vault hides
    it: `added` (by the person, with who and when), `list` (the matter's seed list), `party`,
    `identifier` (of a listed name), `mailbox` (a name read from an e-mail address), `pattern` (an
    e-mail address, IBAN, card or phone number), `name part`, `address`, `vault` (learned earlier). It
    also gives the counts by type, any token the walk could not place, and `candidates`: words that
    look like a person's or an organisation's name but are not hidden. Those are judged by the vault's
    own rules for a person's name and by its legal forms for an organisation, once a phrase's opening
    words ("The", "Dear") are dropped.
  - *The person hides a name.* `POST entity` with `{"type": "person" | "organization" | "vessel" |
    "account" | "other", "name", "role"?}` enrols the name at once, so the next request carries its
    token, and answers with the token. The name is kept with who hid it and when, in a seed file only
    its owner reads (`<the vault's folder>/entities/<scope>-<hash>.toml`, 0600). It is read again at
    every start, so a new vault hides it too. `GET entities` lists these names.
  - *The model's echo of a name hidden since.* Every turn, the proxy resends the model's own words
    exactly as the model wrote them, so the prompt cache holds. A name the model wrote in clear, before
    the person hid it, would then cross in clear on every later turn. Now those words go back
    tokenized instead, both in Messages and in Responses (`anthropic::safe_restores`;
    `tests/restore_after_enrolment.rs`).
  - *What went to the model.* Every request is recorded as it leaves: the audit line, the task or
    review it served (`run`), and its bytes, which are kept until they pass 64 MB in all. `GET
    crossings` (`?run=` for one run) lists them, and a `crossing` event announces each one. `GET
    crossing?id=` gives one request's bytes with a check made on the spot:
    - the bytes hash to the audit line's SHA-256;
    - no value the vault now hides is among them. A request checked after the person hid a name it
      carried says so.
  - *The person's approval* (`--approve-transfers`). A task (`POST prompt`) or a review (`POST review`)
    sends nothing until the person answers its `approval` event. The event carries a summary: the task
    as the model reads it, or the document and its SHA-256; what is hidden, by type; the likely names
    left in clear; where the requests go; the agent and the model. The person answers with `POST
    approve {"id", "verdict": "approve" | "refuse", "remember"?}`:
    - `remember` covers that document while it is unchanged, or tasks for the session;
    - refused, a task ends with `turn_end` (`not_approved`), and a review with `failed` and `refused`;
    - `cancel` withdraws the question.

    Each approval and refusal is kept in `.legix/transfers/<id>.json` with who gave it and when, and a
    standing approval is recorded each time it is used. `GET transfers` lists them.
  - *The matter as a graph.* `GET graph` returns nodes and edges:
    - nodes: the folder's documents; the people and organisations in them, hidden or in clear (a
      name part counts as its owner; e-mail addresses and numbers are counted on the document, not
      drawn); and the events its reviews found (who does what, for whom, and when, linked to the
      tokens their who and for-whom name);
    - edges: a document mentions a party, an event is in a document, a party acts in an event or it
      is done for them.

  `tests/disclosure.rs` drives all of this end to end with deid-agent itself, the scripted agent and a
  stand-in model server.
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
