# Connect a project to deid-proxy: instructions for a coding agent

You are a coding agent, such as Claude Code or Codex. The user asked you to set up deid-kit for
a project, so that the agents working on it read and write the real files while the model
provider receives tokens instead of names. Follow the steps in order.

- **Parts 1–4** set things up once per machine and once per project.
- **Part 5** is the block to add to the project's `CLAUDE.md` or `AGENTS.md`, so every later
  session knows how to work through the proxy.
- **Part 6** covers daily maintenance.

Background is in [coding-agents.md](coding-agents.md).

## Rules for you while you set this up

- **The seed file.** It lists real names and lives in `~/.deid/seeds/`, outside every
  repository. Never write it, copy it, or paste its contents into the project, a commit, a log
  or your reply.
- **What you never print.** The vault (`~/.deid/vault.sqlite`), the seed files and the proxy's
  record. `deid-proxy check` reports counts and masked shapes only; use it rather than reading
  those files.
- **Credentials.** Never read, copy or print API keys or tokens. The proxy passes the agent's
  own credentials through; nothing in this setup needs them.
- **Names.** Ask the user for the names to protect. Do not guess them, and do not collect them
  from other sources without being asked.
- **Order.** Do not point an agent at the proxy for real work until the check in step 4.4
  passes.
- **If the proxy is down.** An agent pointed at it fails to connect. Never fix that by pointing
  the agent back at the provider. Tell the user.

## 1. Install (once per machine)

```sh
python3 -m venv ~/.deid/venv
~/.deid/venv/bin/pip install "deid-kit[proxy] @ git+https://github.com/aicumene/deid-kit"
~/.deid/venv/bin/deid-proxy --help
mkdir -p ~/.deid/seeds && chmod 700 ~/.deid ~/.deid/seeds
```

Python 3.11 or newer.

## 2. Choose the scope and write the seed file (once per client or project)

A scope is one token namespace: one per client, matter or project, named with a short ASCII id
such as `client-a`. The same person gets unrelated tokens in two scopes. Ask the user which scope
the project belongs to; several projects of one client can share it.

Write `~/.deid/seeds/<scope>.toml` with the people and organisations the user names:

```toml
scope = "client-a"
paths = ["~/matters/client-a"] # this client's project folders

[[entity]]
type = "individual"            # individual | company | vessel | account | property
name = "Ada Brenner"
role = "Director"              # optional

[[entity]]
type = "company"
name = "Harrowgate Freight Ltd"
jurisdiction_country = "GB"    # optional
```

Then run `chmod 600 ~/.deid/seeds/<scope>.toml`.

- **Spellings.** List each person once, under the name as usually written. The vault derives
  spellings, transliterations, initials-with-surname and inflected forms itself.
- **Patterns.** E-mail addresses, IBANs, payment cards and international phone numbers are
  detected without being listed. A person whose address has the form `first.last@` is enrolled
  from it.
- **Folders.** `paths` lists the client's project folders. An agent working in one of them, or
  in a folder beneath it, gets this scope without any setting in the agent. This is how Codex
  and the editor windows get the right scope.

## 3. Run the proxy (once per machine; it serves every scope)

One proxy process serves every project and both agents. It loads every seed file in
`~/.deid/seeds/` and listens on `127.0.0.1:8787`. Run it as a login service so that it is always
there.

**macOS**: write `~/Library/LaunchAgents/ai.aicumene.deid-proxy.plist`, replacing `YOU` with the
user name:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.aicumene.deid-proxy</string>
  <key>ProgramArguments</key><array>
    <string>/Users/YOU/.deid/venv/bin/deid-proxy</string>
    <string>--seeds-dir</string><string>/Users/YOU/.deid/seeds</string>
    <string>--require-scope</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/Users/YOU/.deid/proxy.log</string>
</dict></plist>
```

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.aicumene.deid-proxy.plist
```

**Linux**: write `~/.config/systemd/user/deid-proxy.service`:

```ini
[Unit]
Description=deid-proxy

[Service]
ExecStart=%h/.deid/venv/bin/deid-proxy --seeds-dir %h/.deid/seeds --require-scope
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now deid-proxy
```

`--require-scope` refuses a request from a folder that no seed file lists and that sends no
scope header, instead of letting it go out under a default scope. Give personal projects a scope
of their own: a seed file with `scope` and `paths` and no names.

Check that it answers:

```sh
curl -s http://127.0.0.1:8787/deid/health        # {"ok": true, "service": "deid-proxy"}
```

## 4. Point the agents at it (once per project)

### 4.1 Claude Code

Claude Code reads `env` from the project's settings once the user trusts the folder (and at once
in `-p` mode). Choose one of two files:

- `.claude/settings.local.json` — for this user only. Add it to `.gitignore` if Claude Code did
  not create it.
- `.claude/settings.json` — committed, for everyone on the project. Every teammate then needs the
  proxy running, and without it their Claude Code fails to connect. That is the safe way to
  fail.

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "ANTHROPIC_CUSTOM_HEADERS": "x-deid-scope: client-a",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "skipWebFetchPreflight": true
}
```

When the project's folder is listed under `paths` (step 2), `ANTHROPIC_CUSTOM_HEADERS` can be
left out: the proxy takes the scope from the folder. Keep it for a project outside its client's
folders. Merge these keys into the file if it already exists; do not overwrite other settings.
`ANTHROPIC_CUSTOM_HEADERS` needs Claude Code 2.1.227 or later. A claude.ai subscription sign-in
works through the proxy as it is, and no API key is needed.

### 4.2 Codex

Codex ignores provider settings in a project's `.codex/config.toml`. The provider therefore goes
into the user's `~/.codex/config.toml` itself, **without** a scope header; the proxy takes the
scope from the folder (step 2). This one setting covers the terminal, the Codex window of the
ChatGPT app and the editor extensions alike.

```toml
model_provider = "deid"

[model_providers.deid]
name = "OpenAI through deid-kit"
base_url = "http://127.0.0.1:8787/v1"
requires_openai_auth = true                     # ChatGPT sign-in or an API key

[features]
enable_request_compression = false

[analytics]
enabled = false

[feedback]
enabled = false
```

Put `model_provider` above the first `[table]`, and merge the rest into the existing file.

- **What changes.** From then on every Codex session goes through the proxy. With
  `--require-scope`, a session in an unlisted folder is refused, so list every folder Codex is
  used in (step 2).
- **Profiles.** A profile file (`~/.codex/<name>.config.toml`, `codex -p <name>`) works in the
  terminal only. The ChatGPT app and the editor integrations cannot select one.
- **Where Codex lives.** If `codex` is not on the `PATH`, the ChatGPT desktop app on macOS ships
  it at `/Applications/ChatGPT.app/Contents/Resources/codex`.
- **What never passes the proxy.** Codex cloud tasks and the IDE's `/cloud` command run in
  OpenAI's containers.

### 4.3 Editors and apps

The proxy is in the path whenever the agent runs on this machine and reads the settings above.

| where you work | what to set |
|---|---|
| Claude app, Code tab | the project's `.claude/settings*.json` (4.1). The app reads it as the terminal does, once you trust the folder. |
| ChatGPT app, Codex window | `~/.codex/config.toml` (4.2) |
| VS Code or VSCodium | Claude Code extension: the same `.claude/settings*.json`. Codex extension: `~/.codex/config.toml`. VSCodium installs both from Open VSX. |
| Zed | the agent server's `env` in Zed's settings, shown below |
| JetBrains IDEs | a custom agent in `~/.jetbrains/acp.json` with the same `env`. Do not use the built-in Claude Agent and Codex entries, whose routing is not documented. |

Zed, in its `settings.json`:

```json
{
  "agent_servers": {
    "claude-acp": { "type": "registry", "env": {
      "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
      "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1" } },
    "codex-acp": { "type": "registry", "env": { "MODEL_PROVIDER": "deid" } }
  }
}
```

Watch for these:

- **Other AI features.** The built-in AI of Cursor and Windsurf sends files to their own servers
  and never passes the proxy. Turn it off, or use another editor.
- **Clients with their own gateway.** A client that signs agents in through its own gateway
  (ACP's gateway authentication) replaces the base URL. Such a client is unsuitable.
- **A host that manages the provider.** When the host sets `CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST`,
  Claude Code ignores the base URL in settings files.
- **Worktrees.** A new git worktree has no `.claude/settings.local.json`. Claude Code 2.1.211 or
  later reads the main checkout's copy, and a worktree inside a listed folder gets its scope from
  `paths` anyway.
- **The only hard guarantee is the network.** Any setting can be bypassed by a misconfigured
  client. On a machine used for client work, add an outbound firewall rule that lets only the
  proxy's process reach `api.anthropic.com`, `api.openai.com` and `chatgpt.com`.

### 4.4 Check before real work

Run one short task through a separate proxy, on a separate scope, with invented names, and check
what crossed. Nothing here touches the project or the real vault.

1. Make the check folder and its seed file:

   ```sh
   mkdir -p /tmp/deid-check && cd /tmp/deid-check
   ```

   ```sh
   printf 'scope = "deid-check"\n[[entity]]\ntype = "individual"\nname = "Tamsin Okafor"\n[[entity]]\ntype = "company"\nname = "Brightwater Maritime Ltd"\n' > /tmp/deid-check/seeds.toml
   ```

   ```sh
   printf 'From: Tamsin Okafor, Director, Brightwater Maritime Ltd\nEmail: tamsin.okafor@brightwater.example\nPhone: +44 20 7946 0123\n\nBrightwater Maritime Ltd asks for payment to IBAN DE89 3704 0044 0532 0130 00.\n\nTamsin Okafor\n' > /tmp/deid-check/letter.md
   ```

2. Start a check proxy in the background on port 8788, with its own vault and a record:

   ```sh
   ~/.deid/venv/bin/deid-proxy --port 8788 --vault /tmp/deid-check/vault.sqlite --audit /tmp/deid-check/audit.jsonl --seeds /tmp/deid-check/seeds.toml --scope deid-check --record /tmp/deid-check/record.jsonl > /tmp/deid-check/proxy.log 2>&1 &
   ```

   ```sh
   curl -s http://127.0.0.1:8788/deid/health
   ```

3. Run each agent the project will use through it, from `/tmp/deid-check`. The input must be
   closed (`< /dev/null`), or the agent waits for it. If you are yourself running inside an agent
   session, start the child in a clean environment (`env -i HOME="$HOME" USER="$USER"
   PATH="$PATH" …`) so that your session's variables do not leak into it.

   ```sh
   ANTHROPIC_BASE_URL=http://127.0.0.1:8788 ANTHROPIC_CUSTOM_HEADERS="x-deid-scope: deid-check" CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude -p "Read letter.md and write summary.md with one line: who wrote, for which company, and the sender's email. Then reply with that line." --allowedTools Read Write --permission-mode acceptEdits < /dev/null
   ```

   ```sh
   codex exec --ignore-user-config --ephemeral --skip-git-repo-check -s workspace-write -c model_provider=deid -c 'model_providers.deid={name="deid",base_url="http://127.0.0.1:8788/v1",requires_openai_auth=true,http_headers={"x-deid-scope"="deid-check"}}' -c features.enable_request_compression=false "Read letter.md and write summary.md with one line: who wrote, for which company, and the sender's email. Then reply with that line." < /dev/null
   ```

4. Check what crossed:

   ```sh
   ~/.deid/venv/bin/deid-proxy check --record /tmp/deid-check/record.jsonl --vault /tmp/deid-check/vault.sqlite --scope deid-check
   ```

   The check passes when:
   - the command ends with `RESULT: no known value crossed as a word of its own` (exit status 0);
   - `summary.md` contains the invented names, not tokens;
   - the agent's reply names Tamsin Okafor.

   Words listed as "inside other words" are noise when the masked shape is plainly another word
   (`fi****me` is `filename`).

5. Stop the check proxy and delete the check folder:

   ```sh
   kill %1 2>/dev/null || pkill -f "deid-proxy --port 8788"
   rm -rf /tmp/deid-check
   ```

If the check fails, stop. Report the check's output to the user and do not point the project at
the proxy.

## 5. Add the working rules to the project

Append this block to the project's `CLAUDE.md` (Claude Code) and `AGENTS.md` (Codex), creating
the files if needed. It contains no names and may be committed.

Without it, a model that notices the tokens tends to remark that the files are "already
de-identified" and that its output "carries the tokens through". Once the proxy puts the names
back, the user reads that remark next to real names. The block tells the model the tokens are
expected.

```markdown
## Working through deid-proxy

This project's model traffic goes through deid-proxy (deid-kit). The model provider sees tokens
instead of the names of people and organisations, e-mail addresses, phone numbers, IBANs and card
numbers.

- You will see tokens such as `PERSON_48170392`, `ORG_20514477`, `EMAIL_…`, `PHONE_…`, `IBAN_…`.
  Places, dates and amounts are real.
- Write tokens exactly as you see them, in file edits, commands and answers. The proxy puts the
  real values back before a file is written or a command runs, and the user reads names.
- Do not invent tokens, and do not try to work out the real value behind one.
- A token stands for a person or organisation, not for one spelling of the name. A command with
  a token searches for the name as the vault writes it; when a search has to match a particular
  spelling (a surname alone, initials), ask the user.
- Web fetches, web search and MCP tools receive tokens, not names.
- Images and PDFs are withheld. To read a document, convert it to text on this machine first
  (`pdftotext file.pdf -`, `textutil -convert txt file.docx`) and read the text.
- If you see a name, e-mail address or number in clear where a token belongs, tell the user: it is
  missing from the seed file in `~/.deid/seeds/`.
- Never copy anything from `~/.deid/` into this repository.
```

## 6. Maintenance

- **A new name.** Add it to the scope's seed file and restart the proxy: on macOS
  `launchctl kickstart -k gui/$(id -u)/ai.aicumene.deid-proxy`, on Linux
  `systemctl --user restart deid-proxy`. Seed files are read at start.
- **A new client.** Write a new seed file with its own `scope` and `paths`, then restart the
  proxy. Claude Code projects inside those folders need the settings file from 4.1 (the header
  may be left out). Codex needs nothing more.
- **Checking a real session.** Run the proxy once with `--record ~/.deid/record.jsonl`. Then run
  `deid-proxy check --record ~/.deid/record.jsonl --scope <scope>`, and delete the record
  afterwards: it holds the session in tokens, with places, dates and amounts in clear.
- **Backup.** Back up `~/.deid/vault.sqlite` with the same care as the client files. It is the
  only way to turn earlier answers back into names.
- **Turning it off for a project.** Remove the `env` keys from the project's Claude Code settings
  and stop using the Codex profile. From then on the agents send real content to the provider;
  make sure the user wants that.
