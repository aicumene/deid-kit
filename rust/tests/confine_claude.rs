// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! Claude Code confined whole (`agents::confine`), tried under the kernel: the profiles deid-agent writes are
//! applied with sandbox-exec to small programs that reach for what a confined Claude Code may and may not —
//! a home folder of the test's own (no file of the person's is touched), two matters side by side, deid-agent's
//! port and another one. Every refusal stands beside the same act allowed one step over, so a refusal is the
//! profile's and not a missing file's.
#![cfg(target_os = "macos")]

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

use deidkit::agents::confine;

struct Case {
    base: PathBuf,
    home: PathBuf,
    matter: PathBuf,
    other: PathBuf,
    session: PathBuf,
    port: u16,
    elsewhere: u16,
    confined: confine::ConfinedClaude,
    _listeners: (TcpListener, TcpListener),
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A Claude Code that starts and is signed in: what the profile is tried with before the real one starts.
const CLAUDE: &str = "#!/bin/sh\ncase \"$1\" in --version) echo '9.9.9 (Claude Code)';; auth) echo '{\"loggedIn\": true}';; esac\n";

fn case(name: &str) -> Case {
    case_full(name, &[], CLAUDE, false)
}

fn case_writing(name: &str, journals: &[&str]) -> Case {
    case_full(name, journals, CLAUDE, false)
}

/// A case with a Claude Code of `script` (`OTHER` in it is the other matter), and the launcher copied from a
/// quarantined file when `quarantined`, as in an app from a downloaded disk image.
fn case_full(name: &str, journals: &[&str], script: &str, quarantined: bool) -> Case {
    let base = std::env::temp_dir().join(format!("deid-confine-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let base = {
        std::fs::create_dir_all(&base).unwrap();
        base.canonicalize().unwrap()
    };
    let (home, matter, other, session) =
        (base.join("home"), base.join("cases/Matter A"), base.join("cases/Matter B"), base.join("session"));
    write(&home.join(".claude/settings.json"), "{}\n");
    write(&home.join(".claude/history.jsonl"), "{\"display\":\"another matter's question\"}\n");
    write(&home.join(".claude/projects/-elsewhere/t.jsonl"), "another session\n");
    write(&home.join(".claude.json"), "{}\n");
    std::fs::create_dir_all(home.join("Library/Keychains")).unwrap();
    write(&matter.join("contract.txt"), "Delivery: 30 days.\n");
    write(&other.join("notes.txt"), "Matter B's notes.\n");
    let claude = base.join("install/versions/9.9.9");
    write(&claude, &script.replace("OTHER", &s(&other)));
    std::fs::set_permissions(&claude, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let mut launcher = PathBuf::from(env!("CARGO_BIN_EXE_agent-shell"));
    if quarantined {
        let copy = base.join("app/agent-shell");
        std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
        std::fs::copy(&launcher, &copy).unwrap();
        assert!(Command::new("/usr/bin/xattr").args(["-w", "com.apple.quarantine", "0081;6ad2c0ba;Telegram;"]).arg(&copy)
            .status().unwrap().success());
        launcher = copy;
    }
    let (a, b) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
    let (port, elsewhere) = (a.local_addr().unwrap().port(), b.local_addr().unwrap().port());
    let confined = confine::confined_claude(&home.to_string_lossy(), &matter.to_string_lossy(), Some(&s(&claude)), &launcher,
                                            &session, &confine::network(port, &[]), &[],
                                            &journals.iter().map(|w| s(&base.join(w))).collect::<Vec<_>>())
        .expect("laid out");
    Case { base, home, matter, other, session, port, elsewhere, confined, _listeners: (a, b) }
}

/// `program args…` under Claude Code's own profile: did it succeed?
fn claude(c: &Case, program: &str, args: &[&str]) -> bool {
    Command::new(confine::SANDBOX_EXEC).arg("-f").arg(c.session.join("claude.sb")).arg(program).args(args)
        .current_dir(&c.matter).output().unwrap().status.success()
}

/// What `program args…` says on stderr under Claude Code's own profile.
fn claude_err(c: &Case, program: &str, args: &[&str]) -> String {
    let out = Command::new(confine::SANDBOX_EXEC).arg("-f").arg(c.session.join("claude.sb")).arg(program).args(args)
        .current_dir(&c.matter).output().unwrap();
    assert!(!out.status.success());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A shell command as Claude Code runs it: through the launcher, from inside its own profile.
fn command(c: &Case, script: &str) -> bool {
    claude(c, &c.session.join("bin/zsh").to_string_lossy(), &["-c", script])
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

#[test]
fn claude_code_reads_its_matter_and_its_own_settings_but_no_other_matter_or_session() {
    let c = case("read");
    assert!(claude(&c, "/bin/cat", &[&s(&c.matter.join("contract.txt"))]));
    assert!(!claude(&c, "/bin/cat", &[&s(&c.other.join("notes.txt"))]));
    assert!(claude(&c, "/bin/cat", &[&s(&c.home.join(".claude/settings.json"))]));
    assert!(claude(&c, "/bin/cat", &[&s(&c.home.join(".claude.json"))]));
    assert!(!claude(&c, "/bin/cat", &[&s(&c.home.join(".claude/history.jsonl"))]));
    assert!(!claude(&c, "/bin/cat", &[&s(&c.home.join(".claude/projects/-elsewhere/t.jsonl"))]));
}

#[test]
fn claude_code_writes_its_matter_and_its_transcript_but_not_what_would_run_outside_later() {
    let c = case("write");
    let sh = |script: String| claude(&c, "/bin/sh", &["-c", &script]);
    assert!(sh(format!("echo note > '{}'", s(&c.matter.join("notes.txt")))));
    let transcripts = confine::claude_projects(&s(&c.home), &s(&c.matter));
    let own = transcripts.trim_start_matches("(subpath \"").trim_end_matches("\")");
    assert!(sh(format!("mkdir -p '{own}' && echo turn >> '{own}/t.jsonl'")));
    // its settings and the person's: a hook or an MCP server written there would run at the next claude, unsandboxed
    assert!(!sh(format!("echo x >> '{}'", s(&c.home.join(".claude/settings.json")))));
    assert!(!sh(format!("echo x >> '{}'", s(&c.home.join(".claude.json")))));
    // and the same in the matter folder: git's settings and hooks, Claude Code's and Codex's, the MCP servers
    assert!(!sh(format!("mkdir -p '{0}/.git' && echo x > '{0}/.git/config'", s(&c.matter))));
    assert!(!sh(format!("mkdir -p '{0}/sub/.claude' && echo x > '{0}/sub/.claude/settings.json'", s(&c.matter))));
    assert!(!sh(format!("echo x > '{}'", s(&c.matter.join(".mcp.json")))));
    assert!(sh(format!("mkdir -p '{0}/sub' && echo x > '{0}/sub/claude-notes.md'", s(&c.matter))));
}

#[test]
fn claude_code_reaches_deid_agent_and_no_other_port() {
    let c = case("net");
    assert!(claude(&c, "/usr/bin/nc", &["-z", "-G", "3", "127.0.0.1", &c.port.to_string()]));
    assert!(!claude(&c, "/usr/bin/nc", &["-z", "-G", "3", "127.0.0.1", &c.elsewhere.to_string()]));
}

#[test]
fn its_shell_commands_get_the_matter_folder_and_nothing_else_of_the_persons() {
    let c = case("cmd");
    assert!(command(&c, &format!("cat '{}'", s(&c.matter.join("contract.txt")))));
    assert!(command(&c, &format!("echo draft > '{}'", s(&c.matter.join("draft.txt")))));
    assert!(!command(&c, &format!("cat '{}'", s(&c.other.join("notes.txt")))));
    // what Claude Code itself may read, its commands may not: its settings, its sign-in through the Keychain's command
    assert!(!command(&c, &format!("cat '{}'", s(&c.home.join(".claude/settings.json")))));
    assert!(command(&c, "/bin/echo the shell runs"));
    assert!(!command(&c, "/usr/bin/security list-keychains"));
    // no network at all, not even deid-agent
    assert!(!command(&c, &format!("/usr/bin/nc -z -G 3 127.0.0.1 {}", c.port)));
    assert!(command(&c, &format!("echo temp > \"$TMPDIR/x\" 2>/dev/null || echo temp > '{}/x'", s(&c.session.join("tmp")))));
}

#[test]
fn the_launcher_opens_nothing_wider_from_inside() {
    let c = case("again");
    let launcher = s(&c.session.join("bin/zsh"));
    let contract = s(&c.matter.join("contract.txt"));
    // a command that starts the launcher again is already in a sandbox: it cannot read the profile, nor start a second
    let again = claude_err(&c, &launcher, &["-c", &format!("{launcher} -c \"cat '{contract}'\" ; exit $?")]);
    assert!(again.contains("command.sb: Operation not permitted") || again.contains("sandbox_apply"), "{again}");
    // nor does a copy of it elsewhere, with a profile of its own beside it, start outside Claude Code's profile
    let copy = s(&c.matter.join("zsh"));
    assert!(claude(&c, "/bin/cp", &[&launcher, &copy]));
    assert!(claude(&c, "/bin/sh", &["-c", &format!("echo '(version 1)(allow default)' > '{}'", s(&c.matter.join("command.sb")))]));
    let copied = claude_err(&c, &copy, &["-c", &format!("cat '{}'", s(&c.other.join("notes.txt")))]);
    assert!(copied.contains("sandbox_apply"), "{copied}");
    // and the launcher itself cannot be changed, nor its profile
    assert!(!claude(&c, "/bin/sh", &["-c", &format!("echo '(version 1)(allow default)' > '{}'", s(&c.session.join("bin/command.sb")))]));
}

#[test]
fn the_launcher_runs_only_as_a_shell_with_its_profile_beside_it() {
    let base = std::env::temp_dir().join(format!("deid-launcher-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let named = base.join("sh");
    std::fs::copy(env!("CARGO_BIN_EXE_agent-shell"), &named).unwrap();
    let out = Command::new(&named).args(["-c", "echo ran"]).output().unwrap();
    assert_eq!(out.status.code(), Some(126));
    assert!(String::from_utf8_lossy(&out.stderr).contains("runs as zsh or bash"));
    let zsh = base.join("zsh");
    std::fs::copy(env!("CARGO_BIN_EXE_agent-shell"), &zsh).unwrap();
    let out = Command::new(&zsh).args(["-c", "echo ran"]).output().unwrap();
    assert_eq!(out.status.code(), Some(126));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no profile beside it"));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn seatbelt_names_a_server_by_its_port() {
    assert_eq!(confine::seatbelt_address("http://127.0.0.1:18097/v1").as_deref(), Some("localhost:18097"));
    assert_eq!(confine::seatbelt_address("http://localhost:8012").as_deref(), Some("localhost:8012"));
    assert_eq!(confine::seatbelt_address("https://models.example.org/v1").as_deref(), Some("*:443"));
    assert_eq!(confine::seatbelt_address("http://[::1]:9000/").as_deref(), Some("localhost:9000"));
    assert_eq!(confine::seatbelt_address("http://user@10.0.0.5:8097").as_deref(), Some("*:8097"));
    assert_eq!(confine::seatbelt_address("not a url"), None);
}

#[test]
fn transcripts_are_named_as_claude_code_names_them() {
    assert_eq!(confine::claude_projects("/Users/p", "/Users/p/Matters/Case A.1"),
               "(subpath \"/Users/p/.claude/projects/-Users-p-Matters-Case-A-1\")");
    // a letter beyond ASCII is one '-' for each UTF-16 unit
    assert_eq!(confine::claude_projects("/Users/p", "/Users/p/Дело"), "(subpath \"/Users/p/.claude/projects/-Users-p-----\")");
    let long = format!("/Users/p/{}", "x".repeat(300));
    let rule = confine::claude_projects("/Users/p", &long);
    assert!(rule.starts_with("(regex #\"^/Users/p/\\.claude/projects/-Users-p-xxx") && rule.ends_with("x\")"));
}

#[test]
fn agent_guard_keeps_this_matters_journal_and_its_side_files_but_not_another_matters() {
    let c = case_writing("journal", &["guard/case-a.jsonl"]);
    write(&c.base.join("guard/case-b.jsonl"), "another matter's tool calls\n");
    let sh = |script: String| claude(&c, "/bin/sh", &["-c", &script]);
    let guard = s(&c.base.join("guard"));
    assert!(sh(format!("echo '{{}}' >> '{guard}/case-a.jsonl'")));
    assert!(sh(format!("echo '{{}}' > '{guard}/case-a.resmon' && echo '{{}}' > '{guard}/case-a.dedup.json'")));
    assert!(!sh(format!("cat '{guard}/case-b.jsonl'")));
    assert!(!sh(format!("echo x > '{guard}/case-ab.jsonl'")));
}

#[test]
fn a_launcher_from_a_downloaded_app_still_starts_the_commands() {
    // copied out of a quarantined app, macOS kills it at its start (137) unless deid-agent takes the mark off
    let c = case_full("quarantine", &[], CLAUDE, true);
    assert!(command(&c, &format!("cat '{}'", s(&c.matter.join("contract.txt")))));
    let marks = Command::new("/usr/bin/xattr").arg(c.session.join("bin/zsh")).output().unwrap();
    assert!(!String::from_utf8_lossy(&marks.stdout).contains("com.apple.quarantine"));
}

#[test]
fn claude_codes_folders_are_made_before_it_starts() {
    let c = case("folders");
    for d in [".claude/projects", ".claude/sessions", ".claude/session-env"] {
        assert!(c.home.join(d).is_dir(), "{d}");
    }
}

fn check(c: &Case, sign_in: bool) -> Result<(), String> {
    let env: Vec<(String, String)> = vec![("HOME".into(), s(&c.home)), ("PATH".into(), "/usr/bin:/bin".into()),
                                          ("TMPDIR".into(), format!("{}/", s(&c.confined.temp)))];
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
        .block_on(confine::check_claude(&c.confined, &c.matter, &env, sign_in))
}

#[test]
fn the_check_passes_a_claude_code_that_starts_signed_in_and_whose_commands_start() {
    let c = case("check-ok");
    assert_eq!(check(&c, true), Ok(()));
}

#[test]
fn the_check_finds_a_claude_code_installed_where_its_profile_does_not_reach() {
    // as Claude Code under nvm: its Node lies in the home folder, closed to the profile
    let c = case_full("check-start", &[], "#!/bin/sh\ncat 'OTHER/notes.txt' >/dev/null || exit 1\necho 9.9.9\n", false);
    let why = check(&c, true).unwrap_err();
    assert!(why.starts_with("Claude Code did not start in it"), "{why}");
}

#[test]
fn the_check_finds_a_sign_in_read_outside_but_not_inside() {
    let script = "#!/bin/sh\ncase \"$1\" in --version) echo 9.9.9;; auth) if cat 'OTHER/notes.txt' >/dev/null 2>&1; \
                  then echo '{\"loggedIn\": true}'; else echo '{\"loggedIn\": false}'; fi;; esac\n";
    let c = case_full("check-sign-in", &[], script, false);
    assert_eq!(check(&c, true), Err("inside it Claude Code could not read the person's sign-in".into()));
    // with the organization's key there is no sign-in of the person's to read
    assert_eq!(check(&c, false), Ok(()));
}

#[test]
fn the_check_finds_commands_that_cannot_start() {
    let c = case("check-shell");
    std::fs::set_permissions(c.session.join("bin"), std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    std::fs::remove_file(c.session.join("bin/command.sb")).unwrap();
    let why = check(&c, true).unwrap_err();
    assert!(why.starts_with("its shell commands could not start"), "{why}");
}
