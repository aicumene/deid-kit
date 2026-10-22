// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026 Alexandra Bernadotte
//! What crosses to the model, end to end with deid-agent itself (the scripted agent tests/fake_acp_agent.py,
//! a stand-in for the model's server): each request recorded with its bytes and checked; a name the person
//! hides, at once and after a start with a new vault; the review's account of what is hidden and of the
//! likely names that are not; a task or a review sent only when the person approves it, each answer kept
//! with who gave it; and the matter as a graph. Needs python3 (the scripted agent).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const TOKEN: &str = "disclosure-test-token";
const CLIENT: &str = "Harriet Wexley";
const FIRM: &str = "Brightwater Maritime Ltd";
const HIDDEN_LATER: &str = "Jane Holloway";
const IN_CLEAR: &str = "Marta Lindqvist";
const PERSON: &str = "Test Person";
const LETTER: &str = "# Letter\n\nDear Ms Wexley, we act for Brightwater Maritime Ltd in the charter dispute.\n\
The Tenant may break the lease on two weeks' notice.\n\
Please copy Jane Holloway and Marta Lindqvist on all replies to harriet.wexley@example.org.\n\nHarriet Wexley\n";

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A stand-in for the model's server: keeps each request's body, answers with one short message.
fn model() -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let got = Arc::new(Mutex::new(Vec::new()));
    let g = got.clone();
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut length = 0usize;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; length];
            r.read_exact(&mut body).unwrap();
            g.lock().unwrap().push(String::from_utf8_lossy(&body).into_owned());
            let answer = json!({"id": "msg_1", "type": "message", "role": "assistant", "model": "stand-in",
                                "content": [{"type": "text", "text": "Noted."}], "stop_reason": "end_turn",
                                "usage": {"input_tokens": 10, "output_tokens": 2}}).to_string();
            let mut s = s;
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                           answer.len());
        }
    });
    (port, got)
}

struct Matter {
    root: PathBuf,
    folder: PathBuf,
    upstream: u16,
    got: Arc<Mutex<Vec<String>>>,
}

fn matter(name: &str) -> Matter {
    let root = std::env::temp_dir().join(format!("deid-disclosure-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let folder = root.join("matter");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::create_dir_all(root.join("home")).unwrap();
    std::fs::write(folder.join("letter.md"), LETTER).unwrap();
    std::fs::write(root.join("seeds.toml"), format!("scope = \"test-matter\"\n\n[[entity]]\ntype = \"individual\"\nname = \"{CLIENT}\"\n\
role = \"client\"\n\n[[entity]]\ntype = \"company\"\nname = \"{FIRM}\"\n")).unwrap();
    let (upstream, got) = model();
    Matter { root, folder, upstream, got }
}

struct Agent {
    child: Child,
    port: u16,
}

impl Drop for Agent {
    fn drop(&mut self) {
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let _ = self.child.wait();
    }
}

fn start(m: &Matter, vault: &str, approve: bool) -> Agent {
    let port = free_port();
    let fake = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fake_acp_agent.py");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_deid-agent"));
    cmd.args(["--folder", m.folder.to_str().unwrap(), "--scope", "test-matter", "--port", &port.to_string()])
        .args(["--vault", m.root.join("agent").join(vault).to_str().unwrap()])
        .args(["--audit", m.root.join("agent/audit.jsonl").to_str().unwrap(), "--usage-log", ""])
        .args(["--seeds", m.root.join("seeds.toml").to_str().unwrap()])
        .args(["--agent-command", &format!("python3 {} Solo review", fake.display())])
        .args(["--upstream", &format!("http://127.0.0.1:{}", m.upstream), "--person", PERSON, "--no-guard", "--dev-login"])
        .env("DEID_AGENT_TOKEN", TOKEN).env("HOME", m.root.join("home")).env_remove("ANTHROPIC_API_KEY")
        .stdout(Stdio::null()).stderr(Stdio::null());
    if approve {
        cmd.arg("--approve-transfers");
    }
    let agent = Agent { child: cmd.spawn().unwrap(), port };
    let until = Instant::now() + Duration::from_secs(30);
    while Instant::now() < until {
        if let Some((200, state)) = try_call(port, "GET", "/deid-agent/api/state", None) {
            if state["status"] == "ready" {
                return agent;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("deid-agent did not get ready");
}

fn try_call(port: u16, method: &str, path: &str, body: Option<&Value>) -> Option<(u16, Value)> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(30))).ok()?;
    let data = body.map(Value::to_string).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\nContent-Type: application/json\r\n\
x-api-key: the-persons-own\r\nanthropic-version: 2023-06-01\r\nContent-Length: {}\r\n\r\n{data}", data.len()).ok()?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).ok()?;
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n")?;
    let status = head.split(' ').nth(1)?.parse().ok()?;
    Some((status, serde_json::from_str(rest).unwrap_or(Value::String(rest.to_string()))))
}

fn call(a: &Agent, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    try_call(a.port, method, path, body.as_ref()).unwrap_or_else(|| panic!("{method} {path}: no answer"))
}

/// The page's events, as they come.
fn events(a: &Agent) -> mpsc::Receiver<Value> {
    let (tx, rx) = mpsc::channel();
    let mut s = TcpStream::connect(("127.0.0.1", a.port)).unwrap();
    write!(s, "GET /deid-agent/api/events HTTP/1.0\r\nHost: 127.0.0.1\r\nx-deid-agent-token: {TOKEN}\r\n\r\n").unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(s).lines().map_while(Result::ok) {
            if let Some(data) = line.strip_prefix("data: ") {
                if tx.send(serde_json::from_str(data).unwrap()).is_err() {
                    break;
                }
            }
        }
    });
    rx
}

/// The next event `want` takes, the ones before it in `seen`.
fn next(rx: &mpsc::Receiver<Value>, seen: &mut Vec<Value>, want: impl Fn(&Value) -> bool) -> Value {
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let left = until.saturating_duration_since(Instant::now());
        let e = rx.recv_timeout(left).unwrap_or_else(|_| panic!("no such event; seen: {seen:#?}"));
        if want(&e) {
            return e;
        }
        seen.push(e);
    }
}

fn ask_model(a: &Agent, m: &Matter, text: &str) -> String {
    let before = m.got.lock().unwrap().len();
    let (status, _) = call(a, "POST", "/v1/messages", Some(json!({"model": "stand-in", "max_tokens": 10,
                                                                  "messages": [{"role": "user", "content": text}]})));
    assert_eq!(status, 200);
    let got = m.got.lock().unwrap();
    assert_eq!(got.len(), before + 1, "one request reached the model's server");
    got[before].clone()
}

#[test]
fn what_crossed_is_recorded_and_checked_and_a_name_the_person_hides_stays_hidden() {
    let m = matter("crossings");
    let text = format!("Write to {CLIENT} and {HIDDEN_LATER} at {FIRM}.");
    {
        let a = start(&m, "vault.sqlite", false);
        let sent = ask_model(&a, &m, &text);
        assert!(!sent.contains(CLIENT) && !sent.contains(FIRM), "the listed names crossed: {sent}");
        assert!(sent.contains(HIDDEN_LATER), "control: a name nobody hid crosses as written: {sent}");

        let (_, list) = call(&a, "GET", "/deid-agent/api/crossings", None);
        let list = list["crossings"].as_array().unwrap().clone();
        assert_eq!(list.len(), 1, "{list:?}");
        assert_eq!(list[0]["to"], json!(format!("http://127.0.0.1:{}", m.upstream)));
        assert_eq!(list[0]["action"], json!("/v1/messages"));
        assert!(list[0]["tokens"].as_u64().unwrap() >= 2);
        let id = list[0]["id"].as_u64().unwrap();
        let (status, one) = call(&a, "GET", &format!("/deid-agent/api/crossing?id={id}"), None);
        assert_eq!(status, 200);
        assert_eq!(one["sent"], json!(sent), "the record is the bytes the model's server got");
        assert_eq!(one["check"]["sha256_matches"], json!(true));
        assert_eq!(one["check"]["clean"], json!(true), "{:?}", one["check"]);

        let (status, added) = call(&a, "POST", "/deid-agent/api/entity", Some(json!({"type": "person", "name": HIDDEN_LATER})));
        assert_eq!(status, 200, "{added}");
        assert_eq!(added["by"], json!(PERSON));
        assert_eq!(added["kept"], json!(true));
        let token = added["token"].as_str().unwrap().to_string();
        assert!(token.starts_with("PERSON_"), "{token}");
        let files: Vec<PathBuf> = std::fs::read_dir(m.root.join("agent/entities")).unwrap().flatten().map(|e| e.path()).collect();
        assert_eq!(files.len(), 1);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&files[0]).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(std::fs::read_to_string(&files[0]).unwrap().contains(HIDDEN_LATER));

        // the first request, checked again: it carried a name hidden since
        let (_, again) = call(&a, "GET", &format!("/deid-agent/api/crossing?id={id}"), None);
        assert_eq!(again["check"]["clean"], json!(false));
        assert_eq!(again["check"]["found"], json!([HIDDEN_LATER]));

        let sent = ask_model(&a, &m, &text);
        assert!(!sent.contains(HIDDEN_LATER), "the hidden name crossed: {sent}");
        assert!(sent.contains(&token), "{sent}");

        let (status, view) = call(&a, "GET", "/deid-agent/api/review?path=letter.md", None);
        assert_eq!(status, 200);
        let hidden = &view["hidden"];
        for name in [CLIENT, FIRM, HIDDEN_LATER, "harriet.wexley@example.org"] {
            assert!(!hidden["model_text"].as_str().unwrap().contains(name), "{name} in what the model reads");
        }
        let tokens = hidden["tokens"].as_array().unwrap();
        let why = |name: &str| tokens.iter().find(|t| t["name"] == json!(name)).map(|t| t["why"].clone()).unwrap_or(Value::Null);
        assert_eq!(why(HIDDEN_LATER), json!("added"));
        assert_eq!(why(CLIENT), json!("list"));
        assert_eq!(why(FIRM), json!("list"));
        assert_eq!(why("harriet.wexley@example.org"), json!("pattern"));
        assert!(tokens.iter().find(|t| t["name"] == json!(HIDDEN_LATER)).unwrap()["by"] == json!(PERSON));
        let clear: Vec<&str> = hidden["candidates"].as_array().unwrap().iter().map(|c| c["text"].as_str().unwrap()).collect();
        assert_eq!(clear, vec![IN_CLEAR], "the likely names not hidden");
        assert!(hidden["counts"]["PERSON"].as_u64().unwrap() >= 2, "{}", hidden["counts"]);
    }
    // a start with a new vault: the name the person hid is hidden still, and who hid it is known
    let a = start(&m, "new-vault.sqlite", false);
    let sent = ask_model(&a, &m, &text);
    assert!(!sent.contains(HIDDEN_LATER) && !sent.contains(CLIENT), "{sent}");
    let (_, list) = call(&a, "GET", "/deid-agent/api/entities", None);
    let list = list["entities"].as_array().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], json!(HIDDEN_LATER));
    assert_eq!(list[0]["by"], json!(PERSON));
    assert!(list[0]["token"].as_str().is_some_and(|t| t.starts_with("PERSON_")));
    // what the page cannot hide is said so
    let (status, _) = call(&a, "POST", "/deid-agent/api/entity", Some(json!({"type": "planet", "name": "Mars"})));
    assert_eq!(status, 400);
    let (status, _) = call(&a, "POST", "/deid-agent/api/entity", Some(json!({"type": "person", "name": "x"})));
    assert_eq!(status, 400);
}

fn records(m: &Matter) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(m.folder.join(".legix/transfers")).into_iter().flatten().flatten()
        .map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).unwrap()).unwrap()).collect();
    out.sort_by(|a: &Value, b: &Value| a["at"].as_str().cmp(&b["at"].as_str()));
    out
}

#[test]
fn a_task_and_a_review_go_only_when_the_person_approves_them() {
    let m = matter("approval");
    let a = start(&m, "vault.sqlite", true);
    let rx = events(&a);
    let mut seen = Vec::new();
    let (_, state) = call(&a, "GET", "/deid-agent/api/state", None);
    assert_eq!(state["approve_transfers"], json!(true));

    // a task, refused: it never reaches the agent
    let task = format!("Summarise the letter for {CLIENT}.");
    let (status, started) = call(&a, "POST", "/deid-agent/api/prompt", Some(json!({"text": task})));
    assert_eq!(status, 202);
    let asked = next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "asked");
    assert_eq!(asked["what"], json!("task"));
    assert_eq!(asked["run"], started["run"]);
    assert_eq!(asked["summary"]["task"], json!(task));
    assert!(!asked["summary"]["model_text"].as_str().unwrap().contains(CLIENT), "{}", asked["summary"]);
    assert_eq!(asked["summary"]["to"], json!(format!("http://127.0.0.1:{}", m.upstream)));
    let (status, _) = call(&a, "POST", "/deid-agent/api/prompt", Some(json!({"text": "another"})));
    assert_eq!(status, 409, "one task waits for its answer at a time");
    let id = asked["id"].as_str().unwrap().to_string();
    let (status, _) = call(&a, "POST", "/deid-agent/api/approve", Some(json!({"id": id, "verdict": "refuse"})));
    assert_eq!(status, 200);
    next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "refused");
    let end = next(&rx, &mut seen, |e| e["kind"] == "turn_end");
    assert_eq!(end["stopReason"], json!("not_approved"));
    next(&rx, &mut seen, |e| e["kind"] == "files");
    assert!(!seen.iter().any(|e| e["kind"] == "user"), "the refused task reached the agent: {seen:#?}");
    let (status, _) = call(&a, "POST", "/deid-agent/api/approve", Some(json!({"id": id, "verdict": "approve"})));
    assert_eq!(status, 404, "an answered request is answered");

    // a review, approved and remembered for the document as it is
    let (status, started) = call(&a, "POST", "/deid-agent/api/review", Some(json!({"path": "letter.md"})));
    assert_eq!(status, 202);
    let asked = next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "asked");
    assert_eq!(asked["what"], json!("review"));
    assert_eq!(asked["summary"]["document"], json!("letter.md"));
    assert!(asked["summary"]["sha"].is_string());
    assert!(asked["summary"]["hidden"]["PERSON"].as_u64().is_some_and(|n| n >= 2), "{}", asked["summary"]);
    let clear: Vec<&Value> = asked["summary"]["left_in_clear"].as_array().unwrap().iter().map(|c| &c["text"]).collect();
    assert!(clear.contains(&&json!(IN_CLEAR)) && clear.contains(&&json!(HIDDEN_LATER)), "{clear:?}");
    assert_eq!(asked["summary"]["left_in_clear"][0]["kind"], json!("person"));
    // the person hides a name before answering: the record says what went when they answered
    let (status, _) = call(&a, "POST", "/deid-agent/api/entity", Some(json!({"type": "person", "name": HIDDEN_LATER})));
    assert_eq!(status, 200);
    seen.clear();
    let (status, _) = call(&a, "POST", "/deid-agent/api/approve",
                           Some(json!({"id": asked["id"], "verdict": "approve", "remember": true})));
    assert_eq!(status, 200);
    let approved = next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "approved");
    assert_eq!(approved["by"], json!(PERSON));
    assert!(approved["record"].as_str().unwrap().starts_with(".legix/transfers/"));
    next(&rx, &mut seen, |e| e["kind"] == "review" && e["state"] == "started" && e["run"] == started["run"]);
    let done = next(&rx, &mut seen, |e| e["kind"] == "review" && e["state"] != "started");
    assert_eq!(done["state"], json!("done"), "{done}");

    // the same document, unchanged: not asked again, recorded all the same
    let (status, _) = call(&a, "POST", "/deid-agent/api/review", Some(json!({"path": "letter.md"})));
    assert_eq!(status, 202);
    seen.clear();
    let standing = next(&rx, &mut seen, |e| e["kind"] == "approval");
    assert_eq!(standing["state"], json!("approved"));
    assert_eq!(standing["standing"], json!(true));
    next(&rx, &mut seen, |e| e["kind"] == "review" && e["state"] == "done");

    // a task asked about, then stopped: nothing recorded for it
    let (status, _) = call(&a, "POST", "/deid-agent/api/prompt", Some(json!({"text": "Draft a reply."})));
    assert_eq!(status, 202);
    next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "asked");
    call(&a, "POST", "/deid-agent/api/cancel", None);
    next(&rx, &mut seen, |e| e["kind"] == "approval" && e["state"] == "cancelled");
    next(&rx, &mut seen, |e| e["kind"] == "turn_end");

    let kept = records(&m);
    let verdicts: Vec<(&str, &str, bool)> = kept.iter()
        .map(|r| (r["what"].as_str().unwrap(), r["verdict"].as_str().unwrap(), r["standing"] == json!(true))).collect();
    assert_eq!(verdicts, vec![("task", "refused", false), ("review", "approved", false), ("review", "approved", true)]);
    assert!(kept.iter().all(|r| r["by"] == json!(PERSON)));
    assert_eq!(kept[1]["document"], json!("letter.md"));
    let clear: Vec<&Value> = kept[1]["left_in_clear"].as_array().unwrap().iter().map(|c| &c["text"]).collect();
    assert!(clear.contains(&&json!(IN_CLEAR)) && !clear.contains(&&json!(HIDDEN_LATER)), "{clear:?}");
    let (_, listed) = call(&a, "GET", "/deid-agent/api/transfers", None);
    assert_eq!(listed["transfers"].as_array().unwrap().len(), 3);

    // the matter as a graph: the document, its people hidden and in clear, the event its review found
    let (status, g) = call(&a, "GET", "/deid-agent/api/graph", None);
    assert_eq!(status, 200);
    let nodes = g["graph"]["nodes"].as_array().unwrap();
    let edges = g["graph"]["edges"].as_array().unwrap();
    let node = |f: &dyn Fn(&Value) -> bool| nodes.iter().find(|n| f(n)).cloned();
    let doc = node(&|n| n["kind"] == "document" && n["label"] == "letter.md").expect("the document");
    let client = node(&|n| n["label"] == json!(CLIENT)).expect("the client");
    assert_eq!(client["kind"], json!("person"));
    assert_eq!(client["hidden"], json!(true));
    assert_eq!(client["why"], json!("list"));
    let firm = node(&|n| n["label"] == json!(FIRM)).expect("the firm");
    assert_eq!(firm["kind"], json!("organization"));
    let clear = node(&|n| n["label"] == json!(IN_CLEAR)).expect("the name in clear");
    assert_eq!(clear["hidden"], json!(false));
    let event = node(&|n| n["kind"] == "event").expect("an event from the review");
    assert_eq!(event["what"], json!("break the lease"));
    let edge = |from: &Value, to: &Value, kind: &str| edges.iter().any(|e| e["from"] == from["id"] && e["to"] == to["id"] && e["kind"] == kind);
    assert!(edge(&doc, &client, "mentions") && edge(&doc, &firm, "mentions") && edge(&doc, &clear, "mentions"));
    assert!(edge(&event, &doc, "in"));
    assert!(!nodes.iter().any(|n| n["label"] == json!("harriet.wexley@example.org")), "an address is counted, not drawn");
    assert!(doc["identifiers"].as_u64().unwrap() >= 1);
}

#[test]
fn without_the_flag_nothing_is_asked() {
    let m = matter("no-approval");
    let a = start(&m, "vault.sqlite", false);
    let rx = events(&a);
    let mut seen = Vec::new();
    let (status, _) = call(&a, "POST", "/deid-agent/api/review", Some(json!({"path": "letter.md"})));
    assert_eq!(status, 202);
    next(&rx, &mut seen, |e| e["kind"] == "review" && e["state"] == "done");
    assert!(!seen.iter().any(|e| e["kind"] == "approval"), "{seen:#?}");
    assert!(records(&m).is_empty());
}

/// A contract's own title and headings are no names to hide (Bernio, 22.10.2026: "Supply Agreement" was offered as a
/// name the model could read); the people and companies in it still are.
#[test]
fn a_title_or_a_heading_is_not_offered_as_a_name() {
    let text = "# Supply Agreement\n\n## General Terms\n\nAnnex Standard Contractual Clauses apply. \
                Tamsin Okafor signs for Brightwater Maritime Ltd under the Liefervertrag Anlage.";
    let found: Vec<(String, String)> = deidkit::agents::disclosure::candidates(text, &[]).iter()
        .map(|c| (c["text"].as_str().unwrap().to_string(), c["kind"].as_str().unwrap().to_string())).collect();
    let texts: Vec<&str> = found.iter().map(|(t, _)| t.as_str()).collect();
    assert!(!texts.iter().any(|t| t.contains("Agreement") || t.contains("Terms") || t.contains("Clauses") || t.contains("Anlage")),
            "no title or heading: {found:?}");
    assert!(found.contains(&("Tamsin Okafor".into(), "person".into())), "{found:?}");
    assert!(found.iter().any(|(t, k)| t.contains("Brightwater Maritime Ltd") && k == "organization"), "{found:?}");
}
