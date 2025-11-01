import os, io, json, time, sys, threading, pathlib, hmac, hashlib, subprocess, re
import pandas as pd
import requests
from flask import Flask, request, jsonify, abort
from typing import Optional

# ========= CONFIG =========
OWNER   = "miriambenouaghrem"
REPO    = "test_rep"
PATH    = "Articles.csv"     # path inside the repo
BRANCH  = "main"

# ========= LOCAL SAVE LOCATION =========
# Automatically detect your project folder and create a 'data' subfolder for mirrored files.
try:
    PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
except NameError:  # e.g. running inside a notebook
    PROJECT_ROOT = pathlib.Path.cwd().resolve()

LOCAL_DIR   = os.getenv("NEWSBOT_LOCAL_DIR", str(PROJECT_ROOT / "data"))
CSV_PATH    = str(pathlib.Path(LOCAL_DIR) / "Articles.csv")
STATUS_PATH = str(pathlib.Path(LOCAL_DIR) / "mirror_status.json")
LOG_PATH    = str(pathlib.Path(LOCAL_DIR) / "webhook.log")

pathlib.Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

# ========= AUTH & SERVER =========
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", None)  # optional (avoid GitHub rate limit or for private repos)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")  # must match your GitHub webhook secret (leave empty to disable check)

PORT        = 8081
AUTO_TUNNEL = True  # requires `cloudflared` on PATH

# ========= HELPERS =========
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line); sys.stdout.flush()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def write_status(state: str, *, sha: Optional[str]=None, rows: Optional[int]=None, detail: Optional[str]=None):
    payload = {
        "state": state,               # "idle" | "running" | "done" | "error"
        "sha": sha,
        "rows": rows,
        "last_update_ts": time.time(),
        "csv_path": CSV_PATH,
        "detail": detail,
    }
    pathlib.Path(STATUS_PATH).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def read_status() -> dict:
    p = pathlib.Path(STATUS_PATH)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def verify_github_signature(raw_body: bytes, signature_header: str) -> bool:
    """Validate X-Hub-Signature-256 if WEBHOOK_SECRET set."""
    if not WEBHOOK_SECRET:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    sent = signature_header.split("=", 1)[-1]
    mac = hmac.new(WEBHOOK_SECRET.encode("utf-8"), msg=raw_body, digestmod=hashlib.sha256)
    return hmac.compare_digest(sent, mac.hexdigest())

def fetch_github_csv_text() -> tuple[str, str]:
    """Return (sha, csv_text) from GitHub Contents API."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{PATH}?ref={BRANCH}"
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    js = r.json()

    if js.get("encoding") == "base64" and js.get("content"):
        import base64
        text = base64.b64decode(js["content"]).decode("utf-8", errors="replace")
    else:
        dl = requests.get(js["download_url"], timeout=30)
        dl.raise_for_status()
        text = dl.text

    return js.get("sha") or "", text

def robust_rows_from_text(text: str) -> int:
    try:
        return len(pd.read_csv(io.StringIO(text)))
    except Exception:
        return len(pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip"))

# ========= FLASK APP =========
app = Flask(__name__)
JOB = {"state": "idle", "sha": None, "rows": None, "detail": None, "thread": None}

def worker_save(sha: str, text: str):
    try:
        JOB.update({"state": "running", "sha": sha, "rows": None, "detail": None})
        write_status("running", sha=sha)

        # Save CSV to disk
        pathlib.Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(text)

        rows = robust_rows_from_text(text)
        JOB.update({"state": "done", "rows": rows})
        write_status("done", sha=sha, rows=rows)
        log(f"✅ Mirrored Articles.csv • rows={rows} • sha={sha[:7]}")
    except Exception as e:
        msg = str(e)
        JOB.update({"state": "error", "detail": msg})
        write_status("error", sha=sha, detail=msg)
        log(f"❌ Error while mirroring CSV: {msg}")

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, time=time.time())

@app.get("/status")
def status():
    persisted = read_status()
    out = {
        "job_state": JOB["state"],
        "job_sha": JOB["sha"],
        "job_rows": JOB["rows"],
        "job_detail": JOB["detail"],
        "persisted": persisted,
    }
    return jsonify(out)

@app.post("/refresh")
def refresh():
    raw = request.get_data()
    sig_hdr = request.headers.get("X-Hub-Signature-256", "")
    if not verify_github_signature(raw, sig_hdr):
        abort(401, description="Bad signature")

    log("🚀 /refresh received")

    sha, text = fetch_github_csv_text()

    persisted = read_status()
    if persisted.get("sha") == sha and persisted.get("state") == "done":
        log("ℹ️ SHA unchanged; skipping write.")
        return jsonify(status="nochange", sha=sha[:7]), 200

    if JOB["state"] == "running" and JOB.get("sha") == sha:
        return jsonify(status="already_running", sha=sha[:7]), 202

    th = threading.Thread(target=worker_save, args=(sha, text), daemon=True)
    JOB.update({"state":"running", "sha": sha, "rows": None, "detail": None, "thread": th})
    write_status("running", sha=sha)
    th.start()
    return jsonify(status="started", sha=sha[:7]), 202

# ========= CLOUDFLARE TUNNEL =========
def try_start_cloudflared():
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except FileNotFoundError:
        log("ℹ️ cloudflared not found on PATH. Install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")
        return

    public_url = None
    for line in proc.stdout:
        m = re.search(r'https://[-\w]+\.trycloudflare\.com', line)
        if m:
            public_url = m.group(0)
            break
    if public_url:
        print("HEALTH :", public_url + "/healthz")
        print("STATUS :", public_url + "/status")
        print("REFRESH:", public_url + "/refresh")
        log(f"🌐 Webhook URL (Cloudflare): {public_url}/refresh")
    else:
        log("⚠️ Could not detect Cloudflare public URL (tunnel may still be running).")

# ========= MAIN =========
if __name__ == "__main__":
    log(f"📁 Local dir: {LOCAL_DIR}")
    pathlib.Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    threading.Thread(target=lambda: app.run(port=PORT), daemon=True).start()

    if AUTO_TUNNEL:
        try_start_cloudflared()

    while True:

        time.sleep(3600)
