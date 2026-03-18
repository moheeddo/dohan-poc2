"""
Vercel Serverless Function — Collab Portal API
Handles /api/v1/collab/* routes via rewrite in vercel.json.

Storage priority:
  1. GitHub Gist (persistent) — GIST_STORAGE_ID + GIST_STORAGE_TOKEN
  2. /tmp JSON file (ephemeral fallback)

Data separation:
  - collab_comments.json — comments + next_id (write-on-comment only)
  - collab_online.json   — online users (write-on-poll, never touches comments)
"""
from http.server import BaseHTTPRequestHandler
import json, os, time, uuid, fcntl
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# ────────────────────────────────────────────
# GitHub Gist Persistent Storage
# ────────────────────────────────────────────
GIST_ID = os.environ.get("GIST_STORAGE_ID", "")
GIST_TOKEN = os.environ.get("GIST_STORAGE_TOKEN", "")
COMMENTS_FILE = "collab_comments.json"
ONLINE_FILE = "collab_online.json"

# Per-file caches (short TTL to reduce API calls)
_cache = {
    COMMENTS_FILE: {"data": None, "ts": 0},
    ONLINE_FILE: {"data": None, "ts": 0},
}

EMPTY_COMMENTS = {"comments": [], "next_id": 0}
EMPTY_ONLINE = {"online": {}}


def gist_ok():
    return bool(GIST_ID and GIST_TOKEN)


def _gist_headers():
    return {
        "Authorization": f"token {GIST_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "collab-api",
    }


def _gist_load(filename, default, force=False):
    """Load a single file from the Gist. Uses cache unless force=True."""
    c = _cache[filename]
    now = time.time()
    if not force and c["data"] is not None and now - c["ts"] < 5:
        return c["data"]
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{GIST_ID}",
            headers=_gist_headers(),
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            gist = json.loads(resp.read())
            # Update cache for ALL files we find in the gist
            for fname, fobj in gist.get("files", {}).items():
                if fname in _cache:
                    try:
                        _cache[fname]["data"] = json.loads(fobj["content"])
                        _cache[fname]["ts"] = now
                    except (json.JSONDecodeError, KeyError):
                        pass
            if _cache[filename]["data"] is not None:
                return _cache[filename]["data"]
            return default.copy()
    except Exception as e:
        print(f"[Gist Load {filename}] {e}")
        return c["data"] if c["data"] is not None else default.copy()


def _gist_save(filename, data):
    """Save a single file to the Gist (PATCH only that file)."""
    _cache[filename]["data"] = data
    _cache[filename]["ts"] = time.time()
    try:
        body = json.dumps({
            "files": {
                filename: {
                    "content": json.dumps(data, ensure_ascii=False)
                }
            }
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/gists/{GIST_ID}",
            data=body,
            headers={**_gist_headers(), "Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
    except Exception as e:
        print(f"[Gist Save {filename}] {e}")


# ────────────────────────────────────────────
# /tmp File Storage (fallback)
# ────────────────────────────────────────────
TMP_COMMENTS = "/tmp/collab_comments.json"
TMP_ONLINE = "/tmp/collab_online.json"


def _tmp_load(filepath, default):
    try:
        with open(filepath, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return default.copy()


def _tmp_save(filepath, data):
    with open(filepath, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)


# Unified load/save — comments
def _load_comments(force=False):
    if gist_ok():
        return _gist_load(COMMENTS_FILE, EMPTY_COMMENTS, force=force)
    return _tmp_load(TMP_COMMENTS, EMPTY_COMMENTS)


def _save_comments(data):
    if gist_ok():
        _gist_save(COMMENTS_FILE, data)
    else:
        _tmp_save(TMP_COMMENTS, data)


# Unified load/save — online
def _load_online():
    if gist_ok():
        return _gist_load(ONLINE_FILE, EMPTY_ONLINE)
    return _tmp_load(TMP_ONLINE, EMPTY_ONLINE)


def _save_online(data):
    if gist_ok():
        _gist_save(ONLINE_FILE, data)
    else:
        _tmp_save(TMP_ONLINE, data)


# ────────────────────────────────────────────
# Migration: move old single-file data to split files
# ────────────────────────────────────────────
_migrated = False

def _maybe_migrate():
    global _migrated
    if _migrated:
        return
    _migrated = True
    if not gist_ok():
        return
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{GIST_ID}",
            headers=_gist_headers(),
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            gist = json.loads(resp.read())
        files = gist.get("files", {})
        # If old single file exists and new files don't
        if "collab_data.json" in files and COMMENTS_FILE not in files:
            old = json.loads(files["collab_data.json"]["content"])
            comments_data = {
                "comments": old.get("comments", []),
                "next_id": old.get("next_id", 0),
            }
            online_data = {
                "online": old.get("online", {}),
            }
            # Write new split files and delete old file
            body = json.dumps({
                "files": {
                    COMMENTS_FILE: {"content": json.dumps(comments_data, ensure_ascii=False)},
                    ONLINE_FILE: {"content": json.dumps(online_data, ensure_ascii=False)},
                    "collab_data.json": None,  # delete old file
                }
            }).encode()
            req2 = urllib.request.Request(
                f"https://api.github.com/gists/{GIST_ID}",
                data=body,
                headers={**_gist_headers(), "Content-Type": "application/json"},
                method="PATCH",
            )
            with urllib.request.urlopen(req2, timeout=8) as resp2:
                resp2.read()
            print("[Migration] Split collab_data.json → comments + online files")
            _cache[COMMENTS_FILE] = {"data": comments_data, "ts": time.time()}
            _cache[ONLINE_FILE] = {"data": online_data, "ts": time.time()}
    except Exception as e:
        print(f"[Migration] {e}")


# ────────────────────────────────────────────
# Users Database
# ────────────────────────────────────────────
USERS = {
    "KHNP-KDH-2026": {"name": "김도한", "title": "차장", "org": "한수원", "role": "PM", "color": "#1e40af"},
    "KHNP-PSY-2026": {"name": "박소연", "title": "주임", "org": "한수원", "role": "실무", "color": "#7c3aed"},
    "KHNP-SJS-2026": {"name": "서진수", "title": "교수", "org": "한수원", "role": "자문위원", "color": "#0891b2"},
    "KHNP-JSS-2026": {"name": "조성수", "title": "교수", "org": "한수원", "role": "자문위원", "color": "#059669"},
    "KHNP-KJI-2026": {"name": "구진일", "title": "교수", "org": "한수원", "role": "자문위원", "color": "#d97706"},
    "EY-COLLAB-2026": {"name": "EY 컨설팅팀", "title": "", "org": "EY", "role": "컨설팅", "color": "#ffe600"},
    "UPSTAGE-COLLAB-2026": {"name": "Upstage 기술팀", "title": "", "org": "Upstage", "role": "AI 기술", "color": "#6366f1"},
}


# ────────────────────────────────────────────
# Handler
# ────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _route(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")
        try:
            idx = parts.index("collab")
            rest = parts[idx + 1:]
        except (ValueError, IndexError):
            rest = []
        endpoint = rest[0] if rest else ""
        extra = rest[1] if len(rest) > 1 else ""
        params = parse_qs(parsed.query)
        return endpoint, extra, params

    def _auth(self, access_code="", session_id=""):
        user = USERS.get(access_code)
        if user:
            return user
        return None

    # ── CORS ──
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ──
    def do_GET(self):
        _maybe_migrate()
        ep, extra, params = self._route()
        if ep == "comments":
            self._get_comments(params)
        elif ep == "online":
            self._get_online(params)
        elif ep == "health":
            self._json(200, {"status": "ok", "storage": "gist" if gist_ok() else "tmpfile", "gist_id": GIST_ID[:8] + "..." if GIST_ID else ""})
        else:
            self._json(404, {"detail": "Not found"})

    # ── POST ──
    def do_POST(self):
        _maybe_migrate()
        ep, extra, params = self._route()
        body = self._body()
        if ep == "login":
            self._login(body)
        elif ep == "comments":
            self._post_comment(body)
        else:
            self._json(404, {"detail": "Not found"})

    # ── DELETE ──
    def do_DELETE(self):
        _maybe_migrate()
        ep, extra, params = self._route()
        if ep == "comments" and extra:
            access_code = params.get("access_code", [""])[0]
            self._delete_comment(extra, access_code)
        else:
            self._json(404, {"detail": "Not found"})

    # ──────────────────────────────────────────
    # Login
    # ──────────────────────────────────────────
    def _login(self, body):
        code = body.get("access_code", "")
        user = USERS.get(code)
        if not user:
            self._json(401, {"detail": "유효하지 않은 접속 코드입니다."})
            return
        session_id = str(uuid.uuid4())
        self._touch_online(user)
        self._json(200, {"session_id": session_id, "user": user})

    # ──────────────────────────────────────────
    # Get Comments
    # ──────────────────────────────────────────
    def _get_comments(self, params):
        section = params.get("section", [""])[0]
        since_id = int(params.get("since_id", ["0"])[0])

        data = _load_comments()
        comments = data.get("comments", [])

        if section:
            comments = [c for c in comments if c.get("section") == section]
        if since_id:
            comments = [c for c in comments if c.get("id", 0) > since_id]

        self._json(200, {"comments": comments})

    # ──────────────────────────────────────────
    # Post Comment
    # ──────────────────────────────────────────
    def _post_comment(self, body):
        access_code = body.get("access_code", "")
        session_id = body.get("session_id", "")
        section = body.get("section", "")
        content = body.get("content", "").strip()
        parent_id = body.get("parent_id", "")
        images = body.get("images", [])
        # Limit: max 3 images, each max ~1MB base64
        if isinstance(images, list):
            images = [img for img in images[:3] if isinstance(img, str) and len(img) < 1_500_000]
        else:
            images = []

        if not content and not images:
            self._json(400, {"detail": "내용 또는 이미지를 입력하세요."})
            return

        user = self._auth(access_code, session_id)
        if not user:
            self._json(401, {"detail": "인증 실패. 다시 로그인하세요."})
            return

        # Force-reload comments to get latest state (prevent stale overwrites)
        data = _load_comments(force=True)
        data["next_id"] = data.get("next_id", 0) + 1
        comment_id = data["next_id"]

        comment = {
            "id": comment_id,
            "section": section,
            "content": content,
            "parent_id": parent_id,
            "images": images,
            "author": user.get("name", ""),
            "org": user.get("org", ""),
            "role": user.get("role", ""),
            "color": user.get("color", "#6b7280"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "timestamp": time.time(),
        }

        data.setdefault("comments", []).append(comment)
        _save_comments(data)

        # Update online separately (won't touch comments file)
        self._touch_online(user)
        self._json(200, {"comment": comment})

    # ──────────────────────────────────────────
    # Delete Comment
    # ──────────────────────────────────────────
    def _delete_comment(self, comment_id_str, access_code):
        try:
            comment_id = int(comment_id_str)
        except ValueError:
            self._json(400, {"detail": "Invalid comment ID"})
            return

        user = self._auth(access_code)
        if not user:
            self._json(401, {"detail": "인증 실패."})
            return

        # Force-reload to get latest
        data = _load_comments(force=True)
        is_admin = user.get("name") == "김도한"
        data["comments"] = [
            c for c in data.get("comments", [])
            if not (c.get("id") == comment_id and (c.get("author") == user.get("name") or is_admin))
        ]
        _save_comments(data)
        self._json(200, {"ok": True})

    # ──────────────────────────────────────────
    # Online Users (separate from comments!)
    # ──────────────────────────────────────────
    def _get_online(self, params=None):
        access_code = (params or {}).get("access_code", [""])[0]
        user = USERS.get(access_code)
        if user:
            self._touch_online(user)

        data = _load_online()
        online = data.get("online", {})
        users = [v for v in online.values() if time.time() - v.get("last_seen", 0) < 300]
        self._json(200, {"online": users})

    def _touch_online(self, user):
        """Update online status — writes ONLY to online file, never touches comments."""
        info = {"name": user["name"], "org": user["org"], "color": user["color"], "last_seen": time.time()}
        data = _load_online()
        data.setdefault("online", {})[user["name"]] = info
        _save_online(data)

    def log_message(self, format, *args):
        pass
