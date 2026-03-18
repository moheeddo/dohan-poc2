"""
Vercel Serverless Function — Collab Portal API
Handles /api/v1/collab/* routes via rewrite in vercel.json.

Storage priority:
  1. GitHub Gist (persistent) — GIST_STORAGE_ID + GIST_STORAGE_TOKEN
  2. /tmp JSON file (ephemeral fallback)
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
GIST_FILE = "collab_data.json"

# In-memory cache to reduce API calls (refreshed every 5s)
_gist_cache = {"data": None, "ts": 0}


def gist_ok():
    return bool(GIST_ID and GIST_TOKEN)


def _gist_load():
    """Load data from GitHub Gist with in-memory caching."""
    now = time.time()
    if _gist_cache["data"] is not None and now - _gist_cache["ts"] < 5:
        return _gist_cache["data"]
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{GIST_ID}",
            headers={
                "Authorization": f"token {GIST_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "collab-api",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            gist = json.loads(resp.read())
            content = gist["files"][GIST_FILE]["content"]
            data = json.loads(content)
            _gist_cache["data"] = data
            _gist_cache["ts"] = now
            return data
    except Exception as e:
        print(f"[Gist Load] {e}")
        return _gist_cache["data"] or {"comments": [], "next_id": 0, "online": {}}


def _gist_save(data):
    """Save data to GitHub Gist."""
    _gist_cache["data"] = data
    _gist_cache["ts"] = time.time()
    try:
        body = json.dumps({
            "files": {
                GIST_FILE: {
                    "content": json.dumps(data, ensure_ascii=False)
                }
            }
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/gists/{GIST_ID}",
            data=body,
            headers={
                "Authorization": f"token {GIST_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "collab-api",
            },
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
    except Exception as e:
        print(f"[Gist Save] {e}")


# ────────────────────────────────────────────
# /tmp File Storage (fallback)
# ────────────────────────────────────────────
TMP_FILE = "/tmp/collab_data.json"


def _tmp_load():
    try:
        with open(TMP_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"comments": [], "next_id": 0, "online": {}}


def _tmp_save(data):
    with open(TMP_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)


# Unified load/save that prefers Gist
def _load():
    if gist_ok():
        return _gist_load()
    return _tmp_load()


def _save(data):
    if gist_ok():
        _gist_save(data)
    else:
        _tmp_save(data)


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

        data = _load()
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

        data = _load()
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
        _save(data)

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

        data = _load()
        is_admin = user.get("name") == "김도한"
        data["comments"] = [
            c for c in data.get("comments", [])
            if not (c.get("id") == comment_id and (c.get("author") == user.get("name") or is_admin))
        ]
        _save(data)
        self._json(200, {"ok": True})

    # ──────────────────────────────────────────
    # Online Users
    # ──────────────────────────────────────────
    def _get_online(self, params=None):
        access_code = (params or {}).get("access_code", [""])[0]
        user = USERS.get(access_code)
        if user:
            self._touch_online(user)

        data = _load()
        online = data.get("online", {})
        users = [v for v in online.values() if time.time() - v.get("last_seen", 0) < 300]
        self._json(200, {"online": users})

    def _touch_online(self, user):
        info = {"name": user["name"], "org": user["org"], "color": user["color"], "last_seen": time.time()}
        data = _load()
        data.setdefault("online", {})[user["name"]] = info
        _save(data)

    def log_message(self, format, *args):
        pass
