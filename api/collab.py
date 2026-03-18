"""
Vercel Serverless Function — Collab Portal API
Handles /api/v1/collab/* routes via rewrite in vercel.json.

Storage priority:
  1. Upstash Redis (persistent, shared) — if UPSTASH_REDIS_REST_URL configured
  2. /tmp JSON file (ephemeral, shared within warm instance) — zero config fallback
"""
from http.server import BaseHTTPRequestHandler
import json, os, time, uuid, fcntl
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# ────────────────────────────────────────────
# Upstash Redis REST API (optional)
# ────────────────────────────────────────────
REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")


def _redis(cmd_args):
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    try:
        data = json.dumps(cmd_args).encode()
        req = urllib.request.Request(
            REDIS_URL, data=data,
            headers={"Authorization": f"Bearer {REDIS_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("result")
    except Exception as e:
        print(f"[Redis] {e}")
        return None


def redis_ok():
    return bool(REDIS_URL and REDIS_TOKEN)


# ────────────────────────────────────────────
# /tmp File Storage (fallback, zero config)
# ────────────────────────────────────────────
TMP_FILE = "/tmp/collab_data.json"


def _tmp_load():
    """Load data from /tmp file with file locking."""
    try:
        with open(TMP_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"comments": [], "next_id": 0, "online": {}}


def _tmp_save(data):
    """Save data to /tmp file with file locking."""
    with open(TMP_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)


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
        """Authenticate user by access_code (stateless) or session."""
        user = USERS.get(access_code)
        if user:
            return user
        if redis_ok() and session_id:
            raw = _redis(["GET", f"session:{session_id}"])
            if raw:
                return json.loads(raw) if isinstance(raw, str) else raw
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
            self._json(200, {"status": "ok", "redis": redis_ok(), "storage": "redis" if redis_ok() else "tmpfile"})
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
            session_id = params.get("session_id", [""])[0]
            access_code = params.get("access_code", [""])[0]
            self._delete_comment(extra, access_code, session_id)
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

        if redis_ok():
            session_data = {**user, "session_id": session_id, "access_code": code}
            _redis(["SET", f"session:{session_id}", json.dumps(session_data), "EX", "86400"])

        # Track online
        self._touch_online(user)
        self._json(200, {"session_id": session_id, "user": user})

    # ──────────────────────────────────────────
    # Get Comments
    # ──────────────────────────────────────────
    def _get_comments(self, params):
        section = params.get("section", [""])[0]
        since_id = int(params.get("since_id", ["0"])[0])

        if redis_ok():
            raw_list = _redis(["LRANGE", "collab:comments", "0", "-1"]) or []
            comments = []
            for raw in raw_list:
                try:
                    c = json.loads(raw) if isinstance(raw, str) else raw
                    comments.append(c)
                except:
                    continue
        else:
            data = _tmp_load()
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

        if not content:
            self._json(400, {"detail": "내용을 입력하세요."})
            return

        user = self._auth(access_code, session_id)
        if not user:
            self._json(401, {"detail": "인증 실패. 다시 로그인하세요."})
            return

        # Generate ID
        if redis_ok():
            comment_id = int(_redis(["INCR", "collab:comment_id"]) or int(time.time() * 1000) % 2147483647)
        else:
            data = _tmp_load()
            data["next_id"] = data.get("next_id", 0) + 1
            comment_id = data["next_id"]

        comment = {
            "id": comment_id,
            "section": section,
            "content": content,
            "parent_id": parent_id,
            "author": user.get("name", ""),
            "org": user.get("org", ""),
            "role": user.get("role", ""),
            "color": user.get("color", "#6b7280"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "timestamp": time.time(),
        }

        if redis_ok():
            _redis(["RPUSH", "collab:comments", json.dumps(comment)])
        else:
            data = _tmp_load()
            data["next_id"] = comment_id
            data.setdefault("comments", []).append(comment)
            _tmp_save(data)

        self._touch_online(user)
        self._json(200, {"comment": comment})

    # ──────────────────────────────────────────
    # Delete Comment
    # ──────────────────────────────────────────
    def _delete_comment(self, comment_id_str, access_code, session_id):
        try:
            comment_id = int(comment_id_str)
        except ValueError:
            self._json(400, {"detail": "Invalid comment ID"})
            return

        user = self._auth(access_code, session_id)
        if not user:
            self._json(401, {"detail": "인증 실패."})
            return

        if redis_ok():
            raw_list = _redis(["LRANGE", "collab:comments", "0", "-1"]) or []
            for raw in raw_list:
                try:
                    c = json.loads(raw) if isinstance(raw, str) else raw
                    if c.get("id") == comment_id:
                        is_admin = user.get("name") == "김도한"
                        if c.get("author") != user.get("name") and not is_admin:
                            self._json(403, {"detail": "본인의 댓글만 삭제할 수 있습니다."})
                            return
                        _redis(["LREM", "collab:comments", "1", raw if isinstance(raw, str) else json.dumps(c)])
                        break
                except:
                    continue
        else:
            data = _tmp_load()
            is_admin = user.get("name") == "김도한"
            data["comments"] = [
                c for c in data.get("comments", [])
                if not (c.get("id") == comment_id and (c.get("author") == user.get("name") or is_admin))
            ]
            _tmp_save(data)

        self._json(200, {"ok": True})

    # ──────────────────────────────────────────
    # Online Users
    # ──────────────────────────────────────────
    def _get_online(self, params=None):
        access_code = (params or {}).get("access_code", [""])[0]
        user = USERS.get(access_code)
        if user:
            self._touch_online(user)

        if redis_ok():
            raw = _redis(["HGETALL", "online_users"]) or []
            users = []
            if isinstance(raw, list):
                for i in range(0, len(raw), 2):
                    try:
                        u = json.loads(raw[i + 1]) if isinstance(raw[i + 1], str) else raw[i + 1]
                        if time.time() - u.get("last_seen", 0) < 300:
                            users.append(u)
                    except:
                        continue
            self._json(200, {"online": users})
        else:
            data = _tmp_load()
            online = data.get("online", {})
            users = [v for v in online.values() if time.time() - v.get("last_seen", 0) < 300]
            self._json(200, {"online": users})

    def _touch_online(self, user):
        """Update user's last_seen timestamp."""
        info = {"name": user["name"], "org": user["org"], "color": user["color"], "last_seen": time.time()}
        if redis_ok():
            _redis(["HSET", "online_users", user["name"], json.dumps(info)])
        else:
            data = _tmp_load()
            data.setdefault("online", {})[user["name"]] = info
            _tmp_save(data)

    def log_message(self, format, *args):
        pass
