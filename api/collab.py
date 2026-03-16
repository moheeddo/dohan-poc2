"""
Vercel Serverless Function — Collab Portal API
Handles /api/v1/collab/* routes via rewrite in vercel.json.
Uses Upstash Redis REST API for persistent shared storage.
"""
from http.server import BaseHTTPRequestHandler
import json, os, time, uuid
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# ────────────────────────────────────────────
# Upstash Redis REST API
# ────────────────────────────────────────────
REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")


def _redis(cmd_args):
    """Execute a single Redis command via Upstash REST API. Returns result or None."""
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    try:
        data = json.dumps(cmd_args).encode()
        req = urllib.request.Request(
            REDIS_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {REDIS_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            return body.get("result")
    except Exception as e:
        print(f"[Redis Error] {e}")
        return None


def _redis_pipeline(commands):
    """Execute multiple Redis commands in a pipeline."""
    if not REDIS_URL or not REDIS_TOKEN:
        return [None] * len(commands)
    try:
        data = json.dumps(commands).encode()
        req = urllib.request.Request(
            f"{REDIS_URL}/pipeline",
            data=data,
            headers={
                "Authorization": f"Bearer {REDIS_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read())
            return [r.get("result") for r in results]
    except Exception as e:
        print(f"[Redis Pipeline Error] {e}")
        return [None] * len(commands)


def redis_ok():
    return bool(REDIS_URL and REDIS_TOKEN)


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

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _route(self):
        """Return (endpoint, extra) from the request path."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        # /api/v1/collab/comments/123 → endpoint='comments', extra='123'
        parts = path.split("/")
        # Find 'collab' in path and get what follows
        try:
            idx = parts.index("collab")
            rest = parts[idx + 1:]
        except (ValueError, IndexError):
            rest = []
        endpoint = rest[0] if rest else ""
        extra = rest[1] if len(rest) > 1 else ""
        params = parse_qs(parsed.query)
        return endpoint, extra, params

    # ── OPTIONS (CORS preflight) ──
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ──
    def do_GET(self):
        endpoint, extra, params = self._route()

        if endpoint == "comments":
            self._get_comments(params)
        elif endpoint == "online":
            self._get_online()
        elif endpoint == "health":
            self._send_json(200, {"status": "ok", "redis": redis_ok()})
        else:
            self._send_json(404, {"detail": "Not found"})

    # ── POST ──
    def do_POST(self):
        endpoint, extra, params = self._route()
        body = self._read_body()

        if endpoint == "login":
            self._login(body)
        elif endpoint == "comments":
            self._post_comment(body)
        else:
            self._send_json(404, {"detail": "Not found"})

    # ── DELETE ──
    def do_DELETE(self):
        endpoint, extra, params = self._route()

        if endpoint == "comments" and extra:
            session_id = params.get("session_id", [""])[0]
            self._delete_comment(extra, session_id)
        else:
            self._send_json(404, {"detail": "Not found"})

    # ────────────────────────────────────────
    # Login
    # ────────────────────────────────────────
    def _login(self, body):
        code = body.get("access_code", "")
        user = USERS.get(code)
        if not user:
            self._send_json(401, {"detail": "유효하지 않은 접속 코드입니다."})
            return

        session_id = str(uuid.uuid4())
        session_data = {**user, "session_id": session_id, "access_code": code, "login_at": time.time()}

        if redis_ok():
            _redis_pipeline([
                ["SET", f"session:{session_id}", json.dumps(session_data), "EX", "86400"],
                ["HSET", "online_users", user["name"], json.dumps({
                    "name": user["name"], "org": user["org"], "color": user["color"],
                    "last_seen": time.time()
                })],
            ])

        self._send_json(200, {"session_id": session_id, "user": user})

    # ────────────────────────────────────────
    # Get Comments
    # ────────────────────────────────────────
    def _get_comments(self, params):
        section = params.get("section", [""])[0]
        since_id = int(params.get("since_id", ["0"])[0])

        if not redis_ok():
            self._send_json(200, {"comments": []})
            return

        raw_list = _redis(["LRANGE", "collab:comments", "0", "-1"])
        if not raw_list:
            self._send_json(200, {"comments": []})
            return

        comments = []
        for raw in raw_list:
            try:
                c = json.loads(raw) if isinstance(raw, str) else raw
                if section and c.get("section") != section:
                    continue
                if since_id and c.get("id", 0) <= since_id:
                    continue
                comments.append(c)
            except:
                continue

        self._send_json(200, {"comments": comments})

    # ────────────────────────────────────────
    # Post Comment
    # ────────────────────────────────────────
    def _post_comment(self, body):
        session_id = body.get("session_id", "")
        access_code = body.get("access_code", "")
        section = body.get("section", "")
        content = body.get("content", "").strip()
        parent_id = body.get("parent_id", "")

        if not content:
            self._send_json(400, {"detail": "내용을 입력하세요."})
            return

        # Validate: try access_code first (stateless), then session (Redis)
        user = USERS.get(access_code)
        if not user and redis_ok():
            raw = _redis(["GET", f"session:{session_id}"])
            if raw:
                user = json.loads(raw) if isinstance(raw, str) else raw
        if not user:
            self._send_json(401, {"detail": "인증 실패. 다시 로그인하세요."})
            return

        if not redis_ok():
            self._send_json(503, {"detail": "서버 스토리지(Redis) 미설정. 관리자에게 문의하세요."})
            return

        # Generate ID
        comment_id = int(time.time() * 1000) % 2147483647  # Unique enough for this use case
        if redis_ok():
            counter = _redis(["INCR", "collab:comment_id"])
            if counter:
                comment_id = int(counter)

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
            # Update last_seen
            _redis(["HSET", "online_users", user["name"], json.dumps({
                "name": user["name"], "org": user["org"], "color": user["color"],
                "last_seen": time.time()
            })])

        self._send_json(200, {"comment": comment})

    # ────────────────────────────────────────
    # Delete Comment
    # ────────────────────────────────────────
    def _delete_comment(self, comment_id_str, session_id):
        try:
            comment_id = int(comment_id_str)
        except ValueError:
            self._send_json(400, {"detail": "Invalid comment ID"})
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        access_code = params.get("access_code", [""])[0]

        user = USERS.get(access_code)
        if not user and redis_ok():
            raw = _redis(["GET", f"session:{session_id}"])
            if raw:
                user = json.loads(raw) if isinstance(raw, str) else raw
        if not user:
            self._send_json(401, {"detail": "인증 실패."})
            return

        if redis_ok():
            raw_list = _redis(["LRANGE", "collab:comments", "0", "-1"])
            for raw in (raw_list or []):
                try:
                    c = json.loads(raw) if isinstance(raw, str) else raw
                    if c.get("id") == comment_id:
                        if c.get("author") != user.get("name"):
                            self._send_json(403, {"detail": "본인의 댓글만 삭제할 수 있습니다."})
                            return
                        _redis(["LREM", "collab:comments", "1", raw if isinstance(raw, str) else json.dumps(c)])
                        break
                except:
                    continue

        self._send_json(200, {"ok": True})

    # ────────────────────────────────────────
    # Online Users
    # ────────────────────────────────────────
    def _get_online(self):
        if not redis_ok():
            self._send_json(200, {"online": []})
            return

        raw = _redis(["HGETALL", "online_users"])
        users = []
        if raw and isinstance(raw, list):
            for i in range(0, len(raw), 2):
                try:
                    u = json.loads(raw[i + 1]) if isinstance(raw[i + 1], str) else raw[i + 1]
                    # Only show users active in last 5 minutes
                    if time.time() - u.get("last_seen", 0) < 300:
                        users.append(u)
                except:
                    continue

        self._send_json(200, {"online": users})

    def log_message(self, format, *args):
        """Suppress default logging to avoid noise in Vercel logs."""
        pass
