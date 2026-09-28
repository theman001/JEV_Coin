"""모니터링/제어 웹: 표준 라이브러리 HTTP 서버 + HTTP Basic 인증. 의존성 없음.

GET  /             UI (src/static/index.html)
GET  /api/status   Engine.snapshot() JSON
POST /api/start | /api/stop | /api/symbol {"symbol": "SOL/USDT"}
GET  /healthz      인증 없음 (Docker healthcheck용)
"""
import base64
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX = (Path(__file__).parent / "static" / "index.html").read_bytes()


def make_server(engine, host: str, port: int, user: str, password: str) -> ThreadingHTTPServer:
    expected = ("Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if code == 401:
                self.send_header("WWW-Authenticate", 'Basic realm="jev-coin"')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode())

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "").encode("latin-1")
            return hmac.compare_digest(got, expected)

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, b"ok", "text/plain")
            if not self._authed():
                return self._send(401, b"unauthorized", "text/plain")
            if self.path == "/":
                return self._send(200, INDEX, "text/html")
            if self.path == "/favicon.ico":  # 브라우저가 자동 요청 — 404 소음 방지
                return self._send(204, b"", "image/x-icon")
            if self.path == "/api/status":
                return self._json(200, engine.snapshot())
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return self._send(401, b"unauthorized", "text/plain")
            # 브라우저가 Basic 자격증명을 자동 첨부하므로 교차 사이트 폼 POST(CSRF)를 막는다: JSON 콘텐츠 타입만 허용
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
                if self.path == "/api/start":
                    engine.start()
                elif self.path == "/api/stop":
                    engine.stop()
                elif self.path == "/api/symbol":
                    engine.set_symbol(body.get("symbol"))
                else:
                    return self._json(404, {"error": "not found"})
            except (ValueError, AttributeError) as e:  # 잘못된 JSON / 알 수 없는 코인
                return self._json(400, {"error": str(e)})
            self._json(200, engine.snapshot())

        def log_message(self, *args):  # 2초마다 오는 폴링 로그로 컨테이너 로그가 묻히지 않게 끈다
            pass

    return ThreadingHTTPServer((host, port), Handler)
