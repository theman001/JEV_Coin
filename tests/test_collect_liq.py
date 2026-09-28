"""청산 수집기: 원시 메시지 파싱·일자 분할·연결이 끊겨도 재연결 (로컬 웹소켓 서버, 외부 네트워크 없음)."""
import asyncio
import json

from aiohttp import web

import src.collect_liq as cl

REAL_SLEEP = asyncio.sleep  # 테스트가 asyncio.sleep 을 monkeypatch 하므로 서버 핸들러는 원본을 써야 한다

RAW = {"e": "forceOrder", "E": 1790599940083, "o": {"s": "ONEUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC", "q": "401929", "p": "0.0024236", "ap": "0.0024582", "X": "FILLED", "l": "178252", "z": "401929", "T": 1790599939148}}


def test_rows_parse_and_append_split_by_utc_day(tmp_path):
    (r,) = cl.rows(json.dumps(RAW))
    assert r["t"] == 1790599939148 and r["sym"] == "ONEUSDT" and r["side"] == "sell" and r["price"] == 0.0024582 and r["qty"] == 401929.0
    assert abs(r["usd"] - 0.0024582 * 401929) < 1e-9
    assert cl.rows(json.dumps({"result": None, "id": 1})) == []  # 구독 응답 등 비청산 메시지는 무시
    cl.append(tmp_path, r)
    cl.append(tmp_path, {**r, "t": r["t"] + 86_400_000})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-09-28.jsonl", "2026-09-29.jsonl"]  # T=1790599939 → 2026-09-28 (KST 21시대, UTC 12시대)


def test_run_reconnects_after_server_drop_and_keeps_writing(tmp_path, monkeypatch):
    conns = []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conns.append(1)
        await ws.send_str(json.dumps({**RAW, "o": {**RAW["o"], "T": RAW["o"]["T"] + len(conns)}}))
        if len(conns) == 1:
            await ws.close()  # 첫 연결은 바로 끊는다 → 재연결해야 한다
        else:
            await REAL_SLEEP(1)
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/ws", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        real_sleep = asyncio.sleep
        monkeypatch.setattr(cl.asyncio, "sleep", lambda s: real_sleep(0.05))  # 백오프 대기 단축
        task = asyncio.create_task(cl.run(tmp_path, f"ws://127.0.0.1:{port}/ws"))
        for _ in range(100):
            if len(list(tmp_path.glob("*.jsonl"))) and sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.jsonl")) >= 2:
                break
            await real_sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()

    asyncio.run(go())
    assert len(conns) >= 2 and sum(len(p.read_text().splitlines()) for p in tmp_path.glob("*.jsonl")) >= 2


def test_silent_stream_is_detected_warned_and_reconnected(tmp_path, monkeypatch, caplog):
    """연결은 살아있는데 이벤트가 안 오는 '조용한 무수신' (ccxt 옛 경로에서 실제로 겪음)."""
    conns = []

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conns.append(1)
        await REAL_SLEEP(1)  # 아무것도 안 보낸다
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/ws", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        real_sleep = asyncio.sleep
        monkeypatch.setattr(cl.asyncio, "sleep", lambda s: real_sleep(0.05))
        task = asyncio.create_task(cl.run(tmp_path, f"ws://127.0.0.1:{port}/ws", silence=0.2))
        for _ in range(100):
            if len(conns) >= 2:
                break
            await real_sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()

    with caplog.at_level("WARNING"):
        asyncio.run(go())
    assert len(conns) >= 2 and "may be broken" in caplog.text
