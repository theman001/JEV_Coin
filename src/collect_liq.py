"""Binance USDⓈ-M 강제청산 수집기 — 무료 과거 이력이 없어서 웹소켓으로 직접 쌓는다 (STRATEGY_RESEARCH.md §13).

`python -m src.collect_liq` → data/liq/YYYY-MM-DD.jsonl (UTC 일자별, 한 줄 = 청산 1건, 전 심볼). 봇과 무관한 별도 프로세스.
이력은 '수집을 켠 시점부터'만 존재한다 — 몇 달 쌓은 뒤에야 검정할 수 있다.

ccxt.pro 를 안 쓰는 이유: Binance 가 선물 웹소켓 경로를 /ws/ → /market/ws/ 로 옮겼다(2026-09-28 실측: 옛 경로는 aggTrade 조차 0건, 새 경로만 수신).
ccxt 4.5.84 는 옛 경로라 오류 없이 조용히 아무것도 못 받는다 → 원시 웹소켓으로 직접 붙는다. (aiohttp 는 ccxt 의 의존성으로 이미 설치됨)
한계: 거래소가 1초에 심볼당 마지막 청산 1건만 스냅샷으로 보낸다 (연쇄 청산 땐 과소집계).
"""
import asyncio
import datetime as dt
import json
import logging
import os
import pathlib
import time

import aiohttp

URL = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
SILENCE_S = 300  # 전 심볼 청산은 보통 수 초마다 온다. 이보다 오래 조용하면 스트림이 죽은 것 (ccxt 가 옛 경로에서 그랬듯 오류 없이 조용히 멈춘다)


def rows(msg: str) -> list[dict]:
    """원시 메시지 → 저장 행. side: SELL = 롱 포지션이 청산됨, BUY = 숏 포지션이 청산됨 (거래소 원본 그대로)."""
    d = json.loads(msg)
    if d.get("e") != "forceOrder":
        return []
    o = d["o"]
    price, qty = float(o["ap"]), float(o["z"])
    return [{"t": o["T"], "sym": o["s"], "side": o["S"].lower(), "price": price, "qty": qty, "usd": price * qty}]


def append(out_dir: pathlib.Path, r: dict) -> None:
    day = dt.datetime.fromtimestamp(r["t"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    with open(out_dir / f"{day}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def _times(path: pathlib.Path) -> list[int]:
    out = []
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return out
    for ln in lines:
        try:
            out.append(json.loads(ln)["t"])
        except (ValueError, KeyError):
            pass  # 수집기가 쓰는 중인 마지막 줄
    return out


def stats(out_dir: pathlib.Path, now: float | None = None) -> dict | None:
    """수집 현황(웹 표시용): 오늘(UTC) 건수·최근 1시간 건수·마지막 수신 시각(초). 수집기가 한 번도 안 떴으면(폴더 없음) None."""
    now = now or time.time()
    if not out_dir.is_dir():
        return None
    day = lambda k: dt.datetime.fromtimestamp(now - k * 86400, dt.timezone.utc).strftime("%Y-%m-%d")
    today, yesterday = _times(out_dir / f"{day(0)}.jsonl"), _times(out_dir / f"{day(1)}.jsonl")  # 자정 직후에도 '최근 1시간'이 맞게 어제 파일까지
    allt = yesterday + today
    return {"today": len(today), "last_hour": sum(t >= (now - 3600) * 1000 for t in allt), "last_ts": max(allt) / 1000 if allt else None}


async def run(out_dir: pathlib.Path, url: str = URL, silence: float = SILENCE_S) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    delay = 1
    async with aiohttp.ClientSession() as s:
        while True:
            try:
                async with s.ws_connect(url, heartbeat=20) as ws:
                    logging.info("connected")
                    while True:
                        try:
                            m = await asyncio.wait_for(ws.receive(), silence)  # ws.receive(timeout=) 는 시간초과를 예외가 아니라 CLOSED 메시지로 돌려줘 원인을 구분할 수 없다
                        except asyncio.TimeoutError:
                            logging.warning("no liquidation events for %ss: stream may be broken, reconnecting", silence)
                            break
                        if m.type != aiohttp.WSMsgType.TEXT:
                            break  # CLOSE/ERROR → 재연결
                        delay = 1
                        for r in rows(m.data):
                            append(out_dir, r)
                logging.warning("liq stream closed (retry in %ss)", delay)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # 끊김·일시 오류: 루프를 죽이지 않고 재연결
                logging.warning("liq stream error: %s (retry in %ss)", e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run(pathlib.Path(os.getenv("DATA_DIR", "data")) / "liq"))
    except KeyboardInterrupt:
        pass
