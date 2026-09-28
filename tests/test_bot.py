"""네트워크·API 키 없이 동작. Jev SDK는 MockTransport로 실제 요청/응답 경로를 통과시킨다."""
import base64
import json
import threading
import time
import urllib.error
import urllib.request

import httpx2
import numpy as np
import pandas as pd
import pytest
from typesafe_sdk import RetryPolicy, SystemOneResponse, TypeSafeClient

from src.broker import PaperBroker
from src.data import StaleData, fetch_closed
from src.judge import QUESTIONS, SAFE, FakeJudge, JevJudge, Judgment, parse
from src.engine import Engine
from src.policy import COST_PER_SIDE, MAX_POS_FRAC, STOP_LOSS_PCT, Signal, decide
from src.state import build_state
from src.web import make_server

HOUR_MS = 3_600_000


def candles(n=120, drift=0.5, end_ms=None):
    """드리프트가 있는 합성 1h 캔들. end_ms = 마지막 캔들의 open ts."""
    end_ms = end_ms or (int(time.time() * 1000) // HOUR_MS - 1) * HOUR_MS
    close = 100 + drift * np.arange(n) + np.random.default_rng(0).normal(0, 0.3, n)
    ts = end_ms - HOUR_MS * np.arange(n)[::-1]
    return pd.DataFrame({"ts": ts, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0})


class StubExchange:
    def __init__(self, df, price=None, fail=False):
        self.rows = df.values.tolist()
        self.price, self.fail = price or float(df.close.iloc[-1]), fail

    def fetch_ticker(self, symbol):
        if self.fail:
            raise ConnectionError("exchange down")
        return {"last": self.price}

    def parse_timeframe(self, tf):
        return 3600

    def fetch_ohlcv(self, symbol, tf, limit):
        return self.rows + [self.rows[-1]]  # 마지막은 '진행 중' 캔들 (버려져야 함)


# --- state ---
def test_state_uptrend_and_min_bars():
    s = build_state(candles(drift=0.5), "BTC/USDT", "1h")
    assert s["trend"] == "up" and s["price_vs_ema50"] == "above"
    assert s["current_position"] == "flat" and s["position_pnl"] == "none"
    with pytest.raises(ValueError):
        build_state(candles(30), "BTC/USDT", "1h")


def test_state_position_labels_and_cost_gate():
    df = candles()
    assert build_state(df, "X", "1h", ("long", 0.5))["position_pnl"] == "profitable"
    assert build_state(df, "X", "1h", ("short", -0.5))["position_pnl"] == "losing"
    assert build_state(df, "X", "1h", ("long", 0.0))["position_pnl"] == "near breakeven"
    assert build_state(df, "X", "1h")["volatility_covers_costs"] is True  # ATR ~1% >> 왕복 비용 0.14%
    tiny = df.assign(open=100.0, close=100.0, high=100.001, low=99.999)  # 갭도 없고 ATR ~0.002%
    assert build_state(tiny, "X", "1h")["volatility_covers_costs"] is False


# --- policy ---
@pytest.mark.parametrize("j, loss, side, size", [
    (Judgment("long", 0.9, 0.1, "t"), 0.0, "long", MAX_POS_FRAC),
    (Judgment("short", 0.7, 0.1, "t"), 0.0, "short", MAX_POS_FRAC / 2),  # 낮은 confidence → 축소
    (Judgment("long", 0.5, 0.1, "t"), 0.0, "flat", 0),  # confidence 미달
    (Judgment("long", 0.9, 0.9, "t"), 0.0, "flat", 0),  # 반전 위험
    (Judgment("long", 0.9, 0.1, "t"), 0.06, "flat", 0),  # 리스크 거부권이 Jev를 이긴다
    (SAFE, 0.0, "flat", 0),  # Jev 실패 폴백
])
def test_policy(j, loss, side, size):
    sig = decide(j, loss)
    assert (sig.side, sig.size) == (side, size)


def test_policy_cost_gate_blocks_entry():
    sig = decide(Judgment("long", 0.95, 0.0, "t"), 0.0, cost_covered=False)
    assert sig.side == "flat" and "costs" in sig.reason


# --- broker ---
def test_broker_long_short_pnl_and_no_churn():
    b = PaperBroker(10_000)
    b.rebalance(Signal("long", 0.25, ""), 100)
    entry_fee = 2_500 * COST_PER_SIDE
    assert b.qty == pytest.approx(25) and b.cash == pytest.approx(10_000 - entry_fee)
    b.rebalance(Signal("long", 0.5, ""), 110)  # 같은 방향 → 변화 없음
    assert b.qty == pytest.approx(25)
    b.rebalance(Signal("flat", 0, ""), 110)  # +250 - 청산 수수료
    assert b.side == "flat" and b.cash == pytest.approx(10_000 - entry_fee + 250 - 25 * 110 * COST_PER_SIDE)
    b.rebalance(Signal("short", 0.25, ""), 100)
    assert b.equity(90) > b.equity(100)  # 숏은 하락에서 이익
    assert b.session_loss_pct(1e9) > 0.05  # 숏이 급등하면 손실 한도에 걸림


def test_broker_stop_loss_closes_without_judge():
    b = PaperBroker(10_000)
    b.rebalance(Signal("long", 0.25, ""), 100)
    assert b.check_stop(100 * (1 - STOP_LOSS_PCT / 100 * 0.5), STOP_LOSS_PCT) is False  # 손절선 안쪽
    assert b.pnl_pct(99) == pytest.approx(-1.0)
    assert b.check_stop(99, STOP_LOSS_PCT) is True and b.side == "flat"
    b.rebalance(Signal("short", 0.25, ""), 100)
    assert b.check_stop(101, STOP_LOSS_PCT) is True  # 숏은 상승이 손실
    assert PaperBroker().check_stop(1, STOP_LOSS_PCT) is False  # 플랫이면 no-op


# --- judge (실제 SDK 경로) ---
def _resp(side="long", conf=0.9, rev=0.2):
    rest = (1 - conf) / 2
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {
        "side": {"type": "choice", "choice": side, "confidence": conf,
                 "probabilities": {"long": rest, "short": rest, "flat": rest, side: conf}},
        "reversal_risk": {"type": "noul", "noul": rev}}}


def _jev(handler):
    client = TypeSafeClient(api_key="test", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0))
    return JevJudge(client)


def test_parse():
    r = SystemOneResponse.model_validate_json(json.dumps(_resp("short", 0.8, 0.3)))
    assert parse(r) == Judgment("short", 0.8, 0.3, "jev")


def test_jev_request_shape_and_success():
    seen = {}

    def handler(req):
        seen["body"], seen["auth"] = json.loads(req.content), req.headers["authorization"]
        return httpx2.Response(200, json=_resp())

    j = _jev(handler)({"trend": "up"})
    assert j.side == "long" and j.source == "jev"
    assert seen["auth"] == "Bearer test" and seen["body"]["state"] == {"trend": "up"}
    assert {k: v["type"] for k, v in seen["body"]["questions"].items()} == {"side": "choice", "reversal_risk": "noul"}
    assert set(seen["body"]["questions"]) == set(QUESTIONS)


@pytest.mark.parametrize("status", [429, 500])
def test_jev_failure_falls_back_to_flat(status):
    assert _jev(lambda req: httpx2.Response(status, json={"error": "x"}))({"trend": "up"}) == SAFE


def test_jev_garbage_side_falls_back_to_flat():
    assert _jev(lambda req: httpx2.Response(200, json=_resp(side="moon")))({}) == SAFE


# --- data / pipeline ---
def test_fetch_drops_open_candle_and_flags_stale():
    df = candles()
    assert len(fetch_closed(StubExchange(df), "X", "1h", limit=len(df))) == len(df)
    old = candles(end_ms=(int(time.time() * 1000) // HOUR_MS - 10) * HOUR_MS)
    with pytest.raises(StaleData):
        fetch_closed(StubExchange(old), "X", "1h", limit=len(old))
    now_ms = int(time.time() * 1000)
    lag = candles(end_ms=now_ms - int(2.2 * HOUR_MS))  # 마감 후 1.2캔들: 마감 직후 거래소 지연 → 오탐하면 안 됨
    assert len(fetch_closed(StubExchange(lag), "X", "1h", limit=len(lag))) == len(lag)
    late = candles(end_ms=now_ms - int(2.6 * HOUR_MS))  # 1.6캔들: 진짜 stale
    with pytest.raises(StaleData):
        fetch_closed(StubExchange(late), "X", "1h", limit=len(late))


def test_broker_trade_record_is_net_of_costs():
    b = PaperBroker(10_000)
    b.rebalance(Signal("long", 0.25, ""), 100)
    b.rebalance(Signal("flat", 0, ""), 110)
    t = b.trades[-1]
    assert t["pnl"] == pytest.approx(250 - 2_500 * COST_PER_SIDE - 2_750 * COST_PER_SIDE)
    assert b.cash - b.start_equity == pytest.approx(t["pnl"]) and (b.n_trades, b.n_wins) == (1, 1)


# --- engine ---
class Scripted:
    """항상 같은 판단을 내는 테스트용 judge. hook이 있으면 판단 도중 호출한다 (락 밖 실행 경합 재현)."""
    name = "test"

    def __init__(self, side="long", hook=None):
        self.side, self.hook, self.calls = side, hook, 0

    def __call__(self, state):
        self.calls += 1
        self.hook and self.hook()
        return Judgment(self.side, 0.9, 0.1, "test")


def engine(tmp_path, judge=None, ex=None, symbols=("X/USDT", "Y/USDT")):
    return Engine(ex or StubExchange(candles()), judge or Scripted(), list(symbols), timeframe="1h", interval=0.01, data_dir=tmp_path)


def test_engine_one_judgment_per_candle_then_code_stop_loss(tmp_path):
    ex, j = StubExchange(candles()), Scripted()
    e = engine(tmp_path, j, ex)
    e.start()
    e.step()
    assert e.broker.side == "long" and j.calls == 1 and len(e.curve) == 1
    e.step()
    assert j.calls == 1  # 같은 캔들 → Jev 재호출 없음
    ex.price *= 1 - (STOP_LOSS_PCT + 0.1) / 100
    e.step()
    assert e.broker.side == "flat" and e.judgments[-1]["kind"] == "stop_loss"
    e.step()
    assert j.calls == 1 and e.broker.side == "flat"  # 손절 직후 같은 캔들에서 재진입 안 함
    json.dumps(e.snapshot(), allow_nan=False)  # 웹 응답은 엄격한 JSON (NaN 금지)
    assert abs(e.snapshot()["candle_ts"] - time.time()) < 2 * 3600  # 초 단위 (거래소 ms 아님)


def test_engine_no_reentry_when_stop_and_new_candle_coincide(tmp_path):
    ex, j = StubExchange(candles()), Scripted()
    e = engine(tmp_path, j, ex)
    e.start()
    e.step()
    assert j.calls == 1 and e.broker.side == "long"
    ex.rows = candles(end_ms=(int(time.time() * 1000) // HOUR_MS) * HOUR_MS).values.tolist()  # 다음 캔들 마감
    ex.price *= 1 - (STOP_LOSS_PCT + 0.1) / 100  # 같은 폴링에서 손절선도 이탈
    e.step()
    assert e.broker.side == "flat" and j.calls == 1  # 손절한 폴링의 새 캔들에선 Jev를 부르지도, 재진입하지도 않는다
    e.step()
    assert j.calls == 1  # 그 캔들은 소비된 것으로 처리


def test_engine_stop_flattens_and_start_rejudges(tmp_path):
    j = Scripted()
    e = engine(tmp_path, j)
    e.start()
    e.step()
    e.stop()
    assert not e.running and e.broker.side == "flat" and e.judgments[-1]["kind"] == "manual_flat"
    e.start()
    e.step()
    assert j.calls == 2 and e.broker.side == "long"  # 재시작 시 마지막 마감 캔들로 바로 판단
    e.start()  # 이미 실행 중이면 no-op (재판단 없음)
    e.step()
    assert j.calls == 2


def test_engine_set_symbol_flattens_and_state_survives_restart(tmp_path):
    e = engine(tmp_path)
    e.start()
    e.step()
    with pytest.raises(ValueError):
        e.set_symbol("NOPE/USDT")
    e.set_symbol("Y/USDT")
    assert e.symbol == "Y/USDT" and e.broker.side == "flat" and e.broker.n_trades == 1
    e2 = engine(tmp_path)  # 재시작 = 같은 data_dir로 새 엔진
    assert (e2.symbol, e2.running, e2.broker.n_trades) == ("Y/USDT", True, 1)
    assert e2.broker.cash == pytest.approx(e.broker.cash)
    assert len((tmp_path / "judgments.jsonl").read_text().splitlines()) >= 2


def test_engine_discards_result_if_stopped_during_judge(tmp_path):
    e = engine(tmp_path)
    e.judge = Scripted(hook=lambda: e.stop())  # Jev 응답을 기다리는 동안 사용자가 중지
    e.start()
    e.step()
    assert e.broker.side == "flat" and not any(r["kind"] == "judgment" for r in e.judgments)


def test_engine_kill_switch_after_repeated_errors(tmp_path):
    e = engine(tmp_path, ex=StubExchange(candles(), fail=True))
    e.start()
    threading.Thread(target=e.run_forever, daemon=True).start()
    for _ in range(300):
        if not e.running:
            break
        time.sleep(0.01)
    assert not e.running and "kill switch" in e.error


def test_corrupt_state_file_is_kept_and_reported(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    e = engine(tmp_path)
    assert "unreadable" in e.error and (tmp_path / "state.json").read_text() == "{not json"


# --- web ---
@pytest.fixture
def web(tmp_path):
    srv = make_server(engine(tmp_path), "127.0.0.1", 0, "u", "pw")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def call(base, path, body=None, auth="u:pw", ctype="application/json"):
    headers = {"Content-Type": ctype}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_web_auth_and_ui(web):
    assert call(web, "/healthz", auth=None)[0] == 200
    assert call(web, "/", auth=None)[0] == 401 and call(web, "/api/status", auth="u:wrong")[0] == 401
    code, body = call(web, "/")
    assert code == 200 and b"JEV Coin" in body


def test_web_controls(web):
    snap = lambda: json.loads(call(web, "/api/status")[1])
    assert snap()["running"] is False and snap()["symbols"] == ["X/USDT", "Y/USDT"]
    assert json.loads(call(web, "/api/start", {})[1])["running"] is True
    assert json.loads(call(web, "/api/symbol", {"symbol": "Y/USDT"})[1])["symbol"] == "Y/USDT"
    assert call(web, "/api/symbol", {"symbol": "NOPE"})[0] == 400
    assert call(web, "/api/symbol", {})[0] == 400
    assert call(web, "/api/start", {}, ctype="text/plain")[0] == 415  # CSRF 방어
    assert call(web, "/api/start", {}, auth=None)[0] == 401
    assert json.loads(call(web, "/api/stop", {})[1])["running"] is False
    assert call(web, "/api/nope", {})[0] == 404
