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
from src.engine import MAX_CONSECUTIVE_ERRORS, Engine
from src.policy import COST_PER_SIDE, PROFILES, Signal, decide

DEFAULT = PROFILES["default"]
MAX_POS_FRAC, STOP_LOSS_PCT = DEFAULT.max_pos_frac, DEFAULT.stop_loss_pct
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
        self.prices, self.down = {}, set()  # 코인별 가격 override / 조회 실패 코인

    def fetch_ticker(self, symbol):
        if self.fail or symbol in self.down:
            raise ConnectionError("exchange down")
        return {"last": self.prices.get(symbol, self.price)}

    def parse_timeframe(self, tf):
        return 3600

    def fetch_ohlcv(self, symbol, tf, limit):
        if self.fail or symbol in self.down:
            raise ConnectionError("exchange down")
        return self.rows + [self.rows[-1]]  # 마지막은 '진행 중' 캔들 (버려져야 함)


# --- state ---
def test_state_uptrend_and_min_bars():
    s = build_state(candles(drift=0.5), "BTC/USDT", "1h")
    assert s["trend"] == "up" and s["price_vs_ema50"] == "above"
    assert s["current_position"] == "flat" and s["position_pnl"] == "none"
    with pytest.raises(ValueError):
        build_state(candles(30), "BTC/USDT", "1h")


def test_state_labels_for_a_dead_market_are_low_and_thin_not_normal():
    """가격 불변·거래량 0 (1초봉에서 흔함): 0/0 이 NaN 이 되어 라벨이 조용히 'normal' 이 되면 안 되고, 경고도 없어야 한다."""
    import warnings
    n = 120
    dead = pd.DataFrame({"ts": np.arange(n) * 1000, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st = build_state(dead, "X/USDT", "1s")
        woke = dead.copy()
        woke.loc[n - 1, "volume"] = 5.0  # 죽어 있던 시장에 갑자기 체결이 생김 → x/0 = inf → spike
        spike = build_state(woke, "X/USDT", "1s")
    assert st["volatility"] == "low" and st["volume"] == "thin" and st["volatility_covers_costs"] is False
    assert spike["volume"] == "spike"


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


def test_1s_profile_loosens_confidence_and_reversal_thresholds():
    """0.5 확신도: 기본 정책(min_conf 0.6)은 관망, 1s 정책(0.45)은 진입. 반전위험 0.8: 기본(0.7 초과)은 관망, 1s(0.85 이내)는 진입."""
    mid_conf = Judgment("long", 0.5, 0.1, "t")
    assert decide(mid_conf, 0.0).side == "flat"
    assert decide(mid_conf, 0.0, profile=PROFILES["1s"]).side == "long"
    mid_reversal = Judgment("long", 0.9, 0.8, "t")
    assert decide(mid_reversal, 0.0).side == "flat"
    assert decide(mid_reversal, 0.0, profile=PROFILES["1s"]).side == "long"


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


def test_stale_check_has_a_floor_for_one_second_candles():
    """1초봉: 마지막 닫힌 봉이 1.3초 늦는 건 정상(실측)인데 배수(1.5초)만 쓰면 자주 오탐한다 → 최소 5초."""
    class OneSec(StubExchange):
        def parse_timeframe(self, tf):
            return 1

    def sec_candles(age_ms, n=60):
        ts = int(time.time() * 1000) - age_ms - 1000 - 1000 * np.arange(n)[::-1]  # 마지막 닫힌 봉이 age_ms 전에 마감
        c = 100 + np.zeros(n)
        return pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})

    ok = sec_candles(3_000)  # 3초 전 마감: 1.5초 규칙이면 stale 이지만 정상
    assert len(fetch_closed(OneSec(ok), "X", "1s", limit=len(ok))) == len(ok)
    late = sec_candles(8_000)
    with pytest.raises(StaleData):
        fetch_closed(OneSec(late), "X", "1s", limit=len(late))


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

    def __init__(self, side="long", hook=None, conf=0.9, reversal=0.1):
        self.side, self.hook, self.calls, self.conf, self.reversal = side, hook, 0, conf, reversal

    def __call__(self, state):
        self.calls += 1
        self.hook and self.hook()
        return Judgment(self.side, self.conf, self.reversal, "test")


SYMS = ("X/USDT", "Y/USDT")
STOP_PX = 1 - (STOP_LOSS_PCT + 0.1) / 100  # 손절선을 넘는 하락 배율


def engine(tmp_path, judge=None, ex=None, symbols=SYMS):
    return Engine(ex or StubExchange(candles()), judge or Scripted(), list(symbols), timeframe="1h", interval=0.01, data_dir=tmp_path, equity=1000.0)


def sides(e):
    return [e.slots[s].side for s in SYMS]


def test_engine_one_judgment_per_candle_per_coin_then_code_stop_loss_only_that_coin(tmp_path):
    ex, j = StubExchange(candles()), Scripted()
    e = engine(tmp_path, j, ex)
    e.start()
    e.step()
    assert sides(e) == ["long", "long"] and j.calls == 2 and len(e.curve) == 1
    assert all(b.start_equity == 500.0 for b in e.slots.values())  # 자본 ÷ 코인 수
    e.step()
    assert j.calls == 2  # 같은 캔들 → Jev 재호출 없음
    ex.prices["X/USDT"] = ex.price * STOP_PX
    e.step()
    assert sides(e) == ["flat", "long"]  # 손절은 그 코인만
    assert e.judgments[-1]["kind"] == "stop_loss" and e.judgments[-1]["symbol"] == "X/USDT"
    e.step()
    assert j.calls == 2 and sides(e) == ["flat", "long"]  # 손절 직후 같은 캔들에서 재진입 안 함
    json.dumps(e.snapshot(), allow_nan=False)  # 웹 응답은 엄격한 JSON (NaN 금지)
    assert abs(e.snapshot()["coins"][0]["candle_ts"] - time.time()) < 2 * 3600  # 초 단위 (거래소 ms 아님)


def test_engine_no_reentry_when_stop_and_new_candle_coincide(tmp_path):
    ex, j = StubExchange(candles()), Scripted()
    e = engine(tmp_path, j, ex)
    e.start()
    e.step()
    assert j.calls == 2 and sides(e) == ["long", "long"]
    ex.rows = candles(end_ms=(int(time.time() * 1000) // HOUR_MS) * HOUR_MS).values.tolist()  # 다음 캔들 마감
    ex.prices["X/USDT"] = ex.price * STOP_PX  # 같은 폴링에서 X 만 손절선도 이탈
    e.step()
    assert sides(e) == ["flat", "long"] and j.calls == 3  # X: Jev 를 부르지도 재진입하지도 않음 / Y: 정상 판단
    e.step()
    assert j.calls == 3  # X 의 그 캔들은 소비된 것으로 처리


def test_engine_judgments_run_in_parallel(tmp_path):
    """Jev 가 느려도 다른 코인의 손절 확인이 밀리지 않도록 판단은 동시에 부른다 (순차 호출이면 배리어가 시간초과)."""
    barrier = threading.Barrier(2, timeout=3)
    j = Scripted(hook=barrier.wait)
    e = engine(tmp_path, j)
    e.start()
    e.step()
    assert j.calls == 2 and sides(e) == ["long", "long"] and e.error is None


def test_engine_stop_flattens_everything_and_start_rejudges(tmp_path):
    j = Scripted()
    e = engine(tmp_path, j)
    e.start()
    e.step()
    e.stop()
    assert not e.running and sides(e) == ["flat", "flat"] and {r["kind"] for r in list(e.judgments)[-2:]} == {"manual_flat"}
    e.start()
    e.step()
    assert j.calls == 4 and sides(e) == ["long", "long"]  # 재시작 시 코인마다 마지막 마감 캔들로 바로 판단
    e.start()  # 이미 실행 중이면 no-op (재판단 없음)
    e.step()
    assert j.calls == 4


def test_engine_no_symbol_selection_and_state_survives_restart(tmp_path):
    e = engine(tmp_path)
    e.start()
    e.step()
    with pytest.raises(ValueError):
        e.set_symbol("X/USDT")  # 코인은 항상 전부 동시에 동작
    e2 = engine(tmp_path, symbols=("OTHER/USDT",))  # 재시작 = 같은 data_dir, 설정이 달라도 저장된 포트폴리오가 우선
    assert e2.symbols == list(SYMS) and e2.running and e2.start_equity == 1000.0 and sides(e2) == ["long", "long"]
    assert e2._equity() == pytest.approx(e._equity()) and (tmp_path / "jev_state.json").exists()
    assert len((tmp_path / "judgments.jsonl").read_text().splitlines()) >= 2


def test_engine_discards_result_if_stopped_during_judge(tmp_path):
    e = engine(tmp_path)
    e.judge = Scripted(hook=lambda: e.stop())  # Jev 응답을 기다리는 동안 사용자가 중지
    e.start()
    e.step()
    assert sides(e) == ["flat", "flat"] and not any(r["kind"] == "judgment" for r in e.judgments)


def test_engine_isolates_a_failing_coin_flattens_it_when_blind_and_kills_only_when_all_blind(tmp_path):
    ex = StubExchange(candles())
    e = engine(tmp_path, ex=ex)
    e.start()
    e.step()
    ex.down.add("X/USDT")
    for _ in range(MAX_CONSECUTIVE_ERRORS - 1):
        e.step()
    assert e.running and sides(e) == ["long", "long"] and "X/USDT" in e.error and "Y/USDT" not in e.error  # 아직 임계 전: 표시만
    e.step()
    assert e.running and sides(e) == ["flat", "long"]  # X: 감시 불가라 청산, Y 는 계속
    assert any(r["kind"] == "blind_flat" and r["symbol"] == "X/USDT" for r in e.judgments)
    ex.down.add("Y/USDT")
    for _ in range(MAX_CONSECUTIVE_ERRORS):
        e.step()
    assert not e.running and sides(e) == ["flat", "flat"] and "kill switch" in e.error  # 전 코인이 감시 불가 → 킬스위치


def test_engine_error_counter_resets_when_a_coin_recovers(tmp_path):
    ex = StubExchange(candles())
    e = engine(tmp_path, ex=ex)
    e.start()
    e.step()
    ex.down.add("X/USDT")
    for _ in range(MAX_CONSECUTIVE_ERRORS - 1):
        e.step()
    ex.down.clear()
    e.step()
    assert e.errors["X/USDT"] == 0 and e.error is None
    ex.down.add("X/USDT")
    for _ in range(MAX_CONSECUTIVE_ERRORS - 1):
        e.step()
    assert sides(e) == ["long", "long"]  # 연속 오류가 아니면 청산하지 않는다


def quiet_candles(end_ms=None, n=120):
    """변동폭이 왕복 비용(0.14%)보다 훨씬 작은 캔들 (1초봉의 현실): ATR% ≈ 0.001% → 비용 게이트 차단."""
    end_ms = end_ms or (int(time.time() * 1000) // HOUR_MS - 1) * HOUR_MS
    close = 100 + 0.001 * np.sin(np.arange(n))
    ts = end_ms - HOUR_MS * np.arange(n)[::-1]
    return pd.DataFrame({"ts": ts, "open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close, "volume": 1.0})


def test_gated_coins_still_get_a_jev_judgment_by_default(tmp_path):
    """기본(5분봉 봇)은 게이트에 막혀도 판단을 부르고 기록한다 (보정 분석용 원자료). 진입만 게이트가 막는다."""
    j = Scripted()
    e = engine(tmp_path, j, StubExchange(quiet_candles()))
    e.start()
    e.step()
    assert j.calls == 2 and sides(e) == ["flat", "flat"] and "below trading costs" in e.judgments[-1]["reason"]


def mid_vol_candles(end_ms=None, n=120):
    """ATR% ≈ 0.012% (계산 확인) — 기본 정책(0.14%)은 막지만 1s 정책(0.0014%)은 통과하는 중간 변동폭."""
    end_ms = end_ms or (int(time.time() * 1000) // HOUR_MS - 1) * HOUR_MS
    close = 100 + 0.01 * np.sin(np.arange(n))
    ts = end_ms - HOUR_MS * np.arange(n)[::-1]
    return pd.DataFrame({"ts": ts, "open": close, "high": close + 0.005, "low": close - 0.005, "close": close, "volume": 1.0})


def test_1s_profile_loosens_the_cost_gate_but_1s_profile_still_blocks_a_truly_dead_market(tmp_path):
    """사용자 확정(2026-09-29): 1초봉은 처음부터 끝까지 관망이었다 → policy.PROFILES["1s"] 로 게이트를 크게 완화."""
    mid = StubExchange(mid_vol_candles())
    default_engine = Engine(mid, Scripted(), list(SYMS), timeframe="1h", interval=0.01, data_dir=tmp_path / "d", equity=1000.0)
    oneS_engine = Engine(mid, Scripted(), list(SYMS), timeframe="1s", interval=0.01, data_dir=tmp_path / "s", equity=1000.0)
    assert oneS_engine.profile.name == "1s" and default_engine.profile.name == "default"
    default_engine.start()
    default_engine.step()
    assert sides(default_engine) == ["flat", "flat"] and "below trading costs" in default_engine.judgments[-1]["reason"]
    oneS_engine.start()
    oneS_engine.step()
    assert sides(oneS_engine) == ["long", "long"]  # 같은 시세인데 1s 정책은 진입한다

    dead = Engine(StubExchange(quiet_candles()), Scripted(), list(SYMS), timeframe="1s", interval=0.01, data_dir=tmp_path / "z", equity=1000.0)
    dead.start()
    dead.step()
    assert sides(dead) == ["flat", "flat"] and dead.judge.calls == 2  # 완전 무변동(0.0012% < 0.0014%)은 1s 정책도 막지만, 판단은 여전히 기록


def test_engine_applies_the_1s_profile_confidence_threshold_not_the_default(tmp_path):
    """엔진이 decide() 에 self.profile 을 실제로 넘기는지: 확신도 0.5 는 기본 정책이면 관망, 1s 정책이면 진입."""
    j = Scripted(conf=0.5)
    e = Engine(StubExchange(candles()), j, ["X/USDT"], timeframe="1s", interval=0.01, data_dir=tmp_path, equity=1000.0)
    e.start()
    e.step()
    assert e.slots["X/USDT"].side == "long"


def test_1s_judge_cooldown_limits_calls_but_indicators_keep_updating(tmp_path, monkeypatch):
    """1초봉 쿨다운(30초): 캔들이 새로 마감돼도 쿨다운 안이면 Jev 를 안 부르고, 지표는 계속 갱신되며, 쿨다운이 지나면 그때 판단한다."""
    base = (int(time.time() * 1000) // HOUR_MS) * HOUR_MS
    ex, j = StubExchange(candles(end_ms=base)), Scripted()
    e = Engine(ex, j, ["X/USDT"], timeframe="1s", interval=0.01, data_dir=tmp_path, equity=1000.0)
    assert e.profile.judge_cooldown_s == 30.0
    clock = [1_700_000_000.0]
    monkeypatch.setattr("src.engine.time.time", lambda: clock[0])
    e.start()
    e.step()
    assert j.calls == 1 and e.slots["X/USDT"].side == "long"  # 첫 판단은 즉시 (쿨다운 시작)
    price1 = e.ind["X/USDT"]["price"]

    clock[0] += 5  # 5초 뒤 새 캔들 도착 — 쿨다운(30초) 안
    ex.rows = candles(end_ms=base + 1000, drift=50).values.tolist()
    e.step()
    assert j.calls == 1 and e.ind["X/USDT"]["price"] != price1  # Jev 는 안 불렀지만 지표는 최신 캔들로 갱신
    assert e.last_ts["X/USDT"] == base  # 캔들을 소비 처리하지 않음 (다음 폴링에 다시 시도)

    clock[0] += 26  # 총 31초 경과: 쿨다운을 지남
    e.step()
    assert j.calls == 2 and e.last_ts["X/USDT"] == base + 1000  # 쿨다운이 풀리자 그제서야 판단


def test_engine_survives_dead_flat_market_with_zero_volume(tmp_path):
    """1초봉에선 거래 없는 구간이 흔하다: 가격 불변·거래량 0 이어도 지표가 NaN/0 나눗셈으로 엔진을 죽이면 안 된다."""
    n = 120
    df = pd.DataFrame({"ts": (int(time.time() * 1000) // HOUR_MS - 1) * HOUR_MS - HOUR_MS * np.arange(n)[::-1], "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0})
    e = Engine(StubExchange(df), Scripted(), list(SYMS), timeframe="1h", interval=0.01, data_dir=tmp_path, equity=1000.0)
    e.start()
    e.step()
    assert e.error is None and sides(e) == ["flat", "flat"]
    json.dumps(e.snapshot(), allow_nan=False)


def test_run_forever_is_fixed_rate_so_one_second_candles_are_not_missed(tmp_path):
    """폴링에 걸린 시간을 sleep 에서 빼야 한다 (interval 0.1, step 0.06 → 주기 0.1: 1초에 ~10회. 단순 sleep 이면 0.16 → ~6회)."""
    e = engine(tmp_path)
    e.interval, calls = 0.1, []
    e.step = lambda: (calls.append(1), time.sleep(0.06))
    e.running = True
    threading.Thread(target=e.run_forever, daemon=True).start()
    time.sleep(1.0)
    e.running = False
    assert len(calls) >= 8


def test_engine_writes_disk_only_when_state_changes(tmp_path):
    """폴링마다 저장하지 않는다 (SD/eMMC 마모): 새 캔들 판단·손절·시작/정지에서만."""
    ex = StubExchange(candles())
    e = engine(tmp_path, ex=ex)
    e.start()
    e.step()
    saves = []
    e._save = lambda: saves.append(1)
    for _ in range(4):
        e.step()  # 새 캔들도 손절도 없음
    assert not saves
    ex.prices["X/USDT"] = ex.price * STOP_PX
    e.step()
    assert saves == [1]  # 손절 1회


def test_engine_stale_data_is_skipped_not_counted_as_error(tmp_path):
    old = candles(end_ms=(int(time.time() * 1000) // HOUR_MS - 10) * HOUR_MS)
    e = engine(tmp_path, ex=StubExchange(old))
    e.start()
    for _ in range(MAX_CONSECUTIVE_ERRORS * 2):
        e.step()
    assert e.running and "skip" in e.error and sides(e) == ["flat", "flat"] and not e.errors.get("X/USDT")  # 킬스위치로 가지 않고, 주문도 없다


def test_engine_kill_switch_after_repeated_errors_via_loop(tmp_path):
    e = engine(tmp_path, ex=StubExchange(candles(), fail=True))
    e.start()
    threading.Thread(target=e.run_forever, daemon=True).start()
    for _ in range(300):
        if not e.running:
            break
        time.sleep(0.01)
    assert not e.running and "kill switch" in e.error


def test_portfolio_loss_limit_flattens_all_coins_immediately_and_blocks_reentry(tmp_path):
    ex = StubExchange(candles())
    e = engine(tmp_path, ex=ex)
    e.start()
    e.step()
    assert sides(e) == ["long", "long"]
    e.slots["X/USDT"].cash *= 0.5  # 손실 시뮬레이션: 포트폴리오 자산 -25%
    e.step()  # 새 캔들 없이도 폴링 즉시 (리스크 거부권은 Jev 판단을 기다리지 않는다)
    assert sides(e) == ["flat", "flat"] and {r["kind"] for r in list(e.judgments)[-2:]} == {"risk_flat"}
    ex.rows = candles(end_ms=(int(time.time() * 1000) // HOUR_MS) * HOUR_MS).values.tolist()  # 다음 캔들: Jev 는 롱을 원하지만
    e.step()
    assert sides(e) == ["flat", "flat"] and "risk veto" in e.judgments[-1]["reason"]  # 한도 초과 상태에서는 신규 진입 차단


def test_snapshot_math_and_strict_json(tmp_path):
    e = engine(tmp_path)
    e.start()
    e.step()
    s = e.snapshot()
    json.dumps(s, allow_nan=False)
    assert s["mode"] == "jev" and s["symbols"] == list(SYMS) and s["exposure_pct"] == 100
    assert s["equity"] == pytest.approx(sum(c["equity"] for c in s["coins"])) and s["return_pct"] == pytest.approx((s["equity"] / 1000 - 1) * 100)
    assert s["realized_pnl"] + s["unrealized_pnl"] == pytest.approx(s["equity"] - 1000 + sum(b.entry_cost for b in e.slots.values()))
    assert all(c["judgment"]["signal"] == "long" and c["indicators"]["price"] and c["state"]["trend"] for c in s["coins"])
    assert s["session_loss_pct"] >= 0 and s["loss_limit_pct"] == 5.0


def test_corrupt_state_file_is_kept_and_reported(tmp_path):
    (tmp_path / "jev_state.json").write_text("{not json")
    e = engine(tmp_path)
    assert "unreadable" in e.error and (tmp_path / "jev_state.json").read_text() == "{not json"


def test_legacy_single_coin_state_is_ignored_but_kept(tmp_path):
    (tmp_path / "state.json").write_text('{"symbol": "X/USDT", "running": true, "broker": {}}')
    e = engine(tmp_path)
    assert e.error is None and not e.running and e.start_equity == 1000.0  # 새 멀티코인 계좌로 시작
    assert (tmp_path / "state.json").exists()


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
    assert snap()["running"] is False and snap()["symbols"] == list(SYMS) and [c["symbol"] for c in snap()["coins"]] == list(SYMS)
    assert json.loads(call(web, "/api/start", {})[1])["running"] is True
    assert call(web, "/api/symbol", {"symbol": "Y/USDT"})[0] == 400  # 코인 선택 없음: 전부 동시에
    assert call(web, "/api/symbol", {})[0] == 400
    assert call(web, "/api/start", {}, ctype="text/plain")[0] == 415  # CSRF 방어
    assert call(web, "/api/start", {}, auth=None)[0] == 401
    assert json.loads(call(web, "/api/stop", {})[1])["running"] is False
    assert call(web, "/api/nope", {})[0] == 404
