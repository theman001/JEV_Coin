"""추세추종 포트폴리오 모의 엔진 (STRATEGY_RESEARCH.md §14). 네트워크 없이: 코인별 캔들 리플레이 거래소로 검증한다."""
import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

import src.data as data
import src.main as main_mod
from src.backtest import simulate
from src.policy import COST_PER_SIDE, Signal
from src.strategies import sma_cross
from src.trend import FAST, SLOW, TrendEngine, sma_signal

FAST_FIXED, SLOW_FIXED = 20, 50  # 사전 고정 (§14.2) — 엔진 상수를 import 해 비교하면 상수가 잘못 바뀌어도 양쪽이 같이 바뀌어 못 잡는다
from src.web import make_server

TF = 4 * 3_600_000


def walk(seed, n=700, start=100.0):
    """추세 국면이 바뀌는 합성 4h 캔들 (SMA 교차가 여러 번 나온다). df.ts[n-1] = 진행 중 캔들, n-2 = 방금 마감된 캔들."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(0.005 * np.sin(np.arange(n) / 22 + seed) + rng.normal(0, 0.008, n)))
    op = np.r_[start, close[:-1]] * np.exp(rng.normal(0, 0.002, n))  # 시가 갭: 체결가를 '직전 종가'로 잘못 쓰면 백테스트와 어긋나게
    ts = (int(time.time() * 1000) // TF) * TF - TF * np.arange(n)[::-1]
    return pd.DataFrame({"ts": ts, "open": op, "high": np.maximum(op, close) * 1.001, "low": np.minimum(op, close) * 0.999, "close": close, "volume": 1.0})


class Replay:
    """k = 마지막으로 마감된 캔들의 위치. 진행 중 캔들(k+1)의 시가가 현재가(= 백테스트의 '다음 봉 시가 체결')."""

    def __init__(self, dfs):
        self.dfs, self.k, self.down, self.px = dfs, len(next(iter(dfs.values()))) - 2, set(), {}

    def fetch_ticker(self, sym):
        if sym in self.down:
            raise ConnectionError(f"{sym} down")
        return {"last": self.px.get(sym, float(self.dfs[sym]["open"].iloc[self.k + 1]))}

    def parse_timeframe(self, tf):
        return TF // 1000

    def fetch_ohlcv(self, sym, tf, limit):
        if sym in self.down:
            raise ConnectionError(f"{sym} down")
        return self.dfs[sym].values.tolist()[max(0, self.k + 2 - limit): self.k + 2]  # 마지막 행 = 진행 중 캔들


def replay(n_coins=2, n=700, **kw):
    syms = [f"C{i}/KRW" for i in range(n_coins)]
    ex = Replay({s: walk(i + 1, n) for i, s in enumerate(syms)})
    return ex, TrendEngine(ex, syms, timeframe="4h", interval=0.01, equity=1000.0, **kw), syms


@pytest.fixture(autouse=True)
def no_stale(monkeypatch):
    monkeypatch.setattr(data, "STALE_TF_MULT", 1e9)  # 과거 캔들을 리플레이하므로 stale 검사 끔 (stale 자체는 test_bot 이 검사)


# ── 신호 ──
def test_window_signal_equals_full_history_backtest_signal():
    """라이브는 최근 200봉만 본다. 백테스트(전체 이력)와 모든 봉에서 같은 신호여야 한다."""
    df = walk(3, 900)
    full = sma_cross(df, FAST_FIXED, SLOW_FIXED).to_numpy()
    assert 0.2 < full[SLOW_FIXED:].mean() < 0.8  # 합성 데이터가 실제로 교차를 만든다 (검사 검출력)
    assert (FAST, SLOW) == (20, 50)
    for k in range(SLOW_FIXED, len(df) - 1, 7):
        w = df.iloc[max(0, k + 1 - 200): k + 1].reset_index(drop=True)
        hold, fast, slow = sma_signal(w)
        assert hold == bool(full[k]) and hold == (fast > slow)


def test_engine_positions_and_final_equity_match_backtest_simulation():
    """§14.3 의 '백테스트와 일치': 캔들 단위로 재생하면 매 봉 포지션이 백테스트 신호와 같고, 최종 자산이 simulate 와 근사하게 같다."""
    ex, e, syms = replay(2, 700)
    k0 = 60
    changes = {s: 0 for s in syms}
    prev = {s: "flat" for s in syms}
    for k in range(k0, len(ex.dfs[syms[0]]) - 1):
        ex.k = k
        e.step()
        for s in syms:
            pos = sma_cross(ex.dfs[s], FAST_FIXED, SLOW_FIXED).to_numpy()
            assert e.slots[s].side == ("long" if pos[k] else "flat"), (s, k)
            changes[s] += e.slots[s].side != prev[s]
            prev[s] = e.slots[s].side
    assert all(c >= 8 for c in changes.values()), changes  # 진입·청산이 충분히 발생 (검출력)
    n = len(ex.dfs[syms[0]])
    for s in syms:  # 마지막 가격에서 청산하고 백테스트(끝에서 청산한다고 보고 비용 반영)와 비교
        e.slots[s].rebalance(Signal("flat", 0.0, "end"), ex.dfs[s]["open"].iloc[n - 1])
        df, pos = ex.dfs[s].iloc[k0:].reset_index(drop=True), sma_cross(ex.dfs[s], FAST_FIXED, SLOW_FIXED).to_numpy()[k0:]
        expect = 500.0 * (1 + simulate(df, pos)["ret"]).prod()
        assert e.slots[s].cash == pytest.approx(expect, rel=2e-3), (s, e.slots[s].cash, expect)  # 비용 차감 방식 차이는 2차 효과


# ── 엔진 동작 ──
def test_starts_in_uptrend_enters_immediately_and_records_lag():
    ex, e, syms = replay(2)
    ex.k = 400
    e.step()
    up = [s for s in syms if sma_signal(ex.dfs[s].iloc[200:401].reset_index(drop=True))[0]]
    assert [s for s in syms if e.slots[s].side == "long"] == up and up  # 이미 상승 중인 코인은 시작 직후 진입 (백테스트의 상태 기반 신호)
    assert all(ev["lag_s"] >= 0 for ev in e.events) and len(e.events) == 2
    snap = e.snapshot()
    assert snap["exposure_pct"] == 100 * len(up) / 2 and snap["start_equity"] == 1000.0 and all(c["start_equity"] == 500.0 for c in snap["coins"])


def test_no_new_candle_means_no_trade_no_event_no_disk_write(tmp_path):
    ex, e, syms = replay(2, data_dir=tmp_path)
    ex.k = 400
    e.step()
    saves, events, n_curve = [], len(e.events), len(e.curve)
    e._save = lambda: saves.append(1)  # 폴링마다 디스크에 쓰지 않는다 (SD/eMMC 마모)
    for _ in range(3):
        e.step()
    assert not saves and len(e.events) == events and len(e.curve) == n_curve
    ex.k = 401
    e.step()
    assert saves == [1] and len(e.events) == events + 2 and len(e.curve) == n_curve + 1  # 새 캔들이면 저장 1회, 곡선 1점
    assert len((tmp_path / "trend.jsonl").read_text().splitlines()) == events + 2


def test_signal_flip_closes_and_pause_keeps_position_then_resume_reconciles():
    ex, e, syms = replay(1, 800)
    s, df = syms[0], ex.dfs[syms[0]]
    pos = sma_cross(df, FAST_FIXED, SLOW_FIXED).to_numpy()
    k_up = next(k for k in range(100, 700) if pos[k] == 1 and pos[k - 1] == 0)  # 골든크로스 봉
    k_dn = next(k for k in range(k_up + 1, 780) if pos[k] == 0)  # 그 뒤 데드크로스 봉
    ex.k = k_up
    e.step()
    assert e.slots[s].side == "long"
    e.stop()
    assert e.slots[s].side == "long" and not e.running  # 일시정지: 포지션 유지 (§14.6)
    ex.k = k_dn
    e.step()
    assert e.slots[s].side == "long"  # 정지 중엔 손대지 않는다
    e.start()
    e.step()
    assert e.slots[s].side == "flat" and e.slots[s].n_trades == 1  # 재개하면 현재 신호로 맞춘다


def test_one_coin_failure_is_isolated_and_no_kill_switch():
    ex, e, syms = replay(2)
    ex.k = 400
    ex.down.add(syms[1])
    for _ in range(8):  # Jev 봇이라면 5회 연속 오류에서 청산·정지했을 횟수
        e.step()
    assert e.running and syms[1] in e.error and syms[0] not in e.error  # 실패한 코인만 표시
    assert syms[0] in {ev["symbol"] for ev in e.events} and syms[1] not in {ev["symbol"] for ev in e.events}
    ex.down.clear()
    e.step()
    assert e.error is None and {ev["symbol"] for ev in e.events} == set(syms)  # 복구되면 그 코인도 처리


def test_stale_or_short_history_skips_the_tick(monkeypatch):
    ex, e, syms = replay(1)
    monkeypatch.setattr(data, "STALE_TF_MULT", 1.5)
    ex.k = 300  # 최신 캔들이 아님 → stale
    e.step()
    assert "skip" in e.error and not e.events and e.slots[syms[0]].side == "flat"
    monkeypatch.setattr(data, "STALE_TF_MULT", 1e9)
    ex.k = SLOW_FIXED - 5  # SMA50 을 못 만드는 이력
    e.step()
    assert "skip" in e.error and not e.events


def test_result_discarded_if_paused_during_network_call():
    ex, e, syms = replay(1)
    ex.k = 400
    real = ex.fetch_ohlcv
    ex.fetch_ohlcv = lambda *a, **k: (e.stop(), real(*a, **k))[1]  # 네트워크 대기 중 사용자가 정지
    e.step()
    assert not e.events and e.slots[syms[0]].side == "flat"  # 락 밖 실행 경합: 돌아온 결과는 버린다


def test_state_survives_restart_without_retrading_and_saved_portfolio_wins(tmp_path):
    ex, e, syms = replay(2, data_dir=tmp_path)
    for k in (400, 420):
        ex.k = k
        e.step()
    eq, trades = e._equity(), sum(b.n_trades for b in e.slots.values())
    e2 = TrendEngine(ex, ["OTHER/KRW"], timeframe="4h", equity=5.0, data_dir=tmp_path)  # 설정이 달라도 저장된 포트폴리오가 우선
    assert e2.symbols == syms and e2.start_equity == 1000.0 and e2._equity() == pytest.approx(eq) and len(e2.curve) == 2
    assert e2.running and e2.last_ts == e.last_ts and e2.start_px == e.start_px
    assert list(e2.events) == list(e.events) and len(e2.events) == 4  # 화면의 캔들 마감 이력이 재시작 후에도 남는다
    e2.step()  # 같은 캔들 → 재진입/중복 거래 없음
    assert sum(b.n_trades for b in e2.slots.values()) == trades and len(e2.events) == 4


def test_corrupt_state_is_kept_and_reported(tmp_path):
    (tmp_path / "trend_state.json").write_text("{not json")
    ex, e, _ = replay(2, data_dir=tmp_path)
    assert "unreadable" in e.error and (tmp_path / "trend_state.json").read_text() == "{not json"


def test_snapshot_math_and_strict_json():
    ex, e, syms = replay(2)
    first = {sy: float(ex.dfs[sy]["open"].iloc[301]) for sy in syms}  # k=300 에서 처음 본 현재가
    for k in range(300, 330):
        ex.k = k
        e.step()
    assert e.start_px == first  # 단순 보유 기준가는 첫 관측가에서 고정 (이후 갱신 금지)
    s = e.snapshot()
    json.dumps(s, allow_nan=False)  # 웹 응답은 엄격한 JSON
    assert s["equity"] == pytest.approx(sum(c["equity"] for c in s["coins"]))
    assert s["return_pct"] == pytest.approx((s["equity"] / 1000 - 1) * 100)
    r = [((1 - COST_PER_SIDE) * c["price"] / e.start_px[c["symbol"]] - 1) for c in s["coins"]]
    assert s["bh_return_pct"] == pytest.approx(100 * sum(r) / 2) and s["mdd_pct"] <= 0 and s["bh_mdd_pct"] <= 0
    assert s["n_trades"] == sum(b.n_trades for b in e.slots.values()) and len(s["curve"]) == len(e.curve)


# ── 웹 / 엔트리 ──
@pytest.fixture
def web():
    ex, e, _ = replay(2)
    srv = make_server(e, "127.0.0.1", 0, "u", "pw")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", e
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Basic " + base64.b64encode(b"u:pw").decode()})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_web_serves_trend_page_and_controls(web):
    base, e = web
    code, body = call(base, "/")
    assert code == 200 and "추세추종".encode() in body and b"JEV Coin" in body
    assert json.loads(call(base, "/api/status")[1])["mode"] == "trend"
    assert call(base, "/api/symbol", {"symbol": "C0/KRW"})[0] == 400  # 고정 포트폴리오: 코인 선택 없음
    assert json.loads(call(base, "/api/stop", {})[1])["running"] is False
    assert json.loads(call(base, "/api/start", {})[1])["running"] is True


@pytest.mark.parametrize("mode,expect", [("trend", "trend"), ("jev", "jev")])
def test_main_once_selects_engine_by_mode(monkeypatch, mode, expect):
    """MODE 분기(엔트리 리팩터링): 두 모드 모두 --once 가 끝까지 돌고, 모드별 기본값이 적용된다."""
    syms = ["A/KRW"] if mode == "trend" else ["A/USDT", "B/USDT"]
    ex = Replay({s: walk(1, 300) for s in syms})
    seen = {}
    monkeypatch.setattr(main_mod, "make_exchange", lambda name: seen.setdefault("exchange", name) and ex)
    monkeypatch.setenv("MODE", mode)
    monkeypatch.setenv("SYMBOLS", ",".join(syms))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TIMEFRAME", raising=False)
    monkeypatch.delenv("EXCHANGE", raising=False)
    monkeypatch.delenv("PAPER_EQUITY", raising=False)
    if mode == "jev":
        ex.parse_timeframe = lambda tf: 300  # 5m
    out = io.StringIO()
    with redirect_stdout(out):
        assert main_mod.main(["--once"]) == 0
    snap = json.loads(out.getvalue())
    assert seen["exchange"] == ("upbit" if mode == "trend" else "binance")
    assert snap["timeframe"] == ("4h" if mode == "trend" else "5m") and snap["start_equity"] == (1_000_000.0 if mode == "trend" else 100.0)
    assert snap.get("mode") == expect


def test_main_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("MODE", "nope")
    with pytest.raises(SystemExit):
        main_mod.main(["--once"])


def test_snapshot_shows_liquidation_collector_status_with_cache(tmp_path):
    """jev-liq 가 같은 볼륨에 쌓는 파일을 읽어 헤더에 표시. UI 가 2초마다 폴링해도 파일은 30초에 한 번만 읽는다."""
    import src.collect_liq as cl
    ex, e, _ = replay(1, data_dir=tmp_path)
    assert e.snapshot()["liq"] is None  # 수집기 폴더 없음 → UI 는 '없음' 경고
    (tmp_path / "liq").mkdir()
    cl.append(tmp_path / "liq", {"t": int(time.time() * 1000) - 5000, "sym": "X", "side": "buy", "price": 1.0, "qty": 1.0, "usd": 1.0})
    assert e.snapshot()["liq"] is None  # 캐시: 30초 안에는 다시 읽지 않는다
    e._liq_at = 0.0
    s = e.snapshot()
    assert s["liq"]["today"] >= 1 and s["liq"]["last_hour"] == 1 and abs(s["liq"]["last_ts"] - (time.time() - 5)) < 3
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize("page", ["index.html", "trend.html"])
def test_page_inline_js_has_no_syntax_errors(page, tmp_path):
    """UI 스크립트 문법 오류는 페이지 전체를 죽인다 (const 중복 선언으로 실제 발생). node 가 없으면 건너뜀."""
    import re
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    html = (Path(__file__).parent.parent / "src" / "static" / page).read_text()
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    f = tmp_path / "page.js"
    f.write_text(js)
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("tf,expect", [(None, "default"), ("5m", "default"), ("15m", "default"), ("1s", "1s")])
def test_main_selects_policy_profile_from_timeframe(monkeypatch, tf, expect):
    """TIMEFRAME 만으로 엔진이 policy.PROFILES 를 자동 선택한다 (JUDGE_GATED 는 쿨다운으로 대체돼 삭제됨)."""
    ex = Replay({s: walk(1, 300) for s in ("A/USDT", "B/USDT")})
    ex.parse_timeframe = lambda t: 300
    monkeypatch.setattr(main_mod, "make_exchange", lambda name: ex)
    monkeypatch.setenv("MODE", "jev")
    monkeypatch.setenv("SYMBOLS", "A/USDT,B/USDT")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TIMEFRAME", raising=False)
    if tf is not None:
        monkeypatch.setenv("TIMEFRAME", tf)
    out = io.StringIO()
    with redirect_stdout(out):
        assert main_mod.main(["--once"]) == 0
    snap = json.loads(out.getvalue())
    assert snap["timeframe"] == (tf or "5m") and snap["profile"] == expect
