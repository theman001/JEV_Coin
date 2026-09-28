"""규칙·백테스트 하네스 검증. 네트워크·API 키 불필요."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (apply_filter, entries, label, reality_check, simulate, spearman_perm, state_frame, stats, trades)
from src.strategies import RULES


def walk(n=400, seed=0, vol=0.01):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, vol, n)))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": np.arange(n) * 3_600_000, "open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close, "volume": r.uniform(1, 2, n)})


# ── 룩어헤드: t 이후 데이터를 잘라내도 t 시점 신호가 같아야 한다 ──
@pytest.mark.parametrize("name", list(RULES))
def test_rules_have_no_lookahead(name):
    """미래 정보는 주로 '신호가 바뀌는 순간'에만 티가 난다 → 전환 시점을 전수(최대 60개) + 무작위 시점을 검사."""
    df, fn = walk(1500, seed=3), RULES[name]
    full = fn(df).to_numpy()
    changes = np.flatnonzero(np.diff(full) != 0) + 1
    assert len(changes) >= 4, "검사할 전환 시점이 너무 적음 (테스트 데이터 문제)"
    ts = set(changes[:60].tolist()) | set(np.random.default_rng(1).choice(np.arange(120, 1500), 15, replace=False).tolist())
    for t in sorted(t for t in ts if t >= 120):
        assert fn(df.iloc[: t + 1].reset_index(drop=True)).iloc[-1] == full[t], f"{name} 가 t={t} 이후 정보를 사용"


def test_rules_direction_sanity():
    up = walk(300, seed=5, vol=0.002).assign(close=lambda d: 100 * 1.01 ** np.arange(300))
    up["open"], up["high"], up["low"] = up.close.shift(1).fillna(100), up.close * 1.001, up.close * 0.999
    dn = up.assign(close=lambda d: 300 * 0.99 ** np.arange(300))
    dn["open"], dn["high"], dn["low"] = dn.close.shift(1).fillna(300), dn.close * 1.001, dn.close * 0.999
    for name, fn in RULES.items():
        assert fn(up).iloc[-1] == 1, f"{name}: 지속 상승장에서 롱이어야 함"
        assert fn(dn).iloc[-1] == 0, f"{name}: 지속 하락장에서 현금이어야 함"


# ── 시뮬레이션: 체결 시점·비용 ──
def test_simulate_uses_next_bar_not_same_bar():
    """교대로 +1%/−1% 인 시가에서, '지금 봉이 오른다는 걸 아는' 신호(치팅)가 이득을 보면 안 된다 (다음 봉에 적용되므로 오히려 손실)."""
    o = np.tile([100.0, 101.0], 100)
    df = pd.DataFrame({"ts": np.arange(200), "open": o, "high": o, "low": o, "close": o, "volume": 1.0})
    oo = np.r_[o[1:] / o[:-1] - 1, np.nan]
    cheat = (oo > 0).astype(float)  # pos[t]=1 iff 봉 t 의 시가→다음 시가가 오름 (미래 정보)
    r = simulate(df, np.nan_to_num(cheat), cost=0.0)
    assert (1 + r["ret"]).prod() < 1


def test_simulate_cost_accounting():
    n = 12
    df = pd.DataFrame({"ts": np.arange(n), "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0})
    pos = np.zeros(n)
    pos[2:4] = 1  # 진입 1회 + 청산 1회
    r = simulate(df, pos, cost=0.001)
    assert r["ret"].sum() == pytest.approx(-0.002)
    assert r["pos"].tolist()[:6] == [0, 0, 0, 1, 1, 0]  # 신호 t=2 → 체결 t=3 (다음 봉)


def test_buy_and_hold_pays_entry_and_final_exit_only():
    n = 50
    o = 100 * 1.01 ** np.arange(n)
    df = pd.DataFrame({"ts": np.arange(n), "open": o, "high": o, "low": o, "close": o, "volume": 1.0})
    r = simulate(df, np.ones(n), cost=0.002)
    assert r["ret"].iloc[0] == 0 and r["ret"].iloc[1] == pytest.approx(0.01 - 0.002) and r["ret"].iloc[-1] == pytest.approx(0.01 - 0.002)
    assert np.allclose(r["ret"].iloc[2:-1], 0.01)
    assert stats(r, 365)["연진입"] > 0


def test_entries_trades_and_filter():
    pos = np.array([0, 1, 1, 0, 0, 1, 0])
    assert entries(pos).tolist() == [1, 5]
    o = np.array([100, 101, 102, 103, 104, 105, 106.0])
    df = pd.DataFrame({"ts": np.arange(7), "open": o, "high": o, "low": o, "close": o, "volume": 1.0})
    tr = dict(trades(df, pos, cost=0.0))
    assert tr[1] == pytest.approx(o[4] / o[2] - 1)  # 신호 t=1 → 체결 t=2 시가, 첫 청산 신호 u=3 → 체결 u+1=4 시가
    assert tr[5] == pytest.approx(0.0)  # 데이터 끝이라 진입과 같은 봉에서 마감
    assert apply_filter(pos, {1: False, 5: True}).tolist() == [0, 0, 0, 0, 0, 1, 0]  # 차단된 포지션은 다음 청산 신호까지 통째로 건너뜀
    assert apply_filter(pos, {}).tolist() == pos.tolist()  # 판정 없음 = 필터 미적용


# ── 통계 검정: 오탐이 낮고 실제 우위는 잡아야 한다 ──
def test_reality_check_false_positive_rate_and_power():
    ps = [reality_check(np.random.default_rng(s).normal(0, 1, (1500, 10)), B=300, seed=s) for s in range(10)]
    assert sum(p < 0.05 for p in ps) <= 2, f"순수 노이즈에서 유의 판정이 너무 많음: {ps}"
    D = np.random.default_rng(0).normal(0, 1, (1500, 10))
    D[:, 3] += 0.25  # 실제 우위 (t ≈ 9.7)
    assert reality_check(D, B=300) < 0.01


def test_spearman_perm():
    x = np.random.default_rng(0).normal(size=200)
    assert spearman_perm(x, x * 2 + 1, B=500)[1] < 0.01
    y = np.random.default_rng(1).normal(size=200)
    assert spearman_perm(x, y, B=500)[1] > 0.01


# ── Jev 상태: 마스킹·인과성 ──
def test_jev_state_is_masked_and_causal():
    df = walk(400, seed=9)
    sf = state_frame(df)
    assert label(sf.iloc[10], "SMA", "4h") is None  # 워밍업 구간은 필터 미적용
    st = label(sf.iloc[300], "채널돌파 20/10", "4h")
    blob = " ".join(f"{k} {v}" for k, v in st.items())
    assert not any(w in blob for w in ("USDT", "BTC", "ETH", "SOL", "2021", "2022", "2023", "2024", "2025"))  # 코인명·날짜 없음
    assert st == label(state_frame(df.iloc[:301].reset_index(drop=True)).iloc[300], "채널돌파 20/10", "4h")  # 미래를 잘라도 같은 상태
