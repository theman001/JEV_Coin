"""차트 특징 라이브러리 검증: 인과성(전 열), 독립 구현 대조(ta), 손계산."""
import numpy as np
import pandas as pd
import pytest
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

import src.features as ft

T0 = 1_699_920_000_000  # 2023-11-14 00:00:00 UTC (일 경계)
M15 = 900_000


def make(n=1500, seed=0, step=M15, t0=T0):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.003, n)))
    o = np.r_[100.0, close[:-1]]
    df = pd.DataFrame({"ts": t0 + np.arange(n) * step, "open": o, "high": np.maximum(o, close) * (1 + np.abs(r.normal(0, 0.001, n))),
                       "low": np.minimum(o, close) * (1 - np.abs(r.normal(0, 0.001, n))), "close": close, "volume": r.uniform(50, 150, n)})
    df["taker_buy_base"] = df["volume"] * r.uniform(0.3, 0.7, n)
    df["n_trades"] = (df["volume"] * r.uniform(5, 9, n)).astype(int)
    return df


def btc_of(df, seed=1):
    b = make(len(df), seed=seed, t0=int(df["ts"].iloc[0]))
    return b


# ── 구성·인과성 ──
def test_columns_match_and_no_nan_after_warmup():
    df = make()
    F = ft.compute(df, btc_of(df))
    assert list(F.columns) == ft.FEATURES
    nan = F.iloc[400:].isna().mean()
    assert nan[nan > 0].empty, nan[nan > 0].to_dict()
    assert ft.compute(df).filter(regex="btc|rel_strength").isna().all().all()  # btc 없으면 교차 자산 열은 NaN
    no_flow = ft.compute(df.drop(columns=["taker_buy_base", "n_trades"]))
    assert no_flow[["f_delta", "f_cvd12", "f_cvd_div"]].isna().all().all()


def test_every_feature_is_causal():
    """t 이후 데이터를 잘라내도 t 시점의 모든 특징이 같아야 한다 — 일 경계·프랙탈 확정·VWAP 리셋 시점 포함."""
    df, btc = make(1500), btc_of(make(1500))
    full = ft.compute(df, btc)
    day_starts = np.flatnonzero(df["ts"].to_numpy() % ft.DAY_MS == 0)
    cuts = sorted(set(np.random.default_rng(2).choice(np.arange(300, 1500), 20, replace=False).tolist()) | {int(i) for i in day_starts[3:8]} | {int(i) + 1 for i in day_starts[3:6]})
    for t in cuts:
        part = ft.compute(df.iloc[: t + 1].reset_index(drop=True), btc.iloc[: t + 1].reset_index(drop=True)).iloc[-1]
        both = pd.concat([part, full.iloc[t]], axis=1).T.astype(float)
        bad = [c for c in ft.FEATURES if not np.isclose(both[c].iloc[0], both[c].iloc[1], rtol=1e-7, atol=1e-9, equal_nan=True)]
        assert not bad, f"t={t} 에서 미래 정보를 쓰는 열: {bad}"


# ── 독립 구현(ta) 대조 ──
def test_indicators_match_ta_library():
    df = make(2500)
    c = df["close"]
    assert np.allclose(ft.rsi(c, 14).iloc[300:], RSIIndicator(c, 14).rsi().iloc[300:], atol=1e-6)
    assert np.allclose(ft.atr(df, 14).iloc[600:], AverageTrueRange(df["high"], df["low"], c, 14).average_true_range().iloc[600:], rtol=1e-5)
    m = MACD(c)
    assert np.allclose((ft.ema(c, 12) - ft.ema(c, 26)).iloc[300:], m.macd().iloc[300:], atol=1e-6)
    bb = BollingerBands(c, 20, 2)
    F = ft.compute(df)
    assert np.allclose(F["f_bb_pctb"].iloc[100:], bb.bollinger_pband().iloc[100:], atol=1e-9)


def test_rolling_linreg_end_matches_polyfit():
    y = pd.Series(np.random.default_rng(3).normal(size=200)).cumsum()
    got = ft.rolling_linreg_end(y, 20)
    for t in (30, 90, 199):
        w = y.iloc[t - 19: t + 1].to_numpy()
        slope, icpt = np.polyfit(np.arange(20), w, 1)
        assert got.iloc[t] == pytest.approx(icpt + slope * 19)
    assert got.iloc[:19].isna().all()


# ── 손계산 ──
def tail_with(rows, n=80, seed=5):
    """워밍업용 무작위 캔들 뒤에 손으로 만든 캔들을 붙인다. rows = [(o,h,l,c), ...]"""
    df = make(n, seed=seed)
    base = float(df["close"].iloc[-1])
    extra = pd.DataFrame(rows, columns=["open", "high", "low", "close"]) * base / 10
    extra["ts"] = int(df["ts"].iloc[-1]) + M15 * np.arange(1, len(rows) + 1)
    extra["volume"], extra["taker_buy_base"], extra["n_trades"] = 100.0, 75.0, 700
    return pd.concat([df, extra], ignore_index=True)


def test_candle_patterns_handcrafted():
    cases = {
        "f_bull_engulf": [(10, 10.05, 8.95, 9), (8.9, 10.2, 8.85, 10.1)],
        "f_bear_engulf": [(9, 10.05, 8.95, 10), (10.1, 10.15, 8.8, 8.9)],
        "f_hammer": [(10, 10.1, 9.9, 10.05), (10.0, 10.15, 9.0, 10.1)],
        "f_shooting_star": [(10, 10.1, 9.9, 10.05), (10.1, 11.1, 10.0, 10.0)],
        "f_doji": [(10, 10.1, 9.9, 10.05), (10.0, 10.5, 9.5, 10.02)],
        "f_inside_bar": [(10, 11, 9, 10.5), (10.2, 10.8, 9.5, 10.6)],
    }
    for col, rows in cases.items():
        F = ft.compute(tail_with(rows))
        assert F[col].iloc[-1] == 1.0, col
        others = [c for c in ("f_bull_engulf", "f_bear_engulf") if c != col]
        if col in ("f_bull_engulf", "f_bear_engulf"):
            assert F[others[0]].iloc[-1] == 0.0
    assert ft.compute(tail_with(cases["f_bull_engulf"]))["f_candle_signal"].iloc[-1] >= 1
    assert ft.compute(tail_with(cases["f_bear_engulf"]))["f_candle_signal"].iloc[-1] <= -1


def test_anchored_vwap_and_prior_day_levels():
    df = make(400)  # 15m × 400 = 4.2일
    F, a = ft.compute(df), ft.atr(df, 14)
    day = df["ts"] // ft.DAY_MS
    tp = (df["high"] + df["low"] + df["close"]) / 3
    for t in (150, 200, 300, 399):
        m = (day == day.iloc[t]) & (df.index <= t)
        manual = (tp[m] * df["volume"][m]).sum() / df["volume"][m].sum()
        assert df["close"].iloc[t] - F["f_vwap_dev"].iloc[t] * a.iloc[t] == pytest.approx(manual)
        prev = df[day == day.iloc[t] - 1]
        assert df["close"].iloc[t] - F["f_pdh"].iloc[t] * a.iloc[t] == pytest.approx(prev["high"].max())
        assert df["close"].iloc[t] - F["f_pdl"].iloc[t] * a.iloc[t] == pytest.approx(prev["low"].min())
    first = int(np.flatnonzero(day.to_numpy() == day.iloc[-1])[0])  # 일 시작 봉: VWAP = 그 봉의 대표가격
    assert df["close"].iloc[first] - F["f_vwap_dev"].iloc[first] * a.iloc[first] == pytest.approx(tp.iloc[first])


def test_fractal_swing_is_confirmed_only_after_two_bars():
    df = make(120, seed=7)
    j = 100
    df.loc[j, ["high"]] = df["high"].iloc[j - 5: j + 5].max() * 1.05  # 뚜렷한 고점
    df.loc[j, "open"] = df.loc[j, "close"] = df["close"].iloc[j - 1]
    F, a = ft.compute(df), ft.atr(df, 14)
    at = lambda t: df["close"].iloc[t] - F["f_dist_swing_high"].iloc[t] * a.iloc[t]  # 복원한 '마지막 확정 스윙 고점'
    assert at(j + 1) != pytest.approx(df["high"].iloc[j])  # 우측 2봉이 지나기 전에는 모름
    assert at(j + 2) == pytest.approx(df["high"].iloc[j])


def test_squeeze_on_then_fired():
    df = make(140, seed=9, )
    n0 = 100
    base = float(df["close"].iloc[n0 - 1])
    for i in range(n0, 130):  # 종가는 거의 고정, 봉의 고저 폭은 큼 → 볼린저 폭 << 켈트너 폭
        df.loc[i, ["open", "close"]] = base
        df.loc[i, "high"], df.loc[i, "low"] = base * 1.01, base * 0.99
    F = ft.compute(df)
    assert F["f_squeeze_on"].iloc[125] == 1.0 and F["f_squeeze_len"].iloc[129] > F["f_squeeze_len"].iloc[125] >= 5
    for i in range(130, 140):  # 방향성 급등 → 종가 표준편차 폭증
        df.loc[i, ["open"]] = base * (1 + 0.03 * (i - 130)); df.loc[i, "close"] = base * (1 + 0.03 * (i - 129))
        df.loc[i, "high"], df.loc[i, "low"] = df.loc[i, "close"] * 1.001, df.loc[i, "open"] * 0.999
    F = ft.compute(df)
    fired = np.flatnonzero(F["f_squeeze_fired"].iloc[130:].to_numpy() == 1.0)
    assert len(fired) >= 1 and F["f_squeeze_on"].iloc[139] == 0.0


def test_trend_helpers_on_monotone_series():
    n = 200
    up = pd.DataFrame({"ts": T0 + np.arange(n) * M15, "open": 100 * 1.004 ** np.arange(n), "volume": 1.0})
    up["close"] = up["open"] * 1.004
    up["high"], up["low"] = up["close"] * 1.001, up["open"] * 0.999
    dn = up.copy()
    dn["open"] = 100 * 0.996 ** np.arange(n)
    dn["close"] = dn["open"] * 0.996
    dn["high"], dn["low"] = dn["open"] * 1.001, dn["close"] * 0.999
    fu, fd = ft.compute(up), ft.compute(dn)
    assert fu["f_st_dir"].iloc[-1] == 1 and fd["f_st_dir"].iloc[-1] == -1
    assert fu["f_ha_streak"].iloc[-1] == 10 and fd["f_ha_streak"].iloc[-1] == -10  # cap
    assert fu["f_ema_stack"].iloc[-1] == 1 and fd["f_ema_stack"].iloc[-1] == -1
    assert fu["f_rsi14"].iloc[-1] == 100.0 and fd["f_rsi14"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert fu["f_hh"].iloc[-1] in (0.0, 1.0)  # 단조 상승엔 프랙탈이 거의 없음 — 오류만 없으면 됨


def test_flow_and_time_features_handcrafted():
    df = make(200)
    df["taker_buy_base"] = df["volume"] * 0.75  # 공격적 매수 75% → delta = +0.5
    F = ft.compute(df)
    assert np.allclose(F["f_taker_ratio"].dropna(), 0.75) and np.allclose(F["f_delta"].dropna(), 0.5) and np.allclose(F["f_cvd12"].dropna(), 0.5)
    def last_at(ts_last):  # 마지막 봉의 UTC 시각이 ts_last 가 되도록 40봉을 만든다
        d = make(40, t0=ts_last - 39 * M15)
        return ft.compute(d).iloc[-1]
    f1 = last_at(T0 + 12 * 3_600_000)  # 화요일 12:00 UTC
    assert f1["f_tod_sin"] == pytest.approx(0, abs=1e-9) and f1["f_tod_cos"] == pytest.approx(-1)
    assert (f1["f_is_eu"], f1["f_is_us"], f1["f_is_asia"], f1["f_is_weekend"]) == (1.0, 0.0, 0.0, 0.0)
    f2 = last_at(T0 + 4 * ft.DAY_MS + 15 * 3_600_000)  # 토요일 15:00 UTC
    assert (f2["f_is_weekend"], f2["f_is_us"], f2["f_is_eu"]) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        ft.compute(make(10))


def test_cross_asset_alignment():
    df = make(300)
    btc = btc_of(df).iloc[5:].reset_index(drop=True)  # BTC 는 앞 5봉이 없음 → 해당 시점은 NaN
    F = ft.compute(df, btc)
    b = btc.set_index("ts")["close"]
    t = 100
    assert F["f_btc_ret3"].iloc[t] == pytest.approx((b[df["ts"].iloc[t]] / b[df["ts"].iloc[t - 3]] - 1) * 100)
    assert F["f_rel_strength12"].iloc[t] == pytest.approx((df["close"].iloc[t] / df["close"].iloc[t - 12] - 1) * 100 - F["f_btc_ret12"].iloc[t])
    assert F["f_btc_ret1"].iloc[:5].isna().all()
