"""미시구조 로더·특징 검증: 파서, 5분 집계, 인과성(공개 시각), 손계산."""
import numpy as np
import pandas as pd
import pytest

import src.micro as mi

TF = 900_000
T0 = 1_700_000_000_000 // mi.DAY * mi.DAY  # 일 경계


def iso(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")


# ── 파서 ──
def test_parse_metrics_and_premium():
    raw = pd.DataFrame({"create_time": [iso(T0 + 600_000), iso(T0)], "symbol": "X", "sum_open_interest": [11.0, 10.0], "sum_open_interest_value": [1.1, 1.0],
                        "count_toptrader_long_short_ratio": [1.2, 1.1], "sum_toptrader_long_short_ratio": [2.2, 2.1], "count_long_short_ratio": [0.9, 0.8],
                        "sum_taker_long_short_vol_ratio": [1.5, 0.5]})
    m = mi.parse_metrics(raw)
    assert m["t"].tolist() == [T0, T0 + 600_000] and m["oi"].tolist() == [10.0, 11.0] and m["top_pos_ls"].tolist() == [2.1, 2.2] and m["glob_ls"].tolist() == [0.8, 0.9]
    pr = mi.parse_premium(pd.DataFrame({"open_time": [T0, T0 + mi.M5], "close": [-0.001, 0.002]}))
    assert pr["t"].tolist() == [T0 + mi.M5, T0 + 2 * mi.M5] and pr["v"].tolist() == [-0.001, 0.002]  # 봉 마감 시각 기준


def test_aggregate_depth_buckets_last_snapshot_and_mean_imbalance():
    def snap(t, b02, a02, b1, a1, b5, a5):
        rows = [(-5.0, b5), (-1.0, b1), (-0.2, b02), (0.2, a02), (1.0, a1), (5.0, a5)]
        return [{"timestamp": iso(t), "percentage": p, "depth": 1.0, "notional": v} for p, v in rows]
    rows = snap(T0 + 10_000, 100, 100, 1000, 1000, 5000, 5000) + snap(T0 + 40_000, 300, 100, 1100, 900, 5100, 4900) + snap(T0 + mi.M5 + 5_000, 50, 150, 900, 1100, 4900, 5100)
    a = mi.aggregate_depth(pd.DataFrame(rows))
    assert a["t"].tolist() == [T0 + mi.M5, T0 + 2 * mi.M5]  # 버킷 끝 시각
    first = a.iloc[0]
    assert (first["b02"], first["a02"], first["b1"], first["a1"]) == (300, 100, 1100, 900)  # 버킷의 마지막 스냅샷
    assert first["imb02_mean"] == pytest.approx((0.0 + 0.5) / 2)  # 스냅샷 불균형 (0, (300-100)/400=0.5) 의 평균
    assert a.iloc[1]["imb02_mean"] == pytest.approx(-0.5)


def test_day_list_excludes_today_and_is_utc():
    days = mi.day_list(T0, T0 + 3 * mi.DAY + 5_000)
    assert len(days) == 3 and days[0] == iso(T0)[:10] and days[-1] == iso(T0 + 2 * mi.DAY)[:10]  # 진행 중인 4번째 날은 제외


# ── 합성 프레임 ──
def frames(seed=0, days=60):
    rng = np.random.default_rng(seed)
    n5 = days * 288
    t5 = T0 + np.arange(n5) * mi.M5
    fund_t = T0 + np.arange(days * 3) * mi.H8
    return {
        "funding": pd.DataFrame({"t": fund_t + 1000, "rate": rng.normal(1e-4, 5e-5, len(fund_t))}),
        "premium": pd.DataFrame({"t": t5 + mi.M5, "v": rng.normal(0, 5e-4, n5)}),
        "metrics": pd.DataFrame({"t": t5, "oi": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.002, n5))), "oi_val": 1.0, "top_pos_ls": np.exp(rng.normal(0.5, 0.1, n5)),
                                 "top_acc_ls": 1.0, "glob_ls": np.exp(rng.normal(0, 0.1, n5)), "taker_ls": np.exp(rng.normal(0, 0.4, n5))}),
        "depth": pd.DataFrame({"t": t5 + mi.M5, "b02": rng.uniform(1e7, 2e7, n5), "a02": rng.uniform(1e7, 2e7, n5), "b1": rng.uniform(1e8, 2e8, n5), "a1": rng.uniform(1e8, 2e8, n5),
                               "b5": rng.uniform(5e8, 6e8, n5), "a5": rng.uniform(5e8, 6e8, n5), "imb02_mean": rng.uniform(-0.3, 0.3, n5)}),
    }


def bars(start_day=45, days=8, seed=1):
    rng = np.random.default_rng(seed)
    n = days * 96
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"ts": T0 + start_day * mi.DAY + np.arange(n) * TF, "open": np.r_[100.0, close[:-1]], "close": close})


def truncate(fr, T):
    """시각 T 에 실제로 공개된 행만 남긴다 (metrics 는 5분 지연)."""
    return {"funding": fr["funding"][fr["funding"].t <= T], "premium": fr["premium"][fr["premium"].t <= T],
            "metrics": fr["metrics"][fr["metrics"].t + mi.M5 <= T], "depth": fr["depth"][fr["depth"].t <= T]}


# ── 인과성 ──
def test_micro_features_are_causal():
    fr, df = frames(), bars()
    full = mi.compute_micro(df, fr, TF)
    assert list(full.columns) == mi.MICRO_FEATURES and full.iloc[300:].notna().all().all()
    for i in list(np.random.default_rng(3).choice(np.arange(150, len(df)), 12, replace=False)) + [len(df) - 1]:
        T = int(df["ts"].iloc[i]) + TF
        part = mi.compute_micro(df.iloc[: i + 1].reset_index(drop=True), truncate(fr, T), TF).iloc[-1]
        bad = [c for c in mi.MICRO_FEATURES if not np.isclose(part[c], full[c].iloc[i], rtol=1e-9, atol=1e-12, equal_nan=True)]
        assert not bad, f"bar {i}: 공개 전 데이터를 쓰는 열 {bad}"


def test_metrics_lag_and_staleness():
    fr = frames()
    T = T0 + 22 * mi.DAY + 3 * 3_600_000 + TF  # 봉 마감 시각
    me = fr["metrics"].copy()
    me.loc[me.t == T - mi.M5, "oi"] = 777.0  # T−5분 관측: 5분 지연 후 공개 시각 = T → 사용 가능
    me.loc[me.t == T, "oi"] = 999.0  # T 관측: 공개 시각 T+5분 → 사용 금지
    fr["metrics"] = me
    df = pd.DataFrame({"ts": [T - TF], "open": [100.0], "close": [101.0]})
    got = mi._asof(me, [T], ["oi"], lag=mi.M5)
    assert got["oi"].iloc[0] == 777.0
    # 오래된 관측은 결측 처리: 마지막 metrics 행보다 30분 뒤의 봉
    late = pd.DataFrame({"ts": [int(me.t.iloc[-1]) + 1_800_000 - TF], "open": [100.0], "close": [101.0]})
    stale = mi.compute_micro(late, fr, TF)
    assert stale[["f_oi_chg1", "f_top_ls", "f_taker_fut", "f_premium", "f_premium_chg", "f_bd_imb02", "f_bd_asym1", "f_bd_slope"]].isna().all().all()  # 모든 프레임에서 결측 처리
    assert stale["f_funding"].notna().all()  # 펀딩비는 8시간 유효


# ── 손계산 ──
def const_frames(days=30):
    n5 = days * 288
    t5 = T0 + np.arange(n5) * mi.M5
    fund_t = np.array([T0 + 10 * mi.DAY, T0 + 10 * mi.DAY + mi.H8, T0 + 10 * mi.DAY + 2 * mi.H8])
    return {"funding": pd.DataFrame({"t": fund_t, "rate": [1e-4, 3e-4, 2e-4]}),
            "premium": pd.DataFrame({"t": t5 + mi.M5, "v": np.linspace(0.0, 0.0287, n5)}),
            "metrics": pd.DataFrame({"t": t5, "oi": 1000.0 * np.exp(0.001 * np.arange(n5)), "oi_val": 1.0, "top_pos_ls": 2.0, "top_acc_ls": 1.0, "glob_ls": 0.5, "taker_ls": 1.0}),
            "depth": pd.DataFrame({"t": t5 + mi.M5, "b02": 300.0, "a02": 100.0, "b1": 1000.0, "a1": 500.0, "b5": 4000.0, "a5": 2000.0, "imb02_mean": 0.5})}


def test_handcrafted_feature_values():
    fr = const_frames()
    ts = T0 + 10 * mi.DAY + mi.H8 + 3 * 3_600_000  # 두 번째 정산 3시간 뒤 봉 시작
    df = pd.DataFrame({"ts": [ts - 60 * TF, ts], "open": [100.0, 100.0], "close": [100.0, 102.0]})
    F = mi.compute_micro(df, fr, TF).iloc[-1]
    assert F["f_funding"] == pytest.approx(3.0) and F["f_funding_chg"] == pytest.approx(2.0)  # 3bp, 직전 대비 +2bp
    T = ts + TF
    assert F["f_funding_sin"] == pytest.approx(np.sin(2 * np.pi * ((T % mi.H8) / mi.H8))) and F["f_funding_near"] == 0.0
    assert F["f_bd_imb02"] == pytest.approx(0.5) and F["f_bd_imb1"] == pytest.approx(1 / 3) and F["f_bd_imb5"] == pytest.approx(1 / 3)  # (b−a)/(b+a)
    assert F["f_bd_slope"] == pytest.approx(np.log(6000 / 1500)) and F["f_bd_asym1"] == pytest.approx(np.log(2)) and F["f_bd_imb02_chg"] == pytest.approx(0.0)
    assert F["f_top_ls"] == pytest.approx(np.log(2.0)) and F["f_glob_ls"] == pytest.approx(np.log(0.5)) and F["f_top_glob_div"] == pytest.approx(np.log(4.0))
    assert F["f_taker_fut"] == pytest.approx(0.0)  # ln(1)
    assert F["f_oi_chg1"] == pytest.approx((np.exp(0.001 * 3) - 1) * 100, rel=1e-6)  # 15분 = 5분 × 3 (한 주기 지연 후에도 같은 간격)
    assert F["f_oi_x_ret"] == pytest.approx(F["f_oi_chg1"] * 2.0)  # 종가 102 / 시가 100 → +2%
    assert F["f_premium_chg"] == pytest.approx((0.0287 / (30 * 288 - 1)) * 3 * 1e4, rel=1e-3)


def test_funding_window_flag():
    fr = const_frames()
    settle = T0 + 12 * mi.DAY + mi.H8  # 정산 시각
    near = pd.DataFrame({"ts": [settle - 20 * 60_000 - TF, settle - 3 * 3_600_000 - TF], "open": [1.0, 1.0], "close": [1.0, 1.0]})
    F = mi.compute_micro(near, fr, TF)
    assert F["f_funding_near"].tolist() == [1.0, 0.0]  # 정산 20분 전 봉 마감은 근접, 3시간 전은 아님


def test_aggregate_depth_tolerates_missing_02_band_and_int_columns():
    """2026-01-15 이전 파일은 ±0.2% 밴드가 없고 percentage 가 정수형이다 → 해당 열은 결측이고 나머지는 정상이어야 한다."""
    rows = [{"timestamp": iso(T0 + 10_000), "percentage": p, "depth": 1.0, "notional": v} for p, v in ((-5, 5000), (-1, 1000), (1, 800), (5, 4000))]
    a = mi.aggregate_depth(pd.DataFrame(rows))
    assert a[["b02", "a02", "imb02_mean"]].isna().all().all() and (a["b1"].iloc[0], a["a1"].iloc[0], a["b5"].iloc[0], a["a5"].iloc[0]) == (1000, 800, 5000, 4000)


def test_zero_oi_glitch_row_is_missing_not_inf_or_minus_100pct():
    """실데이터의 OI=0 행: 그 봉과 그 봉을 기준으로 삼는 봉은 결측이어야 하고, ±inf·가짜 -100% 가 새면 안 된다 (스크리닝이 inf 특징을 통째로 조용히 버렸던 버그)."""
    fr, df = frames(), bars()
    T = int(df["ts"].iloc[150]) + TF  # 150번째 봉 마감 시각
    me = fr["metrics"].copy()
    me.loc[me.t + mi.M5 == T, "oi"] = 0.0  # 공개 시각 == T 인 관측이 0
    fr["metrics"] = me
    F = mi.compute_micro(df, fr, TF)
    assert not np.isinf(F.to_numpy(float)).any()
    assert np.isnan(F["f_oi_chg1"].iloc[150]) and np.isnan(F["f_oi_chg1"].iloc[151]) and np.isnan(F["f_oi_chg12"].iloc[150])  # 글리치 봉·다음 봉(기준)·12봉 변화 모두 결측
    assert F["f_oi_chg1"].iloc[149] > -50 and F["f_oi_chg1"].iloc[152] > -50  # 이웃 봉은 정상값 (가짜 -100% 없음)
