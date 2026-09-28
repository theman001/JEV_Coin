"""특징 선별·이벤트 스터디 통계 검증: 검출력·오탐·부호·비겹침·규칙 손계산."""
import numpy as np
import pandas as pd
import pytest

import src.screen as sc
import src.setups as st
from src.features import FEATURES, compute
from tests.test_features import make

DAY = sc.DAY_MS


# ── 통계 도구 ──
def test_bh_reject_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])  # q=0.05, m=5 → 임계 0.01, 0.02, 0.03, 0.04, 0.05
    assert sc.bh_reject(p, 0.05).tolist() == [True, True, False, False, False]
    assert not sc.bh_reject(np.array([0.5, 0.7, 0.9]), 0.1).any()
    assert sc.bh_reject(np.array([0.9, 0.001]), 0.1).tolist() == [False, True]  # 순서 무관


def test_cluster_se_grows_with_within_day_correlation():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(200), 10)
    iid = rng.normal(0, 1, 2000)
    m1, se1, n = sc.cluster_mean(iid, day)
    assert se1 == pytest.approx(1 / np.sqrt(2000), rel=0.25)  # 독립이면 일반 표준오차와 비슷
    clustered = np.repeat(rng.normal(0, 1, 200), 10) + rng.normal(0, 0.1, 2000)  # 하루 안의 값이 거의 같음
    _, se2, _ = sc.cluster_mean(clustered, day)
    assert se2 > 2.5 * se1  # 상관을 무시하면 과대평가할 유의성을 군집 SE 가 바로잡는다
    assert np.isnan(sc.cluster_mean([1.0] * 10, np.arange(10))[0])  # 표본 부족은 판단 안 함


# ── 합성 패널 ──
def fake_panels(n=9000, seed=0, effect=0.0, sign=1, oos_effect=None, cols=("f_roc3",), n_coins=5):
    """730일에 걸친 합성 패널. cols 의 특징이 전방 수익률에 effect 만큼 실제로 영향(IS), oos_effect 는 OOS 구간 효과."""
    rng = np.random.default_rng(seed)
    step = int(730 * DAY / n)
    t0 = 1_700_000_000_000
    ts = t0 + np.arange(n) * step
    oos_effect = effect if oos_effect is None else oos_effect
    is_oos = ts >= t0 + 366 * DAY
    panels = {}
    for c in sc.COINS[:n_coins]:
        F = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        z = F[list(cols)[0]].to_numpy()
        eff = np.where(is_oos, oos_effect, effect)
        y = sign * eff * z + rng.normal(0, 0.3, n)
        panels[c] = {"df": pd.DataFrame({"ts": ts}), "F": F, "fwd": {h: pd.Series(y) for h in sc.HS}, "sig": {}}
    return panels, t0


def test_subsample_is_nonoverlapping_and_respects_halves():
    panels, t0 = fake_panels()
    hv = sc.halves(t0)
    p = panels["BTC/USDT"]
    for h in sc.HS:
        idx = sc.subsample(p, h, hv["IS"])
        assert (np.diff(idx) % h == 0).all() and (np.diff(idx) >= h).all() and idx.min() >= sc.WARM
        assert (p["df"]["ts"].to_numpy()[idx] < t0 + 365 * DAY).all()
        assert (p["df"]["ts"].to_numpy()[sc.subsample(p, h, hv["OOS"])] >= t0 + 366 * DAY).all()


def run_stage1(panels, t0, tf="15m"):
    t1 = sc.stage1(panels, t0, tf)
    conf = sc.finalize(sc.stage1_oos(panels, t0, t1), int(t1["cand"].sum()))
    return t1, conf


def test_stage1_finds_planted_feature_and_no_false_validations():
    panels, t0 = fake_panels(effect=0.6)
    t1, conf = run_stage1(panels, t0)
    v = conf[conf["validated"]]
    assert set(v["feature"]) == {"f_roc3"} and len(v) >= 3  # 세 h 모두에서 심은 특징만 검증됨
    assert (v["ic_is"] > 0).all() and (v["ls_net"] > 0).all()


def test_stage1_noise_yields_no_validated_feature():
    for seed in (1, 2, 3):
        panels, t0 = fake_panels(effect=0.0, seed=seed)
        _, conf = run_stage1(panels, t0)
        assert conf.empty or not conf["validated"].any(), f"seed {seed}: 노이즈가 검증됨"


def test_stage1_rejects_effect_that_disappears_out_of_sample():
    panels, t0 = fake_panels(effect=0.6, oos_effect=0.0, seed=4)
    _, conf = run_stage1(panels, t0)
    assert not conf["validated"].any()  # IS 에서만 있던 효과는 OOS 확인에서 탈락
    assert conf["feature"].eq("f_roc3").any()  # 후보로는 올라왔음


def test_negative_relation_uses_opposite_direction_and_long_only_takes_low_side():
    panels, t0 = fake_panels(effect=0.6, sign=-1, seed=5)
    hv = sc.halves(t0)
    e = sc.extreme_expectancy(panels, "f_roc3", 3, -1, hv["IS"], hv["OOS"])
    assert e["ls"]["gross"] > 0.8 and e["ls"]["net"] > 0 and e["long"]["gross"] > 0.8  # 하위 10% 에서 롱, 상위 10% 에서 숏
    e_wrong = sc.extreme_expectancy(panels, "f_roc3", 3, +1, hv["IS"], hv["OOS"])
    assert e_wrong["ls"]["gross"] < -0.8  # 방향을 거꾸로 잡으면 손실 → 부호 처리가 결과를 좌우함


def test_effective_sample_correction_penalizes_correlated_coins():
    """모든 코인의 전방 수익률이 같은 공통 충격을 따르면 유효 표본이 줄어 같은 IC 라도 p 가 커져야 한다."""
    panels, t0 = fake_panels(effect=0.05, seed=6)
    hv = sc.halves(t0)
    base = sc.ic_rows(panels, 3, hv["IS"], ["f_roc3"]).loc["f_roc3"]
    common = np.random.default_rng(9).normal(0, 1, len(panels["BTC/USDT"]["df"]))
    for p in panels.values():
        p["fwd"][3] = p["fwd"][3] * 0.3 + common  # 공통 성분을 크게
    corr = sc.ic_rows(panels, 3, hv["IS"], ["f_roc3"]).loc["f_roc3"]
    assert corr["n_eff"] < 0.5 * base["n_eff"]


# ── 셋업 규칙 ──
def F_rows(**cols):
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in cols.items()}, index=range(n))


def test_setup_conditions_handcrafted():
    df = pd.DataFrame({"close": [100.0] * 60 + [99.0, 101.0]})
    n = len(df)
    up = np.ones(n)
    F = pd.DataFrame({"f_ema9_21": up, "f_ema21_50": up, "f_vwap_dev": up}, index=range(n))
    s1 = st.s1_ema_vwap_pullback(df, F)
    assert s1.iloc[-1] == 1 and s1.iloc[-2] == 0 and (s1.iloc[:-1] == 0).all()  # 직전 봉 EMA9 아래 → 이번 봉 재돌파에서만
    Fs = F_rows(f_squeeze_fired=[0, 1, 1, 1], f_ttm_mom=[1, 1, -1, 1], f_vol_ratio=[2, 2, 2, 1.0])
    assert st.s2_squeeze_breakout(None, Fs).tolist() == [0, 1, -1, 0]
    F3 = F_rows(f_bb_pctb=[-0.1, 1.2, -0.1, 0.5], f_rsi14=[25, 75, 45, 50], f_vwap_dev=[-1.5, 1.5, -1.5, 0])
    assert st.s3_vwap_bb_reversion(None, F3).tolist() == [1, -1, 0, 0]
    F4 = F_rows(f_rsi2=[5, 95, 5, 50], f_c_ema200=[1, -1, -1, 1])
    assert st.s4_connors_rsi2(None, F4).tolist() == [1, -1, 0, 0]
    F5 = F_rows(f_don20_up=[0.5, -1, -1], f_don20_dn=[-1, 0.5, -1], f_vol_ratio=[2, 2, 2], f_delta=[0.3, -0.3, 0])
    assert st.s5_breakout_volume_delta(None, F5).tolist() == [1, -1, 0]
    F6 = F_rows(f_hammer=[1, 0, 0], f_bull_engulf=[0, 0, 0], f_shooting_star=[0, 1, 0], f_bear_engulf=[0, 0, 0], f_rangepos20=[0.1, 0.9, 0.5])
    assert st.s6_candle_reversal_at_extreme(None, F6).tolist() == [1, -1, 0]
    F7 = F_rows(f_cvd_div=[1, -1, 1], f_rangepos20=[0.2, 0.8, 0.9])
    assert st.s7_cvd_divergence(None, F7).tolist() == [1, -1, 0]
    F8 = F_rows(f_btc_ret3=[2.0, -2.0, np.nan, 2.0], f_atr_pct=[1.0, 1.0, 1.0, 1.0], f_rel_strength12=[-1, 1, -1, 1])
    assert st.s8_btc_leads_alt(None, F8).tolist() == [1, -1, 0, 0]  # BTC 자신(NaN)·앞서가는 알트는 신호 없음
    F9 = F_rows(f_st_dir=[-1, 1, 1, -1])
    assert st.s9_supertrend_flip(None, F9).tolist() == [0, 1, 0, -1]
    F10 = F_rows(f_pdh=[-1, 0.5, 0.6, -1], f_pdl=[1, 1, 1, -0.5], f_vol_ratio=[2, 2, 2, 2])
    assert st.s10_prev_day_level_break(None, F10).tolist() == [0, 1, 0, -1]


def test_setups_run_on_real_feature_frame():
    df = make(1200)
    F = compute(df, make(1200, seed=2))
    for name, fn in st.SETUPS.items():
        sig = fn(df, F)
        assert len(sig) == len(df) and set(np.unique(sig)) <= {-1, 0, 1}, name


def test_events_onset_and_min_gap():
    sig = pd.Series([0, 1, 1, 0, -1, -1, 0, 1])
    assert st.events(sig, 1).tolist() == [1, 4, 7]
    assert st.events(sig, 3).tolist() == [1, 4, 7]
    assert st.events(sig, 4).tolist() == [1, 7]  # 직전 채택과 4봉 이상 떨어진 것만
    assert st.events(pd.Series([1, 1, 1]), 1).tolist() == [0]


def test_stage2_detects_planted_setup_and_ignores_noise():
    panels, t0 = fake_panels(seed=7, n=8000)
    rng = np.random.default_rng(7)
    for c, p in panels.items():
        n = len(p["df"])
        s = pd.Series(np.where(rng.random(n) < 0.03, np.where(rng.random(n) < 0.5, 1, -1), 0))
        y = p["fwd"][3].to_numpy().copy()
        y = np.where(s != 0, s * 0.5, y)  # 셋업 발동 후 방향대로 평균 +0.5% (비용 0.14% 초과)
        p["fwd"] = {h: pd.Series(y + rng.normal(0, 0.1, n)) for h in sc.HS}
        p["sig"] = {"PLANTED": s, "NOISE": pd.Series(np.where(rng.random(n) < 0.03, 1, 0))}
    orig = sc.SETUPS
    try:
        sc.SETUPS = {"PLANTED": None, "NOISE": None}
        T2 = sc.stage2(panels, t0, "15m")
    finally:
        sc.SETUPS = orig
    pl, nz = T2[T2["setup"] == "PLANTED"], T2[T2["setup"] == "NOISE"]
    assert (pl["OOS_ls_net"] > 0.25).all() and (pl["OOS_ls_p"] < 1e-6).all()
    assert (nz["OOS_ls_net"] < 0).all()  # 노이즈는 비용만큼 손실


def test_cost_is_subtracted_in_both_stages():
    """신호가 통계적으로 뚜렷해도 총이익이 왕복 비용보다 작으면 '검증'되면 안 된다 (통계적 예측력 ≠ 수익성)."""
    panels, t0 = fake_panels(effect=0.06, seed=8)  # 상위 10% 평균 ≈ 0.10% < 비용 0.14%
    t1, conf = run_stage1(panels, t0)
    assert conf["ic_confirmed"].any() and conf["ls_gross"].between(0.05, 0.14).all()
    assert (conf["ls_net"] < 0).all() and not conf["validated"].any()
    assert np.allclose(conf["ls_net"], conf["ls_gross"] - sc.COST)
    n = 9000
    for p in panels.values():
        p["sig"] = {"X": pd.Series(np.random.default_rng(1).choice([0, 1], n, p=[0.97, 0.03]))}
    orig = sc.SETUPS
    try:
        sc.SETUPS = {"X": None}
        T2 = sc.stage2(panels, t0, "15m")
    finally:
        sc.SETUPS = orig
    assert np.allclose(T2["OOS_ls_net"], T2["OOS_ls_gross"] - sc.COST) and np.allclose(T2["IS_long_net"], T2["IS_long_gross"] - sc.COST)
