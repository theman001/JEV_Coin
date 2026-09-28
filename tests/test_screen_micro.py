"""미시구조 규칙·병합·플라시보 검증."""
import numpy as np
import pandas as pd
import pytest

import src.screen as sc
import src.setups as st
from src.features import FEATURES
from src.micro import MICRO_FEATURES
from tests.test_features import make
from tests.test_micro import frames
from tests.test_screen import fake_panels


def F_rows(**cols):
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: pd.Series(v, dtype=float) for k, v in cols.items()}, index=range(n))


def test_micro_setup_conditions_handcrafted():
    assert st.m1_funding_extreme(None, F_rows(f_funding_z=[-3, 3, 0])).tolist() == [1, -1, 0]
    F2 = F_rows(f_oi_z=[3, 3, 3, 1], f_body=[-1, 1, 1, -1], f_funding_z=[-2, 2, -2, -2])
    assert st.m2_overheated_leverage(None, F2).tolist() == [1, -1, 0, 0]  # 하락+펀딩 음수 → 롱, 상승+펀딩 양수 → 숏
    assert st.m3_book_imbalance(None, F_rows(f_bd_imb02=[0.4, -0.4, 0.4], f_bd_imb1=[0.2, -0.2, 0.0])).tolist() == [1, -1, 0]  # 두 밴드가 같은 쪽이어야
    assert st.m4_premium_extreme(None, F_rows(f_premium_z=[-3, 3, 0])).tolist() == [1, -1, 0]
    assert st.m5_top_trader_follow(None, F_rows(f_top_ls_chg12=[0.2, -0.2, 0.05])).tolist() == [1, -1, 0]
    assert st.m6_taker_flow_book(None, F_rows(f_taker_fut=[0.6, -0.6, 0.6], f_bd_imb02=[0.1, -0.1, -0.1])).tolist() == [1, -1, 0]
    ph = np.array([0.99, 0.01, 0.99, 0.99])  # 정산 직전(≈1), 직후(≈0)
    F7 = F_rows(f_funding_sin=np.sin(2 * np.pi * ph), f_funding_cos=np.cos(2 * np.pi * ph), f_funding_near=[1, 1, 1, 0], f_funding=[4, 4, -1, 4])
    assert st.m7_funding_settlement(None, F7).tolist() == [-1, 0, 1, 0]  # 직전 + 펀딩 높음 → 숏, 직후는 제외, 음수 → 롱, 정산 근처 아니면 없음
    assert st.m8_thin_book_squeeze(None, F_rows(f_squeeze_fired=[1, 1, 0], f_bd_depth1_rel=[-0.3, -0.3, -0.3], f_ttm_mom=[1, -1, 1])).tolist() == [1, -1, 0]
    assert st.m9_oi_flush_reversal(None, F_rows(f_oi_z=[-3, -3, 0], f_body=[-1, 1, -1])).tolist() == [1, -1, 0]
    assert st.m8_thin_book_squeeze(None, F_rows(f_squeeze_fired=[1], f_bd_depth1_rel=[np.nan], f_ttm_mom=[1])).tolist() == [0]  # 결측이면 신호 없음


def test_build_with_micro_merges_columns_and_horizons(monkeypatch):
    monkeypatch.setattr(sc, "load_klines", lambda c, tf, years, ex: make(3000, seed=sc.COINS.index(c) + 1))
    monkeypatch.setattr(sc.micro, "load_frames", lambda c, start_ms, end_ms=None: frames(days=60))
    panels, t0 = sc.build("15m", 2, None, hs=sc.HS_MICRO, setups=st.MICRO_SETUPS, with_micro=True)
    p = panels["ETH/USDT"]
    assert list(p["F"].columns) == FEATURES + MICRO_FEATURES
    assert set(p["fwd"]) == {1, 3, 12} and set(p["sig"]) == set(st.MICRO_SETUPS)
    assert p["F"]["f_bd_imb02"].iloc[2000:].notna().all() and p["F"]["f_funding"].iloc[500:].notna().all()
    c = p["df"]["close"].to_numpy()
    assert p["fwd"][1].iloc[100] == pytest.approx((c[101] / c[100] - 1) * 100)  # h=1 = 다음 봉 종가
    p0 = sc.build("15m", 2, None)[0]["ETH/USDT"]  # 기본 모드: 특징 87개, 미시구조 열 없음
    assert list(p0["F"].columns) == FEATURES and set(p0["fwd"]) == {3, 6, 12}


def test_placebo_kills_planted_signal_and_restores_returns():
    panels, t0 = fake_panels(effect=0.6, seed=11)
    real = sc.stage1(panels, t0, "15m", ["f_roc3", "f_rsi14", "f_cci20"], sc.HS)
    assert real.groupby("h")["cand"].sum().ge(1).all()  # 심은 신호는 모든 h 에서 후보
    before = {c: {h: p["fwd"][h].copy() for h in sc.HS} for c, p in panels.items()}
    out = sc.placebo(panels, t0, "15m", ["f_roc3", "f_rsi14", "f_cci20"], sc.HS, shifts=(500, 1500, 3000))
    assert max(out) == 0, out  # 대응을 끊으면 후보가 사라진다
    assert all((panels[c]["fwd"][h].equals(before[c][h])) for c in panels for h in sc.HS)  # 원래 값 복원


def test_stage1_and_placebo_survive_empty_features_and_long_shifts():
    """전부 결측인 특징(예: ±0.2% 밴드가 IS 에 없음)과 데이터보다 긴 밀기 폭에서도 죽지 않아야 한다 (실제 실행에서 중단된 원인)."""
    panels, t0 = fake_panels(effect=0.6, seed=12)
    for p in panels.values():
        p["F"]["f_roc6"] = np.nan  # 표본 없는 특징
    feats = ["f_roc3", "f_roc6"]
    t1 = sc.stage1(panels, t0, "15m", feats, sc.HS)
    assert set(t1["feature"]) == {"f_roc3"}  # 결측 특징은 조용히 제외
    for c, p in panels.items():
        p["fwd"] = {h: pd.Series(np.nan, index=p["fwd"][h].index) for h in sc.HS}
    assert sc.stage1(panels, t0, "15m", feats, sc.HS).empty  # 전방 수익률이 전부 결측이어도 빈 표를 돌려준다
    panels, t0 = fake_panels(effect=0.6, seed=13)
    out = sc.placebo(panels, t0, "15m", FEATURES, sc.HS)  # 기본 밀기 폭 = 길이 비율. 특징 87개 전체 → BH-FDR 이 의미 있는 조건 (특징 1개면 p<0.1 기준이라 우연 통과가 잦다)
    assert len(out) == 3 and sum(out) <= 1, out
    with pytest.raises(ValueError):  # 밀기 폭이 데이터보다 크면 조용히 0 을 돌려주지 않는다 (가짜 안심 방지)
        sc.placebo(panels, t0, "15m", FEATURES, sc.HS, shifts=(10**9,))
