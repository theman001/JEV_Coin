"""유튜브 인기 단타 패턴을 결정론적 규칙(셋업)으로 — STRATEGY_RESEARCH.md §12.3. 파라미터는 교과서 값으로 고정, 튜닝 금지.

각 함수는 (df, F) → 봉 마감 시점 신호 Series(+1 롱 / −1 숏 / 0 없음). F = features.compute(df, btc) 결과.
신호는 close[t] 까지의 정보만 쓴다 (F 가 인과적이고 여기서는 shift(양수)만 사용).
"""
import numpy as np
import pandas as pd

from .features import ema


def _sig(long_: pd.Series, short_: pd.Series) -> pd.Series:
    return pd.Series(np.where(long_, 1, np.where(short_, -1, 0)), index=long_.index)


def s1_ema_vwap_pullback(df, F):
    """EMA9/21/50 정렬 + VWAP 위(아래) + 직전 봉이 EMA9 아래(위)였다가 이번 봉이 EMA9 재돌파."""
    c, e9 = df["close"], ema(df["close"], 9)
    up, dn = (F["f_ema9_21"] > 0) & (F["f_ema21_50"] > 0), (F["f_ema9_21"] < 0) & (F["f_ema21_50"] < 0)
    return _sig(up & (F["f_vwap_dev"] > 0) & (c.shift(1) <= e9.shift(1)) & (c > e9),
                dn & (F["f_vwap_dev"] < 0) & (c.shift(1) >= e9.shift(1)) & (c < e9))


def s2_squeeze_breakout(df, F):
    """볼린저 스퀴즈 해제 봉 + TTM 모멘텀 방향 + 거래량 급증."""
    fired, vol = F["f_squeeze_fired"] == 1, F["f_vol_ratio"] > 1.2
    return _sig(fired & (F["f_ttm_mom"] > 0) & vol, fired & (F["f_ttm_mom"] < 0) & vol)


def s3_vwap_bb_reversion(df, F):
    """볼린저 밖 + RSI 극단 + VWAP 에서 1ATR 넘게 이격 → 평균회귀."""
    return _sig((F["f_bb_pctb"] < 0) & (F["f_rsi14"] < 30) & (F["f_vwap_dev"] < -1),
                (F["f_bb_pctb"] > 1) & (F["f_rsi14"] > 70) & (F["f_vwap_dev"] > 1))


def s4_connors_rsi2(df, F):
    """Connors RSI(2): 추세(EMA200) 방향으로 RSI2 극단 눌림."""
    return _sig((F["f_rsi2"] < 10) & (F["f_c_ema200"] > 0), (F["f_rsi2"] > 90) & (F["f_c_ema200"] < 0))


def s5_breakout_volume_delta(df, F):
    """직전 20봉 고점(저점) 돌파 + 거래량 급증 + 공격적 매수(매도) 우위."""
    vol = F["f_vol_ratio"] > 1.5
    return _sig((F["f_don20_up"] > 0) & vol & (F["f_delta"] > 0.1), (F["f_don20_dn"] > 0) & vol & (F["f_delta"] < -0.1))


def s6_candle_reversal_at_extreme(df, F):
    """해머/상승 장악형(샛별형 대용)이 20봉 구간 하단에서, 슈팅스타/하락 장악형이 상단에서."""
    return _sig(((F["f_hammer"] == 1) | (F["f_bull_engulf"] == 1)) & (F["f_rangepos20"] < 0.15),
                ((F["f_shooting_star"] == 1) | (F["f_bear_engulf"] == 1)) & (F["f_rangepos20"] > 0.85))


def s7_cvd_divergence(df, F):
    """가격은 안 오르는데 CVD(공격적 매수 누적)는 오르는 강세 다이버전스가 구간 하단에서 (약세는 대칭)."""
    return _sig((F["f_cvd_div"] > 0) & (F["f_rangepos20"] < 0.3), (F["f_cvd_div"] < 0) & (F["f_rangepos20"] > 0.7))


def s8_btc_leads_alt(df, F):
    """BTC 가 알트 ATR 의 1.5배 넘게 움직였는데 알트는 상대적으로 뒤처짐 → 알트 추격 (BTC 자신은 NaN → 신호 없음)."""
    big = 1.5 * F["f_atr_pct"]
    return _sig((F["f_btc_ret3"] > big) & (F["f_rel_strength12"] < 0), (F["f_btc_ret3"] < -big) & (F["f_rel_strength12"] > 0))


def s9_supertrend_flip(df, F):
    """Supertrend 방향 전환."""
    d = F["f_st_dir"]
    return _sig((d == 1) & (d.shift(1) == -1), (d == -1) & (d.shift(1) == 1))


def s10_prev_day_level_break(df, F):
    """전일 고가 상향 돌파(전일 저가 하향 이탈) + 거래량 급증. 직전 봉은 아직 돌파 전."""
    vol = F["f_vol_ratio"] > 1.2
    return _sig((F["f_pdh"] > 0) & (F["f_pdh"].shift(1) <= 0) & vol, (F["f_pdl"] < 0) & (F["f_pdl"].shift(1) >= 0) & vol)


SETUPS = {
    "S1 EMA+VWAP 눌림": s1_ema_vwap_pullback, "S2 스퀴즈 돌파": s2_squeeze_breakout, "S3 VWAP·볼린저 회귀": s3_vwap_bb_reversion,
    "S4 Connors RSI2": s4_connors_rsi2, "S5 돌파+거래량+델타": s5_breakout_volume_delta, "S6 캔들 반전": s6_candle_reversal_at_extreme,
    "S7 CVD 다이버전스": s7_cvd_divergence, "S8 BTC 선행": s8_btc_leads_alt, "S9 Supertrend 전환": s9_supertrend_flip, "S10 전일 레벨 돌파": s10_prev_day_level_break,
}


def events(sig: pd.Series, min_gap: int) -> np.ndarray:
    """이벤트 = 신호가 새로 켜지는(직전 봉과 다른) 봉. 같은 코인에서 min_gap 봉 이상 떨어진 것만 탐욕적으로 채택(전방 수익률 구간이 겹치지 않게)."""
    s = sig.to_numpy()
    idx = np.flatnonzero((s != 0) & (s != np.r_[0, s[:-1]]))
    kept, last = [], -10**9
    for i in idx:
        if i - last >= min_gap:
            kept.append(i)
            last = i
    return np.array(kept, dtype=int)


# ───────────── 미시구조 규칙 (STRATEGY_RESEARCH.md §13.3) — 펀딩·OI·프리미엄·주문서 ─────────────
def m1_funding_extreme(df, F):
    """펀딩 z 극단의 역추세 (한쪽 쏠림이 과열)."""
    return _sig(F["f_funding_z"] < -2, F["f_funding_z"] > 2)


def m2_overheated_leverage(df, F):
    """OI 급증 + 가격 방향 + 펀딩 쏠림: 롱 과열(상승·펀딩 고)이면 숏, 숏 과열(하락·펀딩 저)이면 롱."""
    surge = F["f_oi_z"] > 2
    return _sig(surge & (F["f_body"] < 0) & (F["f_funding_z"] < -1), surge & (F["f_body"] > 0) & (F["f_funding_z"] > 1))


def m3_book_imbalance(df, F):
    """±0.2% 와 ±1% 주문서가 같은 쪽으로 쏠림 → 그 방향."""
    return _sig((F["f_bd_imb02"] > 0.3) & (F["f_bd_imb1"] > 0.15), (F["f_bd_imb02"] < -0.3) & (F["f_bd_imb1"] < -0.15))


def m4_premium_extreme(df, F):
    """프리미엄(베이시스) z 극단 회귀: 선물이 싸면 롱, 비싸면 숏."""
    return _sig(F["f_premium_z"] < -2, F["f_premium_z"] > 2)


def m5_top_trader_follow(df, F):
    """상위 트레이더 롱숏비가 12봉 사이 크게 늘면 롱 (추종)."""
    return _sig(F["f_top_ls_chg12"] > 0.10, F["f_top_ls_chg12"] < -0.10)


def m6_taker_flow_book(df, F):
    """선물 테이커 매수 우위(ln 롱숏비 > 0.5) + 주문서 매수 우위."""
    return _sig((F["f_taker_fut"] > 0.5) & (F["f_bd_imb02"] > 0), (F["f_taker_fut"] < -0.5) & (F["f_bd_imb02"] < 0))


def m7_funding_settlement(df, F):
    """펀딩 정산 직전 30분: 펀딩이 높으면 숏(롱이 정산금을 내기 전 이탈), 음수면 롱."""
    phase = (np.arctan2(F["f_funding_sin"], F["f_funding_cos"]) / (2 * np.pi)) % 1.0
    pre = (F["f_funding_near"] == 1) & (phase > 0.5)
    return _sig(pre & (F["f_funding"] < 0), pre & (F["f_funding"] > 3))


def m8_thin_book_squeeze(df, F):
    """스퀴즈 해제 + 주문서가 평소보다 얇음 → 모멘텀 방향."""
    go = (F["f_squeeze_fired"] == 1) & (F["f_bd_depth1_rel"] < -0.2)
    return _sig(go & (F["f_ttm_mom"] > 0), go & (F["f_ttm_mom"] < 0))


def m9_oi_flush_reversal(df, F):
    """OI 급감(청산·투매) 봉: 하락이면 롱(바닥 확인), 상승이면 숏(숏스퀴즈 소진)."""
    flush = F["f_oi_z"] < -2
    return _sig(flush & (F["f_body"] < 0), flush & (F["f_body"] > 0))


MICRO_SETUPS = {
    "M1 펀딩 극단": m1_funding_extreme, "M2 과열 레버리지": m2_overheated_leverage, "M3 주문서 불균형": m3_book_imbalance,
    "M4 프리미엄 극단": m4_premium_extreme, "M5 상위 트레이더": m5_top_trader_follow, "M6 테이커+호가": m6_taker_flow_book,
    "M7 펀딩 정산 직전": m7_funding_settlement, "M8 얇은 호가 스퀴즈": m8_thin_book_squeeze, "M9 OI 급감 반전": m9_oi_flush_reversal,
}
