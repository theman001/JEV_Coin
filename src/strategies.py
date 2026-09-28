"""문헌 기반 차트 규칙 (STRATEGY_RESEARCH.md §4). 수학은 전부 코드가 한다.

계약: 입력은 마감된 캔들 DataFrame(open/high/low/close/volume). 출력은 각 봉 t 의 **마감 시점** 보유 신호 Series(0/1, 롱 전용).
  - close[t] 까지의 정보만 사용한다 (룩어헤드 없음 — tests/test_strategies.py 가 "t 이후를 잘라도 신호 불변"을 검사).
  - 체결은 t+1 봉 이후 (backtest.simulate 가 다음 봉 시가로 처리).
파라미터는 문헌의 통상값으로 고정한다 (튜닝 금지: 튜닝하면 표본 내/외를 분리해야 한다).
"""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator


def hold(enter, exit_) -> pd.Series:
    """진입/청산 조건 → 보유 상태(1/0). 어느 쪽도 아니면 직전 상태 유지 (히스테리시스)."""
    return pd.Series(np.where(enter, 1.0, np.where(exit_, 0.0, np.nan))).ffill().fillna(0.0).astype(int)


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50, band: float = 0.0) -> pd.Series:
    """SMA_fast > SMA_slow·(1+band) 이면 진입, < SMA_slow·(1−band) 이면 청산. 밴드는 휩쏘·비용 완화용."""
    f, s = df["close"].rolling(fast).mean(), df["close"].rolling(slow).mean()
    return hold(f > s * (1 + band), f < s * (1 - band))


def donchian(df: pd.DataFrame, n: int = 20, m: int = 10) -> pd.Series:
    """채널 돌파: 직전 n봉 고점 돌파 시 진입, 직전 m봉 저점 이탈 시 청산 (윈도우는 shift(1) 로 현재 봉 제외)."""
    hi, lo = df["high"].rolling(n).max().shift(1), df["low"].rolling(m).min().shift(1)
    return hold(df["close"] > hi, df["close"] < lo)


def rsi_trend(df: pd.DataFrame, n: int = 14, level: float = 50) -> pd.Series:
    """RSI 를 역추세(70/30)가 아니라 추세 필터로: RSI > 50 이면 롱 허용."""
    return (RSIIndicator(df["close"], n).rsi() > level).astype(int)


def filter_rule(df: pd.DataFrame, x: float = 0.05) -> pd.Series:
    """Alexander 필터: 직전 저점 대비 +x% 이면 진입, 직전 고점 대비 −x% 이면 청산 (상태 기계)."""
    pos, out, peak, trough = 0, [], -np.inf, np.inf
    for c in df["close"].to_numpy():
        peak, trough = max(peak, c), min(trough, c)
        if pos == 0 and c >= trough * (1 + x):
            pos, peak = 1, c
        elif pos == 1 and c <= peak * (1 - x):
            pos, trough = 0, c
        out.append(pos)
    return pd.Series(out)


RULES = {
    "SMA 20/50": sma_cross,
    "SMA 20/50+밴드0.5%": lambda d: sma_cross(d, band=0.005),
    "채널돌파 20/10": donchian,
    "RSI>50": rsi_trend,
    "필터 5%": filter_rule,
}
