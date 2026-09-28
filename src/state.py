"""OHLCV(+현재 포지션) → 지표 / Jev에 줄 압축 State. 산술은 전부 여기(코드)에서 하고, Jev에는 의미 라벨만 준다."""
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from .policy import COST_PER_SIDE, MIN_ATR_COST_MULT

MIN_BARS = 60  # EMA50 + 여유


def _bucket(x: float, lo: float, hi: float, names: tuple[str, str, str]) -> str:
    return names[0] if x < lo else names[2] if x > hi else names[1]


def indicators(df: pd.DataFrame) -> dict:
    """마지막 마감 캔들 기준 수치 지표 (웹 표시 + 라벨 계산용). Jev에는 직접 전달하지 않는다."""
    if len(df) < MIN_BARS:
        raise ValueError(f"need >= {MIN_BARS} bars, got {len(df)}")
    close = df["close"]
    atr_pct = AverageTrueRange(df["high"], df["low"], close, 14).average_true_range() / close * 100
    return {
        "price": close.iloc[-1],
        "ema20": EMAIndicator(close, 20).ema_indicator().iloc[-1],
        "ema50": EMAIndicator(close, 50).ema_indicator().iloc[-1],
        "rsi": RSIIndicator(close, 14).rsi().iloc[-1],
        "atr_pct": atr_pct.iloc[-1],
        "volatility_ratio": atr_pct.iloc[-1] / atr_pct.iloc[-50:].median(),  # 최근 변동성 vs 자기 평소
        "volume_ratio": df["volume"].iloc[-1] / df["volume"].iloc[-21:-1].median(),  # 직전 봉 vs 최근 20봉 중앙값
        "return_3_bars_pct": (close.iloc[-1] / close.iloc[-4] - 1) * 100,
        "return_24_bars_pct": (close.iloc[-1] / close.iloc[-25] - 1) * 100,
    }


def build_state(df: pd.DataFrame, symbol: str, timeframe: str, position: tuple[str, float] = ("flat", 0.0)) -> dict:
    """position = (side, 진입가 대비 미실현 손익 %). Jev가 보유/전환/청산을 고를 수 있게 라벨로 넘긴다."""
    i = indicators(df)
    price, side, pnl_pct = i["price"], position[0], position[1]
    return {
        "market": f"{symbol} {timeframe} candles, last closed bar",
        "trend": "up" if price > i["ema20"] > i["ema50"] else "down" if price < i["ema20"] < i["ema50"] else "sideways",
        "price_vs_ema50": "above" if price > i["ema50"] else "below",
        "rsi": round(float(i["rsi"]), 1),
        "rsi_zone": "overbought" if i["rsi"] >= 70 else "oversold" if i["rsi"] <= 30 else "neutral",
        "return_24_bars_pct": round(float(i["return_24_bars_pct"]), 2),
        "return_3_bars": _bucket(i["return_3_bars_pct"], -0.1, 0.1, ("falling", "flat", "rising")),
        "volatility": _bucket(i["volatility_ratio"], 0.7, 1.5, ("low", "normal", "high")),
        "volume": _bucket(i["volume_ratio"], 0.5, 2.0, ("thin", "normal", "spike")),
        # 변동폭이 왕복 수수료·슬리피지를 넘는가 (코드가 계산한 참/거짓; 정책의 진입 게이트로도 쓴다)
        "volatility_covers_costs": bool(i["atr_pct"] >= MIN_ATR_COST_MULT * 2 * COST_PER_SIDE * 100),
        "current_position": side,
        "position_pnl": "none" if side == "flat" else _bucket(pnl_pct, -0.1, 0.1, ("losing", "near breakeven", "profitable")),
    }
