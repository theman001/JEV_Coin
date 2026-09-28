"""시장 데이터 수집 (ccxt 공개 엔드포인트, API 키 불필요)."""
import time

import ccxt
import pandas as pd


STALE_TF_MULT = 1.5  # 마지막 마감 캔들의 나이가 이 배수(캔들 주기 기준)를 넘으면 stale. 1.0 이면 캔들 마감 직후마다 오탐
STALE_MIN_MS = 5_000  # 1초봉처럼 아주 짧은 주기는 배수(1.5초)만으로는 네트워크 지연에 자주 오탐(실측: 마지막 닫힌 봉 나이 ~1.3초) → 최소 5초


class StaleData(Exception):
    """마지막 닫힌 캔들이 너무 오래됨 → 이번 틱은 건너뛴다 (stale state 금지)."""


def make_exchange(name: str = "binance") -> ccxt.Exchange:
    return getattr(ccxt, name)({"enableRateLimit": True, "timeout": 10_000})


def fetch_closed(ex: ccxt.Exchange, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    """닫힌 캔들만 반환한다 (진행 중인 마지막 캔들은 버림)."""
    rows = ex.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).iloc[:-1]
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    age_ms = time.time() * 1000 - (df["ts"].iloc[-1] + tf_ms)  # 마지막 닫힌 캔들이 닫힌 지 얼마나 됐나
    if age_ms > max(STALE_TF_MULT * tf_ms, STALE_MIN_MS):
        raise StaleData(f"last closed candle is {age_ms / 1000:.0f}s old")
    return df.reset_index(drop=True)
