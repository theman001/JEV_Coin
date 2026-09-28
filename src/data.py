"""시장 데이터 수집 (ccxt 공개 엔드포인트, API 키 불필요)."""
import time

import ccxt
import pandas as pd


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
    if age_ms > tf_ms:
        raise StaleData(f"last closed candle is {age_ms / 1000:.0f}s old")
    return df.reset_index(drop=True)
