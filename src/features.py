"""차트 예측 특징(계산식) 라이브러리 — 단타 지표를 파이썬으로 수치화한다 (STRATEGY_RESEARCH.md §12).

계약: 입력은 마감된 캔들 DataFrame(ts[ms], open, high, low, close, volume [, taker_buy_base, n_trades]).
      출력은 같은 인덱스의 f_* 열들. **모든 열은 close[t] 까지의 정보만 사용한다 (룩어헤드 없음 — tests 가 전 열을 검사).**
      가격 거리·기울기는 ATR 단위로 정규화해 봉 주기·코인과 무관하게 비교할 수 있게 한다.
      taker_buy_base(체결 방향별 거래량)·n_trades 가 없으면 주문흐름 열은 NaN, btc 가 없으면 교차 자산 열은 NaN.
"""
import numpy as np
import pandas as pd
from ta.momentum import StochRSIIndicator, WilliamsRIndicator
from ta.trend import ADXIndicator, CCIIndicator
from ta.volume import ChaikinMoneyFlowIndicator, MFIIndicator

DAY_MS = 86_400_000


# ───────────── 기본 계산식 ─────────────
def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. 하락 평균이 0 이면 100(상승 평균도 0 이면 50)."""
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + up / dn.where(dn > 0))
    return out.where(dn > 0, pd.Series(np.where(up > 0, 100.0, 50.0), index=close.index)).where(up.notna() & dn.notna())


def rolling_linreg_end(y: pd.Series, n: int) -> pd.Series:
    """길이 n 창의 최소제곱 직선이 현재 봉에서 갖는 값 (TTM Squeeze 모멘텀의 '선형회귀 평활')."""
    x = np.arange(n, dtype=float)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    v = np.nan_to_num(y.to_numpy(float))
    slope = np.convolve(v, w[::-1], mode="full")[: len(v)]
    out = y.rolling(n).mean().to_numpy() + slope * (n - 1 - x.mean())
    out[: n + int(y.isna().sum()) - 1] = np.nan
    return pd.Series(out, index=y.index)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Supertrend(n, mult): 밴드는 유리한 방향으로만 이동하고, 종가가 반대 밴드를 넘으면 방향이 뒤집힌다. 반환 (방향 +1/−1, 선)."""
    a = atr(df, n).to_numpy()
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    ub, lb = (h + l) / 2 + mult * a, (h + l) / 2 - mult * a
    m = len(c)
    fub, flb, dirn, line = ub.copy(), lb.copy(), np.ones(m), np.full(m, np.nan)
    for i in range(1, m):
        if np.isnan(a[i]):
            continue
        prev_ok = not np.isnan(fub[i - 1])
        fub[i] = ub[i] if (not prev_ok or ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (not prev_ok or lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        dirn[i] = dirn[i - 1]
        if prev_ok:
            if dirn[i - 1] < 0 and c[i] > fub[i - 1]:
                dirn[i] = 1
            elif dirn[i - 1] > 0 and c[i] < flb[i - 1]:
                dirn[i] = -1
        line[i] = flb[i] if dirn[i] > 0 else fub[i]
    dirn[np.isnan(line)] = np.nan
    return pd.Series(dirn, index=df.index), pd.Series(line, index=df.index)


def heikin_ashi_streak(df: pd.DataFrame, cap: int = 10) -> pd.Series:
    """하이킨아시 봉 색(종가>시가)의 연속 길이. 양수=연속 상승색, 음수=연속 하락색 (|값| ≤ cap)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hac = (o + h + l + c) / 4
    hao = np.empty(len(c))
    hao[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        hao[i] = (hao[i - 1] + hac[i - 1]) / 2
    color, streak = np.sign(hac - hao), np.zeros(len(c))
    for i in range(len(c)):
        streak[i] = (streak[i - 1] + color[i] if i and color[i] == np.sign(streak[i - 1]) else color[i]) if color[i] != 0 else 0
    return pd.Series(np.clip(streak, -cap, cap), index=df.index)


def run_length(flag: pd.Series) -> pd.Series:
    f = flag.fillna(False).astype(int)
    return f.groupby((f != f.shift()).cumsum()).cumsum()


# ───────────── 특징 계산 ─────────────
MIN_BARS = 30  # 이보다 짧으면 지표 워밍업 자체가 불가능 (ta 의 ADX 등이 오류를 낸다)


def compute(df: pd.DataFrame, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    if len(df) < MIN_BARS:
        raise ValueError(f"need >= {MIN_BARS} bars, got {len(df)}")
    o, h, l, c, v, ts = (df[k] for k in ("open", "high", "low", "close", "volume", "ts"))
    a = atr(df, 14)
    F = pd.DataFrame(index=df.index)
    day = ts // DAY_MS

    # 추세 ------------------------------------------------------------
    e9, e21, e50, e200 = (ema(c, n) for n in (9, 21, 50, 200))
    F["f_ema9_21"], F["f_ema21_50"], F["f_c_ema200"] = (e9 - e21) / a, (e21 - e50) / a, (c - e200) / a
    F["f_ema21_slope5"] = (e21 - e21.shift(5)) / a
    F["f_ema_stack"] = np.where((e9 > e21) & (e21 > e50), 1, np.where((e9 < e21) & (e21 < e50), -1, 0)).astype(float)
    adx = ADXIndicator(h, l, c, window=14)
    F["f_adx"], F["f_di_diff"] = adx.adx(), adx.adx_pos() - adx.adx_neg()
    sd, sl = supertrend(df)
    F["f_st_dir"], F["f_st_dist"] = sd, (c - sl) / a
    F["f_ha_streak"] = heikin_ashi_streak(df)

    # 시장 구조 (윌리엄스 프랙탈: 좌우 2봉보다 높은/낮은 고점·저점, 우측 2봉이 지나야 확정 → 확정 시점부터만 사용) ------
    sh_ev = h.shift(2).where((h.shift(2) > h.shift(3)) & (h.shift(2) > h.shift(4)) & (h.shift(2) > h.shift(1)) & (h.shift(2) > h))
    sl_ev = l.shift(2).where((l.shift(2) < l.shift(3)) & (l.shift(2) < l.shift(4)) & (l.shift(2) < l.shift(1)) & (l.shift(2) < l))
    last_sh, last_sl = sh_ev.ffill(), sl_ev.ffill()
    prev_sh = sh_ev.dropna().shift(1).reindex(df.index).ffill()
    prev_sl = sl_ev.dropna().shift(1).reindex(df.index).ffill()
    F["f_dist_swing_high"], F["f_dist_swing_low"] = (c - last_sh) / a, (c - last_sl) / a
    F["f_bos_up"], F["f_bos_dn"] = (c > last_sh).astype(float), (c < last_sl).astype(float)
    F["f_hh"], F["f_hl"] = (last_sh > prev_sh).astype(float), (last_sl > prev_sl).astype(float)
    hi20, lo20, hi50, lo50 = h.rolling(20).max(), l.rolling(20).min(), h.rolling(50).max(), l.rolling(50).min()
    F["f_rangepos20"], F["f_rangepos50"] = (c - lo20) / (hi20 - lo20).where(hi20 > lo20), (c - lo50) / (hi50 - lo50).where(hi50 > lo50)
    F["f_don20_up"], F["f_don20_dn"] = (c - hi20.shift(1)) / a, (lo20.shift(1) - c) / a

    # 일봉 기준 레벨 (UTC 일 경계 = 09:00 KST). 전일 값은 어제 마감 데이터만 사용 ----------------------------------
    d = pd.DataFrame({"h": h, "l": l, "c": c}).groupby(day).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pdh, pdl, pdc = (day.map(d[k].shift(1)) for k in ("h", "l", "c"))
    pp = (pdh + pdl + pdc) / 3
    F["f_pdh"], F["f_pdl"], F["f_pivot"] = (c - pdh) / a, (c - pdl) / a, (c - pp) / a
    dh, dl = h.groupby(day).cummax(), l.groupby(day).cummin()
    F["f_day_rangepos"] = (c - dl) / (dh - dl).where(dh > dl)
    F["f_day_ret"] = (c / o.groupby(day).transform("first") - 1) * 100

    # 모멘텀 ----------------------------------------------------------
    r14 = rsi(c, 14)
    F["f_rsi14"], F["f_rsi2"] = r14, rsi(c, 2)
    F["f_rsi14_slope3"] = r14 - r14.shift(3)
    st = StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    F["f_stochrsi_k"], F["f_stochrsi_d"] = st.stochrsi_k(), st.stochrsi_d()
    macd_line = ema(c, 12) - ema(c, 26)
    hist = macd_line - ema(macd_line.dropna(), 9).reindex(df.index)
    F["f_macd_line"], F["f_macd_hist"], F["f_macd_hist_delta"] = macd_line / a, hist / a, (hist - hist.shift(1)) / a
    for k in (3, 6, 12, 24):
        F[f"f_roc{k}"] = (c / c.shift(k) - 1) * 100
    F["f_cci20"], F["f_willr14"] = CCIIndicator(h, l, c, window=20).cci(), WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    ttm_delta = c - ((h.rolling(20).max() + l.rolling(20).min()) / 2 + c.rolling(20).mean()) / 2
    F["f_ttm_mom"] = rolling_linreg_end(ttm_delta, 20) / a

    # 변동성 · 스퀴즈 ---------------------------------------------------
    F["f_atr_pct"] = a / c * 100
    mid, sd20 = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    bbu, bbl = mid + 2 * sd20, mid - 2 * sd20
    F["f_bb_pctb"], F["f_bb_bw"] = (c - bbl) / (bbu - bbl).where(bbu > bbl), (bbu - bbl) / mid
    F["f_bb_bw_rank100"] = F["f_bb_bw"].rolling(100).rank(pct=True)
    a20 = atr(df, 20)
    squeeze = ((bbu < mid + 1.5 * a20) & (bbl > mid - 1.5 * a20)).where(a20.notna() & sd20.notna())  # 볼린저가 켈트너 안쪽
    F["f_squeeze_on"], F["f_squeeze_len"] = squeeze.astype(float), run_length(squeeze).where(squeeze.notna())
    F["f_squeeze_fired"] = ((squeeze.shift(1) == True) & (squeeze == False)).astype(float).where(squeeze.notna() & squeeze.shift(1).notna())
    lr = np.log(c).diff()
    F["f_volr_fast_slow"] = lr.rolling(12).std() / lr.rolling(72).std()
    F["f_range_exp"] = true_range(df) / a
    F["f_parkinson20"] = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))) * 100

    # 거래량 · 주문흐름 --------------------------------------------------
    F["f_vol_ratio"] = v / v.rolling(20).median()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    F["f_obv_slope20"] = (obv - obv.shift(20)) / v.rolling(20).sum()
    F["f_mfi14"], F["f_cmf20"] = MFIIndicator(h, l, c, v, window=14).money_flow_index(), ChaikinMoneyFlowIndicator(h, l, c, v, window=20).chaikin_money_flow()
    tp = (h + l + c) / 3
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()  # UTC 일 시작 앵커 VWAP
    F["f_vwap_dev"] = (c - vwap) / a
    F["f_vwap_dev_change3"] = F["f_vwap_dev"] - F["f_vwap_dev"].shift(3)
    rv = (tp * v).rolling(96).sum() / v.rolling(96).sum()
    F["f_rvwap96_dev"] = (c - rv) / a
    if "taker_buy_base" in df:
        tb, n_tr = df["taker_buy_base"], df["n_trades"]
        delta = (2 * tb - v) / v.where(v > 0)  # 공격적 매수 − 매도 (거래량 대비, −1~+1)
        F["f_taker_ratio"], F["f_delta"], F["f_delta_ema3"] = tb / v.where(v > 0), delta, delta.ewm(span=3, adjust=False, min_periods=3).mean()
        cvd12 = (2 * tb - v).rolling(12).sum() / v.rolling(12).sum()
        F["f_cvd12"] = cvd12
        F["f_delta_z96"] = (delta - delta.rolling(96).mean()) / delta.rolling(96).std()
        F["f_trades_ratio"] = n_tr / n_tr.rolling(20).median()
        size = v / n_tr.where(n_tr > 0)
        F["f_avg_trade_size_ratio"] = size / size.rolling(20).median()
        ret12 = c / c.shift(12) - 1
        F["f_cvd_div"] = np.where(np.sign(cvd12) != np.sign(ret12), np.sign(cvd12), 0.0)  # +: 가격은 안 오르는데 매수 우위(강세 다이버전스)
        F.loc[cvd12.isna() | ret12.isna(), "f_cvd_div"] = np.nan
    else:
        for k in ("taker_ratio", "delta", "delta_ema3", "cvd12", "delta_z96", "trades_ratio", "avg_trade_size_ratio", "cvd_div"):
            F[f"f_{k}"] = np.nan

    # 캔들 --------------------------------------------------------------
    body, top, bot, rng = c - o, np.maximum(o, c), np.minimum(o, c), h - l
    uw, lw, ab = h - top, bot - l, (c - o).abs()
    F["f_body"], F["f_upper_wick"], F["f_lower_wick"], F["f_range"] = body / a, uw / a, lw / a, rng / a
    pc, po = c.shift(1), o.shift(1)
    bull_eng = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear_eng = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    hammer = (lw >= 2 * ab) & (uw <= ab) & (rng > 0)
    star = (uw >= 2 * ab) & (lw <= ab) & (rng > 0)
    F["f_bull_engulf"], F["f_bear_engulf"], F["f_hammer"], F["f_shooting_star"] = (x.astype(float) for x in (bull_eng, bear_eng, hammer, star))
    F["f_doji"] = (ab < 0.1 * rng.where(rng > 0)).astype(float)
    F["f_inside_bar"] = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(float)
    F["f_candle_signal"] = F["f_bull_engulf"] + F["f_hammer"] - F["f_bear_engulf"] - F["f_shooting_star"]
    F.loc[a.isna() | pc.isna(), ["f_bull_engulf", "f_bear_engulf", "f_hammer", "f_shooting_star", "f_doji", "f_inside_bar", "f_candle_signal"]] = np.nan

    # 시간대 (UTC) ------------------------------------------------------
    tod = (ts % DAY_MS) / DAY_MS
    F["f_tod_sin"], F["f_tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    dow = (day + 3) % 7  # 월=0 (epoch 일 0 = 목요일)
    F["f_is_weekend"] = (dow >= 5).astype(float)
    hour = (ts % DAY_MS) // 3_600_000
    F["f_is_asia"], F["f_is_eu"], F["f_is_us"] = ((hour < 8).astype(float), ((hour >= 7) & (hour < 16)).astype(float), ((hour >= 13) & (hour < 22)).astype(float))

    # 교차 자산: BTC 가 알트를 선행하는가 -------------------------------------
    keys = ("btc_ret1", "btc_ret3", "btc_ret12", "rel_strength12", "btc_delta3")
    if btc is not None:
        b = btc.set_index("ts")
        bc = b["close"].reindex(ts.to_numpy()).to_numpy()
        for k in (1, 3, 12):
            F[f"f_btc_ret{k}"] = (pd.Series(bc, index=df.index) / pd.Series(bc, index=df.index).shift(k) - 1) * 100
        F["f_rel_strength12"] = (c / c.shift(12) - 1) * 100 - F["f_btc_ret12"]
        if "taker_buy_base" in b:
            bd = ((2 * b["taker_buy_base"] - b["volume"]) / b["volume"].where(b["volume"] > 0)).reindex(ts.to_numpy()).to_numpy()
            F["f_btc_delta3"] = pd.Series(bd, index=df.index).rolling(3).mean()
        else:
            F["f_btc_delta3"] = np.nan
    else:
        for k in keys:
            F[f"f_{k}"] = np.nan
    return F


FEATURES = [
    "f_ema9_21", "f_ema21_50", "f_c_ema200", "f_ema21_slope5", "f_ema_stack", "f_adx", "f_di_diff", "f_st_dir", "f_st_dist", "f_ha_streak",
    "f_dist_swing_high", "f_dist_swing_low", "f_bos_up", "f_bos_dn", "f_hh", "f_hl", "f_rangepos20", "f_rangepos50", "f_don20_up", "f_don20_dn",
    "f_pdh", "f_pdl", "f_pivot", "f_day_rangepos", "f_day_ret",
    "f_rsi14", "f_rsi2", "f_rsi14_slope3", "f_stochrsi_k", "f_stochrsi_d", "f_macd_line", "f_macd_hist", "f_macd_hist_delta",
    "f_roc3", "f_roc6", "f_roc12", "f_roc24", "f_cci20", "f_willr14", "f_ttm_mom",
    "f_atr_pct", "f_bb_pctb", "f_bb_bw", "f_bb_bw_rank100", "f_squeeze_on", "f_squeeze_len", "f_squeeze_fired", "f_volr_fast_slow", "f_range_exp", "f_parkinson20",
    "f_vol_ratio", "f_obv_slope20", "f_mfi14", "f_cmf20", "f_vwap_dev", "f_vwap_dev_change3", "f_rvwap96_dev",
    "f_taker_ratio", "f_delta", "f_delta_ema3", "f_cvd12", "f_delta_z96", "f_trades_ratio", "f_avg_trade_size_ratio", "f_cvd_div",
    "f_body", "f_upper_wick", "f_lower_wick", "f_range", "f_bull_engulf", "f_bear_engulf", "f_hammer", "f_shooting_star", "f_doji", "f_inside_bar", "f_candle_signal",
    "f_tod_sin", "f_tod_cos", "f_is_weekend", "f_is_asia", "f_is_eu", "f_is_us",
    "f_btc_ret1", "f_btc_ret3", "f_btc_ret12", "f_rel_strength12", "f_btc_delta3",
]
