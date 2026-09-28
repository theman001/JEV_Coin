"""미시구조 데이터·특징 (Binance USDⓈ-M 선물) — STRATEGY_RESEARCH.md §13.

  펀딩비(8h) · 프리미엄 지수(선물−지수 베이시스, 5m) · metrics(미결제약정·롱숏비율·테이커 롱숏 거래량비, 5m) · bookDepth(주문서 ±0.2~5% 누적 명목가, 30초)

인과성 계약: 봉 마감 시각 T(= 봉 시작 + 주기) 특징은 **T 이전에 공개된 행만** 쓴다(merge_asof backward).
  - metrics 는 관측 시각(create_time)의 의미가 문서화돼 있지 않아 **한 주기(5분) 지연**을 보수적으로 둔다.
  - 청산은 무료 이력이 없어 여기 없다 → src/collect_liq.py 로 앞으로 수집한다.
"""
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE, REST = "https://data.binance.vision/data/futures/um", "https://fapi.binance.com"
DIR = Path("data") / "micro"
DAY, M5, H8 = 86_400_000, 300_000, 8 * 3_600_000
TOL = 15 * 60_000  # 이 시간 넘게 오래된 관측은 쓰지 않는다 (결측 처리)


# ───────────── 파서 (순수 함수) ─────────────
def to_ms(s: pd.Series) -> pd.Series:
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(1, "ms")).astype("int64")


def read_zip_csv(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]))


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"t": to_ms(df["create_time"]), "oi": df["sum_open_interest"], "oi_val": df["sum_open_interest_value"],
                        "top_pos_ls": df["sum_toptrader_long_short_ratio"], "top_acc_ls": df["count_toptrader_long_short_ratio"],
                        "glob_ls": df["count_long_short_ratio"], "taker_ls": df["sum_taker_long_short_vol_ratio"]})
    return out.sort_values("t").reset_index(drop=True)


def parse_premium(df: pd.DataFrame) -> pd.DataFrame:
    """프리미엄 지수 5m 봉 → t = 봉 마감 시각(open+5m), v = 종가(비율, 0.001 = 0.1%)."""
    return pd.DataFrame({"t": df["open_time"].astype("int64") + M5, "v": df["close"].astype(float)}).sort_values("t").reset_index(drop=True)


def aggregate_depth(df: pd.DataFrame) -> pd.DataFrame:
    """bookDepth 하루치(30초 스냅샷 × 12 밴드) → 5분 버킷(t = 버킷 끝). b*/a* = 그 버킷 마지막 스냅샷의 ±x% 이내 누적 명목가(매수/매도),
    imb02_mean = 버킷 내 스냅샷 평균 ±0.2% 불균형."""
    d = df.assign(t=to_ms(df["timestamp"]))
    wide = d.pivot_table(index="t", columns="percentage", values="notional", aggfunc="last")
    band = lambda x: wide[x] if x in wide.columns else pd.Series(np.nan, index=wide.index)  # ±0.2% 밴드는 2026-01-15 이후 파일에만 있다
    snap = pd.DataFrame({"b02": band(-0.2), "a02": band(0.2), "b1": band(-1.0), "a1": band(1.0), "b5": band(-5.0), "a5": band(5.0)})
    snap["imb"] = (snap["b02"] - snap["a02"]) / (snap["b02"] + snap["a02"])
    bucket = (snap.index.to_numpy() // M5) * M5 + M5
    g = snap.groupby(bucket)
    out = g[["b02", "a02", "b1", "a1", "b5", "a5"]].last()
    out["imb02_mean"] = g["imb"].mean()
    out.index.name = "t"
    return out.reset_index()


# ───────────── 다운로드 (캐시) ─────────────
def _get(url: str, tries: int = 5):
    """없는 파일(403/404)은 None, 그 외 오류는 백오프 재시도."""
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(0.5 * 2 ** k)
        except Exception:  # 네트워크 일시 오류
            time.sleep(0.5 * 2 ** k)
    raise RuntimeError(f"다운로드 반복 실패: {url}")


def day_list(start_ms: int, end_ms: int) -> list[str]:
    """[start, end) 안의 '완결된' UTC 일자 (오늘은 제외)."""
    return [time.strftime("%Y-%m-%d", time.gmtime(d * 86400)) for d in range(start_ms // DAY, end_ms // DAY)]


def _daily(sym: str, kind: str, url_fn, parse, start_ms: int, end_ms: int, workers: int = 16) -> pd.DataFrame:
    DIR.mkdir(parents=True, exist_ok=True)
    cache, done_p = DIR / f"{sym.replace('/', '_')}_{kind}.csv", DIR / f"{sym.replace('/', '_')}_{kind}.days"
    done = set(done_p.read_text().split()) if done_p.exists() else set()
    frames = [pd.read_csv(cache)] if cache.exists() and done else []
    todo = [d for d in day_list(start_ms, end_ms) if d not in done]

    def one(d):
        raw = _get(url_fn(d))
        return d, (parse(read_zip_csv(raw)) if raw else None)

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo))
    ok = [(d, f) for d, f in res if f is not None and len(f)]
    if ok:
        frames += [f for _, f in ok]
        pd.concat(frames, ignore_index=True).drop_duplicates("t").sort_values("t").to_csv(cache, index=False)
        done_p.write_text(" ".join(sorted(done | {d for d, _ in ok})))
    return pd.read_csv(cache) if cache.exists() else pd.DataFrame()


def load_funding(sym: str, start_ms: int) -> pd.DataFrame:
    """REST 펀딩비 이력 (1000건/요청). t = 정산 시각, rate = 직전 8시간 구간에 적용된 펀딩비."""
    mid, rows, since = sym.replace("/", ""), [], start_ms
    while True:
        r = json.loads(urllib.request.urlopen(f"{REST}/fapi/v1/fundingRate?symbol={mid}&startTime={since}&limit=1000", timeout=30).read())
        if not r:
            break
        rows += r
        since = r[-1]["fundingTime"] + 1
        if len(r) < 1000:
            break
    return pd.DataFrame({"t": [int(x["fundingTime"]) for x in rows], "rate": [float(x["fundingRate"]) for x in rows]}).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def load_frames(sym: str, start_ms: int, end_ms: int | None = None) -> dict:
    """한 코인의 미시구조 프레임 4종. start_ms 는 z-점수 워밍업을 위해 창 시작보다 넉넉히 앞서 줘야 한다."""
    end_ms = end_ms or int(time.time() * 1000)
    mid = sym.replace("/", "")
    metrics = lambda d: f"{BASE}/daily/metrics/{mid}/{mid}-metrics-{d}.zip"
    depth = lambda d: f"{BASE}/daily/bookDepth/{mid}/{mid}-bookDepth-{d}.zip"
    premium = lambda d: f"{BASE}/daily/premiumIndexKlines/{mid}/5m/{mid}-5m-{d}.zip"
    return {"funding": load_funding(sym, start_ms),
            "premium": _daily(sym, "premium", premium, parse_premium, start_ms, end_ms),
            "metrics": _daily(sym, "metrics", metrics, parse_metrics, start_ms, end_ms),
            "depth": _daily(sym, "depth", depth, aggregate_depth, start_ms, end_ms, workers=12)}


# ───────────── 특징 ─────────────
def _asof(frame: pd.DataFrame, q, cols: list[str], lag: int = 0, tol: int = TOL) -> pd.DataFrame:
    """각 질의 시각 q 이전(≤ q − lag)에 공개된 가장 최근 행의 cols. 없거나 tol 보다 오래됐으면 NaN. q 는 정렬돼 있지 않아도 된다."""
    q = np.asarray(q, dtype="int64")
    order = np.argsort(q, kind="stable")
    right = frame[["t", *cols]].copy()
    right["t"] = right["t"].astype("int64") + lag
    got = pd.merge_asof(pd.DataFrame({"q": q[order]}), right.sort_values("t"), left_on="q", right_on="t", direction="backward", tolerance=tol)[cols].to_numpy(float)
    out = np.empty_like(got)
    out[order] = got
    return pd.DataFrame(out, columns=cols)


def _ln(x: pd.Series, lo: float = 0.05, hi: float = 20.0) -> pd.Series:
    return np.log(x.clip(lo, hi))


MICRO_FEATURES = [
    "f_funding", "f_funding_chg", "f_funding_z", "f_funding_sin", "f_funding_cos", "f_funding_near",
    "f_premium", "f_premium_z", "f_premium_chg",
    "f_oi_chg1", "f_oi_chg12", "f_oi_z", "f_oi_x_ret", "f_top_ls", "f_top_ls_chg12", "f_glob_ls", "f_top_glob_div", "f_taker_fut",
    "f_bd_imb02", "f_bd_imb1", "f_bd_imb5", "f_bd_imb02_mean", "f_bd_imb02_chg", "f_bd_depth1_rel", "f_bd_slope", "f_bd_asym1",
]


def compute_micro(df: pd.DataFrame, fr: dict, tf_ms: int) -> pd.DataFrame:
    """봉(df: ts=시작 ms, open, close 필요) × 미시구조 프레임 → f_* 열. 봉 마감 시각 T = ts + tf_ms 이전 정보만 사용."""
    T = df["ts"].to_numpy("int64") + tf_ms
    n5 = max(1, tf_ms // M5)
    F = pd.DataFrame(index=df.index)
    nan = lambda: pd.Series(np.nan, index=df.index)

    # 펀딩비: 정산 시각의 값 (정산 시점에 공개). 8시간 위상·정산 근접 여부도 함께.
    fu = fr["funding"].copy()
    fu["prev"], fu["z"] = fu["rate"].shift(1), (fu["rate"] - fu["rate"].rolling(90).mean()) / fu["rate"].rolling(90).std()
    g = _asof(fu, T, ["rate", "prev", "z"], tol=9 * 3_600_000).set_axis(df.index)
    F["f_funding"], F["f_funding_chg"], F["f_funding_z"] = g["rate"] * 1e4, (g["rate"] - g["prev"]) * 1e4, g["z"]
    phase = (T % H8) / H8
    F["f_funding_sin"], F["f_funding_cos"] = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)
    F["f_funding_near"] = (np.minimum(phase, 1 - phase) * 480 <= 30).astype(float)

    # 프리미엄 지수(선물 − 지수, 비율): 베이시스
    pr = fr["premium"].copy()
    pr["z"] = (pr["v"] - pr["v"].rolling(864).mean()) / pr["v"].rolling(864).std()
    g0, g1 = _asof(pr, T, ["v", "z"]).set_axis(df.index), _asof(pr, T - tf_ms, ["v"]).set_axis(df.index)
    F["f_premium"], F["f_premium_z"], F["f_premium_chg"] = g0["v"] * 1e4, g0["z"], (g0["v"] - g1["v"]) * 1e4

    # metrics (5분 지연): 미결제약정 변화, 상위 트레이더/전체 계정 롱숏비, 선물 테이커 롱숏 거래량비
    me = fr["metrics"].copy()
    me.loc[me["oi"] <= 0, ["oi", "oi_val"]] = np.nan  # 실데이터에 OI=0 행이 코인당 ~25개(점검 구간 추정) → 그대로 두면 변화율이 ±inf/가짜 -100% 가 된다
    me["tl"] = _ln(me["taker_ls"]).rolling(n5).mean()
    cols = ["oi", "top_pos_ls", "glob_ls", "tl"]
    m0, m1, m12 = (_asof(me, T - k * tf_ms, cols, lag=M5, tol=TOL + M5).set_axis(df.index) for k in (0, 1, 12))
    chg1 = (m0["oi"] / m1["oi"] - 1) * 100
    F["f_oi_chg1"], F["f_oi_chg12"] = chg1, (m0["oi"] / m12["oi"] - 1) * 100
    F["f_oi_z"] = (chg1 - chg1.rolling(200).mean()) / chg1.rolling(200).std()
    F["f_oi_x_ret"] = chg1 * ((df["close"] / df["open"] - 1) * 100)  # +: 가격과 같은 방향으로 OI 증가(레버리지 추세), −: OI 감소·청산/이익실현
    F["f_top_ls"], F["f_glob_ls"] = _ln(m0["top_pos_ls"]), _ln(m0["glob_ls"])
    F["f_top_ls_chg12"], F["f_top_glob_div"] = F["f_top_ls"] - _ln(m12["top_pos_ls"]), F["f_top_ls"] - F["f_glob_ls"]
    F["f_taker_fut"] = m0["tl"]

    # 주문서 깊이: ±0.2/1/5% 누적 명목가의 매수·매도 불균형과 유동성 두께
    de = fr["depth"].copy()
    de["imb_roll"] = de["imb02_mean"].rolling(n5).mean()
    de["d1"] = de["b1"] + de["a1"]
    de["d1_rel"] = np.log(de["d1"] / de["d1"].rolling(2016).median())
    imb = lambda b, a: (b - a) / (b + a)
    d0 = _asof(de, T, ["b02", "a02", "b1", "a1", "b5", "a5", "imb_roll", "d1_rel"]).set_axis(df.index)
    d1 = _asof(de, T - tf_ms, ["b02", "a02"]).set_axis(df.index)
    F["f_bd_imb02"], F["f_bd_imb1"], F["f_bd_imb5"] = imb(d0["b02"], d0["a02"]), imb(d0["b1"], d0["a1"]), imb(d0["b5"], d0["a5"])
    F["f_bd_imb02_mean"], F["f_bd_imb02_chg"] = d0["imb_roll"], F["f_bd_imb02"] - imb(d1["b02"], d1["a02"])
    F["f_bd_depth1_rel"] = d0["d1_rel"]
    F["f_bd_slope"], F["f_bd_asym1"] = np.log((d0["b5"] + d0["a5"]) / (d0["b1"] + d0["a1"])), np.log(d0["b1"] / d0["a1"])
    return F[MICRO_FEATURES]
