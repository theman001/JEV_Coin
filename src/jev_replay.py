"""Jev 입력 비교 실험 — 사전 고정 설계 (STRATEGY_RESEARCH.md §11). 결과를 보고 설계를 바꾸지 않는다.

같은 표본(코인당 무작위 캔들 마감 시점)에 대해 세 가지 입력으로 Jev 를 호출하고 방향 판단의 정보량을 비교한다:
  labels      = 라이브 봇의 build_state 그대로 (대조군)
  raw         = 마지막 48개 마감 캔들 OHLCV 만
  raw+labels  = 둘 다
원시 데이터는 가격을 마지막 종가=100 으로 재기준화하고 거래량은 48봉 중앙값 대비 배수로 준다 (절대 가격·날짜가 코인과 시기를 식별해
학습 데이터 기억이 새는 것을 막기 위한 마스킹). 코인명·날짜·타임스탬프는 넣지 않는다.

  python -m src.jev_replay run [--n-per-coin 2000]     # Jev 호출 (캐시됨, 재실행하면 이어서)
  python -m src.jev_replay analyze                     # 캐시에서 분석
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import load
from .data import make_exchange
from .judge import QUESTIONS
from .policy import COST_PER_SIDE
from .state import build_state

CACHE = Path("data") / "jev_replay" / "cache.json"
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
VARIANTS = ("labels", "raw", "raw+labels")
WINDOW, HIST, H = 48, 200, 3  # 원시 캔들 수 / 라벨 계산용 히스토리(라이브 봇과 동일) / 1차 전방 수익률 캔들 수
RT_COST = 2 * COST_PER_SIDE * 100  # 왕복 비용 %


# ───────────── 입력 구성 ─────────────
def sample_indices(n_bars: int, n: int, seed: int) -> np.ndarray:
    """라벨용 히스토리와 h 캔들 뒤 종가가 모두 있는 판단 시점 중 무작위 n개."""
    return np.sort(np.random.default_rng(seed).choice(np.arange(HIST - 1, n_bars - H), n, replace=False))


def raw_block(df: pd.DataFrame, i: int) -> dict:
    """봉 i 마감 시점까지의 마지막 WINDOW 개 캔들. 가격은 마지막 종가=100 으로 재기준화, 거래량은 창 내 중앙값 대비 배수."""
    w = df.iloc[i - WINDOW + 1: i + 1]
    ref, vmed = float(w["close"].iloc[-1]), float(w["volume"].median()) or 1.0
    rb = lambda s: [round(float(x) / ref * 100, 3) for x in s]
    return {"open": rb(w["open"]), "high": rb(w["high"]), "low": rb(w["low"]), "close": rb(w["close"]),
            "volume_vs_median": [round(float(v) / vmed, 2) for v in w["volume"]]}


def make_state(df: pd.DataFrame, i: int, variant: str) -> dict:
    state = {"timeframe": "5m"}
    if "labels" in variant:  # 라이브 봇의 상태 (코인명만 가림). 포지션은 없음(flat)
        state = build_state(df.iloc[i - HIST + 1: i + 1].reset_index(drop=True), "asset", "5m")
    if "raw" in variant:
        state = {**state, "recent_candles_oldest_to_newest": raw_block(df, i)}
    return state


# ───────────── 호출 ─────────────
def _questions():
    from typesafe_sdk import Noul
    return {**QUESTIONS, "up3": Noul(instructions="The close price 3 candles from now will be higher than the current close.")}


def call(client, q, state) -> dict | None:
    from typesafe_sdk import TypeSafeError
    try:
        a = client.system_one(state=state, questions=q).answers
        p = a["side"].probabilities
        return {"side": a["side"].choice, "conf": a["side"].confidence, "p_long": p.get("long", 0.0), "p_short": p.get("short", 0.0),
                "p_flat": p.get("flat", 0.0), "rev": a["reversal_risk"].noul, "up3": a["up3"].noul}
    except (TypeSafeError, KeyError):
        return None


def run(n_per_coin: int, workers: int = 4, seed: int = 0) -> None:
    from typesafe_sdk import TypeSafeClient
    ex, client, q = make_exchange(), TypeSafeClient(), _questions()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    todo = []
    for ci, coin in enumerate(COINS):
        df = load(coin, "5m", 2, ex, 1000)
        for i in sample_indices(len(df), n_per_coin, seed + ci):
            for v in VARIANTS:
                key = f"{v}|{coin}|{int(df['ts'].iloc[i])}"
                if key not in cache:
                    todo.append((key, make_state(df, int(i), v)))
    print(f"표본 {len(COINS) * n_per_coin}개 × {len(VARIANTS)}변형 = {len(COINS) * n_per_coin * len(VARIANTS)}회 중 신규 {len(todo)}회 (캐시 {len(COINS) * n_per_coin * len(VARIANTS) - len(todo)}회)", flush=True)
    t0, fails = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for n, (key, r) in enumerate(pool.map(lambda kv: (kv[0], call(client, q, kv[1])), todo), 1):
            if r is None:
                fails += 1
            else:
                cache[key] = r
            if n % 2000 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s, 실패 {fails})", flush=True)
    CACHE.write_text(json.dumps(cache))
    print(f"완료 {time.time() - t0:.0f}s, 실패 {fails}건 (재실행하면 실패분만 이어서 호출)")


# ───────────── 분석 ─────────────
def rank_corr(x, y) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1]
    return 0.0 if np.isnan(r) else float(r)  # 상수 점수(전부 같은 값)는 정보 없음 = 0


def build_table(cache: dict, data: dict) -> pd.DataFrame:
    """캐시 → (코인, 시각)별 한 행: 세 변형이 모두 응답한 표본만. 전방 수익률·모멘텀은 캔들 데이터에서 계산."""
    grouped = {}
    for key, r in cache.items():
        v, coin, ts = key.split("|")
        grouped.setdefault((coin, int(ts)), {})[v] = r
    pos = {c: pd.Series(np.arange(len(df)), index=df["ts"].to_numpy()) for c, df in data.items()}
    rows = []
    for (coin, ts), d in grouped.items():
        if len(d) < len(VARIANTS):
            continue
        close, i = data[coin]["close"].to_numpy(float), int(pos[coin][ts])
        row = {"coin": coin, "ts": ts, "mom3": (close[i] / close[i - 3] - 1) * 100, "fwd1": (close[i + 1] / close[i] - 1) * 100, "fwd3": (close[i + H] / close[i] - 1) * 100}
        for v, r in d.items():
            row |= {f"s_{v}": r["p_long"] - r["p_short"], f"side_{v}": r["side"], f"conf_{v}": r["conf"], f"up3_{v}": r["up3"]}
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ic(T: pd.DataFrame, col: str, ycol: str = "fwd3") -> float:
    return float(np.mean([rank_corr(g[col], g[ycol]) for _, g in T.groupby("coin")]))


def perm_p_ic(T: pd.DataFrame, col: str, B: int = 1000, seed: int = 0) -> float:
    """코인 내 전방 수익률을 섞어 만든 평균 IC 분포 대비 양측 p."""
    rng, obs = np.random.default_rng(seed), mean_ic(T, col)
    parts = [(pd.Series(g[col]).rank().to_numpy(), pd.Series(g["fwd3"]).rank().to_numpy()) for _, g in T.groupby("coin")]
    null = np.array([np.mean([np.corrcoef(rs, rng.permutation(rf))[0, 1] for rs, rf in parts]) for _ in range(B)])
    return float((1 + (np.abs(np.nan_to_num(null)) >= abs(obs)).sum()) / (B + 1))


def delta_ic_ci(T: pd.DataFrame, a: str, b: str, B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """평균 IC(b) − 평균 IC(a)의 짝지은 부트스트랩(같은 행을 함께 재추출). 반환 (관측, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    arrs = [(g[f"s_{a}"].to_numpy(), g[f"s_{b}"].to_numpy(), g["fwd3"].to_numpy()) for _, g in T.groupby("coin")]
    out = np.empty(B)
    for k in range(B):
        d = []
        for sa, sb, f in arrs:
            idx = rng.integers(0, len(f), len(f))
            d.append(rank_corr(sb[idx], f[idx]) - rank_corr(sa[idx], f[idx]))
        out[k] = np.mean(d)
    return mean_ic(T, f"s_{b}") - mean_ic(T, f"s_{a}"), *np.percentile(out, [2.5, 97.5])


def boot_ci(x, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x, rng = np.asarray(x, float), np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), *np.percentile(m, [2.5, 97.5])


def accuracy(T: pd.DataFrame, v: str, B: int = 1000, seed: int = 0) -> dict:
    """argmax 가 long/short 인 표본의 방향 정확도와, 코인 내 전방 수익률을 섞은 무작위 대비 단측 순열 p."""
    side = T[f"side_{v}"].map({"long": 1, "short": -1, "flat": 0}).to_numpy()
    m = (side != 0) & (T["fwd3"].to_numpy() != 0)
    hit = float((side[m] == np.sign(T["fwd3"].to_numpy()[m])).mean()) if m.any() else np.nan
    rng, null = np.random.default_rng(seed), []
    groups = [np.flatnonzero((T["coin"] == c).to_numpy()) for c in T["coin"].unique()]
    f = T["fwd3"].to_numpy()
    for _ in range(B):
        fp = f.copy()
        for g in groups:
            fp[g] = rng.permutation(f[g])
        mm = (side != 0) & (fp != 0)
        null.append((side[mm] == np.sign(fp[mm])).mean())
    return {"n": int(m.sum()), "hit": hit, "chance": float(np.mean(null)), "p": float((1 + (np.array(null) >= hit).sum()) / (B + 1))}


def analyze() -> None:
    ex = make_exchange()
    data = {c: load(c, "5m", 2, ex, 1000) for c in COINS}
    cache = json.loads(CACHE.read_text())
    T = build_table(cache, data)
    n_all = len({k.split("|", 1)[1] for k in cache})
    print(f"\n분석 표본: 세 변형이 모두 응답한 {len(T)}개 (호출 표본 {n_all}개 중; 코인별 {T.groupby('coin').size().to_dict()})")
    print(f"기준: h={H} 캔들(15분) 뒤 수익률, 왕복 비용 {RT_COST:.2f}%. 전방 수익률 평균 {T.fwd3.mean():+.4f}%, 상승 비율 {(T.fwd3 > 0).mean() * 100:.1f}%")

    print("\n[1차] 코인 내 Spearman IC (점수 = P(long) − P(short) vs 3캔들 뒤 수익률)")
    print(f"  {'변형':<12}{'평균 IC':>9}{'순열 p':>8}   코인별 IC")
    for v in VARIANTS:
        per = {c.split('/')[0]: rank_corr(g[f"s_{v}"], g["fwd3"]) for c, g in T.groupby("coin")}
        print(f"  {v:<12}{mean_ic(T, f's_{v}'):+9.4f}{perm_p_ic(T, f's_{v}'):8.3f}   " + "  ".join(f"{k} {x:+.3f}" for k, x in per.items()))
    print(f"  {'(참고) 직전3봉 모멘텀':<12}{np.mean([rank_corr(g['mom3'], g['fwd3']) for _, g in T.groupby('coin')]):+9.4f}")
    d, lo, hi = delta_ic_ci(T, "labels", "raw+labels")
    print(f"\n  ★ ΔIC (V3 raw+labels − V1 labels) = {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  → {'CI 가 0 을 제외하고 양수' if lo > 0 else 'CI 가 0 을 포함하거나 음수'}")
    for a, b in (("labels", "raw"), ("raw", "raw+labels")):
        d2, l2, h2 = delta_ic_ci(T, a, b)
        print(f"    (참고) ΔIC ({b} − {a}) = {d2:+.4f}   95% CI [{l2:+.4f}, {h2:+.4f}]")

    print("\n[2차] 방향 정확도 (argmax long/short) — 무작위(전방 수익률 섞기) 기대치 대비")
    for v in VARIANTS:
        a = accuracy(T, v)
        sides = T[f"side_{v}"].value_counts(normalize=True).mul(100).round(0).to_dict()
        print(f"  {v:<12} n={a['n']:5d}  정확도 {a['hit'] * 100:5.1f}%  (무작위 기대 {a['chance'] * 100:5.1f}%, 단측 p={a['p']:.3f})   판단 분포 {sides}")

    print(f"\n[2차] 거래당 기대수익 % (long=+전방, short=−전방, 진입=판단 캔들 종가 → 3캔들 뒤 종가; 왕복 {RT_COST:.2f}% 차감)")
    print(f"  {'변형':<12}{'조건':<14}{'n':>6}{'총 평균':>9}{'비용 후 평균 [95% CI]':>26}")
    for v in VARIANTS:
        sd = T[f"side_{v}"].map({"long": 1, "short": -1, "flat": 0}).to_numpy()
        gross = sd * T["fwd3"].to_numpy()
        for name, m in (("전체", sd != 0), ("확신도≥0.6", (sd != 0) & (T[f"conf_{v}"].to_numpy() >= 0.6)), ("롱 전용", sd == 1)):
            if m.sum() > 30:
                mean, lo, hi = boot_ci(gross[m] - RT_COST)
                print(f"  {v:<12}{name:<14}{int(m.sum()):6d}{gross[m].mean():+9.4f}{mean:+14.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  (참고) 항상 롱 {T.fwd3.mean() - RT_COST:+.4f} / 항상 숏 {-T.fwd3.mean() - RT_COST:+.4f} (비용 후 평균)")

    print("\n[2차] up3 (Noul: 3캔들 뒤 종가가 더 높다) IC")
    for v in VARIANTS:
        print(f"  {v:<12}{mean_ic(T, f'up3_{v}'):+9.4f}   (순열 p {perm_p_ic(T, f'up3_{v}'):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--n-per-coin", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.n_per_coin, a.workers) if a.cmd == "run" else analyze()
