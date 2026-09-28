"""Jev 판단 시험 — 풍부한 계산 특징 입력 (STRATEGY_RESEARCH.md §12.6, 사전 고정 설계). 결과를 보고 설계를 바꾸지 않는다.

입력: 계산식 87개 특징 값 + 활성 셋업 목록 (코인명·날짜·절대 가격 없음).
질문: 비용 인식 크기 질문(3/12캔들 뒤 종가가 0.28%=왕복 비용×2 넘게 높다/낮다) + 라이브 봇의 side(Choice).

  python -m src.jev_rich run [--tfs 5m 15m]      # 표본 구성 + Jev 호출 (캐시, 이어서 실행 가능)
  python -m src.jev_rich analyze
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .jev_replay import rank_corr
from .judge import QUESTIONS
from .data import make_exchange
from .screen import COINS, COST, DAY_MS, WARM, build, cluster_mean, halves, norm_p1

DIR = Path("data") / "jev_rich"
N_PER_COIN, THRESH, H_LIST, TFS = 1000, 0.28, (3, 12), ("5m", "15m")
SETUP_IDS = {"S1 EMA+VWAP 눌림": "ema_vwap_pullback", "S2 스퀴즈 돌파": "squeeze_breakout", "S3 VWAP·볼린저 회귀": "vwap_bollinger_reversion",
             "S4 Connors RSI2": "connors_rsi2", "S5 돌파+거래량+델타": "breakout_volume_delta", "S6 캔들 반전": "candle_reversal_at_extreme",
             "S7 CVD 다이버전스": "cvd_divergence", "S8 BTC 선행": "btc_leads_alt", "S9 Supertrend 전환": "supertrend_flip", "S10 전일 레벨 돌파": "prev_day_level_break"}


# ───────────── 입력·표본 ─────────────
def rich_state(f_row: pd.Series, active: dict, tf: str) -> dict:
    """특징 값(소수 3자리, NaN 생략, 'f_' 접두 제거) + 활성 셋업(이름: long/short). 식별 정보 없음."""
    return {"timeframe": tf, "features": {k[2:]: round(float(v), 3) for k, v in f_row.items() if not np.isnan(v)}, "active_setups": active}


def make_samples(tf: str, ex=None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """OOS 구간(뒤 365일)에서 코인당 무작위 N_PER_COIN 개. 반환 (표본 표, key → 상태)."""
    panels, t0 = build(tf, 2, ex)
    oos, rows, states = halves(t0)["OOS"], [], {}
    for ci, c in enumerate(COINS):
        p = panels[c]
        ts = p["df"]["ts"].to_numpy()
        ok = np.flatnonzero((np.arange(len(ts)) >= WARM) & oos(ts) & p["fwd"][3].notna().to_numpy() & p["fwd"][12].notna().to_numpy())
        for i in np.sort(np.random.default_rng(seed + ci).choice(ok, N_PER_COIN, replace=False)):
            key = f"{tf}|{c}|{int(ts[i])}"
            active = {SETUP_IDS[n]: ("long" if s.iloc[i] > 0 else "short") for n, s in p["sig"].items() if s.iloc[i] != 0}
            states[key] = rich_state(p["F"].iloc[i], active, tf)
            F = p["F"].iloc[i]
            rows.append({"key": key, "tf": tf, "coin": c, "ts": int(ts[i]), "fwd3": float(p["fwd"][3].iloc[i]), "fwd12": float(p["fwd"][12].iloc[i]),
                         **{k: float(F[f"f_{k}"]) for k in ("rsi14", "roc6", "cci20", "bb_pctb", "roc3")}})
    return pd.DataFrame(rows), states


# ───────────── 호출 ─────────────
def questions() -> dict:
    from typesafe_sdk import Noul
    q = {"side": QUESTIONS["side"]}
    for h in H_LIST:
        q[f"up{h}"] = Noul(instructions=f"The close price {h} candles from now will be more than {THRESH}% higher than the current close.")
        q[f"down{h}"] = Noul(instructions=f"The close price {h} candles from now will be more than {THRESH}% lower than the current close.")
    return q


def call(client, q: dict, state: dict) -> dict | None:
    from typesafe_sdk import TypeSafeError
    try:
        a = client.system_one(state=state, questions=q).answers
        p = a["side"].probabilities
        return {"side": a["side"].choice, "p_long": p.get("long", 0.0), "p_short": p.get("short", 0.0), "conf": a["side"].confidence,
                **{k: a[k].noul for k in q if k != "side"}}
    except (TypeSafeError, KeyError):
        return None


def run(tfs, workers: int = 4) -> None:
    from typesafe_sdk import TypeSafeClient
    ex, client, q = make_exchange(), TypeSafeClient(), questions()
    DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DIR / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for tf in tfs:
        samples, states = make_samples(tf, ex)
        samples.to_csv(DIR / f"samples_{tf}.csv", index=False)
        todo = [(k, s) for k, s in states.items() if k not in cache]
        print(f"[{tf}] 표본 {len(states)}개 중 신규 호출 {len(todo)}회 (상태 크기 예: {len(json.dumps(todo[0][1])) if todo else 0}자)", flush=True)
        t0, fails = time.time(), 0
        with ThreadPoolExecutor(workers) as pool:
            for n, (k, r) in enumerate(pool.map(lambda kv: (kv[0], call(client, q, kv[1])), todo), 1):
                if r is None:
                    fails += 1
                else:
                    cache[k] = r
                if n % 1000 == 0:
                    cache_path.write_text(json.dumps(cache))
                    print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s, 실패 {fails})", flush=True)
        cache_path.write_text(json.dumps(cache))
        print(f"[{tf}] 완료 {time.time() - t0:.0f}s, 실패 {fails}건", flush=True)


# ───────────── 분석 ─────────────
def extreme_decile(T: pd.DataFrame, score: str, ycol: str, sign: int = 1) -> dict:
    """코인 내 점수 상·하위 10% 거래 (상위=롱, 하위=숏; sign=−1 이면 반대). 거래당 총이익·순기대수익(왕복 비용 차감)과 일 군집 SE 단측 p."""
    rets, days, longs, ldays = [], [], [], []
    for _, g in T.groupby("coin"):
        s = g[score].to_numpy()
        lo, hi = np.quantile(s, 0.1), np.quantile(s, 0.9)
        for mask, d in ((s > hi, sign), (s < lo, -sign)):
            r, dd = g[ycol].to_numpy()[mask], (g["ts"].to_numpy() // DAY_MS)[mask]
            rets += list(d * r)
            days += list(dd)
            if d == 1:
                longs += list(r)
                ldays += list(dd)
    out = {}
    for name, (v, dd) in (("ls", (rets, days)), ("long", (longs, ldays))):
        m, se, n = cluster_mean(v, dd)
        z = (m - COST) / se if se and se > 0 else np.nan
        out[name] = {"n": n, "gross": m, "net": m - COST, "p": norm_p1(z) if not np.isnan(z) else np.nan}
    return out


def mean_ic(T: pd.DataFrame, score: str, ycol: str) -> float:
    return float(np.mean([rank_corr(g[score], g[ycol]) for _, g in T.groupby("coin")]))


def delta_ic(T: pd.DataFrame, a: str, b: str, ycol: str, B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """평균 IC(b) − 평균 IC(a) 의 짝지은 부트스트랩 (같은 행 재추출). 반환 (관측, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    arrs = [(g[a].to_numpy(), g[b].to_numpy(), g[ycol].to_numpy()) for _, g in T.groupby("coin")]
    out = np.empty(B)
    for k in range(B):
        d = []
        for sa, sb, y in arrs:
            i = rng.integers(0, len(y), len(y))
            d.append(rank_corr(sb[i], y[i]) - rank_corr(sa[i], y[i]))
        out[k] = np.mean(d)
    return mean_ic(T, b, ycol) - mean_ic(T, a, ycol), *np.percentile(out, [2.5, 97.5])


def add_scores(T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    for h in H_LIST:
        T[f"s_move{h}"] = T[f"up{h}"] - T[f"down{h}"]
    T["s_side"] = T["p_long"] - T["p_short"]
    r = lambda col: T.groupby("coin")[col].rank(pct=True)  # 코인 내 순위 (표본 안)
    T["b_rsi"] = -r("rsi14")
    T["b_reversal"] = (-r("roc6") - r("rsi14") - r("cci20") - r("bb_pctb")) / 4
    T["b_mom3"] = T["roc3"]
    return T


def analyze() -> None:
    cache = json.loads((DIR / "cache.json").read_text())
    for tf in TFS:
        S = pd.read_csv(DIR / f"samples_{tf}.csv")
        R = pd.DataFrame([{"key": k, **v} for k, v in cache.items() if k.startswith(tf + "|")])
        T = add_scores(S.merge(R, on="key"))
        print(f"\n{'=' * 78}\n[{tf}] 표본 {len(T)}개 (호출 표본 {len(S)}개 중; 코인별 {T.groupby('coin').size().to_dict()})\n{'=' * 78}")
        print(f"전방 수익률: 3봉 σ {T.fwd3.std():.3f}% / 12봉 σ {T.fwd12.std():.3f}% | 0.28% 초과 상승 빈도 3봉 {(T.fwd3 > THRESH).mean() * 100:.1f}% 12봉 {(T.fwd12 > THRESH).mean() * 100:.1f}%, 하락 3봉 {(T.fwd3 < -THRESH).mean() * 100:.1f}% 12봉 {(T.fwd12 < -THRESH).mean() * 100:.1f}%")
        print(f"Jev 응답 평균: up3 {T.up3.mean():.3f} down3 {T.down3.mean():.3f} up12 {T.up12.mean():.3f} down12 {T.down12.mean():.3f}  (5~95% up12 [{T.up12.quantile(.05):.2f}, {T.up12.quantile(.95):.2f}])")
        print(f"\n[IC] 점수 vs 전방 수익률 (코인 내 Spearman 평균)")
        print(f"  {'점수':<22}{'h=3':>9}{'h=12':>9}")
        for name, col in (("Jev 크기(up−down) 3", "s_move3"), ("Jev 크기(up−down) 12", "s_move12"), ("Jev side(P롱−P숏)", "s_side"),
                          ("기준선 −RSI14", "b_rsi"), ("기준선 반전 합성", "b_reversal"), ("기준선 직전3봉 모멘텀", "b_mom3")):
            print(f"  {name:<22}{mean_ic(T, col, 'fwd3'):+9.4f}{mean_ic(T, col, 'fwd12'):+9.4f}")
        print(f"\n[★1차] 극단 10% 거래(h=12, 상위=롱·하위=숏) 거래당 기대수익 % (왕복 {COST}% 차감, 일 군집 SE 단측 p)")
        print(f"  {'점수':<22}{'n':>6}{'총이익':>9}{'순기대수익':>11}{'p':>8}   | 롱 전용 n / 순기대")
        for name, col in (("Jev s_move12", "s_move12"), ("Jev s_side", "s_side"), ("기준선 반전 합성", "b_reversal"), ("기준선 −RSI14", "b_rsi")):
            for sign, tag in ((1, ""), (-1, " (부호 반전, 참고)")):
                e = extreme_decile(T, col, "fwd12", sign)
                print(f"  {name + tag:<22}{e['ls']['n']:6d}{e['ls']['gross']:+9.4f}{e['ls']['net']:+11.4f}{e['ls']['p']:8.3f}   | {e['long']['n']} / {e['long']['net']:+.4f}")
        print(f"\n[2차] h=3 극단 10% (Jev s_move3): " + "  ".join(f"{k} 총이익 {v['gross']:+.4f} 순 {v['net']:+.4f}" for k, v in extreme_decile(T, 's_move3', 'fwd3').items() if k == 'ls'))
        for base in ("b_reversal", "b_rsi"):
            d, lo, hi = delta_ic(T, base, "s_move12", "fwd12")
            print(f"  ΔIC(h=12) Jev s_move12 − {base}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        for h in H_LIST:  # 보정: 예측 평균 vs 실현 빈도 + 순위 상관
            print(f"  보정 h={h}: up 예측 {T[f'up{h}'].mean():.3f} vs 실현 {(T[f'fwd{h}'] > THRESH).mean():.3f} (순위상관 {rank_corr(T[f'up{h}'], (T[f'fwd{h}'] > THRESH).astype(float)):+.3f}) | "
                  f"down 예측 {T[f'down{h}'].mean():.3f} vs 실현 {(T[f'fwd{h}'] < -THRESH).mean():.3f} (순위상관 {rank_corr(T[f'down{h}'], (T[f'fwd{h}'] < -THRESH).astype(float)):+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--tfs", nargs="+", default=list(TFS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.tfs, a.workers) if a.cmd == "run" else analyze()
