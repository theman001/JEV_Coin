"""백테스트 하네스 — 사전 고정 설계 (STRATEGY_RESEARCH.md §7). 결과를 보기 전에 정한 것이며, 결과를 보고 바꾸지 않는다.

체결   : 신호는 t 봉 마감, 체결은 t+1 봉 시가. 진입·청산마다 편도 비용(기본 COST_PER_SIDE=0.07%).
데이터 : 5코인 × 1h/4h/1d × 최근 5년. 규칙 파라미터는 문헌 통상값 고정(튜닝 없음). 롱 전용.
비교   : 단순 보유(B&H) / 규칙만 / 규칙 + Jev 필터.
검정   : (1) 규칙 25개(5규칙×5코인)의 "노출 보정 초과수익"(= 전략수익 − 시장노출비율 × B&H)을 White Reality Check(정상 부트스트랩)로 다중검정
         (2) 표본 내(전반부) 최적 규칙 → 표본 외(후반부) 성과 (Hudson & Urquhart Table 9 방식)
         (3) Jev 필터의 1차 지표 = 진입별 Noul 값과 그 거래의 순수익률의 Spearman 상관(순열검정). 보조: 임계값(0.5, 고정) 기준 통과/차단 거래의 평균 수익.
Jev    : 진입 신호 시점에만 Noul 1회 호출. 상태에서 코인명·날짜를 가린다(모델 학습 데이터에 이 기간 가격이 있을 수 있어 기억에 의한 누수 방지).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from .data import make_exchange
from .policy import COST_PER_SIDE
from .strategies import RULES

DATA = Path("data")
BPY = {"1h": 8760, "4h": 2190, "1d": 365}  # 연간 봉 수 (24/7)
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
JEV_THRESHOLD = 0.5  # 사전 고정. 조정 금지


# ───────────── 데이터 ─────────────
def load(sym: str, tf: str, years: int = 5, ex=None) -> pd.DataFrame:
    path = DATA / "candles" / f"{sym.replace('/', '_')}_{tf}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 12 * 3600:  # 12시간 캐시
        return pd.read_csv(path)
    ex = ex or make_exchange()
    ms, since, rows = ex.parse_timeframe(tf) * 1000, int(time.time() * 1000 - years * 365 * 86_400_000), []
    while True:
        r = ex.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not r:
            break
        rows += r
        since = r[-1][0] + ms
        if len(r) < 1000:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts").iloc[:-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ───────────── 시뮬레이션 ─────────────
def simulate(df: pd.DataFrame, pos, cost: float = COST_PER_SIDE) -> pd.DataFrame:
    """pos[t] = 봉 t 마감 시점 신호(0/1) → 봉 t+1 시가에 체결. 반환(인덱스=봉 시작 ts): ret(순수익률, 시가→다음 시가), pos(체결 포지션), oo(총수익률)."""
    o = df["open"].to_numpy(float)
    oo = np.r_[o[1:] / o[:-1] - 1, np.nan]
    ep = np.r_[0.0, np.asarray(pos, float)[:-1]]  # ep[t] = pos[t-1]: 봉 t 시가에 보유 중인 포지션
    trade = np.abs(np.diff(np.r_[0.0, ep]))  # 포지션이 바뀔 때마다 편도 비용
    ret = ep * oo - trade * cost
    out = pd.DataFrame({"ret": ret, "pos": ep, "oo": oo}, index=df["ts"].to_numpy()).iloc[:-1].copy()
    out.iloc[-1, 0] -= cost * out["pos"].iloc[-1]  # 끝까지 들고 있으면 마지막에 청산한다고 보고 비용 반영 (B&H 와 동일 취급)
    return out


def stats(r: pd.DataFrame, bpy: int) -> dict:
    ret, n = r["ret"], len(r)
    eq, years, sd = (1 + ret).cumprod(), n / bpy, r["ret"].std()
    entries_ = int((r["pos"].diff().fillna(r["pos"]) == 1).sum())
    return {"연수익%": (eq.iloc[-1] ** (1 / years) - 1) * 100, "샤프": ret.mean() / sd * np.sqrt(bpy) if sd > 0 else np.nan,
            "MDD%": (eq / eq.cummax() - 1).min() * 100, "연진입": entries_ / years, "노출%": r["pos"].mean() * 100}


def entries(pos) -> np.ndarray:
    p = np.asarray(pos)
    return np.flatnonzero((p == 1) & (np.r_[0, p[:-1]] == 0))


def trades(df: pd.DataFrame, pos, cost: float = COST_PER_SIDE) -> list[tuple[int, float]]:
    """규칙 단독 거래 [(진입 신호 봉 t, 순수익률)]: 진입 t+1 시가 → 첫 청산 신호 봉 u 의 다음 봉 시가(끝이면 마지막 시가)."""
    o, p, out = df["open"].to_numpy(float), np.asarray(pos), []
    n = len(p)
    for t in entries(pos):
        if t + 1 >= n:
            continue
        u = t + 1
        while u < n and p[u] == 1:
            u += 1
        out.append((int(t), o[min(u + 1, n - 1)] / o[t + 1] - 1 - 2 * cost))
    return out


def apply_filter(pos, allow: dict) -> np.ndarray:
    """진입 신호마다 allow[t] 가 False 면 그 포지션(다음 청산 신호까지)을 통째로 건너뛴다. 청산 후 다음 진입은 새로 판정."""
    p, out, blocked = np.asarray(pos), np.zeros(len(pos), int), False
    for t in range(len(p)):
        if p[t] == 0:
            blocked = False
        else:
            if t == 0 or p[t - 1] == 0:
                blocked = not allow.get(t, True)
            out[t] = 0 if blocked else 1
    return out


# ───────────── 통계 ─────────────
def reality_check(D: np.ndarray, B: int | None = None, seed: int = 0) -> float:
    """White(2000) Reality Check: H0 = 어떤 전략도 평균 초과수익이 없다. D[n,K] = 초과수익. 정상 부트스트랩(평균 블록 n^(1/3)), 반환 p 값."""
    n, K = D.shape
    B = B or (1000 if n < 20_000 else 400)
    L, rng, mean = max(2, round(n ** (1 / 3))), np.random.default_rng(seed), D.mean(0)
    v_obs, v = np.sqrt(n) * mean.max(), np.empty(B)
    for b in range(B):
        restart = rng.random(n) < 1 / L
        restart[0] = True
        starts, pos_ = rng.integers(0, n, n), np.flatnonzero(restart)
        seg = np.cumsum(restart) - 1
        idx = (starts[pos_][seg] + np.arange(n) - pos_[seg]) % n
        v[b] = np.sqrt(n) * (D[idx].mean(0) - mean).max()
    return float((1 + (v >= v_obs).sum()) / (B + 1))


def spearman_perm(x, y, B: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Spearman 상관과 양측 순열검정 p 값."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho, rng = np.corrcoef(rx, ry)[0, 1], np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(B)])
    return float(rho), float((1 + (np.abs(perm) >= abs(rho)).sum()) / (B + 1))


# ───────────── Jev 진입 필터 ─────────────
def state_frame(df: pd.DataFrame) -> pd.DataFrame:
    """t 시점 특징 (모두 인과적: 재귀형 지표·rolling 만 사용). 봉 주기와 무관하도록 ATR 단위로 정규화."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    atr = AverageTrueRange(h, l, c, 14).average_true_range() / c * 100
    rng = h.rolling(20).max() - l.rolling(20).min()
    return pd.DataFrame({
        "close": c, "ema20": EMAIndicator(c, 20).ema_indicator(), "ema50": EMAIndicator(c, 50).ema_indicator(),
        "rsi": RSIIndicator(c, 14).rsi(), "atr": atr, "vol_ratio": atr / atr.rolling(100).median(),
        "volu_ratio": v / v.rolling(20).median().shift(1), "ret5": (c / c.shift(5) - 1) * 100,
        "range_pos": (c - l.rolling(20).min()) / rng.where(rng > 0)})


def _b(x, cuts, names):
    return names[int(np.searchsorted(cuts, x, side="right"))]


def label(row, rule: str, tf: str) -> dict | None:
    """마스킹된 상태 (코인명·날짜 없음). 특징이 NaN(워밍업)이면 None → 필터 미적용."""
    if row.isna().any() or row["atr"] <= 0:
        return None
    a, c = row["atr"], row["close"]
    return {
        "signal": f"{rule} long entry just triggered", "timeframe": tf,
        "trend": "up" if c > row["ema20"] > row["ema50"] else "down" if c < row["ema20"] < row["ema50"] else "sideways",
        "rsi": round(float(row["rsi"]), 1),
        "rsi_zone": "overbought" if row["rsi"] >= 70 else "oversold" if row["rsi"] <= 30 else "neutral",
        "recent_move_5_bars": _b(row["ret5"] / a, [-2, -0.5, 0.5, 2], ["strong down", "down", "flat", "up", "strong up"]),
        "extension_from_ema50": _b((c - row["ema50"]) / c * 100 / a, [-2, -1, 1, 2], ["far below", "below", "near", "above", "far above"]),
        "volatility_regime": _b(row["vol_ratio"], [0.7, 1.5], ["low", "normal", "high"]),
        "volume": _b(row["volu_ratio"], [0.5, 2.0], ["thin", "normal", "spike"]),
        "position_in_20_bar_range": _b(row["range_pos"], [1 / 3, 2 / 3], ["lower third", "middle", "upper third"]),
    }


def ask_jev(client, items: list, workers: int = 4) -> dict:
    """items = [(key, state)] → {key: noul 또는 None(실패)}. 레이트 리밋(1,200/분) 안에서 병렬."""
    from typesafe_sdk import Noul, TypeSafeError
    q = {"entry": Noul(instructions="After this long entry signal, the price is likely to be higher over the next several candles.")}

    def one(kv):
        try:
            return kv[0], client.system_one(state=kv[1], questions=q).answers["entry"].noul
        except (TypeSafeError, KeyError):
            return kv[0], None
    done, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (k, v) in enumerate(ex.map(one, items), 1):
            done[k] = v
            if i % 500 == 0:
                print(f"    Jev {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    return done


def allow_map(df, pos, sym: str, tf: str, rule: str, cache: dict) -> dict:
    """진입 봉 t → Jev Noul ≥ 임계값 여부. 캐시에 없는 진입(워밍업·호출 실패)은 판정하지 않는다 = 필터 미적용."""
    out = {}
    for t in entries(pos):
        v = cache.get(f"{sym}|{tf}|{rule}|{int(df.ts.iloc[t])}")
        if v is not None:
            out[int(t)] = v >= JEV_THRESHOLD
    return out


# ───────────── 실행 ─────────────
def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.2f}")


def ew(series: dict) -> pd.Series:
    """코인별 수익률 → 동일가중 포트폴리오 (공통 시각만)."""
    return pd.concat(series, axis=1).dropna().mean(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfs", nargs="+", default=["1h", "4h", "1d"])
    ap.add_argument("--jev-tfs", nargs="*", default=["4h", "1d"], help="Jev 필터를 평가할 주기 (빈 값이면 생략)")
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--years", type=int, default=5)
    a = ap.parse_args(argv)
    ex, out = make_exchange(), {}
    client = None
    if set(a.tfs) & set(a.jev_tfs):
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()
    cache_path = DATA / "backtest" / "jev_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    for tf in a.tfs:
        bpy = BPY[tf]
        data = {s: load(s, tf, a.years, ex) for s in a.coins}
        sims = {}  # (coin, rule) -> DataFrame
        for s, df in data.items():
            sims[(s, "B&H")] = simulate(df, np.ones(len(df)))
            for name, fn in RULES.items():
                sims[(s, name)] = simulate(df, fn(df).to_numpy())
        span = {s: (pd.to_datetime(df.ts.iloc[0], unit="ms").date(), pd.to_datetime(df.ts.iloc[-1], unit="ms").date(), len(df)) for s, df in data.items()}
        print(f"\n{'=' * 78}\n[{tf}] {a.years}년, 코인 {len(data)}개, 비용 편도 {COST_PER_SIDE * 100:.2f}%   기간 예: {list(span.values())[0]}\n{'=' * 78}")

        # (1) 규칙별 성과: 5코인 동일가중
        names = ["B&H"] + list(RULES)
        rows = {}
        for n_ in names:
            port = ew({s: sims[(s, n_)]["ret"] for s in data})
            pos_ = ew({s: sims[(s, n_)]["pos"] for s in data})
            rows[n_] = stats(pd.DataFrame({"ret": port, "pos": pos_}), bpy)
        # 연진입/노출은 코인 평균으로 (포트폴리오 pos 평균은 진입 횟수 의미가 없음)
        for n_ in names:
            st = [stats(sims[(s, n_)], bpy) for s in data]
            rows[n_]["연진입"], rows[n_]["노출%"] = np.mean([x["연진입"] for x in st]), np.mean([x["노출%"] for x in st])
        print(f"\n[규칙별 성과 — 5코인 동일가중, 비용 반영]\n{fmt(pd.DataFrame(rows).T)}")
        sharpe = pd.DataFrame({n_: {s: stats(sims[(s, n_)], bpy)["샤프"] for s in data} for n_ in names})
        print(f"\n[코인별 샤프]\n{fmt(sharpe)}")

        # (1') Reality Check: 규칙 25개의 노출 보정 초과수익
        cols, D = [], {}
        for s in data:
            for n_ in RULES:
                r = sims[(s, n_)]
                D[(s, n_)] = r["ret"] - r["pos"].mean() * r["oo"]  # 시장 노출만큼의 B&H 를 뺀 '타이밍' 초과수익
        Dm = pd.DataFrame(D).dropna()
        best = Dm.mean().idxmax()
        p_rc = reality_check(Dm.to_numpy())
        print(f"\n[Reality Check] 규칙 {Dm.shape[1]}개 중 최고 = {best}, 연 타이밍 초과수익 {Dm.mean()[best] * bpy * 100:+.2f}%  → 다중검정 p = {p_rc:.3f}"
              f"  ({'유의(5%)' if p_rc < 0.05 else '유의하지 않음'})")
        pos_share = (Dm.mean() > 0).mean() * 100
        print(f"                (25개 중 타이밍 초과수익이 양수인 비율 {pos_share:.0f}%)")

        # (2) 표본 내 최적 → 표본 외
        oos_rows = []
        for s in data:
            mid = data[s]["ts"].iloc[len(data[s]) // 2]
            is_sh = {n_: stats(sims[(s, n_)][sims[(s, n_)].index < mid], bpy)["샤프"] for n_ in RULES}
            b_ = max(is_sh, key=is_sh.get)
            o_ = stats(sims[(s, b_)][sims[(s, b_)].index >= mid], bpy)
            bh = stats(sims[(s, "B&H")][sims[(s, "B&H")].index >= mid], bpy)
            oos_rows.append({"코인": s, "표본내 최적": b_, "IS샤프": is_sh[b_], "OOS샤프": o_["샤프"], "OOS연수익%": o_["연수익%"], "OOS MDD%": o_["MDD%"],
                             "B&H OOS샤프": bh["샤프"], "B&H OOS연수익%": bh["연수익%"], "B&H OOS MDD%": bh["MDD%"]})
        oos = pd.DataFrame(oos_rows).set_index("코인")
        print(f"\n[표본 내 최적 규칙 → 표본 외(후반부)]\n{fmt(oos)}")
        print(f"   평균: OOS샤프 {oos['OOS샤프'].mean():.2f} vs B&H {oos['B&H OOS샤프'].mean():.2f} | OOS MDD {oos['OOS MDD%'].mean():.1f}% vs B&H {oos['B&H OOS MDD%'].mean():.1f}%")

        # (3) 연도별 (동일가중)
        yr = {}
        for n_ in names:
            port = ew({s: sims[(s, n_)]["ret"] for s in data})
            yr[n_] = ((1 + port).groupby(pd.to_datetime(port.index, unit="ms").year).prod() - 1) * 100
        print(f"\n[연도별 수익률% — 5코인 동일가중]\n{fmt(pd.DataFrame(yr).T)}")
        out[tf] = {"rules": pd.DataFrame(rows).T.to_dict(), "reality_check_p": p_rc, "oos": oos.to_dict()}

        # (4) Jev 진입 필터
        if tf in a.jev_tfs:
            reqs, meta = [], {}
            for s, df in data.items():
                sf = state_frame(df)
                for n_, fn in RULES.items():
                    pos = fn(df).to_numpy()
                    for t, tr in trades(df, pos):
                        key = f"{s}|{tf}|{n_}|{int(df.ts.iloc[t])}"
                        st = label(sf.iloc[t], n_, tf)
                        meta[key] = (s, n_, t, tr, st is not None)
                        if st is not None and key not in cache:
                            reqs.append((key, st))
            print(f"\n[Jev 진입 필터] 진입 {len(meta)}건 (워밍업 제외 {sum(m[4] for m in meta.values())}건), 신규 호출 {len(reqs)}회 (캐시 {sum(k in cache for k in meta)}건)")
            if reqs:
                t0 = time.time()
                got = ask_jev(client, reqs)
                fails = sum(v is None for v in got.values())
                cache.update({k: v for k, v in got.items() if v is not None})
                cache_path.write_text(json.dumps(cache))
                print(f"    완료 {time.time() - t0:.0f}s, 실패 {fails}건 (실패는 필터 미적용으로 처리)")
            ent = pd.DataFrame([{"coin": m[0], "rule": m[1], "t": m[2], "ret": m[3] * 100, "noul": cache.get(k)} for k, m in meta.items()])
            e = ent.dropna(subset=["noul"])
            if len(e) > 30:
                rho, p = spearman_perm(e["noul"], e["ret"])
                acc, blk = e[e["noul"] >= JEV_THRESHOLD], e[e["noul"] < JEV_THRESHOLD]
                print(f"    ★ 1차 지표: Spearman(noul, 거래 순수익률) = {rho:+.3f}  (n={len(e)}, 순열검정 p = {p:.3f}; 2개 주기 검정 → 유의수준 0.025)")
                print(f"      noul 분포: 평균 {e.noul.mean():.2f}, 5~95% [{e.noul.quantile(.05):.2f}, {e.noul.quantile(.95):.2f}]")
                print(f"      임계 {JEV_THRESHOLD} 통과 {len(acc)}건 평균 {acc.ret.mean():+.2f}% (승률 {(acc.ret > 0).mean() * 100:.0f}%)  |  차단 {len(blk)}건 평균 {blk.ret.mean():+.2f}% (승률 {(blk.ret > 0).mean() * 100:.0f}%)  |  전체 {e.ret.mean():+.2f}%")
                out[tf]["jev"] = {"rho": rho, "p": p, "n": len(e), "accepted_mean": acc.ret.mean(), "blocked_mean": blk.ret.mean()}
            # 포트폴리오: 규칙만 vs 규칙 + Jev 필터
            jr, jd = {}, {}
            for n_ in RULES:
                ret_r, ret_j = {}, {}
                for s, df in data.items():
                    pos = RULES[n_](df).to_numpy()
                    ret_r[s] = sims[(s, n_)]["ret"]
                    ret_j[s] = simulate(df, apply_filter(pos, allow_map(df, pos, s, tf, n_, cache)))["ret"]
                pr, pj = ew(ret_r), ew(ret_j)
                sr, sj = (stats(pd.DataFrame({"ret": x, "pos": 1.0}), bpy) for x in (pr, pj))  # pos 는 성과 통계에 쓰이지 않음
                jr[n_] = {"규칙만 연수익%": sr["연수익%"], "규칙+Jev 연수익%": sj["연수익%"], "규칙만 샤프": sr["샤프"], "규칙+Jev 샤프": sj["샤프"],
                          "규칙만 MDD%": sr["MDD%"], "규칙+Jev MDD%": sj["MDD%"]}
                jd[n_] = (pj - pr).to_numpy()
            print(f"\n[포트폴리오: 규칙만 vs 규칙 + Jev 필터 — 5코인 동일가중]\n{fmt(pd.DataFrame(jr).T)}")
            m = min(len(v) for v in jd.values())
            p_j = reality_check(np.column_stack([v[-m:] for v in jd.values()]))
            print(f"    Reality Check(규칙 5개 중 Jev 필터가 규칙만보다 나은 것이 하나라도 있는가): p = {p_j:.3f}")
            out[tf]["jev_portfolio"] = jr
            out[tf]["jev_rc_p"] = p_j

    res = DATA / "backtest" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    res.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(f"\n저장: {res}")


if __name__ == "__main__":
    main()
