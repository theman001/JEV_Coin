"""특징 선별(1단계) + 셋업 이벤트 스터디(2단계) — 사전 고정 설계 (STRATEGY_RESEARCH.md §12). 결과를 보고 설계를 바꾸지 않는다.

  python -m src.screen [--tfs 5m 15m 1h]      # 캔들 수집 → 특징 계산 → 1·2단계 실행 → data/screen/ 에 저장
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import DAY_MS, load_klines
from . import micro
from .data import make_exchange
from .features import FEATURES, compute
from .setups import MICRO_SETUPS, SETUPS, events

COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
TFS, HS, HS_MICRO = ("5m", "15m", "1h"), (3, 6, 12), (1, 3, 12)
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
COST, WARM, Q_FDR, ALPHA = 0.14, 300, 0.10, 0.05  # 왕복 비용 %, 워밍업 봉, BH-FDR q, 유의수준
OUT = Path("data") / "screen"


# ───────────── 통계 도구 ─────────────
def norm_p2(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def norm_p1(z: float) -> float:  # 단측 P(Z ≥ z)
    return 0.5 * math.erfc(z / math.sqrt(2))


def bh_reject(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini–Hochberg: FDR q 로 기각되는 항목의 불리언 마스크."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ok = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    out = np.zeros(len(p), bool)
    if len(ok):
        out[order[: ok.max() + 1]] = True
    return out


def cluster_mean(v, day) -> tuple[float, float, int]:
    """평균과 일 단위 군집 표준오차 (같은 날 여러 코인·이벤트의 상관을 반영). 반환 (평균, se, n)."""
    v = np.asarray(v, float)
    if len(v) < 30:
        return np.nan, np.nan, len(v)
    m = v.mean()
    s = pd.Series(v - m).groupby(np.asarray(day)).sum()
    return float(m), float(math.sqrt((s ** 2).sum()) / len(v)), len(v)


def pct_rank(x) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def rho_bar(ys: dict) -> float:
    """코인 간 전방 수익률의 평균 상관 (유효 표본 보정용). ys: coin -> Series(index=ts)."""
    D = pd.concat(ys, axis=1).dropna()
    C = D.corr().to_numpy()
    k = C.shape[0]
    return float((C.sum() - k) / (k * (k - 1))) if k > 1 else 0.0


# ───────────── 표본 구성 ─────────────
def build(tf: str, years: int = 2, ex=None, hs=HS, setups=None, with_micro: bool = False):
    """코인별 패널: (df, F, fwd{h}, sig{setup}). BTC 는 교차 자산 특징이 NaN. with_micro 면 미시구조 특징 26개를 F 에 덧붙인다. 반환 (panels, 시작 ts)."""
    setups = SETUPS if setups is None else setups
    raw = {c: load_klines(c, tf, years, ex) for c in COINS}
    panels = {}
    for c, df in raw.items():
        F = compute(df, None if c == "BTC/USDT" else raw["BTC/USDT"])
        if with_micro:  # z-점수 워밍업(펀딩 30일·깊이 7일)을 위해 창 시작보다 14일 앞부터 로드
            F = pd.concat([F, micro.compute_micro(df, micro.load_frames(c, int(df["ts"].iloc[0]) - 14 * DAY_MS), TF_MS[tf])], axis=1)
        panels[c] = {"df": df, "F": F, "fwd": {h: (df["close"].shift(-h) / df["close"] - 1) * 100 for h in hs},
                     "sig": {n: fn(df, F) for n, fn in setups.items()}}
    return panels, int(min(p["df"]["ts"].iloc[0] for p in panels.values()))


def halves(t0: int):
    mid = t0 + 365 * DAY_MS
    return {"IS": lambda ts: ts < mid, "OOS": lambda ts: ts >= mid + DAY_MS}


def subsample(p, h: int, half) -> np.ndarray:
    """h 봉 간격(겹치지 않는 전방 수익률)·워밍업 이후·해당 구간·전방 수익률 존재하는 행 위치."""
    n = len(p["df"])
    ts = p["df"]["ts"].to_numpy()
    return np.flatnonzero((np.arange(n) % h == 0) & (np.arange(n) >= WARM) & half(ts) & p["fwd"][h].notna().to_numpy())


# ───────────── 1단계: 특징 선별 ─────────────
def ic_rows(panels, h: int, half, feats=FEATURES) -> pd.DataFrame:
    """(특징)별 코인 합산 순위상관 IC 와 유효 표본 보정 p. 반환 index=특징, 열: ic, n, n_eff, p, k."""
    subs = {c: subsample(p, h, half) for c, p in panels.items()}
    rb = rho_bar({c: pd.Series(p["fwd"][h].to_numpy()[subs[c]], index=p["df"]["ts"].to_numpy()[subs[c]]) for c, p in panels.items()})
    rows = {}
    for f in feats:
        xs, ys = [], []
        for c, p in panels.items():
            x, y = p["F"][f].to_numpy()[subs[c]], p["fwd"][h].to_numpy()[subs[c]]
            m = ~np.isnan(x)
            if m.sum() > 50 and np.nanstd(x[m]) > 0:
                xs.append(pct_rank(x[m]))
                ys.append(pct_rank(y[m]))
        if not xs:
            continue  # 이 특징은 해당 구간에 유효한 표본이 없음 (예: ±0.2% 밴드는 2026-01-15 이후만 존재)
        X, Y, k = np.concatenate(xs), np.concatenate(ys), len(xs)
        ic = float(np.corrcoef(X, Y)[0, 1])
        n_eff = len(X) / (1 + (k - 1) * max(rb, 0.0))
        t = ic * math.sqrt(max(n_eff - 2, 1) / max(1 - ic ** 2, 1e-12))
        rows[f] = {"ic": ic, "n": len(X), "n_eff": n_eff, "p": norm_p2(t), "k": k}
    return pd.DataFrame(rows, index=["ic", "n", "n_eff", "p", "k"]).T if rows else pd.DataFrame(columns=["ic", "n", "n_eff", "p", "k"], dtype=float)


def extreme_expectancy(panels, f: str, h: int, sign: int, is_half, oos_half) -> dict:
    """IS 분위수(상·하위 10%, 엄격 부등호)로 정한 극단 구간을 OOS 에서 거래. 방향 = IS IC 부호. 순기대수익은 왕복 비용 차감."""
    rets, days, longs, ldays = [], [], [], []
    for c, p in panels.items():
        x_is = p["F"][f].to_numpy()[subsample(p, h, is_half)]
        x_is = x_is[~np.isnan(x_is)]
        if len(x_is) < 200:
            continue
        q10, q90 = np.quantile(x_is, 0.1), np.quantile(x_is, 0.9)
        idx = subsample(p, h, oos_half)
        x, y, ts = p["F"][f].to_numpy()[idx], p["fwd"][h].to_numpy()[idx], p["df"]["ts"].to_numpy()[idx]
        for mask, d in ((x > q90, sign), (x < q10, -sign)):
            mask = mask & ~np.isnan(x)
            rets += list(d * y[mask])
            days += list(ts[mask] // DAY_MS)
            if d == 1:
                longs += list(y[mask])
                ldays += list(ts[mask] // DAY_MS)
    out = {}
    for name, (v, dd) in (("ls", (rets, days)), ("long", (longs, ldays))):
        m, se, n = cluster_mean(v, dd)
        net = m - COST
        out[name] = {"n": n, "gross": m, "net": net, "z": net / se if se and se > 0 else np.nan}
        out[name]["p"] = norm_p1(out[name]["z"]) if not np.isnan(out[name]["z"]) else np.nan
    return out


def stage1(panels, t0: int, tf: str, feats=FEATURES, hs=HS) -> pd.DataFrame:
    hv = halves(t0)
    rows = []
    for h in hs:
        is_, oos = ic_rows(panels, h, hv["IS"], feats), ic_rows(panels, h, hv["OOS"], feats)
        if is_.empty:
            continue
        cand = bh_reject(is_["p"].to_numpy(), Q_FDR)
        for i, f in enumerate(is_.index):
            r = {"tf": tf, "h": h, "feature": f, "ic_is": is_.loc[f, "ic"], "p_is": is_.loc[f, "p"], "cand": bool(cand[i])}
            if f in oos.index:
                r |= {"ic_oos": oos.loc[f, "ic"], "p_oos": oos.loc[f, "p"]}
            rows.append(r)
    return pd.DataFrame(rows, columns=["tf", "h", "feature", "ic_is", "p_is", "cand", "ic_oos", "p_oos"]) if not rows else pd.DataFrame(rows)


def stage1_oos(panels, t0: int, table: pd.DataFrame) -> pd.DataFrame:
    """IS 후보의 극단 구간 OOS 기대수익 통계 (판정은 finalize — 전체 후보 수가 정해져야 Bonferroni α 를 알 수 있다)."""
    hv, out = halves(t0), []
    for _, r in table[table["cand"]].iterrows():
        e = extreme_expectancy(panels, r["feature"], int(r["h"]), int(np.sign(r["ic_is"])), hv["IS"], hv["OOS"])
        out.append({**r.to_dict(), **{f"{k}_{m}": v for k, d in e.items() for m, v in d.items()}})
    return pd.DataFrame(out)


def finalize(conf: pd.DataFrame, n_cand_total: int) -> pd.DataFrame:
    """OOS 확인: IS 와 같은 부호 + IC p < α/후보수 + 극단 구간 OOS 순기대수익 > 0 (같은 α, 일 군집 SE)."""
    if conf.empty:
        return conf
    alpha = ALPHA / max(n_cand_total, 1)
    conf = conf.copy()
    conf["alpha"] = alpha
    conf["ic_confirmed"] = (np.sign(conf["ic_oos"]) == np.sign(conf["ic_is"])) & (conf["p_oos"] < alpha)
    conf["validated"] = conf["ic_confirmed"] & (conf["ls_p"] < alpha) & (conf["ls_net"] > 0)
    return conf


# ───────────── 2단계: 셋업 이벤트 스터디 ─────────────
def stage2(panels, t0: int, tf: str, hs=HS) -> pd.DataFrame:
    hv, rows = halves(t0), []
    for name in panels[COINS[0]]["sig"]:
        for h in hs:
            row = {"tf": tf, "setup": name, "h": h}
            for hn, half in hv.items():
                g, gl, dd, dl = [], [], [], []
                for c, p in panels.items():
                    ev = events(p["sig"][name], h)
                    ts = p["df"]["ts"].to_numpy()
                    ev = ev[(ev >= WARM) & half(ts[ev])]
                    y = p["fwd"][h].to_numpy()[ev]
                    s = p["sig"][name].to_numpy()[ev]
                    ok = ~np.isnan(y)
                    g += list((s * y)[ok]); dd += list(ts[ev][ok] // DAY_MS)
                    gl += list(y[ok & (s == 1)]); dl += list(ts[ev][ok & (s == 1)] // DAY_MS)
                for nm, (v, d_) in (("ls", (g, dd)), ("long", (gl, dl))):
                    m, se, n = cluster_mean(v, d_)
                    net = m - COST
                    z = net / se if se and se > 0 else np.nan
                    row |= {f"{hn}_{nm}_n": n, f"{hn}_{nm}_gross": m, f"{hn}_{nm}_net": net, f"{hn}_{nm}_p": norm_p1(z) if not np.isnan(z) else np.nan,
                            f"{hn}_{nm}_hit": float(np.mean(np.asarray(v) > 0)) if len(v) else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


# ───────────── 실행 ─────────────
def placebo(panels, t0: int, tf: str, feats, hs, shifts=None) -> list[int]:
    """전방 수익률을 시간축으로 밀어 특징과의 대응을 끊고 1단계를 다시 실행 → IS 후보 수(거의 0 이어야 한다 = 유의성이 과대평가되지 않음)."""
    orig = {c: {h: p["fwd"][h].copy() for h in hs} for c, p in panels.items()}
    n = len(panels[COINS[0]]["df"])
    out = []
    for off in shifts or (n // 20, n // 10, n // 5):  # 데이터 길이의 5%·10%·20%: 자기상관 시간 척도(수 시간~일)보다 훨씬 길면서 IS(앞 절반)에 유효 표본이 남는다
        for c, p in panels.items():
            for h in hs:
                p["fwd"][h] = orig[c][h].shift(off)
        t1 = stage1(panels, t0, tf, feats, hs)
        if t1.empty:  # 밀기 폭이 데이터보다 커서 검정 자체가 사라짐 → '후보 0개'를 유의성 보정 통과로 오해하지 않게 오류로 알린다
            raise ValueError(f"placebo shift {off} leaves no valid rows (n={n})")
        out.append(int(t1["cand"].sum()))
    for c, p in panels.items():
        p["fwd"] = orig[c]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfs", nargs="+", default=list(TFS))
    ap.add_argument("--micro", action="store_true", help="미시구조 모드: 특징 26개(§13), 규칙 9개, h∈{1,3,12}, 플라시보 점검")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # 전부 결측인 열의 통계 계산 경고
    hs, feats, setups, tag = (HS_MICRO, micro.MICRO_FEATURES, MICRO_SETUPS, "micro_") if a.micro else (HS, FEATURES, SETUPS, "")
    ex, s1, s1o, s2, plc = make_exchange(), [], [], [], {}
    for tf in a.tfs:  # 주기마다 패널을 만들고 쓰고 버린다 (5m 패널은 수백 MB)
        t = time.time()
        panels, t0 = build(tf, 2, ex, hs, setups, a.micro)
        print(f"[{tf}] 패널 구성 {time.time() - t:.0f}s | 코인당 봉 수 {len(panels['BTC/USDT']['df'])} | 특징 {len(feats)}개", flush=True)
        t1 = stage1(panels, t0, tf, feats, hs)
        s1.append(t1)
        s1o.append(stage1_oos(panels, t0, t1))
        s2.append(stage2(panels, t0, tf, hs))
        if a.micro:
            plc[tf] = placebo(panels, t0, tf, feats, hs)
        del panels
        pd.concat(s1, ignore_index=True).to_csv(OUT / f"{tag}partial_stage1_ic.csv", index=False)  # 중간 저장: 나중 주기에서 죽어도 앞 결과는 남는다
        pd.concat(s1o, ignore_index=True).to_csv(OUT / f"{tag}partial_stage1_oos.csv", index=False)
        pd.concat(s2, ignore_index=True).to_csv(OUT / f"{tag}partial_stage2_setups.csv", index=False)
        print(f"[{tf}] 1·2단계 계산 완료 ({time.time() - t:.0f}s)" + (f" | 플라시보 IS 후보 {plc[tf]}" if a.micro else ""), flush=True)
    T1, T2 = pd.concat(s1, ignore_index=True), pd.concat(s2, ignore_index=True)
    n_cand = int(T1["cand"].sum())
    conf = finalize(pd.concat(s1o, ignore_index=True), n_cand)
    T1.to_csv(OUT / f"{tag}stage1_ic.csv", index=False)
    conf.to_csv(OUT / f"{tag}stage1_confirm.csv", index=False)
    T2.to_csv(OUT / f"{tag}stage2_setups.csv", index=False)
    report(T1, conf, T2, n_cand, plc)


def report(T1, conf, T2, n_cand, plc=None):
    print(f"\n{'=' * 78}\n[1단계] 특징 선별 — IS 후보(BH-FDR q={Q_FDR}) {n_cand}개 / 전체 {len(T1)}개 검정\n{'=' * 78}")
    print(T1.groupby(["tf", "h"])["cand"].sum().unstack().to_string())
    if plc:
        print(f"플라시보(전방 수익률을 시간축으로 밀어 대응 끊기, 밀기 3종) IS 후보 수: {plc}  ← 거의 0 이어야 유의성이 과대평가되지 않은 것")
    if n_cand:
        print(f"\n후보 → OOS 확인 (Bonferroni α = {ALPHA}/{n_cand} = {ALPHA / n_cand:.5f}): IC 확인 {int(conf['ic_confirmed'].sum())}개, "
              f"극단 구간 순기대수익까지 통과(검증됨) {int(conf['validated'].sum())}개")
        show = conf.sort_values("p_is").head(15)[["tf", "h", "feature", "ic_is", "ic_oos", "p_oos", "ic_confirmed", "ls_n", "ls_gross", "ls_net", "ls_p", "long_net", "validated"]]
        print(show.to_string(float_format=lambda x: f"{x:.4f}", index=False))
        if conf["validated"].any():
            print("\n★ 검증된 특징:\n" + conf[conf["validated"]][["tf", "h", "feature", "ic_is", "ic_oos", "ls_n", "ls_gross", "ls_net", "long_n", "long_net"]].to_string(float_format=lambda x: f"{x:.4f}", index=False))
        else:
            print("\n★ 검증된 특징: 없음")
    print(f"\n{'=' * 78}\n[2단계] 셋업 이벤트 스터디 — OOS 구간, 왕복 비용 {COST}% 차감, Bonferroni α = {ALPHA}/{len(T2)} = {ALPHA / len(T2):.5f}\n{'=' * 78}")
    a2 = ALPHA / len(T2)
    T2 = T2.assign(sig_ls=(T2["OOS_ls_p"] < a2) & (T2["OOS_ls_net"] > 0), sig_long=(T2["OOS_long_p"] < a2) & (T2["OOS_long_net"] > 0))
    cols = ["tf", "setup", "h", "IS_ls_n", "IS_ls_gross", "OOS_ls_n", "OOS_ls_gross", "OOS_ls_net", "OOS_ls_p", "OOS_ls_hit", "OOS_long_n", "OOS_long_net", "sig_ls"]
    print(T2.sort_values("OOS_ls_net", ascending=False).head(15)[cols].to_string(float_format=lambda x: f"{x:.4f}", index=False))
    print(f"\n유의(방향 합산) {int(T2['sig_ls'].sum())}개, 유의(롱 전용) {int(T2['sig_long'].sum())}개 / {len(T2)}개. "
          f"OOS 총이익(비용 전) 평균이 양수인 조합 {(T2['OOS_ls_gross'] > 0).mean() * 100:.0f}%, 비용 후 양수인 조합 {(T2['OOS_ls_net'] > 0).mean() * 100:.0f}%")
    print(f"비용 전 총이익 중앙값 {T2['OOS_ls_gross'].median():+.4f}% (왕복 비용 {COST}%), 최대 {T2['OOS_ls_gross'].max():+.4f}%")


if __name__ == "__main__":
    main()
