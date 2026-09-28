"""Jev 입력 비교 실험 검증. 네트워크·API 키 불필요 (Jev SDK 는 MockTransport)."""
import json

import httpx2
import numpy as np
import pandas as pd
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

import src.jev_replay as jr


def candles(n=400, seed=0, base=65_432.1):
    r = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(r.normal(0, 0.002, n)))
    o = np.r_[base, close[:-1]]
    return pd.DataFrame({"ts": 1_700_000_000_000 + np.arange(n) * 300_000, "open": o, "high": np.maximum(o, close) * 1.001,
                         "low": np.minimum(o, close) * 0.999, "close": close, "volume": r.uniform(4_000, 6_000, n)})


def numbers(x):
    if isinstance(x, dict):
        return [n for v in x.values() for n in numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in numbers(v)]
    return [float(x)] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


# ── 입력 구성: 마스킹·인과성 ──
def test_variants_contain_the_right_parts_and_no_absolute_prices():
    df, i = candles(), 300
    lab, raw, both = (jr.make_state(df, i, v) for v in jr.VARIANTS)
    assert "trend" in lab and "recent_candles_oldest_to_newest" not in lab  # 라벨만
    assert "trend" not in raw and "rsi" not in raw and len(raw["recent_candles_oldest_to_newest"]["close"]) == jr.WINDOW  # 원시만
    assert "trend" in both and "recent_candles_oldest_to_newest" in both  # 둘 다
    blk = both["recent_candles_oldest_to_newest"]
    assert blk["close"][-1] == 100.0 and 90 < min(blk["low"]) and max(blk["high"]) < 110  # 마지막 종가 = 100 으로 재기준화
    assert max(numbers(raw)) < 1000 and max(numbers(both)) < 1000  # 절대 가격(6만대)·절대 거래량(수천)이 없다
    assert not any(w in json.dumps(both) for w in ("BTC", "USDT", "65432", "2023"))  # 코인명·가격·날짜 없음


@pytest.mark.parametrize("variant", jr.VARIANTS)
def test_state_is_causal(variant):
    df, i = candles(seed=3), 300
    assert jr.make_state(df, i, variant) == jr.make_state(df.iloc[: i + 1].reset_index(drop=True), i, variant)  # 미래를 잘라도 같은 상태


def test_sample_indices_valid_and_deterministic():
    idx = jr.sample_indices(5000, 500, seed=7)
    assert len(set(idx)) == 500 and idx.min() >= jr.HIST - 1 and idx.max() <= 5000 - 1 - jr.H
    assert (idx == jr.sample_indices(5000, 500, seed=7)).all() and not (idx == jr.sample_indices(5000, 500, seed=8)).all()
    n_all = 300 - jr.H - (jr.HIST - 1)  # 후보를 전부 뽑으면 정확히 [HIST-1, n-H) 여야 한다 (경계가 우연히 안 뽑히는 것을 배제)
    assert jr.sample_indices(300, n_all, seed=1).tolist() == list(range(jr.HIST - 1, 300 - jr.H))


# ── SDK 경로 ──
def test_call_uses_three_questions_and_parses_answers():
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen.update(body)
        return httpx2.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {
            "side": {"type": "choice", "choice": "short", "confidence": 0.7, "probabilities": {"long": 0.1, "short": 0.8, "flat": 0.1}},
            "reversal_risk": {"type": "noul", "noul": 0.25}, "up3": {"type": "noul", "noul": 0.4}}})

    client = TypeSafeClient(api_key="t", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0))
    r = jr.call(client, jr._questions(), {"x": 1})
    assert r == {"side": "short", "conf": 0.7, "p_long": 0.1, "p_short": 0.8, "p_flat": 0.1, "rev": 0.25, "up3": 0.4}
    assert set(seen["questions"]) == {"side", "reversal_risk", "up3"} and seen["state"] == {"x": 1}
    bad = TypeSafeClient(api_key="t", transport=httpx2.MockTransport(lambda req: httpx2.Response(500, json={})), retry=RetryPolicy(max_retries=0))
    assert jr.call(bad, jr._questions(), {}) is None  # 실패는 None (필터/분석에서 제외)


# ── 표 구성 ──
def test_build_table_forward_returns_and_incomplete_samples():
    df = candles(300)
    data = {"BTC/USDT": df}
    ts = int(df["ts"].iloc[250])
    resp = {"side": "long", "conf": 0.9, "p_long": 0.8, "p_short": 0.1, "p_flat": 0.1, "rev": 0.2, "up3": 0.6}
    cache = {f"{v}|BTC/USDT|{ts}": resp for v in jr.VARIANTS}
    cache[f"labels|BTC/USDT|{int(df['ts'].iloc[200])}"] = resp  # 변형이 하나뿐 → 제외되어야 함
    T = jr.build_table(cache, data)
    assert len(T) == 1
    c = df["close"].to_numpy()
    assert T["fwd3"].iloc[0] == pytest.approx((c[253] / c[250] - 1) * 100) and T["fwd1"].iloc[0] == pytest.approx((c[251] / c[250] - 1) * 100)
    assert T["mom3"].iloc[0] == pytest.approx((c[250] / c[247] - 1) * 100) and T["s_raw"].iloc[0] == pytest.approx(0.7)


# ── 통계: 실제 정보는 잡고 노이즈는 오탐하지 않는다 ──
def table(n=1500, seed=0, edge=0.0):
    rng, rows = np.random.default_rng(seed), []
    for coin in ("A", "B", "C"):
        f = rng.normal(0, 1, n)
        rows.append(pd.DataFrame({"coin": coin, "fwd3": f, "s_labels": rng.normal(0, 1, n), "s_raw": edge * f + rng.normal(0, 1, n),
                                  "s_raw+labels": rng.normal(0, 1, n), "side_raw": np.where(edge * f + rng.normal(0, 1, n) > 0, "long", "short")}))
    return pd.concat(rows, ignore_index=True)


def test_delta_ic_detects_planted_edge_and_not_noise():
    T = table(edge=0.5)
    d, lo, hi = jr.delta_ic_ci(T, "labels", "raw", B=300)
    assert d > 0.1 and lo > 0  # raw 에만 정보가 있음
    assert jr.perm_p_ic(T, "s_raw", B=300) < 0.01 and jr.perm_p_ic(T, "s_labels", B=300) > 0.01
    d0, lo0, hi0 = jr.delta_ic_ci(table(edge=0.0, seed=1), "labels", "raw+labels", B=300)
    assert lo0 < 0 < hi0  # 둘 다 노이즈면 CI 가 0 을 포함


def test_accuracy_permutation():
    T = table(edge=2.0, seed=2)
    T["side_raw"] = np.where(T["fwd3"] > 0, "long", "short")  # 완벽한 예측
    a = jr.accuracy(T, "raw", B=200)
    assert a["hit"] > 0.99 and a["p"] < 0.01 and a["chance"] == pytest.approx(0.5, abs=0.03)
    T2 = table(edge=0.0, seed=3)
    assert jr.accuracy(T2, "raw", B=200)["p"] > 0.01


def test_rank_corr_constant_score_is_zero():
    assert jr.rank_corr(np.ones(50), np.arange(50)) == 0.0
