"""Jev 판단 시험(풍부한 특징 입력) 검증. 네트워크·API 키 불필요."""
import json

import httpx2
import numpy as np
import pandas as pd
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

import src.jev_rich as jr
from src.features import compute
from tests.test_features import make
from tests.test_screen import DAY, fake_panels


def test_state_is_masked_scale_free_and_nan_free():
    df = make(600)
    big = df.assign(open=df.open * 1000, high=df.high * 1000, low=df.low * 1000, close=df.close * 1000, volume=df.volume * 7, taker_buy_base=df.taker_buy_base * 7)
    a = jr.rich_state(compute(df, make(600, seed=2)).iloc[500], {"connors_rsi2": "long"}, "5m")
    b = jr.rich_state(compute(big, make(600, seed=2).assign(close=lambda d: d.close * 3)).iloc[500], {"connors_rsi2": "long"}, "5m")
    assert a["features"].keys() == b["features"].keys() and len(a["features"]) > 80
    for k in a["features"]:  # 가격·거래량 단위를 바꿔도 모든 특징이 같다 = 절대 가격·거래량이 상태에 없다 (BTC 상대 열은 BTC 가격 배율 무관해야 함)
        assert a["features"][k] == pytest.approx(b["features"][k], rel=1e-6, abs=1e-6), k
    blob = json.dumps(a)
    assert not any(w in blob for w in ("BTC", "USDT", "f_", "ts", "nan", "NaN")) and a["active_setups"] == {"connors_rsi2": "long"}
    assert a["timeframe"] == "5m"
    early = jr.rich_state(compute(df).iloc[35], {}, "5m")  # 워밍업 구간: NaN 키는 빠진다
    assert all(np.isfinite(v) for v in early["features"].values()) and len(early["features"]) < len(a["features"])


def test_questions_and_call_parsing():
    q = jr.questions()
    assert set(q) == {"side", "up3", "down3", "up12", "down12"}
    assert "0.28%" in q["up12"].instructions and "higher" in q["up12"].instructions and "lower" in q["down3"].instructions and "12 candles" in q["down12"].instructions
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        ans = {"side": {"type": "choice", "choice": "long", "confidence": 0.6, "probabilities": {"long": 0.7, "short": 0.2, "flat": 0.1}}}
        ans |= {k: {"type": "noul", "noul": v} for k, v in zip(("up3", "down3", "up12", "down12"), (0.1, 0.2, 0.3, 0.4))}
        return httpx2.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": ans})

    client = TypeSafeClient(api_key="t", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0))
    r = jr.call(client, q, {"x": 1})
    assert r == {"side": "long", "p_long": 0.7, "p_short": 0.2, "conf": 0.6, "up3": 0.1, "down3": 0.2, "up12": 0.3, "down12": 0.4}
    assert set(seen["questions"]) == set(q)
    bad = TypeSafeClient(api_key="t", transport=httpx2.MockTransport(lambda req: httpx2.Response(500, json={})), retry=RetryPolicy(max_retries=0))
    assert jr.call(bad, q, {}) is None


def test_make_samples_oos_only_unique_and_deterministic(monkeypatch):
    panels, t0 = fake_panels(n=9000, seed=3)
    for p in panels.values():
        p["sig"] = {name: pd.Series(np.zeros(9000)) for name in jr.SETUP_IDS}
        p["sig"]["S3 VWAP·볼린저 회귀"] = pd.Series(np.where(np.arange(9000) % 2 == 0, 1, -1))
    monkeypatch.setattr(jr, "build", lambda tf, years, ex: (panels, t0))
    monkeypatch.setattr(jr, "N_PER_COIN", 150)
    S, states = jr.make_samples("15m")
    assert len(S) == 5 * 150 and S["key"].is_unique and set(S["key"]) == set(states)
    assert (S["ts"] >= t0 + 366 * DAY).all() and set(S["coin"]) == set(panels)  # OOS 구간만, 5코인 모두
    assert all(list(v["active_setups"]) == ["vwap_bollinger_reversion"] and v["active_setups"]["vwap_bollinger_reversion"] in ("long", "short") for v in states.values())
    assert not any("USDT" in json.dumps(v) for v in states.values())
    S2, _ = jr.make_samples("15m")
    assert S["key"].tolist() == S2["key"].tolist()
    c0 = panels["BTC/USDT"]
    row = S[S.coin == "BTC/USDT"].iloc[0]
    i = int(np.flatnonzero(c0["df"]["ts"].to_numpy() == row.ts)[0])
    assert row.fwd3 == pytest.approx(c0["fwd"][3].iloc[i]) and row.fwd12 == pytest.approx(c0["fwd"][12].iloc[i])


def synth(n=3000, seed=0, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in ("A", "B", "C", "D", "E"):
        z = rng.normal(size=n)
        y12 = edge * z + rng.normal(0, 0.8, n)
        rows.append(pd.DataFrame({"coin": c, "ts": 1_700_000_000_000 + np.arange(n) * 900_000 * 4, "s": z, "noise": rng.normal(size=n), "fwd12": y12, "fwd3": y12 / 2}))
    return pd.concat(rows, ignore_index=True)


def test_extreme_decile_direction_and_cost():
    T = synth(edge=0.6)  # 점수가 높을수록 수익 ↑ : 상위 10% 롱 + 하위 10% 숏 → 총이익 ≈ 0.6×1.755
    e = jr.extreme_decile(T, "s", "fwd12")
    assert e["ls"]["gross"] == pytest.approx(0.6 * 1.755, abs=0.12) and e["ls"]["net"] == pytest.approx(e["ls"]["gross"] - jr.COST) and e["ls"]["p"] < 1e-6
    assert e["long"]["gross"] > 0.8 and e["long"]["n"] * 2 == pytest.approx(e["ls"]["n"], rel=0.05)
    rev = jr.extreme_decile(T, "s", "fwd12", sign=-1)  # 반대로 거래하면 손실
    assert rev["ls"]["gross"] < -0.8 and rev["ls"]["p"] > 0.99
    small = jr.extreme_decile(synth(edge=0.05, seed=1), "s", "fwd12")  # 통계적으로 뚜렷해도 총이익 < 비용이면 순기대수익 < 0
    assert 0 < small["ls"]["gross"] < jr.COST and small["ls"]["net"] < 0
    assert jr.extreme_decile(synth(edge=0.0, seed=2), "noise", "fwd12")["ls"]["p"] > 0.01


def test_delta_ic_generic_target_and_scores():
    T = synth(edge=0.6)
    d, lo, hi = jr.delta_ic(T, "noise", "s", "fwd12", B=200)
    assert d > 0.2 and lo > 0
    d0, lo0, hi0 = jr.delta_ic(synth(edge=0.0, seed=4), "noise", "s", "fwd12", B=200)
    assert lo0 < 0 < hi0
    T["up3"], T["down3"], T["up12"], T["down12"] = 0.5, 0.2, 0.6, 0.1
    T["p_long"], T["p_short"] = 0.7, 0.2
    for k in ("rsi14", "roc6", "cci20", "bb_pctb", "roc3"):
        T[k] = np.random.default_rng(0).normal(size=len(T))
    T = jr.add_scores(T)
    assert np.allclose(T["s_move12"], 0.5) and np.allclose(T["s_side"], 0.5)
    assert T["b_reversal"].between(-1, 0).all() and T["b_rsi"].between(-1, 0).all()
