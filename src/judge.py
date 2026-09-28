"""판단 계층: State → Judgment. Jev(TypeSafe System One) 또는 오프라인 FakeJudge.

Jev는 판단만 한다. 임계값·사이즈·주문은 policy/broker(코드)가 담당한다.
어떤 실패도 안전한 쪽(flat, confidence 0)으로 떨어진다.
"""
import logging
from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient, TypeSafeError

log = logging.getLogger(__name__)

SIDES = ("long", "short", "flat")

QUESTIONS = {
    "side": Choice(
        # 실행 분기: 목표 포지션을 고른다. 현재 포지션과 같으면 유지, 다르면 전환/청산 (state에 current_position이 있다).
        instructions="Given this market state and the current position, which position is best to hold over the next few candles?",
        criteria={
            "long": "The price is likely to rise over the next few candles; holding or opening a long is favored",
            "short": "The price is likely to fall over the next few candles; holding or opening a short is favored",
            "flat": "There is no clear edge, or the current position should be closed; holding no position is favored",
        },
    ),
    "reversal_risk": Noul(instructions="The recent price move is likely to reverse sharply soon."),
}


@dataclass(frozen=True)
class Judgment:
    side: str  # long | short | flat
    confidence: float  # side 분포의 집중도 (정답 확률 아님)
    reversal_risk: float  # Noul: P(yes)
    source: str


SAFE = Judgment("flat", 0.0, 1.0, "fallback")


def parse(resp) -> Judgment:
    side, rev = resp.answers["side"], resp.answers["reversal_risk"]
    if side.choice not in SIDES:
        raise ValueError(f"unexpected side {side.choice!r}")
    return Judgment(side.choice, side.confidence, rev.noul, "jev")


class JevJudge:
    name = "jev"

    def __init__(self, client: TypeSafeClient | None = None):
        # 틱 하나가 오래 막히지 않도록 짧은 총 재시도 예산 (기본 30s → 10s)
        self.client = client or TypeSafeClient(retry=RetryPolicy(max_retries=2, timeout=10.0))

    def __call__(self, state: dict) -> Judgment:
        try:
            return parse(self.client.system_one(state=state, questions=QUESTIONS))
        except (TypeSafeError, KeyError, ValueError) as e:
            log.warning("jev failed, staying flat: %s: %s", type(e).__name__, e)
            return SAFE


class FakeJudge:
    """오프라인 스텁 (Jev 아님). 파이프라인/정책 테스트용 단순 규칙."""

    name = "fake"

    def __call__(self, state: dict) -> Judgment:
        rsi_ok = state["rsi_zone"] == "neutral"
        rev = 0.2 if rsi_ok else 0.8
        if state["trend"] == "up" and state["rsi_zone"] != "overbought":
            return Judgment("long", 0.7, rev, "fake")
        if state["trend"] == "down" and state["rsi_zone"] != "oversold":
            return Judgment("short", 0.7, rev, "fake")
        return Judgment("flat", 0.9, rev, "fake")
