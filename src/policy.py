"""결정론적 정책 + 리스크 거부권. Jev 판단은 여기서 임계값을 통과해야만 주문이 된다."""
from dataclasses import dataclass

from .judge import Judgment

MIN_CONF = 0.6  # 미만이면 진입 안 함
FULL_CONF = 0.8  # 미만이면 사이즈 절반 (confidence 낮으면 포기가 아니라 축소)
MAX_REVERSAL = 0.7  # 반전 위험이 이보다 높으면 관망
MAX_POS_FRAC = 0.25  # 자본 대비 최대 포지션 비중 (증거금 기준, 레버리지 없음)
MAX_SESSION_LOSS = 0.05  # 세션 시작 대비 손실이 5% 이상이면 청산 후 신규 진입 차단
STOP_LOSS_PCT = 0.5  # 진입가 대비 -0.5%면 캔들 마감을 기다리지 않고 즉시 청산 (Jev 무관, 코드가 폴링마다 확인)

FEE = 0.0005  # 편도 테이커 수수료율
SLIPPAGE = 0.0002  # 편도 슬리피지 가정 (드라이런 체결가에 반영 안 하고 비용으로 차감)
COST_PER_SIDE = FEE + SLIPPAGE
MIN_ATR_COST_MULT = 1.0  # ATR%가 왕복 비용의 이 배수 미만이면 수수료도 못 넘는 장 → 진입 안 함


@dataclass(frozen=True)
class Signal:
    side: str  # long | short | flat
    size: float  # 자본 대비 비중
    reason: str


def decide(j: Judgment, session_loss_pct: float, cost_covered: bool = True) -> Signal:
    # 리스크 거부권: Jev 결과와 무관하게 최우선. flat이면 포지션은 청산되고, 자본이 고정되므로 손실 한도 초과 상태가 유지된다.
    if session_loss_pct >= MAX_SESSION_LOSS:
        return Signal("flat", 0.0, "risk veto: session loss limit")
    if not cost_covered:
        return Signal("flat", 0.0, "volatility below trading costs")
    if j.side == "flat":
        return Signal("flat", 0.0, "judged flat")
    if j.confidence < MIN_CONF:
        return Signal("flat", 0.0, f"low confidence {j.confidence:.2f}")
    if j.reversal_risk > MAX_REVERSAL:
        return Signal("flat", 0.0, f"reversal risk {j.reversal_risk:.2f}")
    size = MAX_POS_FRAC if j.confidence >= FULL_CONF else MAX_POS_FRAC / 2
    return Signal(j.side, size, f"{j.source}: {j.side} conf {j.confidence:.2f}")
