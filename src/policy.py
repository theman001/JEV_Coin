"""결정론적 정책 + 리스크 거부권. Jev 판단은 여기서 임계값을 통과해야만 주문이 된다.

임계값은 봉 주기별 Profile 로 묶는다(5m/1s 기준을 다르게 — 사용자 확정 2026-09-29). 비용(COST_PER_SIDE)·손절·세션 손실 한도처럼
"돈이 실제로 얼마나 새는가"에 관한 값은 주기가 바뀐다고 사실이 바뀌지 않으므로 모든 프로필에서 같다. 달라지는 건 Jev 판단을
얼마나 깐깐하게 걸러 주문으로 옮길지(min_conf/max_reversal), 얼마나 미세한 변동에도 진입을 허용할지(min_atr_cost_mult),
그리고 얼마나 자주 Jev를 부를지(judge_cooldown_s — 1초봉은 캔들이 초당 마감되므로 없으면 하루 40만+ 회 호출된다)뿐이다.
"""
from dataclasses import dataclass

from .judge import Judgment

FEE = 0.0005  # 편도 테이커 수수료율
SLIPPAGE = 0.0002  # 편도 슬리피지 가정 (드라이런 체결가에 반영 안 하고 비용으로 차감)
COST_PER_SIDE = FEE + SLIPPAGE


@dataclass(frozen=True)
class Profile:
    name: str
    min_conf: float  # 미만이면 진입 안 함
    full_conf: float  # 미만이면 사이즈 절반 (confidence 낮으면 포기가 아니라 축소)
    max_reversal: float  # 반전 위험이 이보다 높으면 관망
    max_pos_frac: float  # 자본 대비 최대 포지션 비중 (증거금 기준, 레버리지 없음)
    max_session_loss: float  # 세션 시작 대비 손실이 이 이상이면 청산 후 신규 진입 차단
    stop_loss_pct: float  # 진입가 대비 이 %면 캔들 마감을 기다리지 않고 즉시 청산 (Jev 무관, 코드가 폴링마다 확인)
    min_atr_cost_mult: float  # ATR%가 왕복 비용의 이 배수 미만이면 수수료도 못 넘는 장 → 진입 안 함
    judge_cooldown_s: float = 0.0  # 새 캔들이어도 이 코인을 이보다 최근에 판단했으면 이번엔 Jev 를 부르지 않음 (호출량 상한)


PROFILES = {
    # 5m(및 그 외 모든 주기)의 기본값 — 기존과 동일, 바뀐 게 없다.
    "default": Profile(name="default", min_conf=0.6, full_conf=0.8, max_reversal=0.7, max_pos_frac=0.25,
                        max_session_loss=0.05, stop_loss_pct=0.5, min_atr_cost_mult=1.0, judge_cooldown_s=0.0),
    # 1s: 실측(1000봉, CLAUDE.md) ATR%(14) 중앙값이 BTC 0.005%~DOGE 0.022% 로 기본 게이트(0.14%)를 절대 못 넘어
    # "처음부터 끝까지 관망"이 되므로 사용자 요청(2026-09-29)대로 크게 완화한다: 게이트는 사실상 순수 무변동 봉만 거르고
    # (0.01 × 0.14% = 0.0014%), 확신도/반전위험 문턱도 낮춘다. 대신 judge_cooldown_s 로 호출량을 묶는다
    # (완화 후에도 5코인 × 1초마다면 하루 43만+ 호출/약 $6 — 30초 쿨다운이면 하루 14,400회/약 $0.2).
    # 손절·세션 손실 한도·사이즈는 그대로: 여기서 바꾸는 건 "판단이 실제로 실행으로 이어지는 문턱"이지 리스크 상한이 아니다.
    # 관찰·모니터링 목적이며 성과 근거는 없다(CLAUDE.md 원칙 3·10) — 이 결과를 보고 다시 임계값을 조정하지 않는다.
    "1s": Profile(name="1s", min_conf=0.45, full_conf=0.65, max_reversal=0.85, max_pos_frac=0.25,
                  max_session_loss=0.05, stop_loss_pct=0.5, min_atr_cost_mult=0.01, judge_cooldown_s=30.0),
}


def profile_for(timeframe: str) -> Profile:
    return PROFILES.get(timeframe, PROFILES["default"])


@dataclass(frozen=True)
class Signal:
    side: str  # long | short | flat
    size: float  # 자본 대비 비중
    reason: str


def decide(j: Judgment, session_loss_pct: float, cost_covered: bool = True, profile: Profile = PROFILES["default"]) -> Signal:
    # 리스크 거부권: Jev 결과와 무관하게 최우선. flat이면 포지션은 청산되고, 자본이 고정되므로 손실 한도 초과 상태가 유지된다.
    if session_loss_pct >= profile.max_session_loss:
        return Signal("flat", 0.0, "risk veto: session loss limit")
    if not cost_covered:
        return Signal("flat", 0.0, "volatility below trading costs")
    if j.side == "flat":
        return Signal("flat", 0.0, "judged flat")
    if j.confidence < profile.min_conf:
        return Signal("flat", 0.0, f"low confidence {j.confidence:.2f}")
    if j.reversal_risk > profile.max_reversal:
        return Signal("flat", 0.0, f"reversal risk {j.reversal_risk:.2f}")
    size = profile.max_pos_frac if j.confidence >= profile.full_conf else profile.max_pos_frac / 2
    return Signal(j.side, size, f"{j.source}: {j.side} conf {j.confidence:.2f}")
