"""PaperBroker: 드라이런 체결. 실주문 없음 (실거래 브로커는 의도적으로 미구현)."""
import time

from .policy import COST_PER_SIDE, Signal

MAX_TRADES_KEPT = 200


class PaperBroker:
    def __init__(self, equity: float = 10_000.0, symbol: str = ""):
        self.cash = equity  # 실현 손익 반영된 자본 (진입 비용 차감 후)
        self.start_equity = equity
        self.symbol = symbol  # 체결 기록 라벨용
        self.side = "flat"
        self.qty = 0.0
        self.entry = 0.0
        self.entry_cost = 0.0
        self.trades: list[dict] = []  # 최근 청산 기록 (표시용)
        self.n_trades = self.n_wins = 0  # 누적 통계 (trades가 잘려도 유지)

    def equity(self, price: float) -> float:
        return self.cash + self.unrealized(price)

    def session_loss_pct(self, price: float) -> float:
        return max(0.0, 1 - self.equity(price) / self.start_equity)

    def pnl_pct(self, price: float) -> float:
        """진입가 대비 미실현 손익 % (방향 반영). 플랫이면 0."""
        return self.unrealized(price) / (self.entry * self.qty) * 100 if self.qty else 0.0

    def check_stop(self, price: float, stop_pct: float) -> bool:
        """손절선 이탈 시 즉시 청산 (Jev 판단을 기다리지 않는다). 청산했으면 True."""
        if self.side == "flat" or self.pnl_pct(price) > -stop_pct:
            return False
        self._close(price)
        return True

    def rebalance(self, sig: Signal, price: float) -> None:
        # ponytail: 같은 방향이면 유지(리사이즈 없음) — 수수료 낭비 방지. 사이즈 조정이 필요하면 여기서 확장.
        if sig.side == self.side:
            return
        self._close(price)
        if sig.side != "flat":
            self.qty = self.cash * sig.size / price
            self.side, self.entry = sig.side, price
            self.entry_cost = self.qty * price * COST_PER_SIDE
            self.cash -= self.entry_cost

    def snapshot(self) -> dict:
        return {k: getattr(self, k) for k in (
            "cash", "start_equity", "symbol", "side", "qty", "entry", "entry_cost", "trades", "n_trades", "n_wins")}

    @classmethod
    def restore(cls, d: dict) -> "PaperBroker":
        b = cls(d["start_equity"])
        b.__dict__.update(d)
        return b

    def unrealized(self, price: float) -> float:
        d = {"long": 1, "short": -1}.get(self.side, 0)
        return d * (price - self.entry) * self.qty

    def _close(self, price: float) -> None:
        if self.side == "flat":
            return
        gross, exit_cost = self.unrealized(price), self.qty * price * COST_PER_SIDE
        pnl = gross - exit_cost - self.entry_cost  # 진입·청산 비용 모두 반영한 거래 순손익
        self.cash += gross - exit_cost
        self.n_trades += 1
        self.n_wins += pnl > 0
        self.trades = (self.trades + [{
            "ts": time.time(), "symbol": self.symbol, "side": self.side, "entry": self.entry, "exit": price,
            "pnl": pnl, "pnl_pct": pnl / (self.entry * self.qty) * 100}])[-MAX_TRADES_KEPT:]
        self.side, self.qty, self.entry, self.entry_cost = "flat", 0.0, 0.0, 0.0
