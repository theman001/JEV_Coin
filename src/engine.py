"""백그라운드 엔진: 폴링 루프 + 웹에서 호출하는 제어(start/stop/set_symbol) + 상태 스냅샷 + 디스크 영속화.

락 규칙: 상태 접근은 self.lock 안에서. 거래소/Jev 네트워크 호출은 락 밖에서 하고,
돌아온 뒤 (실행 중 & 같은 코인)인지 다시 확인해서 아니면 결과를 버린다.
"""
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

from .broker import PaperBroker
from .data import StaleData, fetch_closed
from .policy import STOP_LOSS_PCT, Signal, decide
from .state import build_state, indicators

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ERRORS = 5  # 넘으면 킬스위치 (청산 후 정지)


class Engine:
    def __init__(self, ex, judge, symbols: list[str], timeframe: str = "5m", interval: float = 10, data_dir=None):
        self.ex, self.judge, self.symbols, self.timeframe, self.interval = ex, judge, list(symbols), timeframe, interval
        self.data_dir = Path(data_dir) if data_dir else None
        self.lock = threading.RLock()
        self.symbol, self.running = self.symbols[0], False
        self.broker = PaperBroker()
        self.broker.symbol = self.symbol
        self.price = self.polled_at = self.last_ts = self.error = None
        self.ind, self.state = {}, {}
        self.judgments = deque(maxlen=100)  # 판단/손절 이벤트 (최신이 뒤)
        self.curve = deque(maxlen=1000)  # (ts, equity) 폴링마다 1점 — 메모리 전용
        self._load()

    # ---------- 제어 (웹 스레드에서 호출) ----------
    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running, self.error, self.last_ts = True, None, None  # last_ts 초기화: 시작 즉시 마지막 마감 캔들로 판단
            self._save()

    def stop(self) -> None:
        """포지션을 청산하고 정지. 정지 중에는 손절도 감시하지 않으므로 포지션을 남기지 않는다."""
        with self.lock:
            self._flatten()
            self.running = False
            self._save()

    def set_symbol(self, symbol: str) -> None:
        if symbol not in self.symbols:
            raise ValueError(f"unknown symbol {symbol!r}")
        with self.lock:
            if symbol == self.symbol:
                return
            self._flatten()
            self.symbol = self.broker.symbol = symbol
            self.price = self.last_ts = None
            self.ind, self.state = {}, {}
            self._save()

    # ---------- 루프 (엔진 스레드) ----------
    def run_forever(self) -> None:
        errors = 0
        while True:
            if self.running:
                try:
                    self.step()
                    errors, self.error = 0, None
                except StaleData as e:  # 정상적인 건너뜀: 오류 카운트에 넣지 않는다
                    self.error = f"skip: {e}"
                except Exception as e:
                    errors += 1
                    self.error = f"{type(e).__name__}: {e} ({errors}/{MAX_CONSECUTIVE_ERRORS})"
                    log.exception("step failed (%d/%d)", errors, MAX_CONSECUTIVE_ERRORS)
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        self.stop()
                        self.error = f"kill switch: stopped after {errors} consecutive errors — {e}"
                        errors = 0
            time.sleep(self.interval)

    def step(self) -> None:
        """폴링 1회: 가격 → 손절 확인 → 지표 갱신 → (새 캔들 마감이면) Jev 판단 → 정책 → 주문."""
        with self.lock:
            sym = self.symbol
        price = float(self.ex.fetch_ticker(sym)["last"])
        df = fetch_closed(self.ex, sym, self.timeframe)
        ts = int(df["ts"].iloc[-1])
        with self.lock:
            if not self._live(sym):
                return
            self.price, self.polled_at = price, time.time()
            stopped = self.broker.check_stop(price, STOP_LOSS_PCT)
            if stopped:
                self._record("stop_loss", price, reason=f"stop loss {STOP_LOSS_PCT}%")
                self._save()
            self.ind = indicators(df)
            self.state = state = build_state(df, sym, self.timeframe, (self.broker.side, self.broker.pnl_pct(price)))
            self.curve.append((self.polled_at, self.broker.equity(price)))
            new_candle = ts != self.last_ts and not stopped  # 손절 직후엔 이 캔들에서 재진입하지 않는다 (휩쏘 반복 방지)
            if stopped:
                self.last_ts = ts
        if not new_candle:  # 디스크 쓰기는 상태가 바뀔 때만 (SD/eMMC 마모 방지): 폴링마다 저장하지 않는다
            return
        j = self.judge(state)  # 느린 Jev 호출은 락 밖에서. 캔들당 1회.
        with self.lock:
            if not self._live(sym):
                return
            sig = decide(j, self.broker.session_loss_pct(price), state["volatility_covers_costs"])
            self.broker.rebalance(sig, price)
            self.last_ts = ts
            self._record("judgment", price, j=j, sig=sig)
            self._save()

    # ---------- 조회 ----------
    def snapshot(self) -> dict:
        with self.lock:
            b, p = self.broker, self.price
            mark = p if p is not None else b.entry
            eq = b.equity(mark)
            return {
                "now": time.time(), "running": self.running, "symbol": self.symbol, "symbols": self.symbols,
                "timeframe": self.timeframe, "judge": self.judge.name, "dry_run": True, "poll_seconds": self.interval,
                "price": p, "polled_at": self.polled_at, "candle_ts": self.last_ts and self.last_ts / 1000, "error": self.error,
                "indicators": {k: (round(float(v), 4) if math.isfinite(v) else None) for k, v in self.ind.items()},
                "state": self.state,
                "position": {"side": b.side, "entry": b.entry, "qty": b.qty, "pnl_pct": b.pnl_pct(mark)},
                "equity": eq, "start_equity": b.start_equity, "return_pct": (eq / b.start_equity - 1) * 100,
                "realized_pnl": b.cash - b.start_equity + b.entry_cost,  # 진입 비용은 청산 시 거래손익에 귀속되므로 되돌림
                "unrealized_pnl": b.unrealized(mark),
                "session_loss_pct": b.session_loss_pct(mark) * 100,
                "n_trades": b.n_trades, "n_wins": b.n_wins,
                "judgments": list(self.judgments)[::-1], "trades": b.trades[-20:][::-1], "curve": list(self.curve),
            }

    # ---------- 내부 ----------
    def _live(self, sym: str) -> bool:
        return self.running and self.symbol == sym

    def _flatten(self) -> None:
        if self.broker.side == "flat":
            return
        try:
            price = float(self.ex.fetch_ticker(self.symbol)["last"])
        except Exception:
            price = self.price  # 거래소 장애 시 마지막으로 본 가격
        if price is None:
            self.error = "cannot flatten: no price available"
            log.error(self.error)
            return
        self.broker.rebalance(Signal("flat", 0.0, "manual"), price)
        self._record("manual_flat", price, reason="stopped/changed by user")

    def _record(self, kind: str, price: float, j=None, sig=None, reason: str = "") -> None:
        row = {"ts": time.time(), "kind": kind, "symbol": self.symbol, "price": price,
               "position": self.broker.side, "equity": self.broker.equity(price), "reason": reason}
        if j and sig:
            row |= {"side": j.side, "confidence": j.confidence, "reversal_risk": j.reversal_risk, "source": j.source,
                    "signal": sig.side, "size": sig.size, "reason": sig.reason}
        self.judgments.append(row)
        if self.data_dir:  # 판단 이력은 나중에 Jev 보정(calibration)을 검증하는 원자료가 된다
            with open(self.data_dir / "judgments.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

    def _save(self) -> None:
        if not self.data_dir:
            return
        tmp = self.data_dir / "state.json.tmp"
        tmp.write_text(json.dumps({"symbol": self.symbol, "running": self.running, "broker": self.broker.snapshot()}))
        os.replace(tmp, self.data_dir / "state.json")

    def _load(self) -> None:
        if not self.data_dir:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            d = json.loads((self.data_dir / "state.json").read_text())
            broker = PaperBroker.restore(d["broker"])
        except FileNotFoundError:
            return
        except (ValueError, KeyError, OSError) as e:  # 손상된 파일: 조용히 지우지 않고 크게 알린다
            log.error("state.json unreadable, starting fresh (file kept): %s", e)
            self.error = f"state.json unreadable, started fresh: {e}"
            return
        self.broker, self.symbol, self.running = broker, d["symbol"], d["running"]
        if self.symbol not in self.symbols:  # SYMBOLS 설정이 바뀌어도 보유 포지션의 코인은 유지
            self.symbols.insert(0, self.symbol)
        log.info("restored: %s running=%s equity=%.2f trades=%d", self.symbol, self.running, broker.cash, broker.n_trades)
