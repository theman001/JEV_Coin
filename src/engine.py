"""백그라운드 엔진(Jev 판단 단타, 멀티코인): 폴링 루프 + 웹에서 호출하는 제어(start/stop) + 상태 스냅샷 + 디스크 영속화.

코인 N개가 코인별 독립 슬롯(자본 ÷ N, PaperBroker)으로 동시에 돈다. 정책(임계값·사이즈·손절)은 슬롯 안에서, 세션 손실 한도는 포트폴리오 전체 기준.
락 규칙: 상태 접근은 self.lock 안에서. 거래소/Jev 네트워크 호출은 락 밖에서 하고,
돌아온 뒤 running 인지 다시 확인해서 아니면 결과를 버린다.
"""
import json
import logging
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .broker import PaperBroker
from .data import StaleData, fetch_closed
from .policy import MAX_SESSION_LOSS, STOP_LOSS_PCT, Signal, decide
from .state import build_state, indicators

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ERRORS = 5  # 한 코인이 이만큼 연속 실패하면 그 코인 청산(감시 불가), 전 코인이 그러면 킬스위치(전체 정지)


class Engine:
    page = "index.html"

    def __init__(self, ex, judge, symbols: list[str], timeframe: str = "5m", interval: float = 10, data_dir=None,
                 equity: float = 100.0):
        self.ex, self.judge, self.symbols, self.timeframe, self.interval = ex, judge, list(symbols), timeframe, interval
        self.data_dir = Path(data_dir) if data_dir else None
        self.lock = threading.RLock()
        self.start_equity = equity
        self.slots = {s: PaperBroker(equity / len(self.symbols), symbol=s) for s in self.symbols}
        self.running = False
        self.last_ts, self.prices, self.ind, self.state, self.last_j = {}, {}, {}, {}, {}  # 코인별
        self.errors, self.err_msg = {}, {}  # 코인별 연속 오류 횟수 / 마지막 오류 메시지
        self.judgments = deque(maxlen=100)  # 판단/손절/청산 이벤트 (최신이 뒤)
        self.curve = deque(maxlen=1000)  # (ts, 포트폴리오 자산) 폴링마다 1점 — 메모리 전용
        self.polled_at = self.error = None
        self._load()

    # ---------- 제어 (웹 스레드에서 호출) ----------
    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running, self.error, self.last_ts = True, None, {}  # last_ts 초기화: 시작 즉시 코인마다 마지막 마감 캔들로 판단
            self.errors.clear()
            self.err_msg.clear()
            self._save()

    def stop(self) -> None:
        """모든 포지션을 청산하고 정지. 정지 중에는 손절도 감시하지 않으므로 포지션을 남기지 않는다."""
        quotes = self._quotes()
        with self.lock:
            self._flatten("manual_flat", "stopped by user", quotes)
            self.running = False
            self._save()

    def set_symbol(self, symbol) -> None:
        raise ValueError("all symbols run simultaneously: no symbol selection")

    # ---------- 루프 (엔진 스레드) ----------
    def run_forever(self) -> None:
        while True:
            if self.running:
                try:
                    self.step()
                except Exception as e:  # step 은 코인별로 예외를 흡수한다. 방어용
                    log.exception("step failed")
                    self.error = f"{type(e).__name__}: {e}"
            time.sleep(self.interval)

    def step(self) -> None:
        """폴링 1회: ① 코인별 [가격 → 손절 → 지표] (Jev 호출 없음) ② 새 캔들이 마감된 코인만 Jev 판단을 병렬로 ③ 정책 → 주문 ④ 포트폴리오 리스크 확인.
        판단을 병렬로 하는 이유: 순차 호출이면 Jev 가 느릴 때 다른 코인의 손절 확인이 그만큼 밀린다."""
        with self.lock:
            syms = list(self.symbols)
        pending = []
        for sym in syms:
            try:
                p = self._observe(sym)
            except StaleData as e:  # 정상적인 건너뜀: 오류 카운트에 넣지 않는다
                self._note(sym, f"skip: {e}", count=False)
            except Exception as e:
                log.exception("%s observe failed", sym)
                self._note(sym, f"{type(e).__name__}: {e}", count=True)
            else:
                with self.lock:
                    self.errors[sym] = 0
                    self.err_msg.pop(sym, None)
                if p:
                    pending.append(p)
        if pending:
            with ThreadPoolExecutor(len(pending)) as pool:
                futs = [pool.submit(self.judge, p["state"]) for p in pending]
                for p, f in zip(pending, futs):
                    try:
                        j = f.result()
                    except Exception as e:
                        log.exception("%s judge failed", p["sym"])
                        self._note(p["sym"], f"judge {type(e).__name__}: {e}", count=True)
                        continue
                    self._apply(p, j)
        self._after_poll()

    def _observe(self, sym: str) -> dict | None:
        price = float(self.ex.fetch_ticker(sym)["last"])
        df = fetch_closed(self.ex, sym, self.timeframe)
        ts = int(df["ts"].iloc[-1])
        with self.lock:
            if not self.running:
                return None
            b = self.slots[sym]
            self.prices[sym] = price
            stopped = b.check_stop(price, STOP_LOSS_PCT)
            if stopped:
                self._record(sym, "stop_loss", price, reason=f"stop loss {STOP_LOSS_PCT}%")
                self._save()
            self.ind[sym] = indicators(df)
            self.state[sym] = st = build_state(df, sym, self.timeframe, (b.side, b.pnl_pct(price)))
            if stopped:  # 손절 직후엔 이 캔들에서 재진입하지 않는다 (휩쏘 반복 방지)
                self.last_ts[sym] = ts
            elif ts != self.last_ts.get(sym):
                return {"sym": sym, "ts": ts, "price": price, "state": st}
        return None

    def _apply(self, p: dict, j) -> None:
        with self.lock:
            if not self.running:  # 락 밖 실행 경합: Jev 를 기다리는 동안 정지됐으면 결과를 버린다
                return
            sym, price = p["sym"], p["price"]
            sig = decide(j, self._loss(), p["state"]["volatility_covers_costs"])
            self.slots[sym].rebalance(sig, price)
            self.last_ts[sym] = p["ts"]
            self.last_j[sym] = (j, sig, time.time())
            self._record(sym, "judgment", price, j=j, sig=sig)
            self._save()

    def _after_poll(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.polled_at = time.time()
            for s, n in self.errors.items():  # 감시할 수 없는 코인은 포지션을 남기지 않는다
                if n >= MAX_CONSECUTIVE_ERRORS:
                    self._flatten("blind_flat", f"{n} consecutive errors", only={s})
            kill = all(self.errors.get(s, 0) >= MAX_CONSECUTIVE_ERRORS for s in self.slots)
            loss = self._loss()
            if loss >= MAX_SESSION_LOSS:  # 리스크 거부권: 포트폴리오 세션 손실 한도 → 즉시 전 코인 청산 (이후 신규 진입은 decide 가 계속 차단)
                self._flatten("risk_flat", f"portfolio session loss {loss * 100:.2f}% >= {MAX_SESSION_LOSS * 100:.0f}% limit")
            self.curve.append((self.polled_at, self._equity()))
            self.error = "; ".join(f"{s}: {m}" for s, m in self.err_msg.items()) or None
            if kill:
                self.running = False
                self.error = f"kill switch: stopped, all symbols failed {MAX_CONSECUTIVE_ERRORS}x in a row — {self.error}"
                self._save()

    # ---------- 조회 ----------
    def snapshot(self) -> dict:
        with self.lock:
            coins, realized, unreal = [], 0.0, 0.0
            for s, b in self.slots.items():
                p = self.prices.get(s)
                mark = p if p is not None else b.entry
                jj = self.last_j.get(s)
                realized += b.cash - b.start_equity + b.entry_cost  # 진입 비용은 청산 시 거래손익에 귀속되므로 되돌림
                unreal += b.unrealized(mark)
                coins.append({
                    "symbol": s, "price": p, "position": b.side, "entry": b.entry, "qty": b.qty, "pnl_pct": b.pnl_pct(mark),
                    "equity": b.equity(mark), "start_equity": b.start_equity, "candle_ts": self.last_ts.get(s) and self.last_ts[s] / 1000,
                    "indicators": {k: (round(float(v), 4) if math.isfinite(v) else None) for k, v in self.ind.get(s, {}).items()},
                    "state": self.state.get(s, {}), "errors": self.errors.get(s, 0),
                    "judgment": jj and {"side": jj[0].side, "confidence": jj[0].confidence, "reversal_risk": jj[0].reversal_risk, "source": jj[0].source,
                                        "signal": jj[1].side, "size": jj[1].size, "reason": jj[1].reason, "ts": jj[2]}})
            eq = self._equity()
            trades = sorted((t for b in self.slots.values() for t in b.trades), key=lambda t: t["ts"])[-20:][::-1]
            return {
                "mode": "jev", "now": time.time(), "running": self.running, "symbols": self.symbols, "timeframe": self.timeframe,
                "judge": self.judge.name, "dry_run": True, "poll_seconds": self.interval, "polled_at": self.polled_at, "error": self.error,
                "coins": coins, "equity": eq, "start_equity": self.start_equity, "return_pct": (eq / self.start_equity - 1) * 100,
                "realized_pnl": realized, "unrealized_pnl": unreal, "session_loss_pct": self._loss() * 100, "loss_limit_pct": MAX_SESSION_LOSS * 100,
                "exposure_pct": 100 * sum(b.side != "flat" for b in self.slots.values()) / len(self.slots),
                "n_trades": sum(b.n_trades for b in self.slots.values()), "n_wins": sum(b.n_wins for b in self.slots.values()),
                "judgments": list(self.judgments)[::-1], "trades": trades, "curve": list(self.curve)}

    # ---------- 내부 ----------
    def _equity(self) -> float:
        return sum(b.equity(self.prices.get(s) or b.entry) for s, b in self.slots.items())

    def _loss(self) -> float:
        return max(0.0, 1 - self._equity() / self.start_equity)

    def _note(self, sym: str, msg: str, count: bool) -> None:
        with self.lock:
            if count:
                self.errors[sym] = self.errors.get(sym, 0) + 1
                msg += f" ({self.errors[sym]}/{MAX_CONSECUTIVE_ERRORS})"
            self.err_msg[sym] = msg

    def _quotes(self) -> dict:
        """열린 포지션 코인의 현재가 (락 밖에서 호출). 거래소 장애 코인은 빠지고, 청산할 땐 마지막으로 본 가격을 쓴다."""
        with self.lock:
            open_ = [s for s, b in self.slots.items() if b.side != "flat"]
        out = {}
        for s in open_:
            try:
                out[s] = float(self.ex.fetch_ticker(s)["last"])
            except Exception:
                pass
        return out

    def _flatten(self, kind: str, reason: str, quotes: dict | None = None, only: set | None = None) -> None:
        """락 안에서 호출. 열린 포지션을 청산한다 (only 가 있으면 그 코인만)."""
        for s, b in self.slots.items():
            if b.side == "flat" or (only is not None and s not in only):
                continue
            price = (quotes or {}).get(s) or self.prices.get(s)
            if price is None:
                self.error = f"cannot flatten {s}: no price available"
                log.error(self.error)
                continue
            b.rebalance(Signal("flat", 0.0, kind), price)
            self._record(s, kind, price, reason=reason)

    def _record(self, sym: str, kind: str, price: float, j=None, sig=None, reason: str = "") -> None:
        row = {"ts": time.time(), "kind": kind, "symbol": sym, "price": price,
               "candle_ts": self.last_ts.get(sym) and self.last_ts[sym] / 1000,  # 판단 근거가 된 마지막 마감 캔들의 시작 시각(초)
               "position": self.slots[sym].side, "equity": self._equity(), "reason": reason}  # equity = 포트폴리오 전체
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
        tmp = self.data_dir / "jev_state.json.tmp"
        tmp.write_text(json.dumps({"running": self.running, "start_equity": self.start_equity, "prices": self.prices,
                                   "slots": {s: b.snapshot() for s, b in self.slots.items()}}))
        os.replace(tmp, self.data_dir / "jev_state.json")

    def _load(self) -> None:
        if not self.data_dir:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            d = json.loads((self.data_dir / "jev_state.json").read_text())
            slots = {s: PaperBroker.restore(v) for s, v in d["slots"].items()}
            state = (d["running"], d["start_equity"], d["prices"])
        except FileNotFoundError:
            if (self.data_dir / "state.json").exists():  # 단일 코인 시절의 파일: 구조가 달라 이어받지 않고 그대로 둔다
                log.info("legacy single-coin state.json ignored (kept); starting a fresh multi-coin account")
            return
        except (ValueError, KeyError, OSError) as e:  # 손상된 파일: 조용히 지우지 않고 크게 알린다
            log.error("jev_state.json unreadable, starting fresh (file kept): %s", e)
            self.error = f"jev_state.json unreadable, started fresh: {e}"
            return
        if set(slots) != set(self.symbols):
            log.warning("SYMBOLS differs from the saved portfolio %s: keeping the saved one", list(slots))
        self.slots, self.symbols = slots, list(slots)  # 저장된 포트폴리오가 우선 (도중에 구성이 바뀌면 자본 배분이 어긋난다)
        self.running, self.start_equity, self.prices = state
        log.info("restored: %d slots running=%s equity=%.2f", len(slots), self.running, self._equity())
