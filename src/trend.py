"""추세추종 포트폴리오 모의 엔진 — STRATEGY_RESEARCH.md §14 (사전 고정 설계).

4h SMA 20/50 교차, 코인별 독립 슬롯(시작자본 ÷ N), 롱 전용, 손절 없음, 모의 체결만. Jev 는 쓰지 않는다.
신호는 백테스트와 같은 strategies.sma_cross, 슬롯은 PaperBroker 를 그대로 쓴다.
락 규칙은 engine.py 와 같다: 네트워크 호출은 락 밖에서, 돌아온 뒤 running 을 재확인해 아니면 결과를 버린다.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from .broker import PaperBroker
from .data import StaleData, fetch_closed
from .policy import COST_PER_SIDE, Signal
from .strategies import sma_cross

log = logging.getLogger(__name__)
FAST, SLOW = 20, 50  # 사전 고정 (§14.2). 바꾸려면 새 사전 설계
RULE = f"SMA {FAST}/{SLOW}"
MAX_CURVE = 3000  # 캔들 마감마다 1점 (4h 봉이면 약 1.4년)


def sma_signal(df) -> tuple[bool, float, float]:
    """마지막 마감 봉 기준 (보유 여부, SMA fast, SMA slow)."""
    if len(df) < SLOW:
        raise StaleData(f"need {SLOW} candles, got {len(df)}")  # 신규 상장 등: 이번 틱은 건너뜀
    c = df["close"]
    return bool(sma_cross(df, FAST, SLOW).iloc[-1]), float(c.rolling(FAST).mean().iloc[-1]), float(c.rolling(SLOW).mean().iloc[-1])


class TrendEngine:
    page = "trend.html"
    label = f"trend {RULE}"

    def __init__(self, ex, symbols: list[str], timeframe: str = "4h", interval: float = 60, data_dir=None, equity: float = 1_000_000.0):
        self.ex, self.symbols, self.timeframe, self.interval = ex, list(symbols), timeframe, interval
        self.data_dir = Path(data_dir) if data_dir else None
        self.lock = threading.RLock()
        self.start_equity = equity
        self.slots = {s: self._slot(s, equity / len(self.symbols)) for s in self.symbols}
        self.running = True  # 전진 검증: 첫 기동부터 돈다 (저장된 상태가 있으면 그 값)
        self.last_ts, self.start_px, self.prices, self.sig = {}, {}, {}, {}
        self.curve, self.events = [], deque(maxlen=60)  # curve: [ts, 전략 자산, 단순보유 배수] 캔들 마감마다
        self.dd = {"peak": equity, "mdd": 0.0, "bh_peak": 1.0, "bh_mdd": 0.0}
        self.polled_at = self.error = None
        self._load()

    @staticmethod
    def _slot(sym: str, equity: float) -> PaperBroker:
        b = PaperBroker(equity)
        b.symbol = sym
        return b

    # ---------- 제어 (웹 스레드) ----------
    def start(self) -> None:
        with self.lock:
            self.running = True
            self._save()

    def stop(self) -> None:
        """일시정지. 포지션은 유지한다 (§14.6): 이 규칙엔 감시할 손절이 없고, 청산하면 검증 대상이 달라진다."""
        with self.lock:
            self.running = False
            self._save()

    def set_symbol(self, symbol) -> None:
        raise ValueError("trend mode trades a fixed portfolio: no symbol selection")

    # ---------- 루프 (엔진 스레드) ----------
    def run_forever(self) -> None:
        while True:
            if self.running:
                try:
                    self.step()
                except Exception as e:  # step 은 코인별로 예외를 흡수한다. 방어용
                    log.exception("trend step failed")
                    self.error = f"{type(e).__name__}: {e}"
            time.sleep(self.interval)

    def step(self) -> None:
        """폴링 1회: 코인마다 [가격 → 마지막 마감 캔들 → (새 캔들이면) 신호 → 슬롯 재조정]. 한 코인의 실패가 다른 코인을 막지 않는다."""
        with self.lock:
            syms = list(self.symbols)
        errs, changed = [], False
        for sym in syms:
            try:
                changed |= self._step_coin(sym)
            except StaleData as e:  # 정상적인 건너뜀
                errs.append(f"{sym}: skip ({e})")
            except Exception as e:
                log.exception("%s step failed", sym)
                errs.append(f"{sym}: {type(e).__name__}: {e}")
        with self.lock:
            self.polled_at, self.error = time.time(), "; ".join(errs) or None
            eq, bh = self._equity(), 1 + self._bh()
            d = self.dd
            d["peak"], d["bh_peak"] = max(d["peak"], eq), max(d["bh_peak"], bh)
            d["mdd"], d["bh_mdd"] = min(d["mdd"], eq / d["peak"] - 1), min(d["bh_mdd"], bh / d["bh_peak"] - 1)
            if changed:  # 디스크 쓰기는 캔들이 마감돼 상태가 바뀔 때만 (SD/eMMC 마모 방지)
                self.curve = (self.curve + [[time.time(), eq, bh]])[-MAX_CURVE:]
                self._save()

    def _step_coin(self, sym: str) -> bool:
        price = float(self.ex.fetch_ticker(sym)["last"])
        df = fetch_closed(self.ex, sym, self.timeframe)
        ts, (hold, fast, slow) = int(df["ts"].iloc[-1]), sma_signal(df)
        tf_ms = self.ex.parse_timeframe(self.timeframe) * 1000
        with self.lock:
            if not self.running:
                return False
            self.prices[sym] = price
            self.start_px.setdefault(sym, price)
            self.sig[sym] = {"hold": hold, "fast": fast, "slow": slow, "candle_ts": ts / 1000}
            if ts == self.last_ts.get(sym):
                return False
            b, before = self.slots[sym], self.slots[sym].side
            b.rebalance(Signal("long" if hold else "flat", 1.0, RULE), price)  # 같은 방향이면 유지 (비용 없음)
            self.last_ts[sym] = ts
            row = {"ts": time.time(), "symbol": sym, "candle_ts": ts / 1000, "lag_s": time.time() - (ts + tf_ms) / 1000,  # 백테스트(다음 봉 시가 체결)와의 체결 지연
                   "fast": fast, "slow": slow, "hold": hold, "before": before, "after": b.side, "price": price, "equity": self._equity()}
            self.events.append(row)
            if self.data_dir:
                with open(self.data_dir / "trend.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
            return True

    # ---------- 조회 ----------
    def snapshot(self) -> dict:
        with self.lock:
            coins = []
            for s, b in self.slots.items():
                p, sg, sp = self.prices.get(s), self.sig.get(s), self.start_px.get(s)
                mark = p or b.entry
                coins.append({
                    "symbol": s, "price": p, "position": b.side, "entry": b.entry, "qty": b.qty, "pnl_pct": b.pnl_pct(mark),
                    "hold": sg and sg["hold"], "fast": sg and sg["fast"], "slow": sg and sg["slow"], "gap_pct": sg and (sg["fast"] / sg["slow"] - 1) * 100,
                    "candle_ts": sg and sg["candle_ts"], "equity": b.equity(mark), "start_equity": b.start_equity,
                    "bh_pct": ((1 - COST_PER_SIDE) * p / sp - 1) * 100 if p and sp else None})
            eq, bh = self._equity(), self._bh()
            trades = sorted((t for b in self.slots.values() for t in b.trades), key=lambda t: t["ts"])[-20:][::-1]
            return {
                "mode": "trend", "now": time.time(), "running": self.running, "rule": RULE, "timeframe": self.timeframe, "dry_run": True,
                "poll_seconds": self.interval, "polled_at": self.polled_at, "error": self.error, "coins": coins,
                "equity": eq, "start_equity": self.start_equity, "return_pct": (eq / self.start_equity - 1) * 100, "bh_return_pct": bh * 100,
                "mdd_pct": self.dd["mdd"] * 100, "bh_mdd_pct": self.dd["bh_mdd"] * 100,
                "exposure_pct": 100 * sum(b.side == "long" for b in self.slots.values()) / len(self.slots),
                "n_trades": sum(b.n_trades for b in self.slots.values()), "n_wins": sum(b.n_wins for b in self.slots.values()),
                "trades": trades, "events": list(self.events)[::-1], "curve": self.curve}

    # ---------- 내부 ----------
    def _equity(self) -> float:
        return sum(b.equity(self.prices.get(s) or b.entry) for s, b in self.slots.items())

    def _bh(self) -> float:
        """같은 시작 시점에 5코인을 동일가중으로 사서 들고 있었다면의 수익률 (진입 비용만 반영, 비율)."""
        r = [(1 - COST_PER_SIDE) * self.prices[s] / self.start_px[s] - 1 for s in self.slots if s in self.prices and s in self.start_px]
        return sum(r) / len(r) if r else 0.0

    def _save(self) -> None:
        if not self.data_dir:
            return
        tmp = self.data_dir / "trend_state.json.tmp"
        tmp.write_text(json.dumps({"running": self.running, "start_equity": self.start_equity, "slots": {s: b.snapshot() for s, b in self.slots.items()},
                                   "last_ts": self.last_ts, "start_px": self.start_px, "prices": self.prices, "curve": self.curve, "dd": self.dd}))
        os.replace(tmp, self.data_dir / "trend_state.json")

    def _load(self) -> None:
        if not self.data_dir:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            d = json.loads((self.data_dir / "trend_state.json").read_text())
            slots = {s: PaperBroker.restore(v) for s, v in d["slots"].items()}
            state = (d["running"], d["start_equity"], d["last_ts"], d["start_px"], d["curve"], d["dd"], d["prices"])
        except FileNotFoundError:
            return
        except (ValueError, KeyError, OSError) as e:  # 손상된 파일: 조용히 지우지 않고 크게 알린다
            log.error("trend_state.json unreadable, starting fresh (file kept): %s", e)
            self.error = f"trend_state.json unreadable, started fresh: {e}"
            return
        if set(slots) != set(self.symbols):
            log.warning("SYMBOLS differs from the saved portfolio %s: keeping the saved one", list(slots))
        self.slots, self.symbols = slots, list(slots)  # 저장된 포트폴리오가 우선 (도중에 구성이 바뀌면 검증 대상이 달라진다)
        self.running, self.start_equity, self.last_ts, self.start_px, self.curve, self.dd, self.prices = state  # prices: 재시작 직후 첫 폴링 전에도 마지막 가격으로 평가
        try:  # 캔들 마감 이력은 로그에서 복원 (재시작해도 화면에 남는다)
            self.events.extend(json.loads(ln) for ln in (self.data_dir / "trend.jsonl").read_text().splitlines()[-self.events.maxlen:])
        except (FileNotFoundError, ValueError):
            pass
        log.info("restored: %d slots running=%s equity=%.0f", len(slots), self.running, self._equity())
