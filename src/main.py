"""엔트리포인트: 엔진(백그라운드 스레드) + 웹(메인 스레드). 설정은 환경변수 (.env.example 참고)."""
import argparse
import json
import logging
import os
import signal
import sys
import threading

from .data import make_exchange
from .engine import Engine
from .judge import FakeJudge, JevJudge
from .trend import TrendEngine
from .web import make_server

# MODE=jev(기본): Jev 판단 단타 봇, MODE=trend: 4h SMA 20/50 추세추종 포트폴리오 (STRATEGY_RESEARCH.md §14). 환경변수를 안 주면 모드별 기본값
DEFAULTS = {
    "jev": {"symbols": "SOL/USDT,ETH/USDT,BTC/USDT,XRP/USDT,DOGE/USDT", "exchange": "binance", "timeframe": "5m", "poll": "10", "equity": "100"},
    "trend": {"symbols": "BTC/KRW,ETH/KRW,SOL/KRW,XRP/KRW,DOGE/KRW", "exchange": "upbit", "timeframe": "4h", "poll": "60", "equity": "1000000"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="웹 없이 폴링 1회 실행 후 스냅샷 출력 (동작 확인용)")
    ap.add_argument("--symbol", help="시작 코인 (기본: SYMBOLS의 첫 번째)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if os.getenv("LIVE") == "1":  # 실주문 경로는 프로토타입에 없다. 조용히 드라이런으로 떨어지지 않도록 거부.
        sys.exit("LIVE=1 is not supported: this prototype is dry-run only")

    mode = os.getenv("MODE", "jev")
    if mode not in DEFAULTS:
        sys.exit(f"MODE must be one of {sorted(DEFAULTS)}, got {mode!r}")
    d = DEFAULTS[mode]
    symbols = [s.strip() for s in os.getenv("SYMBOLS", d["symbols"]).split(",") if s.strip()]
    common = dict(timeframe=os.getenv("TIMEFRAME", d["timeframe"]), interval=float(os.getenv("POLL_SECONDS", d["poll"])),
                  equity=float(os.getenv("PAPER_EQUITY", d["equity"])), data_dir=None if args.once else os.getenv("DATA_DIR", "data"))
    ex = make_exchange(os.getenv("EXCHANGE", d["exchange"]))
    if mode == "trend":
        engine = TrendEngine(ex, symbols, **common)
        label = engine.label
    else:
        if os.getenv("TYPESAFE_API_KEY"):
            judge = JevJudge()
        else:
            logging.warning("TYPESAFE_API_KEY not set: using FakeJudge (NOT Jev)")
            judge = FakeJudge()
        engine = Engine(ex, judge, symbols, **common)
        label = f"judge={judge.name}"
    if args.symbol:
        engine.set_symbol(args.symbol)  # trend 모드는 ValueError (고정 포트폴리오)

    if args.once:
        engine.start()
        engine.step()
        snap = engine.snapshot()
        snap.pop("curve")
        print(json.dumps(snap, ensure_ascii=False, indent=1))
        return 0

    user, password = os.getenv("WEB_USER", "admin"), os.getenv("WEB_PASSWORD", "")
    if not password:  # 시작/중지를 제어하는 웹이므로 비밀번호 없이 열지 않는다
        sys.exit("WEB_PASSWORD is empty: set it in .env (the web UI can start/stop trading)")

    threading.Thread(target=engine.run_forever, daemon=True, name="engine").start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop: PID 1은 기본 핸들러가 없어 명시 필요
    server = make_server(engine, "0.0.0.0", int(os.getenv("PORT", "8787")), user, password)
    logging.info("web ui: http://0.0.0.0:%s  (%s, symbols=%s, running=%s)", server.server_port, label, symbols, engine.running)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        logging.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
