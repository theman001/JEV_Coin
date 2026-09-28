# JEV Coin

가상화폐 모의매매 봇 모음. 컨테이너 3개가 같은 이미지·볼륨을 씁니다.

| 서비스 | 하는 일 | 웹 |
|---|---|---|
| `jev-coin` | 단타(롱/숏) 모의매매. 지표 계산은 파이썬 코드가, 롱/숏 확률 판단은 [Jev AI (TypeSafe)](https://docs.typesafe.ai)가 함 (Binance) | `:8787` |
| `jev-trend` | **4h SMA 20/50 추세추종 5코인 동일가중** 모의 포트폴리오, Upbit KRW, 롱 전용, Jev 없음. 낙폭 방어 용도의 전진 검증 ([설계 §14](STRATEGY_RESEARCH.md)) | `:8788` |
| `jev-liq` | Binance 강제청산 수집기 (앞으로 쌓는 데이터, 몇 달 뒤 검정용) | 없음 |

> ⚠️ **모의매매(드라이런) 전용입니다.** 실제 주문 기능은 의도적으로 구현하지 않았고 `LIVE=1`로 실행하면 거부합니다. Jev가 수익을 낸다는 검증된 근거는 아직 없으니, 모의 결과를 충분히 쌓아 확인한 뒤에 다음 단계를 결정하세요. 자세한 내용은 [CLAUDE.md](CLAUDE.md), [JEV_AI_GUIDE.md](JEV_AI_GUIDE.md).

## 웹 화면 (jev-coin)

- **매매 실행 / 중지** — 중지하면 보유 포지션을 청산하고 멈춥니다.
- **코인 선택** — `SYMBOLS` 목록에서 선택 (변경 시 보유 포지션 청산).
- **실시간 지표** — 종가, EMA20/50, RSI, ATR%, 변동성·거래량 배율 등 + Jev에 전달되는 라벨.
- **판단 모니터링** — 캔들 마감마다 Jev 판단(롱/숏/관망, 확신도, 반전위험)과 실제 실행 결과, 사유.
- **수익률** — 총 수익률, 실현/미실현 손익, 자산 곡선, 청산 거래 내역, 승률.

## 추세추종 화면 (jev-trend, `:8788`)

전략 vs 동일가중 단순 보유의 **수익률·최대낙폭**(1차 지표) 비교, 코인별 신호(SMA20/50·이격)·슬롯 자산, 캔들 마감 이력(체결 지연 포함), 청산 거래. **일시정지해도 보유 포지션은 유지**되고 재개하면 현재 신호로 맞춥니다. 수익 우위를 주장하지 않는 낙폭 방어 검증이며, 판정은 6개월·거래 50건 이후 1회입니다.

## OMV(Docker Compose)로 배포

1. 소스는 GitHub 공개 저장소 `theman001/JEV_Coin`에 있습니다. (비공개로 바꾸면 [docker-compose.example.yml](docker-compose.example.yml)의 context를 `https://<TOKEN>@github.com/...` 형태로)
2. OMV 웹 GUI → **Services → Compose → Files → 추가**.
   - 이름: `jev-coin`
   - 내용: [docker-compose.example.yml](docker-compose.example.yml)을 `docker-compose.yml`로 복사해 `environment:` 값을 직접 적은 뒤 그 내용을 붙여넣기 (이미 `theman001/JEV_Coin`을 가리킴)
     - `docker-compose.yml`은 API 키·비밀번호가 들어가는 **로컬 전용 파일**이라 `.gitignore` 대상입니다. 커밋/공유 금지.
     - 대안: 예시 파일을 그대로 붙여넣고, [.env.example](.env.example)을 채운 `.env`를 OMV 환경변수 칸에 입력해도 됩니다 (`${VAR}` 치환).
3. **Up** — 처음에는 git에서 소스를 받아 이미지를 빌드한 뒤 실행합니다 (ARM 보드에서 수 분 소요).
4. 브라우저에서 `http://<보드IP>:8787` → `WEB_USER` / `WEB_PASSWORD`로 로그인.
5. `jev-coin` 은 코인을 고르고 **매매 실행**. `jev-trend`(`:8788`)는 기동하면 바로 돌아갑니다. `jev-liq` 는 `docker logs jev-liq` 로 수신 여부를 확인합니다 (5분간 이벤트가 없으면 경고).
   - 필요 없는 서비스는 compose 에서 통째로 지우면 됩니다.

- 코드를 수정해 다시 push한 뒤에는 **Build**(또는 Pull and build) → **Up**.
- 호스트에 `git`이 필요합니다 (`apt install git`).
- 상태·이력은 `jev-data` 볼륨에 저장돼 재시작해도 이어집니다 (`state.json`·`judgments.jsonl` / `trend_state.json`·`trend.jsonl` / `liq/일자.jsonl`). 모의계좌를 초기화하려면 컨테이너를 내리고 볼륨을 삭제하세요.
- 웹은 HTTP Basic 인증(평문 HTTP)입니다. 집 내부망에서만 쓰고 인터넷에 포트를 열지 마세요.

## 로컬 개발

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q                       # 테스트 (네트워크·키 불필요)
set -a; . ./.env; set +a
.venv/bin/python -m src.main --once                 # 웹 없이 폴링 1회 (Jev 연결 확인)
.venv/bin/python -m src.main                        # 웹 서버 + 엔진 → http://localhost:8787
MODE=trend .venv/bin/python -m src.main --once      # 추세추종: Upbit 공개 시세로 1회 (주문 없음)
.venv/bin/python -m src.collect_liq                 # 청산 수집기 → data/liq/
```
