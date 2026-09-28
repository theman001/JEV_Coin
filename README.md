# JEV Coin

가상화폐 단타(롱/숏) 모의매매 봇. 지표 계산은 파이썬 코드가, 롱/숏 확률 판단은 [Jev AI (TypeSafe)](https://docs.typesafe.ai)가 하고, 웹에서 실행/중지·코인 선택·지표·판단·수익률을 모니터링합니다.

> ⚠️ **모의매매(드라이런) 전용입니다.** 실제 주문 기능은 의도적으로 구현하지 않았고 `LIVE=1`로 실행하면 거부합니다. Jev가 수익을 낸다는 검증된 근거는 아직 없으니, 모의 결과를 충분히 쌓아 확인한 뒤에 다음 단계를 결정하세요. 자세한 내용은 [CLAUDE.md](CLAUDE.md), [JEV_AI_GUIDE.md](JEV_AI_GUIDE.md).

## 웹 화면

- **매매 실행 / 중지** — 중지하면 보유 포지션을 청산하고 멈춥니다.
- **코인 선택** — `SYMBOLS` 목록에서 선택 (변경 시 보유 포지션 청산).
- **실시간 지표** — 종가, EMA20/50, RSI, ATR%, 변동성·거래량 배율 등 + Jev에 전달되는 라벨.
- **판단 모니터링** — 캔들 마감마다 Jev 판단(롱/숏/관망, 확신도, 반전위험)과 실제 실행 결과, 사유.
- **수익률** — 총 수익률, 실현/미실현 손익, 자산 곡선, 청산 거래 내역, 승률.

## OMV(Docker Compose)로 배포

1. 소스는 GitHub 공개 저장소 `theman001/JEV_Coin`에 있습니다. (비공개로 바꾸면 [docker-compose.yml](docker-compose.yml)의 context를 `https://<TOKEN>@github.com/...` 형태로)
2. OMV 웹 GUI → **Services → Compose → Files → 추가**.
   - 이름: `jev-coin`
   - 내용: [docker-compose.yml](docker-compose.yml)을 그대로 붙여넣기 (이미 `theman001/JEV_Coin`을 가리킴)
   - 환경변수(.env): [.env.example](.env.example)을 복사해 `TYPESAFE_API_KEY`, `WEB_PASSWORD`를 채워 붙여넣기
3. **Up** — 처음에는 git에서 소스를 받아 이미지를 빌드한 뒤 실행합니다 (ARM 보드에서 수 분 소요).
4. 브라우저에서 `http://<보드IP>:8787` → `WEB_USER` / `WEB_PASSWORD`로 로그인.
5. 코인을 고르고 **매매 실행**.

- 코드를 수정해 다시 push한 뒤에는 **Build**(또는 Pull and build) → **Up**.
- 호스트에 `git`이 필요합니다 (`apt install git`).
- 모의계좌 상태와 판단 이력은 `jev-data` 볼륨(`state.json`, `judgments.jsonl`)에 저장돼 재시작해도 이어집니다. 모의계좌를 초기화하려면 컨테이너를 내리고 볼륨을 삭제하세요.
- 웹은 HTTP Basic 인증(평문 HTTP)입니다. 집 내부망에서만 쓰고 인터넷에 포트를 열지 마세요.

## 로컬 개발

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q                       # 테스트 (네트워크·키 불필요)
set -a; . ./.env; set +a
.venv/bin/python -m src.main --once                 # 웹 없이 폴링 1회 (Jev 연결 확인)
.venv/bin/python -m src.main                        # 웹 서버 + 엔진 → http://localhost:8787
```
