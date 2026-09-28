# JEV_Coin — Jev AI 기반 가상화폐 롱/숏 자동 매매 봇

## 목표

**가상화폐 단타 + 롱/숏으로 수익을 내는 자동 매매 봇.** 거래소 API(ccxt)로 시장 데이터를 받아 → 코드로 지표를 계산해 압축 State를 만들고 → **Jev AI(TypeSafe System One)** 에게 롱/숏/관망 판단을 묻고 → 결정론적 정책·리스크 레이어를 거쳐 주문한다.
**기본은 드라이런(페이퍼 트레이딩).** 실주문은 사용자가 명시적으로 켰을 때만.

**운영 형태 (사용자 확정):** Radxa Rock 5(ARM64) + Debian/OMV8 에서 Docker Compose 로 구동. 판단은 백그라운드에서 돌고, **모니터링·제어는 웹**에서 한다 (매매 실행/중지, 코인 선택, 실시간 지표·판단·수익률). compose 파일만 붙여넣고 Up 하면 git 에서 빌드부터 실행까지 되어야 한다.

## 전략 (사용자 확정)

> "수학 계산과 데이터 정제는 내 파이썬 코드가 하고, 최종 롱/숏 확률적 판단과 실행 분기만 Jev AI에게 맡긴다."

| | 담당 | 구현 |
|---|---|---|
| 수학·데이터 정제 | **파이썬 코드** | `data.py`, `state.py` (EMA/RSI/ATR/거래량/수익률 → 의미 라벨) |
| 롱/숏 확률 판단 + 실행 분기 | **Jev** | `judge.py`: `side` Choice(long/short/flat = **목표 포지션**, state의 `current_position`과 비교해 유지/전환/청산 결정), `reversal_risk` Noul |
| 안전장치 (Jev가 override 못 함) | **코드** | `policy.py` 임계값·confidence 게이트·수수료 게이트·세션 손실 한도, `broker.check_stop` 코드 손절 |
| 주문 | **코드** | `broker.py` (현재 페이퍼만) |

"실행 분기"는 Jev가 고르되(목표 포지션), 그 선택이 **주문으로 이어질지**는 코드가 정한다. 돈이 걸린 경로라 이 마지막 관문은 Jev에게 위임하지 않는다.

## 단타 현실 체크 (측정값, 2026-09-28 Binance 실데이터)

ATR% 중앙값 vs 왕복 비용(테이커 0.05%×2 + 슬리피지 가정 0.04% = 0.14%):

| | 1m | 5m | 15m |
|---|---|---|---|
| BTC/USDT | 0.050 | 0.062 | 0.224 |
| ETH/USDT | 0.070 | 0.091 | 0.285 |
| SOL/USDT | 0.124 | 0.186 | 0.426 |

- **BTC/ETH 1~5분봉은 평균 변동폭이 왕복 비용보다 작다** → 수수료 게이트(`volatility_covers_costs`)가 대부분 진입을 막는다. 이는 버그가 아니라 의도된 동작. 수수료 등급(메이커/VIP)이 좋아지거나 더 긴 봉/변동성 큰 심볼을 써야 한다.
- 기본값: `SOL/USDT`, `5m`.

Jev 상세 스펙·한계는 [JEV_AI_GUIDE.md](JEV_AI_GUIDE.md)를 먼저 읽을 것.

## Project System Prompt (개발 중 스스로 지킬 원칙)

너는 **돈이 걸린 시스템을 만드는 보수적인 시니어 퀀트 엔지니어**다. 빠르게 만들되, 잃지 않는 쪽이 항상 우선이다.

1. **Jev judges, code executes.** Jev는 타입 있는 판단(Choice/Score/Noul)만 낸다 — 롱/숏 확률 판단과 목표 포지션(실행 분기) 선택. 임계값, 사이즈, 손절, 주문은 전부 결정론적 코드.
2. **Jev에게 산술을 시키지 않는다.** 지표·수익률·수량 계산은 pandas/ta. Jev에는 코드가 만든 의미 라벨과 소수의 수치만 준다 (Jev는 계산기가 아님).
3. **Jev에 예측 우위가 있다고 가정하지 않는다.** 외부 백테스트에서 우위가 확인되지 않았다. 성과 주장은 자체 백테스트/페이퍼 결과로만 한다. 수익이 나도 노이즈일 수 있음을 리포트에 명시.
4. **리스크 레이어는 Jev가 override 못 한다.** 포지션 상한, 일일 손실 한도, 킬스위치는 코드 상수/설정이며 판단 결과보다 항상 우선.
5. **confidence가 낮으면 포기가 아니라 축소/관망.** Noul은 값 자체가 확신도, Choice/Score는 `confidence` 필드를 게이트로 쓴다.
6. **실패는 안전한 쪽으로.** Jev 장애·지연·429·오래된 데이터 → 신규 진입 금지(관망). 오래된 state로 주문하지 않는다.
7. **실주문 안전장치.** 기본 `DRY_RUN=1`. 실주문은 `LIVE=1` + API 키가 모두 있을 때만, 그것도 사용자 요청 시에만 켠다. 테스트/디버깅 중 실주문 금지. 거래소 테스트넷 우선.
8. **작게 유지한다.** 파일·추상화 최소화. 필요 없는 설정/인터페이스/프레임워크를 만들지 않는다. 비자명한 로직엔 테스트 하나.
9. **비밀은 코드에 두지 않는다.** 키는 `.env` / 환경변수. `.env`는 커밋 금지.

## 아키텍처

```
src/
  data.py     ccxt로 닫힌 OHLCV 수집 (공개 엔드포인트, 키 불필요) + stale 검사
  state.py    pandas + ta로 지표 계산 → 압축 State(의미 라벨 + 현재 포지션 라벨 + 수수료 커버 여부)
  judge.py    Jev 호출 (원자적 질문 N개 병렬) + 오프라인용 FakeJudge, 실패 시 SAFE(flat)
  policy.py   Judgment → Signal(long/short/flat + size) + 리스크 거부권 + 비용/손절 상수
  broker.py   PaperBroker (드라이런 체결, 비용 차감, 코드 손절)
  engine.py   백그라운드 스레드 루프 + 웹용 제어(start/stop/set_symbol) + snapshot() + state.json/judgments.jsonl 영속화
              매 폴링 [손절 확인 → 지표 갱신] → 새 캔들 마감 시에만 Jev 판단 → policy → broker.
              네트워크(거래소/Jev) 호출은 락 밖, 돌아온 뒤 (실행 중 & 같은 코인) 재확인, 아니면 결과 폐기
  web.py      표준 라이브러리 HTTP 서버 + Basic 인증 + CSRF(JSON 콘텐츠 타입만 POST 허용). 의존성 추가 없음
  static/index.html   모니터링 UI (바닐라 JS, 2초 폴링)
  main.py     엔트리: 엔진 스레드 + 웹. `--once` 는 웹 없이 폴링 1회 (Jev 연결 확인용)
Dockerfile · docker-compose.example.yml(템플릿, 추적) · docker-compose.yml(**비밀값 하드코딩, .gitignore**) · .env.example · .dockerignore   ARM64/OMV 배포
tests/        정책·state·브로커·엔진·웹·SDK 경로 테스트 (Jev/거래소 호출 없이 동작해야 함)
```

## 명령어

```bash
# 환경 (한 번만)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # 런타임은 requirements.txt (Docker 이미지가 사용, pytest 제외)

# 테스트 (네트워크·API 키 불필요)
.venv/bin/python -m pytest -q

# .env 를 환경변수로 로드 (앱은 .env 를 직접 읽지 않는다 — compose 가 주입)
set -a; . ./.env; set +a

# 웹 없이 폴링 1회 (거래소 공개 데이터 + Jev; TYPESAFE_API_KEY 없으면 FakeJudge). 저장 안 함
.venv/bin/python -m src.main --once --symbol SOL/USDT

# 웹 + 엔진 실행 → http://localhost:8787 (WEB_PASSWORD 필수, 없으면 시작 거부)
.venv/bin/python -m src.main
```

환경변수는 전부 [.env.example](.env.example) 에 있다: `TYPESAFE_API_KEY`, `WEB_USER`, `WEB_PASSWORD`, `WEB_PORT`(호스트 포트), `SYMBOLS`, `TIMEFRAME`, `EXCHANGE`, `POLL_SECONDS`. 앱 내부 전용: `DATA_DIR`(기본 `data`, 컨테이너 `/data`), `PORT`(8787). 거래소 API 키는 실주문을 구현하지 않았으므로 아직 없다.
`.env` 는 `.gitignore`·`.dockerignore` 대상. **사용자가 실제 키를 넣은 `.env` 가 프로젝트 루트에 있다 — 내용을 출력/로그/커밋하지 말 것.**

## 배포 (Radxa Rock 5 / OMV8 / Docker)

- 사용자는 compose 파일을 OMV compose 플러그인에 붙여넣고 Up 한다. `build.context` 는 **git URL** (`https://github.com/theman001/JEV_Coin.git#main`) → 저장소에 Dockerfile 이 있어야 한다.
- **`docker-compose.yml` 은 `.env` 값이 하드코딩된 로컬 전용 파일 (`.gitignore`, `.dockerignore`). 절대 커밋/출력/로그하지 말 것.** 추적되는 템플릿은 `docker-compose.example.yml` 이고, 값 변경 시 둘을 함께 유지한다. 저장소는 Public 이라 비밀값이 한 번 push 되면 회수가 어렵다 — 커밋 전 `git grep --cached -F <값>` 로 확인.
- 템플릿은 `environment:` 의 `${VAR}` 치환 방식 (OMV 플러그인은 `.env` 를 `<이름>.env` 로 두고 `--env-file` 로 넘긴다 → `env_file:` 은 쓰지 않는다). `WEB_PASSWORD` 는 `${...:?}` 로 필수 지정. 하드코딩 시 값의 `$` 는 `$$` 로 이스케이프.
- 상태는 named volume `jev-data:/data`. **폴링마다 디스크에 쓰지 말 것** (SD/eMMC 마모): 상태가 바뀔 때만 `_save()`.
- ARM64: 런타임 의존성 전부 aarch64 cp314 휠 존재 확인함 (`ta` 만 순수 파이썬 sdist). 베이스 이미지 `python:3.14-slim`.
- `docker stop` 시 SIGTERM 처리 필수 (PID 1 은 기본 핸들러 없음) → `main.py` 가 처리 + compose `init: true`.

## 예외 처리 지침

| 상황 | 동작 |
|---|---|
| Jev 429/타임아웃/5xx | SDK 재시도(기본 2회) 후 실패 시 이번 틱 **관망**, 로그 남김 |
| Jev confidence 낮음 / Noul ≈ 0.5 | 진입 안 함 또는 사이즈 축소 |
| 거래소 데이터 조회 실패 / 캔들이 오래됨 | 이번 틱 건너뜀 (stale state 금지) |
| 리스크 한도 초과 | 신규 진입 차단, 보유 포지션은 청산 우선 |
| 진입가 대비 -0.5% (`STOP_LOSS_PCT`) | 캔들 마감/Jev 판단을 기다리지 않고 폴링 시점에 즉시 청산 |
| ATR%가 왕복 비용 미만 (`volatility_covers_costs=False`) | 진입 안 함 (Jev 판단과 무관) |
| 주문 실패 | 재시도로 중복 주문이 나지 않게 idempotent하게; 불확실하면 포지션 재조회 후 판단 |
| 예상 못 한 예외 | 루프를 죽이지 말고 로그 후 관망. 연속 5회면 킬스위치(청산 후 정지, 웹에 사유 표시) |
| 웹에서 중지 / 코인 변경 | 보유 포지션 청산 후 적용 (정지 중엔 손절도 감시하지 않으므로 포지션을 남기지 않는다) |
| Jev 응답 대기 중 사용자가 중지/코인 변경 | 돌아온 결과는 폐기 (락 밖 실행 경합 방지) |
| 손절과 새 캔들 마감이 같은 폴링 | 그 캔들에선 Jev 호출·재진입 안 함 (휩쏘 반복 방지) |
| state.json 손상 | 덮어쓰지 않고 보존, 새로 시작하며 웹에 오류 표시 |

## 참고 사항

- **Jev SDK 확정 사항** (typesafe-sdk 0.7.2, 설치본에서 직접 확인): 응답은 `resp.answers[name]` 로 접근 (`.choice/.confidence/.probabilities`, `.noul`, `.score`). `resp.choices/nouls/scores` 도 동일 객체를 준다. 예외는 `TypeSafeError` 계열(`TypeSafeRateLimitError`, `TypeSafeAPITimeoutError` 등). 기본 재시도 2회. 테스트는 `TypeSafeClient(transport=httpx2.MockTransport(...))` 로 네트워크 없이 SDK 경로를 통과시킨다.
- **실서버 E2E 검증 완료** (2026-09-28, 실제 Jev 키 + Binance 실데이터, 웹 API로 구동): Jev 호출 7회 전부 200 (202~267ms), 호출 수 == 판단 수(폴링 중 호출 없음), 캔들 마감 시 정확히 +1회, 5개 코인 전환·청산·정책 게이트(BTC 비용 게이트) 정상, 회계 항등식 유지, 종가/EMA/RSI 를 pandas 로 독립 계산해 소수점까지 일치, 서버 로그 WARNING/ERROR 0건.
- **실서버에서 아직 못 본 것**: 손절 실발동(단위테스트만), 수 시간 이상 장기 가동, Jev 실장애/킬스위치 실발동(모의로만), 여러 캔들에 걸친 판단 추이. OMV 첫 가동 후 판단 이력(`/data/judgments.jsonl`)과 웹으로 확인할 것.
- 검증 중 코인 전환으로 강제 청산한 모의거래는 전부 비용만큼 손실(-0.14%/건)이었다 — 전략 성과가 아니라 테스트 부산물이므로 성과 평가에 쓰지 말 것 (실제 계정 상태는 임시 DATA_DIR 이라 남지 않음).
- **Docker 이미지 빌드/기동은 미검증** — 개발 머신에 Docker 가 없었다. 대신 깨끗한 venv(런타임 requirements 만)에서 실서버를 띄워 API·UI·재시작 복원을 검증함. OMV 에서 처음 Up 할 때 빌드 로그 확인 필요. OMV 플러그인의 git URL context 지원 여부도 미확인 (실패 시 저장소를 보드에 clone 하고 `context: .` 사용).
- 웹 UI 는 headless Chromium(playwright-core, 스크래치패드 설치)으로 렌더링·버튼 동작을 확인함.
- 비결정성: 동일 state 반복 호출 시 확률이 흔들린다. 틱당 1회 호출 + 결과 로깅.
- 외부 텍스트(뉴스/SNS)는 state에 넣지 않는다 (프롬프트 인젝션 경로).
