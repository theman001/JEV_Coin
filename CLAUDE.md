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
10. **성과 주장은 사전 고정 설계로만.** 규칙 파라미터·Jev 임계값·프롬프트 문구를 정하고 → 결과를 보고 → 바꾸는 순서를 금지한다. 바꾸려면 새 사전 설계 + 표본 외 + 다중검정(Reality Check). 룩어헤드는 테스트로 상시 검사하고, 검사 자체의 검출력도 변이 검사로 확인한다. Jev 에 주는 백테스트 상태는 코인명·날짜를 가린다.

## 아키텍처

```
src/
  data.py     ccxt로 닫힌 OHLCV 수집 (공개 엔드포인트, 키 불필요) + stale 검사
  state.py    pandas + ta로 지표 계산 → 압축 State(의미 라벨 + 현재 포지션 라벨 + 수수료 커버 여부)
  judge.py    Jev 호출 (원자적 질문 N개 병렬) + 오프라인용 FakeJudge, 실패 시 SAFE(flat)
  policy.py   Judgment → Signal(long/short/flat + size) + 리스크 거부권 + 비용/손절 상수
  broker.py   PaperBroker (드라이런 체결, 비용 차감, 코드 손절)
  strategies.py  문헌 기반 차트 규칙 5종(SMA 교차±밴드, 채널 돌파, RSI>50, 필터 5%) → 보유 신호 0/1. close[t] 까지만 사용(룩어헤드 없음, 롱 전용). 라이브 봇은 아직 안 씀
  backtest.py    백테스트 하네스(사전 고정 설계): 다음 봉 시가 체결·비용, 5코인×1h/4h/1d, Reality Check, 표본 내→외, Jev 진입 필터 평가. 실행은 개발용(`python -m src.backtest`)
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

# 백테스트 (Binance 공개 데이터, 캔들은 data/candles/ 에 12시간 캐시. 결과·Jev 캐시는 data/backtest/ — 모두 gitignore)
.venv/bin/python -m src.backtest --jev-tfs                      # 규칙만, 1h/4h/1d (약 1분)
set -a; . ./.env; set +a; .venv/bin/python -m src.backtest --tfs 4h 1d --jev-tfs 4h 1d   # + Jev 진입 필터 (약 7,400회 호출, ~6분, ≈$0.1)
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

## 모의투자 기록

### 백테스트 (2026-09-28, 상세는 [STRATEGY_RESEARCH.md](STRATEGY_RESEARCH.md) §9)
- 규칙만: 4h·1d 추세추종은 **낙폭을 -80% → -45%대로** 줄이고 샤프를 올렸지만(4h SMA 20/50 연 24.9%/샤프 0.77, 1d RSI>50 연 22.2%/0.73, 단순 보유 10.3%/0.48) **Reality Check p=0.14~0.75 로 수익 우위는 우연과 구분되지 않음**. 1h 는 비용·휩쏘로 붕괴(RSI>50 연 -42%). 이 규칙들은 보유 기간이 수일이라 스윙이지 단타가 아니다.
- **Jev 진입 필터(Noul, 임계 0.5)는 가치 없음**: 사전 지정 1차 지표 Spearman 4h -0.083(p<0.001, 음수), 1d -0.052(p=0.06); 규칙별 층화(사후) 결과도 방향 불일치·효과 |ρ|<0.05; 포트폴리오 Reality Check p=0.445/0.951; 진입을 55~65% 차단해 낙폭은 줄지만 수익도 줄어듦. Noul 값이 0.37~0.61 에 몰려 모델이 이 질문에 확신이 없음.
- 한계: 생존 편향(현재 인기 5코인), 특정 5년, Binance 데이터(Upbit 아님), Jev 질문 문구 1개만 시험.

### 2026-09-28 첫 2시간 런 (14:47~16:47 KST, SOL/USDT 5m, 시작 100 USDT, 실제 Jev, 왕복 비용 0.14% 반영)
- **최종 수익률 +0.013%** (100.0135 USDT). 같은 기간 SOL 단순 보유 -0.764%. 청산 거래 2건(둘 다 숏, 1승 1패), MDD 0.068%, 포지션 보유 43%.
  - 거래 1: 숏 118.78→118.74, 유리한 변동 0.034% < 비용 0.14% → 순손실. 거래 2: 숏 118.62→118.2(+0.354%) → 순이익. 수익은 사실상 거래 1건이 만든 것.
- **정확도** (Jev 방향 판단이 15분 뒤 방향과 일치): SOL 12/20 = **60.0%** (95% CI 39~78%, 50% 대비 p=0.25). 5분 기준 12/22 = 54.5%.
  - 기준선: 같은 구간 **"항상 숏" = 64%** (그 구간에서 가격이 내려간 비율), 직전 3캔들 모멘텀 = 50%. → Jev 는 "항상 숏"보다 낫지 않았다.
  - 확장 표본(코인 5종, 사후 재구성 88건): 51.1% (CI 41~61%), 모멘텀 52.3%. 코인별 BTC 31%·XRP 39%·DOGE 68% 는 표본이 작아(n=15~20) 노이즈 범위.
  - 판단 분포: 25건 중 숏 23·플랫 2·롱 0 (하락 구간이었음). 확신도≥0.6 은 13건, 정책 통과 후 진입 11건. 확신도별 정확도는 ≥0.6 5/8, <0.6 7/12 → 확신도가 정확도를 구분한다는 증거 없음.
- **결론: Jev 의 예측 우위 근거는 확인되지 않았다.** 표본이 작아 "우위 없음"을 입증한 것도 아니다(신뢰구간이 넓음). 수익은 하락장 + 숏 편향 + 거래 2건의 결과로 해석하고 성과로 주장하지 않는다. 롱 거래는 0건이라 Upbit(롱 전용) 조건의 성과는 이번 런으로 알 수 없다.
- **채점 정의 (재현·비교용)**: 대상 = Jev 판단 중 long/short 전부(확신도 게이트 무관, flat 제외, 같은 코인·캔들 재판단은 첫 판단만). 정답 = 판단한 캔들 종가 대비 h캔들 뒤 **마감된** 캔들 종가 방향과 일치(h=3 주 기준, h=1 보조, 동일가 제외). 신뢰구간은 Wilson, 유의성은 50% 대비 단측 이항검정. 기준선 = 직전 3캔들 수익률 부호 추종 / 항상 숏 / 항상 롱.
- 원자료·스크립트: 로컬 `data/paper100-2h/` (gitignore) — `judgments.jsonl`, `state.json`, `driver.py`(실행), `report.py`(채점), `run/`(진행 로그·최종 스냅샷).

## 추후 일감 (Backlog) — 기록만, 미착수

### Upbit 실거래 연동 (실제 자동 투자)

**핵심 제약 — Upbit 은 현물 전용이다: 롱(매수 보유)은 가능, 숏(공매도·마진·선물)은 불가.**
→ 봇은 **롱 전용**이 된다. Jev `long` → 매수, `short`/`flat` → 매도해 현금 대기. `short` 판단은 "하락장 회피/청산 신호"로만 쓰이고, 하락에서 수익을 내는 절반은 사라진다. 현재 모의투자(Binance, 롱/숏 양방향) 결과는 롱 거래만 따로 봐야 Upbit 에 해당한다.

확인된 사실 (2026-09-28 조사):
- 원화마켓 수수료 0.05% (시장가·지정가 동일 — 업비트 고객센터 기준, 계정 등급/이벤트는 착수 시 재확인), 최소 주문금액 5,000 KRW (공식 docs), 호가 단위는 가격대별 표 (지정가는 반올림 필요).
- 요청 제한(공식): 시세 조회 초당 10회/IP, 주문 초당 12회·기본 초당 30회/pocket. 429 → 다음 초에 재시도, 418 → 일시 차단. 응답 헤더 `Remaining-Req`.
- 인증: JWT(access_key, nonce, 파라미터가 있으면 query_hash=SHA-512). ccxt `upbit` 가 JWT·rate limit 을 처리 → 자체 구현 불필요.
- **미확인**: API 키 발급 시 허용 IP 등록 요구 여부(이번 조사에서 공식 문서로 확인 못 함). 요구된다면 OMV 보드의 공인 IP(유동 IP 여부)가 문제가 된다 — 착수 첫 단계에서 확인.

설계 메모:
- 시세·지표는 현재 Binance USDT 캔들이다. Upbit KRW 마켓(KRW-SOL 등) 캔들로 통일하는 것을 권장 (Binance 지표로 Upbit 에서 체결하면 김치프리미엄 갭이 판단과 체결 사이에 낀다). 수수료·변동성 게이트(`COST_PER_SIDE`, ATR%)와 코인별 ATR 표도 KRW 마켓 기준으로 다시 측정.
- `PaperBroker` 와 같은 인터페이스(`rebalance`/`check_stop`/`equity`…)의 `UpbitBroker`(ccxt). 주문은 idempotent(`identifier`), 부분체결·미체결 처리, **체결/잔고를 거래소에 재조회해서 포지션을 갱신**(거래소가 진실, 내부 상태는 캐시). 롱 전용이므로 `side` 는 long/flat 만.
- 코드 손절은 실거래에서 갭·슬리피지 손실이 더 크다 (10초 폴링). 시장가 매도 + 체결 확인, 필요 시 지정 손절 주문 병행 검토.
- 실거래 안전장치(필수): `LIVE=1` 외에 별도 확인 절차, 1회 주문·일일 손실·총 투입액 상한(소액부터, 예: 주문당 5,000~10,000 KRW), 킬스위치, 웹에서 실거래 모드를 크게 표시, 시작 전 잔고 대조. **웹 인증 강화 선행** — 현재 Basic+평문 HTTP·실패 횟수 제한 없음이므로 실주문 제어에는 부족 (HTTPS 리버스 프록시, 로그인 시도 제한).
- 원칙 7 유지: 실주문은 사용자가 요청·확인했을 때만 켠다.

진입 게이트(착수 전 조건): 롱 전용 조건에서 수수료 후 양의 기대값이 **통계적으로 확인**될 것. 현재까지 Jev 예측 우위의 근거는 없다 (외부 백테스트에서 우위 미확인; 자체 2시간 모의투자 결과는 아래 기록 참고). 근거가 없으면 실거래 진행 여부부터 재검토.

권장 순서: ① Upbit KRW 캔들로 롱 전용 백테스트/다일 모의투자 → ② `UpbitBroker` + 초소액 실거래 → ③ 점진 확대.

### 기타
- **전략 조사: [STRATEGY_RESEARCH.md](STRATEGY_RESEARCH.md)** — 문헌상 근거는 일봉·저빈도 추세추종 계열, 5분봉 방향성 단타 근거는 못 찾음. 후보 규칙의 파이썬 수치화 명세와 실측(룩어헤드 검사 통과, 봉 주기별 비용 부담)이 들어 있다. 계산 프로토타입은 로컬 `data/research/strategy_proto.py`(gitignore).
- (완료) 백테스트 하네스 — `src/backtest.py`. 다음 후보: ① Upbit 원화마켓 캔들로 재검증(생존 편향 완화를 위해 코인 확대), ② 롱 전용 4h/1d 추세추종을 라이브(모의) 봇에 연결, ③ Jev 를 국면 분류 등 다른 용도로 시험(새 사전 설계 + 다중검정 필수 — 결과를 보고 문구를 바꾸는 것은 금지).
