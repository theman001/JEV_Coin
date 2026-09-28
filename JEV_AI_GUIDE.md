# Jev AI (TypeSafe AI) 가이드

> 조사일: 2026-09-28. Jev는 2026-09-15 출시된 신생 모델이라 문서·수치가 바뀔 수 있다. 출처는 **[공식]** / **[제3자]** 로 표기했고, 제3자 수치는 검증 전까지 참고용으로만 쓴다.

## 0. 혼동 방지

여기서 "Jev"는 **TypeSafe AI의 System One 모델**만 뜻한다. 동명의 게임 캐릭터, 경제학 용어, 그 밖의 동명 서비스와는 무관하다. (Jev 이름을 붙인 크립토 "내러티브" 기사도 검색에 섞여 나오는데 검증하지 않았으니 기술 자료로 쓰지 말 것.) 검색 시 `TypeSafe`, `System One`, `Noul/Choice/Score`를 함께 쓸 것.

## 1. Jev란 무엇인가

- **TypeSafe AI의 첫 System One 모델.** 상태(state)와 타입이 지정된 질문(question)을 보내면, **텍스트가 아닌 타입 있는 답 + 보정된 확률**을 돌려준다. 코드가 파싱 없이 바로 쓴다. [공식: [docs](https://docs.typesafe.ai/introduction), [System One](https://docs.typesafe.ai/concepts/system-one)]
- **System One**: 빠르고 구조화된 판단(분류·라우팅·점수화)에 특화된 모델 클래스. 코드 생성, 글쓰기, 추론 설명은 하지 않는다. 카너먼의 "System 1(직관)"에서 따온 이름.
- 한 번의 forward pass로 **여러 질문을 병렬 평가**한다 (질문을 추가해도 지연이 거의 늘지 않고 질문 토큰 비용만 추가).
- 입력은 **텍스트 전용**(문자열 / JSON 객체 / 텍스트 배열). 이미지·오디오·영상 미지원. 주 학습 언어는 영어 (타 언어는 정확도 낮음).

### 스펙 요약

| 항목 | 값 | 출처 |
|---|---|---|
| 엔드포인트 | `POST https://api.typesafe.ai/v1/systemone` | 공식 |
| 인증 | `Authorization: Bearer $TYPESAFE_API_KEY` | 공식 |
| 모델 | `jev-latest` (= `jev-1.13.0`, 2026-09-15) | 공식 / 제3자 |
| 응답 지연 | 70–500ms (벤더 주장) | 공식 블로그 |
| 가격 | 입력 $0.042 / 1M 토큰, 출력 무료 | 공식 블로그 / 제3자 |
| 컨텍스트 | 총 64K 토큰, state 예산 32K (state + 가장 긴 질문) | 제3자 ([systemonemodels.org](https://systemonemodels.org/models/jev/)) |
| 레이트 리밋 | ~1,200 req/min, 25만 tok/s (고지 없이 변경 가능) | 제3자 |
| 가입 | console.typesafe.ai, 대기열 없음 (2026-09-20 해제) | 제3자 |

## 2. 세 가지 프리미티브

한 요청에 이름 붙인 질문을 여러 개 담는다: `questions: {"이름": {type: ...}, ...}`.

### 2.1 Noul — 예/아니오 확률

- 요청: `type: "noul"`, `instructions`(참/거짓을 판단할 문장), 선택 `criteria: {"true": "...", "false": "..."}`
- 응답: `noul` = P(yes), 0~1. **별도 confidence 없음** (0.5 = 미정, 값 자체가 확신도).
- 작성 팁: 높은 값이 "예"가 되도록 쓴다. 부정형("~가 없는가") 금지, 한 질문에 조건 하나만. 임계값은 오탐/미탐 비용에 따라 코드에서 정한다.

```python
Noul(instructions="Is momentum bullish on the 1h and 4h timeframes?")
```

### 2.2 Choice — 옵션 중 하나 선택 (최대 255개)

- 요청: `type: "choice"`, `instructions`, `criteria: {옵션명: 설명}`
- 응답: `choice`(최고 확률 옵션), `probabilities`(전 옵션, 합 = 1), `confidence`(0~1, 분포가 얼마나 뾰족한지).
- 롱/숏/관망 같은 **상호 배타 결정**에 사용.

```python
Choice(
    instructions="Which position fits the current market state?",
    criteria={"long": "...", "short": "...", "flat": "..."},
)
```

### 2.3 Score — 순서 있는 척도 (2~10단계)

- 요청: `type: "score"`, `instructions`, `criteria: [레벨0 설명, 레벨1 설명, ...]` (낮음→높음)
- 응답: `score`(확률 가중 평균 레벨 번호, 레벨 사이 소수 가능. 예: 3단계면 0~2), `probabilities`(레벨별), `confidence`, `legend`.
- 소수 점수는 순위 매기기에, 반올림은 이산 판단에 쓴다.
- ⚠️ **Score의 소수 값은 수치를 보간한 것이 아니다.** 아래 §4의 수치 보정 약점 참고.

### 2.4 Confidence (Choice/Score)

- 확률 **분포의 집중도**이지 **정답일 확률이 아니다**. 낮으면 겹치는 레벨 설명 / 다차원 질문 / 정보 부족 신호.
- 공식 문서의 핵심 경고: 보정(calibration)은 **예측 묶음 단위**에서만 성립하고, 개별 답에는 정확도 보장이 없다.
- 권장 용법: 답 = "무엇을", confidence = "행동할지 말지"의 두 번째 축 (confidence-gated routing).

## 3. LLM 대비 장점

| | LLM | Jev (System One) |
|---|---|---|
| 출력 | 자유 텍스트 → 파싱 필요 | 타입 보장된 값 (환각·포맷 오류 없음) |
| 지연 | 초 단위 | 70–500ms (벤더 주장) |
| 비용 | 출력 토큰 과금 | 출력 무료, 입력 $0.042/M |
| 불확실성 | 말로만 자신감 | 확률 + confidence를 매 답변에 제공 |
| 일관성 | 샘플링마다 흔들림 | 의미가 비슷한 입력에 유사한 출력 (단, §5 비결정성 주의) |
| 다중 판단 | 프롬프트/호출 반복 | 한 호출에 질문 N개 병렬 |

※ 속도·비용 배수(예: 40~200x)는 TypeSafe 자체 워크플로 평가이며, 공식 블로그도 "실제 이득의 상단일 수 있고 자사 팀이 만든 워크플로"라고 밝힌다.

## 4. 공식 문서의 한계 (Jev 1.13 "jaggedness")

[공식: [model-jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)]

| 약점 | 내용 | 공식 우회책 |
|---|---|---|
| **산술** | "Jev is not a calculator." 세기·수치 비교·근접도 판단 불안정. Score 레벨 사이를 보간해 정확한 수치를 복원할 수 없음. 숫자 표현(hex 등)은 의미 표현보다 성능 낮음 | 산술은 코드에서. 수치는 코드로 의미 라벨("RSI overbought")로 바꿔서 전달 |
| **날짜/시간** | 날짜를 순서 있는 수량이 아니라 텍스트로 읽음 | 날짜 비교·간격은 코드에서, Choice 옵션으로 넘긴다 |
| 간접 추론 | 이중부정·다단 추론 성능 저하 | 질문을 직접적·리터럴하게 |
| 리터럴 해석 | 부정어·범위어·암묵 조건을 문자 그대로 해석 | 조건 명시, 모호한 질문은 분리 |
| 컨텍스트 팽창 | 무관한 정보가 방해 요소로 작용 | 코드에서 걸러 필요한 필드만 전달 |
| 적대적 입력 | state를 적대적으로 보지 않음 (프롬프트 인젝션 가능) | 명시적 기준, 사전 테스트. **뉴스/SNS 텍스트를 state에 넣을 땐 특히 주의** |
| 구조적 불변성 없음 | 서로 다른 질문의 P(yes)+P(no)=1 보장 없음. 프리미티브 간 임계값 혼용 금지 | 질문 의미를 정확히 표현 |
| 텍스트 생성 | 불가 | Choice 사용 |

## 5. 트레이딩 자동화 시 주의점 (가장 중요)

### 5.1 산술 취약 → 지표 계산은 전부 코드에서

Jev에 원시 OHLCV 숫자를 던지고 "오를까?"를 묻는 설계는 §4의 약점에 정면으로 걸린다. **RSI·EMA·ATR·펀딩비 등은 pandas/ta로 계산하고, Jev에는 압축된 의미 있는 state(수치 + 코드가 만든 라벨)만 준다.** 가격·주문 수량·손익 계산에 Jev 출력을 쓰지 않는다. 공개 프로젝트 조사(제3자 [gist](https://gist.github.com/drillan/6916b16e8ea31a8ec36c8f59d6483150))에서도 state는 <400 토큰으로 압축하는 것이 공통 패턴이다.

### 5.2 예측 우위(edge)의 증거가 아직 없다

[제3자: [TradeRank 백테스트](https://www.traderank.ai/blog/what-is-jev-typesafe), 단일 출처·미검증]

- 2026-02~06, 크립토+미국주식 8,631개 예측: 방향 적중률 **49.34%** (드리프트 공식 48.68%, p=0.66 → 유의미한 우위 없음).
- Brier score 0.3057 (동전 던지기 0.25보다 나쁨), 80% 구간의 실제 포착률 68% (**과신**).
- 동일 조건 3회 실행 수익률 +17.5% / −17.4% / −2.6% → 손익은 실력이 아니라 **노이즈**.
- 동일 요청 5회에서 확률이 최대 0.07 벗어남 → **비결정적**, 라이브 판단은 여러 번 호출해 평균 필요.

**결론: Jev를 "가격 예측기"로 믿고 실계좌에 투입하면 안 된다.** Jev의 실제 강점은 시장 국면 분류, 독성 주문 흐름 판별, 뉴스 해석 같은 *상황 판단*이며, 이 봇의 프로토타입은 **드라이런/페이퍼 트레이딩이 기본**이어야 한다.

### 5.3 "Jev judges, code executes" 아키텍처 (커뮤니티 공통 패턴)

```
시장 데이터(ccxt) → 코드로 지표 계산 → 압축 State
   → Jev (원자적 판단 N개, 병렬)
   → 결정론적 정책 엔진 (임계값·confidence gating)
   → 리스크 거부권(코드, 위임 불가) → 실행
```

- 판단은 **원자적**으로 쪼갠다 (regime / direction / 변동성 위험 등 각각 별도 질문).
- **임계값은 코드에 둔다.** confidence가 낮으면 포지션 포기가 아니라 **사이즈 축소**.
- 리스크(포지션 상한, 드로다운 한도, 킬스위치)는 **Jev가 절대 override 못 하는** 코드 레이어.
- Jev 장애/지연 시: 정지 또는 결정론적 폴백. **오래된 state로 주문하지 않는다.**
- 실주문은 키를 명시 설정했을 때만, 기본은 dry-run.

### 5.4 기타 운영 주의

- 레이트 리밋(~1,200/min)·비용은 낮지만, 429/타임아웃 재시도 정책과 데드라인을 명시한다 (SDK 기본: 재시도 2회, 지수 백오프).
- state에 외부 텍스트(뉴스, 트윗)를 넣으면 프롬프트 인젝션 경로가 된다. 넣지 않는 것이 기본, 넣는다면 별도 격리 질문으로.
- 짧은 시간 내 동일 state를 반복 질의하지 말고 결과를 캐싱(비결정성 때문에 반복 질의 시 값이 흔들림).

## 6. Python SDK 사용법

```bash
pip install typesafe-sdk        # 선택: typesafe-sdk[http2]
export TYPESAFE_API_KEY=...     # console.typesafe.ai 에서 발급
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # 비동기: AsyncTypeSafeClient

with TypeSafeClient() as client:
    r = client.system_one(
        state={"trend_1h": "up", "rsi_1h": "overbought", "funding": "elevated"},
        questions={
            "side": Choice(instructions="Best position now?",
                           criteria={"long": "...", "short": "...", "flat": "..."}),
            "crowded": Noul(instructions="Is the long side crowded?"),
            "risk": Score(instructions="Downside risk?", criteria=["low", "medium", "high"]),
        },
    )
```

> 문서 불일치 해소 (typesafe-sdk 0.7.2 설치본 확인): Quickstart의 `response.answers["x"].noul` 과 SDK 문서의 `response.nouls["x"]` / `.choices` / `.scores` 는 **둘 다 동작**하며 같은 답변 객체를 돌려준다. 예외는 `TypeSafeError` 계열, 기본 재시도 2회(`RetryPolicy`).

## 7. 출처

**공식 (TypeSafe)**
- [Introduction](https://docs.typesafe.ai/introduction) · [llms.txt 색인](https://docs.typesafe.ai/llms.txt) · [Quick Start](https://docs.typesafe.ai/introduction/quickstart)
- [System One](https://docs.typesafe.ai/concepts/system-one) · [State](https://docs.typesafe.ai/concepts/state)
- [Choice](https://docs.typesafe.ai/primitives/choice) · [Score](https://docs.typesafe.ai/primitives/score) · [Noul](https://docs.typesafe.ai/primitives/noul)
- [Jev 1.13 jaggedness (한계)](https://docs.typesafe.ai/model-jaggedness/jev-1.13) · [Python SDK](https://docs.typesafe.ai/sdk/python)
- [출시 블로그](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**제3자 (검증 필요)**
- [systemonemodels.org — 가격/한도](https://systemonemodels.org/models/jev/)
- [TradeRank — 트레이딩 백테스트](https://www.traderank.ai/blog/what-is-jev-typesafe)
- [drillan gist — Jev 트레이딩 프로젝트 조사](https://gist.github.com/drillan/6916b16e8ea31a8ec36c8f59d6483150)
- [buberlo/jev-trader — 레퍼런스 아키텍처](https://github.com/buberlo/jev-trader)
