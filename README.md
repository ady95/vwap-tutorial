# vwap-tutorial

위키독스 「AI로 개선하는 변형 VWAP 단타 자동매매 따라하기」의 예제 코드 저장소입니다.

- 위키독스: https://wikidocs.net/book/21914
- 부제: Python·Optuna·LightGBM을 활용한 매수 신호 최적화와 실전형 자동매매 시스템 구축

> 이 저장소는 교육용 예제입니다. 특정 투자 성과를 보장하지 않으며, 실제 계좌 주문은 코드에서 차단합니다.

## 준비

```bash
# uv 설치 후
uv sync                      # 기본 의존성
uv sync --extra ml           # 06~09장 (Optuna·LightGBM)
cp .env.example .env         # 키움증권 REST API 앱키·계좌 입력
```

키움증권 REST API를 쓰려면 키움 Open API 사이트에서 앱키를 발급받고, 실행할 PC의 공인 IP를 허용 IP로 등록해야 합니다.

## 구성

```text
src/vwap_trading/
├── backtest/    # 이벤트 기반 백테스트·비용 모델·성과 지표 (05장)
├── broker/      # 키움증권 REST API 클라이언트
├── data/        # 분봉 수집·정제·세션·종목 선정 (02장)
├── indicators/  # VWAP·기울기·상태·박스 중심선 (03장)
├── reporting/   # 차트 공통 설정
└── strategy/    # 기본 전략 S0·A: 신호·청산·진입 허용 (04장)
configs/         # 전략 설정 파일 (TOML)
examples/
├── chapter01/   # 장별 실습 예제
├── chapter02/
├── chapter03/
├── chapter04/
└── chapter05/
```

## 장별 예제

| 장 | 예제 | 내용 |
|---|---|---|
| 02 | `examples/chapter02/01_check_connection.py` | 키움 REST API 토큰·시세·분봉 조회 확인 |
| 02 | `examples/chapter02/02_collect_minute_bars.py` | 연속조회로 과거 1분봉 수집 → Parquet 저장 |
| 02 | `examples/chapter02/03_build_universe.py` | 거래대금 상위 보통주로 예제 유니버스 생성 |
| 02 | `examples/chapter02/04_make_selections.py` | 예제 종목 선정 입력(selections.csv) 생성 |

```bash
uv run python examples/chapter02/01_check_connection.py
uv run python examples/chapter02/02_collect_minute_bars.py 005930 --pages 50
uv run python examples/chapter02/03_build_universe.py --n 15
uv run python examples/chapter02/02_collect_minute_bars.py --universe data/sample/universe_20261030.csv
uv run python examples/chapter02/04_make_selections.py --universe data/sample/universe_20261030.csv
uv run pytest -q
```

| 장 | 예제 | 내용 |
|---|---|---|
| 03 | `examples/chapter03/01_session_vwap.py` | 세션 VWAP 계산과 손 검산 |
| 03 | `examples/chapter03/02_rolling_vs_session.py` | 세션·롤링 VWAP 반응 속도와 교차 횟수 비교 |
| 03 | `examples/chapter03/03_state_and_box.py` | 정규화 기울기 상태 판정과 박스 중심선 통계 |
| 03 | `examples/chapter03/04_plot_day.py` | 하루치 가격·기준선·횡보 구간 차트 |

```bash
uv run python examples/chapter03/01_session_vwap.py 005930 2026-10-29
uv run python examples/chapter03/02_rolling_vs_session.py --universe data/sample/universe_20261030.csv
uv run python examples/chapter03/03_state_and_box.py
uv run python examples/chapter03/04_plot_day.py 005930 2026-10-29
```

| 장 | 예제 | 내용 |
|---|---|---|
| 04 | `examples/chapter04/01_signals.py` | S0·A 매수 후보를 조건별로 세기 |
| 04 | `examples/chapter04/02_exits.py` | ATR 손절 폭, 청산 사유 분포, 매매 1건 그림 |
| 04 | `examples/chapter04/03_repaint.py` | 미확정 봉 착시, 다음 봉 시가 체결, 리페인팅 검사 (약 3분) |
| 04 | `examples/chapter04/04_filters.py` | 진입 허용·재진입 대기 조건별로 걸러지는 후보 |
| 04 | `examples/chapter04/05_run_strategy.py` | 설정 파일로 S0·A 실행 → 매매 기록 저장, 재현 확인 |

```bash
uv run python examples/chapter04/01_signals.py
uv run python examples/chapter04/05_run_strategy.py
```

| 장 | 예제 | 내용 |
|---|---|---|
| 05 | `examples/chapter05/01_event_engine.py` | 이벤트 엔진과 04장 결과 일치 확인, 이벤트 로그, 동시 보유 한도 |
| 05 | `examples/chapter05/02_costs.py` | 호가단위·수정주가 점검, 비용 분해, 장중 거래 공백 |
| 05 | `examples/chapter05/03_metrics.py` | 순수익 기준 성과 지표와 누적 손익 곡선 |
| 05 | `examples/chapter05/04_compare.py` | S0 대 A: 신뢰구간, 월별·종목별, 비용 민감도, 손익분기 승률 |
| 05 | `examples/chapter05/05_leakage.py` | 미래 정보·데이터 누수 검사 (약 2분) |

> 수수료율 0.015% 는 가정값입니다. `src/vwap_trading/backtest/costs.py` 의 `CostModel(fee=...)` 를 본인 계좌 수수료율로 바꾸세요.
