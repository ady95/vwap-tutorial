"""시간순 분할과 중첩 레이블 누수 방지 (07장).

- 검증 달마다 그 이전 데이터 전체로 학습한다(확장 창). 06장 Walk-forward 와 같은 검증 달을 쓴다.
- 퍼지(purge): 레이블 관측 기간(진입~청산)이 검증 구간 시작과 겹치는 학습 표본을 뺀다.
- 엠바고(embargo): 검증 구간 바로 앞 몇 거래일을 학습에서 뺀다. 이어진 날의 시장 분위기가
  학습과 검증에 함께 들어가는 것을 줄인다.
- 최종 테스트 구간(2026-09 이후)은 학습에 절대 들어가지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vwap_trading.optimize.search import FINAL_TEST_START, check_before_final


def fold(df: pd.DataFrame, test_month: str, embargo_days: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """검증 달 하나 → (학습 행 위치, 검증 행 위치)."""
    month = df.date.dt.to_period("M").astype(str)
    test = np.flatnonzero(month == test_month)
    start = pd.Period(test_month).start_time
    days = np.sort(df.date[df.date < start].unique())
    if embargo_days:
        days = days[:-embargo_days]
    train_mask = df.date.isin(days) & (df.exit_time < start)          # 엠바고 + 퍼지
    train = np.flatnonzero(train_mask.to_numpy())
    if len(train):
        check_before_final(df.date.iloc[train])
    return train, test


def walk_forward_folds(df: pd.DataFrame, test_months: list[str], embargo_days: int = 1):
    for m in test_months:
        if pd.Period(m).start_time.date() >= FINAL_TEST_START:
            raise ValueError(f"{m} 은 최종 테스트 구간이다. 09장에서 한 번만 평가한다.")
        yield m, *fold(df, m, embargo_days)
