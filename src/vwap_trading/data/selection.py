"""예제 종목 선정 입력 만들기 (02장).

이 책은 종목 선정을 외부 도구에 맡기고 (symbol, selected_at, tradable_from) 만 입력으로 받는다.
실습용 대체 규칙: 매일 개장 후 30분 동안의 거래대금이 같은 시간대 최근 20거래일 평균의
몇 배인지(상대 거래대금)를 계산해 상위 종목을 고른다. 개장 30분 시점까지의 정보만 쓰므로
선정 시각 이전의 미래 정보는 들어가지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from vwap_trading.data.session import regular_session, session_of


def early_value(bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    """날짜별 개장 후 minutes 분 동안의 거래대금 (종가 × 거래량 합)."""
    bars = regular_session(bars)
    out = {}
    for day, g in bars.groupby(bars.index.date):
        start = datetime.combine(day, session_of(day)[0])
        win = g[g.index < start + timedelta(minutes=minutes)]
        out[day] = float((win.close * win.volume).sum())
    return pd.Series(out)


def make_selections(bars_by_symbol: dict[str, pd.DataFrame], top_n: int = 3, minutes: int = 30,
                    lookback: int = 20, min_ratio: float = 1.5) -> pd.DataFrame:
    values = pd.DataFrame({s: early_value(b, minutes) for s, b in bars_by_symbol.items()}).sort_index()
    base = values.rolling(lookback, min_periods=lookback).mean().shift(1)   # 오늘은 빼고 과거 20일만
    ratio = values / base
    rows = []
    for day, r in ratio.dropna(how="all").iterrows():
        picked = r[r >= min_ratio].sort_values(ascending=False).head(top_n)
        selected_at = datetime.combine(day, session_of(day)[0]) + timedelta(minutes=minutes)
        for symbol, v in picked.items():
            rows.append({"symbol": symbol, "selected_at": selected_at, "tradable_from": selected_at,
                         "ratio": round(float(v), 2)})
    return pd.DataFrame(rows)
