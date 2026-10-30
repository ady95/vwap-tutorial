"""장중 종목 선정 (11장). 02장 실습용 선정 규칙(make_selections)을 실시간으로 적용한다.

개장 후 30분 거래대금 ÷ 같은 시간대 최근 20거래일 평균 ≥ 1.5 인 종목 중 상위 3개.
과거 20거래일 평균은 장 시작 전에 저장된 분봉(data/raw)으로 미리 계산하고, 오늘 값은
실시간으로 만든 봉으로 계산한다.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from vwap_trading.data.selection import early_value


def history_base(raw_by_symbol: dict[str, pd.DataFrame], day: date, minutes: int = 30,
                 lookback: int = 20) -> pd.Series:
    """종목별 '개장 후 minutes 분 거래대금'의 day 이전 lookback 거래일 평균. 거래일이 모자라면 빠진다."""
    out = {}
    for s, b in raw_by_symbol.items():
        v = early_value(b[b.index.date < day], minutes)
        if len(v) >= lookback:
            out[s] = v.tail(lookback).mean()
    return pd.Series(out, dtype=float)


def select_live(today_bars: dict[str, pd.DataFrame], base: pd.Series, at: datetime, top_n: int = 3,
                min_ratio: float = 1.5) -> pd.DataFrame:
    """at 이전에 확정된 오늘 봉으로 선정한다. 반환: symbol, selected_at, tradable_from, ratio."""
    value = pd.Series({s: float((b.close * b.volume)[b.index < at].sum()) for s, b in today_bars.items()})
    ratio = (value / base).dropna()
    picked = ratio[ratio >= min_ratio].sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"symbol": picked.index, "selected_at": at, "tradable_from": at,
                         "ratio": picked.round(2).to_numpy()})
