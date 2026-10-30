"""03장 실습 1 — 세션 VWAP 을 계산하고 손으로 검산한다.

실행: uv run python examples/chapter03/01_session_vwap.py 005930 2026-10-29
"""

import sys

import pandas as pd

from vwap_trading.data.session import regular_session
from vwap_trading.indicators.vwap import session_vwap, typical_price

sys.stdout.reconfigure(encoding="utf-8")
symbol, date = sys.argv[1], sys.argv[2]

bars = regular_session(pd.read_parquet(f"data/raw/{symbol}_1m.parquet")).loc[date]
day = bars.assign(tp=typical_price(bars), svwap=session_vwap(bars))

print(day[["high", "low", "close", "volume", "tp", "svwap"]].head(3).round(2).to_string())
print("...")
print(day[["close", "volume", "svwap"]].tail(3).round(2).to_string())

# 손 계산: 처음 세 봉의 거래대금 합 / 거래량 합
first3 = day.head(3)
by_hand = (first3.tp * first3.volume).sum() / first3.volume.sum()
print(f"\n세 번째 봉 세션 VWAP: 함수 {day.svwap.iloc[2]:.2f} / 손 계산 {by_hand:.2f}")

above = (day.close > day.svwap).mean()
print(f"종가가 세션 VWAP 위에 있던 봉 비율: {above:.1%} ({len(day)}봉)")
