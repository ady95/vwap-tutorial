"""04장 실습 1 — S0·A 매수 후보를 조건별로 세어 본다 (시간·포지션 조건 적용 전).

실행: uv run python examples/chapter04/01_signals.py
"""

import sys

import pandas as pd

from vwap_trading.indicators.vwap import FLAT, load_indicators
from vwap_trading.strategy.rules import StrategyParams, raw_signal

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
p = StrategyParams()
ind = load_indicators(sel.symbol, p.vwap)

# 선정된 종목-일의 봉만 모은다
days = [ind[r.symbol].loc[str(r.tradable_from.date())].assign(symbol=r.symbol) for r in sel.itertuples()]
x = pd.concat(days)
print(f"선정 종목-일 {len(sel)}개, 1분봉 {len(x):,}개")

# 전략 A: 조건을 하나씩 더한다
c1 = (x.state.shift() == FLAT) & (x.slope >= p.vwap.theta) & (x.symbol == x.symbol.shift())
c2 = c1 & (x.close > x.mvwap.shift() + p.delta * x.atr_s)
c3 = c2 & (x.rvol >= p.rho)
print("\n[A] 조건별 봉 수")
print(f"  (1) 횡보 → 상승 전환           {c1.sum():6,}")
print(f"  (2) + 박스 위 0.3 ATR 돌파     {c2.sum():6,}")
print(f"  (3) + 상대 거래량 1.5배 이상   {c3.sum():6,}")

s0 = pd.concat(raw_signal(d, StrategyParams(strategy="S0")) for d in days)
a = pd.concat(raw_signal(d, p) for d in days)
assert a.sum() == c3.sum()
print(f"\n[S0] 세션 VWAP 상향 돌파 + 상대 거래량: {s0.sum():,}")
print(f"[A]  변형 VWAP 매수 후보:              {a.sum():,}")
print(f"종목-일당 S0 {s0.sum() / len(sel):.2f}개, A {a.sum() / len(sel):.2f}개")

hour = pd.DataFrame({"S0": s0[s0].index.hour.value_counts(), "A": a[a].index.hour.value_counts()})
print("\n시간대별 후보 수 (시)")
print(hour.fillna(0).astype(int).sort_index().T.to_string())
