"""05장 실습 2 — 체결·비용 모델: 호가단위 확인, 비용 분해, 같은 봉 손절·목표가 동시 도달.

실행: uv run python examples/chapter05/02_costs.py
"""

import sys

import numpy as np
import pandas as pd

from vwap_trading.backtest.costs import CostModel, tick_size
from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
ind = load_indicators(sel.symbol)

# 1. 분봉 가격이 호가단위에 맞는가
ticks = np.vectorize(tick_size)
print("[호가단위에 맞지 않는 종가 비율]")
for s in ("005930", "000660", "196170", "207940", "328130"):
    b = pd.read_parquet(f"data/raw/{s}_1m.parquet")
    px = b.close.to_numpy()
    t = ticks(px)
    off = px % t != 0
    half = off & (px % (t / 2) == 0)
    month = pd.Series(off, index=b.index).groupby(b.index.to_period("M")).mean()
    print(f"  {s}: 전체 {off.mean():.1%} (그중 반 호가 {half.sum() / max(off.sum(), 1):.0%}), "
          f"월별 최대 {month.max():.0%} 최소 {month.min():.0%}")

# 2. 호가 한 칸이 가격의 몇 % 인가 (선정 종목-일의 진입가 기준)
p = StrategyParams()
bt = backtest(ind, sel, p, CostModel(fee=0, slip_ticks=0, tax=False))
tick_pct = bt.entry.map(tick_size) / bt.entry
print(f"\n[호가 한 칸 / 가격] A 매매 {len(bt)}건 진입가 기준: 중앙값 {tick_pct.median():.3%}, "
      f"최소 {tick_pct.min():.3%}, 최대 {tick_pct.max():.3%}")

# 3. 비용 분해: 비용을 하나씩 더한다
steps = [("비용 전", CostModel(fee=0, slip_ticks=0, tax=False)),
         ("+ 수수료 0.015% × 2", CostModel(slip_ticks=0, tax=False)),
         ("+ 증권거래세", CostModel(slip_ticks=0)),
         ("+ 슬리피지 0.5호가 × 2", CostModel(slip_ticks=0.5)),
         ("+ 슬리피지 1호가 × 2 (기준)", CostModel(slip_ticks=1))]
for name in ("S0", "A"):
    q = StrategyParams(strategy=name)
    print(f"\n[{name} 비교 2] 거래당 평균 수익률")
    prev = None
    for label, cost in steps:
        r = backtest(ind, sel, q, cost).net
        gap = "" if prev is None else f"  ({r.mean() - prev:+.3%}p)"
        print(f"  {label:<24}{r.mean():+.3%}  양수 {(r > 0).mean():.1%}{gap}")
        prev = r.mean()

# 4. 같은 봉에서 손절가·목표가에 모두 닿은 매매
both = 0
for tr in bt[bt.reason == "stop"].itertuples():
    bar = ind[tr.symbol].loc[tr.exit_time]
    both += bar.high >= tr.target
print(f"\n[같은 봉 동시 도달] A 손절 {int((bt.reason == 'stop').sum())}건 중 {both}건은 같은 봉에서 목표가에도 닿음 → 손절로 처리")

# 5. 장중 거래 공백: 시장 전체가 20분 이상 체결이 없던 구간 (삼성전자 기준)
b = pd.read_parquet("data/raw/005930_1m.parquet")
b = b[b.index.strftime("%H:%M") < "15:20"]
gap = b.index.to_series().groupby(b.index.date).diff().dt.total_seconds().div(60)
gap = gap[gap >= 20]
print(f"\n[장중 공백] {len(gap)}일: " + ", ".join(f"{t:%Y-%m-%d %H:%M} 재개({v:.0f}분)" for t, v in gap.items()))
for name in ("S0", "A"):
    t = backtest(ind, sel, StrategyParams(strategy=name))
    held = sum(((t.entry_time < ts) & (t.exit_time >= ts)).sum() for ts in gap.index)
    after = sum(((t.signal_time >= ts) & (t.signal_time < ts + pd.Timedelta(minutes=10))).sum() for ts in gap.index)
    print(f"  {name}: 공백을 넘겨 보유한 매매 {held}건, 재개 후 10분 안 신호 {after}건")
