"""04장 실습 4 — 진입 가능 시간·재진입 대기 조건이 매수 후보를 얼마나 걸러 내는지 본다.

실행: uv run python examples/chapter04/04_filters.py
"""

import sys
from datetime import date

import pandas as pd

from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import StrategyParams, day_clock, raw_signal, run

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
base = StrategyParams()
ind = load_indicators(sel.symbol, base.vwap)

for name in ("S0", "A"):
    p = StrategyParams(strategy=name)
    steps = dict.fromkeys(["지표 조건만", "+ 워밍업 (44봉 이후)", "+ tradable_from 이후",
                           "+ 진입 마감 (14:45) 이전", "+ ATR 하한 이상"], 0)
    for r in sel.itertuples():
        x = ind[r.symbol].loc[str(r.tradable_from.date())]
        cutoff, _ = day_clock(x.index[0].date(), p)
        masks = [raw_signal(x, p), x.bar_no > p.vwap.warmup_end, x.index >= r.tradable_from,
                 x.index.time <= cutoff, ~x.atr_low]
        m = masks[0]
        for key, cond in zip(steps, masks):
            m = m & cond
            steps[key] += int(m.sum())
    trades = run(ind, sel, p)
    print(f"[{name}]")
    for key, v in steps.items():
        print(f"  {key:<24}{v:6,}")
    print(f"  {'+ 무포지션·재진입 대기 = 매매':<24}{len(trades):6,}")
    hist = trades.signal_time.dt.hour.value_counts().sort_index()
    print("  매매 시간대(시):", hist.to_dict())

print("\n예외일 시간표")
for d in (date(2026, 10, 29), date(2025, 11, 13), date(2026, 1, 2)):
    cutoff, force = day_clock(d, base)
    print(f"  {d}: 신규 진입 마감 {cutoff:%H:%M}, 장 마감 청산 판정 봉 {force:%H:%M}")
