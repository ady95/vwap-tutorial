"""05장 실습 1 — 이벤트 기반 엔진이 04장 규칙과 같은 매매를 내는지 확인하고, 이벤트 순서를 본다.

실행: uv run python examples/chapter05/01_event_engine.py
"""

import sys
import time

import pandas as pd

from vwap_trading.backtest.engine import SymbolDay, backtest
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import StrategyParams, run

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
ind = load_indicators(sel.symbol)

# 1. 04장 run() 과 이벤트 엔진의 매매 기록 비교
cols = ["symbol", "signal_time", "entry_time", "exit_time", "reason", "bars", "ret"]
for strategy in ("S0", "A"):
    for compare in (1, 2):
        p = StrategyParams(strategy=strategy, compare=compare)
        t0 = time.perf_counter()
        bt = backtest(ind, sel, p)
        sec = time.perf_counter() - t0
        a = bt[cols].sort_values(["symbol", "signal_time"]).reset_index(drop=True)
        b = run(ind, sel, p)[cols].sort_values(["symbol", "signal_time"]).reset_index(drop=True)
        print(f"[{strategy} 비교 {compare}] 엔진 {len(a)}건, 04장 {len(b)}건, 완전 일치 {a.equals(b)}, {sec:.1f}초")

# 2. 매매 1건의 이벤트 순서 (전략 A, 2026-10-29 006400)
p = StrategyParams()
x = ind["006400"].loc["2026-10-29"]
sd = SymbolDay("006400", x, p, pd.Timestamp("2026-10-29 09:30"))
print("\n006400 2026-10-29 이벤트 로그 (전략 A)")
for i in range(len(x)):
    before = (sd.pending_buy, sd.pending_sell, sd.pos is not None)
    done = sd.on_bar(i)
    t = x.index[i].strftime("%H:%M")
    if sd.pending_buy is not None and before[0] is None:
        print(f"  {t} 봉 확정: 매수 후보 → 다음 봉 시가 매수 주문")
    if sd.pos is not None and before[0] is not None:
        print(f"  {t} 봉 시가 {x.open.iloc[i]:,.0f} 매수 체결 (손절 {sd.pos.stop:,.0f}, 목표 {sd.pos.target:,.0f})")
    if sd.pending_sell is not None and before[1] is None:
        print(f"  {t} 봉 확정: 청산 판정({sd.pending_sell[0]}) → 다음 봉 시가 매도 주문")
    if done is not None:
        print(f"  {t} 봉 매도 체결 {done['exit']:,.0f} ({done['reason']}, 비용 전 {done['ret']:+.2%})")
        if x.index[i] > pd.Timestamp("2026-10-29 11:30"):
            break

# 3. 동시 보유 한도의 효과 (전략 A 비교 2)
print()
for k in (None, 3, 1):
    bt = backtest(ind, sel, p, max_positions=k)
    print(f"동시 보유 한도 {k or '없음'}: {len(bt)}건")
