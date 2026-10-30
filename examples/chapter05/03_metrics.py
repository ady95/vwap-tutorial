"""05장 실습 3 — 비용 차감 순수익 기준 성과 지표와 누적 손익 곡선.

실행: uv run python examples/chapter05/03_metrics.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vwap_trading.backtest.costs import CostModel
from vwap_trading.backtest.engine import backtest
from vwap_trading.backtest.metrics import equity_curve, summarize
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.reporting.plotting import COLORS, use_korean_font
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
ind = load_indicators(sel.symbol)

rows, curves = [], {}
for strategy in ("S0", "A"):
    for compare in (1, 2):
        bt = backtest(ind, sel, StrategyParams(strategy=strategy, compare=compare))
        Path("reports").mkdir(exist_ok=True)
        bt.to_csv(f"reports/backtest_{strategy}_cmp{compare}.csv", index=False)
        m = summarize(bt, days=len(sel))
        g = summarize(bt, col="ret")
        rows.append({"전략": f"{strategy} 비교{compare}", "매매": m["trades"], "하루 매매": round(m["trades_per_day"], 2),
                     "비용 전 평균": f"{g['mean']:+.3%}", "순 평균": f"{m['mean']:+.3%}", "승률": f"{m['win_rate']:.1%}",
                     "손익비": round(m["payoff"], 2), "PF": round(m["profit_factor"], 2),
                     "누적": f"{m['total']:+.2f}", "MDD": f"{m['mdd']:.2f}"})
        if compare == 2:
            curves[strategy] = (equity_curve(bt), equity_curve(bt, "ret"))

print(f"선정 종목-일 {len(sel)}개 · 거래당 투입 금액 1 기준 (누적·MDD 는 투입 금액의 배수)")
print(pd.DataFrame(rows).to_string(index=False))

use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4.5))
for strategy, color in (("S0", COLORS["svwap"]), ("A", COLORS["mvwap"])):
    net, gross = curves[strategy]
    ax.plot(net.index, net.to_numpy(), color=color, lw=2, label=f"{strategy} 순손익 (비용 차감)")
    ax.plot(gross.index, gross.to_numpy(), color=color, lw=1, ls="--", label=f"{strategy} 비용 전")
ax.axhline(0, color="#888888", lw=0.8)
ax.set_title("누적 손익 (비교 2, 거래당 투입 금액 = 1)", fontsize=11)
ax.set_ylabel("투입 금액 배수")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("reports/05-3-equity-curve.png", dpi=110)
print("그림: reports/05-3-equity-curve.png")
