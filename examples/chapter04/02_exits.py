"""04장 실습 2 — ATR 기반 손절의 근거와 청산 사유 분포, 매매 1건 그림.

실행: uv run python examples/chapter04/02_exits.py
"""

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vwap_trading.indicators.vwap import FLAT, load_indicators
from vwap_trading.reporting.plotting import COLORS, hhmm_axis, shade, use_korean_font
from vwap_trading.strategy.rules import StrategyParams, run

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
p = StrategyParams()
ind = load_indicators(sel.symbol, p.vwap)

# 1. 같은 1.5 ATR 이 종목마다 몇 % 인가
pct = pd.Series({s: (p.lam * x.atr_s / x.close).between_time("10:00", "14:45").median() for s, x in ind.items()})
print(f"1.5 ATR 손절 폭 (가격 대비, 종목별 중앙값): 최소 {pct.min():.2%}, 중앙값 {pct.median():.2%}, 최대 {pct.max():.2%}")

# 2. 청산 사유 분포
for cmp in (1, 2):
    tr = run(ind, sel, replace(p, compare=cmp))
    share = tr.reason.value_counts(normalize=True).round(3).to_dict()
    print(f"[A 비교 {cmp}] {len(tr)}건, 평균 보유 {tr.bars.mean():.1f}봉, 청산 사유 {share}")

# 3. 목표가에 도달한 매매 1건을 그린다
tr = run(ind, sel, p)
t = tr[tr.reason == "target"].iloc[-1]
day = ind[t.symbol].loc[str(t.entry_time.date())]
w = day.loc[t.signal_time - pd.Timedelta(minutes=60): t.exit_time + pd.Timedelta(minutes=20)]
use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(w.index, w.close, color=COLORS["close"], lw=1, label="종가")
ax.plot(w.index, w.mvwap, color=COLORS["mvwap"], lw=2, label="변형 VWAP")
shade(ax, w.index, (w.state == FLAT).to_numpy())
hold = w.loc[t.entry_time: t.exit_time].index
ax.plot(hold, [t.stop] * len(hold), color="#1f77b4", ls="--", lw=1.2, label=f"손절가 {t.stop:,.0f}")
ax.plot(hold, [t.target] * len(hold), color="#2ca02c", ls="--", lw=1.2, label=f"목표가 {t.target:,.0f}")
ax.scatter([t.signal_time], [day.close[t.signal_time]], s=60, color="#d62728", zorder=5, label="신호 봉 (확정)")
ax.scatter([t.entry_time], [t.entry], s=60, marker="^", color="black", zorder=5, label=f"진입 {t.entry:,.0f}")
ax.scatter([t.exit_time], [t.exit], s=60, marker="v", color="#2ca02c", zorder=5, label=f"청산 {t.exit:,.0f}")
ax.set_title(f"{t.symbol} {t.entry_time.date()}: 전략 A 매매 1건 (비용 전 {t.ret:+.2%})", fontsize=11)
ax.legend(loc="upper left", fontsize=7)
hhmm_axis(ax)
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/04-2-trade-example.png", dpi=110)
print(f"그림: {t.symbol} 신호 {t.signal_time:%Y-%m-%d %H:%M}, 진입 {t.entry:,.0f}, 손절 {t.stop:,.0f}, "
      f"목표 {t.target:,.0f}, 청산 {t.exit_time:%H:%M} {t.exit:,.0f} ({t.ret:+.2%}, {t.bars}봉)")
