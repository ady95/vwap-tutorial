"""05장 실습 4 — S0 대 A 비교: 신뢰구간, 월별·종목별 분포, 비용 가정 민감도.

실행: uv run python examples/chapter05/04_compare.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vwap_trading.backtest.costs import CostModel
from vwap_trading.backtest.engine import backtest
from vwap_trading.backtest.metrics import mean_ci
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.reporting.plotting import COLORS, use_korean_font
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
ind = load_indicators(sel.symbol)


def diff_ci(a, b, n_boot=2000, seed=0):
    """평균 차이(a − b)의 부트스트랩 95% 신뢰구간."""
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return np.quantile(d, [0.025, 0.975])


bts = {(s, c): backtest(ind, sel, StrategyParams(strategy=s, compare=c)) for s in ("S0", "A") for c in (1, 2)}

# 1. 거래당 평균 (비용 전·순) 과 95% 신뢰구간
print("[거래당 평균 수익률과 95% 신뢰구간]")
for c in (1, 2):
    for col, label in (("ret", "비용 전"), ("net", "순")):
        a, s0 = bts[("A", c)][col].to_numpy(), bts[("S0", c)][col].to_numpy()
        lo_a, hi_a = mean_ci(a)
        lo_s, hi_s = mean_ci(s0)
        lo_d, hi_d = diff_ci(a, s0)
        print(f"  비교 {c} {label:<4}: A {a.mean():+.3%} [{lo_a:+.3%}, {hi_a:+.3%}]  "
              f"S0 {s0.mean():+.3%} [{lo_s:+.3%}, {hi_s:+.3%}]  A−S0 [{lo_d:+.3%}, {hi_d:+.3%}]")

# 2. 월별 (비교 2, 순수익 합계 = 그달 누적 손익)
print("\n[월별 순손익 합계 (비교 2, 거래당 투입 금액 1 기준)]")
month = pd.DataFrame({s: bts[(s, 2)].groupby(bts[(s, 2)].exit_time.dt.to_period("M")).net.sum() for s in ("S0", "A")})
month["A 매매"] = bts[("A", 2)].groupby(bts[("A", 2)].exit_time.dt.to_period("M")).size()
print(month.round(3).to_string())
print(f"A 가 S0 보다 손실이 작은 달: {(month.A > month.S0).sum()}/{len(month)}")

# 3. 비용 전 기준으로 이익을 낸 달·종목 (신호 자체에 우위가 있는 구간이 있는가)
a2 = bts[("A", 2)]
g_month = a2.groupby(a2.exit_time.dt.to_period("M")).ret.mean()
g_sym = a2.groupby("symbol").agg(n=("ret", "size"), gross=("ret", "mean"), net=("net", "mean"))
print(f"\n[A 비교 2] 비용 전 평균이 양수인 달 {(g_month > 0).sum()}/{len(g_month)}, "
      f"종목 {(g_sym.gross > 0).sum()}/{len(g_sym)} (순 평균이 양수인 종목 {(g_sym.net > 0).sum()})")
print("매매 10건 이상 종목:")
print(g_sym[g_sym.n >= 10].sort_values("gross", ascending=False).round(4).to_string())

# 4. 슬리피지 가정 민감도 (비교 2, 순 평균)
print("\n[슬리피지 가정별 순 평균]")
for k in (0, 0.5, 1, 2):
    vals = [backtest(ind, sel, StrategyParams(strategy=s), CostModel(slip_ticks=k)).net.mean() for s in ("S0", "A")]
    print(f"  {k}호가: S0 {vals[0]:+.3%}  A {vals[1]:+.3%}")

# 5. 청산 사유별 손익과 손익분기 승률 (A 비교 2)
by = a2.groupby("reason").agg(n=("net", "size"), gross=("ret", "mean"), net=("net", "mean"),
                              net_win=("net", lambda r: (r > 0).mean()))
print("\n[A 비교 2 청산 사유별]")
print(by.round(4).to_string())
cost = (a2.ret - a2.net).mean()
width = (a2.target / a2.entry - 1).median()
win, loss = a2.net[a2.net > 0].mean(), a2.net[a2.net <= 0].mean()
print(f"매매당 비용 {cost:.3%}, 목표 폭 중앙값 {width:.3%}, 평균 이익 {win:+.3%}, 평균 손실 {loss:+.3%}")
print(f"손익분기 승률 {-loss / (win - loss):.1%}, 현재 승률 {(a2.net > 0).mean():.1%}")

# 그림: 월별 순손익
use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(month))
ax.bar(x - 0.2, month.S0, width=0.4, color=COLORS["svwap"], label="S0")
ax.bar(x + 0.2, month.A, width=0.4, color=COLORS["mvwap"], label="A")
ax.set_xticks(x, [str(m) for m in month.index], rotation=45, fontsize=8)
ax.axhline(0, color="#888888", lw=0.8)
ax.set_title("월별 순손익 합계 (비교 2, 거래당 투입 금액 = 1)", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/05-4-monthly.png", dpi=110)
print("그림: reports/05-4-monthly.png")
