"""09장 실습 3 — 시장 국면과 종목 특성별 성과 (S0·A, 선정 종목-일 전 기간).

S0·A 는 파라미터를 학습하지 않으므로 전 기간(2025-11~2026-10)을 그대로 쓸 수 있다.
국면은 그날 장이 끝난 뒤에 아는 정보로 나누므로, 매매 규칙이 아니라 사후 분석용이다.
실행: uv run python examples/chapter09/03_regimes.py
"""

import sys

import numpy as np
import pandas as pd

from vwap_trading.backtest.costs import tick_size
from vwap_trading.backtest.engine import backtest
from vwap_trading.backtest.metrics import mean_ci
from vwap_trading.data.session import regular_session
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
uni = pd.read_csv("data/sample/universe_20261030.csv", dtype={"symbol": str}, encoding="utf-8-sig")
ind = load_indicators(sel.symbol)

# 시장 국면: 유니버스 종목들의 그날 시가→종가 수익률 중앙값(방향)과 (고가−저가)/시가 중앙값(변동성)
daily = []
for s in uni.symbol:
    try:
        b = regular_session(pd.read_parquet(f"data/raw/{s}_1m.parquet"))
    except FileNotFoundError:
        continue
    g = b.groupby(b.index.normalize())
    daily.append(pd.DataFrame({"ret": g.close.last() / g.open.first() - 1,
                               "rng": (g.high.max() - g.low.min()) / g.open.first()}))
daily = pd.concat(daily).groupby(level=0).median()
daily["방향"] = pd.qcut(daily.ret, 3, labels=["하락일", "보합일", "상승일"])
daily["변동성"] = pd.qcut(daily.rng, 3, labels=["저변동", "중변동", "고변동"])
print(f"시장 국면 기준: 유니버스 일별 중앙값, {len(daily)}거래일을 3등분")
print(f"  방향 경계 {daily.ret.quantile(1 / 3):+.2%} / {daily.ret.quantile(2 / 3):+.2%}, "
      f"변동성 경계 {daily.rng.quantile(1 / 3):.2%} / {daily.rng.quantile(2 / 3):.2%}")

trades = []
for name in ("S0", "A"):
    tr = backtest(ind, sel, StrategyParams(strategy=name)).assign(strategy=name)
    trades.append(tr)
tr = pd.concat(trades, ignore_index=True)
day = tr.entry_time.dt.normalize()
tr["방향"] = day.map(daily["방향"])
tr["변동성"] = day.map(daily["변동성"])
tr["시장"] = tr.symbol.map(uni.set_index("symbol").market)
tr["호가비율"] = tr.entry.map(tick_size) / tr.entry
atr_pct = [ind[s].atr_s[t] / ind[s].close[t] for s, t in zip(tr.symbol, tr.signal_time)]
tr["ATR비율"] = atr_pct
for c in ("호가비율", "ATR비율"):
    tr[c + "구간"] = pd.qcut(tr[c], 3, labels=["낮음", "중간", "높음"])
tr["시간대"] = pd.cut(tr.signal_time.dt.hour * 60 + tr.signal_time.dt.minute, [0, 660, 780, 1000],
                     labels=["~11시", "11~13시", "13시~"])


def show(col):
    rows = []
    for (s, k), g in tr.groupby(["strategy", col], observed=True, sort=False):
        lo, hi = mean_ci(g.net)
        rows.append({"전략": s, col: k, "매매": len(g), "비용 전": f"{g.ret.mean():+.3%}", "순 평균": f"{g.net.mean():+.3%}",
                     "95% 구간": f"[{lo:+.2%}, {hi:+.2%}]", "승률": f"{(g.net > 0).mean():.1%}"})
    print(f"\n[{col}]")
    print(pd.DataFrame(rows).sort_values(["전략", col]).to_string(index=False))


print(f"\n선정 {len(sel)}종목-일 (2025-11~2026-10): S0 {int((tr.strategy == 'S0').sum())}건, A {int((tr.strategy == 'A').sum())}건")
for col in ("방향", "변동성", "시장", "호가비율구간", "ATR비율구간", "시간대"):
    show(col)
best = tr[tr.strategy == "A"].groupby(["호가비율구간", "ATR비율구간"], observed=True).net.agg(["size", "mean"])
print("\n[A: 호가비율 × ATR비율] 순 평균")
print(best.assign(mean=best["mean"].map("{:+.3%}".format)).to_string())
