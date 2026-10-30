"""06장 실습 3 — 목적함수 비교: 순수익 합계·거래당 평균·보수적 평균(평균 − 표준오차).

첫 번째 창(학습 2025-11~2026-02, 검증 2026-03)에서 목적함수만 바꿔 100회씩 탐색한다.
실행: uv run python examples/chapter06/03_objectives.py   (약 5분)
"""

import sys

import optuna
import pandas as pd

from vwap_trading.backtest.metrics import summarize
from vwap_trading.optimize.search import evaluate, optimize, params_from, selected_bars, walk_forward_windows
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
month = sel.tradable_from.dt.to_period("M").astype(str)
train, test = walk_forward_windows(sorted(month.unique()))[0]
tr_sel, te_sel = sel[month.isin(train)], sel[month.isin(test)]
tr_bars, te_bars = selected_bars(tr_sel), selected_bars(te_sel)

rows = []
for kind, min_trades in (("total", 1), ("mean", 1), ("mean", 30), ("lcb", 30)):
    if kind == "total" or min_trades == 1:
        label = {"total": "순수익 합계", "mean": "거래당 평균"}[kind] + " (최소 거래 없음)"
    else:
        label = {"mean": "거래당 평균", "lcb": "평균 − 표준오차"}[kind] + f" (최소 {min_trades}건)"
    study = optimize(tr_bars, tr_sel, n_trials=100, kind=kind, min_trades=min_trades)
    p = params_from(study.best_params)
    a, b = summarize(evaluate(tr_bars, tr_sel, p)), summarize(evaluate(te_bars, te_sel, p))
    rows.append({"목적함수": label, "학습 거래": a["trades"], "학습 평균": f"{a.get('mean', 0):+.3%}",
                 "학습 합계": f"{a.get('total', 0):+.2f}", "학습 MDD": f"{a.get('mdd', 0):.2f}",
                 "검증 거래": b["trades"], "검증 평균": f"{b['mean']:+.3%}" if b["trades"] else "-",
                 "파라미터": {k: round(v, 3) for k, v in study.best_params.items()}})
a = summarize(evaluate(tr_bars, tr_sel, StrategyParams()))
b = summarize(evaluate(te_bars, te_sel, StrategyParams()))
rows.append({"목적함수": "초기값 (탐색 없음)", "학습 거래": a["trades"], "학습 평균": f"{a['mean']:+.3%}",
             "학습 합계": f"{a['total']:+.2f}", "학습 MDD": f"{a['mdd']:.2f}", "검증 거래": b["trades"],
             "검증 평균": f"{b['mean']:+.3%}", "파라미터": "N30 k5 θ0.05 m10 δ0.3 ρ1.5"})
print(f"학습 {train[0]}~{train[-1]} → 검증 {test[0]}")
for r in rows:
    print(r)
