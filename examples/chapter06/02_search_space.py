"""06장 실습 2 — 탐색 공간 점검: 파라미터를 하나씩만 바꿔 거래 수와 순수익이 어떻게 변하는지 본다.

첫 번째 학습 구간(2025-11~2026-02)만 쓴다. 검증·최종 테스트 구간은 보지 않는다.
실행: uv run python examples/chapter06/02_search_space.py
"""

import sys
from dataclasses import replace

import pandas as pd

from vwap_trading.optimize.search import SPACE, evaluate, params_from, selected_bars, walk_forward_windows
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
month = sel.tradable_from.dt.to_period("M").astype(str)
train, _ = walk_forward_windows(sorted(month.unique()))[0]
tr_sel = sel[month.isin(train)]
bars = selected_bars(tr_sel)
base = StrategyParams()
default = {"N": 30, "k": 5, "theta": 0.05, "m": 10, "delta": 0.3, "rho": 1.5}

grid = {"N": [10, 20, 30, 50, 70, 90], "k": [3, 5, 8, 11, 15], "theta": [0.02, 0.05, 0.08, 0.11, 0.15],
        "m": [5, 10, 15, 20, 30], "delta": [0.0, 0.3, 0.5, 0.75, 1.0], "rho": [1.0, 1.5, 2.0, 2.5, 3.0]}
assert grid.keys() == SPACE.keys()
print(f"학습 구간 {train[0]}~{train[-1]}, 나머지 파라미터는 초기값 {default}")
for name, values in grid.items():
    cells = []
    for v in values:
        p = params_from({**default, name: v}, base)
        tr = evaluate(bars, tr_sel, p)
        cells.append(f"{v}: {len(tr):3d}건 {tr.net.mean():+.2%}" if len(tr) else f"{v}: 0건")
    warm = "" if name not in ("N", "k", "m") else \
        f"  (워밍업 {params_from({**default, name: values[0]}).vwap.warmup_end}~{params_from({**default, name: values[-1]}).vwap.warmup_end}봉)"
    print(f"[{name}] " + " | ".join(cells) + warm)
