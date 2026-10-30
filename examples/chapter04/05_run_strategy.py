"""04장 실습 5 — 설정 파일로 S0·A 전략을 돌려 매매 기록을 저장하고, 재현되는지 확인한다.

실행: uv run python examples/chapter04/05_run_strategy.py
결과: reports/trades_S0_cmp2.csv 등 4개 (비용 차감 전. 비용은 05장에서 뺀다)
"""

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import load_params, run

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
Path("reports").mkdir(exist_ok=True)

rows = []
for cfg in ("configs/strategy_s0.toml", "configs/strategy_a.toml"):
    p = load_params(cfg)
    ind = load_indicators(sel.symbol, p.vwap)
    for cmp in (1, 2):
        q = replace(p, compare=cmp)
        tr = run(ind, sel, q)
        out = Path(f"reports/trades_{q.strategy}_cmp{cmp}.csv")
        tr.to_csv(out, index=False)
        again = run(ind, sel, q).to_csv(index=False)
        same = hashlib.sha256(again.encode()).hexdigest() == hashlib.sha256(tr.to_csv(index=False).encode()).hexdigest()
        rows.append({"전략": q.strategy, "비교": cmp, "매매": len(tr), "평균 수익률": f"{tr.ret.mean():+.3%}",
                     "양수 비율": f"{(tr.ret > 0).mean():.1%}", "평균 보유(봉)": round(tr.bars.mean(), 1),
                     "손절": (tr.reason == "stop").sum(), "목표": (tr.reason == "target").sum(),
                     "이탈": (tr.reason == "base").sum(), "시간": (tr.reason == "time").sum(),
                     "마감": (tr.reason == "close").sum(), "재실행 동일": same})

print(f"선정 종목-일 {len(sel)}개 · 비용 차감 전")
print(pd.DataFrame(rows).to_string(index=False))
print("저장: reports/trades_S0_cmp1.csv, trades_S0_cmp2.csv, trades_A_cmp1.csv, trades_A_cmp2.csv")
