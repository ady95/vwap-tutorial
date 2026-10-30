"""08장 실습 4 — 전략 C: 전략 A 의 매수 후보를 LightGBM 이 승인할 때만 주문한다.

선정 종목-일, 검증 달 2026-03~08 (06장 B 와 같은 기간). 02 실행 후.
실행: uv run python examples/chapter08/04_integrate.py
결과: reports/c_trades.parquet
"""

import sys

import pandas as pd

from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
sel = sel[(sel.tradable_from >= "2026-03-01") & (sel.tradable_from < "2026-09-01")]
ind = load_indicators(sel.symbol)
p = StrategyParams()


def approver(pred: pd.DataFrame, col: str):
    """예측 표(종목, 신호 시각, 승인 여부) → backtest 에 넘길 승인 함수. 예측이 없는 후보는 거부."""
    ok = set(zip(pred.symbol[pred[col]], pred.time[pred[col]]))

    def approve(symbol, t):
        return (symbol, t) in ok
    return approve


variants = {"A (필터 없음)": None}
for label, name in (("y", "(1)"), ("y_net", "(2)")):
    pred = pd.read_parquet(f"reports/ml_pred_{label}.parquet")
    variants[f"C{name} 기대 순수익 규칙"] = approver(pred, "approve")
    variants[f"C{name} 상위 30% (비교용)"] = approver(pred, "top30")

out = []
for name, ap in variants.items():
    t = backtest(ind, sel, p, approve=ap).assign(variant=name)
    out.append(t)
    print(f"{name:<22} 매매 {len(t):3d}건" + (f", 순 평균 {t.net.mean():+.3%}, 누적 {t.net.sum():+.2f}" if len(t) else ""))
pd.concat(out).to_parquet("reports/c_trades.parquet")
print(f"선정 종목-일 {len(sel)}개 (2026-03~08), 저장: reports/c_trades.parquet")
