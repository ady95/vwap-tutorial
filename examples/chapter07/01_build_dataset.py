"""07장 실습 1 — 유니버스 전체의 전략 A 매수 후보로 학습 데이터셋을 만든다.

실행: uv run python examples/chapter07/01_build_dataset.py   (약 40초)
결과: data/ml/candidates_A.parquet
"""

import sys
from pathlib import Path

import pandas as pd

from vwap_trading.indicators.vwap import load_indicators
from vwap_trading.ml.dataset import build
from vwap_trading.strategy.rules import StrategyParams, run

sys.stdout.reconfigure(encoding="utf-8")
uni = pd.read_csv("data/sample/universe_20261030.csv", dtype={"symbol": str}, encoding="utf-8-sig").symbol
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])

df = build(uni)
Path("data/ml").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/ml/candidates_A.parquet")
print(f"유니버스 {len(uni)}종목 중 후보가 있는 {df.symbol.nunique()}종목, {df.date.nunique()}거래일, 후보 {len(df):,}건")
print(f"기간 {df.date.min():%Y-%m-%d} ~ {df.date.max():%Y-%m-%d}, 저장: data/ml/candidates_A.parquet")

# 레이블 통계는 최종 테스트 구간(2026-09~)을 빼고 본다. 그 구간은 건수만 확인한다
pre = df[df.date < "2026-09-01"]
print(f"최종 테스트 구간 후보 {len(df) - len(pre):,}건은 결과를 보지 않음 (09장)")
print(f"\n[레이블] 2025-10 ~ 2026-08, 후보 {len(pre):,}건")
print(f"  y=1 (성공) 비율 {pre.y.mean():.1%}")
by = pre.groupby("reason").agg(n=("y", "size"), y=("y", "mean"), ret=("ret", "mean"), net=("net", "mean"))
print(by.round(4).to_string())
print(f"  순수익 평균 {pre.net.mean():+.3%}, 비용 전 {pre.ret.mean():+.3%}")

# 학습 대상(유니버스 전체)과 매매 대상(선정 종목-일)
key = sel.assign(date=sel.tradable_from.dt.normalize())
s = pre.merge(key, on=["symbol", "date"])
s = s[s.time >= s.tradable_from]
print(f"\n[선정 종목-일] 2025-11 ~ 2026-08 후보 {len(s)}건 ({len(s) / len(pre):.1%}), "
      f"y=1 비율 {s.y.mean():.1%}, 순수익 평균 {s.net.mean():+.3%}")

# 레이블이 04장 매매 시뮬레이션과 같은가 (비교 1: 기준선 이탈 청산 없음, 04장은 전 기간)
tr = run(load_indicators(sel.symbol), sel, StrategyParams(compare=1))
m = tr.merge(df, left_on=["symbol", "signal_time"], right_on=["symbol", "time"], suffixes=("", "_lab"))
same = ((m.reason == m.reason_lab) & (m.exit_time == m.exit_time_lab) & ((m.ret - m.ret_lab).abs() < 1e-12)).sum()
print(f"[대조] 04장 비교 1 매매 {len(tr)}건 중 레이블과 청산 사유·시각·수익률이 모두 같은 매매 {same}건")
