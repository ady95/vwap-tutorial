"""07장 실습 2 — 피처별로 후보를 5분위로 나눠 성공 비율과 순수익을 본다.

탐색은 첫 검증 달(2026-03) 이전 데이터(2025-10~2026-02)로만 한다.
실행: uv run python examples/chapter07/02_features.py   (01 실행 후)
"""

import sys

import pandas as pd

from vwap_trading.ml.dataset import FEATURES

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
eda = df[df.date < "2026-03-01"]
print(f"탐색용 {eda.date.min():%Y-%m} ~ {eda.date.max():%Y-%m}: 후보 {len(eda):,}건, y=1 {eda.y.mean():.1%}, 순 평균 {eda.net.mean():+.3%}")
print(eda[FEATURES].describe(percentiles=[0.5]).T[["mean", "50%", "min", "max"]].round(4).to_string())

rows = []
for f in FEATURES:
    q = pd.qcut(eda[f].rank(method="first"), 5, labels=False)
    g = eda.groupby(q).agg(y=("y", "mean"), net=("net", "mean"))
    rows.append({"피처": f, **{f"y Q{i + 1}": f"{v:.0%}" for i, v in enumerate(g.y)},
                 **{f"순 Q{i + 1}": f"{v:+.2%}" for i, v in enumerate(g.net)},
                 "순 Q5−Q1": round(g.net.iloc[-1] - g.net.iloc[0], 4)})
out = pd.DataFrame(rows).set_index("피처")
out = out.reindex(out["순 Q5−Q1"].abs().sort_values(ascending=False).index)
print("\n피처 5분위별 성공 비율(y)과 순수익 평균 (Q1 = 가장 작은 20%)")
print(out.to_string())
