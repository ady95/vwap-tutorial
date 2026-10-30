"""08장 실습 2 — 검증 달(2026-03~08) 예측 평가: AUC·Brier 와, 점수 구간별 실제 순수익.

실행: uv run python examples/chapter08/02_evaluate.py   (약 30초)
결과: reports/ml_pred_y.parquet, reports/ml_pred_y_net.parquet (08-3·08-4 에서 사용)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from vwap_trading.ml.model import LABELS, walk_forward

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
df = df[df.date < "2026-09-01"].reset_index(drop=True)
months = [str(p) for p in pd.period_range("2026-03", "2026-08", freq="M")]
Path("reports").mkdir(exist_ok=True)

for label, name in LABELS.items():
    r = walk_forward(df, label, months)
    r.to_parquet(f"reports/ml_pred_{label}.parquet")
    base = r[label].mean()
    print(f"\n[{name}] 검증 {len(r):,}건, 실제 비율 {base:.1%}")
    print(f"  AUC {roc_auc_score(r[label], r.p_raw):.3f} (월별 "
          + ", ".join(f"{m[-2:]}월 {roc_auc_score(g[label], g.p_raw):.2f}" for m, g in r.groupby("month")) + ")")
    print(f"  Brier {brier_score_loss(r[label], r.p_raw):.4f} (항상 실제 비율로 예측하면 {base * (1 - base):.4f})")
    q = pd.qcut(r.p_raw.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    t = r.groupby(q, observed=True).agg(점수=("p_raw", "mean"), 레이블=(label, "mean"), 목표가도달=("y", "mean"),
                                        순이익비율=("y_net", "mean"), 순평균=("net", "mean"))
    print("  점수 5분위 (Q5 = 모델이 가장 좋게 본 20%)")
    print(t.to_string(formatters={"점수": "{:.3f}".format, "레이블": "{:.1%}".format, "목표가도달": "{:.1%}".format,
                                  "순이익비율": "{:.1%}".format, "순평균": "{:+.3%}".format}))

r = pd.read_parquet("reports/ml_pred_y_net.parquet")
hit = r[r.reason == "target"]
print(f"\n[(2) 모델이 구분하는 것] 목표가 도달 표본 {len(hit)}건 안에서 순이익 여부 AUC "
      f"{roc_auc_score(hit.y_net, hit.p_raw):.3f}, 목표가 먼저 여부((1)) AUC {roc_auc_score(r.y, r.p_raw):.3f}")
