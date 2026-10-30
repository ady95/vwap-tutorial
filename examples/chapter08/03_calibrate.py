"""08장 실습 3 — 확률 보정과 승인 임계값: 기대 순수익이 0 보다 큰 후보만 승인한다.

실행: uv run python examples/chapter08/03_calibrate.py   (02 실행 후)
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from vwap_trading.ml.model import LABELS

sys.stdout.reconfigure(encoding="utf-8")

for label, name in LABELS.items():
    r = pd.read_parquet(f"reports/ml_pred_{label}.parquet")
    print(f"\n[{name}] 검증 {len(r):,}건")
    # 1. 신뢰도 표: 예측 확률 구간별 실제 비율
    bins = pd.cut(r.p_raw, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], include_lowest=True)
    t = r.groupby(bins, observed=True).agg(건수=(label, "size"), 원확률=("p_raw", "mean"), 보정확률=("p_cal", "mean"),
                                           실제=(label, "mean"))
    print(t.to_string(formatters={k: "{:.3f}".format for k in ("원확률", "보정확률", "실제")}))
    print(f"  Brier 원확률 {brier_score_loss(r[label], r.p_raw):.4f} → 보정 {brier_score_loss(r[label], r.p_cal):.4f}")

    # 2. 승인 규칙: 기대 순수익 = p × 성공 평균 + (1 − p) × 실패 평균 > 0
    w = r.groupby("month")[["win", "loss"]].first()
    w["손익분기 p"] = (-w.loss / (w.win - w.loss)).where(w.win > 0, np.inf)
    w["최대 보정확률"] = r.groupby("month").p_cal.max()
    w["승인"] = r.groupby("month").approve.sum()
    print("  검증 달별 (성공·실패 평균은 그 달 이전 학습 구간 값)")
    print(w.to_string(formatters={"win": "{:+.3%}".format, "loss": "{:+.3%}".format, "손익분기 p": "{:.3f}".format,
                                  "최대 보정확률": "{:.3f}".format}))
    a = r[r.approve]
    print(f"  기대 순수익 규칙 승인 {len(a)}건" + (f", 실제 순 평균 {a.net.mean():+.3%}" if len(a) else ""))
    top = r[r.top30]
    print(f"  (비교) 상위 30% 승인 {len(top)}건 순 평균 {top.net.mean():+.3%} | 거부 {int((~r.top30).sum())}건 "
          f"{r[~r.top30].net.mean():+.3%} | 전체 {r.net.mean():+.3%}")

# 3. 비용 산술: 목표가 먼저일 확률이 얼마여야 손익분기인가 (변동성별)
r = pd.read_parquet("reports/ml_pred_y.parquet")
cost = (r.ret - r.net).mean()
print(f"\n[비용 산술] 매매당 평균 비용 c = {cost:.3%}, 손절 1.5 ATR, 목표 2.0 ATR 일 때 손익분기 p = (1.5a + c) / 3.5a")
for qv in (0.1, 0.5, 0.9):
    a_ = r.atr_pct.quantile(qv)
    print(f"  ATR/가격 a 의 {int(qv * 100)}% 분위 {a_:.3%}: 손익분기 p = {(1.5 * a_ + cost) / (3.5 * a_):.2f}")
print(f"  검증 표본의 실제 목표가 도달 비율 {r.y.mean():.1%}")
