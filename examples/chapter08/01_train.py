"""08장 실습 1 — 두 레이블로 LightGBM 을 학습하고, 무엇을 중요하게 보는지 확인한다.

실행: uv run python examples/chapter08/01_train.py   (07장 01 실행 후)
"""

import sys
import time

import pandas as pd

from vwap_trading.ml.dataset import FEATURES
from vwap_trading.ml.model import LABELS, PARAMS, fit_calibrated
from vwap_trading.ml.split import fold

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
df = df[df.date < "2026-09-01"].reset_index(drop=True)          # 최종 테스트 구간은 쓰지 않는다
print("LightGBM 설정:", PARAMS)
print(f"표본 {len(df):,}건, 레이블 비율: " + ", ".join(f"{LABELS[k]} {df[k].mean():.1%}" for k in LABELS))

print("\n[검증 달별 학습 구간]")
for m in [str(p) for p in pd.period_range("2026-03", "2026-08", freq="M")]:
    tr, te = fold(df, m)
    train = df.iloc[tr]
    cal_from = train.date.dt.to_period("M").max() - 1
    print(f"  {m}: 학습 {len(tr):,}건 ({train.date.min():%Y-%m} ~ {train.date.max():%m-%d}), "
          f"그중 보정용 {int((train.date.dt.to_period('M') >= cal_from).sum())}건 ({cal_from}~), 검증 {len(te)}건")

# 마지막 창(검증 2026-08)의 모델로 피처 중요도를 본다
tr, _ = fold(df, "2026-08")
for label, name in LABELS.items():
    t0 = time.perf_counter()
    model, _ = fit_calibrated(df.iloc[tr], label)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    share = (gain / gain.sum()).sort_values(ascending=False)
    print(f"\n[{name}] 학습 {time.perf_counter() - t0:.1f}초, 중요도(gain 비중) 상위 6개:")
    print("  " + ", ".join(f"{k} {v:.1%}" for k, v in share.head(6).items()))
