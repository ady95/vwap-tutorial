"""07장 실습 4 — 분할 방법에 따라 검증 성능이 얼마나 부풀려지는가.

같은 LightGBM 설정으로 세 가지 분할을 비교한다 (최종 테스트 구간 2026-09~ 은 쓰지 않는다).
  1. 무작위 5겹: 표본을 섞어 나눔
  2. 날짜 묶음 5겹: 같은 날짜는 한쪽에만, 순서는 무시
  3. 시간순 Walk-forward: 검증 달 이전 데이터로만 학습 (엠바고 1거래일, 퍼지)
실행: uv run python examples/chapter07/04_split_leakage.py   (01 실행 후)
"""

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from vwap_trading.ml.dataset import FEATURES
from vwap_trading.ml.split import walk_forward_folds

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
df = df[df.date < "2026-09-01"].reset_index(drop=True)          # 최종 테스트 구간 제외
X, y = df[FEATURES], df.y.to_numpy()


def model():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def oof_auc(splits):
    pred = np.full(len(df), np.nan)
    for tr, te in splits:
        pred[te] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    m = ~np.isnan(pred)
    return roc_auc_score(y[m], pred[m]), int(m.sum())


print(f"표본 {len(df):,}건 ({df.date.min():%Y-%m} ~ {df.date.max():%Y-%m}), y=1 {y.mean():.1%}")
auc, n = oof_auc(KFold(5, shuffle=True, random_state=0).split(X))
print(f"[1] 무작위 5겹          AUC {auc:.3f} (검증 {n:,}건)")
auc, n = oof_auc(GroupKFold(5).split(X, groups=df.date))
print(f"[2] 날짜 묶음 5겹       AUC {auc:.3f} (검증 {n:,}건)")
months = [str(m) for m in pd.period_range("2026-03", "2026-08", freq="M")]
folds = list(walk_forward_folds(df, months, embargo_days=1))
auc, n = oof_auc([(tr, te) for _, tr, te in folds])
print(f"[3] 시간순 Walk-forward AUC {auc:.3f} (검증 {n:,}건, 2026-03~08)")
for m, tr, te in folds:
    p = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    print(f"    {m}: 학습 {len(tr):,}건 (~{df.date.iloc[tr].max():%m-%d}) → 검증 {len(te)}건 AUC {roc_auc_score(y[te], p):.3f}")
