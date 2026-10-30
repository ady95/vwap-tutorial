"""LightGBM 매수 후보 필터 (08장).

검증 달마다 그 이전 데이터로 학습한다 (07장 split.fold: 확장 창·엠바고·퍼지·최종 테스트 차단).
학습 구간의 마지막 CALIB_MONTHS 개월은 확률 보정에만 쓰고, 나머지로 모델을 맞춘다.

승인 규칙 (결과를 보기 전에 정함):
  기대 순수익 = p × (학습 구간 성공 표본의 순수익 평균) + (1 − p) × (실패 표본의 순수익 평균) > 0
  p 는 보정된 확률. 학습 구간의 평균값만 쓰므로 검증 구간 정보는 들어가지 않는다.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from vwap_trading.ml.dataset import FEATURES
from vwap_trading.ml.split import fold

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
CALIB_MONTHS = 2
LABELS = {"y": "(1) 목표가 먼저", "y_net": "(2) 순이익 > 0"}


def fit(train: pd.DataFrame, label: str) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**PARAMS).fit(train[FEATURES], train[label])


def fit_calibrated(train: pd.DataFrame, label: str):
    """학습 구간을 (모델용, 보정용)으로 시간순으로 나눠 모델과 등위 회귀 보정기를 만든다."""
    months = train.date.dt.to_period("M")
    cut = months.max() - (CALIB_MONTHS - 1)
    fit_part, cal_part = train[months < cut], train[months >= cut]
    model = fit(fit_part, label)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(model.predict_proba(cal_part[FEATURES])[:, 1], cal_part[label])
    return model, iso


def expected_net(p: np.ndarray, win: float, loss: float) -> np.ndarray:
    return p * win + (1 - p) * loss


def walk_forward(df: pd.DataFrame, label: str, test_months: list[str], embargo_days: int = 1) -> pd.DataFrame:
    """검증 달마다 학습·보정한 모델로 예측한다. 반환: 검증 표본 + p_raw, p_cal, exp_net, approve, q70."""
    out = []
    for m in test_months:
        tr, te = fold(df, m, embargo_days)
        train, test = df.iloc[tr], df.iloc[te]
        model, iso = fit_calibrated(train, label)
        p_raw = model.predict_proba(test[FEATURES])[:, 1]
        p_cal = iso.predict(p_raw)
        win, loss = train.net[train[label] == 1].mean(), train.net[train[label] == 0].mean()
        months = train.date.dt.to_period("M")
        cal_part = train[months >= months.max() - (CALIB_MONTHS - 1)]
        q70 = np.quantile(model.predict_proba(cal_part[FEATURES])[:, 1], 0.7)   # 비교용: 상위 30% 승인
        e = expected_net(p_cal, win, loss)
        out.append(test.assign(month=m, p_raw=p_raw, p_cal=p_cal, exp_net=e, approve=e > 0,
                               top30=p_raw >= q70, win=win, loss=loss, n_train=len(train)))
    return pd.concat(out)
