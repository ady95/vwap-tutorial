"""LightGBM 필터의 실시간 연결 (10장). 08장 C 와 같은 모델·보정·승인 규칙을 쓴다.

mode
  "log"  : 확률과 기대 순수익을 계산해 기록만 하고 매수는 모두 승인한다 (11장 모의매매 기본값)
  "rule" : 08장 승인 규칙 (기대 순수익 > 0) 을 통과한 후보만 승인한다

11장 모의매매의 목적은 백테스트 가정(체결·봉 시각·슬리피지)의 검증이다. 08·09장에서 승인 규칙은
8개월 동안 한 건도 승인하지 않았으므로, "rule" 로 돌리면 매매가 없어 검증할 체결도 없다.
그래서 기본값은 "log" 다.
"""

from __future__ import annotations

import time
from datetime import date

import joblib
import numpy as np
import pandas as pd

from vwap_trading.ml.dataset import FEATURES, features
from vwap_trading.ml.model import expected_net, fit_calibrated


class LiveFilter:
    def __init__(self, model, iso, win: float, loss: float, trained_until: str, mode: str = "log"):
        if mode not in ("log", "rule"):
            raise ValueError(f"mode 는 log 또는 rule: {mode}")
        self.model, self.iso, self.win, self.loss = model, iso, win, loss
        self.trained_until, self.mode = trained_until, mode

    @classmethod
    def train(cls, df: pd.DataFrame, day: date, label: str = "y", mode: str = "log") -> "LiveFilter":
        """day 이전 거래일의 후보만으로 학습한다. 오늘 데이터는 들어가지 않는다."""
        train = df[df.date < pd.Timestamp(day)]
        model, iso = fit_calibrated(train, label)
        win, loss = train.net[train[label] == 1].mean(), train.net[train[label] == 0].mean()
        return cls(model, iso, float(win), float(loss), f"{train.date.max():%Y-%m-%d}", mode)

    def save(self, path) -> None:
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path, mode: str | None = None) -> "LiveFilter":
        d = joblib.load(path)
        if mode:
            d["mode"] = mode
        return cls(**d)

    def __call__(self, symbol: str, x: pd.DataFrame) -> tuple[bool, dict]:
        """마지막 확정 봉(신호 봉)의 피처로 승인 여부를 정한다."""
        t0 = time.perf_counter()
        f = features(x).iloc[[-1]][FEATURES]
        p_raw = float(self.model.predict_proba(f)[:, 1][0])
        p_cal = float(self.iso.predict(np.array([p_raw]))[0])
        e = float(expected_net(p_cal, self.win, self.loss))
        ok = True if self.mode == "log" else e > 0
        return ok, {"p_raw": round(p_raw, 4), "p_cal": round(p_cal, 4), "exp_net": round(e, 6),
                    "rule_ok": e > 0, "ms": round((time.perf_counter() - t0) * 1000, 1)}
