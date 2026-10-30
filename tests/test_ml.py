"""07장 학습 데이터 테스트 — Triple Barrier 레이블, 피처 시점, 시간순 분할."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vwap_trading.backtest.costs import CostModel
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.ml.dataset import FEATURES, features, triple_barrier
from vwap_trading.ml.split import fold, walk_forward_folds
from vwap_trading.optimize.search import FinalTestLeak
from vwap_trading.strategy.rules import StrategyParams

P = StrategyParams()
NO_COST = CostModel(fee=0, slip_ticks=0, tax=False)


def flat_day(n=390, start="2026-03-03 09:00"):
    idx = pd.date_range(start, periods=n, freq="1min", name="time")
    x = pd.DataFrame({"open": 10000.0, "high": 10000.0, "low": 10000.0, "close": 10000.0}, index=idx)
    x["atr_s"] = 10.0                                   # 손절 10000 − 15, 목표 10000 + 20
    return x


def test_target_before_stop_is_success():
    x = flat_day()
    x.iloc[105, x.columns.get_loc("high")] = 10021.0
    x.iloc[108, x.columns.get_loc("low")] = 9980.0
    lab = triple_barrier(x, np.array([100]), P, NO_COST).iloc[0]
    assert lab.reason == "target" and lab.y == 1 and lab.bars == 5


def test_same_bar_stop_and_target_is_failure():
    x = flat_day()
    x.iloc[105, [x.columns.get_loc("high"), x.columns.get_loc("low")]] = [10021.0, 9980.0]
    lab = triple_barrier(x, np.array([100]), P, NO_COST).iloc[0]
    assert lab.reason == "stop" and lab.y == 0


def test_time_exit_label_follows_net_return():
    x = flat_day()
    x.iloc[131, x.columns.get_loc("open")] = 10010.0     # 30봉 보유 후 다음 봉 시가 +0.1%
    up = triple_barrier(x, np.array([100]), P, NO_COST).iloc[0]
    assert up.reason == "time" and up.net > 0 and up.y == 1
    down = triple_barrier(x, np.array([100]), P, CostModel()).iloc[0]   # 비용을 빼면 손실
    assert down.reason == "time" and down.net < 0 and down.y == 0


def test_features_do_not_change_when_future_changes():
    rng = np.random.default_rng(1)
    n = 390
    close = 20000 + np.cumsum(rng.integers(-3, 4, n)) * 10.0
    idx = pd.date_range("2026-03-03 09:00", periods=n, freq="1min", name="time")
    bars = pd.DataFrame({"open": close, "high": close + 20, "low": close - 20, "close": close,
                         "volume": rng.integers(1, 3000, n).astype(float)}, index=idx)
    changed = bars.copy()
    cut = 200
    changed.iloc[cut + 1:, :4] *= 1.2
    changed.iloc[cut + 1:, 4] *= 5
    a = features(compute_indicators(bars)).iloc[: cut + 1]
    b = features(compute_indicators(changed)).iloc[: cut + 1]
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def sample_frame():
    days = pd.bdate_range("2026-01-26", "2026-03-31")
    df = pd.DataFrame({"date": days, "time": days + pd.Timedelta(hours=10)})
    df["exit_time"] = df.time + pd.Timedelta(minutes=20)
    return df


def test_fold_embargo_and_purge():
    df = sample_frame()
    train, test = fold(df, "2026-03", embargo_days=1)
    assert df.date.iloc[test].dt.month.eq(3).all()
    assert df.date.iloc[train].max() == pd.Timestamp("2026-02-26")      # 2026-02-27 은 엠바고로 제외
    df.loc[df.date == "2026-02-25", "exit_time"] = pd.Timestamp("2026-03-02 09:10")   # 관측 기간이 검증으로 넘어감
    train, _ = fold(df, "2026-03", embargo_days=1)
    assert pd.Timestamp("2026-02-25") not in set(df.date.iloc[train])              # 퍼지


def test_final_test_month_cannot_be_used():
    df = sample_frame()
    with pytest.raises(ValueError):
        list(walk_forward_folds(df, ["2026-09"]))
    late = pd.DataFrame({"date": [pd.Timestamp("2026-09-01")], "time": [pd.Timestamp("2026-09-01 10:00")],
                         "exit_time": [pd.Timestamp("2026-09-01 10:20")]})
    with pytest.raises(FinalTestLeak):
        fold(pd.concat([late, pd.DataFrame({"date": [pd.Timestamp("2026-10-01")],
                                            "time": [pd.Timestamp("2026-10-01 10:00")],
                                            "exit_time": [pd.Timestamp("2026-10-01 10:20")]})]), "2026-10", 0)
