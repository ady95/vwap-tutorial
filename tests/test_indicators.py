"""03장 지표 테스트 — 손으로 계산한 값과 대조, 소급·누수 방지 확인."""

import numpy as np
import pandas as pd
import pytest

from vwap_trading.indicators.vwap import (FLAT, UP, VwapParams, atr, box_center, classify_state,
                                          compute_indicators, compute_indicators_reference, relative_volume,
                                          rolling_vwap, session_vwap)


def day(rows, start="2026-10-29 09:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", name="time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


def test_session_vwap_by_hand():
    d = day([[100, 102, 98, 100, 100], [100, 104, 100, 102, 300], [102, 103, 99, 101, 100]])
    # TP = 100, 102, 101 → (100*100 + 102*300 + 101*100) / 500 = 101.4
    assert session_vwap(d).iloc[-1] == pytest.approx(101.4)
    assert session_vwap(d).iloc[0] == pytest.approx(100.0)


def test_rolling_vwap_no_partial_window():
    d = day([[100, 102, 98, 100, 100], [100, 104, 100, 102, 300], [102, 103, 99, 101, 100]])
    r = rolling_vwap(d, 2)
    assert np.isnan(r.iloc[0])
    assert r.iloc[2] == pytest.approx((102 * 300 + 101 * 100) / 400)


def test_atr_first_bar_uses_today_only():
    d = day([[100, 105, 95, 100, 1], [100, 101, 99, 100, 1]])
    tr_first = 105 - 95
    assert atr(d, 1).iloc[0] == tr_first


def test_relative_volume_excludes_current_bar():
    d = day([[1, 1, 1, 1, 10], [1, 1, 1, 1, 10], [1, 1, 1, 1, 40]])
    assert relative_volume(d, 2).iloc[2] == pytest.approx(4.0)    # 40 / 평균(10, 10)


def test_state_flat_after_m_bars_without_backfill():
    s = np.array([0.2, 0.01, 0.01, 0.01, 0.2])
    st = classify_state(s, theta=0.05, m=3)
    # 1~2번째 작은 기울기 봉은 직전 상태(UP)를 유지하고, 3번째에서 FLAT. 소급해서 바꾸지 않는다
    assert list(st) == [UP, UP, UP, FLAT, UP]


def test_box_center_fixed_during_flat():
    tp = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    vol = np.ones(5)
    state = np.array([UP, FLAT, FLAT, FLAT, UP], dtype=float)
    box = box_center(tp, vol, state, m=2)
    assert box[1] == pytest.approx(15.0)                          # t0=1 에서 최근 2봉 평균
    assert box[2] == box[3] == box[1]                             # FLAT 동안 고정
    assert np.isnan(box[4])


def test_indicators_reset_each_day_and_do_not_use_future():
    rng = np.random.default_rng(0)
    rows = lambda n: [[p, p + 1, p - 1, p, v] for p, v in zip(100 + rng.normal(0, 1, n).cumsum(), rng.integers(1, 100, n))]
    bars = pd.concat([day(rows(60), "2026-10-28 09:00"), day(rows(60), "2026-10-29 09:00")])
    p = VwapParams(N=10, k=3, m=4)
    full = compute_indicators(bars, p)
    # 둘째 날 첫 봉의 세션 VWAP 은 그 봉의 TP 와 같아야 한다 (전일 누적 없음)
    second = full.loc["2026-10-29"]
    assert second.svwap.iloc[0] == pytest.approx(second.tp.iloc[0])
    assert np.isnan(second.rvwap.iloc[p.N - 2])
    # 미래 봉을 지워도 과거 봉의 지표 값이 바뀌지 않아야 한다 (리페인팅 없음)
    cut = compute_indicators(bars.loc[:"2026-10-29 09:40"], p)
    cols = ["svwap", "rvwap", "atr", "slope", "state", "mvwap"]
    pd.testing.assert_frame_equal(full.loc[cut.index, cols], cut[cols])


@pytest.mark.parametrize("params", [VwapParams(), VwapParams(N=10, k=3, theta=0.15, m=5), VwapParams(N=90, k=15, m=30)])
def test_vectorized_matches_reference(params):
    """06장에서 빠르게 바꾼 compute_indicators 가 03장 기준 구현과 같은 값을 내는가."""
    rng = np.random.default_rng(7)
    days = []
    for d in pd.bdate_range("2026-03-02", periods=3):
        n = 390
        close = 10000 + np.cumsum(rng.integers(-2, 3, n)) * 10       # 정수 호가 → 기울기 동률이 생긴다
        idx = pd.date_range(d + pd.Timedelta(hours=9), periods=n, freq="1min", name="time")
        days.append(pd.DataFrame({"open": close, "high": close + 10, "low": close - 10, "close": close,
                                  "volume": rng.integers(0, 50, n)}, index=idx))
    bars = pd.concat(days)
    fast, ref = compute_indicators(bars, params), compute_indicators_reference(bars, params)
    cols = ["bar_no", "svwap", "rvwap", "atr", "atr_s", "slope", "state", "box", "mvwap", "rvol"]
    pd.testing.assert_frame_equal(fast[cols], ref[cols], check_dtype=False, atol=1e-6)
    assert (fast.atr_low == ref.atr_low).all()
