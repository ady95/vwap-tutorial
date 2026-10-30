"""02장 데이터 정제·세션 필터·종목 선정 테스트 (네트워크 없이 실행)."""

import pandas as pd

from vwap_trading.data.minute import to_frame
from vwap_trading.data.selection import make_selections
from vwap_trading.data.session import regular_session


def bars_at(times, close=100.0, volume=10):
    idx = pd.DatetimeIndex(pd.to_datetime(times), name="time")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx)


def test_to_frame_strips_sign_sorts_and_dedups():
    rows = [
        {"cntr_tm": "20261030090100", "open_pric": "-100", "high_pric": "+102", "low_pric": "-99", "cur_prc": "-101", "trde_qty": "5"},
        {"cntr_tm": "20261030090000", "open_pric": "100", "high_pric": "101", "low_pric": "99", "cur_prc": "100", "trde_qty": "7"},
        {"cntr_tm": "20261030090100", "open_pric": "-100", "high_pric": "+102", "low_pric": "-99", "cur_prc": "-101", "trde_qty": "5"},
    ]
    df = to_frame(rows)
    assert list(df.index.strftime("%H:%M")) == ["09:00", "09:01"]
    assert df.loc["2026-10-30 09:01", "close"] == 101
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_to_frame_empty_has_datetime_index():
    assert isinstance(to_frame([]).index, pd.DatetimeIndex)


def test_regular_session_drops_after_market():
    df = regular_session(bars_at(["2026-10-29 08:59", "2026-10-29 09:00", "2026-10-29 15:30", "2026-10-29 15:40"]))
    assert list(df.index.strftime("%H:%M")) == ["09:00", "15:30"]


def test_regular_session_special_day_suneung():
    df = regular_session(bars_at(["2025-11-13 09:30", "2025-11-13 10:00", "2025-11-13 16:00", "2025-11-13 16:30"]))
    assert list(df.index.strftime("%H:%M")) == ["10:00", "16:00", "16:30"]


def test_selection_uses_only_past_days():
    days = pd.bdate_range("2026-01-05", periods=22)
    def stock(last_day_volume):
        times = [f"{d.date()} 09:00" for d in days]
        df = bars_at(times, volume=10)
        df.iloc[-1, df.columns.get_loc("volume")] = last_day_volume
        return df
    sel = make_selections({"AAA": stock(100), "BBB": stock(10)}, top_n=3, lookback=20, min_ratio=1.5)
    # 21번째 날부터 과거 20일 평균이 생기고, 마지막 날 AAA 만 거래대금이 10배
    assert list(sel.symbol) == ["AAA"]
    assert str(sel.selected_at.iloc[0]) == f"{days[-1].date()} 09:30:00"
