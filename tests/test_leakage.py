"""05장 누수 테스트 — 미래 봉을 바꿔도 과거 매매가 같은가, 선정 시각 전에는 매매하지 않는가."""

import numpy as np
import pandas as pd

from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.strategy.rules import StrategyParams


def random_day(seed, day="2026-03-03", n=390):
    rng = np.random.default_rng(seed)
    close = 20000 + np.cumsum(rng.normal(0, 25, n)).round(-1)
    idx = pd.date_range(f"{day} 09:00", periods=n, freq="1min", name="time")
    return pd.DataFrame({"open": close, "high": close + 30, "low": close - 30, "close": close,
                         "volume": rng.integers(100, 5000, n).astype(float)}, index=idx)


def test_changing_future_bars_keeps_past_trades():
    bars = random_day(3)
    sel = pd.DataFrame({"symbol": ["000001"], "tradable_from": [pd.Timestamp("2026-03-03 09:30")]})
    for strategy in ("S0", "A"):
        p = StrategyParams(strategy=strategy, rho=0.0)
        base = backtest({"000001": compute_indicators(bars, p.vwap)}, sel, p)
        cut = pd.Timestamp("2026-03-03 12:00")
        changed = bars.copy()
        after = changed.index > cut
        changed.loc[after, ["open", "high", "low", "close"]] *= 1.3      # 미래 봉을 30% 올린다
        changed.loc[after, "volume"] *= 10
        other = backtest({"000001": compute_indicators(changed, p.vwap)}, sel, p)
        done_before = base[base.exit_time <= cut].reset_index(drop=True)
        assert len(done_before) > 0
        pd.testing.assert_frame_equal(done_before, other[other.exit_time <= cut].reset_index(drop=True))


def test_no_trade_before_tradable_from():
    bars = random_day(4)
    p = StrategyParams(strategy="S0", rho=0.0)
    ind = {"000001": compute_indicators(bars, p.vwap)}
    late = pd.Timestamp("2026-03-03 13:00")
    sel = pd.DataFrame({"symbol": ["000001"], "tradable_from": [late]})
    early = backtest(ind, sel.assign(tradable_from=pd.Timestamp("2026-03-03 09:30")), p)
    assert (early.signal_time < late).any()                              # 늦게 선정되지 않았다면 매매했을 신호
    assert (backtest(ind, sel, p).signal_time >= late).all()
