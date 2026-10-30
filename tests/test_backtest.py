"""05장 백테스트 테스트 — 비용 손 계산, 엔진과 04장 규칙의 일치, 성과 지표."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from vwap_trading.backtest.costs import CostModel, sell_tax_rate, tick_size
from vwap_trading.backtest.engine import backtest
from vwap_trading.backtest.metrics import max_drawdown, summarize
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.strategy.rules import StrategyParams, run


@pytest.mark.parametrize("price, tick", [(1_999, 1), (2_000, 5), (19_990, 10), (20_000, 50),
                                         (199_900, 100), (200_000, 500), (500_000, 1_000)])
def test_tick_size_boundaries(price, tick):
    assert tick_size(price) == tick


def test_sell_tax_by_trade_date():
    assert sell_tax_rate(date(2025, 12, 30)) == 0.0015
    assert sell_tax_rate(date(2026, 1, 2)) == 0.0020


def test_net_return_by_hand():
    # 10,000원 매수 → 10,100원 매도, 2026년. 호가 10원, 수수료 0.015%, 세금 0.20%
    buy = 10_010 * 1.00015
    sell = 10_090 * (1 - 0.00015 - 0.0020)
    assert CostModel().net_return(10_000, 10_100, date(2026, 3, 3)) == pytest.approx(sell / buy - 1)
    no_cost = CostModel(fee=0, slip_ticks=0, tax=False)
    assert no_cost.net_return(10_000, 10_100, date(2026, 3, 3)) == pytest.approx(0.01)


def synthetic(seed, symbol_days=6):
    rng = np.random.default_rng(seed)
    days, sel = [], []
    for d in pd.bdate_range("2026-03-02", periods=symbol_days):
        n = 390
        close = 20000 + np.cumsum(rng.normal(0, 25, n)).round(-1)
        idx = pd.date_range(d + pd.Timedelta(hours=9), periods=n, freq="1min", name="time")
        days.append(pd.DataFrame({"open": close, "high": close + 30, "low": close - 30, "close": close,
                                  "volume": rng.integers(100, 5000, n).astype(float)}, index=idx))
        sel.append({"symbol": "000001", "tradable_from": d + pd.Timedelta(hours=9, minutes=30)})
    return {"000001": compute_indicators(pd.concat(days))}, pd.DataFrame(sel)


@pytest.mark.parametrize("strategy, compare", [("S0", 1), ("S0", 2), ("A", 1), ("A", 2)])
def test_engine_matches_chapter4_rules(strategy, compare):
    ind, sel = synthetic(1)
    p = StrategyParams(strategy=strategy, compare=compare, rho=0.0)
    cols = ["symbol", "signal_time", "entry_time", "exit_time", "reason", "bars", "ret"]
    a = backtest(ind, sel, p)[cols].sort_values("signal_time").reset_index(drop=True)
    b = run(ind, sel, p)[cols].sort_values("signal_time").reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_max_positions_limits_overlap():
    ind, sel = synthetic(2)
    sel2 = pd.concat([sel, sel.assign(symbol="000002")])          # 같은 가격의 두 번째 종목
    ind["000002"] = ind["000001"]
    p = StrategyParams(strategy="S0", rho=0.0)
    both = backtest(ind, sel2, p)
    one = backtest(ind, sel2, p, max_positions=1)
    assert len(both) == 2 * len(backtest(ind, sel, p))
    assert 0 < len(one) < len(both)
    for _, a in one.iterrows():                                   # 보유 구간이 겹치지 않는다
        others = one[(one.entry_time < a.exit_time) & (one.exit_time > a.entry_time)]
        assert len(others) == 1


def test_metrics_by_hand():
    t = pd.DataFrame({"exit_time": pd.date_range("2026-03-02 10:00", periods=5, freq="1h"),
                      "net": [0.02, -0.01, -0.03, 0.01, 0.02], "bars": [5, 5, 5, 5, 5]})
    m = summarize(t)
    assert m["win_rate"] == 0.6
    assert m["profit_factor"] == pytest.approx(0.05 / 0.04)
    assert m["payoff"] == pytest.approx((0.05 / 3) / 0.02)
    # 누적: 0.02, 0.01, -0.02, -0.01, 0.01 → 고점 0.02 에서 -0.02 까지 0.04
    assert m["mdd"] == pytest.approx(0.04)
    assert max_drawdown(pd.Series([-0.01, -0.02])) == pytest.approx(0.02)   # 시작점 0 도 고점
