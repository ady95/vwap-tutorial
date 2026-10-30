"""10~11장 실시간 모듈 테스트 — 봉 만들기, 주문 상태, 위험관리, 실계좌 차단, 백테스트와의 판정 일치."""

from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from vwap_trading.backtest.engine import backtest
from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.broker.orders import PaperBroker, RealOrderBlocked
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.live.bars import MinuteBarBuilder, parse_0b
from vwap_trading.live.journal import Journal
from vwap_trading.live.oms import Order, OrderBook
from vwap_trading.live.replay import SimBroker, replay
from vwap_trading.live.risk import RiskEngine, RiskLimits
from vwap_trading.live.runner import Runner
from vwap_trading.strategy.rules import StrategyParams

T = datetime(2026, 11, 2, 10, 15)


def at(sec):
    return T + pd.Timedelta(seconds=sec)


def test_parse_0b_strips_signs():
    ts, px, q = parse_0b({"20": "101503", "10": "-72000", "15": "-30"}, date(2026, 11, 2))
    assert (ts, px, q) == (datetime(2026, 11, 2, 10, 15, 3), 72000.0, 30)


def test_bar_confirmed_by_next_minute_tick():
    b = MinuteBarBuilder()
    assert b.on_tick("A", at(0), 100, 5) == []
    assert b.on_tick("A", at(30), 103, 5) == []
    assert b.on_tick("A", at(59), 99, 5) == []
    done = b.on_tick("A", at(61), 101, 1)                    # 10:16:01 체결 → 10:15 봉 확정
    assert len(done) == 1 and done[0].how == "tick"
    bar = done[0]
    assert (bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.ticks) == (T, 100, 103, 99, 99, 15, 3)


def test_bar_confirmed_by_timer_and_late_tick_dropped():
    b = MinuteBarBuilder(grace=2.0)
    b.on_tick("A", at(10), 100, 5)
    assert b.due(at(61)) == []                               # 끝난 지 1초: 아직
    [(sym, bar)] = b.due(at(62))                             # 끝난 지 2초: 확정
    assert sym == "A" and bar.how == "timer"
    assert b.on_tick("A", at(50), 120, 9) == []              # 확정된 10:15 봉의 늦은 체결
    assert b.late["A"] == 1 and "A" not in b.cur


def ev(no, status, qty, open_qty, amount=0, kind="+매수", orig="0000000", reject=""):
    return {"9203": no, "913": status, "900": str(qty), "902": str(open_qty), "903": str(amount), "904": orig,
            "905": kind, "919": reject}


def test_fills_are_cumulative_and_idempotent():
    book = OrderBook()
    o = book.add(Order("k", "A", "buy", 10, None))
    book.sent(o, "0000123", T)
    book.on_event(ev("0000123", "체결", 10, 6, 400_000), T)
    book.on_event(ev("0000123", "체결", 10, 6, 400_000), T)   # 같은 이벤트가 두 번
    assert (o.filled, o.status) == (4, "PARTIAL")
    book.on_event(ev("123", "체결", 10, 0, 1_000_000), T)     # 0 채움 자릿수가 달라도 같은 주문
    assert (o.filled, o.status, o.avg) == (10, "FILLED", 100_000)


def test_event_before_order_response_is_kept():
    book = OrderBook()
    o = book.add(Order("k", "A", "buy", 5, None))
    assert book.on_event(ev("0000007", "체결", 5, 0, 500_000), T) is None   # 주문번호를 아직 모른다
    changed = book.sent(o, "0000007", T)
    assert changed == [o] and o.status == "FILLED"


def test_cancel_and_reject_events():
    book = OrderBook()
    o = book.add(Order("k", "A", "buy", 5, 1000))
    book.sent(o, "0000001", T)
    book.on_event(ev("0000002", "확인", 5, 0, kind="매수취소", orig="0000001"), T)
    assert o.status == "CANCELLED"
    r = book.add(Order("k2", "A", "buy", 5, 1000))
    book.sent(r, "0000003", T)
    book.on_event(ev("0000003", "접수", 5, 5, reject="주문가능금액 부족"), T)
    assert r.status == "REJECTED"
    with pytest.raises(ValueError):
        book.add(Order("k", "A", "buy", 5, None))            # 같은 키로 두 번 주문하지 않는다


def test_risk_limits(tmp_path):
    lim = RiskLimits(budget=1_000_000, max_positions=2, daily_loss=30_000, kill_file=str(tmp_path / "STOP"))
    r = RiskEngine(lim)
    assert r.qty(70_000) == 14
    assert r.check_entry(70_000, 1, 0) is None
    assert "동시 보유" in r.check_entry(70_000, 2, 0)
    assert "1주 가격" in r.check_entry(1_500_000, 0, 0)
    assert "일일 손실" in r.check_entry(70_000, 0, -30_000)
    assert "일일 손실" in r.check_entry(70_000, 0, +50_000)  # 한 번 걸리면 그날은 풀리지 않는다
    k = RiskEngine(lim)
    (tmp_path / "STOP").touch()
    assert "STOP" in k.check_entry(70_000, 0, 0)


def test_orders_blocked_unless_paper(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    with pytest.raises(RealOrderBlocked):
        PaperBroker(KiwoomClient("k", "s", env="real"))
    paper = KiwoomClient("k", "s", env="paper")
    monkeypatch.setenv("TRADING_MODE", "offline")
    with pytest.raises(RealOrderBlocked):
        PaperBroker(paper)
    monkeypatch.setenv("TRADING_MODE", "paper")
    broker = PaperBroker(paper)
    broker.client.host = "https://api.kiwoom.com"             # 만든 뒤에 주소를 바꿔도
    with pytest.raises(RealOrderBlocked):
        broker.buy("005930", 1)                               # 보내기 직전에 다시 막는다


def random_day(seed, day="2026-10-29"):
    rng = np.random.default_rng(seed)
    n = 390
    close = 10000 + np.cumsum(rng.normal(0, 12, n)).round(-1)
    idx = pd.date_range(f"{day} 09:00", periods=n, freq="1min", name="time")
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) + 20, "low": np.minimum(o, close) - 20,
                         "close": close, "volume": rng.integers(100, 5000, n).astype(float)}, index=idx)


@pytest.mark.parametrize("strategy", ["S0", "A"])
def test_replay_matches_backtest_decisions(strategy):
    """같은 봉을 넣으면 실시간 실행기와 백테스트 엔진의 매매 판정(시각·사유)이 같아야 한다."""
    p = StrategyParams(strategy=strategy, rho=0.0)
    day = date(2026, 10, 29)
    bars = {"AAA": random_day(3), "BBB": random_day(4)}
    sel = pd.DataFrame({"symbol": list(bars), "tradable_from": pd.Timestamp("2026-10-29 09:30")})
    bt = backtest({s: compute_indicators(b, p.vwap) for s, b in bars.items()}, sel, p)
    r = Runner(p, SimBroker(), Journal(":memory:"), RiskEngine(RiskLimits(budget=10**9, max_positions=99, daily_loss=10**15,
                                                                           max_entries=999)), day)
    replay(r, r.broker, bars, day, {s: datetime(2026, 10, 29, 9, 30) for s in bars})
    live = pd.DataFrame(r.trades)
    keys = ["symbol", "signal_time", "entry_time", "exit_time", "reason"]
    assert len(bt) >= 3
    a = bt[keys].astype(str).sort_values(keys).reset_index(drop=True)
    b = live[keys].astype(str).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert (bt.sort_values(keys).entry.to_numpy() == live.sort_values(keys).entry.to_numpy()).all()


def test_max_positions_blocks_second_symbol():
    p = StrategyParams(strategy="S0", rho=0.0)
    day = date(2026, 10, 29)
    same = random_day(3)
    bars = {"AAA": same, "BBB": same.copy()}                # 두 종목이 같은 시각에 같은 신호
    limits = RiskLimits(budget=10**9, max_positions=1, daily_loss=10**15)
    r = Runner(p, SimBroker(), Journal(":memory:"), RiskEngine(limits), day)
    replay(r, r.broker, bars, day, {s: datetime(2026, 10, 29, 9, 30) for s in bars})
    blocks = r.journal.table("events").query("kind == 'risk_block'")
    assert len(blocks) > 0 and blocks.data.str.contains("동시 보유").all()
    t = pd.DataFrame(r.trades).sort_values("entry_time")
    assert (t.entry_time.iloc[1:].to_numpy() >= t.exit_time.iloc[:-1].to_numpy()).all()   # 보유 구간이 겹치지 않는다


def test_restart_does_not_resend_same_order(tmp_path):
    p = StrategyParams(strategy="S0", rho=0.0)
    day = date(2026, 10, 29)
    bars = {"AAA": random_day(1)}
    j = Journal(tmp_path / "j.sqlite")
    r = Runner(p, SimBroker(), j, RiskEngine(RiskLimits(budget=10**9)), day)
    o = Order(f"{day}:AAA:buy:1000", "AAA", "buy", 1, None, "entry")
    r.broker.last["AAA"] = 10000
    assert r._submit(o, T)
    r2 = Runner(p, SimBroker(), Journal(tmp_path / "j.sqlite"), RiskEngine(), day)   # 다시 시작
    assert not r2._submit(replace(o, status="NEW"), T)
    assert "duplicate_skip" in set(r2.journal.table("events").kind)
