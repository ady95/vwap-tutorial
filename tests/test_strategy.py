"""04장 전략 규칙 테스트 — 진입 가능 시간, 청산 우선순위, 재진입 대기, 리페인팅 방지."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vwap_trading.indicators.vwap import FLAT, UP, compute_indicators
from vwap_trading.strategy.rules import StrategyParams, load_params, raw_signal, run_day

S0 = StrategyParams(strategy="S0")


def frame(n=390, start="2026-10-29 09:00"):
    """지표가 이미 계산된 하루치 가짜 프레임. 가격 100, ATR* 1, 세션 VWAP 100 으로 평평하다."""
    idx = pd.date_range(start, periods=n, freq="1min", name="time")
    x = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=idx)
    x["bar_no"] = np.arange(1, n + 1)
    x["svwap"] = x["mvwap"] = 100.0
    x["atr_s"] = 1.0
    x["atr_low"] = False
    x["rvol"] = 2.0
    x["state"] = float(UP)
    x["slope"] = 0.0
    return x


def cross(x, at):
    """S0 상향 돌파를 봉 at 에 만든다 (종가 100 → 101)."""
    x.iloc[at, x.columns.get_loc("close")] = 101.0
    x.iloc[at, x.columns.get_loc("high")] = 101.0


def test_no_signal_before_warmup_tradable_from_or_after_cutoff():
    x = frame()
    for at in (40, 60, 350):                     # 09:40 워밍업 전, 10:00 tradable_from 전, 14:50 마감 후
        cross(x, at)
    tf = pd.Timestamp("2026-10-29 10:30")
    assert raw_signal(x, S0).iloc[[40, 60, 350]].all()
    assert run_day(x, S0, tf) == []


def test_signal_uses_closed_bar_and_enters_next_open():
    x = frame()
    cross(x, 100)
    x.iloc[101, x.columns.get_loc("open")] = 100.5
    tr = run_day(x, S0)[0]
    assert tr["signal_time"] == x.index[100]
    assert tr["entry_time"] == x.index[101] and tr["entry"] == 100.5


def test_stop_wins_when_stop_and_target_hit_in_same_bar():
    x = frame()
    cross(x, 100)
    x.iloc[105, x.columns.get_loc("high")] = 103.0      # 목표가 102 도달
    x.iloc[105, x.columns.get_loc("low")] = 98.0        # 손절가 98.5 도달
    tr = run_day(x, S0)[0]
    assert tr["reason"] == "stop" and tr["exit"] == 98.5


def test_gap_below_stop_fills_at_open():
    x = frame()
    cross(x, 100)
    x.iloc[105, [x.columns.get_loc(k) for k in ("open", "low")]] = [97.0, 96.0]
    tr = run_day(x, S0)[0]
    assert tr["reason"] == "stop" and tr["exit"] == 97.0


def test_base_break_fills_next_open_and_only_in_compare_2():
    x = frame()
    cross(x, 100)
    x.iloc[110, x.columns.get_loc("close")] = 99.7       # 세션 VWAP 100 − 0.2 아래
    x.iloc[110, x.columns.get_loc("low")] = 99.7
    x.iloc[111, x.columns.get_loc("open")] = 99.9
    tr = run_day(x, S0)[0]
    assert tr["reason"] == "base" and tr["exit_time"] == x.index[111] and tr["exit"] == 99.9
    assert run_day(x, replace(S0, compare=1))[0]["reason"] == "time"


def test_time_exit_and_forced_close():
    x = frame()
    cross(x, 100)                                         # 진입 101번째 봉 → 30봉 보유 후 청산
    tr = run_day(x, S0)[0]
    assert tr["reason"] == "time" and tr["bars"] == 30 and tr["exit_time"] == x.index[131]
    y = frame()
    cross(y, 344)                                         # 14:44 신호 → 15:14 확정 시 장 마감 청산
    tr = run_day(y, S0)[0]
    assert tr["reason"] == "close" and tr["exit_time"] == pd.Timestamp("2026-10-29 15:15")


def test_cooldown_blocks_reentry():
    x = frame()
    cross(x, 100)
    x.iloc[102, x.columns.get_loc("low")] = 98.0          # 102번째 봉에서 손절
    cross(x, 105)                                         # 대기 중 신호 → 무시
    cross(x, 112)                                         # 대기(102~111) 이후 신호 → 진입
    trades = run_day(x, S0)
    assert [t["signal_time"] for t in trades] == [x.index[100], x.index[112]]


def test_strategy_a_needs_flat_before_and_box_breakout():
    x = frame()
    x.iloc[90:100, x.columns.get_loc("state")] = float(FLAT)
    x.iloc[100, [x.columns.get_loc(k) for k in ("slope", "close", "mvwap")]] = [0.06, 100.4, 100.4]
    a = StrategyParams(strategy="A")
    assert raw_signal(x, a).iloc[100]                     # 100.4 > 박스 100 + 0.3
    x.iloc[100, x.columns.get_loc("close")] = 100.2
    assert not raw_signal(x, a).iloc[100]


def test_signals_do_not_repaint():
    """봉 t 까지만 넣고 계산한 신호와 하루 전체로 계산한 신호가 같아야 한다."""
    rng = np.random.default_rng(0)
    n = 390
    close = 10000 + np.cumsum(rng.normal(0, 8, n)).round()
    idx = pd.date_range("2026-10-29 09:00", periods=n, freq="1min", name="time")
    bars = pd.DataFrame({"open": close, "high": close + 10, "low": close - 10, "close": close,
                         "volume": rng.integers(100, 5000, n).astype(float)}, index=idx)
    for strategy in ("S0", "A"):
        p = StrategyParams(strategy=strategy, rho=0.0)
        full = raw_signal(compute_indicators(bars, p.vwap), p)
        for t in range(50, n, 7):
            part = raw_signal(compute_indicators(bars.iloc[: t + 1], p.vwap), p)
            assert part.iloc[-1] == full.iloc[t]


def test_load_params(tmp_path):
    f = tmp_path / "a.toml"
    f.write_text('strategy = "A"\nrho = 2.0\n[vwap]\nN = 20\n', encoding="utf-8")
    p = load_params(f)
    assert p.rho == 2.0 and p.vwap.N == 20 and p.vwap.warmup_end == 34
    f.write_text("rhoo = 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_params(f)
