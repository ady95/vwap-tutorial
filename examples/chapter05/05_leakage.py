"""05장 실습 5 — 미래 정보 사용·데이터 누수 검사.

1. 매매 기록의 시각 불변식 (선정 전 신호, 신호 봉 이전 진입 등)
2. 미래 봉 교란 검사: 신호 봉 이후 봉을 엉뚱한 값으로 바꿔도 신호·손절가·목표가가 같은가
3. 선정 입력 재현: 선정 시각 이후 봉을 모두 지우고 다시 골라도 같은 종목이 나오는가
4. 수정주가: 과거 가격에 들어 있는 미래 정보

실행: uv run python examples/chapter05/05_leakage.py   (약 2분)
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vwap_trading.backtest.costs import tick_size
from vwap_trading.backtest.engine import backtest
from vwap_trading.data.selection import make_selections
from vwap_trading.data.session import session_of
from vwap_trading.indicators.vwap import compute_indicators, load_indicators
from vwap_trading.strategy.rules import StrategyParams, day_clock, entry_allowed, raw_signal

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["selected_at", "tradable_from"])
tf = sel.set_index([sel.symbol, sel.tradable_from.dt.date]).tradable_from
ind = load_indicators(sel.symbol)
p = StrategyParams()
trades = {s: backtest(ind, sel, StrategyParams(strategy=s)) for s in ("S0", "A")}

# 1. 시각 불변식
for s, bt in trades.items():
    tfrom = pd.Series([tf[(r.symbol, r.signal_time.date())] for r in bt.itertuples()], index=bt.index)
    cut = pd.Series([day_clock(t.date(), p)[0] for t in bt.signal_time], index=bt.index)
    checks = {
        "선정 전 신호": int((bt.signal_time < tfrom).sum()),
        "신호 봉 이전·같은 봉 진입": int((bt.entry_time <= bt.signal_time).sum()),
        "진입 마감 이후 신호": int((bt.signal_time.dt.time > cut).sum()),
        "진입 전 청산": int((bt.exit_time < bt.entry_time).sum()),
        "날짜를 넘긴 보유": int((bt.exit_time.dt.date != bt.entry_time.dt.date).sum()),
    }
    print(f"[1] {s} {len(bt)}건 시각 불변식 위반: {checks}")

# 2. 미래 봉 교란 검사 (전략 A 매매 전체)
rng = np.random.default_rng(0)
ohlc = ["open", "high", "low", "close"]
bt, changed = trades["A"], 0
for tr in bt.itertuples():
    bars = pd.read_parquet(f"data/raw/{tr.symbol}_1m.parquet").loc[str(tr.signal_time.date())].astype(float)
    future = bars.index > tr.signal_time
    k = int(future.sum())
    bars.loc[future, ohlc] = bars.loc[future, ohlc].mul(rng.uniform(0.5, 1.5, k), axis=0)
    bars.loc[future, "high"] = bars.loc[future, ohlc].max(axis=1)
    bars.loc[future, "low"] = bars.loc[future, ohlc].min(axis=1)
    bars.loc[future, "volume"] = rng.integers(1, 10**6, k)
    x = compute_indicators(bars, p.vwap)
    ok = (raw_signal(x, p) & entry_allowed(x, p, tf[(tr.symbol, tr.signal_time.date())]))[tr.signal_time]
    risk = x.atr_s[tr.signal_time]
    same = np.isclose(tr.stop, tr.entry - p.lam * risk) and np.isclose(tr.target, tr.entry + p.tau * risk)
    changed += int(not (ok and same))
print(f"\n[2] 미래 봉 교란: A 매매 {len(bt)}건 중 신호·손절가·목표가가 달라진 매매 {changed}건")

# 3. 선정 입력 재현: 매일 선정 시각 이후 봉을 지우고 다시 고른다
uni = pd.read_csv("data/sample/universe_20261030.csv", dtype={"symbol": str}, encoding="utf-8-sig").symbol
cut_bars = {}
for s in uni:
    try:
        b = pd.read_parquet(f"data/raw/{s}_1m.parquet")
    except FileNotFoundError:
        continue
    sel_at = [datetime.combine(d, session_of(d)[0]) + timedelta(minutes=30) for d in b.index.date]
    cut_bars[s] = b[b.index < pd.DatetimeIndex(sel_at)]
again = make_selections(cut_bars)
same = sel[["symbol", "selected_at"]].reset_index(drop=True).equals(again[["symbol", "selected_at"]].reset_index(drop=True))
print(f"\n[3] 선정 시각 이후 봉 삭제 후 재선정: {len(again)}건 (원래 {len(sel)}건), 완전 일치 {same}")

# 4. 수정주가: 종가의 절반 이상이 호가단위에 맞지 않는 날 = 권리 변동 전 수정된 가격
print("\n[4] 수정주가 구간 (종가의 50% 이상이 호가단위에 맞지 않는 날)")
adjusted = {}
for s in dict.fromkeys(sel.symbol):
    b = pd.read_parquet(f"data/raw/{s}_1m.parquet")
    px = b.close.to_numpy()
    off = pd.Series(px % np.vectorize(tick_size)(px) != 0, index=b.index).groupby(b.index.date).mean()
    days = off[off >= 0.5].index
    if len(days):
        adjusted[s] = set(days)
        print(f"  {s}: {len(days)}일, {min(days)} ~ {max(days)}")
for s, bt in trades.items():
    hit = sum(t.entry_time.date() in adjusted.get(t.symbol, ()) for t in bt.itertuples())
    print(f"  {s}: 수정주가 구간 매매 {hit}/{len(bt)}건")
