"""10장 실습 2 — 실시간 실행기를 과거 분봉으로 재생해 백테스트 엔진과 판정이 같은지 확인한다.

실행: uv run python examples/chapter10/02_replay_parity.py [시작달] [끝달]   (기본 2026-10 한 달)
주문은 내지 않는다(SimBroker). 결과: reports/ch10/parity.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd

from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.live.journal import Journal
from vwap_trading.live.replay import SimBroker, replay
from vwap_trading.live.risk import RiskEngine, RiskLimits
from vwap_trading.live.runner import Runner
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
start, end = (sys.argv[1:3] + ["2026-10", "2026-10"])[:2] if len(sys.argv) > 1 else ("2026-10", "2026-10")
p = load_params("configs/strategy_a.toml")
sel = pd.read_csv("data/sample/selections.csv", parse_dates=["selected_at", "tradable_from"],
                  dtype={"symbol": str})
sel = sel[sel.tradable_from.dt.to_period("M").astype(str).between(start, end)]
raw = {s: pd.read_parquet(f"data/raw/{s}_1m.parquet") for s in sel.symbol.unique()}
NO_LIMIT = RiskLimits(budget=10**12, max_positions=99, daily_loss=10**15, max_entries=999)
KEYS = ["symbol", "signal_time", "entry_time", "exit_time", "reason"]

t0 = time.time()
bt_all, live_all = [], []
for day, g in sel.groupby(sel.tradable_from.dt.date):
    bars = {s: raw[s][raw[s].index.date == day] for s in g.symbol}
    ind = {s: compute_indicators(b, p.vwap) for s, b in bars.items()}
    bt_all.append(backtest(ind, g, p))
    r = Runner(p, SimBroker(), Journal(":memory:"), RiskEngine(NO_LIMIT), day)   # 로그는 메모리에만
    replay(r, r.broker, bars, day, dict(zip(g.symbol, g.tradable_from.dt.to_pydatetime())),
           select_at=g.tradable_from.min().to_pydatetime())
    live_all.append(pd.DataFrame(r.trades))

bt = pd.concat(bt_all, ignore_index=True)
live = pd.concat([d for d in live_all if len(d)], ignore_index=True)
for d in (bt, live):
    for c in ("signal_time", "entry_time", "exit_time"):
        d[c] = pd.to_datetime(d[c]).astype("datetime64[us]")
m = bt[KEYS + ["entry", "exit"]].merge(live[KEYS + ["entry", "exit"]], on=KEYS, how="outer",
                                         suffixes=("_bt", "_live"), indicator=True)
both = m[m._merge == "both"]
print(f"기간 {start}~{end}, 선정 {len(sel)}종목-일, {time.time() - t0:.0f}초")
print(f"백테스트 매매 {len(bt)}건, 실시간 실행기(재생) 매매 {len(live)}건")
print(f"판정 일치 (종목·신호·진입·청산 시각·사유): {len(both)}건, 백테스트에만 {(m._merge == 'left_only').sum()}건, "
      f"재생에만 {(m._merge == 'right_only').sum()}건")
print(f"진입가 일치: {(both.entry_bt == both.entry_live).sum()}/{len(both)}")
for reason, g in both.groupby("reason"):
    same = (g.exit_bt == g.exit_live).sum()
    print(f"  청산 {reason:<6} {len(g):3d}건  청산가 같음 {same:3d}  재생−백테스트 평균 {((g.exit_live / g.exit_bt) - 1).mean():+.4%}")
Path("reports/ch10").mkdir(parents=True, exist_ok=True)
m.to_csv("reports/ch10/parity.csv", index=False)
print("저장: reports/ch10/parity.csv")
