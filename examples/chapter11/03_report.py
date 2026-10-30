"""11장 실습 3 — 모의매매 하루 리포트: 체결 오차, 봉 대조, 같은 날 백테스트와의 판정 비교. 주문 없음.

실행: uv run python examples/chapter11/03_report.py [YYYYMMDD]          장 마감 후 (키움 분봉을 조회해 대조)
      uv run python examples/chapter11/03_report.py --journal 경로 --offline   대조용 분봉을 data/raw 에서 읽음
"""

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from vwap_trading.backtest.costs import CostModel, tick_size
from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.live.journal import Journal
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
ap = argparse.ArgumentParser()
ap.add_argument("day", nargs="?", default=f"{datetime.now(ZoneInfo(os.getenv('MARKET_TZ', 'Asia/Seoul'))):%Y%m%d}")
ap.add_argument("--journal")
ap.add_argument("--offline", action="store_true")
ap.add_argument("--config", default="configs/strategy_a.toml")
args = ap.parse_args()
j = Journal(args.journal or f"reports/live/{args.day}.sqlite")
ev, trades, bars, orders = (j.table(t) for t in ("events", "trades", "bars", "orders"))
day = pd.Timestamp(bars.time.min()).date() if len(bars) else pd.Timestamp(args.day).date()
p = load_params(args.config)
cnt = ev.kind.value_counts()

# 1. 하루 요약
sel_ev = ev[ev.kind == "selection"]
picked = json.loads(sel_ev.data.iloc[-1])["picked"] if len(sel_ev) else \
    [{"symbol": s, "tradable_from": json.loads(d)["tradable_from"]} for s, d in ev[ev.kind == "selected"][["symbol", "data"]].values]
print(f"[{day}] 선정 {[x['symbol'] for x in picked]}")
print(f"  신호 {cnt.get('signal', 0)} | AI 거부 {cnt.get('filtered', 0)} | 위험관리 거부 {cnt.get('risk_block', 0)} | "
      f"주문 {len(orders)} (거부 {cnt.get('order_rejected', 0)}, 응답 유실 {cnt.get('order_unknown', 0)}, "
      f"취소 {cnt.get('cancel_sent', 0)}) | 매매 {len(trades)} | 시세 재접속 {max(cnt.get('ws_connected', 0) - 2, 0)}")
sig = ev[ev.kind == "signal"].data.map(json.loads)
if len(sig) and "p_cal" in sig.iloc[0]:
    print(f"  AI 필터(기록만): 신호 {len(sig)}건 중 08장 승인 규칙 통과 {sum(s['rule_ok'] for s in sig)}건, "
          f"p_cal 평균 {np.mean([s['p_cal'] for s in sig]):.3f}, 예측 {np.median([s['ms'] for s in sig]):.0f}ms")

# 2. 체결 오차: 실제 체결가 vs 백테스트가 가정한 가격 (호가 단위)
cost = CostModel(slip_ticks=0)
if len(trades):
    t = trades.copy()
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c])
    bar_open = {(r.symbol, pd.Timestamp(r.time)): r.open for r in bars.itertuples()}
    t["entry_bar_open"] = [bar_open.get((r.symbol, r.entry_time)) for r in t.itertuples()]   # 진입 봉은 장 마감 뒤에는 확정돼 있다
    t["exit_bar_open"] = [bar_open.get((r.symbol, r.exit_time)) for r in t.itertuples()]
    t["buy_slip"] = (t.buy_avg - t.entry_bar_open) / t.entry_bar_open.map(tick_size)
    t["sell_ref"] = np.where(t.reason == "stop", t.stop, np.where(t.reason == "target", t.target, t.exit_bar_open))
    t["sell_slip"] = (t.sell_ref - t.sell_avg) / t.sell_ref.map(tick_size)
    t["net"] = [cost.net_return(r.buy_avg, r.sell_avg, r.exit_time.date()) for r in t.itertuples()]
    print("\n[체결 오차] 매수: 실제 평균가 − 진입 봉 시가 / 매도: 기준가(손절·목표가 또는 다음 봉 시가) − 실제 평균가, 호가 단위")
    num_cols = ["entry_bar_open", "buy_avg", "buy_slip", "sell_ref", "sell_avg", "sell_slip", "net"]
    print(t[["symbol", "entry_time", "reason", "qty"]].join(t[num_cols].astype(float).round(3)).to_string(index=False))
    print(f"  매수 평균 {t.buy_slip.mean():+.2f}호가, 매도 평균 {t.sell_slip.mean():+.2f}호가 (백테스트 가정: 각 +1호가) | "
          f"순수익(실제 체결가, 수수료·세금 차감) 평균 {t.net.mean():+.3%}")

# 3. 봉 대조: 실시간으로 만든 봉 vs 키움 분봉 (같은 시각 라벨)
symbols = sorted(bars.symbol.unique())
if args.offline:
    ref_bars = {s: pd.read_parquet(f"data/raw/{s}_1m.parquet") for s in symbols}
else:
    from vwap_trading.broker.kiwoom import KiwoomClient
    from vwap_trading.data.minute import fetch_minute_bars
    client = KiwoomClient.from_env("real")
    ref_bars = {s: fetch_minute_bars(client, s, max_pages=1) for s in symbols}
rows = []
for s in symbols:
    mine = bars[(bars.symbol == s) & (bars.how != "backfill")].assign(time=lambda d: pd.to_datetime(d.time)).set_index("time")
    k = ref_bars[s][ref_bars[s].index.date == day]
    for label, kk in (("같은 라벨", k), ("키움 라벨=구간 끝 가정", k.shift(-1, freq="1min"))):
        m = mine.merge(kk, left_index=True, right_index=True, suffixes=("", "_k"))
        rows.append({"종목": s, "비교": label, "봉": len(m),
                     "OHLC 일치": int(((m.open == m.open_k) & (m.high == m.high_k) & (m.low == m.low_k) & (m.close == m.close_k)).sum()),
                     "거래량 일치": int((m.volume == m.volume_k).sum()),
                     "종가 일치": int((m.close == m.close_k).sum())})
print("\n[봉 대조] 실시간 체결로 만든 봉 vs 키움 분봉")
print(pd.DataFrame(rows).to_string(index=False))
live_b = bars[bars.how != "backfill"].copy()
if len(live_b):
    delay = (pd.to_datetime(live_b.confirmed_at) - pd.to_datetime(live_b.time)).dt.total_seconds() - 60
    print(f"  확정 방식 {live_b.how.value_counts().to_dict()} | 봉이 끝난 뒤 확정까지 중앙 {delay.median():.2f}초, "
          f"95% {delay.quantile(0.95):.2f}초")

# 4. 같은 날 백테스트: 키움 분봉으로 05장 엔진을 돌려 판정(신호·진입·청산 시각·사유)을 비교
sel = pd.DataFrame(picked)[["symbol", "tradable_from"]].assign(tradable_from=lambda d: pd.to_datetime(d.tradable_from))
ind = {s: compute_indicators(ref_bars[s][ref_bars[s].index.date == day], p.vwap) for s in sel.symbol if s in ref_bars}
bt = backtest(ind, sel[sel.symbol.isin(ind)], p) if ind else pd.DataFrame()
keys = ["symbol", "signal_time", "entry_time", "exit_time", "reason"]
a = bt[keys].astype(str) if len(bt) else pd.DataFrame(columns=keys)
b = trades[keys].astype(str) if len(trades) else pd.DataFrame(columns=keys)
both = a.merge(b, on=keys)
print(f"\n[같은 날 백테스트] 백테스트 {len(a)}건, 모의매매 {len(b)}건, 판정 일치 {len(both)}건")
print("  백테스트에만:", a.merge(b, on=keys, how="left", indicator=True).query("_merge == 'left_only'")[keys].values.tolist())
print("  모의매매에만:", b.merge(a, on=keys, how="left", indicator=True).query("_merge == 'left_only'")[keys].values.tolist())
