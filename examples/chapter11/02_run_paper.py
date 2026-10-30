"""11장 실습 2 — 실시간 모의매매 실행.

실행: uv run python examples/chapter11/02_run_paper.py [--filter log|rule|off] [--config configs/strategy_a.toml]
  - 장 시작 전(08:50 무렵)에 켜 두면 09:00 부터 시세를 받아 봉을 만들고, 09:30 에 종목을 선정한다.
  - 장중에 다시 켜면: 미체결 취소 → 보유 복구 → 오늘 분봉을 조회해 채운 뒤 이어서 실행한다.
  - 중단: 작업 폴더에 STOP 파일을 만든다 (신규 진입 중단 + 보유 전량 시장가 매도).
    Ctrl+C 는 프로그램만 끈다(보유는 그대로). 다시 켜면 recover() 가 이어받는다.
시세는 실전 서버(조회 전용 키), 주문과 주문체결 이벤트는 모의투자 서버. 실계좌 주문은 코드에서 막는다.
기록: reports/live/YYYYMMDD.sqlite, reports/live/YYYYMMDD_ticks.csv
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.broker.orders import PaperBroker
from vwap_trading.data.minute import fetch_minute_bars
from vwap_trading.live.bars import parse_0b, to_frame
from vwap_trading.live.feed import KiwoomSocket
from vwap_trading.live.filter import LiveFilter
from vwap_trading.live.journal import Journal
from vwap_trading.live.risk import RiskEngine, RiskLimits
from vwap_trading.live.runner import Runner
from vwap_trading.live.select import history_base, select_live
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
ap = argparse.ArgumentParser()
ap.add_argument("--filter", default="log", choices=["log", "rule", "off"])
ap.add_argument("--config", default="configs/strategy_a.toml")
ap.add_argument("--universe", default="data/sample/universe_20261030.csv")
ap.add_argument("--max-minutes", type=float, default=None, help="이 시간(분)이 지나면 종료 (점검용)")
args = ap.parse_args()

TZ = ZoneInfo(os.getenv("MARKET_TZ", "Asia/Seoul"))


def now() -> datetime:
    return datetime.now(TZ).replace(tzinfo=None)


p = load_params(args.config)
day = now().date()
universe = pd.read_csv(args.universe, dtype={"symbol": str}).symbol.tolist()
paper = KiwoomClient.from_env("paper")
real = KiwoomClient.from_env("real")
broker = PaperBroker(paper)                          # 모의투자가 아니면 여기서 RealOrderBlocked
journal = Journal(f"reports/live/{day:%Y%m%d}.sqlite")
flt = None if args.filter == "off" else LiveFilter.load("models_out/live_filter.joblib", mode=args.filter)
runner = Runner(p, broker, journal, RiskEngine(RiskLimits()), day, approve=flt)
# 선정 기준값: 02장에서 모은 분봉(data/raw)에, 그 뒤 거래일의 분봉을 조회해 덧붙인다 (data/raw 는 고치지 않는다)
hist = {}
for s in universe:
    recent = fetch_minute_bars(real, s, max_pages=3)
    old = pd.read_parquet(f"data/raw/{s}_1m.parquet") if Path(f"data/raw/{s}_1m.parquet").exists() else recent
    hist[s] = pd.concat([old, recent]).loc[lambda d: ~d.index.duplicated(keep="last")].sort_index()
base = history_base(hist, day)
SELECT_AT = runner.open_t + timedelta(minutes=30)
journal.event(now(), "start", filter=args.filter, config=args.config, universe=len(universe), base=len(base),
              model=getattr(flt, "trained_until", None))
print(f"{now():%H:%M:%S} 시작: 종목 {len(universe)}개, 선정 기준값 {len(base)}종목, 필터 {args.filter}")

# 재시작: 미체결 취소·보유 복구, 장중이면 오늘 분봉을 채운다
runner.recover(now())
if now() > runner.open_t + timedelta(minutes=1):
    cut = now().replace(second=0, microsecond=0)
    for s in universe:
        b = fetch_minute_bars(real, s, max_pages=1)
        runner.load_bars(s, b[(b.index.date == day) & (b.index >= runner.open_t) & (b.index < cut)], now())
    journal.event(now(), "backfill", bars=sum(len(v) for v in runner.bars.values()))
    print(f"{now():%H:%M:%S} 오늘 분봉 채움: {sum(len(v) for v in runner.bars.values())}봉")

tick_file = open(f"reports/live/{day:%Y%m%d}_ticks.csv", "a", newline="", encoding="utf-8")
ticks = csv.writer(tick_file)
lat: list[float] = []                                # 체결 시각 → 수신 시각 지연 (초)


def on_quote(kind, item, v):
    if kind != "0B":
        return
    t = now()
    ts, price, qty = parse_0b(v, day)
    lat.append((t - ts).total_seconds())
    ticks.writerow([t.isoformat(timespec="milliseconds"), item, v.get("20"), price, v.get("15"), v.get("27"),
                    v.get("28"), v.get("290")])
    runner.on_tick(item, ts, price, qty, t)


def on_order(kind, item, v):
    if kind == "00":
        runner.on_order_event(v, now())


def status(name):
    return lambda kind, detail: (journal.event(now(), f"ws_{kind}", feed=name, detail=detail),
                                 print(f"{now():%H:%M:%S} [{name}] {kind} {detail}"))


async def clock(stop: asyncio.Event):
    started = now()
    selected = any(s.tradable_from for s in runner.syms.values())
    while not stop.is_set():
        t = now()
        runner.on_timer(t)
        tick_file.flush()
        if not selected and t >= SELECT_AT + timedelta(seconds=3):
            today = {s: to_frame(b) for s, b in runner.bars.items()}
            sel = select_live(today, base, SELECT_AT)
            for r in sel.itertuples():
                runner.select(r.symbol, SELECT_AT, t)
            journal.event(t, "selection", picked=sel.to_dict("records"))
            print(f"{t:%H:%M:%S} 선정: {sel[['symbol', 'ratio']].to_dict('records')}")
            selected = True
        if t.second == 0:
            n_bars = sum(len(b) for b in runner.bars.values())
            med = sorted(lat)[len(lat) // 2] if lat else float("nan")
            print(f"{t:%H:%M} 봉 {n_bars} | 체결 {len(lat)} 지연 중앙 {med:.2f}s | "
                  f"보유 {[s.symbol for s in runner.syms.values() if s.state != 'FLAT']} | "
                  f"손익 {runner.pnl_today():+,.0f}원 | 중단 {runner.risk.halted}")
            journal.event(t, "minute", ticks=len(lat), lat_med=med, pnl=runner.pnl_today())
            lat.clear()
        if t.time() >= time(15, 31) or (args.max_minutes and t - started > timedelta(minutes=args.max_minutes)):
            stop.set()
        await asyncio.sleep(1 - (now().microsecond / 1e6))


async def main():
    quotes = KiwoomSocket(real, on_quote, status("시세"))
    orders = KiwoomSocket(paper, on_order, status("주문"))
    await quotes.subscribe("1", universe, ["0B"])
    await orders.subscribe("2", [""], ["00"])
    stop = asyncio.Event()
    tasks = [asyncio.create_task(quotes.run()), asyncio.create_task(orders.run())]
    try:
        await clock(stop)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Ctrl+C: 프로그램만 종료합니다 (보유는 그대로)")
finally:
    journal.event(now(), "stop", trades=len(runner.trades), realized=runner.realized,
                  late_ticks=dict(runner.builder.late))
    tick_file.close()
    print(f"{now():%H:%M:%S} 종료: 매매 {len(runner.trades)}건, 실현 {runner.realized:+,.0f}원 → {journal.path}")
