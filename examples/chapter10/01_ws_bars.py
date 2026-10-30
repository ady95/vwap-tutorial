"""10장 실습 1 — 실시간 체결을 받아 1분봉을 만들고, 키움 분봉(ka10080)과 대조한다. 주문 없음.

실행: uv run python examples/chapter10/01_ws_bars.py [분] [종목 ...]   (기본 10분, 005930 000660 035420)
장중에 실행한다. 장이 열리지 않은 날은 LOGIN·PING 만 오고 체결은 오지 않는다.
"""

import asyncio
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.data.minute import fetch_minute_bars
from vwap_trading.live.bars import MinuteBarBuilder, parse_0b, to_frame
from vwap_trading.live.feed import KiwoomSocket

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
TZ = ZoneInfo(os.getenv("MARKET_TZ", "Asia/Seoul"))
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10
symbols = sys.argv[2:] or ["005930", "000660", "035420"]
client = KiwoomClient.from_env("real")                # 시세는 실전 서버 (조회 전용)
builder = MinuteBarBuilder(grace=2.0)
bars = {s: [] for s in symbols}
delays = []


def now():
    return datetime.now(TZ).replace(tzinfo=None)


def on_real(kind, item, v):
    if kind != "0B":
        return
    t = now()
    ts, price, qty = parse_0b(v, t.date())
    delays.append((t - ts).total_seconds())
    for b in builder.on_tick(item, ts, price, qty, t):
        bars[item].append(b)
        print(f"{t:%H:%M:%S.%f}"[:-3], item, f"{b.time:%H:%M} 봉 확정({b.how}, 끝난 뒤 "
              f"{(t - b.time).total_seconds() - 60:.2f}초) O{b.open:,.0f} H{b.high:,.0f} L{b.low:,.0f} "
              f"C{b.close:,.0f} V{b.volume:,} 체결 {b.ticks}건")


async def main():
    ws = KiwoomSocket(client, on_real, lambda k, d: print(f"{now():%H:%M:%S} [{k}] {d}"))
    await ws.subscribe("1", symbols, ["0B"])
    task = asyncio.create_task(ws.run())
    t0 = now()
    while (now() - t0).total_seconds() < minutes * 60:
        for s, b in builder.due(now()):
            bars[s].append(b)
            print(f"{now():%H:%M:%S}", s, f"{b.time:%H:%M} 봉 확정(timer)")
        await asyncio.sleep(0.5)
    task.cancel()


asyncio.run(main())
if delays:
    d = pd.Series(delays)
    print(f"\n체결 {len(d):,}건, 체결 시각(초 단위) → 수신 지연: 중앙 {d.median():.2f}초, 95% {d.quantile(0.95):.2f}초, "
          f"최대 {d.max():.2f}초, 늦게 와서 버린 체결 {dict(builder.late)}")

# 키움 분봉과 대조: 같은 시각 라벨의 봉끼리 OHLCV 를 비교한다
print("\n[대조] 실시간으로 만든 봉 vs 키움 분봉 (ka10080, 같은 시각 라벨)")
for s in symbols:
    mine = to_frame(bars[s])
    if mine.empty:
        print(f"  {s}: 확정된 봉 없음")
        continue
    k = fetch_minute_bars(client, s, max_pages=1)
    m = mine.merge(k, left_index=True, right_index=True, suffixes=("", "_k"))
    shifted = mine.merge(k.shift(-1, freq="1min"), left_index=True, right_index=True, suffixes=("", "_k"))   # 키움 라벨이 구간 끝이라고 가정
    same = lambda df: int(((df.open == df.open_k) & (df.high == df.high_k) & (df.low == df.low_k)
                           & (df.close == df.close_k) & (df.volume == df.volume_k)).sum())
    print(f"  {s}: 내 봉 {len(mine)}개 | 같은 라벨 OHLCV 완전 일치 {same(m)}/{len(m)}, 종가 일치 {(m.close == m.close_k).sum()}"
          f", 거래량 차이 중앙 {(m.volume - m.volume_k).abs().median():.0f}주 | 키움 라벨=구간 끝 가정 시 완전 일치 {same(shifted)}/{len(shifted)}")
