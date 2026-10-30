"""10장 실습 4 — 모의투자 서버에서 매수·정정·취소, 시장가 매수·매도를 해 보고 주문체결(00) 이벤트를 본다.

실행: TRADING_MODE=paper 로 두고
  uv run python examples/chapter10/04_orders.py [종목]            지정가 매수 → 정정 → 취소 (체결 없음)
  uv run python examples/chapter10/04_orders.py [종목] --market   시장가 1주 매수 → 시장가 1주 매도
실계좌 키로는 실행되지 않는다(RealOrderBlocked). 장이 열린 영업일에만 주문이 받아진다.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import find_dotenv, load_dotenv

from vwap_trading.backtest.costs import tick_size
from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.broker.orders import OrderRejected, PaperBroker
from vwap_trading.live.feed import KiwoomSocket

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
TZ = ZoneInfo(os.getenv("MARKET_TZ", "Asia/Seoul"))
symbol = next((a for a in sys.argv[1:] if not a.startswith("--")), "005930")
market = "--market" in sys.argv
client = KiwoomClient.from_env("paper")
broker = PaperBroker(client)
t_sent: dict[str, float] = {}

FIDS = {"9203": "주문번호", "904": "원주문", "905": "주문구분", "913": "상태", "900": "주문수량", "901": "주문가",
        "902": "미체결", "903": "체결누계금액", "909": "체결번호", "910": "체결가", "911": "체결량", "908": "시각",
        "919": "거부사유"}


def on_event(kind, item, v):
    if kind == "00":
        no = v.get("9203", "").lstrip("0")
        dt = f"+{(time.monotonic() - t_sent[no]) * 1000:.0f}ms" if no in t_sent else ""
        print(f"    [00 {dt}] " + ", ".join(f"{name} {v[k]}" for k, name in FIDS.items() if v.get(k, "").strip()))


def send(label, fn, *a):
    t0 = time.monotonic()
    try:
        no = fn(*a)
    except OrderRejected as e:
        print(f"  {label}: 거부 {e.code} {e.msg}")
        return None
    t_sent[no.lstrip("0")] = t0
    print(f"  {label}: 주문번호 {no} (응답 {(time.monotonic() - t0) * 1000:.0f}ms)")
    return no


async def main():
    ws = KiwoomSocket(client, on_event, lambda k, d: print(f"  [소켓] {k} {d}"))
    await ws.subscribe("2", [""], ["00"])
    task = asyncio.create_task(ws.run())
    await asyncio.sleep(2)
    info, _ = client.post("/api/dostk/stkinfo", "ka10001", {"stk_cd": symbol})
    price = abs(float(info["cur_prc"]))
    print(f"{datetime.now(TZ):%H:%M:%S} {symbol} 현재가 {price:,.0f}원, 예수금 {broker.cash():,}원")
    if not market:
        low = (price * 0.95) // tick_size(price) * tick_size(price)        # 체결되지 않을 낮은 가격
        no = send(f"지정가 매수 1주 @{low:,.0f}", broker.buy, symbol, 1, low)
        await asyncio.sleep(2)
        if no:
            print("  미체결 조회:", broker.open_orders().to_dict("records"))
            low2 = low - tick_size(low)
            no2 = send(f"정정 → @{low2:,.0f}", broker.modify, no, symbol, 1, low2)
            await asyncio.sleep(2)
            send("취소 (남은 수량 전부)", broker.cancel, no2 or no, symbol, 0)
            await asyncio.sleep(2)
            print("  미체결 조회:", broker.open_orders().to_dict("records"))
    else:
        if send("시장가 매수 1주", broker.buy, symbol, 1, None):
            await asyncio.sleep(3)
            print("  잔고:", broker.holdings().to_dict("records"))
            send("시장가 매도 1주", broker.sell, symbol, 1, None)
            await asyncio.sleep(3)
            print("  당일 체결:", broker.fills().to_dict("records"))
    task.cancel()


asyncio.run(main())
