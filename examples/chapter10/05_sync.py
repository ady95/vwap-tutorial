"""10장 실습 5 — 계좌(키움 모의투자)와 거래 로그(SQLite)의 상태를 대조한다. 주문 없음.

실행: uv run python examples/chapter10/05_sync.py [YYYYMMDD]   (기본 오늘)
프로그램이 기억하는 보유·미체결과 계좌의 실제 보유·미체결이 다르면, 그 종목은 프로그램이 모르는
위험을 안고 있는 것이다. 장중 재시작(11-4)과 장 마감 후 점검에 쓴다.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.broker.orders import PaperBroker
from vwap_trading.live.journal import Journal

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
day = sys.argv[1] if len(sys.argv) > 1 else f"{datetime.now(ZoneInfo(os.getenv('MARKET_TZ', 'Asia/Seoul'))):%Y%m%d}"
broker = PaperBroker(KiwoomClient.from_env("paper"), read_only=True)   # 조회만 한다 (주문 메서드는 막힘)

hold = broker.holdings()
opens = broker.open_orders()
fills = broker.fills()
print(f"[계좌] 예수금 {broker.cash():,}원 | 보유 {len(hold)}종목 | 미체결 {len(opens)}건 | 당일 체결 주문 {len(fills)}건")
for r in hold.itertuples():
    print(f"  보유 {r.symbol} {r.qty}주 평균 {r.avg_price:,.0f}원")

path = Path(f"reports/live/{day}.sqlite")
if not path.exists():
    print(f"[로그] {path} 없음 — 이 날 모의매매를 실행하지 않았습니다")
    sys.exit()
j = Journal(path)
state = j.load_state()
orders = j.table("orders")
open_log = orders[orders.status.isin(["NEW", "SENT", "PARTIAL", "UNKNOWN"])]
print(f"[로그] 보유 {len(state)}종목 | 열린 주문 {len(open_log)}건 | 전체 주문 {len(orders)}건")

problems = []
acct = dict(zip(hold.symbol, hold.qty))
for s in sorted(set(acct) | set(state)):
    a, b = acct.get(s, 0), state.get(s, {}).get("qty", 0)
    if a != b:
        problems.append(f"보유 수량 불일치 {s}: 계좌 {a}주, 로그 {b}주")
log_nos = set(open_log.ord_no.dropna().str.lstrip("0"))
for r in opens.itertuples():
    if r.ord_no.lstrip("0") not in log_nos:
        problems.append(f"로그에 없는 미체결 {r.symbol} 주문번호 {r.ord_no} ({r.open_qty}주)")
acct_nos = set(opens.ord_no.str.lstrip("0"))
for r in open_log.itertuples():
    if r.ord_no is None or r.ord_no.lstrip("0") not in acct_nos:
        problems.append(f"로그는 열려 있는데 계좌에는 없는 주문 {r.symbol} {r.key} ({r.status})")
filled = fills.groupby("ord_no").filled.sum().rename(lambda s: s.lstrip("0"))
for r in orders[orders.ord_no.notna()].itertuples():
    f = int(filled.get(r.ord_no.lstrip("0"), 0))
    if f != (r.filled or 0):
        problems.append(f"체결 수량 불일치 {r.symbol} {r.key}: 계좌 {f}주, 로그 {r.filled}주")
print("\n[대조] " + ("OK — 계좌와 로그가 같습니다" if not problems else f"{len(problems)}건"))
for p in problems:
    print("  " + p)
