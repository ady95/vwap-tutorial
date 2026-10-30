"""02장 실습 2 — 연속조회로 과거 1분봉을 모아 Parquet 으로 저장한다.

실행: uv run python examples/chapter02/02_collect_minute_bars.py 005930 --pages 50
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.data.minute import fetch_minute_bars

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))

parser = argparse.ArgumentParser()
parser.add_argument("symbol")
parser.add_argument("--pages", type=int, default=50, help="최대 페이지 수 (페이지당 약 900봉)")
args = parser.parse_args()


def on_page(page, rows):
    if rows:
        print(f"  page {page:3d}: {len(rows)}봉, 가장 오래된 봉 {rows[-1]['cntr_tm']}", flush=True)


client = KiwoomClient.from_env("real")   # 시세 조회는 실전 서버 (주문은 하지 않는다)
bars = fetch_minute_bars(client, args.symbol, max_pages=args.pages, on_page=on_page)
client.revoke()

out = Path(os.getenv("DATA_DIR", "./data")) / "raw" / f"{args.symbol}_1m.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
bars.to_parquet(out)

days = bars.index.normalize().nunique()
print(f"\n저장: {out}")
print(f"기간: {bars.index.min()} ~ {bars.index.max()} ({days}거래일, {len(bars)}봉)")
print("일별 봉 수(최근 5일):")
print(bars.groupby(bars.index.date).size().tail(5).to_string())
