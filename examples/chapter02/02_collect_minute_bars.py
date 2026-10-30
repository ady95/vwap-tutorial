"""02장 실습 2 — 연속조회로 과거 1분봉을 모아 Parquet 으로 저장한다.

실행:
  uv run python examples/chapter02/02_collect_minute_bars.py 005930 --pages 50
  uv run python examples/chapter02/02_collect_minute_bars.py --universe data/sample/universe_20261030.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.data.minute import fetch_minute_bars

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))

parser = argparse.ArgumentParser()
parser.add_argument("symbols", nargs="*", help="종목코드 (여러 개 가능)")
parser.add_argument("--universe", help="symbol 열이 있는 CSV")
parser.add_argument("--pages", type=int, default=400, help="종목당 최대 페이지 수 (페이지당 약 900봉)")
parser.add_argument("--force", action="store_true", help="이미 받은 종목도 다시 받기")
args = parser.parse_args()

symbols = list(args.symbols)
if args.universe:
    symbols += pd.read_csv(args.universe, dtype={"symbol": str})["symbol"].tolist()

raw_dir = Path(os.getenv("DATA_DIR", "./data")) / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
client = KiwoomClient.from_env("real")   # 시세 조회는 실전 서버 (주문은 하지 않는다)


def on_page(page, rows):
    if rows and page % 25 == 0:
        print(f"    page {page:3d}: 가장 오래된 봉 {rows[-1]['cntr_tm']}", flush=True)


for i, symbol in enumerate(symbols, 1):
    out = raw_dir / f"{symbol}_1m.parquet"
    if out.exists() and not args.force:
        print(f"[{i}/{len(symbols)}] {symbol}: 이미 있음, 건너뜀")
        continue
    print(f"[{i}/{len(symbols)}] {symbol} 수집 중", flush=True)
    bars = fetch_minute_bars(client, symbol, max_pages=args.pages, on_page=on_page)
    if not bars.attrs["complete"] or bars.empty:
        print(f"  실패: 끝까지 받지 못해 저장하지 않습니다 ({len(bars)}봉). 다시 실행하면 이어서 시도합니다")
        continue
    bars.to_parquet(out)
    days = bars.index.normalize().nunique()
    print(f"  저장: {out} | {bars.index.min()} ~ {bars.index.max()} ({days}거래일, {len(bars)}봉)", flush=True)
    # 수집 기록: 언제 어떤 범위를 받았는지 남겨 두면 나중에 같은 데이터를 다시 만들 수 있다
    manifest = raw_dir / "_manifest.csv"
    pd.DataFrame([{"symbol": symbol, "collected_at": pd.Timestamp.now().floor("s"), "first": bars.index.min(),
                   "last": bars.index.max(), "days": days, "bars": len(bars)}]).to_csv(
        manifest, mode="a", header=not manifest.exists(), index=False)

client.revoke()
