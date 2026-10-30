"""02장 실습 4 — 수집한 분봉으로 예제 종목 선정 입력(selections.csv)을 만든다.

실행: uv run python examples/chapter02/04_make_selections.py --universe data/sample/universe_20261030.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from vwap_trading.data.selection import make_selections

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--universe", required=True)
parser.add_argument("--top", type=int, default=3, help="하루 최대 선정 종목 수")
parser.add_argument("--min-ratio", type=float, default=1.5, help="최소 상대 거래대금")
args = parser.parse_args()

symbols = pd.read_csv(args.universe, dtype={"symbol": str})["symbol"]
bars = {s: pd.read_parquet(f"data/raw/{s}_1m.parquet") for s in symbols if Path(f"data/raw/{s}_1m.parquet").exists()}
print(f"분봉 파일 {len(bars)}/{len(symbols)}종목")

sel = make_selections(bars, top_n=args.top, min_ratio=args.min_ratio)
out = Path("data/sample/selections.csv")
sel[["symbol", "selected_at", "tradable_from"]].to_csv(out, index=False)

days = sel.selected_at.dt.date.nunique()
print(sel.head(6).to_string(index=False))
print(f"\n저장: {out} | {len(sel)}건, {days}거래일, 하루 평균 {len(sel) / days:.2f}종목")
print("종목별 선정 횟수 상위 5:")
print(sel.symbol.value_counts().head(5).to_string())
