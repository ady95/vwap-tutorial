"""03장 실습 2 — 세션 VWAP 과 롤링 VWAP 의 반응 속도·교차 횟수 비교.

실행: uv run python examples/chapter03/02_rolling_vs_session.py --universe data/sample/universe_20261030.csv
"""

import argparse
import sys

import pandas as pd

from vwap_trading.indicators.vwap import VwapParams, compute_indicators

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--universe", required=True)
parser.add_argument("--N", type=int, default=30)
args = parser.parse_args()

manifest = pd.read_csv("data/raw/_manifest.csv", dtype={"symbol": str})
symbols = manifest.loc[manifest.days >= 250, "symbol"]          # 13개월치가 있는 종목만
rows = []
for s in symbols:
    x = compute_indicators(pd.read_parquet(f"data/raw/{s}_1m.parquet"), VwapParams(N=args.N))
    x = x.dropna(subset=["rvwap", "atr"])
    hhmm = x.index.hour * 100 + x.index.minute
    for name, line in (("세션", x.svwap), ("롤링", x.rvwap)):
        move = (line.groupby(x.index.date).diff().abs() / x.atr_s)          # 봉당 기준선 이동 (ATR* 배수)
        side = (x.close > line).astype(int)
        cross = side.groupby(x.index.date).diff().abs().groupby(x.index.date).sum()
        rows.append({"종목": s, "기준선": name,
                     "오전 이동": move[(hhmm >= 1000) & (hhmm < 1100)].mean(),
                     "오후 이동": move[(hhmm >= 1400) & (hhmm < 1445)].mean(),
                     "하루 교차": cross.mean()})

res = pd.DataFrame(rows).groupby("기준선")[["오전 이동", "오후 이동", "하루 교차"]].mean()
print(f"{len(symbols)}종목 평균 (이동 = 봉당 기준선 변화 / ATR, 교차 = 종가가 기준선을 넘나든 횟수)")
print(res.round(3).to_string())
