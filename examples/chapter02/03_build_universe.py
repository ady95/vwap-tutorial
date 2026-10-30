"""02장 실습 3 — 거래대금 상위 보통주로 예제 유니버스를 만든다.

실행: uv run python examples/chapter02/03_build_universe.py --n 15
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient
from vwap_trading.data.universe import top_trading_value

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=15, help="시장별 종목 수")
args = parser.parse_args()

client = KiwoomClient.from_env("real")
universe = top_trading_value(client, args.n)
client.revoke()

out = Path("data/sample") / f"universe_{date.today():%Y%m%d}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
universe.to_csv(out, index=False, encoding="utf-8-sig")
print(universe.to_string(index=False))
print(f"\n저장: {out} ({len(universe)}종목)")
