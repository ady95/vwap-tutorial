"""02장 실습 1 — 키움 REST API 연결 확인 (조회 전용, 주문 없음).

실행: uv run python examples/chapter02/01_check_connection.py
"""

import sys

from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient, KiwoomError
from vwap_trading.data.minute import fetch_minute_bars

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))

for env in ("real", "paper"):
    print(f"===== {env}")
    try:
        client = KiwoomClient.from_env(env)
        client.token()
        print("  토큰 발급: 성공")
        info, _ = client.post("/api/dostk/stkinfo", "ka10001", {"stk_cd": "005930"})
        print(f"  ka10001 주식기본정보: {info['stk_nm']} 현재가 {info['cur_prc'].lstrip('+-')}")
        bars = fetch_minute_bars(client, "005930", max_pages=1)
        print(f"  ka10080 1분봉: {len(bars)}봉, {bars.index.min()} ~ {bars.index.max()}")
        print(bars.tail(3).to_string())
        client.revoke()
    except KiwoomError as e:
        print("  실패:", e)
