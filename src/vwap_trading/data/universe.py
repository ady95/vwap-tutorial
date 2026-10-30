"""예제 종목 유니버스 (02장).

이 책은 종목 탐색을 다루지 않는다. 실습용으로 키움 거래대금 상위(ka10032)에서
코스피·코스닥 보통주를 골라 고정 유니버스를 만든다.
주의: '오늘' 거래대금 상위로 고른 유니버스이므로 과거 구간에는 생존 편향이 있다.
"""

from __future__ import annotations

import pandas as pd

from vwap_trading.broker.kiwoom import KiwoomClient

MARKETS = {"KOSPI": "001", "KOSDAQ": "101"}
# ETF·ETN·스팩 등 개별 기업이 아닌 상품을 이름으로 거른다
EXCLUDE_WORDS = ("KODEX", "TIGER", "RISE", "KBSTAR", "ACE", "SOL", "HANARO", "KOSEF", "ARIRANG", "PLUS",
                 "1Q", "TIME", "BNK", "WON", "KIWOOM", "ETN", "레버리지", "인버스", "선물", "스팩")


def is_common_stock(code: str, name: str) -> bool:
    if not code.isdigit() or code[-1] != "0":      # 우선주(끝자리 5·7 등) 제외
        return False
    return not any(w in name for w in EXCLUDE_WORDS)


def top_trading_value(client: KiwoomClient, n_per_market: int = 15) -> pd.DataFrame:
    rows = []
    for market, code in MARKETS.items():
        data, _ = client.post("/api/dostk/rkinfo", "ka10032",
                              {"mrkt_tp": code, "mang_stk_incls": "0", "stex_tp": "1"})
        picked = [r for r in data["trde_prica_upper"] if is_common_stock(r["stk_cd"], r["stk_nm"])]
        for r in picked[:n_per_market]:
            rows.append({"symbol": r["stk_cd"], "name": r["stk_nm"], "market": market,
                         "rank": int(r["now_rank"]), "trading_value": int(r["trde_prica"])})
    return pd.DataFrame(rows)
