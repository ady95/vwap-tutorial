"""키움 분봉(ka10080) 수집과 정제 (02장).

키움 응답의 가격 문자열에는 전일 대비 부호(+/-)가 붙어 오므로 절댓값으로 바꾼다.
응답은 최신 봉부터 오므로 시각 오름차순으로 다시 정렬한다.
"""

from __future__ import annotations

import pandas as pd

from vwap_trading.broker.kiwoom import KiwoomClient, KiwoomError

PRICE_COLS = {"open_pric": "open", "high_pric": "high", "low_pric": "low", "cur_prc": "close"}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """ka10080 응답 행 → 정제된 분봉 DataFrame (index: 봉 시각)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.DataFrame({
        "time": pd.to_datetime(df["cntr_tm"], format="%Y%m%d%H%M%S"),
        **{new: pd.to_numeric(df[old].str.lstrip("+-")) for old, new in PRICE_COLS.items()},
        "volume": pd.to_numeric(df["trde_qty"]),
    })
    return out.drop_duplicates("time").set_index("time").sort_index()


def fetch_minute_bars(client: KiwoomClient, symbol: str, tick: int = 1, max_pages: int = 10,
                      on_page=None) -> pd.DataFrame:
    """연속조회로 최근 분봉을 최대 max_pages 페이지(페이지당 약 900봉)까지 받는다.

    도중에 오류가 나면 그때까지 받은 봉을 돌려주고 경고를 출력한다.
    """
    body = {"stk_cd": symbol, "tic_scope": str(tick), "upd_stkpc_tp": "1"}   # 1: 수정주가
    rows: list[dict] = []
    try:
        for page_rows, page in client.post_all("/api/dostk/chart", "ka10080", body,
                                               "stk_min_pole_chart_qry", max_pages):
            rows.extend(page_rows)
            if on_page:
                on_page(page, page_rows)
    except KiwoomError as e:
        print(f"[경고] 수집 중단, 받은 봉까지만 저장합니다: {e}")
    return to_frame(rows)
