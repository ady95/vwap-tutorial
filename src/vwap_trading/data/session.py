"""정규장 세션 필터 (02장).

KRX 정규장은 09:00~15:30 이지만, 수능일·새해 첫 거래일처럼 장 시간이 바뀌는 날이 있다.
그런 날은 SPECIAL_SESSIONS 에 (개장, 마감) 을 등록한다. 2026-09-14 이후 분봉에 섞여 오는
애프터마켓(15:40~) 봉은 이 필터로 걸러진다.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd

REGULAR = (time(9, 0), time(15, 30))
SPECIAL_SESSIONS: dict[date, tuple[time, time]] = {
    date(2025, 11, 13): (time(10, 0), time(16, 30)),   # 2026학년도 수능일
    date(2026, 1, 2): (time(10, 0), time(15, 30)),     # 새해 첫 거래일
}


def session_of(day: date) -> tuple[time, time]:
    return SPECIAL_SESSIONS.get(day, REGULAR)


def regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    """날짜별 정규장 범위 안의 봉만 남긴다."""
    days = pd.Series(bars.index.date, index=bars.index)
    opens = days.map(lambda d: session_of(d)[0])
    closes = days.map(lambda d: session_of(d)[1])
    clock = pd.Series(bars.index.time, index=bars.index)
    return bars[(clock >= opens) & (clock <= closes)]
