"""실시간 체결 → 1분봉 (10장).

키움 실시간 주식체결(0B)의 체결시간(FID 20, HHMMSS)으로 봉을 나눈다. 봉 시각은 구간의 시작이다
(04-3: 10:15 봉 = 10:15:00~10:15:59 체결). 봉 t 는 다음 둘 중 먼저 오는 때에 확정한다.

  1. 그 종목의 t+1분 이후 체결이 들어왔을 때 (how="tick")
  2. 체결이 없더라도 로컬 시계가 t+1분 + grace 초를 지났을 때 (how="timer", 거래가 뜸한 종목)

확정한 봉에 속하는 체결이 늦게 도착하면 봉을 고치지 않고 버린 뒤 개수만 센다. 고치면
이미 판정한 신호가 바뀌는 리페인팅이 되기 때문이다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

MINUTE = timedelta(minutes=1)


@dataclass
class Bar:
    time: datetime       # 봉 시각 (구간 시작)
    open: float
    high: float
    low: float
    close: float
    volume: int
    ticks: int = 1       # 봉에 들어간 체결 건수
    confirmed_at: datetime | None = None
    how: str = ""        # 확정 방식: "tick" | "timer"


def parse_0b(values: dict, day: date) -> tuple[datetime, float, int]:
    """0B 의 values → (체결 시각, 체결가, 체결량). 가격·수량에는 부호가 붙어 온다 (+ 매수 체결, − 매도 체결)."""
    hhmmss = values["20"]
    ts = datetime.combine(day, time(int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])))
    return ts, abs(float(values["10"])), abs(int(values["15"]))


class MinuteBarBuilder:
    def __init__(self, grace: float = 2.0):
        self.grace = timedelta(seconds=grace)
        self.cur: dict[str, Bar] = {}
        self.last_closed: dict[str, datetime] = {}
        self.late = Counter()          # 종목별 확정 뒤 도착해 버린 체결 수

    def on_tick(self, symbol: str, ts: datetime, price: float, qty: int, now: datetime | None = None) -> list[Bar]:
        """체결 하나를 넣고, 이 체결로 확정된 봉 목록을 돌려준다 (0개 또는 1개)."""
        minute = ts.replace(second=0, microsecond=0)
        last = self.last_closed.get(symbol)
        if last is not None and minute <= last:
            self.late[symbol] += 1
            return []
        done = []
        b = self.cur.get(symbol)
        if b is not None and minute > b.time:
            done.append(self._close(symbol, now or ts, "tick"))
            b = None
        if b is None:
            self.cur[symbol] = Bar(minute, price, price, price, price, qty)
        else:
            b.high, b.low, b.close = max(b.high, price), min(b.low, price), price
            b.volume += qty
            b.ticks += 1
        return done

    def due(self, now: datetime) -> list[tuple[str, Bar]]:
        """로컬 시계 기준으로 끝난 지 grace 초가 지난 봉을 확정한다."""
        out = []
        for symbol, b in list(self.cur.items()):
            if now >= b.time + MINUTE + self.grace:
                out.append((symbol, self._close(symbol, now, "timer")))
        return out

    def _close(self, symbol: str, now: datetime, how: str) -> Bar:
        b = self.cur.pop(symbol)
        b.confirmed_at, b.how = now, how
        self.last_closed[symbol] = b.time
        return b


def to_frame(bars: list[Bar]) -> pd.DataFrame:
    """확정 봉 목록 → 02장 분봉과 같은 형식의 DataFrame (index: time, open·high·low·close·volume)."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="time"))
    idx = pd.DatetimeIndex([b.time for b in bars], name="time")
    return pd.DataFrame({k: [getattr(b, k) for b in bars] for k in ("open", "high", "low", "close", "volume")},
                        index=idx)
