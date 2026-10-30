"""거래 로그 (10장). 하루에 SQLite 파일 하나 (reports/live/YYYYMMDD.sqlite).

  events : 일어난 모든 일 (시세 연결, 신호, 위험관리 거부, 주문 요청·응답, 체결 이벤트 원문, 오류)
  bars   : 실시간으로 만든 확정 봉 (장 마감 후 키움 분봉과 대조, 12장)
  orders : 주문의 마지막 상태
  trades : 끝난 매매. 백테스트 매매 기록(TRADE_COLS)과 같은 열 + 실제 체결 정보
  state  : 보유 포지션 (재시작할 때 복구용)

모든 기록은 그 자리에서 커밋한다. 프로그램이 죽어도 직전까지의 기록은 남는다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (ts TEXT, kind TEXT, symbol TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS bars (symbol TEXT, time TEXT, open REAL, high REAL, low REAL, close REAL,
    volume INTEGER, ticks INTEGER, confirmed_at TEXT, how TEXT, PRIMARY KEY (symbol, time));
CREATE TABLE IF NOT EXISTS orders (key TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty INTEGER, price REAL,
    reason TEXT, ord_no TEXT, status TEXT, filled INTEGER, avg REAL, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS trades (symbol TEXT, signal_time TEXT, entry_time TEXT, entry REAL, stop REAL,
    target REAL, exit_time TEXT, exit REAL, reason TEXT, bars INTEGER, ret REAL, qty INTEGER,
    buy_avg REAL, sell_avg REAL, entry_bar_open REAL, trigger_price REAL, buy_key TEXT, sell_key TEXT);
CREATE TABLE IF NOT EXISTS state (symbol TEXT PRIMARY KEY, data TEXT);
"""


def _s(v):
    return v.isoformat(sep=" ") if isinstance(v, datetime) else v


class Journal:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    def event(self, ts: datetime, kind: str, symbol: str = "", **data) -> None:
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)",
                        (_s(ts), kind, symbol, json.dumps(data, ensure_ascii=False, default=str)))
        self.db.commit()

    def bar(self, symbol: str, b) -> None:
        d = asdict(b)
        self.db.execute("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (symbol, _s(d["time"]), d["open"], d["high"], d["low"], d["close"], d["volume"],
                         d["ticks"], _s(d["confirmed_at"]), d["how"]))
        self.db.commit()

    def order(self, o) -> None:
        self.db.execute("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (o.key, o.symbol, o.side, o.qty, o.price, o.reason, o.ord_no, o.status, o.filled,
                         o.avg, _s(o.created), _s(o.updated)))
        self.db.commit()

    def order_keys(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT key FROM orders")}

    def trade(self, t: dict) -> None:
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(trades)")]
        self.db.execute(f"INSERT INTO trades VALUES ({','.join('?' * len(cols))})", [_s(t.get(c)) for c in cols])
        self.db.commit()

    def save_state(self, symbol: str, data: dict | None) -> None:
        if data is None:
            self.db.execute("DELETE FROM state WHERE symbol=?", (symbol,))
        else:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (symbol, json.dumps(data, default=str)))
        self.db.commit()

    def load_state(self) -> dict[str, dict]:
        return {s: json.loads(d) for s, d in self.db.execute("SELECT symbol, data FROM state")}

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {name}", self.db)

    def close(self) -> None:
        self.db.close()
