"""11장 실습 4 — 위험관리·장애 대응 훈련. 과거 하루를 재생하면서 일부러 문제를 만든다. 주문은 내지 않는다.

실행: uv run python examples/chapter11/04_drills.py [날짜] [설정]   (기본 2026-10-12, configs/strategy_a.toml)
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from vwap_trading.broker.orders import OrderUnknown
from vwap_trading.live.journal import Journal
from vwap_trading.live.replay import SimBroker, replay
from vwap_trading.live.risk import RiskEngine, RiskLimits
from vwap_trading.live.runner import Runner
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-12").date()
p = load_params(sys.argv[2] if len(sys.argv) > 2 else "configs/strategy_a.toml")
sel = pd.read_csv("data/sample/selections.csv", parse_dates=["tradable_from"], dtype={"symbol": str})
sel = sel[sel.tradable_from.dt.date == day]
bars = {s: (b := pd.read_parquet(f"data/raw/{s}_1m.parquet"))[b.index.date == day] for s in sel.symbol}
SEL = dict(zip(sel.symbol, sel.tradable_from.dt.to_pydatetime()))
AT = sel.tradable_from.min().to_pydatetime()
tmp = Path(tempfile.mkdtemp())


def run(name, limits=RiskLimits(), broker=None, journal=None, **kw):
    broker = broker or SimBroker()
    j = journal or Journal(":memory:")
    r = Runner(p, broker, j, RiskEngine(limits), day)
    replay(r, broker, bars, day, SEL, select_at=AT, **kw)
    ev = j.table("events")
    return r, ev


def show(name, r, ev, kinds):
    counts = ev.kind.value_counts()
    t = pd.DataFrame(r.trades)
    trades = ", ".join(f"{x.symbol} {x.entry_time:%H:%M}→{x.exit_time:%H:%M} {x.reason}" for x in t.itertuples()) if len(t) else "없음"
    print(f"\n[{name}] 매매 {len(t)}건: {trades}")
    print("  이벤트: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in kinds) + f" | 실현 {r.realized:+,.0f}원")
    for row in ev[ev.kind.isin(kinds[1:])].head(4).itertuples():
        print(f"    {row.ts[11:19]} {row.kind} {row.symbol} {row.data[:110]}")


print(f"훈련 날짜 {day}, 선정 {list(SEL)}, 거래당 {RiskLimits().budget:,}원")
r, ev = run("기준")
show("0. 기준 (한도 기본값)", r, ev, ["signal", "risk_block", "order_sent", "trade"])

r, ev = run("일일 손실", RiskLimits(daily_loss=5_000))
show("1. 일일 손실 한도 5,000원", r, ev, ["signal", "risk_block", "trade"])


class Rejecting(SimBroker):
    """매수 주문을 처음 n번 거부한다."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def buy(self, symbol, qty, price=None):
        if self.n > 0:
            self.n -= 1
            self.fail_next = "reject"
        return super().buy(symbol, qty, price)


r, ev = run("거부", broker=Rejecting(3))
show("2. 주문 거부 연속 3회", r, ev, ["signal", "order_rejected", "risk_block", "trade"])


class LostResponse(SimBroker):
    """첫 매수는 서버에 들어가 체결되지만 응답을 받지 못한다 (네트워크 시간 초과)."""
    lost = False

    def buy(self, symbol, qty, price=None):
        no = super().buy(symbol, qty, price)
        if not self.lost:
            self.lost = True
            raise OrderUnknown("[kt10000] 응답 없음: Timeout")
        return no


r, ev = run("응답 유실", broker=LostResponse())
show("3. 주문 응답 유실 → 조회로 확인", r, ev, ["order_unknown", "unknown_found", "unknown_absent", "trade"])


class SlowSell(SimBroker):
    """첫 매도는 체결되지 않고 걸려 있다."""
    stuck = False

    def sell(self, symbol, qty, price=None):
        if not self.stuck:
            self.stuck = True
            no = self._no()
            self.orders[no] = {"symbol": symbol, "qty": qty, "price": None, "kind": "-매도", "open": qty}
            return no
        return super().sell(symbol, qty, price)


r, ev = run("매도 미체결", broker=SlowSell())
show("4. 매도 미체결 30초 → 취소 후 다시 매도", r, ev, ["exit", "cancel_sent", "order_sent", "trade"])

kill = tmp / "STOP"
box = {}


def stop_when_long(ts):
    r = box["r"]
    if not kill.exists() and any(s.state == "LONG" for s in r.syms.values()):
        kill.touch()                                  # 처음 보유가 생긴 순간 STOP 파일을 만든다


broker = SimBroker()
box["r"] = Runner(p, broker, Journal(":memory:"), RiskEngine(RiskLimits(kill_file=str(kill))), day)
replay(box["r"], broker, bars, day, SEL, select_at=AT, on_tick=stop_when_long)
show("5. 보유 중 STOP 파일 → 전량 매도·신규 중단", box["r"], box["r"].journal.table("events"),
     ["signal", "exit", "risk_block", "trade"])
kill.unlink()

# 7. 재시작: 한 종목을 보유한 채 프로그램이 죽고, 몇 분 뒤 다시 켠다
j = Journal(tmp / "restart.sqlite")
broker = SimBroker()
first = Runner(p, broker, j, RiskEngine(), day)
crash = {}


def crash_when_long(ts):
    if any(s.state == "LONG" for s in first.syms.values()) and "at" not in crash:
        crash["at"] = ts
        return False


replay(first, broker, bars, day, SEL, select_at=AT, on_tick=crash_when_long)
restart = crash["at"].replace(second=0) + pd.Timedelta(minutes=3)
print(f"\n[6. 재시작] {crash['at']:%H:%M:%S} 보유 {broker.holdings().to_dict('records')} 상태에서 종료 → {restart:%H:%M} 재시작")
second = Runner(p, broker, j, RiskEngine(), day)
second.recover(restart)
for s, b in bars.items():
    second.load_bars(s, b[b.index < restart], restart)
for s, tf in SEL.items():
    second.select(s, tf, restart)
replay(second, broker, bars, day, start=restart)
ev = j.table("events")
show("6. 재시작 후", second, ev[ev.ts >= f"{restart}"], ["recover_hold", "recover_unknown", "duplicate_skip", "trade"])
print(f"  첫 실행 매매 {len(first.trades)}건 + 재시작 후 {len(second.trades)}건, 마지막 잔고 {broker.holdings().to_dict('records')}")
