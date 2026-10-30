"""재생 모드 (10장). 과거 1분봉을 체결 시세처럼 흘려보내 실시간 실행기를 장 밖에서 시험한다.

봉 하나를 체결 다섯 개로 쪼갠다.
  :00 시가 (수량 0)  — 직전 봉을 확정시키는 첫 체결
  :01 시가 · :15 저가 · :30 고가 · :45 종가 — 거래량을 넷으로 나눔
저가를 고가보다 먼저 두는 것은 백테스트의 "같은 봉에서 손절·목표가 모두 닿으면 손절"과 맞추기
위해서다. 시장가 주문은 "그 종목의 마지막 체결 가격"에 바로 채운다. 그래서 봉 확정 직후 낸
주문은 다음 봉 시가에 체결되고, 손절은 손절가가 아니라 손절을 발동시킨 체결 가격(저가 등)에
체결된다. 이 점만 백테스트와 다르다.

SimBroker 는 PaperBroker 와 같은 메서드를 갖고, 체결 결과를 실제와 같은 형식의 주문체결(00)
이벤트로 돌려준다. 이벤트는 outbox 에 쌓았다가 재생기가 실행기에 넘긴다(실제로도 이벤트는
주문 응답과 따로 도착한다).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from vwap_trading.broker.orders import OrderRejected


class SimBroker:
    def __init__(self):
        self.last: dict[str, float] = {}
        self.outbox: list[dict] = []
        self.orders: dict[str, dict] = {}
        self.pos: dict[str, int] = {}
        self.seq = 0
        self.fail_next: str | None = None        # 시험용: 다음 주문을 "reject" 또는 "unknown" 으로 만든다

    def _no(self) -> str:
        self.seq += 1
        return f"{self.seq:07d}"

    def buy(self, symbol, qty, price=None):
        return self._new(symbol, qty, price, "+매수")

    def sell(self, symbol, qty, price=None):
        return self._new(symbol, qty, price, "-매도")

    def _new(self, symbol, qty, price, kind):
        if self.fail_next == "reject":
            self.fail_next = None
            raise OrderRejected("sim", 20, "시험용 거부")
        no = self._no()
        self.orders[no] = {"symbol": symbol, "qty": qty, "price": price, "kind": kind, "open": qty}
        if price is None:
            self._fill(no, self.last[symbol])
        return no

    def _fill(self, no, px):
        o = self.orders[no]
        q = o["open"]
        o["open"] = 0
        self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0) + (q if o["kind"] == "+매수" else -q)
        self.outbox.append({"9203": no, "9001": "A" + o["symbol"], "913": "체결", "900": str(o["qty"]),
                            "901": str(o["price"] or 0), "902": "0", "903": str(px * q), "904": "0000000",
                            "905": o["kind"], "909": self._no(), "910": str(px), "911": str(q), "919": ""})

    def on_price(self, symbol, price):
        """지정가 주문: 가격이 닿으면 체결."""
        self.last[symbol] = price
        for no, o in self.orders.items():
            if o["symbol"] == symbol and o["open"] and o["price"] is not None:
                if (o["kind"] == "+매수" and price <= o["price"]) or (o["kind"] == "-매도" and price >= o["price"]):
                    self._fill(no, o["price"])

    def cancel(self, orig, symbol, qty=0):
        no = self._no()
        o = self.orders[orig]
        o["open"] = 0
        self.outbox.append({"9203": no, "9001": "A" + symbol, "913": "확인", "904": orig,
                            "905": o["kind"][1:] + "취소", "902": "0", "919": ""})
        return no

    def modify(self, orig, symbol, qty, price):
        self.orders[orig]["price"] = price
        return self._no()

    def open_orders(self):
        rows = [{"ord_no": n, "symbol": o["symbol"], "side": o["kind"], "qty": o["qty"], "open_qty": o["open"],
                 "price": o["price"]} for n, o in self.orders.items() if o["open"]]
        return pd.DataFrame(rows, columns=["ord_no", "symbol", "side", "qty", "open_qty", "price"])

    def fills(self):
        rows = [{"ord_no": n, "symbol": o["symbol"], "side": o["kind"], "qty": o["qty"], "filled": o["qty"] - o["open"],
                 "price": 0.0} for n, o in self.orders.items() if o["qty"] > o["open"]]
        return pd.DataFrame(rows, columns=["ord_no", "symbol", "side", "qty", "filled", "price"])

    def holdings(self):
        return pd.DataFrame([{"symbol": s, "qty": q, "avg_price": 0.0} for s, q in self.pos.items() if q],
                            columns=["symbol", "qty", "avg_price"])

    def cash(self):
        return 100_000_000


def ticks_from_bars(bars_by_symbol: dict[str, pd.DataFrame]) -> list[tuple[datetime, str, float, int]]:
    """봉 → (시각, 종목, 가격, 수량) 체결 목록 (시각 순, 같은 시각이면 종목 입력 순)."""
    out = []
    for k, (s, b) in enumerate(bars_by_symbol.items()):
        for t, r in zip(b.index, b.itertuples()):
            v = int(r.volume)
            q = [v // 4] * 3 + [v - 3 * (v // 4)]
            for sec, px, qty in ((0, r.open, 0), (1, r.open, q[0]), (15, r.low, q[1]), (30, r.high, q[2]),
                                 (45, r.close, q[3])):
                out.append((t + timedelta(seconds=sec), k, s, float(px), qty))
    out.sort(key=lambda e: (e[0], e[1]))
    return [(t, s, px, q) for t, _, s, px, q in out]


def replay(runner, broker: SimBroker, bars_by_symbol: dict[str, pd.DataFrame], day: date,
           selections: dict[str, datetime] | None = None, select_at: datetime | None = None, on_tick=None,
           start: datetime | None = None) -> None:
    """하루를 재생한다. selections: 종목 → tradable_from. select_at 시각에 선정한다(없으면 처음부터).
    start 가 있으면 그 시각 이전 체결은 건너뛴다(재시작 훈련). on_tick(ts) 가 False 를 돌려주면 멈춘다."""
    selections = selections or {}
    chosen = select_at is None
    if chosen:
        for s, tf in selections.items():
            runner.select(s, tf, datetime.combine(day, datetime.min.time()))

    def drain(now):
        while broker.outbox:
            runner.on_order_event(broker.outbox.pop(0), now)

    last_sec = None
    for ts, s, px, q in ticks_from_bars(bars_by_symbol):
        if start is not None and ts < start:
            continue
        if ts != last_sec:
            runner.on_timer(ts)
            drain(ts)
            last_sec = ts
        if not chosen and ts >= select_at:
            for sym, tf in selections.items():
                runner.select(sym, tf, ts)
            chosen = True
        broker.on_price(s, px)
        drain(ts)
        runner.on_tick(s, ts, px, q, ts)
        drain(ts)
        if on_tick and on_tick(ts) is False:
            return
    end = datetime.combine(day, datetime.max.time())
    runner.on_timer(end)
    drain(end)
