"""주문 상태 관리 (10장).

주문은 키(key)로 식별한다. 키는 "날짜:종목:매수/매도:봉 시각:사유" 처럼 같은 판단에서 나온
주문이면 항상 같은 값이 되게 만든다. 같은 키로 두 번 주문하지 않으므로, 프로그램을 다시
시작해 같은 봉을 다시 판정해도 주문이 중복되지 않는다(11-4).

체결은 실시간 주문체결(00) 이벤트로 받는다. 주요 FID:
  9203 주문번호 · 9001 종목코드 · 913 주문상태(접수/체결/확인) · 900 주문수량 · 902 미체결수량
  903 체결누계금액 · 904 원주문번호 · 905 주문구분 · 909 체결번호 · 910 체결가 · 911 체결량 · 919 거부사유
체결 수량은 이벤트 하나하나를 더하지 않고 (주문수량 − 미체결수량) 으로 누계를 다시 계산한다.
같은 이벤트가 두 번 오거나 하나를 놓쳐도 마지막 이벤트만 받으면 누계가 맞는다.
주문 응답(주문번호)보다 체결 이벤트가 먼저 도착할 수 있어, 모르는 주문번호의 이벤트는 보관해 두었다가
주문번호가 등록될 때 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vwap_trading.broker.orders import num

OPEN, DONE = ("NEW", "SENT", "PARTIAL", "UNKNOWN"), ("FILLED", "CANCELLED", "REJECTED")


@dataclass
class Order:
    key: str
    symbol: str
    side: str                  # "buy" | "sell"
    qty: int
    price: float | None        # None 이면 시장가
    reason: str = ""           # entry | stop | target | base | time | close | sweep
    ord_no: str | None = None
    status: str = "NEW"
    filled: int = 0
    amount: float = 0.0        # 체결 금액 누계
    created: datetime | None = None
    updated: datetime | None = None
    cancel_at: datetime | None = None   # 마지막으로 취소를 보낸 시각
    events: list = field(default_factory=list)

    @property
    def avg(self) -> float | None:
        return self.amount / self.filled if self.filled else None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN


def norm(ord_no) -> str:
    """주문번호 정규화. 응답과 이벤트의 0 채움 자릿수가 달라도 같은 주문으로 본다."""
    s = str(ord_no).strip().lstrip("0")
    return s or "0"


class OrderBook:
    def __init__(self):
        self.by_key: dict[str, Order] = {}
        self.by_no: dict[str, Order] = {}
        self.orphans: dict[str, list[dict]] = {}       # 주문번호를 아직 모르는 이벤트

    def has(self, key: str) -> bool:
        return key in self.by_key

    def add(self, order: Order) -> Order:
        if order.key in self.by_key:
            raise ValueError(f"이미 낸 주문입니다: {order.key}")
        self.by_key[order.key] = order
        return order

    def sent(self, order: Order, ord_no: str, now: datetime) -> list[Order]:
        """주문 응답으로 주문번호를 받았다. 먼저 도착해 있던 이벤트를 처리하고 바뀐 주문을 돌려준다."""
        order.ord_no, order.status, order.updated = ord_no, "SENT", now
        self.by_no[norm(ord_no)] = order
        return [o for v in self.orphans.pop(norm(ord_no), []) if (o := self.on_event(v, now))]

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self.by_key.values() if o.is_open and (symbol is None or o.symbol == symbol)]

    def on_event(self, v: dict, now: datetime) -> Order | None:
        """00 이벤트 하나를 반영한다. 상태가 바뀐 주문(없으면 None)을 돌려준다."""
        no, orig = norm(v.get("9203", "")), norm(v.get("904", "0"))
        kind = v.get("905", "")
        if "취소" in kind or "정정" in kind:            # 취소·정정 주문의 이벤트 → 원주문에 반영
            o = self.by_no.get(orig)
            if o is None:
                return None
            o.events.append(v)
            if "취소" in kind and v.get("913") == "확인":
                o.status, o.updated = ("FILLED" if o.filled >= o.qty else "CANCELLED"), now
                return o
            return None
        o = self.by_no.get(no)
        if o is None:
            self.orphans.setdefault(no, []).append(v)
            return None
        o.events.append(v)
        if v.get("919", "").strip():                    # 거부 사유가 있으면 체결이 아니라 거부
            o.status, o.updated = "REJECTED", now
            return o
        if v.get("913") == "체결":
            filled = int(num(v.get("900", o.qty))) - int(num(v.get("902", 0)))
            if filled > o.filled:
                amount = num(v.get("903", 0))
                o.amount = amount if amount > 0 else o.amount + num(v.get("910", 0)) * (filled - o.filled)
                o.filled = filled
            o.status = "FILLED" if o.filled >= o.qty else "PARTIAL"
            o.updated = now
            return o
        return None
