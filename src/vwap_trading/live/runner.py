"""실시간 매매 실행기 (10~11장).

입력은 세 가지뿐이다: 체결 시세(on_tick), 주문체결 이벤트(on_order_event), 1초 타이머(on_timer).
판정 규칙은 05장 백테스트 엔진(SymbolDay.on_bar)과 같다.

  봉 i 확정 → 보유 중: 장 마감·시간·기준선 이탈이면 시장가 매도
            → 무포지션: 매수 후보 + AI 필터 + 위험관리 통과면 시장가 매수
  체결 시세 → 보유 중: 손절가 이하·목표가 이상이면 시장가 매도

백테스트와 다른 점은 체결이다. 백테스트는 "다음 봉 시가"·"손절가 그대로" 체결을 가정하지만,
여기서는 실제 체결 가격(주문체결 이벤트의 체결 금액 / 수량)을 쓰고, 손절가·목표가도 실제
매수 평균가에서 다시 계산한다. 이 차이가 12장에서 재는 "체결 오차"다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from vwap_trading.backtest.costs import CostModel, sell_tax_rate
from vwap_trading.data.session import session_of
from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.live.bars import Bar, MinuteBarBuilder, to_frame
from vwap_trading.live.journal import Journal
from vwap_trading.live.oms import Order, OrderBook
from vwap_trading.live.risk import RiskEngine
from vwap_trading.broker.orders import OrderRejected, OrderUnknown
from vwap_trading.strategy.rules import StrategyParams, base_line, day_clock, entry_allowed, raw_signal

BUY_TIMEOUT = timedelta(seconds=60)      # 시장가 매수가 이 시간 안에 다 체결되지 않으면 남은 수량 취소
SELL_TIMEOUT = timedelta(seconds=30)     # 시장가 매도가 이 시간 안에 다 체결되지 않으면 취소 후 다시 매도
SWEEP_AFTER = timedelta(minutes=4)       # 장 마감 청산 판정 봉(15:14) 이후 4분이 지나도 남은 보유는 강제 매도


@dataclass
class Holding:
    signal_time: datetime
    entry_time: datetime
    entry_idx: int           # 매수 체결 봉 번호 (시간 청산 계산용)
    entry: float             # 실제 매수 평균가
    stop: float
    target: float
    qty: int
    buy_key: str


class LiveSymbol:
    """한 종목-일의 상태. FLAT → BUYING → LONG → SELLING → FLAT."""

    def __init__(self, symbol: str, tradable_from: datetime | None):
        self.symbol, self.tradable_from = symbol, tradable_from
        self.state = "FLAT"
        self.hold: Holding | None = None
        self.signal: tuple[int, datetime, float] | None = None   # (신호 봉, 시각, ATR*)
        self.order: Order | None = None                          # 진행 중인 주문
        self.exit: dict | None = None                            # 청산 사유·판정 봉·발동 가격
        self.next_ok = 0
        self.sell_tries = 0


class Runner:
    def __init__(self, p: StrategyParams, broker, journal: Journal, risk: RiskEngine, day: date,
                 approve=None, cost: CostModel = CostModel(), grace: float = 2.0):
        self.p, self.broker, self.journal, self.risk, self.day = p, broker, journal, risk, day
        self.approve = approve                   # (종목, 지표 x) → (승인 여부, 기록할 정보). None 이면 모두 승인
        self.cost = cost
        self.open_t, self.close_t = (datetime.combine(day, t) for t in session_of(day))
        self.force_t = datetime.combine(day, day_clock(day, p)[1])
        self.builder = MinuteBarBuilder(grace)
        self.bars: dict[str, list[Bar]] = {}
        self.syms: dict[str, LiveSymbol] = {}
        self.book = OrderBook()
        self.used_keys = journal.order_keys()    # 재시작해도 같은 키로 다시 주문하지 않는다
        self.last: dict[str, float] = {}
        self.realized = 0.0
        self.trades: list[dict] = []

    # ── 입력 1: 체결 시세 ─────────────────────────────────────────────────
    def on_tick(self, symbol: str, ts: datetime, price: float, qty: int, now: datetime) -> None:
        if not (self.open_t <= ts < self.close_t + timedelta(minutes=1)):
            return                                               # 정규장 밖 체결은 쓰지 않는다
        self.last[symbol] = price
        for b in self.builder.on_tick(symbol, ts, price, qty, now):
            self._on_bar(symbol, b, now)
        ls = self.syms.get(symbol)
        if ls is not None and ls.state == "LONG":
            h = ls.hold
            if price <= h.stop:                                  # 손절 (봉 안 판정)
                self._exit(ls, "stop", now, price, len(self.bars.get(symbol, [])))
            elif price >= h.target:                              # 목표가
                self._exit(ls, "target", now, price, len(self.bars.get(symbol, [])))

    # ── 입력 2: 주문체결 이벤트 (00) ──────────────────────────────────────
    def on_order_event(self, values: dict, now: datetime) -> None:
        self.journal.event(now, "order_event", values.get("9001", "").lstrip("A"), **values)
        o = self.book.on_event(values, now)
        if o is not None:
            self._on_order_update(o, now)

    # ── 입력 3: 1초 타이머 ────────────────────────────────────────────────
    def on_timer(self, now: datetime) -> None:
        for symbol, b in self.builder.due(now):
            self._on_bar(symbol, b, now)
        if self.risk.kill_requested():
            self.risk.halt("수동 중단 (STOP 파일)")
            for ls in self.syms.values():
                if ls.state == "LONG":
                    self._exit(ls, "kill", now, self.last.get(ls.symbol), len(self.bars.get(ls.symbol, [])))
                elif ls.state == "BUYING" and ls.order is not None and ls.order.is_open:
                    self._cancel(ls.order, now, "수동 중단")
        if any(o.status == "UNKNOWN" and now - o.created > timedelta(seconds=5) for o in self.book.by_key.values()):
            self.reconcile_unknown(now)
        for ls in self.syms.values():
            o = ls.order
            if ls.state == "BUYING" and o is not None and o.is_open and now - o.created > BUY_TIMEOUT:
                self._cancel(o, now, "매수 시간 초과")
            elif ls.state == "SELLING" and o is not None and o.is_open and now - o.created > SELL_TIMEOUT:
                self._cancel(o, now, "매도 시간 초과")
            elif ls.state == "LONG" and now >= self.force_t + SWEEP_AFTER:
                self._exit(ls, "sweep", now, self.last.get(ls.symbol), len(self.bars.get(ls.symbol, [])))

    # ── 종목 선정 ─────────────────────────────────────────────────────────
    def select(self, symbol: str, tradable_from: datetime, now: datetime) -> None:
        ls = self.syms.setdefault(symbol, LiveSymbol(symbol, None))
        if ls.tradable_from is None:                  # 복구한 보유 종목이 선정되면 매매 가능 시각만 채운다
            ls.tradable_from = tradable_from
            self.journal.event(now, "selected", symbol, tradable_from=tradable_from)

    def load_bars(self, symbol: str, df: pd.DataFrame, now: datetime) -> None:
        """재시작할 때 키움 분봉으로 오늘의 확정 봉을 채운다. 이 봉들로는 판정·주문을 하지 않는다."""
        self.bars[symbol] = [Bar(t.to_pydatetime(), r.open, r.high, r.low, r.close, int(r.volume), 0, now, "backfill")
                             for t, r in zip(df.index, df.itertuples())]
        if len(df):
            self.builder.last_closed[symbol] = df.index[-1].to_pydatetime()

    # ── 봉 확정 ───────────────────────────────────────────────────────────
    def _on_bar(self, symbol: str, b: Bar, now: datetime) -> None:
        bars = self.bars.setdefault(symbol, [])
        bars.append(b)
        self.journal.bar(symbol, b)
        ls = self.syms.get(symbol)
        if ls is None:
            return
        i = len(bars) - 1
        if ls.state == "LONG":
            x = compute_indicators(to_frame(bars), self.p.vwap)
            self._exit_rules(ls, x, i, now)
        elif ls.state == "FLAT" and i >= ls.next_ok and i + 1 > self.p.vwap.warmup_end:   # 워밍업 전에는 신호가 없다
            x = compute_indicators(to_frame(bars), self.p.vwap)
            self._entry_rules(ls, x, i, now)

    def _exit_rules(self, ls: LiveSymbol, x: pd.DataFrame, i: int, now: datetime) -> None:
        p, h = self.p, ls.hold
        base = base_line(x, p)
        if x.index[i] >= self.force_t:
            reason = "close"
        elif i - h.entry_idx + 1 >= p.h:
            reason = "time"
        elif base is not None and x.close.iloc[i] < base.iloc[i] - p.eps * x.atr_s.iloc[i]:
            reason = "base"
        else:
            return
        self._exit(ls, reason, now, None, i)

    def _entry_rules(self, ls: LiveSymbol, x: pd.DataFrame, i: int, now: datetime) -> None:
        p = self.p
        sig = raw_signal(x, p) & entry_allowed(x, p, ls.tradable_from)
        if not sig.iloc[i]:
            return
        t = x.index[i]
        info = {}
        if self.approve is not None:
            ok, info = self.approve(ls.symbol, x)
            if not ok:
                self.journal.event(now, "filtered", ls.symbol, bar=t, **info)
                return
        price = self.last.get(ls.symbol, x.close.iloc[i])
        n_open = sum(s.state != "FLAT" for s in self.syms.values())
        blocked = self.risk.check_entry(price, n_open, self.pnl_today())
        if blocked:
            self.journal.event(now, "risk_block", ls.symbol, bar=t, reason=blocked)
            return
        o = Order(f"{self.day}:{ls.symbol}:buy:{t:%H%M}", ls.symbol, "buy", self.risk.qty(price), None, "entry")
        ls.signal = (i, t, float(x.atr_s.iloc[i]))
        self.journal.event(now, "signal", ls.symbol, bar=t, close=x.close.iloc[i], atr_s=ls.signal[2], **info)
        if self._submit(o, now):
            ls.state, ls.order = "BUYING", o
            self.risk.entries += 1

    # ── 주문 ───────────────────────────────────────────────────────────────
    def _submit(self, o: Order, now: datetime) -> bool:
        if o.key in self.used_keys:
            self.journal.event(now, "duplicate_skip", o.symbol, key=o.key)
            return False
        self.used_keys.add(o.key)
        o.created = o.updated = now
        self.book.add(o)
        self.journal.order(o)
        try:
            send = self.broker.buy if o.side == "buy" else self.broker.sell
            ord_no = send(o.symbol, o.qty, o.price)
        except OrderRejected as e:
            o.status, o.updated = "REJECTED", now
            self.journal.order(o)
            self.journal.event(now, "order_rejected", o.symbol, key=o.key, code=e.code, msg=e.msg)
            self.risk.on_order_result(False)
            return False
        except OrderUnknown as e:
            o.status = "UNKNOWN"                     # 들어갔을 수도 있다 → 11-4 의 확인 절차
            self.journal.order(o)
            self.journal.event(now, "order_unknown", o.symbol, key=o.key, msg=str(e))
            self.risk.on_order_result(False)
            return True
        self.risk.on_order_result(True)
        self.journal.event(now, "order_sent", o.symbol, key=o.key, ord_no=ord_no, qty=o.qty, side=o.side)
        for u in self.book.sent(o, ord_no, now):    # 응답보다 먼저 와 있던 체결 이벤트
            self._on_order_update(u, now)
        self.journal.order(o)
        return True

    def _cancel(self, o: Order, now: datetime, why: str) -> None:
        if o.ord_no is None:                          # 주문번호를 모르면 취소할 수 없다 → reconcile_unknown
            return
        if o.cancel_at is not None and now - o.cancel_at < timedelta(seconds=10):
            return                                    # 방금 취소를 보냈다 → 확인 이벤트를 기다린다
        o.cancel_at = now
        try:
            no = self.broker.cancel(o.ord_no, o.symbol, 0)
            self.journal.event(now, "cancel_sent", o.symbol, key=o.key, ord_no=no, why=why)
        except (OrderRejected, OrderUnknown) as e:
            self.journal.event(now, "cancel_failed", o.symbol, key=o.key, why=why, msg=str(e))

    def _exit(self, ls: LiveSymbol, reason: str, now: datetime, trigger: float | None, last_idx: int) -> None:
        h = ls.hold
        ls.state = "SELLING"
        ls.exit = {"reason": reason, "last_idx": last_idx, "trigger": trigger}
        ls.sell_tries += 1
        o = Order(f"{self.day}:{ls.symbol}:sell:{h.entry_time:%H%M}:{ls.sell_tries}", ls.symbol, "sell",
                  h.qty, None, reason)
        self.journal.event(now, "exit", ls.symbol, reason=reason, trigger=trigger, bar_idx=last_idx)
        if self._submit(o, now):
            ls.order = o
        else:
            ls.state = "LONG"                         # 거부되면 보유 상태로 돌아가 다음 판정에서 다시 시도

    # ── 주문 상태 변화 ─────────────────────────────────────────────────────
    def _on_order_update(self, o: Order, now: datetime) -> None:
        self.journal.order(o)
        ls = self.syms.get(o.symbol)
        if ls is None or ls.order is not o or o.status in ("SENT", "PARTIAL"):
            return
        n = len(self.bars.get(o.symbol, []))          # 지금 만들어지는 봉의 번호
        minute = now.replace(second=0, microsecond=0)
        if o.side == "buy":
            if o.filled == 0:                         # 하나도 체결되지 않고 끝남 (취소·거부)
                ls.state, ls.order = "FLAT", None
                return
            i, t, a = ls.signal
            e = o.avg
            ls.hold = Holding(t, minute, n, e, e - self.p.lam * a, e + self.p.tau * a, o.filled, o.key)
            ls.state, ls.order = "LONG", None
            self.journal.save_state(o.symbol, asdict(ls.hold))
            self.journal.event(now, "entered", o.symbol, **asdict(ls.hold))
            return
        h = ls.hold                                   # 매도
        if o.filled < h.qty:                          # 일부만 팔고 끝남 → 남은 수량을 다시 판다
            if o.filled:
                self._record(ls, o, now, partial=True)
            ls.state = "LONG"
            self._exit(ls, ls.exit["reason"], now, ls.exit["trigger"], ls.exit["last_idx"])
            return
        self._record(ls, o, now)
        ls.state, ls.order, ls.hold, ls.exit = "FLAT", None, None, None
        ls.next_ok = n + self.p.cooldown              # 청산 체결 봉부터 cooldown 봉 동안 신호 무시
        self.journal.save_state(o.symbol, None)

    def _record(self, ls: LiveSymbol, o: Order, now: datetime, partial: bool = False) -> None:
        h, ex = ls.hold, ls.exit
        bars = {b.time: b for b in self.bars.get(ls.symbol, [])}
        entry_bar = bars.get(h.entry_time)
        qty = o.filled
        buy_amt, sell_amt = h.entry * qty, o.amount
        pnl = sell_amt * (1 - self.cost.fee - sell_tax_rate(now.date())) - buy_amt * (1 + self.cost.fee)
        self.realized += pnl
        t = {"symbol": ls.symbol, "signal_time": h.signal_time, "entry_time": h.entry_time, "entry": h.entry,
             "stop": h.stop, "target": h.target, "exit_time": now.replace(second=0, microsecond=0),
             "exit": o.avg, "reason": ex["reason"], "bars": ex["last_idx"] - h.entry_idx + 1,
             "ret": o.avg / h.entry - 1, "qty": qty, "buy_avg": h.entry, "sell_avg": o.avg,
             "entry_bar_open": entry_bar.open if entry_bar else None, "trigger_price": ex["trigger"],
             "buy_key": h.buy_key, "sell_key": o.key}
        if partial:
            h.qty -= qty
        self.trades.append(t)
        self.journal.trade(t)
        self.journal.event(now, "trade", ls.symbol, pnl=round(pnl), **{k: t[k] for k in ("reason", "ret", "qty")})

    # ── 장애 대응 (11-4) ───────────────────────────────────────────────────
    def reconcile_unknown(self, now: datetime) -> None:
        """응답을 못 받은 주문(UNKNOWN)이 실제로 들어갔는지 당일 체결·미체결 조회로 확인한다."""
        unknown = [o for o in self.book.by_key.values() if o.status == "UNKNOWN"]
        if not unknown:
            return
        known = set(self.book.by_no)
        seen = pd.concat([self.broker.open_orders()[["ord_no", "symbol", "qty"]],
                          self.broker.fills()[["ord_no", "symbol", "qty"]]]).drop_duplicates("ord_no")
        for o in unknown:
            m = seen[(seen.symbol == o.symbol) & (seen.qty == o.qty)
                     & ~seen.ord_no.map(lambda s: s.lstrip("0") in known)]
            if len(m):                                # 들어가 있었다 → 주문번호를 붙이고 이벤트를 기다린다
                self.journal.event(now, "unknown_found", o.symbol, key=o.key, ord_no=m.ord_no.iloc[0])
                for u in self.book.sent(o, m.ord_no.iloc[0], now):
                    self._on_order_update(u, now)
            else:                                     # 없다 → 주문이 안 들어간 것으로 처리
                o.status, o.updated = "REJECTED", now
                self.journal.event(now, "unknown_absent", o.symbol, key=o.key)
                self._on_order_update(o, now)

    def recover(self, now: datetime) -> None:
        """재시작 직후: 남은 미체결은 취소하고, 보유 종목은 저장해 둔 포지션과 맞춘다."""
        for r in self.broker.open_orders().itertuples():
            self.journal.event(now, "recover_cancel", r.symbol, ord_no=r.ord_no, open_qty=r.open_qty)
            try:
                self.broker.cancel(r.ord_no, r.symbol, 0)
            except (OrderRejected, OrderUnknown) as e:
                self.journal.event(now, "cancel_failed", r.symbol, ord_no=r.ord_no, msg=str(e))
        saved = self.journal.load_state()
        for r in self.broker.holdings().itertuples():
            if r.qty <= 0:
                continue
            ls = self.syms.setdefault(r.symbol, LiveSymbol(r.symbol, None))
            if r.symbol in saved:                     # 이 프로그램이 산 종목 → 손절·목표가를 이어서 감시
                d = saved[r.symbol]
                ls.hold = Holding(**{**d, "signal_time": pd.Timestamp(d["signal_time"]).to_pydatetime(),
                                     "entry_time": pd.Timestamp(d["entry_time"]).to_pydatetime(), "qty": r.qty})
                ls.state = "LONG"
                self.journal.event(now, "recover_hold", r.symbol, qty=r.qty)
            else:                                     # 기록에 없는 보유 → 이유를 모르므로 정리한다
                ls.hold = Holding(now, now.replace(second=0, microsecond=0), 0, r.avg_price, 0.0, float("inf"),
                                  r.qty, "unknown")
                ls.state = "LONG"
                self.journal.event(now, "recover_unknown", r.symbol, qty=r.qty)
                self._exit(ls, "unknown", now, None, 0)

    def pnl_today(self) -> float:
        """실현 손익(비용 차감) + 보유 종목 평가 손익(비용 전), 원."""
        unreal = sum((self.last.get(s.symbol, s.hold.entry) - s.hold.entry) * s.hold.qty
                     for s in self.syms.values() if s.hold is not None)
        return self.realized + unreal
