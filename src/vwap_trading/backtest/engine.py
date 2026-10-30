"""이벤트 기반 백테스트 엔진 (05장).

모든 선정 종목-일의 1분봉을 시각 순서로 하나씩 흘려보내며, 봉이 확정될 때마다 종목별
SymbolDay.on_bar 를 부른다. 한 봉에서 일어나는 일의 순서는 다음과 같다.

  1. 직전 봉에서 낸 주문을 이 봉 시가로 체결
  2. 보유 중이면 봉 안 손절·목표가 판정, 아니면 봉 확정 후 청산 판정 → 다음 봉 시가 매도 주문
  3. 무포지션이면 매수 후보 판정 → 다음 봉 시가 매수 주문

실시간 매매(10~11장)도 같은 순서로 동작하도록, 판정은 봉 i 와 그 이전 값만 쓴다.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import pandas as pd

from vwap_trading.backtest.costs import CostModel
from vwap_trading.strategy.rules import StrategyParams, base_line, day_clock, entry_allowed, raw_signal

TRADE_COLS = ["symbol", "signal_time", "entry_time", "entry", "stop", "target", "exit_time", "exit",
              "reason", "bars", "ret", "buy_fill", "sell_fill", "net"]


@dataclass
class Position:
    signal: int          # 신호 봉
    entry_idx: int       # 진입 봉
    entry: float
    stop: float
    target: float


class SymbolDay:
    """한 종목-일의 매매 상태 기계."""

    def __init__(self, symbol: str, x: pd.DataFrame, p: StrategyParams, tradable_from=None, approve=None):
        self.symbol, self.x, self.p = symbol, x, p
        self.approve = approve                                # (종목, 신호 시각) → 승인 여부. None 이면 모두 승인
        self.sig = (raw_signal(x, p) & entry_allowed(x, p, tradable_from)).to_numpy()
        base = base_line(x, p)
        self.base = None if base is None else base.to_numpy()
        self.o, self.hi, self.lo, self.c = (x[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.atr_s = x.atr_s.to_numpy()
        self.force = x.index.time >= day_clock(x.index[0].date(), p)[1]
        self.n = len(x)
        self.pos: Position | None = None
        self.pending_buy: int | None = None      # 신호 봉 번호
        self.pending_sell: tuple[str, int] | None = None   # (사유, 판정 봉)
        self.next_ok = 0

    def wants_entry(self) -> bool:
        return self.pending_buy is not None

    def on_bar(self, i: int, can_enter=lambda: True) -> dict | None:
        """봉 i 가 확정됐을 때 호출한다. 청산이 체결되면 매매 기록(dict)을 돌려준다."""
        p, done = self.p, None
        # 1. 대기 주문 체결 (이 봉 시가)
        if self.pending_buy is not None:
            t = self.pending_buy
            risk = self.atr_s[t]                                  # ATR* 는 신호 봉 값으로 고정
            entry = self.o[i]
            self.pos = Position(t, i, entry, entry - p.lam * risk, entry + p.tau * risk)
            self.pending_buy = None
        elif self.pending_sell is not None:
            reason, j = self.pending_sell
            done = self._close(reason, self.o[i], i, j)
            self.pending_sell = None

        # 2. 보유 중: 봉 안 판정 → 봉 확정 판정
        pos = self.pos
        if pos is not None:
            if self.lo[i] <= pos.stop:                            # (1) 손절 (같은 봉이면 손절 우선)
                return self._close("stop", min(pos.stop, self.o[i]), i, i)
            if self.hi[i] >= pos.target:                          # (2) 목표가
                return self._close("target", max(pos.target, self.o[i]), i, i)
            if i == self.n - 1:                                   # 데이터 끝 (안전장치)
                return self._close("eod", self.c[i], i, i)
            if self.force[i]:                                     # (5) 장 마감 청산
                self.pending_sell = ("close", i)
            elif i - pos.entry_idx + 1 >= p.h:                    # (4) 시간 청산
                self.pending_sell = ("time", i)
            elif self.base is not None and self.c[i] < self.base[i] - p.eps * self.atr_s[i]:
                self.pending_sell = ("base", i)                   # (3) 기준선 이탈
            return done

        # 3. 무포지션: 매수 후보 → 다음 봉 시가 주문
        if self.sig[i] and i >= self.next_ok and i < self.n - 1 and can_enter():
            if self.approve is None or self.approve(self.symbol, self.x.index[i]):   # 08장 AI 필터
                self.pending_buy = i
        return done

    def _close(self, reason: str, price: float, fill: int, last: int) -> dict:
        pos, x = self.pos, self.x
        self.pos = None
        self.next_ok = fill + self.p.cooldown                     # 청산 체결 봉부터 cooldown 봉 동안 신호 무시
        return {"symbol": self.symbol, "signal_time": x.index[pos.signal], "entry_time": x.index[pos.entry_idx],
                "entry": pos.entry, "stop": pos.stop, "target": pos.target, "exit_time": x.index[fill],
                "exit": price, "reason": reason, "bars": last - pos.entry_idx + 1, "ret": price / pos.entry - 1}


def backtest(ind: dict[str, pd.DataFrame], selections: pd.DataFrame, p: StrategyParams,
             cost: CostModel = CostModel(), max_positions: int | None = None, approve=None) -> pd.DataFrame:
    """선정 종목-일 전체를 시각 순서로 흘려보내 매매 기록을 만든다.

    max_positions: 동시 보유(대기 매수 포함) 종목 수 한도. None 이면 제한 없음(05장 기준 실험).
    approve: (종목, 신호 시각) → bool. 거부된 매수 후보는 주문하지 않는다(08장 LightGBM 필터).
    """
    by_day = {s: dict(tuple(x.groupby(x.index.date))) for s, x in ind.items()}
    traders = []
    for sel in selections.itertuples():
        tf = pd.Timestamp(sel.tradable_from)
        day = by_day.get(sel.symbol, {}).get(tf.date())
        if day is not None:
            traders.append(SymbolDay(sel.symbol, day, p, tf, approve))

    def held():
        return sum(t.pos is not None or t.wants_entry() for t in traders)

    def can_enter():
        return max_positions is None or held() < max_positions

    # 이벤트 큐: (봉 시각, 종목-일 번호, 봉 번호) — 같은 시각이면 선정 입력 순서대로
    events = heapq.merge(*[[(ts, k, i) for i, ts in enumerate(t.x.index)] for k, t in enumerate(traders)])
    rows = []
    for _, k, i in events:
        tr = traders[k].on_bar(i, can_enter)
        if tr is not None:
            rows.append(tr)

    out = pd.DataFrame(rows, columns=TRADE_COLS[:11])
    if out.empty:
        return pd.DataFrame(columns=TRADE_COLS)
    out["buy_fill"] = out.entry.map(cost.buy_price)
    out["sell_fill"] = out.exit.map(cost.sell_price)
    out["net"] = cost.net_returns(out)
    return out.sort_values(["exit_time", "symbol"], kind="stable").reset_index(drop=True)
