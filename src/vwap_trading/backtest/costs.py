"""거래비용 모델 (05장, 수식 명세 v0.2 10절).

- 수수료: 매수·매도 각 fee (가정값 0.015%. 실제 계좌 수수료율로 바꿀 것)
- 증권거래세(매도): 매도 체결일 기준. 2025년까지 0.15%, 2026년부터 0.20%
  (코스피는 거래세 0.05% + 농어촌특별세 0.15%, 코스닥은 거래세 0.20%. 합계가 같아 시장 구분 없이 적용)
- 슬리피지: 매수·매도 각 slip_ticks 호가 (KRX 가격대별 호가단위)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

# (기준 가격 미만, 호가단위) — KRX 유가증권·코스닥 공통 (2023-01-25 개정)
TICK_TABLE = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000

# (적용 시작일, 매도 세율) — 시작일 이후 매도 체결분부터 적용
SELL_TAX = ((date(2025, 1, 1), 0.0015), (date(2026, 1, 1), 0.0020))


def tick_size(price: float) -> int:
    for bound, tick in TICK_TABLE:
        if price < bound:
            return tick
    return TOP_TICK


def sell_tax_rate(d: date) -> float:
    rate = SELL_TAX[0][1]
    for start, r in SELL_TAX:
        if d >= start:
            rate = r
    return rate


@dataclass(frozen=True)
class CostModel:
    fee: float = 0.00015       # 매수·매도 각각 (가정값)
    slip_ticks: float = 1.0    # 매수·매도 각각 몇 호가 불리하게 체결되는가
    tax: bool = True           # 증권거래세 적용 여부 (비용 분해용)

    def buy_price(self, price: float) -> float:
        return price + self.slip_ticks * tick_size(price)

    def sell_price(self, price: float) -> float:
        return price - self.slip_ticks * tick_size(price)

    def net_return(self, entry: float, exit_: float, sell_day: date) -> float:
        """비용을 모두 뺀 거래 수익률. 매수 대금(수수료 포함) 대비."""
        buy = self.buy_price(entry) * (1 + self.fee)
        tax = sell_tax_rate(sell_day) if self.tax else 0.0
        sell = self.sell_price(exit_) * (1 - self.fee - tax)
        return sell / buy - 1

    def net_returns(self, trades) -> np.ndarray:
        """매매 기록(entry, exit, exit_time 열) 전체의 순수익률."""
        return np.array([self.net_return(e, x, t.date())
                         for e, x, t in zip(trades.entry, trades.exit, trades.exit_time)])
