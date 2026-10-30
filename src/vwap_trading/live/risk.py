"""위험관리 (11장). 전략·AI 와 독립적으로, 신규 진입을 허락할지와 주문 수량만 정한다.

위험관리는 매수만 막는다. 청산(손절·목표·시간·장 마감)은 어떤 한도에 걸려도 막지 않는다.
한도에 걸려 매도까지 막으면 손실이 더 커진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiskLimits:
    budget: int = 1_000_000          # 거래당 투입 금액 (원). 05-3 의 "거래당 고정 금액"
    max_positions: int = 3           # 동시 보유 종목 수 (매수 주문 중 포함)
    daily_loss: int = 30_000         # 일일 손실 한도 (실현 손익 + 평가 손익, 원)
    max_entries: int = 20            # 하루 신규 진입 횟수
    max_consecutive_errors: int = 3  # 주문 오류가 연속 이만큼이면 신규 진입 중단
    kill_file: str = "STOP"          # 이 파일이 생기면 신규 진입 중단 + 전량 청산


class RiskEngine:
    def __init__(self, limits: RiskLimits = RiskLimits()):
        self.limits = limits
        self.halted: str | None = None     # 신규 진입 중단 사유 (한 번 걸리면 그날은 풀지 않는다)
        self.entries = 0
        self.errors = 0

    def qty(self, price: float) -> int:
        return int(self.limits.budget // price)

    def halt(self, reason: str) -> None:
        if self.halted is None:
            self.halted = reason

    def kill_requested(self) -> bool:
        return Path(self.limits.kill_file).exists()

    def check_entry(self, price: float, n_open: int, pnl_today: float) -> str | None:
        """신규 진입을 막을 사유. None 이면 허락."""
        lim = self.limits
        if self.kill_requested():
            self.halt("수동 중단 (STOP 파일)")
        if pnl_today <= -lim.daily_loss:
            self.halt(f"일일 손실 한도 ({pnl_today:,.0f}원 ≤ −{lim.daily_loss:,}원)")
        if self.errors >= lim.max_consecutive_errors:
            self.halt(f"주문 오류 연속 {self.errors}회")
        if self.halted:
            return self.halted
        if n_open >= lim.max_positions:
            return f"동시 보유 한도 {lim.max_positions}종목"
        if self.entries >= lim.max_entries:
            return f"하루 진입 한도 {lim.max_entries}회"
        if self.qty(price) < 1:
            return f"1주 가격({price:,.0f}원)이 거래당 금액보다 큼"
        return None

    def on_order_result(self, ok: bool) -> None:
        self.errors = 0 if ok else self.errors + 1
