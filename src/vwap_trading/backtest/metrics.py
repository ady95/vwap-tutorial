"""성과 지표 (05장).

기준 실험은 "거래당 고정 금액"이므로, 매매 1건의 수익률을 그대로 더하면 투입 금액 대비 누적
손익이 된다(복리 아님). 낙폭(MDD)도 이 누적 손익 곡선에서 잰다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(trades: pd.DataFrame, col: str = "net") -> pd.Series:
    """청산 시각 순 누적 손익 (거래당 투입 금액 = 1)."""
    t = trades.sort_values("exit_time", kind="stable")
    return pd.Series(t[col].to_numpy().cumsum(), index=t.exit_time.to_numpy())


def max_drawdown(curve: pd.Series) -> float:
    """누적 손익 곡선의 최대 낙폭 (고점 대비 하락폭, 투입 금액 배수). 시작점 0 을 고점에 포함한다."""
    if curve.empty:
        return 0.0
    v = np.concatenate([[0.0], curve.to_numpy()])
    return float((np.maximum.accumulate(v) - v).max())


def summarize(trades: pd.DataFrame, col: str = "net", days: int | None = None) -> dict:
    """매매 기록 → 성과 지표. days: 기간 중 선정 종목-일 수 (회전율 계산용)."""
    r = trades[col].to_numpy(dtype=float)
    n = len(r)
    if n == 0:
        return {"trades": 0}
    win, loss = r[r > 0], r[r <= 0]
    gross_win, gross_loss = win.sum(), -loss.sum()
    curve = equity_curve(trades, col)
    out = {
        "trades": n,
        "mean": r.mean(),                                        # 거래당 기대수익률
        "total": r.sum(),                                        # 누적 손익 (투입 금액 배수)
        "win_rate": len(win) / n,
        "avg_win": win.mean() if len(win) else 0.0,
        "avg_loss": loss.mean() if len(loss) else 0.0,
        "payoff": (win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.nan,
        "mdd": max_drawdown(curve),
        "avg_bars": trades.bars.mean(),
    }
    if days:
        out["trades_per_day"] = n / days
    return out


def mean_ci(r, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """거래당 평균 수익률의 부트스트랩 신뢰구간."""
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
