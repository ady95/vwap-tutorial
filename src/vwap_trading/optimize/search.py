"""Optuna 로 전략 A 의 진입 파라미터를 찾는다 (06장).

탐색 대상은 진입 쪽 6개(N·k·θ·m·δ·ρ)뿐이다. 손절·목표가·최대 보유·기준선 이탈·재진입 대기는
위험관리 규칙이므로 고정한다(수식 명세 v0.2 11절, 05-4 결정).

최종 테스트 구간(FINAL_TEST_START 이후)의 데이터는 탐색에 절대 넣지 않는다. 학습 데이터를 만드는
모든 함수가 check_before_final 을 거친다.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import optuna
import pandas as pd

from vwap_trading.backtest.costs import CostModel
from vwap_trading.indicators.vwap import VwapParams, compute_indicators
from vwap_trading.strategy.rules import StrategyParams, run

FINAL_TEST_START = date(2026, 9, 1)     # 이 날 이후는 09장에서 한 번만 쓰는 최종 테스트 구간

SPACE = {                               # (최소, 최대, 간격) — 수식 명세 v0.2 11절
    "N": (10, 90, 5),
    "k": (3, 15, 1),
    "theta": (0.02, 0.15, 0.005),
    "m": (5, 30, 1),
    "delta": (0.0, 1.0, 0.05),
    "rho": (1.0, 3.0, 0.1),
}


class FinalTestLeak(RuntimeError):
    """최종 테스트 구간의 데이터가 탐색에 들어가려 할 때."""


def check_before_final(days, limit: date | None = None) -> None:
    """학습 데이터가 limit(기본: 최종 테스트 시작일) 이전인지 확인한다.

    09장의 최종 평가만 limit 을 '평가하는 달의 첫날'로 넘긴다. 그래도 평가하는 달 이전 데이터만 쓴다는
    Walk-forward 원칙은 그대로다.
    """
    limit = limit or FINAL_TEST_START
    last = max(pd.Timestamp(d).date() for d in days)
    if last >= limit:
        raise FinalTestLeak(f"학습 데이터에 {limit} 이후가 들어 있다: {last}")


def suggest(trial: optuna.Trial, base: StrategyParams = StrategyParams()) -> StrategyParams:
    """trial 에서 진입 파라미터 6개를 뽑아 StrategyParams 를 만든다."""
    v = {}
    for name, (lo, hi, step) in SPACE.items():
        if isinstance(lo, int):
            v[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            v[name] = trial.suggest_float(name, lo, hi, step=step)
    vwap = replace(base.vwap, N=v["N"], k=v["k"], theta=v["theta"], m=v["m"])
    return replace(base, delta=v["delta"], rho=v["rho"], vwap=vwap)


def params_from(values: dict, base: StrategyParams = StrategyParams()) -> StrategyParams:
    vwap = replace(base.vwap, **{k: values[k] for k in ("N", "k", "theta", "m")})
    return replace(base, delta=values["delta"], rho=values["rho"], vwap=vwap)


def selected_bars(selections: pd.DataFrame, raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """선정된 종목-일의 1분봉만 모은다. 지표는 하루 단위로 계산하므로 다른 날은 필요 없다."""
    out = {}
    for s, g in selections.groupby("symbol"):
        bars = pd.read_parquet(f"{raw_dir}/{s}_1m.parquet")
        days = set(pd.to_datetime(g.tradable_from).dt.date)
        out[s] = bars[np.isin(bars.index.date, list(days))]
    return out


def evaluate(bars: dict[str, pd.DataFrame], selections: pd.DataFrame, p: StrategyParams,
             cost: CostModel = CostModel()) -> pd.DataFrame:
    """파라미터 p 로 지표를 다시 계산해 매매 기록(순수익 net 열 포함)을 만든다."""
    ind = {s: compute_indicators(b, p.vwap) for s, b in bars.items()}
    tr = run(ind, selections, p)
    tr["net"] = cost.net_returns(tr) if len(tr) else []
    return tr


def score(net: np.ndarray, kind: str = "lcb", min_trades: int = 30) -> float:
    """목적함수 값. 클수록 좋다.

    kind="total": 순수익 합계, "mean": 거래당 평균, "lcb": 평균 − 표준오차 (거래 수가 적으면 깎인다)
    거래가 min_trades 미만이면 -1 (−100%) 을 돌려 탐색에서 밀어낸다.
    """
    n = len(net)
    if n < min_trades:
        return -1.0
    if kind == "total":
        return float(net.sum())
    if kind == "mean":
        return float(net.mean())
    return float(net.mean() - net.std(ddof=1) / np.sqrt(n))


def optimize(bars, selections, n_trials: int = 100, kind: str = "lcb", min_trades: int = 30,
             seed: int = 42, storage: str | None = None, study_name: str | None = None,
             limit: date | None = None) -> optuna.Study:
    """학습 구간 선정 입력으로 Study 를 돌린다. limit(기본: 최종 테스트 시작일) 이후가 섞이면 FinalTestLeak."""
    check_before_final(selections.tradable_from, limit)

    def objective(trial):
        p = suggest(trial)
        tr = evaluate(bars, selections, p)
        net = tr.net.to_numpy(dtype=float)
        trial.set_user_attr("trades", int(len(net)))
        trial.set_user_attr("mean", float(net.mean()) if len(net) else None)
        trial.set_user_attr("total", float(net.sum()))
        return score(net, kind, min_trades)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                storage=storage, study_name=study_name, load_if_exists=False)
    study.optimize(objective, n_trials=n_trials)
    return study


def walk_forward_windows(months: list[str], train: int = 4, test: int = 1) -> list[tuple[list[str], list[str]]]:
    """월 목록 → (학습 월들, 검증 월들) 창. 최종 테스트 구간의 달은 넣지 않는다."""
    months = [m for m in months if pd.Period(m).start_time.date() < FINAL_TEST_START]
    return [(months[i:i + train], months[i + train:i + train + test])
            for i in range(0, len(months) - train - test + 1, test)]
