"""06장 실습 4 — Walk-forward 최적화: 학습 4개월로 찾은 파라미터를 다음 1개월에 고정 적용한다.

실행: uv run python examples/chapter06/04_walk_forward.py   (약 10분)
결과: reports/wf_params.csv (창별 파라미터), reports/wf_trades_B.csv·wf_trades_A.csv (검증 구간 매매)
      reports/optuna_wf.db (창별 Study 기록)
"""

import sys
import time
from pathlib import Path

import optuna
import pandas as pd

from vwap_trading.optimize.search import (FinalTestLeak, evaluate, optimize, params_from, selected_bars,
                                          walk_forward_windows)
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = 100
Path("reports").mkdir(exist_ok=True)
db = Path("reports/optuna_wf.db")
db.unlink(missing_ok=True)                       # 매번 새로 시작 (재현성)
storage = f"sqlite:///{db.as_posix()}"

sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
month = sel.tradable_from.dt.to_period("M").astype(str)
windows = walk_forward_windows(sorted(month.unique()))

# 최종 테스트 구간을 학습에 넣으려 하면 막힌다
try:
    optimize({}, sel, n_trials=1)
except FinalTestLeak as e:
    print(f"[차단 확인] {e}")

rows, trades_b, trades_a = [], [], []
for i, (train, test) in enumerate(windows):
    t0 = time.perf_counter()
    tr_sel, te_sel = sel[month.isin(train)], sel[month.isin(test)]
    study = optimize(selected_bars(tr_sel), tr_sel, n_trials=N_TRIALS, storage=storage, study_name=f"wf_{i}")
    best = params_from(study.best_params)
    te_bars = selected_bars(te_sel)
    b = evaluate(te_bars, te_sel, best).assign(window=i)
    a = evaluate(te_bars, te_sel, StrategyParams()).assign(window=i)
    trades_b.append(b)
    trades_a.append(a)
    rows.append({"window": i, "train": f"{train[0]}~{train[-1]}", "test": test[0], **study.best_params,
                 "train_trades": study.best_trial.user_attrs["trades"],
                 "train_mean": study.best_trial.user_attrs["mean"], "train_score": study.best_value,
                 "test_trades_B": len(b), "test_mean_B": b.net.mean(), "test_trades_A": len(a), "test_mean_A": a.net.mean()})
    r = rows[-1]
    print(f"[창 {i}] 학습 {r['train']} → 검증 {r['test']} ({time.perf_counter() - t0:.0f}초) "
          f"학습 {r['train_trades']}건 {r['train_mean']:+.3%} | 검증 B {len(b)}건 {b.net.mean():+.3%}, "
          f"A {len(a)}건 {a.net.mean():+.3%}")

params = pd.DataFrame(rows)
params.to_csv("reports/wf_params.csv", index=False)
pd.concat(trades_b).to_csv("reports/wf_trades_B.csv", index=False)
pd.concat(trades_a).to_csv("reports/wf_trades_A.csv", index=False)
print("\n창별 최적 파라미터")
print(params[["window", "test", "N", "k", "theta", "m", "delta", "rho"]].to_string(index=False))
B, A = pd.concat(trades_b), pd.concat(trades_a)
print(f"\n검증 구간 합계 (2026-03~08): B {len(B)}건 순 평균 {B.net.mean():+.3%} 누적 {B.net.sum():+.2f} | "
      f"A {len(A)}건 순 평균 {A.net.mean():+.3%} 누적 {A.net.sum():+.2f}")
