"""06장 실습 1 — 첫 Optuna Study: 학습 구간 하나에서 30회 탐색하고, 기록을 저장·재현한다.

실행: uv run python examples/chapter06/01_first_study.py
"""

import sys
from pathlib import Path

import optuna
import pandas as pd

from vwap_trading.optimize.search import optimize, selected_bars, walk_forward_windows

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
month = sel.tradable_from.dt.to_period("M").astype(str)
train, _ = walk_forward_windows(sorted(month.unique()))[0]
tr_sel = sel[month.isin(train)]
bars = selected_bars(tr_sel)
print(f"학습 구간 {train[0]}~{train[-1]}: 선정 {len(tr_sel)}종목-일")

Path("reports").mkdir(exist_ok=True)
db = Path("reports/optuna_intro.db")
db.unlink(missing_ok=True)
storage = f"sqlite:///{db.as_posix()}"
study = optimize(bars, tr_sel, n_trials=30, storage=storage, study_name="intro")

df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
df.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in df.columns]
print("\n상위 5개 trial")
print(df.sort_values("value", ascending=False).head(5).round(4).to_string(index=False))
print(f"\n최고 trial #{study.best_trial.number}: 점수 {study.best_value:+.4%}, {study.best_params}")

# 같은 seed 로 다시 돌리면 같은 결과가 나오는가
again = optimize(bars, tr_sel, n_trials=30)
print(f"같은 seed 재실행: 최고 파라미터 일치 {again.best_params == study.best_params}")

# 저장된 기록 다시 읽기
loaded = optuna.load_study(study_name="intro", storage=storage)
print(f"저장된 Study 다시 읽기: trial {len(loaded.trials)}개, 최고 점수 {loaded.best_value:+.4%}")
