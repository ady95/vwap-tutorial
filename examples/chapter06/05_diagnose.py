"""06장 실습 5 — 최적화 전후 비교(B 대 A)와 과최적화 진단. 04_walk_forward.py 를 먼저 실행한다.

실행: uv run python examples/chapter06/05_diagnose.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from vwap_trading.backtest.metrics import mean_ci, summarize
from vwap_trading.optimize.search import SPACE
from vwap_trading.reporting.plotting import COLORS, use_korean_font

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
params = pd.read_csv("reports/wf_params.csv")
B = pd.read_csv("reports/wf_trades_B.csv", parse_dates=["exit_time"])
A = pd.read_csv("reports/wf_trades_A.csv", parse_dates=["exit_time"])
storage = "sqlite:///reports/optuna_wf.db"

# 1. 검증 구간 성과: B(창마다 최적화) 대 A(초기값 고정)
print("[1] 검증 구간 2026-03~08 (순수익 기준)")
for name, t in (("A 초기값", A), ("B 최적화", B)):
    m = summarize(t)
    lo, hi = mean_ci(t.net)
    print(f"  {name}: {m['trades']}건, 평균 {m['mean']:+.3%} [{lo:+.3%}, {hi:+.3%}], 승률 {m['win_rate']:.1%}, "
          f"PF {m['profit_factor']:.2f}, 누적 {m['total']:+.2f}, MDD {m['mdd']:.2f}")
rng = np.random.default_rng(0)
d = [rng.choice(B.net, len(B)).mean() - rng.choice(A.net, len(A)).mean() for _ in range(2000)]
print(f"  B − A 평균 차이 95% 구간: [{np.quantile(d, 0.025):+.3%}, {np.quantile(d, 0.975):+.3%}]")

# 2. 학습 성과와 검증 성과의 차이
print("\n[2] 창별 학습 → 검증 (거래당 순 평균)")
gap = params[["window", "train", "test", "train_trades", "train_mean", "test_trades_B", "test_mean_B",
              "test_trades_A", "test_mean_A"]].copy()
print(gap.to_string(index=False, float_format=lambda v: f"{v:+.3%}"))
print(f"  학습 평균 {gap.train_mean.mean():+.3%} → 검증 평균 {gap.test_mean_B.mean():+.3%} "
      f"(B 가 A 보다 나은 창 {(gap.test_mean_B > gap.test_mean_A).sum()}/{len(gap)})")

# 3. 최고 trial 은 운이 좋았던 trial 이기도 하다: 학습 점수 분포
print("\n[3] 창별 학습 점수 분포 (100 trial, 최소 거래 미달 제외)")
for i in params.window:
    st = optuna.load_study(study_name=f"wf_{i}", storage=storage)
    v = np.array([t.value for t in st.trials if t.value is not None and t.value > -1])
    print(f"  창 {i}: 유효 {len(v)}개, 중앙값 {np.median(v):+.3%}, 상위 10% {np.quantile(v, 0.9):+.3%}, 최고 {v.max():+.3%}")

# 4. 파라미터 안정성: 창마다 최적값이 얼마나 바뀌는가
print("\n[4] 창별 최적 파라미터")
cols = list(SPACE)
print(params[["test"] + cols].to_string(index=False))
for c in cols:
    lo, hi, _ = SPACE[c]
    width = (params[c].max() - params[c].min()) / (hi - lo)
    print(f"  {c}: {params[c].min()}~{params[c].max()} (탐색 범위의 {width:.0%})")

# 5. 파라미터 중요도 (창별 평균)
imp = pd.DataFrame([optuna.importance.get_param_importances(optuna.load_study(study_name=f"wf_{i}", storage=storage))
                    for i in params.window]).mean().sort_values(ascending=False)
print("\n[5] 파라미터 중요도 (fANOVA, 6개 창 평균):", {k: round(v, 2) for k, v in imp.items()})

# 그림: 창별 학습·검증 평균
use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(gap))
ax.bar(x - 0.27, gap.train_mean * 100, width=0.27, color="#bbbbbb", label="B 학습 구간")
ax.bar(x, gap.test_mean_B * 100, width=0.27, color=COLORS["rvwap"], label="B 검증 구간")
ax.bar(x + 0.27, gap.test_mean_A * 100, width=0.27, color=COLORS["mvwap"], label="A 초기값 검증 구간")
ax.set_xticks(x, gap.test)
ax.axhline(0, color="#888888", lw=0.8)
ax.set_ylabel("거래당 순수익률 (%)")
ax.set_title("Walk-forward: 학습 구간에서 찾은 파라미터의 다음 달 성과", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/06-5-walk-forward.png", dpi=110)
print("그림: reports/06-5-walk-forward.png")
