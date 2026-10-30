"""09장 실습 1 — S0·A·A′·A_B·B·C·D 를 같은 조건에서 월별 Walk-forward 로 실행한다.

평가 달: P1 = 2026-03~08 (06·08장 개발 때 이미 본 검증 달), P2 = 2026-09~10 (최종 테스트, 이 실행에서 처음 연다)
모든 달은 같은 규칙을 따른다: 그 달의 파라미터·모델은 그 달 이전 데이터로만 정한다.
  B : 직전 4개월 선정 입력으로 Optuna 100 trial (06장, 평균−표준오차·최소 30건)
  C : 그 달 이전 유니버스 후보로 LightGBM (08장, 확장 창·엠바고 1일·마지막 2개월 보정)
  D : B 의 그 달 파라미터로 유니버스 후보를 다시 만들고 C 와 같은 방식으로 필터
주 비교는 레이블 (1)·기대 순수익 규칙. (2) 레이블과 상위 30% 규칙은 민감도 분석용.

실행: uv run python examples/chapter09/01_run_experiments.py   (약 25분)
결과: reports/ch9/trades.parquet, reports/ch9/b_params.csv, reports/ch9/manifest.json
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from vwap_trading.backtest.engine import backtest
from vwap_trading.indicators.vwap import VwapParams, compute_indicators
from vwap_trading.ml.dataset import build
from vwap_trading.ml.model import walk_forward
from vwap_trading.optimize.search import optimize, params_from, selected_bars
from vwap_trading.strategy.rules import StrategyParams

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = Path("reports/ch9")
OUT.mkdir(parents=True, exist_ok=True)
P1 = [str(p) for p in pd.period_range("2026-03", "2026-08", freq="M")]
P2 = ["2026-09", "2026-10"]
KAPPA = 2.0                     # A_B 의 박스 폭 (07-2: 2026-02 이전 box_range 중앙값 1.91 근처, 실행 전 고정)

sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
month_of = sel.tradable_from.dt.to_period("M").astype(str)
uni = pd.read_csv("data/sample/universe_20261030.csv", dtype={"symbol": str}, encoding="utf-8-sig").symbol
base_ds = pd.read_parquet("data/ml/candidates_A.parquet")       # 07장: 초기값 A 의 유니버스 후보

FIXED = {
    "S0": StrategyParams(strategy="S0"),
    "A": StrategyParams(),
    "A′": StrategyParams(vwap=VwapParams(box="anchored")),
    "A_B": StrategyParams(vwap=VwapParams(kappa=KAPPA)),
}


def approver(pred, col):
    ok = set(zip(pred.symbol[pred[col]], pred.time[pred[col]]))

    def approve(symbol, t):
        return (symbol, t) in ok
    return approve


def run_month(m: str, b_params: StrategyParams, rows: list):
    te = sel[month_of == m]
    bars = selected_bars(te)
    final = m in P2
    period = "P2" if final else "P1"

    counts = {}

    def add(name, p, approve=None):
        ind = {s: compute_indicators(b, p.vwap) for s, b in bars.items()}
        t = backtest(ind, te, p, approve=approve)
        rows.append(t.assign(strategy=name, period=period, month=m))
        counts[name] = len(t)

    for name, p in FIXED.items():
        add(name, p)
    add("B", b_params)

    for label in ("y", "y_net"):
        pred = walk_forward(base_ds, label, [m], final=final)
        tag = "(1)" if label == "y" else "(2)"
        add(f"C{tag} 규칙", FIXED["A"], approver(pred, "approve"))
        if label == "y":
            add(f"C{tag} 상위30%", FIXED["A"], approver(pred, "top30"))

    ds = build(uni, b_params)                                     # D: B 파라미터로 만든 유니버스 후보
    pred = walk_forward(ds, "y", [m], final=final)
    add("D(1) 규칙", b_params, approver(pred, "approve"))
    add("D(1) 상위30%", b_params, approver(pred, "top30"))
    return counts


commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "examples"], capture_output=True,
                            text=True).stdout.strip())
print(f"코드 커밋 {commit}{' (커밋되지 않은 변경 있음)' if dirty else ''}, 시작 {datetime.now():%Y-%m-%d %H:%M}")

all_months = sorted(month_of.unique())
months = sys.argv[1:] or P1 + P2                                  # 인자로 달을 주면 그 달만 (점검용)
rows, b_rows = [], []
for m in months:
    t0 = time.perf_counter()
    i = all_months.index(m)
    train = all_months[i - 4:i]                                   # B: 직전 4개월
    tr_sel = sel[month_of.isin(train)]
    limit = pd.Period(m).start_time.date()                        # 그 달 이전 데이터만
    study = optimize(selected_bars(tr_sel), tr_sel, n_trials=100, limit=limit)
    b = params_from(study.best_params)
    b_rows.append({"month": m, "train": f"{train[0]}~{train[-1]}", **study.best_params,
                   "train_trades": study.best_trial.user_attrs["trades"]})
    counts = run_month(m, b, rows)
    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    print(f"[{m}] {time.perf_counter() - t0:.0f}초 | B 학습 {train[0]}~{train[-1]} | 매매 수: {summary}")

trades = pd.concat(rows, ignore_index=True)
if sys.argv[1:]:
    print("점검 실행이라 결과를 저장하지 않음")
    sys.exit()
trades.to_parquet(OUT / "trades.parquet")
pd.DataFrame(b_rows).to_csv(OUT / "b_params.csv", index=False)
json.dump({"commit": commit, "dirty": dirty, "finished": f"{datetime.now():%Y-%m-%d %H:%M}", "P1": P1, "P2": P2,
           "kappa": KAPPA, "optuna_trials": 100, "strategy_A": asdict(FIXED["A"])},
          open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(f"저장: {OUT}/trades.parquet ({len(trades)}건), b_params.csv, manifest.json")
