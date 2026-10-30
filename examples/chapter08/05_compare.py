"""08장 실습 5 — AI 필터 적용 전후 비교: 신뢰구간, 거른 매매와 통과한 매매, 월별.

실행: uv run python examples/chapter08/05_compare.py   (04 실행 후)
"""

import sys

import numpy as np
import pandas as pd

from vwap_trading.backtest.metrics import mean_ci, summarize

sys.stdout.reconfigure(encoding="utf-8")
t = pd.read_parquet("reports/c_trades.parquet")
A = t[t.variant == "A (필터 없음)"]
rng = np.random.default_rng(0)

print("[1] 선정 종목-일 2026-03~08, 순수익 기준")
for name, g in t.groupby("variant", sort=False):
    if g.empty:
        continue
    m = summarize(g)
    line = f"  {name:<22} {m['trades']:3d}건 평균 {m['mean']:+.3%}"
    if len(g) >= 10:
        lo, hi = mean_ci(g.net)
        d = [rng.choice(g.net, len(g)).mean() - rng.choice(A.net, len(A)).mean() for _ in range(2000)]
        line += f" [{lo:+.3%}, {hi:+.3%}] 승률 {m['win_rate']:.1%} 누적 {m['total']:+.2f}"
        if name != "A (필터 없음)":
            line += f" | A 대비 차이 [{np.quantile(d, 0.025):+.3%}, {np.quantile(d, 0.975):+.3%}]"
    else:
        line += f" (표본이 적어 신뢰구간 생략) 누적 {m['total']:+.2f}"
    print(line)
print("  C(1) 기대 순수익 규칙: 매매 0건 (모든 후보 거부)")

# 2. 필터가 A 의 매매 중 무엇을 걸렀나: A 의 127건을 각 모델의 판단으로 나눈다
print("\n[2] A 의 매매를 모델 판단으로 나누면 (같은 매매, 같은 순수익)")
for label, name in (("y", "(1)"), ("y_net", "(2)")):
    pred = pd.read_parquet(f"reports/ml_pred_{label}.parquet")[["symbol", "time", "top30", "approve", "p_raw"]]
    m = A.merge(pred, left_on=["symbol", "signal_time"], right_on=["symbol", "time"], how="left")
    for col, rule in (("top30", "상위 30%"), ("approve", "기대 순수익")):
        ok, no = m[m[col] == True], m[m[col] != True]  # noqa: E712
        ok_mean = f"{ok.net.mean():+.3%}" if len(ok) else "-"
        print(f"  {name} {rule:<6}: 승인 {len(ok):3d}건 {ok_mean} | 거부 {len(no):3d}건 {no.net.mean():+.3%}")

# 3. 월별
print("\n[3] 월별 매매 수와 순 평균")
mon = t.assign(month=t.exit_time.dt.to_period("M")).groupby(["month", "variant"], sort=False).net.agg(["size", "mean"])
tab = mon["size"].unstack().fillna(0).astype(int)
print(tab.to_string())
