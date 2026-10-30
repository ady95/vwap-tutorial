"""09장 실습 2 — 전략별 성과표와 사전 등록한 기준에 따른 판단.

실행: uv run python examples/chapter09/02_compare.py   (01 실행 후)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vwap_trading.backtest.metrics import mean_ci, summarize
from vwap_trading.reporting.plotting import use_korean_font

sys.stdout.reconfigure(encoding="utf-8")
t = pd.read_parquet("reports/ch9/trades.parquet")
man = json.load(open("reports/ch9/manifest.json", encoding="utf-8"))
sel = pd.read_csv("data/sample/selections.csv", parse_dates=["tradable_from"])
days = sel.tradable_from.dt.to_period("M").astype(str).value_counts()
ORDER = ["S0", "A", "A′", "A_B", "B", "C(1) 규칙", "D(1) 규칙", "C(2) 규칙", "C(1) 상위30%", "D(1) 상위30%"]
MAIN = {"S0", "A′", "A_B", "B", "C(1) 규칙", "D(1) 규칙"}
print(f"실험 코드 커밋 {man['commit']} (변경 없음: {not man['dirty']}), 완료 {man['finished']}")


def diff_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.quantile(d, [0.025, 0.975])


def table(sub, months):
    n_days = int(days.reindex(months).sum())
    rows = []
    for s in ORDER:
        g = sub[sub.strategy == s]
        if g.empty:
            rows.append({"전략": s, "매매": 0})
            continue
        m = summarize(g, days=n_days)
        lo, hi = mean_ci(g.net) if len(g) >= 2 else (np.nan, np.nan)
        rows.append({"전략": s, "매매": m["trades"], "하루 매매": round(m["trades_per_day"], 2),
                     "비용 전": f"{g.ret.mean():+.3%}", "순 평균": f"{m['mean']:+.3%}",
                     "95% 구간": f"[{lo:+.2%}, {hi:+.2%}]", "승률": f"{m['win_rate']:.1%}",
                     "손익비": round(m["payoff"], 2) if np.isfinite(m["payoff"]) else "-",
                     "PF": round(m["profit_factor"], 2) if np.isfinite(m["profit_factor"]) else "-",
                     "누적": round(m["total"], 2), "MDD": round(m["mdd"], 2)})
    return pd.DataFrame(rows).astype(object).fillna("-")


for label, months in (("P1+P2 (2026-03~10, 8개월)", man["P1"] + man["P2"]), ("P1 (2026-03~08)", man["P1"]),
                      ("P2 최종 테스트 (2026-09~10)", man["P2"])):
    sub = t[t.month.isin(months)]
    print(f"\n[{label}] 선정 {int(days.reindex(months).sum())}종목-일")
    print(table(sub, months).to_string(index=False))

# 사전 등록 기준에 따른 판단 (8개월 합계, A 대비)
A = t[t.strategy == "A"]
print("\n[판단] 거래당 순수익의 A 대비 차이 (8개월, 부트스트랩 95%)")
verdicts = []
for s in ORDER:
    if s == "A":
        continue
    g = t[t.strategy == s]
    p2 = g[g.period == "P2"].net.mean() - A[A.period == "P2"].net.mean() if (g.period == "P2").any() else np.nan
    if len(g) < 30:
        v, rng = "판단 불가 (매매 30건 미만)", ""
    else:
        lo, hi = diff_ci(g.net.to_numpy(), A.net.to_numpy())
        v = "추가 효과 있음" if lo > 0 else ("오히려 나쁨" if hi < 0 else "추가 효과 없음")
        rng = f"[{lo:+.3%}, {hi:+.3%}]"
    mon = pd.DataFrame({"s": g.groupby("month").net.mean(), "a": A.groupby("month").net.mean()}).dropna()
    abs_lo = mean_ci(g.net)[0] if len(g) >= 2 else np.nan
    print(f"  {s:<14}{'(주)' if s in MAIN else '(민감도)':<7} {len(g):3d}건 {rng:<22} {v:<20} "
          f"P2 차이 {'-' if np.isnan(p2) else f'{p2:+.3%}'} | A보다 나은 달 {(mon.s > mon.a).sum()}/{len(mon)} | "
          f"실전 후보 {'예' if abs_lo > 0 else '아니요'}")
a_lo = mean_ci(A.net)[0]
print(f"  A 자체: {len(A)}건, 실전 후보 {'예' if a_lo > 0 else '아니요'} (95% 구간 아래 끝 {a_lo:+.3%})")

# 그림: 거래당 순수익과 95% 구간
use_korean_font()
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, s in enumerate(ORDER):
    for j, (per, color) in enumerate((("P1", "#888888"), ("P2", "#d62728"))):
        g = t[(t.strategy == s) & (t.period == per)].net
        if len(g) >= 2:
            lo, hi = mean_ci(g)
            ax.errorbar(i + (j - 0.5) * 0.25, g.mean() * 100, yerr=[[(g.mean() - lo) * 100], [(hi - g.mean()) * 100]],
                        fmt="o", color=color, capsize=3, label=per if i == 1 else None)
        elif len(g) == 1:
            ax.plot(i + (j - 0.5) * 0.25, g.mean() * 100, "x", color=color)
ax.axhline(0, color="#888888", lw=0.8)
ax.set_xticks(range(len(ORDER)), ORDER, rotation=30, fontsize=8)
ax.set_ylabel("거래당 순수익률 (%)")
ax.set_title("전략별 거래당 순수익과 95% 신뢰구간 (P1: 2026-03~08, P2: 2026-09~10)", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/09-3-strategies.png", dpi=110)
print("그림: reports/09-3-strategies.png")
