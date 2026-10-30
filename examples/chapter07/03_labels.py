"""07장 실습 3 — 레이블 점검: 성공(y=1)과 비용 차감 순이익의 관계, 같은 봉 동시 도달, 관측 기간.

실행: uv run python examples/chapter07/03_labels.py   (01 실행 후)
"""

import sys

import pandas as pd

from vwap_trading.indicators.vwap import compute_indicators

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
df = df[df.date < "2026-09-01"]                                 # 최종 테스트 구간은 보지 않는다
print(f"2025-10 ~ 2026-08 후보 {len(df):,}건")

# 1. y=1 은 "돈을 벌었다"인가
print("[1] 레이블 y 와 순이익(net > 0) 교차표")
print(pd.crosstab(df.y, df.net > 0, margins=True, rownames=["y"], colnames=["net>0"]).to_string())
t = df[df.reason == "target"]
print(f"  목표가 도달 {len(t):,}건: 비용 전 평균 {t.ret.mean():+.3%}, 순 평균 {t.net.mean():+.3%}, 순이익 비율 {(t.net > 0).mean():.1%}")
q = pd.qcut(t.tick_pct.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
print("  호가/가격 5분위별 목표가 도달 매매의 순 평균:",
      {k: f"{v:+.3%}" for k, v in t.groupby(q, observed=True).net.mean().items()})

# 2. 같은 봉에서 손절·목표가 동시 도달 (손절로 처리한 건수)
both = 0
for s, g in df[df.reason == "stop"].groupby("symbol"):
    x = compute_indicators(pd.read_parquet(f"data/raw/{s}_1m.parquet"))
    hi = x.high.reindex(g.exit_time).to_numpy()
    entry = x.open.reindex(g.entry_time).to_numpy()
    atr = x.atr_s.reindex(g.time).to_numpy()
    both += int((hi >= entry + 2.0 * atr).sum())
print(f"\n[2] 손절 {int((df.reason == 'stop').sum()):,}건 중 같은 봉에서 목표가에도 닿은 표본 {both}건 → y=0")

# 3. 관측 기간(진입~청산)의 길이와 겹침
print(f"\n[3] 관측 기간: 평균 {df.bars.mean():.1f}봉, 중앙값 {df.bars.median():.0f}봉, 최대 {df.bars.max()}봉")
over = 0
for _, g in df.groupby(["symbol", "date"]):
    s, e = g.entry_time.to_numpy(), g.exit_time.to_numpy()
    for i in range(len(g)):
        over += int(((s < e[i]) & (e > s[i])).sum() > 1)
print(f"  같은 종목-일에서 관측 기간이 다른 표본과 겹치는 표본 {over:,}건 ({over / len(df):.1%})")
print(f"  날짜를 넘긴 관측 기간 {(df.exit_time.dt.date != df.time.dt.date).sum()}건")

# 4. 완벽한 필터의 상한: 정답을 미리 알고 고른다면 거래당 순수익은 얼마인가
print(f"\n[4] 전체 후보 {len(df):,}건 순 평균 {df.net.mean():+.3%}")
print(f"  y=1 만 정확히 고르면  {int(df.y.sum()):,}건, 순 평균 {df[df.y == 1].net.mean():+.3%}")
print(f"  순이익(net>0)만 고르면 {int((df.net > 0).sum()):,}건, 순 평균 {df[df.net > 0].net.mean():+.3%}")
