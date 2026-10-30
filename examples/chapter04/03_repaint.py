"""04장 실습 3 — 미확정 봉이 만드는 착시와 리페인팅 검사.

실행: uv run python examples/chapter04/03_repaint.py
"""

import sys

import pandas as pd

from vwap_trading.indicators.vwap import FLAT, compute_indicators, load_indicators
from vwap_trading.strategy.rules import StrategyParams, entry_allowed, raw_signal, run

sys.stdout.reconfigure(encoding="utf-8")
sel = pd.read_csv("data/sample/selections.csv", dtype={"symbol": str}, parse_dates=["tradable_from"])
p = StrategyParams()
ind = load_indicators(sel.symbol, p.vwap)
days = [ind[r.symbol].loc[str(r.tradable_from.date())] for r in sel.itertuples()]

# 1. 봉 도중에는 조건을 만족했지만, 봉이 끝났을 때는 아닌 신호
#    현재 봉의 가격 조건만 종가 대신 고가로 판정해 근사한다 (직전 봉은 이미 확정된 값 그대로).
def price_line(x, sp):
    return x.svwap if sp.strategy == "S0" else x.mvwap.shift() + sp.delta * x.atr_s


for name, sp in (("S0", StrategyParams(strategy="S0")), ("A", p)):
    touched = closed = 0
    for x in days:
        sig = raw_signal(x, sp) & entry_allowed(x, sp)
        if sp.strategy == "S0":                                   # 가격 조건을 뺀 나머지 조건
            others = (x.close.shift() <= x.svwap.shift()) & (x.rvol >= sp.rho)
        else:
            others = (x.state.shift() == FLAT) & (x.slope >= sp.vwap.theta) & (x.rvol >= sp.rho)
        touch = (x.high > price_line(x, sp)) & others & entry_allowed(x, sp)
        closed += int(sig.sum())
        touched += int((touch | sig).sum())
    print(f"[{name}] 봉 도중 한 번이라도 신호: {touched:,}  봉 확정 시 신호: {closed:,}  "
          f"사라진 신호 {touched - closed:,}건 ({1 - closed / touched:.0%})")

# 2. 신호 봉 종가에 샀다고 가정하면 (체결 불가능한 가격) 수익률이 얼마나 부풀려지나
tr = run(ind, sel, p)
close_t = pd.Series([ind[s].close[t] for s, t in zip(tr.symbol, tr.signal_time)], index=tr.index)
fake = tr.exit / close_t - 1
print(f"\n[A] {len(tr)}건 평균 수익률 (비용 전): 다음 봉 시가 진입 {tr.ret.mean():+.3%}, "
      f"신호 봉 종가 진입 {fake.mean():+.3%}")
print(f"    다음 봉 시가 − 신호 봉 종가: 평균 {(tr.entry / close_t - 1).mean():+.3%}, "
      f"시가가 더 높음 {(tr.entry > close_t).mean():.0%}, 같음 {(tr.entry == close_t).mean():.0%}, 더 낮음 {(tr.entry < close_t).mean():.0%}")

# 3. 봉 t 까지만 넣고 다시 계산해도 신호가 같은가 (리페인팅 검사)
checked = mismatch = 0
for r in sel.head(20).itertuples():                 # 오래 걸려 20개만 검사 (약 2분)
    raw = pd.read_parquet(f"data/raw/{r.symbol}_1m.parquet")
    bars = raw.loc[str(r.tradable_from.date())]
    full = raw_signal(compute_indicators(bars, p.vwap), p)
    for t in range(p.vwap.warmup_end, len(full)):
        part = raw_signal(compute_indicators(bars.iloc[: t + 1], p.vwap), p)
        checked += 1
        mismatch += int(part.iloc[-1] != full.iloc[t])
print(f"\n리페인팅 검사: 종목-일 20개, {checked:,}개 시점 재계산, 불일치 {mismatch}건")

# 4. 봉 시각은 구간의 시작인가 끝인가 — 15:20~15:30 동시호가 구간의 봉
special = {pd.Timestamp("2025-11-13").date(), pd.Timestamp("2026-01-02").date()}
count, total = pd.Series(0, index=["15:19", "15:20", "15:21", "15:29", "15:30"]), 0
for s in dict.fromkeys(sel.symbol):
    raw = pd.read_parquet(f"data/raw/{s}_1m.parquet")
    raw = raw[~pd.Index(raw.index.date).isin(special)]
    labels = pd.Series(raw.index.strftime("%H:%M"))
    total += raw.index.normalize().nunique()
    count += labels[labels.isin(count.index)].value_counts().reindex(count.index, fill_value=0)
print(f"\n봉 시각별 존재하는 종목-일 (예외일 제외 {total:,}종목-일)")
print(count.to_string())
