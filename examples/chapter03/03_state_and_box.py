"""03장 실습 3 — 정규화 기울기로 상태(FLAT/UP/DOWN)를 판정하고 박스 중심선을 확인한다.

실행: uv run python examples/chapter03/03_state_and_box.py
"""

import sys

import pandas as pd

from vwap_trading.indicators.vwap import FLAT, UP, VwapParams, compute_indicators

sys.stdout.reconfigure(encoding="utf-8")
p = VwapParams()
manifest = pd.read_csv("data/raw/_manifest.csv", dtype={"symbol": str})
symbols = manifest.loc[manifest.days >= 250, "symbol"]

shares, seg_lens, first_flat, flat_to_up = [], [], [], 0
stock_days = 0
for s in symbols:
    x = compute_indicators(pd.read_parquet(f"data/raw/{s}_1m.parquet"), p)
    hhmm = x.index.hour * 100 + x.index.minute
    shares.append(x.loc[(hhmm >= 930) & (hhmm <= 1445), "state"].value_counts(normalize=True, dropna=False))
    seg = (x.state != x.state.shift()).cumsum()
    flat = x.state == FLAT
    seg_lens.append(flat.groupby(seg).sum()[flat.groupby(seg).first()])
    first_flat.append(x[flat].groupby(x[flat].index.date).bar_no.min())
    flat_to_up += int(((x.state.shift() == FLAT) & (x.state == UP)).sum())
    stock_days += x.index.normalize().nunique()

share = pd.concat(shares, axis=1).mean(axis=1).rename({0.0: "FLAT", 1.0: "UP", -1.0: "DOWN"})
print(f"{len(symbols)}종목, {stock_days}종목-일 (파라미터 N={p.N}, k={p.k}, θ={p.theta}, m={p.m})")
print("09:30~14:45 상태 비중:", {str(k): round(v, 3) for k, v in share.items()})
lens = pd.concat(seg_lens)
print(f"FLAT 구간 {len(lens)}개: 길이 중앙값 {lens.median():.0f}봉, 평균 {lens.mean():.1f}봉, 최대 {lens.max()}봉")
ff = pd.concat(first_flat)
print(f"하루 첫 FLAT 봉 순번: 최소 {ff.min()} (워밍업 계산값 {p.warmup_end}), 중앙값 {ff.median():.0f}")
print(f"FLAT→UP 전환: {flat_to_up}회, 종목-일당 {flat_to_up / stock_days:.2f}회")
