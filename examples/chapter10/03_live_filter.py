"""10장 실습 3 — LightGBM 필터를 실시간용으로 저장하고, 실시간 피처가 07장 데이터셋과 같은지 확인한다.

실행: uv run python examples/chapter10/03_live_filter.py [모의매매 날짜]   (기본 2026-11-02)
결과: models_out/live_filter.joblib
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.live.filter import LiveFilter
from vwap_trading.ml.dataset import FEATURES, features
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-11-02").date()
p = load_params("configs/strategy_a.toml")
df = pd.read_parquet("data/ml/candidates_A.parquet")

# 1. 피처 일치: 07장은 종목의 전 기간 분봉으로 지표를 계산했고, 실시간은 그날 봉만 가지고 있다.
#    10월 후보마다 "그날 신호 봉까지의 봉"만으로 다시 계산해 비교한다.
oct_ = df[df.date >= "2026-10-01"]
raw = {s: pd.read_parquet(f"data/raw/{s}_1m.parquet") for s in oct_.symbol.unique()}
diffs, nan_mismatch, t_ind, t_feat = [], 0, [], []
for r in oct_.itertuples():
    b = raw[r.symbol]
    b = b[(b.index.date == r.time.date()) & (b.index <= r.time)]       # 그날 신호 봉까지만
    t0 = time.perf_counter()
    x = compute_indicators(b, p.vwap)
    t1 = time.perf_counter()
    f = features(x).iloc[-1][FEATURES]
    t_feat.append(time.perf_counter() - t1)
    t_ind.append(t1 - t0)
    ref = pd.Series({k: getattr(r, k) for k in FEATURES}).astype(float)
    nan_mismatch += int((f.astype(float).isna() != ref.isna()).sum())
    diffs.append((f.astype(float) - ref).abs())
d = pd.DataFrame(diffs)
print(f"[피처 일치] 2026-10 후보 {len(d)}건 × 피처 {len(FEATURES)}개: 최대 차이 {np.nanmax(d.to_numpy()):.2e}, "
      f"값 없음(NaN) 위치 불일치 {nan_mismatch}칸")
print(f"  봉 확정 → 지표 {np.median(t_ind) * 1000:.1f}ms, 피처 {np.median(t_feat) * 1000:.1f}ms (중앙값)")

# 2. 모의매매 날짜 이전 데이터로 학습해 저장
flt = LiveFilter.train(df, day, label="y", mode="log")
Path("models_out").mkdir(exist_ok=True)
flt.save("models_out/live_filter.joblib")
print(f"\n[학습] {day} 이전 후보 {int((df.date < pd.Timestamp(day)).sum())}건 (마지막 {flt.trained_until}), "
      f"레이블 (1), 성공 평균 {flt.win:+.3%}, 실패 평균 {flt.loss:+.3%}")

# 3. 저장한 모델로 마지막 후보 하나를 판정해 본다
flt = LiveFilter.load("models_out/live_filter.joblib")
r = oct_.iloc[-1]
b = raw[r.symbol][(raw[r.symbol].index.date == r.time.date()) & (raw[r.symbol].index <= r.time)]
ok, info = flt(r.symbol, compute_indicators(b, p.vwap))
print(f"[판정 예] {r.symbol} {r.time:%Y-%m-%d %H:%M} → 승인 {ok} (mode={flt.mode}), {info}")
