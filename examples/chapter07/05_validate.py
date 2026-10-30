"""07장 실습 5 — 학습 데이터셋 검증: 피처 시점 검사, 건수·클래스 비율, 결측, 분포 변화.

실행: uv run python examples/chapter07/05_validate.py   (01 실행 후, 약 1분)
"""

import sys

import numpy as np
import pandas as pd

from vwap_trading.indicators.vwap import compute_indicators
from vwap_trading.ml.dataset import FEATURES, features

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/ml/candidates_A.parquet")
rng = np.random.default_rng(0)
ohlc = ["open", "high", "low", "close"]


def leaky(x):
    """일부러 미래 정보를 넣은 피처 (검사가 잡아내는지 보기 위한 것)."""
    f = features(x)
    f["next_ret"] = x.close.shift(-1) / x.close - 1
    return f


# 1. 피처 시점 검사: 신호 봉 이후 봉을 엉뚱한 값으로 바꿔도 신호 봉의 피처가 같은가
sample = df.sample(300, random_state=0)
changed = {"정상 피처": 0, "미래 정보 피처": 0}
raw_cache = {}
for r in sample.itertuples():
    raw = raw_cache.setdefault(r.symbol, pd.read_parquet(f"data/raw/{r.symbol}_1m.parquet"))
    bars = raw.loc[str(r.date.date())].astype(float)
    fut = bars.index > r.time
    k = int(fut.sum())
    noisy = bars.copy()
    noisy.loc[fut, ohlc] = bars.loc[fut, ohlc].mul(rng.uniform(0.5, 1.5, k), axis=0)
    noisy.loc[fut, "high"] = noisy.loc[fut, ohlc].max(axis=1)
    noisy.loc[fut, "low"] = noisy.loc[fut, ohlc].min(axis=1)
    noisy.loc[fut, "volume"] = rng.integers(1, 10**6, k)
    a, b = leaky(compute_indicators(bars)).loc[r.time], leaky(compute_indicators(noisy)).loc[r.time]
    changed["정상 피처"] += int(not np.allclose(a[FEATURES], b[FEATURES], equal_nan=True))
    changed["미래 정보 피처"] += int(not np.isclose(a.next_ret, b.next_ret))
print(f"[1] 피처 시점 검사 (표본 300건, 신호 봉 이후 봉 교란): 값이 바뀐 표본 {changed}")

# 2. 건수와 클래스 비율
df["month"] = df.date.dt.to_period("M").astype(str)
df["구간"] = np.select([df.date < "2026-03-01", df.date < "2026-09-01"], ["학습 전용", "Walk-forward 검증"], "최종 테스트")
m = df.groupby("month").agg(구간=("구간", "first"), 후보=("y", "size"), 종목=("symbol", "nunique"), y=("y", "mean"),
                            순평균=("net", "mean"))
sealed = m["구간"] == "최종 테스트"                          # 최종 테스트 구간은 건수만 보고 결과는 보지 않는다
m["y"] = m.y.map("{:.1%}".format).where(~sealed, "(봉인)")
m["순평균"] = m["순평균"].map("{:+.3%}".format).where(~sealed, "(봉인)")
print("\n[2] 월별 건수와 성공 비율")
print(m.to_string())
g = df.groupby("구간", sort=False).agg(후보=("y", "size"), y=("y", "mean"))
g["y"] = g.y.map("{:.1%}".format).where(g.index != "최종 테스트", "(봉인)")
print(g.to_string())

# 3. 결측·무한대·중복
num = df[FEATURES].to_numpy(dtype=float)
print(f"\n[3] 결측 {int(np.isnan(num).sum())}개, 무한대 {int(np.isinf(num).sum())}개, "
      f"(종목, 시각) 중복 {int(df.duplicated(['symbol', 'time']).sum())}건")

# 4. 분포 변화: 학습 전용 구간 대비 이후 구간의 중앙값 이동 (학습 구간 표준편차 단위)
base, later = df[df["구간"] == "학습 전용"], df[df["구간"] == "Walk-forward 검증"]
shift = ((later[FEATURES].median() - base[FEATURES].median()) / base[FEATURES].std()).sort_values(key=abs, ascending=False)
print("\n[4] 학습 전용(~2026-02) → 검증(2026-03~08) 중앙값 이동 (표준편차 단위), 큰 순서 5개:")
print({k: round(v, 2) for k, v in shift.head(5).items()})
