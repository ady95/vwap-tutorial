"""LightGBM 학습 데이터 (07장, 수식 명세 v0.2 12절).

- 표본: 유니버스 전 종목·전 거래일의 전략 A 매수 후보 (선정 여부와 무관). 포지션·재진입 대기는
  매매 단계의 조건이므로 표본에서는 보지 않는다.
- 피처: 신호 봉 t 가 확정된 시점까지의 값만 쓴다.
- 레이블: t+1 시가 진입을 가정한 Triple Barrier. 장벽은 전략의 청산 규칙과 같다
  (손절 λ·ATR*, 목표 τ·ATR*, 최대 보유 h봉, 장 마감 청산). 기준선 이탈은 쓰지 않는다.
  y     : 목표가가 먼저 (시간·장 마감 청산이면 순수익 > 0)   — 수식 명세 v0.2 12.1
  y_net : 비용 차감 순수익 > 0                            — 08장 비교용
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vwap_trading.backtest.costs import CostModel, tick_size
from vwap_trading.indicators.vwap import FLAT, compute_indicators
from vwap_trading.strategy.rules import StrategyParams, day_clock, entry_allowed, raw_signal

FEATURES = [
    "breakout",      # (종가 − 직전 박스 중심선) / ATR*    : 박스를 얼마나 벗어났나
    "dist_svwap",    # (종가 − 세션 VWAP) / ATR*
    "dist_rvwap",    # (종가 − 롤링 VWAP) / ATR*
    "slope",         # 정규화 기울기
    "rvol",          # 상대 거래량
    "atr_pct",       # ATR* / 종가                       : 변동성 (가격 대비)
    "tick_pct",      # 호가단위 / 종가                     : 비용의 크기
    "flat_len",      # 직전 횡보 구간 길이 (봉)
    "box_range",     # 직전 횡보 구간의 (최고가 − 최저가) / ATR*
    "bar_range",     # 신호 봉 (고가 − 저가) / ATR*
    "bar_body",      # 신호 봉 (종가 − 시가) / ATR*
    "upper_wick",    # 신호 봉 (고가 − 종가) / ATR*
    "ret_5",         # 최근 5봉 수익률
    "ret_day",       # 당일 시가 대비 수익률
    "from_high",     # (당일 최고가 − 종가) / ATR*
    "from_low",      # (종가 − 당일 최저가) / ATR*
    "minutes",       # 개장 후 경과 분
]


def features(x: pd.DataFrame) -> pd.DataFrame:
    """compute_indicators 결과 → 봉마다 피처. 모든 값은 그 봉 확정 시점까지의 정보만 쓴다."""
    a = x.atr_s
    day = x.index.normalize()
    g = x.groupby(day)
    f = pd.DataFrame(index=x.index)
    f["breakout"] = (x.close - x.mvwap.shift()) / a
    f["dist_svwap"] = (x.close - x.svwap) / a
    f["dist_rvwap"] = (x.close - x.rvwap) / a
    f["slope"] = x.slope
    f["rvol"] = x.rvol
    f["atr_pct"] = a / x.close
    f["tick_pct"] = x.close.map(tick_size) / x.close

    # 직전 횡보 구간: 상태가 바뀔 때마다 구간 번호를 매기고, 한 봉 전 구간이 FLAT 이면 그 구간을 본다
    seg = (x.state != x.state.shift()).cumsum()
    flat = x.state == FLAT
    seg_len = flat.groupby(seg).transform("sum")
    seg_hi = x.high.where(flat).groupby(seg).transform("max")
    seg_lo = x.low.where(flat).groupby(seg).transform("min")
    d = day.to_numpy()
    same_day = np.r_[False, d[1:] == d[:-1]]
    ended = flat.shift(fill_value=False).to_numpy() & ~flat.to_numpy() & same_day   # 한 봉 전에 횡보가 끝남
    f["flat_len"] = np.where(ended, seg_len.shift(), np.nan)
    f["box_range"] = np.where(ended, (seg_hi.shift() - seg_lo.shift()) / a, np.nan)
    # transform 은 구간 전체로 계산하지만, 이미 끝난 구간만 쓰므로 미래 봉이 섞이지 않는다

    f["bar_range"] = (x.high - x.low) / a
    f["bar_body"] = (x.close - x.open) / a
    f["upper_wick"] = (x.high - x.close) / a
    f["ret_5"] = x.close / g.close.shift(5) - 1
    f["ret_day"] = x.close / g.open.transform("first") - 1
    f["from_high"] = (g.high.cummax() - x.close) / a
    f["from_low"] = (x.close - g.low.cummin()) / a
    f["minutes"] = x.bar_no - 1
    return f


def candidates(x: pd.DataFrame, p: StrategyParams) -> np.ndarray:
    """전략 A 매수 후보 봉 위치 (워밍업·진입 마감·ATR 하한 적용, tradable_from 은 보지 않음)."""
    out = []
    pos = 0
    for _, d in x.groupby(x.index.normalize()):
        ok = (raw_signal(d, p) & entry_allowed(d, p)).to_numpy().copy()
        ok[-1] = False                                       # 다음 봉이 없으면 진입 불가
        out.append(np.flatnonzero(ok) + pos)
        pos += len(d)
    return np.concatenate(out) if out else np.array([], dtype=int)


def triple_barrier(x: pd.DataFrame, idx: np.ndarray, p: StrategyParams, cost: CostModel = CostModel()) -> pd.DataFrame:
    """후보 봉마다 t+1 시가 진입을 가정해 손절·목표가·시간·장 마감 중 무엇이 먼저인지 본다."""
    o, hi, lo, c = (x[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr_s = x.atr_s.to_numpy()
    day = x.index.normalize().to_numpy()
    force = np.zeros(len(x), dtype=bool)
    for d in np.unique(day):
        m = day == d
        force[m] = x.index[m].time >= day_clock(pd.Timestamp(d).date(), p)[1]
    rows = []
    for t in idx:
        e = t + 1
        entry = o[e]
        stop, tgt = entry - p.lam * atr_s[t], entry + p.tau * atr_s[t]
        j = e
        while True:
            last = j + 1 >= len(x) or day[j + 1] != day[j]
            if lo[j] <= stop:                                # 같은 봉에서 둘 다 닿으면 손절 (보수적)
                reason, px, fill = "stop", min(stop, o[j]), j
            elif hi[j] >= tgt:
                reason, px, fill = "target", max(tgt, o[j]), j
            elif last:
                reason, px, fill = "eod", c[j], j
            elif force[j]:
                reason, px, fill = "close", o[j + 1], j + 1
            elif j - e + 1 >= p.h:
                reason, px, fill = "time", o[j + 1], j + 1
            else:
                j += 1
                continue
            break
        net = cost.net_return(entry, px, x.index[fill].date())
        y = int(reason == "target" or (reason in ("time", "close", "eod") and net > 0))
        rows.append({"time": x.index[t], "entry_time": x.index[e], "exit_time": x.index[fill],
                     "reason": reason, "bars": j - e + 1, "ret": px / entry - 1, "net": net, "y": y,
                     "y_net": int(net > 0)})
    return pd.DataFrame(rows)


def build(symbols, p: StrategyParams = StrategyParams(), cost: CostModel = CostModel(),
          raw_dir: str = "data/raw") -> pd.DataFrame:
    """종목들의 전 기간 A 매수 후보 → 피처 + 레이블 데이터셋."""
    parts = []
    for s in symbols:
        path = Path(raw_dir) / f"{s}_1m.parquet"
        if not path.exists():
            continue
        x = compute_indicators(pd.read_parquet(path), p.vwap)
        idx = candidates(x, p)
        if len(idx) == 0:
            continue
        lab = triple_barrier(x, idx, p, cost)
        feat = features(x).iloc[idx].reset_index(drop=True)
        parts.append(pd.concat([lab, feat], axis=1).assign(symbol=s))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = df.time.dt.normalize()
    cols = ["symbol", "date", "time", "entry_time", "exit_time", "reason", "bars", "ret", "net", "y", "y_net"] + FEATURES
    return df[cols].sort_values(["time", "symbol"], kind="stable").reset_index(drop=True)
