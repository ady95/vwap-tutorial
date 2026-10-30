"""VWAP 지표와 연구용 변형 VWAP (03장, 수식 명세 v0.2).

모든 계산은 하루 단위로 한다. 정규장 봉만 넣어야 하며(data.session.regular_session),
전일 봉으로 워밍업하지 않는다. 윈도우가 덜 찬 구간은 NaN 으로 둔다(부분 윈도우 금지).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from vwap_trading.data.session import regular_session

FLAT, UP, DOWN = 0, 1, -1
# 기준선·기울기를 이 자릿수로 반올림한다. 정수 가격에서는 기울기가 임계값과 정확히 같아지는 봉이
# 생기는데, 계산 순서에 따른 1e-11 수준의 오차가 상태 판정을 뒤집지 않게 하기 위해서다.
PRICE_DP, SLOPE_DP = 6, 9


@dataclass(frozen=True)
class VwapParams:
    N: int = 30             # 롤링 VWAP 기간 (봉)
    k: int = 5              # 기울기 간격 (봉)
    theta: float = 0.05     # 횡보 임계 (ATR 정규화 기울기)
    m: int = 10             # 횡보 최소 지속 (봉)
    n_atr: int = 14         # ATR 기간
    n_vol: int = 20         # 상대 거래량 기간
    eps_atr: float = 0.0005  # ATR 하한 (가격 대비)

    @property
    def warmup_end(self) -> int:
        """상태 판정이 처음 FLAT 이 될 수 있는 봉 순번 (1부터)."""
        return self.N + self.k + self.m - 1


def typical_price(day: pd.DataFrame) -> pd.Series:
    return (day.high + day.low + day.close) / 3


def session_vwap(day: pd.DataFrame) -> pd.Series:
    pv = typical_price(day) * day.volume
    return pv.cumsum() / day.volume.cumsum()


def rolling_vwap(day: pd.DataFrame, n: int) -> pd.Series:
    pv = typical_price(day) * day.volume
    return pv.rolling(n).sum() / day.volume.rolling(n).sum()


def atr(day: pd.DataFrame, n: int) -> pd.Series:
    prev_close = day.close.shift()
    tr = pd.concat([day.high - day.low, (day.high - prev_close).abs(), (day.low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = day.high.iloc[0] - day.low.iloc[0]     # 당일 첫 봉: 전일 종가를 쓰지 않는다
    return tr.rolling(n).mean()


def relative_volume(day: pd.DataFrame, n: int) -> pd.Series:
    base = day.volume.shift(1).rolling(n).mean()          # 현재 봉 제외
    return day.volume / base.replace(0, np.nan)


def classify_state(s: np.ndarray, theta: float, m: int) -> np.ndarray:
    """정규화 기울기 → 상태(FLAT/UP/DOWN, 미정의는 NaN). 과거로 소급하지 않는다."""
    state = np.full(len(s), np.nan)
    run, prev = 0, np.nan
    for i, v in enumerate(s):
        if np.isnan(v):
            run, prev = 0, np.nan
            continue
        run = run + 1 if abs(v) < theta else 0
        if run >= m:
            cur = FLAT
        elif v >= theta:
            cur = UP
        elif v <= -theta:
            cur = DOWN
        else:
            cur = prev                                     # 아직 m봉 미만 → 직전 상태 유지
        state[i] = cur
        prev = cur
    return state


def box_center(tp: np.ndarray, vol: np.ndarray, state: np.ndarray, m: int) -> np.ndarray:
    """FLAT 이 새로 시작된 봉에서 최근 m봉 VWAP 을 계산해 FLAT 이 끝날 때까지 고정한다."""
    box = np.full(len(state), np.nan)
    level = np.nan
    for i, st in enumerate(state):
        if st == FLAT:
            if i == 0 or state[i - 1] != FLAT:            # FLAT 확정 봉 t0
                w = slice(i - m + 1, i + 1)
                level = (tp[w] * vol[w]).sum() / vol[w].sum()
            box[i] = level
        else:
            level = np.nan
    return box


def day_indicators(day: pd.DataFrame, p: VwapParams) -> pd.DataFrame:
    out = day.copy()
    out["bar_no"] = np.arange(1, len(day) + 1)
    out["tp"] = typical_price(day)
    out["svwap"] = session_vwap(day).round(PRICE_DP)
    out["rvwap"] = rolling_vwap(day, p.N).round(PRICE_DP)
    out["atr"] = atr(day, p.n_atr)
    out["atr_s"] = np.maximum(out.atr, p.eps_atr * day.close)          # ATR* (분모 하한)
    out["atr_low"] = out.atr < p.eps_atr * day.close                     # 신규 진입 금지 표시
    out["slope"] = ((out.rvwap - out.rvwap.shift(p.k)) / (p.k * out.atr_s)).round(SLOPE_DP)
    out["state"] = classify_state(out.slope.to_numpy(), p.theta, p.m)
    out["box"] = box_center(out.tp.to_numpy(), day.volume.to_numpy(dtype=float), out.state.to_numpy(), p.m)
    out["mvwap"] = np.where(out.state == FLAT, out.box, np.where(out.state.isna(), np.nan, out.rvwap))
    out["rvol"] = relative_volume(day, p.n_vol)
    return out


def compute_indicators_reference(bars: pd.DataFrame, p: VwapParams = VwapParams()) -> pd.DataFrame:
    """정규장 봉만 골라 날짜별로 day_indicators 를 돌린다 (03장 설명용 기준 구현, 느림)."""
    bars = regular_session(bars)
    days = [day_indicators(g, p) for _, g in bars.groupby(bars.index.date)]
    out = pd.concat(days)
    out.attrs["params"] = asdict(p)
    return out


def _day_cumsum(v: np.ndarray, first: np.ndarray) -> np.ndarray:
    """날짜마다 0 에서 다시 시작하는 누적합. first: 그날 첫 봉 위치."""
    c = np.cumsum(v)
    start = np.repeat(np.r_[0.0, c][first], np.diff(np.r_[first, len(v)]))
    return c - start


def _day_rolling_sum(v: np.ndarray, n: int, bar_no: np.ndarray, first: np.ndarray) -> np.ndarray:
    """날짜 안에서만 n 봉 이동합. 그날 봉이 n 개 미만이면 NaN (부분 윈도우 금지)."""
    c = _day_cumsum(v, first)
    prev = np.r_[np.full(n, np.nan), c[:-n]] if len(c) > n else np.full(len(c), np.nan)
    out = np.where(bar_no > n, c - prev, c)
    return np.where(bar_no >= n, out, np.nan)


def compute_indicators(bars: pd.DataFrame, p: VwapParams = VwapParams()) -> pd.DataFrame:
    """정규장 봉만 골라 지표를 계산한다. 하루 단위 계산을 배열 연산으로 한 번에 한다.

    결과는 compute_indicators_reference 와 같다(부동소수 오차 범위). 06장 Optuna 처럼
    수백 번 반복할 때를 위해 날짜별 반복을 없앴다.
    """
    bars = regular_session(bars)
    out = bars.copy()
    if bars.empty:
        return compute_indicators_reference(bars, p)
    day = bars.index.normalize().to_numpy()
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    bar_no = np.arange(len(bars)) - np.repeat(first, np.diff(np.r_[first, len(bars)])) + 1
    h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    vol = bars.volume.to_numpy(dtype=float)
    tp = (h + lo + c) / 3
    pv = tp * vol

    out["bar_no"] = bar_no
    out["tp"] = tp
    out["svwap"] = np.round(_day_cumsum(pv, first) / _day_cumsum(vol, first), PRICE_DP)
    rvwap = np.round(_day_rolling_sum(pv, p.N, bar_no, first) / _day_rolling_sum(vol, p.N, bar_no, first), PRICE_DP)
    out["rvwap"] = rvwap
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
    tr[first] = h[first] - lo[first]                       # 당일 첫 봉: 전일 종가를 쓰지 않는다
    atr_ = _day_rolling_sum(tr, p.n_atr, bar_no, first) / p.n_atr
    out["atr"] = atr_
    atr_s = np.maximum(atr_, p.eps_atr * c)
    atr_s[np.isnan(atr_)] = np.nan
    out["atr_s"] = atr_s
    out["atr_low"] = atr_ < p.eps_atr * c
    prev_r = np.r_[np.full(p.k, np.nan), rvwap[:-p.k]]
    slope = np.round(np.where(bar_no > p.k, (rvwap - prev_r) / (p.k * atr_s), np.nan), SLOPE_DP)
    out["slope"] = slope
    state = classify_state(slope, p.theta, p.m)
    out["state"] = state
    box = box_center(tp, vol, state, p.m)
    out["box"] = box
    out["mvwap"] = np.where(state == FLAT, box, np.where(np.isnan(state), np.nan, rvwap))
    vol_prev = np.r_[np.nan, vol[:-1]]
    vol_prev[first] = np.nan
    base = _day_rolling_sum(np.nan_to_num(vol_prev), p.n_vol, bar_no - 1, first) / p.n_vol
    out["rvol"] = vol / np.where(base == 0, np.nan, base)
    out.attrs["params"] = asdict(p)
    return out


def load_indicators(symbols, p: VwapParams = VwapParams(), raw_dir="data/raw") -> dict[str, pd.DataFrame]:
    """02장에서 저장한 1분봉 Parquet 을 읽어 종목별 지표를 계산한다 (종목당 약 0.2초)."""
    return {s: compute_indicators(pd.read_parquet(f"{raw_dir}/{s}_1m.parquet"), p) for s in dict.fromkeys(symbols)}
