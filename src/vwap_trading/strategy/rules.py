"""기본 규칙 기반 전략 S0·A (04장, 수식 명세 v0.2 6~9절).

봉 t 가 확정된 뒤에만 신호를 판정하고, 주문은 t+1 봉 시가로 낸다. 손절·목표가는 봉 안에서
(고가·저가로) 판정하고, 기준선 이탈·시간 청산·장 마감 청산은 확정 봉에서 판정해 다음 봉
시가로 청산한다. 이 모듈은 비용을 빼지 않은 가격만 기록하며, 비용은 05장에서 반영한다.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from vwap_trading.data.session import session_of
from vwap_trading.indicators.vwap import FLAT, VwapParams

STRATEGIES = ("S0", "A")


@dataclass(frozen=True)
class StrategyParams:
    strategy: str = "A"        # "S0" (세션 VWAP) 또는 "A" (연구용 변형 VWAP)
    compare: int = 2           # 1: 기준선 이탈 청산 없음, 2: 기준선 이탈 청산 포함
    delta: float = 0.3         # 돌파 이격 (ATR 배수, A 만)
    rho: float = 1.5           # 상대 거래량
    eps: float = 0.2           # 기준선 이탈 허용 (ATR 배수)
    lam: float = 1.5           # 손절 (ATR 배수) — 위험관리, 최적화 대상 아님
    tau: float = 2.0           # 목표가 (ATR 배수)
    h: int = 30                # 최대 보유 (봉)
    cooldown: int = 10         # 청산 후 재진입 대기 (봉)
    cutoff_min: int = 45       # 신규 진입 마감 = 장 마감 − 45분 (정규장 14:45)
    force_min: int = 16        # 장 마감 청산 판정 봉 = 장 마감 − 16분 (정규장 15:14)
    vwap: VwapParams = field(default_factory=VwapParams)

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy 는 {STRATEGIES} 중 하나: {self.strategy}")
        if self.compare not in (1, 2):
            raise ValueError(f"compare 는 1 또는 2: {self.compare}")


def load_params(path) -> StrategyParams:
    """TOML 설정 파일 → StrategyParams. [vwap] 표는 VwapParams 로 넘긴다."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    vwap = VwapParams(**cfg.pop("vwap", {}))
    known = {f.name for f in fields(StrategyParams)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"알 수 없는 설정: {sorted(unknown)}")
    return StrategyParams(**cfg, vwap=vwap)


def day_clock(d: date, p: StrategyParams) -> tuple[time, time]:
    """그날의 (신규 진입 마감, 장 마감 청산 판정 봉) 시각. 예외일은 장 마감 기준으로 옮긴다."""
    close = datetime.combine(d, session_of(d)[1])
    return (close - timedelta(minutes=p.cutoff_min)).time(), (close - timedelta(minutes=p.force_min)).time()


def raw_signal(x: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """시간·포지션 조건을 빼고, 지표만으로 본 매수 후보 (봉 t 확정 기준)."""
    if p.strategy == "A":
        cond = ((x.state.shift() == FLAT) & (x.slope >= p.vwap.theta)
                & (x.close > x.mvwap.shift() + p.delta * x.atr_s))
    else:
        cond = (x.close > x.svwap) & (x.close.shift() <= x.svwap.shift())
    return cond & (x.rvol >= p.rho)


def entry_allowed(x: pd.DataFrame, p: StrategyParams, tradable_from: pd.Timestamp | None = None) -> pd.Series:
    """워밍업·tradable_from·진입 마감·ATR 하한 조건 (명세 6절). 하루치 x 에만 쓴다."""
    cutoff, _ = day_clock(x.index[0].date(), p)
    ok = (x.bar_no > p.vwap.warmup_end) & (x.index.time <= cutoff) & ~x.atr_low
    if tradable_from is not None:
        ok &= x.index >= tradable_from
    return ok


def base_line(x: pd.DataFrame, p: StrategyParams) -> pd.Series | None:
    """기준선 이탈 청산에 쓰는 선. 비교 1 이면 None (이탈 청산 없음)."""
    if p.compare == 1:
        return None
    return x.mvwap if p.strategy == "A" else x.svwap


def run_day(x: pd.DataFrame, p: StrategyParams, tradable_from: pd.Timestamp | None = None) -> list[dict]:
    """하루치 지표 x 로 매매를 시뮬레이션한다. 종목당 동시 포지션 1개."""
    sig = (raw_signal(x, p) & entry_allowed(x, p, tradable_from)).to_numpy()
    base = base_line(x, p)
    base = None if base is None else base.to_numpy()
    o, hi, lo, c = (x[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_s = x.atr_s.to_numpy()
    _, force_t = day_clock(x.index[0].date(), p)
    force = x.index.time >= force_t
    n = len(x)

    trades, i, next_ok = [], 0, 0
    while i < n - 1:
        if not sig[i] or i < next_ok:
            i += 1
            continue
        t, e = i, i + 1                                     # 신호 봉 t, 진입 봉 e
        entry, risk = o[e], atr_s[t]                        # ATR* 는 신호 봉 값으로 고정
        stop, tgt = entry - p.lam * risk, entry + p.tau * risk
        j, reason, px, fill = e, None, None, None
        while j < n:
            if lo[j] <= stop:                               # (1) 손절 (같은 봉이면 손절 우선)
                reason, px, fill = "stop", min(stop, o[j]), j
            elif hi[j] >= tgt:                              # (2) 목표가
                reason, px, fill = "target", max(tgt, o[j]), j
            elif j == n - 1:                                # 데이터 끝 (안전장치)
                reason, px, fill = "eod", c[j], j
            elif force[j]:                                  # (5) 장 마감 청산
                reason = "close"
            elif j - e + 1 >= p.h:                          # (4) 시간 청산
                reason = "time"
            elif base is not None and c[j] < base[j] - p.eps * atr_s[j]:   # (3) 기준선 이탈
                reason = "base"
            if reason is not None:
                if px is None:                              # (3)(4)(5) 는 다음 봉 시가
                    px, fill = o[j + 1], j + 1
                break
            j += 1
        trades.append({
            "signal_time": x.index[t], "entry_time": x.index[e], "entry": entry,
            "stop": stop, "target": tgt, "exit_time": x.index[fill], "exit": px,
            "reason": reason, "bars": j - e + 1, "ret": px / entry - 1,   # bars: 보유한 봉 수
        })
        next_ok = fill + p.cooldown                         # 청산 체결 봉부터 cooldown 봉 동안 신호 무시
        i = fill
    return trades


def run(ind: dict[str, pd.DataFrame], selections: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """선정 종목-일마다 run_day 를 돌려 매매 기록을 모은다.

    ind: 종목코드 → compute_indicators 결과. selections: symbol, tradable_from 열.
    """
    by_day = {s: dict(tuple(x.groupby(x.index.date))) for s, x in ind.items()}
    rows = []
    for sel in selections.itertuples():
        tf = pd.Timestamp(sel.tradable_from)
        day = by_day.get(sel.symbol, {}).get(tf.date())
        if day is None:
            continue
        for tr in run_day(day, p, tf):
            rows.append({"symbol": sel.symbol, **tr})
    cols = ["symbol", "signal_time", "entry_time", "entry", "stop", "target",
            "exit_time", "exit", "reason", "bars", "ret"]
    return pd.DataFrame(rows, columns=cols)
