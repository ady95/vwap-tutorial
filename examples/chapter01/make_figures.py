"""01장 그림 — 실측 1분봉으로 이동평균선·VWAP·변형 VWAP 을 비교한다.

02장에서 1분봉을 수집한 뒤 실행한다.
실행: uv run python examples/chapter01/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vwap_trading.indicators.vwap import FLAT, compute_indicators
from vwap_trading.reporting.plotting import COLORS, hhmm_axis, shade, use_korean_font

sys.stdout.reconfigure(encoding="utf-8")
use_korean_font()
OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def load(symbol):
    return compute_indicators(pd.read_parquet(f"data/raw/{symbol}_1m.parquet"))


def crossings(close, line):
    side = (close > line).astype(int)[line.notna()]
    return int(side.diff().abs().sum())


# 그림 1 (01-1): 단순 이동평균선과 세션 VWAP
def fig_ma_vs_vwap(symbol="005930", date="2026-10-29"):
    x = load(symbol).loc[date]
    sma = x.close.rolling(20).mean()
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x.index, x.close, color=COLORS["close"], lw=1, label="종가")
    ax.plot(x.index, sma, color=COLORS["sma"], lw=1.5, label="단순 이동평균선 (20봉)")
    ax.plot(x.index, x.svwap, color=COLORS["svwap"], lw=2, label="세션 VWAP")
    ax.set_title(f"{symbol} {date} 1분봉: 이동평균선과 세션 VWAP")
    ax.set_ylabel("가격(원)")
    ax.legend(loc="upper right", fontsize=8)
    axv.bar(x.index, x.volume, width=1 / 1440, color=COLORS["close"])
    axv.set_ylabel("거래량")
    hhmm_axis(axv)
    fig.tight_layout()
    fig.savefig(OUT / "01-1-ma-vs-vwap.png", dpi=110)
    morning = x.between_time("09:00", "09:59").volume.sum() / x.volume.sum()
    print(f"[01-1] {symbol} {date}: 09시대 거래량 비중 {morning:.1%}, "
          f"15:19 종가 {x.close.loc[f'{date} 15:19']:.0f}, 이동평균 {sma.loc[f'{date} 15:19']:.0f}, "
          f"세션 VWAP {x.svwap.loc[f'{date} 15:19']:.0f}")


# 그림 2 (01-2): 횡보 → 상승 전환 매수 후보, 성공 사례와 실패 사례
def fig_breakout(cases=(("005380", "2026-02-19 12:46", "성공 사례"), ("005380", "2026-10-15 11:17", "실패 사례"))):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    for col, (symbol, when, label) in enumerate(cases):
        t = pd.Timestamp(when)
        x = load(symbol).loc[str(t.date())]
        i = x.index.get_loc(t)
        seg = (x.state != x.state.shift()).cumsum()
        flat_start = x.index.get_loc(x[seg == seg.iloc[i - 1]].index[0])
        w = x.iloc[max(flat_start - 25, 0): i + 35]
        ax, axv = axes[0, col], axes[1, col]
        ax.plot(w.index, w.close, color=COLORS["close"], lw=1, label="종가")
        ax.plot(w.index, w.rvwap, color=COLORS["rvwap"], lw=1, ls="--", label="롤링 VWAP")
        ax.plot(w.index, w.mvwap, color=COLORS["mvwap"], lw=2, label="변형 VWAP")
        shade(ax, w.index, (w.state == FLAT).to_numpy())
        ax.scatter([t], [x.close.iloc[i]], s=80, color="#d62728", zorder=5, label="매수 후보 (봉 확정)")
        ret30 = x.close.iloc[i + 30] / x.close.iloc[i] - 1
        ax.set_title(f"{label}: {symbol} {when[:10]}  (30분 뒤 {ret30:+.2%}, 비용 전)", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        axv.bar(w.index, w.volume, width=1 / 1440, color=["#d62728" if ts == t else COLORS["close"] for ts in w.index])
        hhmm_axis(axv)
        print(f"[01-2] {label} {symbol} {when}: 직전 횡보 {i - flat_start}봉, 박스 {x.mvwap.iloc[i - 1]:.0f}, "
              f"신호 봉 종가 {x.close.iloc[i]:.0f}, 상대 거래량 {x.rvol.iloc[i]:.1f}배, 30분 뒤 {ret30:+.2%}")
    axes[0, 0].set_ylabel("가격(원)")
    axes[1, 0].set_ylabel("거래량")
    fig.suptitle("횡보(음영) → 상승 전환 매수 후보: 같은 규칙, 다른 결과", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01-2-breakout-cases.png", dpi=110)


# 그림 3 (01-3): 세 기준선과 가격의 교차
def fig_three_lines(symbol="005930", date="2026-10-29"):
    x = load(symbol).loc[date]
    lines = [("세션 VWAP", x.svwap, COLORS["svwap"]), ("롤링 VWAP (N=30)", x.rvwap, COLORS["rvwap"]),
             ("연구용 변형 VWAP", x.mvwap, COLORS["mvwap"])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, (name, line, color) in zip(axes, lines):
        n = crossings(x.close, line)
        ax.plot(x.index, x.close, color=COLORS["close"], lw=1)
        ax.plot(x.index, line, color=color, lw=2)
        if name.startswith("연구용"):
            shade(ax, x.index, (x.state == FLAT).to_numpy())
        ax.set_title(f"{name} — 종가와 교차 {n}회", fontsize=10, loc="left")
        print(f"[01-3] {symbol} {date} {name}: 교차 {n}회")
    axes[1].set_ylabel("가격(원)")
    hhmm_axis(axes[-1])
    fig.suptitle(f"{symbol} {date}: 같은 가격, 세 가지 기준선", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01-3-three-lines.png", dpi=110)


fig_ma_vs_vwap()
fig_breakout()
fig_three_lines()
print(f"저장: {OUT}/01-1-ma-vs-vwap.png, 01-2-breakout-cases.png, 01-3-three-lines.png")
