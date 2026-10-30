"""차트 공통 설정 (01·03장~)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import font_manager

COLORS = {"close": "#888888", "svwap": "#1f77b4", "rvwap": "#ff7f0e", "mvwap": "#d62728", "sma": "#2ca02c"}


def use_korean_font() -> None:
    """설치된 한글 폰트 중 첫 번째를 쓴다 (Windows 맑은 고딕, macOS 애플고딕, Linux 나눔고딕)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in installed]
    plt.rcParams["font.family"] = fonts or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def hhmm_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def shade(ax, index, mask, color="#d62728", alpha=0.08) -> None:
    """mask 가 참인 1분봉 구간에 음영을 칠한다."""
    for t, on in zip(index, mask):
        if on:
            ax.axvspan(t, t + matplotlib.dates.num2timedelta(1 / 1440), color=color, alpha=alpha, lw=0)
