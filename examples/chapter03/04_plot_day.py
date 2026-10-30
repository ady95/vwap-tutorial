"""03장 실습 4 — 하루치 가격과 세 기준선, 상태를 차트로 그린다.

실행: uv run python examples/chapter03/04_plot_day.py 005930 2026-10-29
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from vwap_trading.indicators.vwap import FLAT, compute_indicators

# 한글 폰트: 설치된 것 중 첫 번째 (Windows 맑은 고딕, macOS 애플고딕, Linux 나눔고딕)
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in installed] or ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")
symbol, date = sys.argv[1], sys.argv[2]
x = compute_indicators(pd.read_parquet(f"data/raw/{symbol}_1m.parquet")).loc[date]

fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.plot(x.index, x.close, color="#888888", lw=1, label="종가")
ax.plot(x.index, x.svwap, color="#1f77b4", lw=1.5, label="세션 VWAP")
ax.plot(x.index, x.rvwap, color="#ff7f0e", lw=1, ls="--", label="롤링 VWAP (N=30)")
ax.plot(x.index, x.mvwap, color="#d62728", lw=2, label="변형 VWAP")

flat = (x.state == FLAT).to_numpy()
for i, t in enumerate(x.index):                      # FLAT 구간 음영
    if flat[i]:
        ax.axvspan(t, t + pd.Timedelta(minutes=1), color="#d62728", alpha=0.08, lw=0)
ax.set_title(f"{symbol} {date} 1분봉과 기준선 (음영: 횡보 FLAT)")
ax.legend(loc="upper right", fontsize=8)
axv.bar(x.index, x.volume, width=1 / 1440, color="#888888")
axv.set_ylabel("거래량")
ax.set_ylabel("가격(원)")
axv.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
fig.tight_layout()

out = Path("reports") / f"ch03_{symbol}_{date.replace('-', '')}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"저장: {out}")
