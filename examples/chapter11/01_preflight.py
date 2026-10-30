"""11장 실습 1 — 장 시작 전 점검. 하나라도 '불가'면 모의매매를 시작하지 않는다. 주문 없음.

실행: uv run python examples/chapter11/01_preflight.py
"""

import os
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

from vwap_trading.broker.kiwoom import KiwoomClient, KiwoomError
from vwap_trading.broker.orders import PaperBroker
from vwap_trading.live.filter import LiveFilter
from vwap_trading.live.risk import RiskLimits
from vwap_trading.strategy.rules import load_params

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(find_dotenv(usecwd=True))
TZ = ZoneInfo(os.getenv("MARKET_TZ", "Asia/Seoul"))
results = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:                            # 점검은 실패해도 다음 항목을 계속 본다
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append(ok)
    print(f"  {'OK  ' if ok else '불가'}  {name}: {detail}")


def mode():
    m = os.getenv("TRADING_MODE", "offline")
    return m == "paper", f"TRADING_MODE={m}"


def keys():
    need = ["KIWOOM_REAL_APP_KEY", "KIWOOM_REAL_APP_SECRET", "KIWOOM_PAPER_APP_KEY", "KIWOOM_PAPER_APP_SECRET"]
    missing = [k for k in need if not os.getenv(k)]
    return not missing, "모두 있음" if not missing else f"없음: {missing}"


def token(env):
    def f():
        KiwoomClient.from_env(env).token()
        return True, f"{env} 서버 토큰 발급"
    return f


def clock():
    """응답 헤더 Date(초 단위, 소수점 버림)와 PC 시계를 비교한다. 다섯 번 재서 가장 작은 차이를 쓴다."""
    skews = []
    for _ in range(5):
        t0 = datetime.now(TZ).replace(tzinfo=None)
        r = requests.post("https://api.kiwoom.com/oauth2/token", json={}, timeout=10)
        t1 = datetime.now(TZ).replace(tzinfo=None)
        server = parsedate_to_datetime(r.headers["Date"]).astimezone(TZ).replace(tzinfo=None)
        skews.append(((t0 + (t1 - t0) / 2) - server).total_seconds() - 0.5)   # Date 는 초 미만을 버린다
    skew = min(skews, key=abs)
    return abs(skew) < 1.5, f"PC 시계 − 서버 시계 ≈ {skew:+.1f}초 (±0.5초, 봉 확정 타이머가 PC 시계를 쓴다)"


def account():
    b = PaperBroker(KiwoomClient.from_env("paper"), read_only=True)
    cash, hold, opens = b.cash(), b.holdings(), b.open_orders()
    need = RiskLimits().budget * RiskLimits().max_positions
    ok = cash >= need and opens.empty
    return ok, f"예수금 {cash:,}원 (필요 {need:,}원), 보유 {len(hold)}종목, 미체결 {len(opens)}건"


def data():
    u = pd.read_csv("data/sample/universe_20261030.csv", dtype={"symbol": str}).symbol
    have = [s for s in u if Path(f"data/raw/{s}_1m.parquet").exists()]
    return len(have) >= 20, f"유니버스 {len(u)}종목 중 분봉 파일 {len(have)}개 (부족분은 실행 시 조회로 채움)"


def model():
    f = LiveFilter.load("models_out/live_filter.joblib")
    return True, f"학습 마지막 날 {f.trained_until}, 모드 {f.mode} (10장 실습 3)"


def config():
    p = load_params("configs/strategy_a.toml")
    return True, f"전략 {p.strategy}, 손절 {p.lam} ATR, 목표 {p.tau} ATR, 최대 {p.h}봉 | {RiskLimits()}"


def stop_file():
    return not Path(RiskLimits().kill_file).exists(), "STOP 파일 없음" if not Path("STOP").exists() else "STOP 파일이 있음 (지우고 시작)"


print(f"장 시작 전 점검 {datetime.now(TZ):%Y-%m-%d %H:%M:%S}")
for name, fn in [("실행 모드", mode), ("앱키", keys), ("시세 서버", token("real")), ("모의 서버", token("paper")),
                 ("시계", clock), ("모의 계좌", account), ("분봉 데이터", data), ("AI 필터", model),
                 ("전략 설정", config), ("중단 파일", stop_file)]:
    check(name, fn)
print("\n결과:", "모두 OK — 시작해도 됩니다" if all(results) else f"불가 {results.count(False)}건 — 고친 뒤 다시 점검하세요")
sys.exit(0 if all(results) else 1)
