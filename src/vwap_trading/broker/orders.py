"""키움 주문·계좌 TR (10장). 주문은 모의투자 서버에서만 낸다.

주문 TR (POST /api/dostk/ordr)
  kt10000 매수 · kt10001 매도 : dmst_stex_tp, stk_cd, ord_qty, ord_uv, trde_tp (0 지정가, 3 시장가), cond_uv
  kt10002 정정               : dmst_stex_tp, orig_ord_no, stk_cd, mdfy_qty, mdfy_uv, mdfy_cond_uv
  kt10003 취소               : dmst_stex_tp, orig_ord_no, stk_cd, cncl_qty ("0" 이면 남은 수량 전부)
계좌 TR (POST /api/dostk/acnt)
  kt00001 예수금 · kt00018 잔고 · ka10075 미체결 · ka10076 체결

조회와 달리 주문은 자동으로 재시도하지 않는다. 응답을 못 받은 주문(네트워크 오류·시간 초과)은
서버에 들어갔는지 알 수 없으므로 OrderUnknown 으로 알리고, 미체결·체결 조회로 확인한 뒤에
다시 낼지 정한다(11-4). 같은 주문을 두 번 내는 것보다 한 번 놓치는 편이 덜 위험하다.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from vwap_trading.broker.kiwoom import HOSTS, KiwoomClient, KiwoomError

TRADE_TYPE = {"limit": "0", "market": "3"}


class RealOrderBlocked(RuntimeError):
    """실계좌 주문 시도. 이 책의 코드는 모의투자 서버로만 주문한다."""


class OrderRejected(KiwoomError):
    """서버가 응답했고 주문을 받지 않았다 (확실히 주문이 없다)."""

    def __init__(self, api_id: str, code, msg: str):
        super().__init__(f"[{api_id}] {code} {msg}")
        self.code, self.msg = code, msg


class OrderUnknown(KiwoomError):
    """응답을 받지 못했다. 주문이 들어갔는지 알 수 없다."""


def ensure_paper(client: KiwoomClient, ordering: bool = True) -> None:
    """실계좌 주문 차단: 세 가지를 모두 확인한다. 조회만 할 때(ordering=False)는 실행 모드를 보지 않는다."""
    if client.env != "paper":
        raise RealOrderBlocked(f"모의투자 클라이언트가 아닙니다: env={client.env}")
    if client.host != HOSTS["paper"]:
        raise RealOrderBlocked(f"모의투자 서버 주소가 아닙니다: {client.host}")
    if ordering and os.getenv("TRADING_MODE", "offline") != "paper":
        raise RealOrderBlocked("TRADING_MODE=paper 일 때만 주문합니다 (.env 확인)")


def num(s) -> float:
    """키움 응답 숫자 문자열 → 숫자. '+72000', '-500', '000000100000000' 같은 형식을 받는다."""
    s = str(s).strip()
    return abs(float(s)) if s not in ("", "-", "+") else 0.0


class PaperBroker:
    def __init__(self, client: KiwoomClient, exchange: str = "KRX", read_only: bool = False):
        ensure_paper(client, ordering=not read_only)
        self.client, self.exchange, self.read_only = client, exchange, read_only

    # ── 주문 ────────────────────────────────────────────────────────────
    def _order(self, api_id: str, body: dict) -> dict:
        if self.read_only:
            raise RealOrderBlocked("조회 전용(read_only)으로 만든 객체입니다")
        ensure_paper(self.client)                            # 보내기 직전에 한 번 더
        c = self.client
        for _ in range(2):                                   # 인증 만료(3)만 토큰을 새로 받아 한 번 더
            wait = c.min_interval - (time.monotonic() - c._last_call)
            if wait > 0:
                time.sleep(wait)
            headers = {"Content-Type": "application/json;charset=UTF-8",
                       "authorization": f"Bearer {c.token()}", "api-id": api_id}
            try:
                data = c.session.post(f"{c.host}/api/dostk/ordr", headers=headers, json=body, timeout=5).json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                raise OrderUnknown(f"[{api_id}] 응답 없음: {type(e).__name__}") from e
            finally:
                c._last_call = time.monotonic()
            if data.get("return_code") == 3:
                c._token = None
                continue
            break
        if data.get("return_code") != 0:
            raise OrderRejected(api_id, data.get("return_code"), data.get("return_msg", ""))
        return data

    def buy(self, symbol: str, qty: int, price: float | None = None) -> str:
        return self._new("kt10000", symbol, qty, price)

    def sell(self, symbol: str, qty: int, price: float | None = None) -> str:
        return self._new("kt10001", symbol, qty, price)

    def _new(self, api_id: str, symbol: str, qty: int, price: float | None) -> str:
        kind = "market" if price is None else "limit"
        data = self._order(api_id, {"dmst_stex_tp": self.exchange, "stk_cd": symbol, "ord_qty": str(qty),
                                    "ord_uv": "" if price is None else str(int(price)),
                                    "trde_tp": TRADE_TYPE[kind], "cond_uv": ""})
        return data["ord_no"]

    def modify(self, orig_ord_no: str, symbol: str, qty: int, price: float) -> str:
        data = self._order("kt10002", {"dmst_stex_tp": self.exchange, "orig_ord_no": orig_ord_no, "stk_cd": symbol,
                                       "mdfy_qty": str(qty), "mdfy_uv": str(int(price)), "mdfy_cond_uv": ""})
        return data["ord_no"]

    def cancel(self, orig_ord_no: str, symbol: str, qty: int = 0) -> str:
        data = self._order("kt10003", {"dmst_stex_tp": self.exchange, "orig_ord_no": orig_ord_no,
                                       "stk_cd": symbol, "cncl_qty": str(qty)})
        return data["ord_no"]

    # ── 계좌 조회 (자동 재시도 가능) ─────────────────────────────────────
    def cash(self) -> int:
        d, _ = self.client.post("/api/dostk/acnt", "kt00001", {"qry_tp": "3"})
        return int(num(d.get("entr", 0)))

    def holdings(self) -> pd.DataFrame:
        """보유 종목: symbol, qty, avg_price. 잔고 TR 의 종목코드에는 'A' 접두어가 붙어 온다."""
        d, _ = self.client.post("/api/dostk/acnt", "kt00018", {"qry_tp": "1", "dmst_stex_tp": self.exchange})
        rows = [{"symbol": r["stk_cd"].lstrip("A"), "qty": int(num(r.get("rmnd_qty", 0))),
                 "avg_price": num(r.get("pur_pric", 0))} for r in d.get("acnt_evlt_remn_indv_tot") or []]
        return pd.DataFrame(rows, columns=["symbol", "qty", "avg_price"])

    def open_orders(self) -> pd.DataFrame:
        """미체결 주문: ord_no, symbol, side, qty, open_qty, price."""
        d, _ = self.client.post("/api/dostk/acnt", "ka10075",
                                {"all_stk_tp": "0", "trde_tp": "0", "stk_cd": "", "stex_tp": "0"})
        rows = [{"ord_no": r["ord_no"], "symbol": r["stk_cd"].lstrip("A"), "side": r.get("io_tp_nm", ""),
                 "qty": int(num(r.get("ord_qty", 0))), "open_qty": int(num(r.get("oso_qty", 0))),
                 "price": num(r.get("ord_pric", 0))} for r in d.get("oso") or []]
        return pd.DataFrame(rows, columns=["ord_no", "symbol", "side", "qty", "open_qty", "price"])

    def fills(self) -> pd.DataFrame:
        """당일 체결 (주문 단위 누계): ord_no, symbol, side, qty, filled, price."""
        d, _ = self.client.post("/api/dostk/acnt", "ka10076",
                                {"stk_cd": "", "qry_tp": "0", "sell_tp": "0", "ord_no": "", "stex_tp": "0"})
        rows = [{"ord_no": r["ord_no"], "symbol": r["stk_cd"].lstrip("A"), "side": r.get("io_tp_nm", ""),
                 "qty": int(num(r.get("ord_qty", 0))), "filled": int(num(r.get("cntr_qty", 0))),
                 "price": num(r.get("cntr_pric", 0))} for r in d.get("cntr") or []]
        return pd.DataFrame(rows, columns=["ord_no", "symbol", "side", "qty", "filled", "price"])
