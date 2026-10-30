"""키움 웹소켓 실시간 수신 (10장).

접속 → LOGIN → REG 를 한 묶음으로 하고, 끊기면 묶음 전체를 다시 한다. 서버는 끊긴 연결의
등록 상태를 기억하지 않으므로, 등록 목록은 클라이언트가 들고 있어야 한다.

실측 (2026-10-31, 실전·모의 서버):
- 서버가 약 10~20초마다 {"trnm":"PING"} 을 보낸다. 받은 그대로 돌려보내지 않으면 약 40초 뒤
  SYSTEM R10002 "Heartbeat(연결유지) 메시지가 오지않아서 접속을 종료합니다" 와 함께 끊는다.
- 같은 앱키로 두 번째 연결이 LOGIN 하면 첫 번째 연결이 close code 1000 "Bye" 로 끊긴다.
  프로그램을 두 개 띄우면 서로를 끊으며 재접속을 반복하게 된다.
- 등록할 수 없는 실시간 타입을 보내면 REG 응답의 return_code 가 0 이 아니다 (예: 305005).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import websockets

from vwap_trading.broker.kiwoom import KiwoomClient

WS_HOSTS = {"real": "wss://api.kiwoom.com:10000", "paper": "wss://mockapi.kiwoom.com:10000"}
WS_PATH = "/api/dostk/websocket"


class LoginFailed(RuntimeError):
    pass


class KiwoomSocket:
    """on_real(type, item, values) 로 실시간 데이터를, on_status(kind, detail) 로 연결 상태를 알린다."""

    def __init__(self, client: KiwoomClient, on_real: Callable, on_status: Callable = lambda kind, detail: None):
        self.client, self.on_real, self.on_status = client, on_real, on_status
        self.url = WS_HOSTS[client.env] + WS_PATH
        self.groups: dict[str, tuple[list[str], list[str]]] = {}   # grp_no → (종목, 실시간 타입)
        self.ws = None
        self.connects = 0

    async def subscribe(self, grp_no: str, items: list[str], types: list[str]) -> None:
        self.groups[grp_no] = (items, types)
        if self.ws is not None:
            await self._reg(grp_no)

    async def _reg(self, grp_no: str) -> None:
        items, types = self.groups[grp_no]
        await self.ws.send(json.dumps({"trnm": "REG", "grp_no": grp_no, "refresh": "1",
                                       "data": [{"item": items, "type": types}]}))

    async def run(self, max_backoff: float = 30.0) -> None:
        """취소될 때까지 수신한다. 끊기면 1, 2, 4 … 초 기다렸다가 다시 접속한다."""
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    await ws.send(json.dumps({"trnm": "LOGIN", "token": self.client.token()}))
                    login = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                    if login.get("return_code") != 0:
                        self.client._token = None            # 토큰 만료·무효 → 다음 접속에서 새로 받는다
                        raise LoginFailed(f"{login.get('return_code')} {login.get('return_msg')}")
                    self.ws = ws
                    self.connects += 1
                    for g in self.groups:
                        await self._reg(g)
                    self.on_status("connected", f"{self.connects}번째 접속")
                    backoff = 1.0
                    async for msg in ws:
                        await self._handle(ws, msg)
                    self.on_status("closed", f"{ws.close_code} {ws.close_reason}")
            except asyncio.CancelledError:
                raise
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException, LoginFailed) as e:
                self.on_status("error", f"{type(e).__name__}: {e}")
            finally:
                self.ws = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    async def _handle(self, ws, msg: str) -> None:
        d = json.loads(msg)
        trnm = d.get("trnm")
        if trnm == "PING":
            await ws.send(msg)                               # 받은 그대로 돌려보낸다
        elif trnm == "REAL":
            for row in d.get("data", []):
                self.on_real(row.get("type"), row.get("item"), row.get("values", {}))
        elif trnm == "REG" and d.get("return_code") != 0:
            self.on_status("reg_error", f"{d.get('return_code')} {d.get('return_msg')}")
        elif trnm == "SYSTEM":
            self.on_status("system", f"{d.get('code')} {d.get('message')}")
