"""키움증권 REST API 클라이언트 (02장: 토큰·조회).

- 토큰: POST /oauth2/token (appkey, secretkey) → 약 24시간 유효
- 조회: POST /api/dostk/... 요청 헤더 api-id 로 TR 을 구분한다.
- 연속조회: 응답 헤더 cont-yn=Y 이면 next-key 를 요청 헤더에 넣어 이어서 받는다.
- 호출 한도를 넘으면 return_code 가 0 이 아니므로 잠시 쉬고 재시도한다.
- 연속조회를 수백 번 하면 새 연결이 시간 초과로 끊기는 일이 있어 Session 으로 연결을 재사용하고,
  네트워크 오류도 재시도한다.
"""

from __future__ import annotations

import os
import time

import requests

HOSTS = {"real": "https://api.kiwoom.com", "paper": "https://mockapi.kiwoom.com"}


class KiwoomError(RuntimeError):
    pass


class KiwoomClient:
    def __init__(self, app_key: str, app_secret: str, env: str = "real", min_interval: float = 0.25):
        self.host = HOSTS[env]
        self.env = env
        self.app_key, self.app_secret = app_key, app_secret
        self.min_interval = min_interval   # 연속 호출 사이 최소 간격(초)
        self._token: str | None = None
        self._last_call = 0.0
        self.session = requests.Session()

    @classmethod
    def from_env(cls, env: str = "real") -> "KiwoomClient":
        p = env.upper()
        key, secret = os.getenv(f"KIWOOM_{p}_APP_KEY"), os.getenv(f"KIWOOM_{p}_APP_SECRET")
        if not key or not secret:
            raise KiwoomError(f"KIWOOM_{p}_APP_KEY / KIWOOM_{p}_APP_SECRET 가 설정되지 않았습니다 (.env 확인)")
        return cls(key, secret, env)

    def token(self) -> str:
        if not self._token:
            body = requests.post(f"{self.host}/oauth2/token", timeout=10, json={
                "grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}).json()
            if "token" not in body:
                raise KiwoomError(f"키움 토큰 발급 실패: {body.get('return_code')} {body.get('return_msg')}")
            self._token = body["token"]
        return self._token

    def revoke(self) -> None:
        if self._token:
            requests.post(f"{self.host}/oauth2/revoke", timeout=10, json={
                "appkey": self.app_key, "secretkey": self.app_secret, "token": self._token})
            self._token = None

    def post(self, path: str, api_id: str, body: dict, next_key: str | None = None,
             retries: int = 5) -> tuple[dict, requests.structures.CaseInsensitiveDict]:
        headers = {"Content-Type": "application/json;charset=UTF-8", "authorization": f"Bearer {self.token()}",
                   "api-id": api_id}
        if next_key:
            headers.update({"cont-yn": "Y", "next-key": next_key})
        for attempt in range(retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.post(f"{self.host}{path}", headers=headers, json=body, timeout=15)
                data = r.json()
            except (requests.ConnectionError, requests.Timeout, ValueError) as e:
                data = {"return_code": None, "return_msg": f"네트워크 오류: {type(e).__name__}"}
            self._last_call = time.monotonic()
            if data.get("return_code") == 0:
                return data, r.headers
            if attempt < retries:   # 호출 한도 초과·연결 끊김 등 일시 오류는 점점 길게 쉬고 재시도
                time.sleep(2.0 * (attempt + 1))
        raise KiwoomError(f"[{api_id}] {data.get('return_code')} {data.get('return_msg')}")

    def post_all(self, path: str, api_id: str, body: dict, list_key: str, max_pages: int = 10):
        """연속조회를 따라가며 페이지마다 (행 목록, 페이지 번호) 를 돌려준다."""
        next_key = None
        for page in range(1, max_pages + 1):
            data, headers = self.post(path, api_id, body, next_key)
            rows = data.get(list_key) or []
            yield rows, page
            if headers.get("cont-yn") != "Y" or not rows:
                return
            next_key = headers.get("next-key")
