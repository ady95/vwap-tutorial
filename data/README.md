# data/

| 폴더 | 내용 | 저장소 포함 |
|---|---|---|
| `raw/` | 키움 REST API 원본 1분봉 (`<종목코드>_1m.parquet`), 수집 기록 `_manifest.csv` | 아니오 (직접 수집) |
| `processed/` | 정규장 필터·지표 계산 결과 | 아니오 |
| `sample/` | 예제 유니버스, 종목 선정 입력(`selections.csv`) | 예 |

원시 시세는 이용 조건 때문에 저장소에 넣지 않습니다. `examples/chapter02/02_collect_minute_bars.py`로 직접 수집하세요.

`selections.csv` 형식:

```text
symbol,selected_at,tradable_from
353200,2025-11-05 09:30:00,2025-11-05 09:30:00
```
