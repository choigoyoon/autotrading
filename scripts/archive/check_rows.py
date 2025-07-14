import pandas as pd
from pathlib import Path
from pandas import DataFrame

# 파일 목록
files: list[tuple[str, str]] = [
    ('1min.parquet', '1분봉'),
    ('3min.parquet', '3분봉'),
    ('5min.parquet', '5분봉'),
    ('10min.parquet', '10분봉'),
    ('15min.parquet', '15분봉'),
    ('30min.parquet', '30분봉'),
    ('1h.parquet', '1시간'),
    ('2h.parquet', '2시간'),
    ('4h.parquet', '4시간'),
    ('6h.parquet', '6시간'),
    ('8h.parquet', '8시간'),
    ('12h.parquet', '12시간'),
    ('1day.parquet', '1일'),
    ('2day.parquet', '2일'),
    ('3day.parquet', '3일'),
    ('1week.parquet', '1주')
]

base_path = Path('data/processed/btc_usdt_kst/resampled_ohlcv')
print(f"--- 데이터 행 수 확인 ({base_path}) ---")

# 각 파일 확인
for f, name in files:
    path = base_path / f
    try:
        if not path.exists():
            print(f"{name}: 파일 없음")
            continue
        df: DataFrame = pd.read_parquet(path)
        print(f"{name}: {len(df):,}행")
    except Exception as e:
        print(f"{name} 오류: {e}")
