import pandas as pd

# 1분봉 데이터 로드
df = pd.read_parquet('data/processed/btc_usdt_kst/resampled_ohlcv/1min.parquet')

# 시작일과 종료일 확인
print(f"1분봉 데이터 범위:")
print(f"시작일: {df.index[0]}")
print(f"종료일: {df.index[-1]}")
print(f"전체 일수: {(df.index[-1] - df.index[0]).days}일")
print(f"1분봉 행 수: {len(df):,}행")

# 일봉 데이터 로드
df_day = pd.read_parquet('data/processed/btc_usdt_kst/resampled_ohlcv/1day.parquet')
print(f"\n일봉 행 수: {len(df_day):,}행")
print(f"일봉 시작일: {df_day.index[0]}")
print(f"일봉 종료일: {df_day.index[-1]}")
