import pandas as pd
from pandas import DataFrame, DatetimeIndex

try:
    # 1분봉 데이터 로드
    df: DataFrame = pd.read_parquet('data/processed/btc_usdt_kst/resampled_ohlcv/1min.parquet')

    # 시작일과 종료일 확인
    if isinstance(df.index, DatetimeIndex) and len(df.index) > 0:
        print("1분봉 데이터 범위:")
        print(f"시작일: {df.index[0]}")
        print(f"종료일: {df.index[-1]}")
        print(f"전체 일수: {(df.index[-1] - df.index[0]).days}일")
        print(f"1분봉 행 수: {len(df):,}행")
    else:
        print("1분봉 데이터의 인덱스가 날짜/시간 형식이 아니거나 비어있습니다.")

    # 일봉 데이터 로드
    df_day: DataFrame = pd.read_parquet('data/processed/btc_usdt_kst/resampled_ohlcv/1day.parquet')
    
    if isinstance(df_day.index, DatetimeIndex) and len(df_day.index) > 0:
        print(f"\n일봉 행 수: {len(df_day):,}행")
        print(f"일봉 시작일: {df_day.index[0]}")
        print(f"일봉 종료일: {df_day.index[-1]}")
    else:
        print("\n일봉 데이터의 인덱스가 날짜/시간 형식이 아니거나 비어있습니다.")

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다 - {e}")
except Exception as e:
    print(f"데이터 처리 중 오류 발생: {e}")
