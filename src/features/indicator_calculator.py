import pandas as pd
import talib
from typing import List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_indicators(df: pd.DataFrame, indicators: List[str] = ['rsi', 'macd']) -> pd.DataFrame:
    """
    주어진 데이터프레임에 기술적 지표들을 계산하여 추가합니다.

    Args:
        df (pd.DataFrame): OHLCV 데이터프레임.
        indicators (List[str]): 추가할 지표 목록. 현재 'rsi', 'macd' 지원.

    Returns:
        pd.DataFrame: 지표가 추가된 데이터프레임.
    """
    logging.info(f"기술적 지표 추가: {indicators}")
    
    df_copy = df.copy()

    if 'rsi' in indicators:
        df_copy['rsi'] = talib.RSI(df_copy['close'], timeperiod=14) # type: ignore
        logging.info("RSI(14) 지표 추가 완료.")

    if 'macd' in indicators:
        macd, macdsignal, macdhist = talib.MACD(df_copy['close'], fastperiod=12, slowperiod=26, signalperiod=9) # type: ignore
        df_copy['macd'] = macd
        df_copy['macdsignal'] = macdsignal
        df_copy['macdhist'] = macdhist
        logging.info("MACD(12, 26, 9) 지표 추가 완료.")
        
    # 다른 지표들도 필요에 따라 추가 가능
    # 예: if 'bollinger' in indicators: ...

    df_copy.dropna(inplace=True)
    logging.info("지표 계산 후 NA 값 제거 완료.")
    
    return df_copy

if __name__ == '__main__':
    # 모듈 테스트 코드
    from pathlib import Path
    import sys

    # 프로젝트 루트 경로 추가
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data.ohlcv_loader import load_ohlcv, resample_ohlcv

    data_file = project_root / 'data' / 'rwa' / 'parquet_converted' / 'btc_kst_1min.parquet'
    
    # 15분봉 데이터 로드
    df_1min = load_ohlcv(data_file, start_date='2024-01-01', end_date='2024-01-31')
    df_15min = resample_ohlcv(df_1min, '15T')
    
    # 지표 추가
    df_with_indicators = add_indicators(df_15min, indicators=['rsi', 'macd'])
    
    print("--- 지표가 추가된 15분봉 데이터 ---")
    print(df_with_indicators.head())
    print("\n--- 데이터프레임 정보 ---")
    df_with_indicators.info()
