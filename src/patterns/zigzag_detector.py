import pandas as pd
from scipy.signal import find_peaks
import logging
from typing import Dict, List

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ZigzagDetector:
    """
    가격 및 지표 데이터에서 고점(Peak)과 저점(Valley)을 찾아내고,
    이를 기반으로 지그재그 패턴 및 W/M 패턴을 식별하는 클래스.
    """
    def __init__(self, prominence_threshold: float = 0.01):
        """
        Args:
            prominence_threshold (float): find_peaks에서 사용할 prominence 값.
                                          가격 변동의 중요도를 결정하며, 노이즈를 필터링하는 역할을 함.
        """
        self.prominence = prominence_threshold

    def find_pivots(self, series: pd.Series) -> pd.Series:
        """
        주어진 Series에서 고점과 저점을 찾아 인덱스를 반환합니다.
        고점은 1, 저점은 -1로 표시합니다.
        """
        # rolling 연산 후 Series 타입 보장
        smoothed = series.rolling(window=3, center=True).mean()
        if isinstance(smoothed, pd.Series):
            series_smooth = smoothed.bfill().ffill()
        else:
            # numpy 배열일 경우를 대비한 처리 (일반적으로는 발생하지 않음)
            temp_series = pd.Series(smoothed)
            series_smooth = temp_series.bfill().ffill()

        peak_indices, _ = find_peaks(series_smooth, prominence=(series.max() - series.min()) * self.prominence)
        valley_indices, _ = find_peaks(-series_smooth, prominence=(series.max() - series.min()) * self.prominence)
        
        pivots = pd.Series(0, index=series.index)
        pivots.iloc[peak_indices] = 1
        pivots.iloc[valley_indices] = -1
        
        return pivots

    def extract_zigzag_patterns(self, df: pd.DataFrame, target_column: str = 'close') -> pd.DataFrame:
        """
        데이터프레임에 'pivots'와 'zigzag' 열을 추가합니다.
        'pivots'는 고점(1)/저점(-1)을 표시하고, 'zigzag'는 지그재그 값을 포함합니다.
        """
        df_copy = df.copy()
        
        # target_column이 Series인지 확인
        target_series = df_copy[target_column]
        if not isinstance(target_series, pd.Series):
             # 에러를 발생시키는 대신, Series로 변환 시도
            target_series = pd.Series(target_series)

        pivots = self.find_pivots(target_series)
        df_copy['pivots'] = pivots
        
        # 지그재그 값 계산
        last_pivot_val = 0
        df_copy.index[0]
        zigzag_values = []

        for idx, row in df_copy.iterrows():
            if row['pivots'] != 0:
                last_pivot_val = row[target_column]
            zigzag_values.append(last_pivot_val)
        
        df_copy['zigzag'] = zigzag_values
        
        logging.info(f"'{target_column}'에 대한 지그재그 패턴 추출 완료.")
        return df_copy
        
    def find_w_patterns(self, df_with_pivots: pd.DataFrame) -> List[Dict]:
        """ W (이중 바닥) 패턴을 찾습니다. """
        patterns = []
        pivots = df_with_pivots[df_with_pivots['pivots'] == -1] # 저점만 필터링
        
        if len(pivots) < 2:
            return patterns

        for i in range(len(pivots) - 1):
            l1_idx, l2_idx = pivots.index[i], pivots.index[i+1]
            l1_price, l2_price = pivots.iloc[i]['low'], pivots.iloc[i+1]['low']
            
            # 두 저점 사이의 고점 (넥라인)
            intermediate_slice = df_with_pivots.loc[l1_idx:l2_idx]
            neckline_idx = intermediate_slice[intermediate_slice['pivots'] == 1].index.max()
            
            if pd.isna(neckline_idx): continue
                
            neckline_price = df_with_pivots.loc[neckline_idx, 'high']
            
            patterns.append({
                'type': 'W',
                'l1_idx': l1_idx, 'l1_price': l1_price,
                'l2_idx': l2_idx, 'l2_price': l2_price,
                'neckline_idx': neckline_idx, 'neckline_price': neckline_price
            })
            
        logging.info(f"{len(patterns)}개의 W 패턴 후보 감지.")
        return patterns

if __name__ == '__main__':
    # 모듈 테스트 코드
    from pathlib import Path
    import sys

    # 프로젝트 루트 경로 추가
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data.ohlcv_loader import load_ohlcv, resample_ohlcv
    from src.features.indicator_calculator import add_indicators

    data_file = project_root / 'data' / 'rwa' / 'parquet_converted' / 'btc_kst_1min.parquet'
    
    # 데이터 준비
    df_1min = load_ohlcv(data_file, start_date='2024-01-01', end_date='2024-02-29')
    df_1h = resample_ohlcv(df_1min, '1H')
    df_1h_indicators = add_indicators(df_1h)

    # 지그재그 패턴 감지
    detector = ZigzagDetector(prominence_threshold=0.015)
    df_with_zigzag = detector.extract_zigzag_patterns(df_1h_indicators, 'close')
    
    print("--- 지그재그 패턴이 추가된 데이터 ---")
    print(df_with_zigzag[df_with_zigzag['pivots'] != 0].head())
    
    # W 패턴 찾기
    w_patterns = detector.find_w_patterns(df_with_zigzag)
    print(f"\n--- 감지된 W 패턴 (상위 5개) ---")
    for p in w_patterns[:5]:
        print(p)
        
    # 시각화 (선택적)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 7))
        plt.plot(df_with_zigzag.index, df_with_zigzag['close'], label='Close Price')
        plt.plot(df_with_zigzag.index, df_with_zigzag['zigzag'], label='Zigzag', color='gray', linestyle='--')
        
        pivots_data = df_with_zigzag[df_with_zigzag['pivots'] != 0]
        plt.scatter(pivots_data.index, pivots_data['close'], c=pivots_data['pivots'], cmap='viridis', s=50, zorder=5)
        
        plt.title("Zigzag Detection on Close Price")
        plt.legend()
        plt.show()
    except ImportError:
        logging.warning("Matplotlib이 설치되지 않아 시각화는 건너뜁니다.") 