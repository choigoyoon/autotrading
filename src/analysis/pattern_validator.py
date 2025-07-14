import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# 모듈 검색 경로에 프로젝트 루트 추가
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from typing import Dict, List, Any
from src.utils.helpers import safe_numeric_conversion

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PatternValidator:
    """
    지그재그 패턴의 통계적 유효성을 검증하는 클래스.
    - 지그재그 이동 주기 분석
    - 패턴과 지표(RSI) 간의 상관관계 분석
    """
    def __init__(self, df: pd.DataFrame, patterns: List[Dict[str, Any]], target_column: str, timeframe: str):
        self.df = df
        self.patterns = patterns
        self.target_column = target_column
        self.timeframe = timeframe
        self.pivots = self._extract_pivots()

    def _extract_pivots(self) -> pd.Series:
        """데이터프레임에서 피봇(-1, 1)만 추출하여 반환합니다."""
        pivots_data = self.df[self.df['pivots'] != 0]['pivots']
        return pd.Series(pivots_data, dtype=float)

    def analyze_zigzag_frequency(self) -> Dict[str, float]:
        """
        지그재그(고점-저점 또는 저점-고점) 이동의 평균 횟수(캔들 수)를 계산합니다.
        """
        if len(self.pivots) < 2:
            return {'avg_candles_per_move': 0.0, 'total_moves': 0.0}
            
        pivot_indices = pd.to_datetime(self.pivots.index)
        time_diffs_seconds = (pivot_indices[1:] - pivot_indices[:-1]).total_seconds()
        
        # timeframe 문자열을 Timedelta로 변환하여 초 단위로 변경
        timeframe_seconds = pd.to_timedelta(self.timeframe).total_seconds()
        
        if timeframe_seconds > 0:
            avg_candles = np.mean(time_diffs_seconds) / timeframe_seconds
        else:
            avg_candles = 0.0

        return {
            'avg_candles_per_move': float(avg_candles),
            'total_moves': float(len(time_diffs_seconds))
        }

    def analyze_rsi_divergence(self, rsi_column='rsi', lookahead_period=24) -> List[Dict[str, Any]]:
        """
        W 패턴에서 RSI 강세 다이버전스의 유효성을 통계적으로 분석합니다.
        """
        divergence_analysis = []
        for p in self.patterns:
            if p.get('type') != 'W': continue

            l1_idx, l2_idx = p.get('l1_idx'), p.get('l2_idx')
            if l1_idx is None or l2_idx is None:
                continue
            
            # W 패턴의 두 저점에서 가격과 RSI 값 추출 (안전한 숫자 변환)
            price_l1 = safe_numeric_conversion(self.df.loc[l1_idx, 'low'], f"price_l1 at {l1_idx}")
            price_l2 = safe_numeric_conversion(self.df.loc[l2_idx, 'low'], f"price_l2 at {l2_idx}")
            rsi_l1 = safe_numeric_conversion(self.df.loc[l1_idx, 'rsi'], f"rsi_l1 at {l1_idx}")
            rsi_l2 = safe_numeric_conversion(self.df.loc[l2_idx, 'rsi'], f"rsi_l2 at {l2_idx}")

            # 하나라도 변환에 실패하면 해당 패턴은 건너뜀
            if any(v is None for v in [price_l1, price_l2, rsi_l1, rsi_l2]):
                logging.warning(f"Skipping pattern due to data conversion issue at {l1_idx} or {l2_idx}.")
                continue

            # pyright가 None 가능성을 계속 제기하므로, assert를 사용하여 None이 아님을 명시
            assert price_l1 is not None and price_l2 is not None
            assert rsi_l1 is not None and rsi_l2 is not None

            # 강세 다이버전스 조건: 가격은 하락/유지, RSI는 상승
            price_condition = price_l2 <= price_l1
            rsi_condition = rsi_l2 > rsi_l1
            
            is_divergence = price_condition and rsi_condition
            
            # 다이버전스 이후 가격 변화 (간단한 백테스트)
            # l2_idx를 Timestamp로 명확히 변환
            l2_timestamp = pd.to_datetime(l2_idx) if isinstance(l2_idx, (str, int)) else l2_idx
            future_end_time = l2_timestamp + pd.Timedelta(hours=lookahead_period)
            future_slice = self.df.loc[l2_timestamp:future_end_time]
            
            price_change_pct = 0.0
            if not future_slice.empty:
                high_series = pd.to_numeric(future_slice['high'], errors='coerce')
                future_high = high_series.max()
                if pd.notna(future_high) and price_l2 > 0:
                    price_change_pct = float(((future_high - price_l2) / price_l2) * 100)
            
            analysis = {
                'pattern_type': 'W',
                'l2_timestamp': l2_idx,
                'is_divergence': is_divergence,
                'rsi_at_l1': rsi_l1,
                'rsi_at_l2': rsi_l2,
                'price_change_pct_24h': price_change_pct
            }
            divergence_analysis.append(analysis)
            
        # logging.info(f"RSI 다이버전스 분석 완료. 총 {len(divergence_analysis)}개 패턴 분석.")
        return divergence_analysis

if __name__ == '__main__':
    # 모듈 테스트 코드
    from src.data.ohlcv_loader import load_ohlcv, resample_ohlcv
    from src.features.indicator_calculator import add_indicators
    from src.patterns.zigzag_detector import ZigzagDetector

    data_file = project_root / 'data' / 'rwa' / 'parquet_converted' / 'btc_kst_1min.parquet'
    
    # 1. 데이터 준비
    df_1min = load_ohlcv(data_file, start_date='2023-01-01', end_date='2023-12-31')
    df_1h = resample_ohlcv(df_1min, '1H')
    df_1h_indicators = add_indicators(df_1h, indicators=['rsi', 'macd'])

    # 2. 패턴 감지
    detector = ZigzagDetector(prominence_threshold=0.01)
    df_with_zigzag = detector.extract_zigzag_patterns(df_1h_indicators, 'close')
    w_patterns = detector.find_w_patterns(df_with_zigzag)
    
    # 3. 패턴 검증 및 분석
    validator = PatternValidator(df_with_zigzag, w_patterns, 'close', '1H')
    
    # 지그재그 주기 분석
    frequency_stats = validator.analyze_zigzag_frequency()
    print(f"\n--- 지그재그 주기 분석 결과 ---")
    print(frequency_stats)
    
    # RSI 다이버전스 분석
    divergence_results = validator.analyze_rsi_divergence()
    df_divergence = pd.DataFrame(divergence_results)
    
    print("\n--- RSI 다이버전스 분석 결과 (상위 5개) ---")
    print(df_divergence.head())
    
    # 다이버전스 발생 여부에 따른 평균 수익률 비교
    avg_return_with_divergence = df_divergence[df_divergence['is_divergence']]['price_change_pct_24h'].mean()
    avg_return_without_divergence = df_divergence[~df_divergence['is_divergence']]['price_change_pct_24h'].mean()
    
    print("\n--- 다이버전스 유무에 따른 24시간 후 평균 가격 변화율 ---")
    print(f"다이버전스 존재 시: {avg_return_with_divergence:.2f}%")
    print(f"다이버전스 미존재 시: {avg_return_without_divergence:.2f}%") 