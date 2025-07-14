# type: ignore
# pylint: disable-all
import pandas as pd
import numpy as np
import logging
# from numba import njit # 현재 로직에서는 Numba 직접 사용 안함
# from typing import List, Tuple, Optional # Optional은 필요할 수 있음
from typing import Optional # Optional만 남김 또는 필요시 추가
import numpy as np
import pandas as pd

def calculate_macd_optimized(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, use_numba: bool = True) -> tuple:
    """
    최적화된 MACD, 시그널, 히스토그램 계산
    
    Args:
        prices: 가격 시리즈
        fast: 단기 이동평균 기간 (기본값: 12)
        slow: 장기 이동평균 기간 (기본값: 26)
        signal: 시그널 기간 (기본값: 9)
        use_numba: Numba 가속 사용 여부 (향후 사용 예정, 현재는 무시됨)
        
    Returns:
        tuple: (macd, signal, histogram)
    """
    # 지수이동평균(EMA) 계산
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    
    # MACD 라인 계산 (단기 EMA - 장기 EMA)
    macd = exp1 - exp2
    
    # 시그널 라인 계산 (MACD의 signal 기간 EMA)
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # 히스토그램 계산 (MACD - 시그널라인)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

logger = logging.getLogger(__name__)

# _find_price_extremes 함수는 더 이상 사용되지 않으므로 주석 처리 또는 삭제
# @njit
# def _find_price_extremes(labels: np.ndarray, high: np.ndarray, low: np.ndarray, segments: List[Tuple[int, int, int]], hist_start_idx: int):
#     ...

class EnhancedMACDLabeler:
    """
    간단한 MACD 히스토그램 임계값 교차 기반 라벨러
    
    이 클래스는 다양한 타임프레임(1분, 5분, 15분, 1시간, 1일 등)의 가격 데이터에 대해
    MACD 히스토그램을 기반으로 매수/매도/관망 라벨을 생성합니다.
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, use_numba: bool = True, **kwargs):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.use_numba_macd = use_numba # MACD 계산 시 Numba 사용 여부
        logger.info(
            f"EnhancedMACDLabeler (Simple Threshold Cross) initialized with MACD params: "
            f"fast={self.fast}, slow={self.slow}, signal={self.signal}, use_numba={self.use_numba_macd}"
        )
        if kwargs:
            logger.debug(f"Unused kwargs in EnhancedMACDLabeler __init__: {kwargs}")

    def generate_enhanced_labels(self, df: pd.DataFrame) -> pd.Series:
        """MACD 히스토그램 부호 기반 라벨링"""
        if 'close' not in df.columns:
            logger.error("DataFrame must contain 'close' column for MACD calculation.")
            raise ValueError("DataFrame must contain 'close' column.")
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty Series.")
            return pd.Series(dtype=np.int8)

        # MACD 계산
        _, _, hist = calculate_macd_optimized(df['close'], 
                                            fast=self.fast, 
                                            slow=self.slow, 
                                            signal=self.signal, 
                                            use_numba=self.use_numba_macd)
        
        if hist.isnull().all():
            logger.warning("MACD histogram is all NaNs. Returning zero labels.")
            return pd.Series(0, index=df.index, dtype=np.int8)
            
        logger.info("Generating labels based on MACD histogram sign.")
        
        # 라벨 초기화
        labels = pd.Series(0, index=df.index, dtype=np.int8)
        
        # 벡터화된 방식으로 라벨 부여 (기존 루프 방식 제거)
        labels[hist < 0] = 1  # 히스토그램 음수 -> 매수
        labels[hist > 0] = -1 # 히스토그램 양수 -> 매도
        
        # 분포 출력 (개선된 포맷)
        total_labels = len(labels)
        buy_count = (labels == 1).sum()
        sell_count = (labels == -1).sum()
        hold_count = (labels == 0).sum()

        buy_pct = (buy_count / total_labels * 100) if total_labels > 0 else 0
        sell_pct = (sell_count / total_labels * 100) if total_labels > 0 else 0
        hold_pct = (hold_count / total_labels * 100) if total_labels > 0 else 0

        # 여러 줄 f-string을 사용하여 로그 메시지 구성
        dist_log_msg = f"""Generated labels distribution (Simple Threshold Cross):
  매수(+1): {buy_pct:>6.2f}% ({buy_count:>7,}개)
  매도(-1): {sell_pct:>6.2f}% ({sell_count:>7,}개)
  관망(0): {hold_pct:>6.2f}% ({hold_count:>7,}개)
  총 라벨: {total_labels:>12,}개"""
        logger.info(dist_log_msg)
        
        return labels

    def generate_enhanced_labels_old(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """MACD 히스토그램 부호 변화 기반 간단한 라벨링."""
        
        if 'close' not in df.columns:
            logger.error("DataFrame must contain 'close' column for MACD calculation.")
            raise ValueError("DataFrame must contain 'close' column.")
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty Series.")
            return pd.Series(dtype=np.int8)

        logger.debug(f"Generating zero-cross MACD labels for df shape: {df.shape}. Received (and unused) kwargs: {kwargs}")

        # MACD 계산
        _, _, hist_series = calculate_macd_optimized(
            df['close'], 
            fast=self.fast, 
            slow=self.slow, 
            signal=self.signal,
            use_numba=self.use_numba_macd 
        )
        
        # hist_series가 numpy 배열일 수 있으므로 Series로 변환하고 df의 인덱스와 맞춤
        if not isinstance(hist_series, pd.Series):
            if len(hist_series) == len(df):
                hist = pd.Series(hist_series, index=df.index)
            elif len(hist_series) < len(df):
                hist = pd.Series(hist_series, index=df.index[len(df)-len(hist_series):])
            else: 
                hist = pd.Series(hist_series[:len(df)], index=df.index)
        else:
            hist = hist_series 
            
        if hist.empty:
            logger.warning("MACD calculation resulted in an empty histogram. Returning zero labels.")
            return pd.Series(0, index=df.index, dtype=np.int8)
            
        # 라벨 초기화 (df의 전체 인덱스 기준)
        labels = pd.Series(0, index=df.index, dtype=np.int8)
        
        # MACD 히스토그램이 계산된 유효한 범위 (NaN 아닌 값) 확보
        hist_for_labeling = hist.dropna()
        if hist_for_labeling.empty:
            logger.warning("MACD histogram contains all NaNs after dropna. Returning zero labels.")
            return labels

        # 부호 변화 지점 찾기
        # hist_for_labeling의 인덱스는 df.index의 부분집합일 수 있음
        # hist_values는 실제 값, hist_indices는 해당 값의 원본 df 인덱스
        hist_values = hist_for_labeling.values
        hist_indices = hist_for_labeling.index

        # 첫 번째 데이터(1행)도 라벨링 대상으로 포함
        for i in range(len(hist_values)):
            if i == 0:
                # 첫 번째 데이터는 이전 값이 없으므로, 부호에 따라 라벨 지정 (예: 음수면 매수, 양수면 매도, 0이면 관망)
                if hist_values[i] < 0:
                    labels.loc[hist_indices[i]] = 1  # 매수
                elif hist_values[i] > 0:
                    labels.loc[hist_indices[i]] = -1  # 매도
                else:
                    labels.loc[hist_indices[i]] = 0  # 관망
            else:
                prev_hist_val = hist_values[i-1]
                current_hist_val = hist_values[i]
                actual_label_idx = hist_indices[i] # 현재 값의 인덱스에 라벨링

                # 음수 -> 양수 (또는 0): 골짜기 통과 (매수)
                if prev_hist_val < 0 and current_hist_val >= 0:
                    labels.loc[actual_label_idx] = 1
                # 양수 -> 음수 (또는 0): 산봉우리 통과 (매도)
                elif prev_hist_val > 0 and current_hist_val <= 0:
                    labels.loc[actual_label_idx] = -1
        
        if not labels.empty:
            buy_count = (labels == 1).sum()
            sell_count = (labels == -1).sum() 
            hold_count = (labels == 0).sum()
            total_labels_generated = len(labels)
            
            buy_pct = buy_count / total_labels_generated * 100 if total_labels_generated > 0 else 0
            sell_pct = sell_count / total_labels_generated * 100 if total_labels_generated > 0 else 0
            hold_pct = hold_count / total_labels_generated * 100 if total_labels_generated > 0 else 0

            logger.info(f"Generated labels distribution (Zero Cross Version):\n"
                       f"  Buy(+1): {buy_pct:>6.2f}% ({buy_count:>7,}개)\n"
                       f"  Sell(-1): {sell_pct:>5.2f}% ({sell_count:>7,}개)\n"
                       f"  Hold(0): {hold_pct:>6.2f}% ({hold_count:>7,}개)\n"
                       f"  Total: {total_labels_generated:>14,}개 on {len(df)} input rows")
        else:
            logger.info("Generated empty labels (Zero Cross Version).")
            
        return labels.astype(np.int8) 


def label_all_timeframes(timeframe_data_dict, config):
    """
    각 타임프레임 데이터 전체(1행~끝행)에 대해 MACD 라벨링을 수행합니다.

    Args:
        timeframe_data_dict (dict): {타임프레임(str): DataFrame} 형태의 딕셔너리
        config: PipelineConfig 인스턴스

    Returns:
        dict: {타임프레임(str): 라벨 Series}
    """
    labeler = EnhancedMACDLabeler()
    labels_per_timeframe = {}

    for tf in config.labeling_timeframes:  # 15개 타임프레임
        df = timeframe_data_dict.get(tf)
        if df is None or df.empty:
            print(f"[{tf}] 데이터가 없습니다. 건너뜁니다.")
            continue
        # 1행~끝행 전체 라벨링
        labels = labeler.generate_enhanced_labels(df)
        labels_per_timeframe[tf] = labels
        print(f"[{tf}] 라벨링 완료: {len(labels)}개")

    return labels_per_timeframe

def save_labels_to_files(labels_dict, ohlcv_dict=None, output_dir="data/labels_macd/btc_usdt_kst"):
    """
    ohlcv 데이터와 라벨을 합쳐서 parquet/csv 파일로 저장합니다.
    labels_dict: {타임프레임: 라벨 Series}
    ohlcv_dict: {타임프레임: DataFrame} (ohlcv 데이터)
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tf, labels in labels_dict.items():
        output_file_parquet = output_dir / f"macd_labels_{tf}.parquet"
        output_file_csv = output_dir / f"macd_labels_{tf}.csv"

        # ohlcv 데이터와 라벨을 합쳐서 저장
        if ohlcv_dict is not None and tf in ohlcv_dict:
            df_ohlcv = ohlcv_dict[tf].copy()
            # 인덱스를 맞추고 concat으로 합침
            df_result = pd.concat([df_ohlcv, labels.rename('label')], axis=1)
            print(f"[디버그] 저장 직전 데이터 샘플 ({tf}):\n{df_result.head()}\n")
            print(f"[디버그] 저장 직전 컬럼명: {df_result.columns}")
            print(f"[디버그] 저장 직전 shape: {df_result.shape}")
            # 혹시 MultiIndex 컬럼이면 단일 컬럼명으로 변환
            if isinstance(df_result.columns, pd.MultiIndex):
                df_result.columns = ['_'.join(col).strip() for col in df_result.columns.values]
                print(f"[디버그] MultiIndex 컬럼 변환 후: {df_result.columns}")
            df_result.to_parquet(output_file_parquet)
            df_result.to_csv(output_file_csv, encoding='utf-8-sig', index=True, header=True)
            # 저장 직후 파일을 다시 읽어서 실제로 컬럼이 어떻게 저장됐는지 확인
            df_check = pd.read_csv(output_file_csv, encoding='utf-8-sig')
            print(f"[디버그] 저장 후 파일에서 읽은 데이터 샘플 ({tf}):\n{df_check.head()}\n")
            print(f"[디버그] 저장 후 컬럼명: {df_check.columns}\n")
        else:
            # ohlcv_dict가 없는 경우 기존 방식대로 라벨만 저장(호환성)
            labels.to_frame(name='label').to_parquet(output_file_parquet)

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.config.settings import PipelineConfig
    config = PipelineConfig()

    # 타임프레임 목록
    timeframes = [
        "1min", "3min", "5min", "10min", "15min", "30min", "1h", "2h", "4h", "6h", "8h", "12h", "1D", "2D", "3D", "1W"
    ]
    ohlcv_base_dir = 'data/processed/btc_usdt_kst/resampled_ohlcv'
    all_labels = {}
    all_ohlcv = {}  # 각 타임프레임별 ohlcv 데이터 저장

    # 1min 데이터 미리 로드
    base_ohlcv_path = f"{ohlcv_base_dir}/1min.parquet"
    base_df = pd.read_parquet(base_ohlcv_path)

    for tf in timeframes:
        ohlcv_path = f"{ohlcv_base_dir}/{tf}.parquet"
        try:
            # 파일이 존재하면 그대로 사용
            if os.path.exists(ohlcv_path):
                df = pd.read_parquet(ohlcv_path)
                print(f"[{tf}] OHLCV 데이터: {len(df)}행, 기간: {df.index[0]} ~ {df.index[-1]}")
            else:
                # 파일이 없으면 1min 데이터에서 리샘플링
                print(f"[{tf}] OHLCV 파일 없음 → 1min 데이터에서 리샘플링 생성")
                df = base_df.resample(tf).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                print(f"[{tf}] 리샘플링 데이터: {len(df)}행, 기간: {df.index[0]} ~ {df.index[-1]}")
            # MACD 라벨 생성 (close 컬럼 기준)
            labeler = EnhancedMACDLabeler()
            labels = labeler.generate_enhanced_labels(df)
            all_labels[tf] = labels
            all_ohlcv[tf] = df  # ohlcv 데이터도 함께 저장
        except Exception as e:
            print(f"[{tf}] 데이터 로드/라벨링 실패(리샘플 포함): {e}")
    # 라벨 파일 저장 (ohlcv 데이터와 함께)
    save_labels_to_files(all_labels, all_ohlcv)
    print("모든 라벨 파일 저장 완료!")