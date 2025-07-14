# src/ml/inflection_data_preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, cast
from scipy.signal import find_peaks

class InflectionDataPreprocessor:
    """
    InflectionMagnitudePredictor 모델 학습을 위한 데이터를 생성하는 전처리기.

    주요 기능:
    1. 가격 데이터에서 의미있는 변곡점을 찾습니다.
    2. 각 변곡점에 대해 '컨텍스트(Context)'와 '순간(Moment)' 특징을 추출합니다.
    3. 각 변곡점 이후의 실제 가격 움직임을 분석하여 정답 레이블(강도, 기간, 수익률)을 생성합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        전처리기 초기화

        Args:
            config (Dict[str, Any]): 데이터 경로 및 전처리 관련 설정.
        """
        self.config = config
        self.raw_data_path = Path(config.get("raw_data_path", "data/rwa/parquet/btc_1min.parquet"))
        # 필요한 다른 데이터 경로들 (예: 타임프레임별 레이블)
        # self.label_base_path = Path(config.get("label_base_path", "data/labels/btc_usdt_kst"))
        self.ohlcv_1d = self._load_and_resample_daily()

    def _load_and_resample_daily(self) -> pd.DataFrame:
        """1분봉 데이터를 로드하여 일봉으로 리샘플링하고 기본 지표를 추가합니다."""
        print("Loading and resampling to daily OHLCV...")
        df = pd.read_parquet(self.raw_data_path)
        df.index = pd.to_datetime(df.index)
        
        daily_df = df['close'].resample('D').ohlc()
        daily_df['volume'] = df['volume'].resample('D').sum()
        
        # 기본 지표 추가 (향후 컨텍스트 특징으로 활용)
        daily_df['ma5'] = daily_df['close'].rolling(window=5).mean()
        daily_df['ma20'] = daily_df['close'].rolling(window=20).mean()
        return daily_df.dropna()

    def find_inflection_points(self, prominence: float = 0.1) -> pd.DataFrame:
        """
        일봉 데이터에서 의미있는 저점(변곡점)을 찾습니다.
        
        Args:
            prominence (float): 피크의 돌출 정도. 클수록 더 중요한 피크만 선택됩니다.

        Returns:
            pd.DataFrame: 변곡점의 타임스탬프와 가격을 담은 데이터프레임.
        """
        print("Finding significant inflection points (lows)...")
        # 저점을 찾기 위해 가격 데이터에 음수를 취함
        lows_indices, _ = find_peaks(-self.ohlcv_1d['low'], prominence=self.ohlcv_1d['low'].mean() * prominence)
        
        inflection_points = self.ohlcv_1d.iloc[lows_indices]
        inflection_df = inflection_points[['low']].copy()
        inflection_df.rename(columns={'low': 'inflection_price'}, inplace=True)
        print(f"Found {len(inflection_df)} potential inflection points.")
        return inflection_df

    def _extract_context_sequence(self, inflection_timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """주어진 변곡점 이전 30일간의 컨텍스트 특징을 추출합니다."""
        end_date = inflection_timestamp
        start_date = end_date - pd.Timedelta(days=30)
        
        context_df = self.ohlcv_1d.loc[start_date:end_date]
        
        if len(context_df) < 30:
            return None # 데이터가 충분하지 않으면 None 반환

        # TODO: 요구사항에 명시된 특징들 추출
        # 1. 하락 지속 기간
        # 2. 거래량 패턴 (고갈 -> 폭증)
        # 3. 변동성 압축 정도 (e.g., Bollinger Band Width)
        # 4. 주요 지지선과의 거리
        # 5. 전체 시장 환경 (e.g., 장기 이평선 배열)
        
        # 임시로 OHLCV와 이동평균선을 특징으로 사용
        features = context_df[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20']].values
        
        # 특징 차원을 10으로 맞추기 위해 임시 패딩
        padded_features = np.pad(features, ((0, 0), (0, 3)), 'constant')

        return padded_features[-30:] # 정확히 30일치만 반환

    def _extract_moment_features(self, inflection_timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """변곡점 순간의 특징을 추출합니다."""
        
        # TODO: 요구사항에 명시된 특징들 추출
        # 1. 다중 타임프레임 합의 강도
        # 2. 거래량 폭증 정도
        # 3. 모멘텀 다이버전스 강도
        # 4. 패턴 완성도 (e.g., Double Bottom, etc.)
        
        # 임시로 15개의 무작위 값을 특징으로 사용
        moment_features = np.random.rand(15)
        return moment_features

    def _calculate_labels(self, inflection_timestamp: pd.Timestamp, inflection_price: float) -> Optional[Tuple[int, int, float]]:
        """변곡점 이후의 가격 움직임을 분석하여 3가지 레이블을 생성합니다."""
        
        future_data = self.ohlcv_1d.loc[inflection_timestamp:]
        if len(future_data) < 90: # 최소 3달의 데이터가 필요하다고 가정
            return None

        # 1. 최대 수익률 계산
        max_price_after = future_data['high'][1:].max() # 변곡점 다음날부터
        max_return = (max_price_after - inflection_price) / inflection_price

        # 2. 반등 강도 분류
        # TODO: 수익률 분포를 보고 경계값(threshold) 재설정 필요
        if max_return > 1.0: magnitude_class = 3 # 폭발적 (100% 초과)
        elif max_return > 0.5: magnitude_class = 2 # 강함 (50% 초과)
        elif max_return > 0.2: magnitude_class = 1 # 보통 (20% 초과)
        else: magnitude_class = 0 # 약함

        # 3. 지속 기간 분류
        # TODO: 실제 고점 도달 시간을 기준으로 재계산 필요
        duration_class = np.random.randint(0, 4) # 임시값
        
        return magnitude_class, duration_class, max_return

    def run(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int], List[float]]:
        """전체 데이터 전처리 파이프라인을 실행합니다."""
        inflection_points = self.find_inflection_points()

        all_contexts = []
        all_moments = []
        all_mag_labels = []
        all_dur_labels = []
        all_ret_labels = []

        print("Generating features and labels for each inflection point...")
        for timestamp, row in inflection_points.iterrows():
            ts = cast(pd.Timestamp, timestamp)
            context_seq = self._extract_context_sequence(ts)
            moment_features = self._extract_moment_features(ts)
            labels = self._calculate_labels(ts, row['inflection_price'])

            if context_seq is not None and moment_features is not None and labels is not None:
                all_contexts.append(context_seq)
                all_moments.append(moment_features)
                all_mag_labels.append(labels[0])
                all_dur_labels.append(labels[1])
                all_ret_labels.append(labels[2])
        
        print(f"Successfully generated {len(all_contexts)} training samples.")
        return all_contexts, all_moments, all_mag_labels, all_dur_labels, all_ret_labels

if __name__ == '__main__':
    # --- 전처리기 테스트 ---
    config = {"raw_data_path": "data/rwa/parquet/btc_1min.parquet"}
    preprocessor = InflectionDataPreprocessor(config)
    
    contexts, moments, mag_labels, dur_labels, ret_labels = preprocessor.run()
    
    if contexts:
        print("\n--- Preprocessor Test Output ---")
        print(f"Number of samples: {len(contexts)}")
        print(f"Context sequence shape: {contexts[0].shape}")
        print(f"Moment features shape: {moments[0].shape}")
        print(f"Sample Magnitude Label: {mag_labels[0]}")
        print(f"Sample Duration Label: {dur_labels[0]}")
        print(f"Sample Return Label: {ret_labels[0]:.4f}") 