# src/ml/data_preprocessor.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

class MultiTimeframeDataPreprocessor:
    """
    MultiTimeframePredictor 모델 학습을 위한 데이터를 준비하는 전처리기.
    
    Holding/Breakeven 거래 로그와 각 타임프레임별 레이블을 기반으로
    모델 입력 특징(features)과 정답(labels)을 생성합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_base_path = Path(config.get("label_base_path", "data/labels/btc_usdt_kst"))
        self.timeframes = config.get("timeframes", [
            '1min', '3min', '5min', '10min', '15min', '30min', '1h', '2h', '4h', 
            '6h', '8h', '12h', '1day', '3day', '1week', '1month'
        ])
        # 원본 레이블 캐시
        self.labels_cache = self._load_all_labels()
        # merge_asof 성능 최적화를 위해 미리 처리된 레이블 캐시
        self.processed_labels_cache = self._preprocess_labels_for_merge()

    def _load_all_labels(self) -> Dict[str, pd.DataFrame]:
        """모든 타임프레임의 레이블 파일을 미리 로드합니다."""
        cache = {}
        for tf in self.timeframes:
            file_path = self.label_base_path / f"macd_zone_{tf}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df.index = pd.to_datetime(df.index)
                df.index.name = 'timestamp'
                cache[tf] = df
        return cache

    def _preprocess_labels_for_merge(self) -> Dict[str, pd.DataFrame]:
        """백테스팅 시 merge_asof의 성능을 위해 레이블을 미리 처리합니다."""
        processed_cache = {}
        for tf, df in self.labels_cache.items():
            if df is not None:
                # 인덱스를 리셋하고 timestamp로 정렬하여 캐시에 저장
                processed_cache[tf] = df.reset_index().sort_values('timestamp')
        return processed_cache

    def load_trade_logs(self) -> pd.DataFrame:
        """Holding 및 Breakeven 거래 로그를 로드하고 통합합니다."""
        holding_path = Path(self.config.get("holding_trades_path"))
        breakeven_path = Path(self.config.get("breakeven_trades_path"))
        
        holding_df = pd.read_csv(holding_path)
        breakeven_df = pd.read_csv(breakeven_path)
        
        holding_df['label'] = 1 # Non-Reversion
        breakeven_df['label'] = 0 # Reversion
        
        combined_df = pd.concat([holding_df, breakeven_df], ignore_index=True)
        combined_df['진입시점'] = pd.to_datetime(combined_df['진입시점'])
        # 수익률 데이터를 float으로 변환
        combined_df['최대수익률(%)'] = pd.to_numeric(combined_df['최대수익률(%)'], errors='coerce').fillna(0.0)
        return combined_df

    def create_features_for_trade(self, trade_row: pd.Series) -> Tuple[np.ndarray, int, float]:
        """단일 거래에 대한 특징 매트릭스, 레이블, 수익률을 생성합니다."""
        timestamp = trade_row['진입시점']
        signal_type = trade_row['신호타입']
        label = trade_row['label']
        max_return = trade_row['최대수익률(%)'] / 100.0 # 백분율을 소수로 변환

        feature_matrix = []
        
        # 합의 및 구조적 강도 계산을 위한 임시 변수
        consensus_signals = []
        has_long_term_signal = False

        for tf in self.timeframes:
            df = self.labels_cache.get(tf)
            has_signal = 0.0
            
            if df is not None:
                idx = df.index.searchsorted(timestamp, side='right') - 1
                if idx >= 0:
                    latest_label = df.iloc[idx]
                    time_diff = timestamp - latest_label.name
                    
                    # 시간 단위를 pandas가 이해할 수 있는 형식으로 변환
                    try:
                        if 'min' in tf:
                            td = pd.to_timedelta(int(tf.replace('min', '')), unit='m')
                        elif 'h' in tf:
                            td = pd.to_timedelta(int(tf.replace('h', '')), unit='h')
                        elif 'day' in tf:
                            td = pd.to_timedelta(int(tf.replace('day', 'd')), unit='d')
                        elif 'week' in tf:
                             td = pd.to_timedelta(int(tf.replace('week', '')), unit='W')
                        elif 'month' in tf:
                             td = pd.to_timedelta(int(tf.replace('month', '')) * 30, unit='d') # 30일로 근사
                        else:
                            td = pd.to_timedelta(0)

                        # 해당 타임프레임의 2캔들 내에 발생한 유효한 신호인지 확인
                        if latest_label['label'] == signal_type and time_diff < (td * 2):
                            has_signal = 1.0
                            consensus_signals.append(tf)
                            if tf in ['1week', '1month']:
                                has_long_term_signal = True
                    except ValueError:
                        pass # 변환 실패 시 무시
            
            # Feature_1: 신호 유무
            # Feature_2: 신호 강도 (Placeholder)
            # Feature_3: 합의 강도 (현재까지의 합의 비율)
            # Feature_4: 구조적 강도 (월/주봉 포함 여부)
            # Feature_5: 신호 일관성 (모두 같은 방향인지, 여기서는 항상 True)
            features = [
                has_signal,
                0.5, # 신호 강도 (임시값)
                len(consensus_signals) / (self.timeframes.index(tf) + 1),
                1.0 if has_long_term_signal else 0.0,
                1.0 # 신호 일관성 (임시값)
            ]
            feature_matrix.append(features)
            
        return np.array(feature_matrix), label, max_return

    def run(self) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """전체 데이터셋에 대한 전처리 파이프라인을 실행합니다."""
        trade_logs = self.load_trade_logs()
        
        all_features = []
        all_labels = []
        all_returns = []
        
        for _, row in trade_logs.iterrows():
            features, label, max_return = self.create_features_for_trade(row)
            all_features.append(features)
            all_labels.append(label)
            all_returns.append(max_return)
            
        return all_features, all_labels, all_returns

    def get_features_for_single_trade(self, timestamp: pd.Timestamp, signal_type: str) -> np.ndarray:
        """단일 거래 시점에 대한 특징 매트릭스를 생성합니다. (백테스팅용)"""
        feature_matrix = []
        consensus_signals = []
        has_long_term_signal = False

        for tf in self.timeframes:
            df = self.labels_cache.get(tf)
            has_signal = 0.0
            
            if df is not None:
                idx = df.index.searchsorted(timestamp, side='right') - 1
                if idx >= 0:
                    latest_label = df.iloc[idx]
                    time_diff = timestamp - latest_label.name
                    
                    try:
                        # 'T' 대신 'min'을 사용하여 FutureWarning 방지
                        if 'min' in tf:
                            td = pd.to_timedelta(int(tf.replace('min', '')), unit='m')
                        elif 'h' in tf:
                            td = pd.to_timedelta(int(tf.replace('h', '')), unit='h')
                        elif 'day' in tf:
                            td = pd.to_timedelta(int(tf.replace('day', 'd')), unit='d')
                        elif 'week' in tf:
                             td = pd.to_timedelta(int(tf.replace('week', '')), unit='W')
                        elif 'month' in tf:
                             td = pd.to_timedelta(int(tf.replace('month', '')) * 30, unit='d') # 30일로 근사
                        else:
                            td = pd.to_timedelta(0)

                        if latest_label['label'] == signal_type and time_diff < (td * 2):
                            has_signal = 1.0
                            consensus_signals.append(tf)
                            if tf in ['1week', '1month']:
                                has_long_term_signal = True
                    except ValueError:
                        pass
            
            features = [
                has_signal,
                0.5,
                len(consensus_signals) / (self.timeframes.index(tf) + 1),
                1.0 if has_long_term_signal else 0.0,
                1.0
            ]
            feature_matrix.append(features)
            
        return np.array(feature_matrix)

    def get_features_for_batch(self, trades_df: pd.DataFrame) -> np.ndarray:
        """
        여러 거래(배치)에 대한 특징 매트릭스를 벡터화 연산을 통해 한 번에 생성합니다.
        (백테스팅 속도 개선용)

        Args:
            trades_df (pd.DataFrame): '진입시점'과 '신호타입' 컬럼을 포함하는 거래 로그.

        Returns:
            np.ndarray: (거래 수, 타임프레임 수, 특징 수) 형태의 3D 배열.
        """
        num_trades = len(trades_df)
        num_timeframes = len(self.timeframes)
        num_features = 5  # 특징 개수

        # 최종 결과를 저장할 3D 배열 초기화
        feature_array = np.zeros((num_trades, num_timeframes, num_features))
        
        # 원본 데이터의 순서를 보존하기 위해 임시 인덱스를 생성
        trades_df = trades_df.copy()
        trades_df['original_order'] = np.arange(num_trades)
        trades_df['진입시점'] = pd.to_datetime(trades_df['진입시점'])

        has_long_term_signal = np.zeros(num_trades, dtype=bool)

        for i, tf in enumerate(self.timeframes):
            # 미리 처리된 레이블 데이터를 캐시에서 가져옴
            df_labels = self.processed_labels_cache.get(tf)
            if df_labels is None:
                continue

            # merge_asof를 사용하여 모든 거래에 대해 가장 가까운 과거 라벨을 한 번에 찾음
            merged = pd.merge_asof(
                trades_df.sort_values('진입시점'),
                df_labels, # 이미 정렬 및 인덱스 리셋됨
                left_on='진입시점',
                right_on='timestamp',
                direction='backward'
            )
            # 원본 순서로 복원하기 위해 'original_order'를 인덱스로 설정
            merged = merged.set_index('original_order').sort_index()

            # 시간 차이 계산
            time_diff = merged['진입시점'] - merged['timestamp']

            # 해당 타임프레임의 유효 기간 (2캔들)
            try:
                # 'T' 대신 'min'을 사용하여 FutureWarning 방지
                td_str = tf.replace('min', 'min').replace('day', 'D').replace('week', 'W').replace('month', 'M')
                td = pd.to_timedelta(td_str)
                time_limit = td * 2
            except (ValueError, TypeError):
                time_limit = pd.to_timedelta(0)

            # 조건에 맞는 신호를 벡터화 연산으로 한 번에 찾음
            valid_signals = (merged['label'] == merged['신호타입']) & (time_diff < time_limit)
            
            # Feature_1: 신호 유무
            feature_array[:, i, 0] = valid_signals.astype(float)
            
            # Feature_2: 신호 강도 (정규화된 MACD 히스토그램 값)
            # 신호가 유효할 때만 값을 채우고, 나머지는 0으로 둠
            # macd_hist 값을 -1과 1 사이로 클리핑하여 이상치 제어 후 0~1 사이로 정규화
            normalized_hist = (np.clip(merged['macd_hist'], -1, 1) + 1) / 2
            feature_array[:, i, 1] = np.where(valid_signals, normalized_hist, 0)

            # Feature_3: 합의 강도
            # 현재 타임프레임까지의 누적 신호 합계를 기반으로 계산
            consensus_signals = (feature_array[:, :i+1, 0].sum(axis=1)) / (i + 1)
            feature_array[:, i, 2] = consensus_signals
            
            # Feature_4: 구조적 강도
            feature_array[:, i, 3] = has_long_term_signal.astype(float)
            
            # Feature_5: 신호 일관성 (Placeholder)
            feature_array[:, i, 4] = 1.0
            
            # valid_signals는 현재 배치의 결과이므로, 이전 배치의 상태를 유지하며 업데이트
            has_long_term_signal |= valid_signals
            
        return feature_array

if __name__ == '__main__':
    config = {
        "label_base_path": "data/labels/btc_usdt_kst",
        "holding_trades_path": "results/top_trades/top_100_holding_trades.csv",
        "breakeven_trades_path": "results/top_trades/top_100_breakeven_trades.csv",
    }
    
    preprocessor = MultiTimeframeDataPreprocessor(config)
    features, labels, returns = preprocessor.run()
    
    print("--- Data Preprocessing Test ---")
    print(f"Number of samples: {len(features)}")
    if len(features) > 0:
        print(f"Feature matrix shape for one sample: {features[0].shape}")
        print(f"Label for one sample: {labels[0]}")
        print(f"Return for one sample: {returns[0]}") 