import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, TypedDict

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame, Index, Series
from sklearn.preprocessing import MinMaxScaler  # type: ignore[reportMissingTypeStubs]
from tqdm import tqdm

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class SequenceConfig(TypedDict):
    sequence_length: int
    test_size: float
    val_size: float
    labels_dir: Path
    ohlcv_path: Path
    output_dir: Path
    feature_cols: List[str]

class SequenceGenerator:
    """
    라벨링된 시계열 데이터로부터 학습용 시퀀스를 생성, 정규화, 분할 및 저장합니다.
    """
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.scaler = MinMaxScaler()

    def run(self) -> bool:
        """전체 시퀀스 생성 파이프라인을 실행합니다."""
        print("🚀 MACD 라벨 기반 시퀀스 생성을 시작합니다...")
        
        # 1. 데이터 로드
        df_features, df_labels, label_column = self._load_and_prepare_data()
        if df_features is None or df_labels is None or label_column is None:
            return False

        # 2. 데이터 정규화 및 배열 변환
        feature_array, label_array = self._scale_and_convert_to_arrays(df_features, df_labels, label_column)

        # 3. 데이터셋 분할 및 저장
        self._split_and_save_sequences(feature_array, label_array, df_features.index)
        
        # 4. 메타데이터 및 스케일러 저장
        self._save_metadata_and_scaler(df_features, df_labels, label_column)
        
        print(f"\n✅ 시퀀스 생성 완료! 저장 위치: {self.config['output_dir']}")
        return True

    def _load_and_prepare_data(self) -> Tuple[Optional[DataFrame], Optional[DataFrame], Optional[str]]:
        """라벨과 OHLCV 데이터를 로드하고 전처리합니다."""
        try:
            # 라벨 파일 선택
            label_file = self._find_label_file()
            if not label_file: return None, None, None
            print(f"🎯 사용할 라벨 파일: {label_file.name}")

            df_labels = pd.read_parquet(label_file)
            df_ohlcv = pd.read_parquet(self.config['ohlcv_path'])

            # 인덱스 및 컬럼 정리
            df_labels, label_column = self._clean_labels(df_labels)
            if label_column is None: return None, None, None
            
            df_ohlcv = self._clean_ohlcv(df_ohlcv)

            # 피처 생성
            df_features = self._create_features(df_ohlcv)

            # 공통 인덱스로 데이터 정렬
            common_idx = df_features.index.intersection(df_labels.index)
            if len(common_idx) < self.config['sequence_length'] * 2:
                print("❌ 공통 인덱스의 데이터가 너무 적습니다.")
                return None, None, None
            
            return df_features.loc[common_idx], df_labels.loc[common_idx], label_column

        except FileNotFoundError as e:
            print(f"❌ 파일 없음: {e}")
            return None, None, None
        except Exception as e:
            print(f"❌ 데이터 준비 실패: {e}")
            return None, None, None

    def _find_label_file(self) -> Optional[Path]:
        """사용할 최적의 라벨 파일을 찾습니다."""
        if not self.config['labels_dir'].exists():
            raise FileNotFoundError(f"라벨 디렉토리 없음: {self.config['labels_dir']}")
        
        label_files = list(self.config['labels_dir'].glob('*.parquet'))
        if not label_files:
            print(f"❌ 라벨 파일이 없습니다: {self.config['labels_dir']}")
            return None
            
        # 우선순위: merged > 1min > 가장 큰 파일
        merged_file = next((f for f in label_files if 'merged' in f.name.lower()), None)
        if merged_file: return merged_file

        one_min_file = next((f for f in label_files if '1min' in f.name.lower()), None)
        if one_min_file: return one_min_file

        return max(label_files, key=lambda f: f.stat().st_size)

    def _clean_labels(self, df: DataFrame) -> Tuple[DataFrame, Optional[str]]:
        """라벨 데이터프레임의 인덱스와 컬럼을 정리합니다."""
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) # Timezone 제거

        label_col = next((c for c in ['label', 'labels', 'target'] if c in df.columns), None)
        if not label_col:
            print("❌ 'label', 'labels' 또는 'target' 컬럼을 찾을 수 없습니다.")
            return df, None
        return df, label_col

    def _clean_ohlcv(self, df: DataFrame) -> DataFrame:
        """OHLCV 데이터프레임의 인덱스를 정리합니다."""
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) # Timezone 제거
        return df
        
    def _create_features(self, df_ohlcv: DataFrame) -> DataFrame:
        """피처를 생성하고 선택합니다."""
        df = df_ohlcv.copy()
        df['return'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['return'].rolling(20).std().fillna(0)
        df['volume_ma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
        
        available_features = [f for f in self.config['feature_cols'] if f in df.columns]
        return df.loc[:, available_features]

    def _scale_and_convert_to_arrays(
        self, 
        df_features: DataFrame, 
        df_labels: DataFrame, 
        label_column: str
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int_]]:
        """데이터를 정규화하고 NumPy 배열로 변환합니다."""
        feature_array: npt.NDArray[np.float32] = self.scaler.fit_transform(df_features).astype(np.float32)
        label_array: npt.NDArray[np.int_] = df_labels[label_column].to_numpy().astype(np.int_)
        return feature_array, label_array

    def _split_and_save_sequences(
        self, 
        feature_array: npt.NDArray[np.float32], 
        label_array: npt.NDArray[np.int_], 
        index: Index
    ) -> None:
        """시퀀스를 생성하고 train/val/test로 분할하여 저장합니다."""
        output_base_dir = self.config['output_dir']
        if output_base_dir.exists():
            shutil.rmtree(output_base_dir)
        
        num_sequences = len(feature_array) - self.config['sequence_length'] + 1
        if num_sequences <= 0:
            print("❌ 시퀀스를 생성하기에 데이터가 부족합니다.")
            return

        train_end = int(num_sequences * (1 - self.config['test_size'] - self.config['val_size']))
        val_end = int(num_sequences * (1 - self.config['test_size']))
        
        datasets = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, num_sequences)}

        for name, (start, end) in datasets.items():
            dataset_dir = output_base_dir / name
            dataset_dir.mkdir(parents=True)
            for i in tqdm(range(start, end), desc=f"'{name}' 데이터 저장 중"):
                seq_end_idx = i + self.config['sequence_length'] - 1
                sequence = feature_array[i : seq_end_idx + 1]
                label = label_array[seq_end_idx]
                timestamp = cast(pd.Timestamp, index[seq_end_idx])

                torch.save({
                    'features': torch.tensor(sequence, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.long),
                    'timestamp': timestamp
                }, dataset_dir / f"{i - start}.pt")

    def _save_metadata_and_scaler(self, df_features: DataFrame, df_labels: DataFrame, label_column: str) -> None:
        """메타데이터와 스케일러를 저장합니다."""
        metadata = {
            'sequence_length': self.config['sequence_length'],
            'num_features': len(df_features.columns),
            'feature_names': list(df_features.columns),
            'label_column': label_column,
            'data_range': {'start': str(df_features.index[0]), 'end': str(df_features.index[-1])},
            'label_distribution': {str(k): v for k, v in df_labels[label_column].value_counts().to_dict().items()}
        }
        with open(self.config['output_dir'] / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        joblib.dump(self.scaler, self.config['output_dir'] / "scaler.joblib")


def main() -> bool:
    """메인 실행 함수"""
    project_root = Path(__file__).resolve().parent.parent
    
    config: SequenceConfig = {
        "sequence_length": 240,
        "test_size": 0.1,
        "val_size": 0.2,
        "labels_dir": project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'labeled',
        "ohlcv_path": project_root / 'data' / 'processed' / 'btc_usdt_kst' / 'resampled_ohlcv' / '1min.parquet',
        "output_dir": project_root / 'data' / 'sequences_macd',
        "feature_cols": ['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'volume_ma']
    }
    
    generator = SequenceGenerator(config)
    return generator.run()

if __name__ == '__main__':
    if main():
        print("\n🎉 시퀀스 생성 성공!")
    else:
        print("\n❌ 시퀀스 생성 실패!")