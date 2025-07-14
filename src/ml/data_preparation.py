import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 로거 설정
logger = logging.getLogger(__name__)


class OnDiskTradingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    디스크에 개별 파일로 저장된 시퀀스 데이터를 로드하는 PyTorch Dataset.
    메모리 사용량을 최소화하기 위해 데이터를 미리 로드하지 않고,
    __getitem__에서 필요할 때마다 파일을 읽어옵니다.
    """
    def __init__(self, data_dir: Union[str, Path]):
        """
        Args:
            data_dir (str or Path): .pt 파일들이 저장된 디렉토리 경로.
        """
        self.data_dir = Path(data_dir)
        self.file_list: List[str] = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith('.pt')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        self.num_samples: int = len(self.file_list)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path: Path = self.data_dir / self.file_list[idx]
        data: Dict[str, torch.Tensor] = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        return data['features'], data['label']


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    On-Disk 데이터셋을 사용하여 학습, 검증, 테스트용 DataLoader를 생성합니다.
    num_workers > 0 으로 데이터 로딩을 병렬화하여 I/O 병목을 해소합니다.
    
    Args:
        batch_size (int): DataLoader의 배치 크기.
        num_workers (int): 데이터 로딩에 사용할 서브프로세스 수.
                           (0은 메인 프로세스에서만 로딩함을 의미)

    Returns:
        dict: 'train', 'val', 'test' 키를 가진 DataLoader 딕셔너리.
    """
    project_root = Path(__file__).parent.parent.parent
    base_data_path = project_root / 'data' / 'sequences_tb'

    train_dir = base_data_path / 'train'
    val_dir = base_data_path / 'val'
    test_dir = base_data_path / 'test'

    logger.info("On-Disk 데이터셋을 생성합니다...")
    train_dataset = OnDiskTradingDataset(data_dir=train_dir)
    val_dataset = OnDiskTradingDataset(data_dir=val_dir)
    test_dataset = OnDiskTradingDataset(data_dir=test_dir)
    
    logger.info(f"  - 훈련 세트: {len(train_dataset):,}")
    logger.info(f"  - 검증 세트: {len(val_dataset):,}")
    logger.info(f"  - 테스트 세트: {len(test_dataset):,}")
    
    dataloaders: Dict[str, Any] = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    scaler_path = base_data_path / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        dataloaders['scaler'] = scaler
        logger.info(f"Scaler 로드 완료: {scaler_path}")

    return dataloaders


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    특징 및 레이블 데이터를 시계열 시퀀스로 변환합니다.
    
    Args:
        features (np.ndarray): (n_samples, n_features) 형태의 특징 배열.
        labels (np.ndarray): (n_samples,) 형태의 레이블 배열.
        sequence_length (int): 각 시퀀스의 길이.

    Returns:
        tuple: (np.ndarray, np.ndarray) 형태의 (X, y) 시퀀스 데이터.
               X: (n_samples - sequence_length, sequence_length, n_features)
               y: (n_samples - sequence_length,)
    """
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y) 