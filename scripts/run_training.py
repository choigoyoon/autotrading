import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.ml.adaptive_learning_system import AdaptiveLearningSystem
from src.ml.hierarchical_trading_transformer import (
    HierarchicalTradingTransformer, ModelConfig
)
from src.ml.training_pipeline_v2 import TrainingPipeline

def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def load_config(config_path: Path) -> Dict[str, Any]:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config

def main() -> None:
    config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    config = load_config(config_path)
    
    logging_config = config.get('logging', {})
    logger = setup_logging(logging_config.get('file'), logging_config.get('level', 'INFO'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_config = config['data']
    model_params = config['model']
    training_params = config['training']

    # 모델 구성 객체 생성 (ModelConfig가 feature_dims를 요구하므로 맞춰줌)
    # feature_dims는 각 타임프레임별 피처의 개수를 담은 딕셔너리여야 합니다.
    # 이 정보는 데이터셋을 분석해야 알 수 있으므로, 임시로 설정하거나 설정 파일에 추가해야 합니다.
    # 여기서는 config 파일에 `feature_dims`가 있다고 가정합니다.
    model_config = ModelConfig(
        feature_dims=model_params['feature_dims'],
        d_model=model_params['d_model'],
        n_heads=model_params['n_heads'],
        n_layers=model_params['n_layers'],
        d_ff=model_params['d_ff'],
        dropout=model_params.get('dropout', 0.1),
        timeframes=data_config['timeframes'],
        max_seq_len=data_config['seq_len']
    )
    
    # TrainingPipeline이 요구하는 인자에 맞춰서 전달
    pipeline = TrainingPipeline(
        data_dir=data_config['data_dir'],
        model_dir=training_params['checkpoint_dir'],
        results_dir=training_params.get('results_dir', 'results'),
        device=str(device),
        timeframes=data_config['timeframes'],
        seq_length=data_config['seq_len'],
        target_length=data_config.get('target_length', 10),
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 0),
        random_seed=training_params.get('random_seed', 42)
    )

    logger.info("Preparing data...")
    pipeline.prepare_data()

    logger.info("Starting training...")
    # train 메소드에 모델과 손실함수 등을 전달해야 할 수 있습니다.
    # TrainingPipelineV2의 train 메소드 시그니처를 확인해야 합니다.
    # 현재는 모델과 설정을 TrainingPipeline 내부에서 생성한다고 가정합니다.
    pipeline.train(epochs=training_params['num_epochs'])
    
    logger.info("Running evaluation on test set...")
    test_metrics = pipeline.test()
    logger.info(f"Test metrics: {test_metrics}")

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
