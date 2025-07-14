import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.absolute()))

from src.ml.training_pipeline_v2 import TrainingPipeline
from src.ml.adaptive_learning_system import AdaptiveLearningSystem
from src.ml.hierarchical_trading_transformer import HierarchicalTradingTransformer, ModelConfig
from src.ml.risk_adjusted_loss import RiskAdjustedLoss, LossConfig

def setup_logging(log_file: str = None, log_level: str = "INFO"):
    """로깅 설정"""
    log_level = getattr(logging, log_level.upper())
    
    # 로거 생성
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 포매터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (파일이 지정된 경우)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 설정 파일 로드
    config_path = "configs/training_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    config = load_config(config_path)
    
    # 로깅 설정
    log_file = config.get('logging', {}).get('file')
    log_level = config.get('logging', {}).get('level', 'INFO')
    logger = setup_logging(log_file, log_level)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 모델 구성
    model_config = ModelConfig(
        input_dim=config['model']['input_dim'],
        model_dim=config['model']['model_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_timeframes=len(config['data']['timeframes']),
        dropout=config['model']['dropout'],
        use_pe=config['model']['use_pe']
    )
    
    # 손실 함수 구성
    loss_weights = config['loss_weights']
    loss_config = LossConfig(
        direction_weight=loss_weights['direction'],
        magnitude_weight=loss_weights['magnitude'],
        duration_weight=loss_weights['duration'],
        confidence_weight=loss_weights['confidence'],
        risk_weight=loss_weights['risk']
    )
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 파이프라인 초기화
    pipeline = TrainingPipeline(
        data_dir=config['data']['data_dir'],
        features=config['data']['features'],
        timeframes=config['data']['timeframes'],
        model_config=model_config,
        loss_config=loss_config,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        batch_size=config['data']['batch_size'],
        seq_len=config['data']['seq_len'],
        num_workers=config['data']['num_workers'],
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=config['training']['log_dir']
    )
    
    # 데이터 로드
    logger.info("Loading data...")
    pipeline.load_data()
    
    # 학습 실행
    logger.info("Starting training...")
    pipeline.train(
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        clip_grad_norm=config['training']['clip_grad_norm'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        use_amp=config['training']['use_amp']
    )
    
    # 테스트 실행
    logger.info("Running evaluation on test set...")
    test_metrics = pipeline.evaluate()
    logger.info(f"Test metrics: {test_metrics}")
    
    # 모델 저장
    model_save_path = checkpoint_dir / "final_model.pt"
    pipeline.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # 온라인 적응 시스템 초기화 (선택 사항)
    if config.get('online_adaptation', {}).get('enable_adaptation', False):
        logger.info("Initializing online adaptation system...")
        
        # 모델 로드 (필요한 경우)
        model = HierarchicalTradingTransformer(model_config).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        
        # 온라인 적응 시스템 초기화
        online_system = AdaptiveLearningSystem(
            model=model,
            device=device,
            checkpoint_dir=checkpoint_dir / "online_models",
            **config['online_adaptation']
        )
        
        # 온라인 적응 시스템 시작
        online_system.start()
        logger.info("Online adaptation system started")
        
        # 여기서는 예시로 1분간 실행 (실제로는 계속 실행되어야 함)
        try:
            import time
            time.sleep(60)
            
            # 상태 확인
            status = online_system.get_status()
            logger.info(f"Online system status: {status}")
            
        except KeyboardInterrupt:
            logger.info("Stopping online adaptation system...")
        finally:
            online_system.stop()
            logger.info("Online adaptation system stopped")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
