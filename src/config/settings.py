# type: ignore
# pylint: disable-all
"""
전역 설정 파일 (통합 버전)
- API 키, 데이터 경로, 시스템 설정, 모델 설정 등 프로젝트 전반의 설정을 관리합니다.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

# ==============================================================================
# === 기본 경로 설정 ===
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'
TEMP_DIR = PROJECT_ROOT / 'temp'

# ==============================================================================
# === 환경 및 시스템 설정 (utils/settings.py 에서 가져옴) ===
# ==============================================================================
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" and torch.cuda.is_available() else 0
CPU_COUNT = os.cpu_count() or 1

# ==============================================================================
# === API 키 설정 (기존 config/settings.py 유지) ===
# ==============================================================================
# 환경 변수에서 API 키를 불러옵니다. .env 파일 등을 사용하여 관리하세요.
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_API_SECRET')

# 지원하는 타임프레임 목록 (17개)
SUPPORTED_TIMEFRAMES: List[str] = [
    # 분봉 (6개)
    '1m', '3m', '5m', '10m', '15m', '30m',
    # 시간봉 (6개)
    '1h', '2h', '4h', '6h', '8h', '12h',
    # 일봉 (3개)
    '1D', '2D', '3D',
    # 주봉 (1개)
    '1W',
    # 월봉 (1개)
    '1M'
]

# ==============================================================================
# === Dataclass 및 Pydantic Model 기반 상세 설정 ===
# ==============================================================================

@dataclass
class LoggingSettingsConfig:
    """로깅 설정"""
    log_dir: Path = LOG_DIR
    log_level: str = "DEBUG" if DEBUG else "INFO"
    max_file_size_mb: int = 10
    backup_count: int = 5
    main_log_filename_pattern: str = "main_{time}.log"
    trade_log_filename_pattern: str = "trades_{time}.log"
    error_log_filename_pattern: str = "errors_{time}.log"
    pipeline_log_filename_pattern: str = "pipeline_{time}.log"

@dataclass
class SystemConfig:
    """시스템 설정"""
    device: str = DEVICE
    gpu_memory: int = GPU_MEMORY
    cpu_count: int = CPU_COUNT
    max_workers: int = min(32, (CPU_COUNT if CPU_COUNT is not None else 1) + 4)
    batch_size: int = 32 if DEVICE == "cpu" else 64
    timeframes: List[str] = field(default_factory=lambda: SUPPORTED_TIMEFRAMES)
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT'])
    logging: LoggingSettingsConfig = field(default_factory=LoggingSettingsConfig)
    version: str = "3.2.1"

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    timeframes: List[str] = field(default_factory=lambda: SUPPORTED_TIMEFRAMES)
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    data_dir_structure: Dict[str, str] = field(default_factory=lambda: {
        'base': '{symbol_clean}/{category}/{timeframe_safe}_basedata.parquet',
        'resampled': '{symbol_clean}/{category}/{timeframe_safe}.parquet',
        'processed': '{symbol_clean}/{category}/{timeframe_safe}_indicators_labels_returns.parquet'
    })
    lookback_days: Optional[int] = None  # None으로 설정하면 전체 기간 데이터 로드
    update_interval_minutes: int = 60
    base_timeframe: str = "1m"
    start_date: str = "2017-08-01"
    exchange_id: str = "binance"
    ensure_complete_data: bool = True
    primary_exchange_for_base_data: str = "binance"
    force_regenerate_base_data: bool = False

    def __post_init__(self):
        pass

    def get_symbol_start_date(self, symbol):
        """심볼별 실제 상장일 반환"""
        symbol_start_dates = {
            'BTC/USDT': '2017-08-17',  # 바이낸스 BTC 상장일
            'ETH/USDT': '2017-08-17',
            # ... 기타 심볼들
        }
        return symbol_start_dates.get(symbol, self.start_date)

@dataclass
class TradingSettingsConfig:
    """매매 및 리스크 관리 설정"""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_capital: float = 100000.0
    
    indicator_buffer_size: int = 500
    min_data_points_for_signal: int = 200
    
    max_risk_per_trade_pct: float = 0.05
    max_drawdown_limit_pct: float = 0.20
    max_position_size_pct: float = 0.20
    max_daily_loss_pct: float = 0.05
    max_concurrent_positions: int = 5

    commission_rate: float = 0.0004
    slippage_avg_pct: float = 0.0002

    min_signal_score: int = 80
    signal_timeout_seconds: int = 300

@dataclass
class ModelConfig:
    """AI 모델 관련 설정"""
    model_save_dir: Path = MODEL_DIR
    cnn_sequence_length: int = 50
    cnn_input_channels: int = 5
    cnn_num_patterns: int = 10
    lstm_sequence_length: int = 20
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    min_accuracy: float = 0.85
    min_f1_score: float = 0.70

# --- 라벨링 설정을 위한 Pydantic 모델들 --- 
class MACDLabelerConfig(BaseModel):
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    peak_min_distance: int = 5
    peak_min_prominence: float = 0.1
    profit_window: int = 20
    stop_loss_window: int = 10
    profit_threshold_pct: float = 0.02
    stop_loss_threshold_pct: float = 0.01
    use_trend_filter: bool = False
    trend_ema_period: int = 50
    use_dynamic_thresholds: bool = False
    dynamic_atr_period: int = 14
    dynamic_profit_atr_multiplier: float = 1.5
    dynamic_stop_loss_atr_multiplier: float = 1.0
    min_volume_threshold: Optional[float] = None
    max_holding_period: Optional[int] = None
    divergence_check_enabled: bool = False
    divergence_lookback_period: int = 14
    min_peak_valley_prominence: float = 0.05
    label_types_to_generate: List[str] = Field(default_factory=lambda: ["standard", "profit_stop", "tri_barrier"])

class TripleBarrierConfig(BaseModel):
    profit_target_multipliers: List[float] = Field(default_factory=lambda: [0.01, 0.02, 0.03])
    stop_loss_multipliers: List[float] = Field(default_factory=lambda: [0.005, 0.01, 0.015])
    barrier_lookforward_periods: List[int] = Field(default_factory=lambda: [5, 10, 20])
    volatility_lookback: int = 20
    volatility_target_type: str = "atr"  # 'atr', 'stddev'
    min_volatility_threshold: float = 0.0001

class LabelingConfig(BaseModel):
    active_labeler: str = "macd"
    macd_labeler: MACDLabelerConfig = Field(default_factory=MACDLabelerConfig)
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)
    output_dir: str = str(DATA_DIR / "processed" / "labels")
    cache_labels: bool = True
    force_relabel: bool = False
    label_horizon_specific_config: Dict[str, Any] = Field(default_factory=dict)
    default_min_return_for_label: float = 0.001
    indicator_settings: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class ExchangeAPIConfig:
    """거래소 API 설정"""
    binance_api_key: str = BINANCE_API_KEY
    binance_api_secret: str = BINANCE_API_SECRET
    rate_limit_per_minute: int = 1200
    timeout_seconds: int = 30
    retry_count: int = 3
    use_testnet: bool = ENVIRONMENT != 'production'

# ==============================================================================
# === 통합 설정 인스턴스 생성 (주로 테스트나 단독 실행 시 사용) ===
# === 파이프라인에서는 ConfigLoader를 통해 YAML에서 로드하는 것을 권장 ===
# ==============================================================================
system_settings = SystemConfig()
data_settings = DataConfig()
trading_settings = TradingSettingsConfig()
model_settings = ModelConfig()
labeling_settings = LabelingConfig()
exchange_api_settings = ExchangeAPIConfig()
logging_settings_instance = LoggingSettingsConfig()

# ==============================================================================
# === 추가 설정 클래스 (config_loader.py 에서 필요) ===
# ==============================================================================

class StrategyConfig(BaseModel):
    """전략 관련 설정"""
    name: str = "DefaultStrategy"
    parameters: Dict[str, Any] = Field(default_factory=dict)

class BacktestingConfig(BaseModel):
    """백테스팅 관련 설정"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

class PipelineConfig(BaseModel):
    """파이프라인 실행 설정"""
    # 데이터 수집 설정
    enable_data_collection: bool = True
    enable_resampling: bool = True
    
    # 라벨링 설정
    enable_labeling: bool = True
    labeling_timeframes: List[str] = Field(
        default_factory=lambda: [
            '1min', '3min', '5min', '10min', '15min', '30min',  # 분봉
            '1h', '2h', '4h', '6h', '8h', '12h',               # 시간봉
            '1D', '2D', '3D', '1W'                             # 일봉/주봉
        ]  # 라벨링을 수행할 타임프레임 (모든 지원 타임프레임)
    )
    
    # 모델 학습 설정
    enable_model_training: bool = True
    training_timeframes: List[str] = Field(default_factory=lambda: ['1h', '4h', '1D'])  # 모델 학습에 사용할 타임프레임
    
    # 백테스팅 설정
    enable_backtesting: bool = True
    backtesting_timeframes: List[str] = Field(default_factory=lambda: ['1h', '4h', '1D'])  # 백테스팅에 사용할 타임프레임
    
    # 병렬 처리 설정
    max_workers: int = 4  # 병렬 처리에 사용할 최대 워커 수
    
    # 로깅 설정
    log_level: str = "INFO"
    log_to_file: bool = True


class DefaultPaths(BaseModel):
    """기본 경로 설정 (DataClass가 아닌 Pydantic 모델로 정의)"""
    project_root: str = str(PROJECT_ROOT)
    data_dir: str = str(DATA_DIR)
    model_dir: str = str(MODEL_DIR)
    log_dir: str = str(LOG_DIR)
    temp_dir: str = str(TEMP_DIR)

def validate_all_settings() -> bool:
    """모든 설정을 검증합니다."""
    errors = []
    
    _data_settings_instance = DataConfig(timeframes=SUPPORTED_TIMEFRAMES)

    required_dirs = [DATA_DIR, MODEL_DIR, LOG_DIR, TEMP_DIR, _data_settings_instance.raw_data_dir, _data_settings_instance.processed_data_dir]
    for dir_path in required_dirs:
        try:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"디렉토리 생성 실패 {dir_path}: {e}")

    _system_settings_instance = SystemConfig()
    if _system_settings_instance.device == "cuda" and _system_settings_instance.gpu_memory < 4 * 1024**3:
        errors.append(f"GPU 메모리 부족 {_system_settings_instance.gpu_memory / 1024**3:.1f}GB (권장: 4GB 이상)")
    
    _exchange_api_settings_instance = ExchangeAPIConfig()
    if ENVIRONMENT == 'production':
        if not _exchange_api_settings_instance.binance_api_key or _exchange_api_settings_instance.binance_api_key == 'YOUR_API_KEY':
            errors.append("BINANCE_API_KEY 환경변수 설정 필요 (프로덕션)")
        if not _exchange_api_settings_instance.binance_api_secret or _exchange_api_settings_instance.binance_api_secret == 'YOUR_API_SECRET':
            errors.append("BINANCE_API_SECRET 환경변수 설정 필요 (프로덕션)")
            
    _trading_settings_instance = TradingSettingsConfig()
    if not (0.001 <= _trading_settings_instance.max_risk_per_trade_pct <= 0.2):
        errors.append(f"max_risk_per_trade_pct 범위 오류: {_trading_settings_instance.max_risk_per_trade_pct*100:.1f}% (권장: 0.1% ~ 20%)")

    if errors:
        print("!!! 설정 검증 실패 !!!")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(">>> 모든 설정 검증 완료 <<<")
    return True

def print_active_config_summary():
    """주요 활성 설정을 요약하여 출력합니다."""
    _system_settings = SystemConfig()
    _data_settings = DataConfig(timeframes=_system_settings.timeframes)
    _trading_settings = TradingSettingsConfig()
    _model_settings = ModelConfig()
    _logging_settings = _system_settings.logging
    _exchange_api_settings = ExchangeAPIConfig()

    summary = f"""
    ======================================================
    =====          활성 시스템 설정 요약           =====
    ======================================================
    환경:                     {ENVIRONMENT} (디버그: {DEBUG})
    실행 디바이스:            {_system_settings.device}
    GPU 메모리:             {_system_settings.gpu_memory / 1024**3:.1f} GB (사용 가능 시)
    CPU 코어:                 {_system_settings.cpu_count} 개
    기본 데이터 경로:         {DATA_DIR}
      - Raw 데이터:         {_data_settings.raw_data_dir}
      - Processed 데이터:   {_data_settings.processed_data_dir}
    모델 저장 경로:           {_model_settings.model_save_dir}
    로그 저장 경로:           {_logging_settings.log_dir} (레벨: {_logging_settings.log_level})
    
    기본 트레이딩 설정:
      - 심볼/타임프레임:      {_trading_settings.symbol} / {_trading_settings.timeframe}
      - 초기 자본금:          {_trading_settings.initial_capital:,.0f}
      - 거래당 최대 리스크:   {_trading_settings.max_risk_per_trade_pct*100:.2f}%
      - 수수료율:             {_trading_settings.commission_rate*100:.4f}%
    
    API 설정 (Binance):
      - API Key 설정됨:     {'예' if BINANCE_API_KEY and BINANCE_API_KEY != 'YOUR_API_KEY' else '아니요/기본값'}
      - Testnet 사용:       {_exchange_api_settings.use_testnet}
    ======================================================
    """
    print(summary)

# 스크립트 로드 시 기본 검증 실행 여부 (선택적)
# if __name__ == "__main__":
#     print_active_config_summary()
#     if not validate_all_settings():
#         print("!!! 시스템 설정에 문제가 있어 실행이 중단될 수 있습니다 !!!")

# 기존 TRADING_SETTINGS dict (참고용, TradingSettingsConfig로 이전됨)
# TRADING_SETTINGS = {
#     "symbol": "BTC/USDT",
#     "timeframe": "1h",
#     "initial_capital": 100000.0,
#     "indicator_buffer_size": 500,
#     "min_data_points_for_signal": 200,
#     "max_risk_per_trade": 0.05,
#     "max_drawdown_limit": 0.20,
# }
# -> trading_settings 인스턴스로 대체하여 사용 권장.

# 기존 LoggingConfig 클래스 (참고용, LoggingSettingsConfig로 이전됨)
# class LoggingConfig:
#     log_dir = LOG_DIR
#     main_log_file = "main.log"
#     trade_log_file = "trade.log"
#     error_log_file = "error.log"
#     log_level = "INFO"
#     max_file_size = "10 MB"
#     backup_count = 7
# -> logging_settings 인스턴스로 대체하여 사용 권장.

# 끝부분 중복 정의 제거됨 (PROJECT_ROOT, DATA_DIR, LOG_DIR, API 키들)

