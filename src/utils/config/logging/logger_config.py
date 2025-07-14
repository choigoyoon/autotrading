# type: ignore
# pylint: disable-all
"""
중앙화된 로깅 시스템
모든 모듈에서 일관된 로깅 제공
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger as loguru_logger

# 통합된 settings.py의 logging_settings를 import
# from ..config.settings import logging_settings # 이전 경로
from src.config.settings import logging_settings # 수정된 절대 경로

class QuantTradingLogger:
    """퀀트매매 시스템 전용 로거"""

    def __init__(self):
        self.log_dir = logging_settings.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.remove() # 기존 핸들러 제거
        
        self.loggers = {} # 현재 사용되지 않으나, 향후 특정 로거 저장용으로 유지
        
        self._setup_main_logger()
        self._setup_trade_logger()
        self._setup_error_logger()
        self._setup_pipeline_logger() # 파이프라인 로거 추가

    def _format_log_path(self, pattern: str) -> Path:
        # Loguru는 파일명에 {time} 포맷을 직접 지원하므로, 여기서는 Path 객체만 반환
        # 다만, settings의 패턴이 단순 파일명이면 그대로 사용
        if "{time" in pattern: # Loguru가 처리할 시간 포맷이 있다면 그대로 사용
            return self.log_dir / pattern
        # 시간 포맷이 없다면, 현재 시간을 기준으로 파일명 생성 (예시, 필요시 수정)
        return self.log_dir / pattern.format(time=datetime.now().strftime("%Y-%M-%d"))


    def _setup_main_logger(self):
        """메인 로거 설정"""
        # main_log_path = self.log_dir / logging_settings.main_log_filename_pattern
        # Loguru가 {time} 포맷을 처리하므로, 파일명 패턴 직접 전달
        main_log_path_pattern = logging_settings.main_log_filename_pattern

        loguru_logger.add(
            sys.stdout,
            level=logging_settings.log_level.upper(), # .upper() 추가
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            # 기본 필터: 에러 및 특정 extra 태그 제외 (예시, 필요에 따라 조정)
            filter=lambda record: record["level"].name != "ERROR" and not record["extra"].get("is_trade_log")
        )
        
        loguru_logger.add(
            self.log_dir / main_log_path_pattern, # Path 객체로 변환
            level=logging_settings.log_level.upper(), # .upper() 추가
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=f"{logging_settings.max_file_size_mb}MB", # MB 단위 사용
            retention=f"{logging_settings.backup_count} days",
            compression="zip",
            encoding='utf-8', # 인코딩 명시
            filter=lambda record: record["level"].name != "ERROR" and not record["extra"].get("is_trade_log")
        )

    def _setup_trade_logger(self):
        """거래 전용 로거 설정"""
        # trade_log_path = self.log_dir / logging_settings.trade_log_filename_pattern
        trade_log_path_pattern = logging_settings.trade_log_filename_pattern
        
        # 거래 로그는 stdout으로도 일부 중요 정보만 출력하거나, 파일로만 기록
        loguru_logger.add(
            self.log_dir / trade_log_path_pattern,
            level="INFO", 
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", # extra 필드는 메시지에 포함
            rotation=f"{logging_settings.max_file_size_mb}MB", 
            retention=f"{logging_settings.backup_count * 2} days",
            compression="zip",
            encoding='utf-8',
            # 필터: extra에 is_trade_log=True가 있는 경우에만
            filter=lambda record: record["extra"].get("is_trade_log") is True
        )

    def _setup_error_logger(self):
        """에러 전용 로거 설정"""
        # error_log_path = self.log_dir / logging_settings.error_log_filename_pattern
        error_log_path_pattern = logging_settings.error_log_filename_pattern
        
        loguru_logger.add(
            self.log_dir / error_log_path_pattern,
            level="ERROR", # 에러 레벨 이상
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            rotation=f"{logging_settings.max_file_size_mb}MB",
            retention=f"{logging_settings.backup_count * 3} days",
            compression="zip",
            encoding='utf-8'
            # 별도 필터 없으면 level="ERROR"가 기본 필터 역할
        )

    def _setup_pipeline_logger(self): # 신규 추가
        """파이프라인 진행 상황 로거 설정"""
        pipeline_log_path_pattern = logging_settings.pipeline_log_filename_pattern

        loguru_logger.add(
            self.log_dir / pipeline_log_path_pattern,
            level="INFO", # 파이프라인 로그는 INFO 레벨부터
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=f"{logging_settings.max_file_size_mb}MB",
            retention=f"{logging_settings.backup_count} days",
            compression="zip",
            encoding='utf-8'
            # filter=lambda record: record["extra"].get("is_pipeline_log") is True # <-- 필터 일시 제거
        )

    def get_logger(self, name: Optional[str] = None) -> loguru_logger:
        """모듈별 또는 일반 로거 반환. name이 없으면 기본 로거."""
        if name:
            return loguru_logger.bind(name=name)
        return loguru_logger # 기본 (전역) 로거

    # 특정 로그 타입 메서드는 bind를 활용하여 컨텍스트 추가 가능
    def get_trade_logger(self) -> loguru_logger:
        return loguru_logger.bind(is_trade_log=True)

    def get_pipeline_logger(self) -> loguru_logger:
        return loguru_logger.bind(is_pipeline_log=True)

    # 기존 log_trade, log_signal, log_performance 메서드는 유지하거나,
    # get_trade_logger().info(...) 형태로 사용하는 것을 권장하여 삭제 또는 수정 가능.
    # 여기서는 일단 유지하되, 내부에서 get_trade_logger() 등을 사용하도록 수정.

    def log_trade(self, action: str, symbol: str, quantity: float, 
                  price: float, side: str, **kwargs):
        trade_logger = self.get_trade_logger()
        message = f"{action} | {symbol} | {side.upper()} | Q:{quantity} | P:{price}"
        if kwargs:
            extra_info = " | ".join([f"{k}:{v}" for k, v in kwargs.items()])
            message += f" | {extra_info}"
        trade_logger.info(message) # extra에 is_trade_log=True 자동 바인딩

    def log_signal(self, signal_type: str, symbol: str, score: int, 
                   timeframe: str, **kwargs):
        # 신호 로그는 일반 로거 사용 (main logger 필터에 걸리지 않도록)
        # 또는 별도 signal_logger 핸들러 추가 고려
        logger_to_use = self.get_logger(name="SIGNAL_LOGIC") 
        message = f"SIGNAL | {signal_type} | {symbol} | TF:{timeframe} | Score:{score}"
        if kwargs:
            extra_info = " | ".join([f"{k}:{v}" for k, v in kwargs.items()])
            message += f" | {extra_info}"
        logger_to_use.info(message)

    def log_performance(self, period: str, metrics: dict):
        # 성과 로그는 파이프라인 로거나 일반 로거 사용
        logger_to_use = self.get_pipeline_logger() # 파이프라인 로그로 분류
        message = f"PERFORMANCE | Period:{period}"
        for key, value in metrics.items():
            message += f" | {key}:{value}"
        logger_to_use.info(message)

# ============== 전역 로거 인스턴스 ==============
# 다른 모듈에서 from src.utils.config.logging.logger_config import quant_logger 로 사용
quant_logger_instance = QuantTradingLogger()

# 전역에서 쉽게 사용할 수 있도록 loguru_logger 자체를 반환하는 함수도 제공 가능
def get_q_logger(name: Optional[str] = None) -> loguru_logger:
    """프로젝트 전역에서 사용할 수 있는 Loguru 로거 인스턴스를 반환합니다."""
    global quant_logger_instance
    if quant_logger_instance is None: # 매우 예외적인 경우 (모듈 로드 순서 등)
        quant_logger_instance = QuantTradingLogger()
    return quant_logger_instance.get_logger(name)

def get_q_trade_logger() -> loguru_logger:
    global quant_logger_instance
    if quant_logger_instance is None: quant_logger_instance = QuantTradingLogger()
    return quant_logger_instance.get_trade_logger()

def get_q_pipeline_logger() -> loguru_logger:
    global quant_logger_instance
    if quant_logger_instance is None: quant_logger_instance = QuantTradingLogger()
    return quant_logger_instance.get_pipeline_logger()

# 이전 quant_logger 변수명 유지 (호환성)
quant_logger = quant_logger_instance.get_logger()


# 사용 예시:
# from src.utils.config.logging.logger_config import get_q_logger, get_q_trade_logger
#
# module_logger = get_q_logger(__name__)
# trade_specific_logger = get_q_trade_logger()
#
# module_logger.info("일반 모듈 로그")
# trade_specific_logger.info("이것은 거래 관련 로그입니다.") 