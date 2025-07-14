import logging
import os
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
import pandas as pd

# utils.progress_logger는 src 폴더를 기준으로 찾을 수 있도록 경로 조정이 필요할 수 있음
# 이 파일(logging_config.py)이 src/config/ 에 있으므로,
# utils.progress_logger는 src/utils/progress_logger.py를 의미함.
# sys.path에 src가 이미 추가되어 있다고 가정하고 진행.

# 타입 체커(Linter)는 이 블록을 보고 ProgressLogger 타입을 인지함
if TYPE_CHECKING:
    from utils.progress_logger import ProgressLogger

# 실제 런타임에 사용될 변수들 초기화
RuntimeProgressLoggerClass: Optional[Any] = None # 실제 임포트될 클래스 또는 None
EMOJI_CLOCK = "🕘"  # 기본값
PROGRESS_LOGGER_MODULE_AVAILABLE = False

try:
    # from utils.progress_logger import ProgressLogger as ImportedPLogger 로 하고, 타입힌트를 ImportedPLogger로 해도 됨
    from utils.progress_logger import ProgressLogger as ActualImportedProgressLogger, EMOJI_CLOCK as ImportedEmojiClock
    RuntimeProgressLoggerClass = ActualImportedProgressLogger
    EMOJI_CLOCK = ImportedEmojiClock
    PROGRESS_LOGGER_MODULE_AVAILABLE = True
    # print("Successfully imported ProgressLogger from utils.progress_logger.") # 디버깅용
except ImportError:
    # print("Warning: Could not import ProgressLogger from utils.progress_logger. Advanced console logging will be disabled.") # 디버깅용
    pass # RuntimeProgressLoggerClass는 None으로 유지, PROGRESS_LOGGER_MODULE_AVAILABLE는 False로 유지

# ==================================================
# 기본 로깅 스타일 설정 (환경 변수 또는 기본값 사용)
# ==================================================
# LOGGING_STYLE 환경 변수가 있으면 그 값을 사용, 없으면 "detailed"
LOGGING_STYLE = os.getenv("LOGGING_STYLE", "detailed").lower()
VALID_LOGGING_STYLES = ["simple", "detailed", "debug"]
if LOGGING_STYLE not in VALID_LOGGING_STYLES:
    print(f"Warning: Invalid LOGGING_STYLE '{LOGGING_STYLE}'. Defaulting to 'detailed'. Valid options: {VALID_LOGGING_STYLES}")
    LOGGING_STYLE = "detailed"

# ==================================================
# 전역 ProgressLogger 인스턴스 (싱글톤처럼 사용 가능)
# ==================================================
# 타입 힌트는 TYPE_CHECKING 블록의 것을 사용
_progress_logger_instance: Optional['ProgressLogger'] = None

def get_progress_logger(logger_name: str = "App", force_new: bool = False) -> Optional['ProgressLogger']:
    """전역 ProgressLogger 인스턴스를 반환하거나 새로 생성합니다. 임포트 실패 시 None 반환."""
    global _progress_logger_instance

    if not PROGRESS_LOGGER_MODULE_AVAILABLE or RuntimeProgressLoggerClass is None:
        return None

    # _progress_logger_instance가 RuntimeProgressLoggerClass의 유효한 인스턴스인지 확인
    is_valid_instance = isinstance(_progress_logger_instance, RuntimeProgressLoggerClass)

    create_new = force_new or not is_valid_instance
    if not create_new and is_valid_instance:
        # hasattr로 logger 속성 존재 여부 먼저 확인 후 접근
        # 타입 체커는 _progress_logger_instance가 Optional[ProgressLogger]임을 알지만,
        # 실제로는 RuntimeProgressLoggerClass의 인스턴스여야 logger.name 접근이 안전함.
        # is_valid_instance 체크로 이미 확인되었으므로, Linter가 경고 시 type: ignore 사용 가능.
        if hasattr(_progress_logger_instance, 'logger') and _progress_logger_instance.logger.name != logger_name: # type: ignore[attr-defined]
            create_new = True
        elif not hasattr(_progress_logger_instance, 'logger'): 
            create_new = True 
            
    if create_new:
        _progress_logger_instance = RuntimeProgressLoggerClass(logger_name=logger_name, logging_style=LOGGING_STYLE)
    
    return _progress_logger_instance

# ==================================================
# 프로젝트 루트 로깅 설정 함수 (메인 스크립트에서 호출)
# ==================================================
PROJECT_ROOT_FOR_LOG = Path(__file__).resolve().parent.parent.parent # E:/trading 가정
LOG_DIR_FOR_SETUP = PROJECT_ROOT_FOR_LOG / "logs"

def setup_project_logging(script_name: str = "main_app", log_level_override: Optional[int] = None):
    """프로젝트 전체의 기본 파일 로깅을 설정합니다.
       ProgressLogger는 주로 콘솔(StreamHandler)에 집중하고,
       파일 로깅은 이 함수를 통해 메인 스크립트에서 한 번 설정합니다.
    """
    LOG_DIR_FOR_SETUP.mkdir(exist_ok=True)
    current_time_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOG_DIR_FOR_SETUP / f"{script_name}_{current_time_str}.log"

    # 설정된 LOGGING_STYLE에 따라 파일 로그 레벨 결정
    if LOGGING_STYLE == "debug":
        effective_log_level = logging.DEBUG
    elif LOGGING_STYLE == "detailed":
        effective_log_level = logging.INFO
    else: # simple
        effective_log_level = logging.WARNING # 간단 모드는 파일에도 경고 이상만 기록 (예시)

    if log_level_override is not None:
        effective_log_level = log_level_override

    # 루트 로거 가져오기
    root_logger = logging.getLogger()
    # 기존 핸들러들 모두 제거 (basicConfig의 force=True 대안)
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # 핸들러를 닫아 파일 잠금 해제 등 리소스 정리
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    emoji_to_use = EMOJI_CLOCK 
    file_formatter = logging.Formatter(
        f'{emoji_to_use} %(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 루트 로거에 핸들러 추가 및 레벨 설정
    root_logger.addHandler(file_handler)
    root_logger.setLevel(effective_log_level)
    
    # ProgressLogger가 사용하는 StreamHandler는 ProgressLogger 내부에서 관리되도록 함
    # logging.basicConfig(force=True) 대신 루트 로거를 직접 설정하는 방식으로 변경하여 중복 basicConfig 호출 방지

    logging.info(f"Project-level file logging setup for '{script_name}'. Log file: {log_file_path}")
    logging.info(f"Current LOGGING_STYLE: {LOGGING_STYLE}")
    if not PROGRESS_LOGGER_MODULE_AVAILABLE:
        logging.warning("ProgressLogger (utils.progress_logger) could not be imported. Advanced console logging is disabled.") 