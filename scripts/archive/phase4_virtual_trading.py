"""
Phase 4: 가상 트레이딩 엔진 실행 스크립트.
현재는 환경 설정만 수행하며, 실제 실행 로직은 주석 처리되어 있습니다.
"""
import sys
import os
import logging
from pathlib import Path

def setup_environment() -> None:
    """스크립트 실행을 위한 환경(콘솔 인코딩, 로깅, 경로)을 설정합니다."""
    
    # 1. Windows 콘솔 인코딩 설정 (UTF-8)
    if os.name == 'nt':
        try:
            import ctypes
            # mypy/pyright가 windows 전용 모듈을 인식하도록 # type: ignore 추가
            ctypes.windll.kernel32.SetConsoleOutputCP(65001) # type: ignore
        except (ImportError, AttributeError, OSError) as e:
            # ctypes가 없거나, 함수 호출에 실패할 경우
            print(f"Warning: Windows 콘솔 인코딩 설정 실패 - {e}")

    # 2. 로깅 설정
    log_format = '[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # 3. 프로젝트 경로 설정
    try:
        # __file__이 정의되지 않은 경우(예: 대화형 환경)를 대비
        project_root = Path(__file__).resolve().parent.parent
    except NameError:
        project_root = Path.cwd()
        
    src_path = project_root / "src"

    # 경로를 sys.path에 중복 추가하지 않도록 확인
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    logging.info(f"프로젝트 루트: {project_root}")
    logging.info(f"sys.path에 추가: {src_path}")

def main() -> None:
    """메인 실행 함수"""
    setup_environment()
    
    # 가상 트레이딩 엔진 로직 (현재 비활성화)
    # ------------------------------------
    # import asyncio
    # from pprint import pprint
    # from src.trading.virtual.virtual_trading_engine import VirtualTradingEngine
    
    logging.info("가상 트레이딩 엔진 실행 준비 완료. (현재 비활성화 상태)")
    # engine = VirtualTradingEngine(symbol="BTCUSDT")
    # asyncio.run(engine.run())
    # ------------------------------------
    pass

if __name__ == "__main__":
    main() 