import sys
from pathlib import Path
import logging
import os

# 콘솔 인코딩 강제(윈도우 한글/특수문자 깨짐 방지)
if os.name == 'nt':
    import ctypes
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

# 로깅 설정(모든 모듈에 적용)
if sys.version_info >= (3, 9):
    handler = logging.StreamHandler(sys.stdout, encoding='utf-8')
else:
    handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(name)s: %(message)s',
    handlers=[handler]
)

project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
from src.trading.virtual.virtual_trading_engine import VirtualTradingEngine
from pprint import pprint

if __name__ == "__main__":
    engine = VirtualTradingEngine(symbol="BTCUSDT")
    asyncio.run(engine.run()) 