"""선택적 타입 검사 스크립트"""

import subprocess
import sys
from pathlib import Path

def check_critical_modules():
    """핵심 모듈만 엄격하게 검사"""
    critical_paths = [
        "src/data/processors/",
        "src/models/",
        "src/backtesting/"
    ]
    
    for path in critical_paths:
        if Path(path).exists():
            result = subprocess.run([
                sys.executable, "-m", "pyright", 
                "--level", "error",  # 오류만 검사
                path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ {path}에서 타입 오류 발견:")
                print(result.stdout)
                return False
    
    print("✅ 핵심 모듈 타입 검사 통과")
    return True

if __name__ == "__main__":
    check_critical_modules() 