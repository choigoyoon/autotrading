"""선택적 타입 검사 스크립트"""

import subprocess
import sys
from pathlib import Path

def check_critical_modules() -> bool:
    """
    핵심 모듈만 엄격하게 검사하여 타입 오류 여부를 반환합니다.
    
    Returns:
        bool: 타입 오류가 없으면 True, 있으면 False를 반환합니다.
    """
    print("🚀 핵심 모듈에 대한 엄격한 타입 검사를 시작합니다...")
    critical_paths: list[str] = [
        "src/data/",
        "src/features/",
        "src/labeling/",
        "src/ml/",
    ]
    
    all_passed = True
    for path_str in critical_paths:
        path = Path(path_str)
        if path.exists():
            print(f"\n🔍 검사 중: {path_str}")
            # sys.executable이 None일 수 있음을 처리
            python_executable = sys.executable
            if not python_executable:
                print("❌ Python 실행 파일을 찾을 수 없습니다.")
                return False

            result = subprocess.run(
                [python_executable, "-m", "pyright", "--level", "error", path_str],
                capture_output=True,
                text=True,
                encoding='utf-8' # 인코딩 명시
            )
            
            if result.returncode != 0:
                print(f"❌ '{path_str}'에서 타입 오류 발견:")
                # stdout과 stderr을 모두 출력하여 자세한 오류 확인
                if result.stdout:
                    print("--- stdout ---")
                    print(result.stdout)
                if result.stderr:
                    print("--- stderr ---")
                    print(result.stderr)
                all_passed = False
            else:
                print(f"✅ '{path_str}' 타입 검사 통과")
        else:
            print(f"⚠️ 경로를 찾을 수 없음 (건너뛰기): {path_str}")
    
    if all_passed:
        print("\n🎉 모든 핵심 모듈 타입 검사를 통과했습니다.")
    else:
        print("\n🚨 일부 핵심 모듈에서 타입 오류가 발견되었습니다.")

    return all_passed

if __name__ == "__main__":
    if not check_critical_modules():
        sys.exit(1) 