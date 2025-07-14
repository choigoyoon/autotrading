#!/usr/bin/env python3
"""
개발용 경량 워크스페이스 생성 스크립트
350만개 데이터 프로젝트 최적화용
"""

import shutil
import subprocess
import sys
from pathlib import Path

def create_dev_workspace() -> None:
    """개발용 경량 워크스페이스 생성"""
    
    # 현재 디렉토리
    current_dir = Path.cwd()
    dev_dir = current_dir / "dev_workspace"
    
    print(f"🚀 개발용 워크스페이스 생성 중: {dev_dir}")
    
    # 개발용 폴더 구조
    dev_structure: dict[str, str] = {
        'src': 'copy',           # 코드는 복사
        'configs': 'copy',       # 설정은 복사  
        'tools': 'copy',         # 도구는 복사
        'templates': 'copy',     # 템플릿은 복사
        'results': 'symlink',    # 결과는 심볼릭 링크
        'models': 'symlink',     # 모델은 심볼릭 링크
        'logs': 'symlink',       # 로그는 심볼릭 링크
        'data': 'symlink'        # 데이터는 심볼릭 링크
    }
    
    # 개발 디렉토리 생성
    dev_dir.mkdir(exist_ok=True)
    
    # 파일 복사
    files_to_copy: list[str] = [
        'pyproject.toml',
        'README.md',
        'TA_Lib-0.4.28-cp310-cp310-win_amd64.whl',
        '.cursor-project',
        '.gitignore'
    ]
    
    for file_name in files_to_copy:
        src_file = current_dir / file_name
        dst_file = dev_dir / file_name
        
        if src_file.exists():
            if src_file.is_dir():
                _ = shutil.copytree(src_file, dst_file, dirs_exist_ok=True)
            else:
                _ = shutil.copy2(src_file, dst_file)
            print(f"📁 복사: {file_name}")
    
    # 폴더 처리
    for folder, action in dev_structure.items():
        src_folder = current_dir / folder
        dst_folder = dev_dir / folder
        
        if not src_folder.exists():
            continue
            
        if action == 'copy':
            # 폴더 복사
            if dst_folder.exists():
                shutil.rmtree(dst_folder)
            _ = shutil.copytree(src_folder, dst_folder, dirs_exist_ok=False)
            print(f"📁 복사: {folder}")
            
        elif action == 'symlink':
            # 심볼릭 링크 생성
            if dst_folder.exists():
                if dst_folder.is_symlink():
                    dst_folder.unlink()
                else:
                    shutil.rmtree(dst_folder)
            
            try:
                # Windows와 Unix/macOS에 맞춰 심볼릭 링크 생성
                if sys.platform == "win32":
                    _ = subprocess.run(
                        ["mklink", "/D", str(dst_folder), str(src_folder)],
                        shell=True,
                        check=True,
                    )
                else:
                    # 'linux', 'darwin'
                    dst_folder.symlink_to(src_folder, target_is_directory=True) # type: ignore[unreachable]

                print(f"🔗 심볼릭 링크: {folder}")
            except Exception as e:
                print(f"⚠️ 심볼릭 링크 실패 ({folder}): {e}")
                # 실패시 폴더 생성
                dst_folder.mkdir(exist_ok=True)
    
    # 개발용 설정 파일 생성
    create_dev_settings(dev_dir)
    
    print(f"\n✅ 개발용 워크스페이스 생성 완료!")
    print(f"📂 위치: {dev_dir}")
    print(f"💡 사용법: Cursor에서 {dev_dir} 폴더를 열어주세요")

def create_dev_settings(dev_dir: Path) -> None:
    """
    개발용 IDE(Cursor, VSCode) 설정 파일을 생성합니다.
    
    Args:
        dev_dir (Path): 설정을 생성할 개발 워크스페이스 경로.
    """
    
    # 개발용 .cursorrules
    dev_cursorrules = dev_dir / ".cursor-project" # .cursorrules에서 .cursor-project로 변경
    with open(dev_cursorrules, 'w', encoding='utf-8') as f:
        _ = f.write("""# 개발용 Cursor 설정
# 경량 워크스페이스용

ai_features: full
file_limit: 5000

exclude_patterns:
  - "**/node_modules"
  - "**/.git"
  - "**/data_backup"
  - "**/*.pt"
  - "**/*.pth"
  - "**/*.csv"
  - "**/*.parquet"
""")
    
    # 개발용 .vscode/settings.json
    dev_vscode_dir = dev_dir / ".vscode"
    dev_vscode_dir.mkdir(exist_ok=True)
    
    dev_settings = dev_vscode_dir / "settings.json"
    with open(dev_settings, 'w', encoding='utf-8') as f:
        _ = f.write("""{
    // 개발용 최적화 설정
    "files.watcherExclude": {
        "**/*.pt": true,
        "**/*.pth": true,
        "**/*.csv": true,
        "**/*.parquet": true
    },
    
    "search.exclude": {
        "**/*.pt": true,
        "**/*.pth": true,
        "**/*.csv": true,
        "**/*.parquet": true
    },
    
    // 개발 기능 활성화
    "python.analysis.autoSearchPaths": true,
    "python.analysis.autoImportCompletions": true,
    
    // AI 기능 활성화
    "cursor.ai.enabled": true,
    "cursor.ai.autoComplete": true,
    
    // 메모리 최적화
    "files.maxMemoryForLargeFilesMB": 2048
}
""")
    
    print("⚙️ 개발용 설정 파일 생성 완료")

def create_data_cleanup_script() -> None:
    """데이터 정리 스크립트 생성"""
    
    cleanup_script = Path.cwd() / "dev_workspace" / "cleanup_heavy_data.py"
    with open(cleanup_script, 'w', encoding='utf-8') as f:
        _ = f.write("""#!/usr/bin/env python3
\"\"\"
무거운 데이터 파일 정리 스크립트
Cursor 성능 최적화용
\"\"\"

import os
import shutil
from pathlib import Path

def cleanup_heavy_files():
    \"\"\"무거운 파일들을 별도 위치로 이동\"\"\"
    
    # 백업 디렉토리
    backup_dir = Path("data_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # 이동할 파일/폴더
    heavy_items = [
        "results/profit_distribution_summary.csv",
        "results/profit_distribution/",
        "results/label_analysis/",
        "models/",
        "logs/"
    ]
    
    for item in heavy_items:
        src_path = Path(item)
        if src_path.exists():
            dst_path = backup_dir / src_path.name
            
            if src_path.is_file():
                shutil.move(str(src_path), str(dst_path))
                print(f"📁 이동: {item}")
            elif src_path.is_dir():
                shutil.move(str(src_path), str(dst_path))
                print(f"📁 이동: {item}")
    
    print(f"\\n✅ 무거운 파일 정리 완료!")
    print(f"📂 백업 위치: {backup_dir}")

if __name__ == "__main__":
    cleanup_heavy_files()
""")
    
    print("🧹 데이터 정리 스크립트 생성 완료")

if __name__ == "__main__":
    try:
        create_dev_workspace()
        create_data_cleanup_script()
        
        print("\n" + "="*50)
        print("🎯 다음 단계:")
        print("1. Cursor를 재시작하세요")
        print("2. 'dev_workspace' 폴더를 Cursor에서 열어주세요")
        print("3. 필요시 'python cleanup_heavy_data.py'를 실행하세요")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1) 