"""
파일명: project_sync_report.py
우주아빠님 + Claude + Cursor AI 3자 동기화용 리포트
"""
import sys
import platform
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import psutil  # type: ignore
import pandas as pd
from loguru import logger

# --- Constants ---
IMPORTANT_PATHS: List[str] = [
    ".", "src", "src/analysis", "src/data", "src/features", "src/labeling",
    "data", "data/processed", "data/raw", "results", "models", "tools", "scripts", "configs"
]

KEY_SCRIPTS: List[str] = [
    'tools/pipeline_runner.py',
    'src/analysis/label_validation_analyzer.py',
    'src/analysis/timeframe_consensus_analyzer.py',
    'src/analysis/divergence_quantifier.py'
]

# --- Helper Functions ---
def get_directory_info(path: Path) -> Dict[str, Any]:
    """주어진 경로의 파일 정보를 딕셔너리로 반환합니다."""
    if not path.is_dir():
        return {}
    
    all_files = list(path.iterdir())
    return {
        'python_files': [f.name for f in all_files if f.name.endswith('.py')],
        'data_files': [f.name for f in all_files if f.name.endswith(('.parquet', '.csv'))],
        'config_files': [f.name for f in all_files if f.name.endswith(('.txt', '.md', '.toml', '.py', '.yaml', '.json'))],
        'total_files': len(all_files)
    }

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    # 이 스크립트가 src/utils에 있다고 가정
    return Path(__file__).resolve().parent.parent.parent

# --- Main Report Generation ---
def generate_sync_report() -> None:
    """프로젝트 동기화 리포트를 생성하고 터미널에 출력하며, JSON 파일로 저장합니다."""
    project_root = get_project_root()
    logger.info(f"Project root directory: {project_root}")
    
    print("🎯 퀀트매매 프로젝트 3자 동기화 리포트")
    print("=" * 80)
    
    # 1. 컴퓨터 사양 정보
    print("\n💻 컴퓨터 사양:")
    print(f"OS: {platform.system()} {platform.release()}")
    try:
        print(f"CPU: {platform.processor()}")
    except Exception:
        print("CPU: 정보를 가져올 수 없습니다.")
    print(f"CPU 코어: {psutil.cpu_count(logical=True)}개 (Physical: {psutil.cpu_count(logical=False)})")
    print(f"메모리: {round(psutil.virtual_memory().total / (1024**3), 1)} GB")
    print(f"Python: {sys.version.split()[0]}")
    
    # 2. 프로젝트 폴더 구조
    print("\n📁 프로젝트 폴더 구조:")
    project_map: Dict[str, Dict[str, Any]] = {}
    for path_str in IMPORTANT_PATHS:
        path = project_root / path_str
        if path.exists():
            dir_info = get_directory_info(path)
            project_map[path_str] = dir_info
            total_files = dir_info.get('total_files', 0)
            print(f"  📂 {path_str:<25}: {total_files}개 파일")
    
    # 3. 현재 진행 단계
    print("\n🎯 현재 진행 단계:")
    progress: Dict[str, str] = {}
    
    def check_path_content(path_str: str, glob_pattern: str) -> bool:
        path = project_root / path_str
        return path.exists() and any(path.glob(glob_pattern))

    progress['Phase1_데이터수집'] = "✅ 완료" if check_path_content('data/raw', '*.parquet') else "❌ 미완료"
    progress['Phase1_데이터가공'] = "✅ 완료" if check_path_content('data/processed', '*_features.parquet') else "❌ 미완료"
    progress['Phase2_라벨링'] = "✅ 완료" if check_path_content('data/processed', '*_labeled.parquet') else "❌ 미완료"
    progress['Phase3_모델학습'] = "✅ 완료" if check_path_content('models', '*.keras') or check_path_content('models', '*.pkl') else "❌ 미완료"
    
    for phase, status in progress.items():
        print(f"  {phase:<25}: {status}")
    
    # 4. 데이터 현황 상세
    print("\n📊 데이터 현황:")
    processed_path = project_root / 'data/processed'
    if processed_path.exists():
        labeled_files = list(processed_path.glob('*_labeled.parquet'))
        print(f"  처리된 라벨링 파일: {len(labeled_files)}개")
        
        if labeled_files:
            sample_file = labeled_files[0]
            try:
                df = pd.read_parquet(sample_file)
                print(f"  샘플 파일: {sample_file.name}")
                if df.index.is_monotonic_increasing:
                    print(f"  데이터 기간: {df.index.min()} ~ {df.index.max()}")
                else:
                    print("  데이터 기간: 인덱스가 정렬되지 않음")
                print(f"  데이터 크기: {len(df)} 행, {len(df.columns)} 열")
                
                if 'macd_label' in df.columns:
                    label_dist = df['macd_label'].value_counts()
                    print(f"  라벨 분포: 매수({label_dist.get(1,0)}), 매도({label_dist.get(-1,0)}), 관망({label_dist.get(0,0)})")
                    
            except Exception as e:
                logger.error(f"데이터 읽기 오류: {e}")
                print(f"  데이터 읽기 오류: {e}")
    
    # 5. 분석 결과 현황
    print("\n📈 분석 결과 현황:")
    results_path = project_root / 'results'
    if results_path.exists():
        result_files = [f.name for f in results_path.glob('*.csv')]
        print(f"  분석 결과 파일: {len(result_files)}개")
        for file_name in result_files:
            print(f"    - {file_name}")
    else:
        print("  분석 결과 없음")
    
    # 6. 실행 가능한 주요 스크립트
    print("\n🐍 실행 가능한 주요 스크립트:")
    for script_path_str in KEY_SCRIPTS:
        script_path = project_root / script_path_str
        status = "✅" if script_path.exists() else "❌"
        print(f"  {status} {script_path_str}")
    
    # 7. 다음 할 일 (간소화된 제안)
    print("\n🚀 다음 추천 작업:")
    next_step = "알 수 없음"
    if progress.get('Phase1_데이터가공') == '❌ 미완료':
        next_step = "1. 데이터 전처리 및 피처 생성 실행"
    elif progress.get('Phase2_라벨링') == '❌ 미완료':
        next_step = "2. 데이터 라벨링 실행"
    elif progress.get('Phase3_모델학습') == '❌ 미완료':
        next_step = "3. 모델 학습 실행"
    else:
        next_step = "4. 백테스팅 및 가상매매 시스템 실행"
    print(f"  ➡️  {next_step}")
    
    # 8. JSON 리포트 저장
    report_data: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'computer_specs': {
            'os': f"{platform.system()} {platform.release()}",
            'cpu_cores': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'python_version': sys.version.split()[0]
        },
        'project_structure': project_map,
        'progress': progress,
        'next_steps': next_step
    }
    
    report_file = project_root / 'project_sync_report.json'
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 상세 리포트 저장: {report_file}")
    except IOError as e:
        logger.error(f"리포트 파일 저장 실패: {e}")
        print(f"\n❌ 상세 리포트 저장 실패: {report_file}")

    print("\n" + "=" * 80)
    print("📋 이 리포트를 Claude와 Cursor AI에게 공유하세요!")

if __name__ == "__main__":
    # 로그 설정
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_dir / "sync_report_{time}.log", level="DEBUG", rotation="10 MB")
    
    generate_sync_report()