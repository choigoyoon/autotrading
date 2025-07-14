# type: ignore
# pylint: disable-all
"""
파일명: project_sync_report.py
우주아빠님 + Claude + Cursor AI 3자 동기화용 리포트
"""

import os
import sys
import psutil
import platform
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def generate_sync_report():
    print("🎯 퀀트매매 프로젝트 3자 동기화 리포트")
    print("=" * 80)
    
    # 1. 컴퓨터 사양 정보
    print("\n💻 컴퓨터 사양:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU 코어: {psutil.cpu_count()} 개")
    print(f"메모리: {round(psutil.virtual_memory().total / (1024**3), 1)} GB")
    print(f"Python: {sys.version.split()[0]}")
    
    # 2. 프로젝트 폴더 구조
    print("\n📁 프로젝트 폴더 구조:")
    project_map = {}
    
    important_paths = [
        ".",
        "src",
        "src/analysis", 
        "src/data",
        "src/indicators",
        "src/labeling",
        "data",
        "data/processed",
        "data/raw", 
        "analysis_results",
        "models",
        "notebooks"
    ]
    
    for path in important_paths:
        if os.path.exists(path):
            files = []
            if os.path.isdir(path):
                all_files = os.listdir(path)
                py_files = [f for f in all_files if f.endswith('.py')]
                data_files = [f for f in all_files if f.endswith(('.parquet', '.csv'))]
                config_files = [f for f in all_files if f.endswith(('.txt', '.md', '.toml', '.py'))]
                
                files = {
                    'python_files': py_files,
                    'data_files': data_files, 
                    'config_files': config_files,
                    'total_files': len(all_files)
                }
            
            project_map[path] = files
            # files가 딕셔너리일 때만 .get()을 사용하도록 수정
            total_files = files.get('total_files', 0) if isinstance(files, dict) else len(files)
            print(f"  📂 {path}: {total_files}개 파일")
    
    # 3. 현재 진행 단계
    print("\n🎯 현재 진행 단계:")
    progress = {}
    
    # Phase 1: 데이터 수집 및 라벨링
    has_raw_data = len(list(Path('data/raw').glob('*.parquet'))) > 0 if os.path.exists('data/raw') else False
    has_processed_data = len(list(Path('data/processed').glob('*_indicators.parquet'))) > 0 if os.path.exists('data/processed') else False
    
    progress['Phase1_데이터수집'] = "✅ 완료" if has_raw_data else "❌ 미완료"
    progress['Phase1_라벨링'] = "✅ 완료" if has_processed_data else "❌ 미완료"
    
    # Phase 2: 라벨링 검증
    has_validation = os.path.exists('analysis_results/label_validation.csv')
    has_consensus = os.path.exists('analysis_results/timeframe_consensus_analysis.csv')
    
    progress['Phase2_라벨검증'] = "✅ 완료" if has_validation else "❌ 미완료"
    progress['Phase2_타임프레임합의'] = "✅ 완료" if has_consensus else "❌ 미완료"
    
    # Phase 3: 딥러닝 분석  
    has_pattern_analysis = os.path.exists('analysis_results/divergence_strength_analysis.csv')
    has_wm_analysis = os.path.exists('analysis_results/zigzag_pattern_analysis.csv')
    
    progress['Phase3_패턴분석'] = "✅ 완료" if has_pattern_analysis else "❌ 미완료"
    progress['Phase3_WM패턴'] = "✅ 완료" if has_wm_analysis else "❌ 미완료"
    
    for phase, status in progress.items():
        print(f"  {phase}: {status}")
    
    # 4. 데이터 현황 상세
    print("\n📊 데이터 현황:")
    if os.path.exists('data/processed'):
        parquet_files = list(Path('data/processed').glob('*.parquet'))
        print(f"  처리된 데이터 파일: {len(parquet_files)}개")
        
        if parquet_files:
            sample_file = parquet_files[0]
            try:
                df = pd.read_parquet(sample_file)
                print(f"  샘플 파일: {sample_file.name}")
                print(f"  데이터 기간: {len(df)}개 캔들")
                print(f"  컬럼 수: {len(df.columns)}개")
                print(f"  라벨링 여부: {'macd_label' in df.columns}")
                
                if 'macd_label' in df.columns:
                    label_dist = df['macd_label'].value_counts()
                    print(f"  라벨 분포: 매수{label_dist.get(1,0)}개, 매도{label_dist.get(-1,0)}개, 관망{label_dist.get(0,0)}개")
                    
            except Exception as e:
                print(f"  데이터 읽기 오류: {e}")
    
    # 5. 분석 결과 현황
    print("\n📈 분석 결과 현황:")
    if os.path.exists('analysis_results'):
        result_files = list(Path('analysis_results').glob('*.csv'))
        print(f"  분석 결과 파일: {len(result_files)}개")
        for file in result_files:
            print(f"    - {file.name}")
    else:
        print("  분석 결과 없음")
    
    # 6. 실행 가능한 스크립트
    print("\n🐍 실행 가능한 스크립트:")
    key_scripts = [
        'pipeline_runner.py',
        'src/analysis/label_validation_analyzer.py',
        'src/analysis/timeframe_consensus_analyzer.py', 
        'src/analysis/divergence_quantifier.py'
    ]
    
    for script in key_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
    
    # 7. 다음 할 일
    print("\n🎯 다음 할 일:")
    
    if not has_processed_data:
        print("  1. pipeline_runner.py 실행해서 라벨링 완료")
    elif not has_validation:
        print("  1. 라벨링 검증 분석 실행")
    elif not has_consensus:
        print("  1. 타임프레임 합의 분석 실행")
    elif not has_pattern_analysis:
        print("  1. 다이버전스 패턴 분석 실행")
    else:
        print("  1. W/M 패턴 지그재그 분석 실행")
        print("  2. 딥러닝 성공/실패 패턴 분석")
        print("  3. 매매법 5단계 검증 시스템 구축")
    
    # 8. JSON 리포트 저장
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'computer_specs': {
            'os': f"{platform.system()} {platform.release()}",
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'python_version': sys.version.split()[0]
        },
        'project_structure': project_map,
        'progress': progress,
        'next_steps': "위 출력 참조"
    }
    
    with open('project_sync_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 상세 리포트 저장: project_sync_report.json")
    print("\n" + "=" * 80)
    print("📋 이 리포트를 Claude와 Cursor AI에게 공유하세요!")

if __name__ == "__main__":
    generate_sync_report() 