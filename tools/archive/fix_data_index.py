from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
import sys

@dataclass
class DataHealthReport:
    raw_data_ok: bool = False
    resampled_data_ok: bool = False
    labeled_data_ok: bool = False
    issues: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """모든 데이터 파이프라인 단계가 정상인지 확인합니다."""
        return self.raw_data_ok and self.resampled_data_ok and self.labeled_data_ok

    def get_recommended_action(self) -> str:
        """데이터 상태에 따라 권장되는 다음 작업을 반환합니다."""
        if not self.raw_data_ok:
            return "data_collection"
        if not self.resampled_data_ok:
            return "resampling"
        if not self.labeled_data_ok:
            return "labeling"
        return "proceed_to_phase2.5"

def check_pipeline_health() -> DataHealthReport:
    """파이프라인 상태 진단 및 필요성 판단"""
    
    print("🔍 파이프라인 상태 진단 시작")
    print("="*50)
    
    report = DataHealthReport()
    
    # 1️⃣ 원본 데이터 체크
    raw_file = Path('data/rwa/parquet/btc_1min.parquet')
    if raw_file.exists():
        try:
            df = pd.read_parquet(raw_file)
            print(f"✅ 원본 데이터: {len(df):,}행 존재")
            report.raw_data_ok = True
        except Exception as e:
            print(f"❌ 원본 데이터 손상: {e}")
            report.issues.append("원본 데이터 손상")
    else:
        print(f"❌ 원본 데이터 없음: {raw_file}")
        report.issues.append("원본 데이터 없음")
    
    # 2️⃣ 리샘플링 데이터 체크
    resample_dir = Path('data/processed/btc_usdt_kst/resampled_ohlcv')
    resample_files = list(resample_dir.glob('*.parquet')) if resample_dir.exists() else []
    
    print(f"\n📊 리샘플링 데이터 체크:")
    if resample_files:
        print(f"   발견된 파일: {len(resample_files)}개")
        
        # 핵심 타임프레임 체크
        key_timeframes = ['1min.parquet', '5min.parquet', '1h.parquet', '1day.parquet']
        missing_key: list[str] = []
        corrupted_files: list[str] = []
        
        for tf_file in key_timeframes:
            file_path = resample_dir / tf_file
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                        print(f"   ✅ {tf_file}: {len(df):,}행, DatetimeIndex")
                    else:
                        print(f"   ⚠️ {tf_file}: 인덱스 문제 or 빈 데이터")
                        corrupted_files.append(tf_file)
                except Exception as e:
                    print(f"   ❌ {tf_file}: 손상됨 ({e})")
                    corrupted_files.append(tf_file)
            else:
                print(f"   ❌ {tf_file}: 없음")
                missing_key.append(tf_file)
        
        if not missing_key and not corrupted_files:
            report.resampled_data_ok = True
        else:
            if missing_key:
                report.issues.append(f"리샘플링 누락: {missing_key}")
            if corrupted_files:
                report.issues.append(f"리샘플링 손상: {corrupted_files}")
    else:
        print("   ❌ 리샘플링 파일 없음")
        report.issues.append("리샘플링 데이터 전체 없음")
    
    # 3️⃣ 라벨링 데이터 체크
    labels_dir = Path('data/processed/btc_usdt_kst/labeled')
    label_files = list(labels_dir.glob('*_macd_labeled.parquet')) if labels_dir.exists() else []
    
    print(f"\n🏷️ 라벨링 데이터 체크:")
    if label_files:
        print(f"   발견된 파일: {len(label_files)}개")
        
        # 샘플 파일 검증
        if label_files:
            sample_file = label_files[0]
            try:
                df = pd.read_parquet(sample_file)
                if 'label' in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    label_counts = df['label'].value_counts()
                    print(f"   ✅ 라벨 샘플 ({sample_file.name}): {dict(label_counts)}")
                    report.labeled_data_ok = True
                else:
                    print(f"   ❌ 라벨 구조 문제: 'label' 컬럼 또는 인덱스 오류")
                    report.issues.append("라벨 데이터 구조 문제")
            except Exception as e:
                print(f"   ❌ 라벨 파일 손상: {e}")
                report.issues.append("라벨 데이터 손상")
    else:
        print("   ❌ 라벨링 파일 없음")
        report.issues.append("라벨링 데이터 전체 없음")
    
    # 4️⃣ 종합 판단
    print(f"\n📋 종합 진단 결과:")
    print(f"   🔒 원본 데이터: {'✅' if report.raw_data_ok else '❌'}")
    print(f"   📊 리샘플링: {'✅' if report.resampled_data_ok else '❌'}")
    print(f"   🏷️ 라벨링: {'✅' if report.labeled_data_ok else '❌'}")
    
    recommended_action = report.get_recommended_action()
    
    if recommended_action == "data_collection":
        print(f"\n🚨 중대한 문제: 원본 데이터 없음!")
        print(f"   해결책: python src/data/collectors/simple_collector.py --safe-update")
    elif recommended_action == "resampling":
        print(f"\n⚠️ 리샘플링 문제 발견")
        print(f"   해결책: python src/data/processors/resample_data.py")
    elif recommended_action == "labeling":
        print(f"\n⚠️ 라벨링 문제 발견")
        print(f"   해결책: python tools/create_macd_zone_labels.py")
    else:
        print(f"\n🎉 모든 데이터 정상!")
        print(f"   다음 단계: python pipeline_runner.py --phase2.5-only")
    
    return report

def smart_repair_decision() -> tuple[bool, str]:
    """스마트한 복구 결정"""
    
    print("🤖 스마트 복구 의사결정 시작")
    print("="*50)
    
    health = check_pipeline_health()
    
    if health.is_healthy():
        print(f"\n✅ 복구 불필요!")
        print(f"🚀 바로 다음 단계 진행 가능:")
        print(f"   python tools/merge_labels.py")
        print(f"   python pipeline_runner.py --phase2.5-only")
        return False, "no_action_needed"
    
    action = health.get_recommended_action()
    
    if action == "data_collection":
        print(f"\n📥 데이터 수집 필요")
        print(f"   실행: python src/data/collectors/simple_collector.py --safe-update")
        return True, "data_collection"
        
    elif action == "resampling":
        print(f"\n🔄 리샘플링만 필요")
        print(f"   실행: python src/data/processors/resample_data.py")
        return True, "resampling_only"
        
    elif action == "labeling":
        print(f"\n🏷️ 라벨링만 필요") 
        print(f"   실행: python tools/create_macd_zone_labels.py")
        return True, "labeling_only"
    
    # 복합 문제인 경우
    print(f"\n🔧 복합 문제 발견 - 전체 복구 필요")
    print(f"   문제점: {health.issues}")
    return True, "full_repair"

def conditional_repair(timeframe: str = '1min'):
    """조건부 복구 실행"""
    
    print(f"🎯 조건부 복구 시작: {timeframe}")
    print("="*60)
    
    # 필요성 체크
    need_repair, action_type = smart_repair_decision()
    
    if not need_repair:
        print(f"\n🎉 복구 스킵! 데이터 상태 양호")
        return True
    
    print(f"\n🔧 필요한 복구 작업: {action_type}")
    
    # 사용자 확인
    response = input(f"\n복구를 진행하시겠습니까? (y/N): ").lower().strip()
    
    if response != 'y':
        print(f"🚫 사용자가 복구를 취소했습니다.")
        return False
    
    # 액션 타입별 실행
    if action_type == "data_collection":
        print(f"📥 데이터 수집 실행...")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'src/data/collectors/simple_collector.py', 
            '--safe-update'
        ])
        return result.returncode == 0
        
    elif action_type == "resampling_only":
        print(f"🔄 리샘플링만 실행...")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'src/data/processors/resample_data.py'
        ])
        return result.returncode == 0
        
    elif action_type == "labeling_only":
        print(f"🏷️ 라벨링만 실행...")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'tools/create_macd_zone_labels.py'
        ])
        return result.returncode == 0
        
    elif action_type == "full_repair":
        print(f"🔧 전체 복구 실행...")
        # 존재하지 않는 함수 호출 대신, 파이프라인을 순차적으로 실행
        
        import subprocess
        
        try:
            # 1. 데이터 수집
            print("\n--- 1. 데이터 수집 ---")
            _ = subprocess.run([sys.executable, 'src/data/collectors/simple_collector.py', '--safe-update'], check=True, capture_output=True, text=True)
            
            # 2. 데이터 리샘플링
            print("\n--- 2. 데이터 리샘플링 ---")
            _ = subprocess.run([sys.executable, 'src/data/processors/resample_data.py'], check=True, capture_output=True, text=True)
            
            # 3. 라벨링
            print("\n--- 3. 라벨링 ---")
            _ = subprocess.run([sys.executable, 'tools/create_macd_zone_labels.py'], check=True, capture_output=True, text=True)
            
            print("\n✅ 전체 복구 프로세스 완료.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 복구 프로세스 중 오류 발생: {e.stderr or 'N/A'}")
            return False
        except FileNotFoundError as e:
            print(f"❌ 실행 파일을 찾을 수 없습니다: {e}")
            return False
    
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="스마트 데이터 복구")
    _ = parser.add_argument('--check-only', action='store_true', help="상태 체크만")
    _ = parser.add_argument('--timeframe', type=str, default='1min', help="대상 타임프레임")
    
    args = parser.parse_args()
    
    # 타입 명시를 통해 Any 타입 문제 해결
    check_only_flag: bool = args.check_only
    timeframe_str: str = args.timeframe
    
    if check_only_flag:
        # 체크만
        _ = check_pipeline_health()
    else:
        # 조건부 복구
        success = conditional_repair(timeframe_str)
        
        if success:
            print(f"\n🎉 작업 완료!")
            print(f"🚀 다음 단계:")
            print(f"   python tools/merge_labels.py")
            print(f"   python pipeline_runner.py --phase2.5-only")
        else:
            print(f"\n❌ 작업 실패 또는 취소")