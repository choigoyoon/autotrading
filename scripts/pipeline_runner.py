import subprocess
import sys
from pathlib import Path
import os
import time
import threading
from queue import Queue, Empty
import argparse

class OptimizedPipelineRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.resolve()
        self.step_times = {}
    
    def setup_environment(self):
        """환경 설정 최적화 + 인코딩 문제 해결"""
        env = os.environ.copy()
        
        # 🔥 인코딩 문제 해결
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSENCODING'] = 'utf-8'
        
        # 다중 경로 설정
        python_paths = [
            str(self.project_root),
            str(self.project_root / 'src'),
            str(self.project_root / 'tools')
        ]
        env['PYTHONPATH'] = os.pathsep.join(python_paths)
        return env
    
    def run_with_timeout_and_output(self, script_path, args=None, timeout=3600):
        """실시간 출력 + 타임아웃 처리 + 인코딩 문제 해결"""
        if args is None:
            args = []
        
        full_script_path = self.project_root / script_path
        command = [sys.executable, str(full_script_path)] + args
        env = self.setup_environment()
        
        print(f"\n🚀 실행 중: {os.path.basename(script_path)}")
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.project_root,
                env=env,
                encoding='utf-8',  # 🔥 UTF-8 강제 설정
                errors='replace'   # 🔥 디코딩 에러 시 대체문자 사용
            )
            
            # 실시간 출력 스레드 (에러 처리 강화)
            def read_output(proc, queue):
                try:
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        # 🔥 안전한 출력 처리
                        try:
                            queue.put(('output', line.rstrip()))
                        except UnicodeDecodeError:
                            queue.put(('output', '[인코딩 에러 - 출력 생략]'))
                    
                    proc.wait()
                    queue.put(('done', proc.returncode))
                except Exception as e:
                    queue.put(('error', str(e)))
            
            output_queue = Queue()
            reader_thread = threading.Thread(
                target=read_output, 
                args=(process, output_queue)
            )
            reader_thread.daemon = True
            reader_thread.start()
            
            # 타임아웃과 실시간 출력 처리
            last_output_time = time.time()
            
            while True:
                try:
                    msg_type, content = output_queue.get(timeout=1.0)
                    
                    if msg_type == 'output':
                        # 🔥 안전한 출력
                        try:
                            print(content)
                        except UnicodeEncodeError:
                            print('[출력 인코딩 에러]')
                        last_output_time = time.time()
                        
                    elif msg_type == 'done':
                        duration = time.time() - start_time
                        print(f"✅ 완료: {duration:.2f}초")
                        return content == 0
                        
                    elif msg_type == 'error':
                        print(f"❌ 오류: {content}")
                        return False
                        
                except Empty:
                    if time.time() - last_output_time > timeout:
                        print(f"⏰ 타임아웃: {timeout}초 초과")
                        process.terminate()
                        return False
                    
                    if process.poll() is not None:
                        break
            
            return process.returncode == 0
            
        except Exception as e:
            print(f"❌ 예외 발생: {e}")
            return False

# 🔥 추가 해결책: 직접 실행 함수
def run_script_direct(script_path, args=None):
    """직접 실행 방식 (인코딩 문제 우회)"""
    if args is None:
        args = []
    
    project_root = Path(__file__).parent.resolve()
    full_script_path = project_root / script_path
    
    # 환경 설정
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONPATH'] = str(project_root) + os.pathsep + str(project_root / 'src')
    
    print(f"\n🚀 직접 실행: {os.path.basename(script_path)}")
    
    try:
        # 🔥 간단한 실행 방식
        result = subprocess.run(
            [sys.executable, str(full_script_path)] + args,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800  # 30분 타임아웃
        )
        
        if result.stdout:
            print("📋 출력:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ 에러:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 성공!")
            return True
        else:
            print(f"❌ 실패 (코드: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 타임아웃 발생")
        return False
    except Exception as e:
        print(f"❌ 예외: {e}")
        return False

# 🔥 전체 파이프라인 (Phase 1 ~ 2.5)
def run_complete_pipeline():
    """완전한 파이프라인 (Phase 1 → 2 → 2.5)"""
    print("=" * 60)
    print("🚀 완전한 퀀트 트레이딩 파이프라인 (Phase 1 ~ 2.5)")
    print("=" * 60)
    
    runner = OptimizedPipelineRunner()
    results = {}
    
    # 📊 Phase 1: 데이터 준비
    print("\n" + "="*50)
    print("📊 Phase 1: 데이터 준비")
    print("="*50)
    
    print("\n🔄 [1.1] 데이터 수집...")
    success1_1 = runner.run_with_timeout_and_output(
        'src/data/collectors/simple_collector.py', 
        ['--safe-update'],
        timeout=300
    )
    
    if not success1_1:
        print("🔄 직접 실행 방식으로 재시도...")
        success1_1 = run_script_direct(
            'src/data/collectors/simple_collector.py',
            ['--safe-update']
        )
    
    results['data_collection'] = success1_1
    
    print("\n⚡ [1.2] 데이터 리샘플링...")
    success1_2 = runner.run_with_timeout_and_output(
        'src/data/processors/resample_data.py',
        timeout=600
    )
    
    if not success1_2:
        print("🔄 직접 실행 방식으로 재시도...")
        success1_2 = run_script_direct('src/data/processors/resample_data.py')
    
    results['resampling'] = success1_2
    
    # 📈 Phase 2: 라벨링 (간단한 확인만)
    print("\n" + "="*50)
    print("📈 Phase 2: 라벨링 상태 확인")
    print("="*50)
    
    # 라벨링 데이터 존재 여부만 확인
    label_files = list(Path('data/processed/btc_usdt_kst/labeled').glob('*.parquet'))
    if label_files:
        print(f"✅ 라벨링 데이터 발견: {len(label_files)}개 파일")
        results['labeling_check'] = True
    else:
        print("⚠️ 라벨링 데이터 없음 - Phase 2 실행 필요")
        results['labeling_check'] = False
    
    # 🔍 Phase 2.5: 라벨 품질 검증
    print("\n" + "="*50)
    print("🔍 Phase 2.5: 라벨 품질 검증 및 분석")
    print("="*50)
    
    print("\n📊 [2.5.1] MACD 라벨 분포 분석...")
    success2_5_1 = runner.run_with_timeout_and_output(
        'tools/analyze_macd_labels.py',
        timeout=600
    )
    
    if not success2_5_1:
        print("🔄 직접 실행 방식으로 재시도...")
        success2_5_1 = run_script_direct('tools/analyze_macd_labels.py')
    
    results['macd_analysis'] = success2_5_1
    
    print("\n📈 [2.5.2] 라벨 성과 분석 (고속)...")
    success2_5_2 = runner.run_with_timeout_and_output(
        'tools/analyze_label_performance_fast.py',
        timeout=900
    )
    
    if not success2_5_2:
        print("🔄 직접 실행 방식으로 재시도...")
        success2_5_2 = run_script_direct('tools/analyze_label_performance_fast.py')
    
    results['performance_fast'] = success2_5_2
    
    print("\n💰 [2.5.3] 수익률 분포 분석...")
    success2_5_3 = runner.run_with_timeout_and_output(
        'tools/analyze_profit_distribution.py',
        timeout=600
    )
    
    if not success2_5_3:
        print("🔄 직접 실행 방식으로 재시도...")
        success2_5_3 = run_script_direct('tools/analyze_profit_distribution.py')
    
    results['profit_distribution'] = success2_5_3
    
    print("\n🎯 [2.5.4] 상세 라벨 성과 분석...")
    success2_5_4 = runner.run_with_timeout_and_output(
        'tools/analyze_label_performance.py',
        timeout=1800
    )
    
    if not success2_5_4:
        print("🔄 직접 실행 방식으로 재시도...")
        success2_5_4 = run_script_direct('tools/analyze_label_performance.py')
    
    results['performance_detailed'] = success2_5_4
    
    # 📋 최종 결과 요약
    print("\n" + "="*60)
    print("📋 전체 파이프라인 실행 결과")
    print("="*60)
    
    phases = {
        'Phase 1 - 데이터': ['data_collection', 'resampling'],
        'Phase 2 - 라벨링': ['labeling_check'],
        'Phase 2.5 - 분석': ['macd_analysis', 'performance_fast', 'profit_distribution', 'performance_detailed']
    }
    
    total_success = 0
    total_tasks = 0
    
    for phase_name, tasks in phases.items():
        print(f"\n{phase_name}:")
        phase_success = 0
        for task in tasks:
            success = results.get(task, False)
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  - {task}: {status}")
            if success:
                phase_success += 1
                total_success += 1
            total_tasks += 1
        
        phase_rate = phase_success / len(tasks) * 100
        print(f"  📊 {phase_name} 성공률: {phase_success}/{len(tasks)} ({phase_rate:.1f}%)")
    
    overall_rate = total_success / total_tasks * 100
    print(f"\n🎯 전체 성공률: {total_success}/{total_tasks} ({overall_rate:.1f}%)")
    
    if overall_rate >= 80:
        print("\n🎉 파이프라인 성공적으로 완료!")
        print("🚀 다음 단계: Phase 3 (AI 모델 학습) 준비 완료!")
    elif overall_rate >= 60:
        print("\n⚠️ 파이프라인 부분 완료")
        print("일부 작업이 실패했지만 진행 가능합니다.")
    else:
        print("\n❌ 파이프라인 실패")
        print("대부분의 작업이 실패했습니다. 설정을 확인해주세요.")
    
    return overall_rate >= 60

# 🔥 Phase 2.5만 실행
def run_phase_2_5_only():
    """Phase 2.5 라벨 품질 검증만 실행"""
    print("=" * 60)
    print("🔍 Phase 2.5: 라벨 품질 검증 및 분석")
    print("=" * 60)
    
    runner = OptimizedPipelineRunner()
    results = {}
    
    # Phase 2.5 스크립트들
    scripts = [
        ('MACD 라벨 분석', 'tools/analyze_macd_labels.py', 600),
        ('라벨 성과 분석 (고속)', 'tools/analyze_label_performance_fast.py', 900),
        ('수익률 분포 분석', 'tools/analyze_profit_distribution.py', 600),
        ('상세 성과 분석', 'tools/analyze_label_performance.py', 1800)
    ]
    
    for i, (name, script_path, timeout) in enumerate(scripts, 1):
        print(f"\n📊 [{i}/4] {name}...")
        
        success = runner.run_with_timeout_and_output(script_path, timeout=timeout)
        if not success:
            print("🔄 직접 실행 방식으로 재시도...")
            success = run_script_direct(script_path)
        
        results[script_path] = success
    
    # 결과 요약
    successful = sum(results.values())
    total = len(results)
    success_rate = successful / total * 100
    
    print(f"\n📊 Phase 2.5 결과: {successful}/{total} 성공 ({success_rate:.1f}%)")
    
    return success_rate >= 75

# 🔥 개별 스크립트 실행
def run_individual_script(script_name):
    """개별 스크립트 실행"""
    script_map = {
        'collect': 'src/data/collectors/simple_collector.py',
        'resample': 'src/data/processors/resample_data.py',
        'macd': 'tools/analyze_macd_labels.py',
        'performance': 'tools/analyze_label_performance_fast.py',
        'profit': 'tools/analyze_profit_distribution.py',
        'detailed': 'tools/analyze_label_performance.py'
    }
    
    if script_name not in script_map:
        print(f"❌ 알 수 없는 스크립트: {script_name}")
        print(f"사용 가능: {list(script_map.keys())}")
        return False
    
    runner = OptimizedPipelineRunner()
    script_path = script_map[script_name]
    
    args = ['--safe-update'] if script_name == 'collect' else []
    
    success = runner.run_with_timeout_and_output(script_path, args, timeout=1800)
    if not success:
        print("🔄 직접 실행 방식으로 재시도...")
        success = run_script_direct(script_path, args)
    
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="퀀트 트레이딩 파이프라인")
    parser.add_argument('--phase2.5-only', action='store_true', 
                        help="Phase 2.5 라벨 분석만 실행")
    parser.add_argument('--individual', 
                        choices=['collect', 'resample', 'macd', 'performance', 'profit', 'detailed'],
                        help="개별 스크립트 실행")
    
    args = parser.parse_args()
    
    if args.individual:
        # 개별 스크립트 실행
        print(f"🚀 개별 실행: {args.individual}")
        success = run_individual_script(args.individual)
        print(f"결과: {'✅ 성공' if success else '❌ 실패'}")
        
    elif getattr(args, 'phase2.5-only', False):
        # Phase 2.5만 실행
        run_phase_2_5_only()
        
    else:
        # 전체 파이프라인 실행
        run_complete_pipeline()
    
    print("\n" + "="*60)
    print("🏁 파이프라인 실행 완료")
    print("="*60)
