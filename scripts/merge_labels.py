import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

def merge_and_sort_labels():
    """
    data/processed/btc_usdt_kst/labeled/ 폴더의 모든 라벨 parquet 파일을
    하나의 파일로 병합하고 타임스탬프 기준으로 정렬합니다.
    """
    # 🔥 경로 수정 - 실제 라벨 생성 경로와 일치
    input_dir = Path('data/processed/btc_usdt_kst/labeled')
    output_dir = Path('data/processed/btc_usdt_kst/labeled')
    output_file = output_dir / 'merged_all_labels.parquet'
    
    print(f"📁 입력 경로: {input_dir}")
    print(f"📁 출력 경로: {output_file}")
    
    # 🔥 실제 파일 패턴으로 수정
    label_files = list(input_dir.glob('*_macd_labeled.parquet'))
    
    # 📊 사용 가능한 파일 확인
    all_files = list(input_dir.glob('*.parquet'))
    print(f"📂 전체 파일 수: {len(all_files)}개")
    if all_files:
        print("📋 발견된 파일들:")
        for file in sorted(all_files):
            print(f"  - {file.name}")
    
    if not label_files:
        print(f"❌ '{input_dir}' 디렉토리에서 라벨 파일을 찾을 수 없습니다.")
        print("🔍 '*_macd_labeled.parquet' 패턴의 파일이 필요합니다.")
        return

    print(f"\n🔄 총 {len(label_files)}개의 라벨 파일을 병합합니다.")

    all_labels_df_list = []
    successful_files = 0
    failed_files = 0
    
    for file in tqdm(label_files, desc="📊 라벨 파일 처리중"):
        try:
            # 🔥 파일명에서 타임프레임 추출 (수정된 패턴)
            match = re.search(r'(.+)_macd_labeled\.parquet', file.name)
            if not match:
                print(f"⚠️ 파일명 패턴 불일치, 건너뛰기: {file.name}")
                failed_files += 1
                continue
                
            timeframe = match.group(1)
            print(f"📈 처리중: {timeframe}")

            df = pd.read_parquet(file)
            
            # 기본 데이터 검증
            if df.empty:
                print(f"⚠️ '{file.name}': 빈 데이터프레임, 건너뛰기")
                failed_files += 1
                continue
            
            # 시간대 정보 통일 (timezone-naive로 변환)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                print(f"  🔧 시간대 정보 제거: {timeframe}")

            # 인덱스 타입 확인
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"❌ '{file.name}': 인덱스가 DatetimeIndex가 아님 ({type(df.index)})")
                
                # 🔧 자동 복구 시도
                try:
                    if 'timestamp' in df.columns:
                        df.index = pd.to_datetime(df.columns['timestamp'])
                        df = df.drop('timestamp', axis=1)
                    else:
                        df.index = pd.to_datetime(df.index)
                    print(f"  ✅ 인덱스 복구 성공: {timeframe}")
                except Exception as e:
                    print(f"  ❌ 인덱스 복구 실패: {e}")
                    failed_files += 1
                    continue

            # 필수 컬럼 확인
            if 'label' not in df.columns:
                print(f"⚠️ '{file.name}': 'label' 컬럼 없음")
                failed_files += 1
                continue
            
            # 라벨 분포 확인
            label_counts = df['label'].value_counts().sort_index()
            non_zero_labels = df[df['label'] != 0]
            
            print(f"  📊 {timeframe} 라벨 분포: {dict(label_counts)}")
            print(f"  🎯 신호 라벨: {len(non_zero_labels):,}개")
            
            # 타임프레임 정보 추가
            df_copy = df.copy()
            df_copy['timeframe'] = timeframe
            df_copy['file_source'] = file.name
            
            all_labels_df_list.append(df_copy)
            successful_files += 1

        except Exception as e:
            print(f"❌ '{file.name}' 처리 중 오류: {e}")
            failed_files += 1

    # 결과 확인
    print(f"\n📊 처리 결과:")
    print(f"  ✅ 성공: {successful_files}개")
    print(f"  ❌ 실패: {failed_files}개")
    
    if not all_labels_df_list:
        print("❌ 처리할 라벨 데이터가 없습니다.")
        return

    # 🔄 모든 데이터프레임 병합
    print("\n🔄 데이터프레임 병합 중...")
    merged_df = pd.concat(all_labels_df_list, ignore_index=False)
    
    # 인덱스 이름 설정 (없는 경우)
    if merged_df.index.name is None:
        merged_df.index.name = 'timestamp'
    
    # 중복 제거 (같은 시간, 같은 타임프레임)
    print("🔄 중복 데이터 제거 중...")
    before_dedup = len(merged_df)
    merged_df = merged_df.reset_index()
    merged_df = merged_df.drop_duplicates(subset=[merged_df.columns[0], 'timeframe'], keep='first')
    after_dedup = len(merged_df)
    
    print(f"  📊 중복 제거: {before_dedup:,} → {after_dedup:,} (제거: {before_dedup-after_dedup:,}개)")
    
    # 타임스탬프 기준 정렬
    print("🔄 타임스탬프 기준 정렬 중...")
    timestamp_col = merged_df.columns[0]  # 첫 번째 컬럼이 타임스탬프
    merged_df = merged_df.sort_values([timestamp_col, 'timeframe'])
    
    # 📊 최종 통계
    total_labels = len(merged_df)
    signal_labels = len(merged_df[merged_df['label'] != 0])
    
    print("\n" + "="*60)
    print("🎉 라벨 병합 완료!")
    print("="*60)
    print(f"📊 총 라벨 수: {total_labels:,}개")
    print(f"🎯 신호 라벨: {signal_labels:,}개 ({signal_labels/total_labels*100:.1f}%)")
    print(f"📈 타임프레임 수: {merged_df['timeframe'].nunique()}개")
    
    # 타임프레임별 통계
    print("\n📊 타임프레임별 라벨 수:")
    tf_stats = merged_df['timeframe'].value_counts().sort_index()
    for tf, count in tf_stats.items():
        print(f"  {tf}: {count:,}개")
    
    # 📁 결과 저장
    print(f"\n💾 결과 저장 중...")
    
    # Parquet 저장
    merged_df.to_parquet(output_file, index=False)
    print(f"  ✅ Parquet: {output_file}")
    
    # CSV 저장 (선택적)
    csv_output_file = output_dir / 'merged_all_labels.csv'
    merged_df.to_csv(csv_output_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ CSV: {csv_output_file}")
    
    # 📋 미리보기
    print(f"\n📋 데이터 미리보기:")
    print("첫 5행:")
    print(merged_df.head())
    print("\n마지막 5행:")
    print(merged_df.tail())
    
    return output_file

if __name__ == "__main__":
    print("🚀 MACD 라벨 병합 시작")
    print("="*60)
    
    result_file = merge_and_sort_labels()
    
    if result_file:
        print(f"\n🎯 다음 단계: Phase 2.5 분석을 다시 실행하세요!")
        print(f"📁 병합된 라벨 파일: {result_file}")
    
    print("\n🏁 라벨 병합 완료!")