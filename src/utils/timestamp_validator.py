import pandas as pd
from pathlib import Path
import sys

def validate_timestamps(file_path: Path):
    """
    CSV 파일의 timestamp 열을 검증하여 리포트를 출력합니다.

    Args:
        file_path (Path): 검증할 CSV 파일 경로.
    """
    if not file_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: '{file_path}'", file=sys.stderr)
        return

    print(f"'{file_path}' 파일 타임스탬프 검증을 시작합니다...")

    try:
        # CSV 파일을 읽고 timestamp를 datetime 객체로 변환
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print("파일 로딩 및 파싱 완료.")

        # 1. 중복 검사
        duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
        if not duplicates.empty:
            print(f"\n[오류] {len(duplicates)}개의 중복된 타임스탬프를 발견했습니다:")
            print(duplicates)
        else:
            print("\n[성공] 중복된 타임스탬프가 없습니다.")

        # 2. 순서 검사
        if not df['timestamp'].is_monotonic_increasing:
            print("\n[오류] 타임스탬프가 시간순으로 정렬되어 있지 않습니다.")
            # 정렬하여 추가 분석 진행
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            print("  - 분석을 위해 데이터를 시간순으로 정렬했습니다.")
        else:
            print("\n[성공] 타임스탬프가 시간순으로 올바르게 정렬되어 있습니다.")

        # 3. 누락 검사 (Gaps)
        df['diff'] = df['timestamp'].diff()
        
        # 예상 간격 (1분)
        expected_diff = pd.Timedelta(minutes=1)
        
        # 예상 간격과 다른 모든 지점 찾기 (첫 행의 NaT는 제외)
        gaps = df[df['diff'] > expected_diff]

        if not gaps.empty:
            print(f"\n[경고] {len(gaps)}개의 누락된 구간(gap)을 발견했습니다.")
            print("아래는 누락된 구간의 시작 지점과 누락된 시간입니다:")
            
            # 누락된 시간 계산 (실제 차이 - 예상 차이)
            gaps_info = gaps[['timestamp', 'diff']].copy()
            gaps_info['missing_duration'] = gaps_info['diff'] - expected_diff
            
            # 보기 쉽게 출력
            for _, row in gaps_info.head(10).iterrows():
                gap_start_time = row['timestamp'] - row['diff']
                gap_end_time = row['timestamp']
                print(f"  - {gap_start_time} 부터 {row['missing_duration']} 동안 데이터 누락 후 {gap_end_time} 에서 재개")
            
            if len(gaps) > 10:
                print(f"  ... (총 {len(gaps)}개 중 상위 10개만 표시)")

        else:
            print("\n[성공] 1분 간격의 누락된 타임스탬프가 없습니다.")
            
        print("\n검증 완료.")

    except Exception as e:
        print(f"검증 중 오류 발생: {e}", file=sys.stderr)

if __name__ == '__main__':
    csv_file = Path('data/rwa/csv/btc_1min.csv')
    validate_timestamps(csv_file) 