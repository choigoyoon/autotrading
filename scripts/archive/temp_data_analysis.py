import pandas as pd
from pandas import DataFrame, Series, Timedelta, DatetimeIndex, Timestamp
from pathlib import Path

def analyze_enhanced_features(file_path: str | Path) -> None:
    """
    강화된 피처 데이터셋 Parquet 파일을 로드하고 분석 결과를 출력합니다.
    
    Args:
        file_path (str | Path): 분석할 Parquet 파일 경로.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {path}")
        return

    try:
        df: DataFrame = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ 오류: '{path}' 파일 로드 실패: {e}")
        return
        
    if 'timestamp' in df.columns:
        # utc=True를 추가하여 타임존 인식 DatetimeIndex로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')  # type: ignore

    print(f"=== 🎯 {path.name} 현황 ===")
    print(f"📊 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")

    if isinstance(df.index, DatetimeIndex) and not df.index.empty:
        start_time: Timestamp = df.index.min()  # type: ignore
        end_time: Timestamp = df.index.max()  # type: ignore
        
        # start_time과 end_time이 Timestamp 객체인지 확인
        print(f"📅 기간: {start_time.date()} ~ {end_time.date()}")
        delta: Timedelta = end_time - start_time
        print(f"⏰ 총 기간: {delta.days}일")
    else:
        print("⏰ 기간 정보를 표시할 수 없습니다 (인덱스가 날짜가 아니거나 비어있음).")

    print("\n=== 📋 컬럼 목록 ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col} ({df[col].dtype})")

    print("\n=== 🔍 샘플 데이터 (첫 3행) ===")
    print(df.head(3))

    print("\n=== 📈 기본 통계 ===")
    print(df.describe())

    print("\n=== ⚠️ 결측치 확인 ===")
    null_counts: Series = df.isnull().sum()
    if null_counts.sum() > 0:
        print(null_counts[null_counts > 0])
    else:
        print("✅ 결측치 없음!")

    # 원래 코드는 주석 처리
    # print("\n=== 🎲 다이버전스 신호 분포 ===")
    # if 'divergence_signal' in df.columns:
    #     print(df['divergence_signal'].value_counts())
    # elif 'label' in df.columns:
    #     print(df['label'].value_counts())
    # else:
    #     print("라벨 컬럼을 찾아서 분포를 확인해주세요") 

if __name__ == "__main__":
    file_to_analyze = 'results/ml_analysis_v2/enhanced_features_dataset.parquet'
    analyze_enhanced_features(file_to_analyze) 