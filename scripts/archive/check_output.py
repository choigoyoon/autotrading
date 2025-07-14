import pandas as pd

output_file = "data/processed/btc_usdt_kst/labeled/divergence_confirmed_labels.parquet"

try:
    df = pd.read_parquet(output_file)
    print(f"✅ 파일 로드 성공: {output_file}")
    print("\n--- 기본 정보 ---")
    df.info()
    print("\n--- 상위 5개 샘플 ---")
    print(df.head())
    print("\n--- 라벨 분포 ---")
    print(df['label'].value_counts())
    print("\n--- 타임프레임별 라벨 분포 ---")
    print(df.groupby('timeframe')['label'].value_counts().unstack(fill_value=0))
    
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {output_file}")
except Exception as e:
    print(f"오류 발생: {e}") 