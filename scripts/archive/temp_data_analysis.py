import pandas as pd
import numpy as np

# 강화된 피처 데이터셋 로드 및 분석
df = pd.read_parquet('results/ml_analysis_v2/enhanced_features_dataset.parquet')
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

print("=== 🎯 강화된 피처 데이터셋 현황 ===")
print(f"📊 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")

if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
    print(f"📅 기간: {df.index.min()} ~ {df.index.max()}")
    delta = df.index.max() - df.index.min()
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
null_counts = df.isnull().sum()
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