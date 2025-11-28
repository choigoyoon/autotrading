"""
15분 데이터를 리샘플링하여 MTF 데이터 생성
- 15분 → 1시간 (4개 합침)
- 15분 → 4시간 (16개 합침)
- 15분 → 1일 (96개 합침)
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("15분 데이터 → MTF 리샘플링")
print("=" * 60)

# 15분 데이터 로드
print("\n15분 데이터 로드 중...")
df_15m = pd.read_csv('output_phase1_labeled.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.set_index('datetime')

print(f"15분 데이터: {len(df_15m):,}개")
print(f"기간: {df_15m.index.min()} ~ {df_15m.index.max()}")

def resample_ohlcv(df, timeframe, name):
    """OHLCV 리샘플링"""

    print(f"\n{'-' * 60}")
    print(f"{name} 리샘플링 시작")
    print(f"{'-' * 60}")

    # 리샘플링
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # 결측치 제거
    resampled = resampled.dropna()

    print(f"  생성: {len(resampled):,}개")
    print(f"  기간: {resampled.index.min()} ~ {resampled.index.max()}")

    # 비율 확인
    ratio = len(df_15m) / len(resampled)
    print(f"  비율: {ratio:.1f}배")

    # datetime 컬럼 추가
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'index': 'datetime'})

    return resampled

# 각 타임프레임 생성
timeframes = {
    '1H': ('1시간', 'btcusdt_1h_raw.csv'),
    '4H': ('4시간', 'btcusdt_4h_raw.csv'),
    '1D': ('1일', 'btcusdt_1d_raw.csv'),
}

results = {}

for tf, (name, filename) in timeframes.items():
    df = resample_ohlcv(df_15m, tf, name)
    results[tf] = df

    # 저장
    df.to_csv(filename, index=False)
    print(f"  저장: {filename}")

# 요약
print("\n" + "=" * 60)
print("리샘플링 완료")
print("=" * 60)

print(f"\n15분 데이터: {len(df_15m):,}개")
for tf in ['1H', '4H', '1D']:
    if tf in results:
        print(f"{timeframes[tf][0]}: {len(results[tf]):,}개")

# 이론적 비율 vs 실제
print(f"\n비율 확인 (15분 기준):")
expected = {'1H': 4, '4H': 16, '1D': 96}
for tf in ['1H', '4H', '1D']:
    if tf in results:
        actual = len(df_15m) / len(results[tf])
        exp = expected[tf]
        print(f"  {timeframes[tf][0]}: {actual:.1f}배 (이론: {exp}배)")

# 데이터 샘플
print(f"\n1시간 데이터 샘플:")
print(results['1H'].head())

print("\n생성된 파일:")
for tf, (name, filename) in timeframes.items():
    print(f"  - {filename}")

print("\n다음 단계:")
print("  1. MTF L/H 라벨링")
print("  2. MTF MACD 계산")
print("  3. 상황 분류 (A~H)")
print("  4. Zone 추출")
