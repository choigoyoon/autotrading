"""
Phase 1 MTF: L/H 라벨링
- 1시간, 4시간, 1일 데이터
- MACD 히스토그램 크로스오버 기반
- L: 음수→양수 전환
- H: 양수→음수 전환
- 나우캐스트: 1봉 딜레이
"""

import pandas as pd
import numpy as np

def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD 계산"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()

    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return macd, signal_line, histogram

def label_hl_from_macd(df):
    """MACD 히스토그램 크로스오버로 L/H 라벨링"""

    df = df.copy()

    # MACD 계산
    macd, signal, histogram = calculate_macd(df)

    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = histogram

    # 크로스오버 감지
    df['label'] = None

    for i in range(1, len(df)):
        prev_hist = df.iloc[i-1]['macd_hist']
        curr_hist = df.iloc[i]['macd_hist']

        # 음수 → 양수: L (저점)
        if prev_hist < 0 and curr_hist >= 0:
            df.iloc[i-1, df.columns.get_loc('label')] = 'L'

        # 양수 → 음수: H (고점)
        elif prev_hist > 0 and curr_hist <= 0:
            df.iloc[i-1, df.columns.get_loc('label')] = 'H'

    return df

def process_timeframe(filename, timeframe_name):
    """타임프레임별 처리"""

    print(f"\n{'='*60}")
    print(f"{timeframe_name} 라벨링")
    print(f"{'='*60}")

    # 데이터 로드
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n원본 데이터: {len(df):,}개")
    print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # L/H 라벨링
    df_labeled = label_hl_from_macd(df)

    # 통계
    l_count = (df_labeled['label'] == 'L').sum()
    h_count = (df_labeled['label'] == 'H').sum()

    print(f"\nL/H 라벨:")
    print(f"  L (저점): {l_count:,}개")
    print(f"  H (고점): {h_count:,}개")
    print(f"  총 라벨: {l_count + h_count:,}개")
    print(f"  라벨 비율: {(l_count + h_count) / len(df) * 100:.2f}%")

    # MACD 통계
    print(f"\nMACD 통계:")
    print(f"  평균: {df_labeled['macd_hist'].mean():.4f}")
    print(f"  표준편차: {df_labeled['macd_hist'].std():.4f}")
    print(f"  최대: {df_labeled['macd_hist'].max():.4f}")
    print(f"  최소: {df_labeled['macd_hist'].min():.4f}")

    # 샘플 출력
    print(f"\nL 샘플:")
    l_samples = df_labeled[df_labeled['label'] == 'L'].head(3)
    for idx, row in l_samples.iterrows():
        print(f"  {row['datetime']}: close={row['close']:.2f}, macd_hist={row['macd_hist']:.4f}")

    print(f"\nH 샘플:")
    h_samples = df_labeled[df_labeled['label'] == 'H'].head(3)
    for idx, row in h_samples.iterrows():
        print(f"  {row['datetime']}: close={row['close']:.2f}, macd_hist={row['macd_hist']:.4f}")

    # 저장
    output_file = filename.replace('_raw.csv', '_labeled.csv')
    df_labeled.to_csv(output_file, index=False)
    print(f"\n저장: {output_file}")

    return df_labeled

# 메인 실행
print("="*60)
print("MTF L/H 라벨링 시작")
print("="*60)

timeframes = [
    ('btcusdt_1h_raw.csv', '1시간'),
    ('btcusdt_4h_raw.csv', '4시간'),
    ('btcusdt_1d_raw.csv', '1일'),
]

results = {}

for filename, name in timeframes:
    try:
        df = process_timeframe(filename, name)
        results[name] = df
        print(f"\n✅ {name} 완료!")
    except Exception as e:
        print(f"\n❌ {name} 실패: {e}")
        import traceback
        traceback.print_exc()

# 전체 요약
print("\n" + "="*60)
print("전체 요약")
print("="*60)

for name, df in results.items():
    l_count = (df['label'] == 'L').sum()
    h_count = (df['label'] == 'H').sum()
    print(f"\n{name}:")
    print(f"  총 개수: {len(df):,}개")
    print(f"  L: {l_count:,}개")
    print(f"  H: {h_count:,}개")
    print(f"  L+H: {l_count + h_count:,}개 ({(l_count + h_count) / len(df) * 100:.1f}%)")

print("\n생성된 파일:")
print("  - btcusdt_1h_labeled.csv")
print("  - btcusdt_4h_labeled.csv")
print("  - btcusdt_1d_labeled.csv")

print("\n다음 단계:")
print("  1. 상황 분류 시스템 (1D/4H/1H MACD 조합 → A~H)")
print("  2. MTF Zone 추출 (L/H 꼬리 범위)")
print("  3. 15분 진입 예측 시스템")
print("  4. 통합 백테스트")

print("\n완료!")
