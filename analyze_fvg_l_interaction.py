"""
FVG + L 라벨 상호작용 분석
- FVG: Fair Value Gap (가격 공백)
- L 이후 FVG 존재 시: 상승 vs 지지 효과 비교
"""

import pandas as pd
import numpy as np

print("="*60)
print("FVG + L 라벨 상호작용 분석")
print("="*60)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n총 데이터: {len(df):,}개")

# 1. FVG 감지 함수
def detect_fvg(df):
    """
    FVG (Fair Value Gap) 감지

    Bullish FVG:
      - Candle 1의 high < Candle 3의 low
      - Candle 2가 갭 생성
      - 미충전 영역 = 지지 가능

    Bearish FVG:
      - Candle 1의 low > Candle 3의 high
      - Candle 2가 갭 생성
      - 미충전 영역 = 저항 가능
    """

    fvg_list = []

    for i in range(2, len(df)):
        candle_1 = df.iloc[i-2]
        candle_2 = df.iloc[i-1]
        candle_3 = df.iloc[i]

        # Bullish FVG
        if candle_1['high'] < candle_3['low']:
            gap_bottom = candle_1['high']
            gap_top = candle_3['low']
            gap_size = (gap_top - gap_bottom) / candle_3['close'] * 100

            fvg_list.append({
                'idx': i-1,  # FVG는 Candle 2 위치
                'datetime': candle_2['datetime'],
                'type': 'bullish',
                'gap_bottom': gap_bottom,
                'gap_top': gap_top,
                'gap_size': gap_size,
                'filled': False,  # 나중에 확인
            })

        # Bearish FVG
        elif candle_1['low'] > candle_3['high']:
            gap_bottom = candle_3['high']
            gap_top = candle_1['low']
            gap_size = (gap_top - gap_bottom) / candle_3['close'] * 100

            fvg_list.append({
                'idx': i-1,
                'datetime': candle_2['datetime'],
                'type': 'bearish',
                'gap_bottom': gap_bottom,
                'gap_top': gap_top,
                'gap_size': gap_size,
                'filled': False,
            })

    return pd.DataFrame(fvg_list)

print("\nFVG 감지 중...")
fvg_df = detect_fvg(df)

print(f"총 FVG: {len(fvg_df):,}개")
print(f"  Bullish FVG: {len(fvg_df[fvg_df['type'] == 'bullish']):,}개")
print(f"  Bearish FVG: {len(fvg_df[fvg_df['type'] == 'bearish']):,}개")
print(f"  평균 갭 크기: {fvg_df['gap_size'].mean():.3f}%")

# 2. L 라벨 + FVG 매칭
print("\n" + "="*60)
print("L 라벨 + FVG 매칭")
print("="*60)

labeled = df[df['label'].notna()].copy()
l_labels = labeled[labeled['label'] == 'L'].copy()

print(f"\nL 라벨: {len(l_labels)}개")

# L 이후 가까운 Bullish FVG 찾기
def find_nearby_fvg(l_idx, l_datetime, fvg_df, max_distance_bars=20):
    """
    L 라벨 근처 FVG 찾기
    - max_distance_bars: L 이후 N봉 이내
    """

    # L 이후 발생한 FVG만
    future_fvgs = fvg_df[
        (fvg_df['idx'] > l_idx) &
        (fvg_df['idx'] <= l_idx + max_distance_bars) &
        (fvg_df['type'] == 'bullish')  # L 이후 Bullish FVG
    ]

    if len(future_fvgs) == 0:
        return None

    # 가장 가까운 FVG
    nearest = future_fvgs.iloc[0]

    return {
        'fvg_idx': nearest['idx'],
        'fvg_datetime': nearest['datetime'],
        'distance_bars': nearest['idx'] - l_idx,
        'gap_size': nearest['gap_size'],
        'gap_bottom': nearest['gap_bottom'],
        'gap_top': nearest['gap_top'],
    }

l_with_fvg = []

for idx, l_row in l_labels.iterrows():
    l_idx = l_row.name
    l_datetime = l_row['datetime']
    l_price = l_row['label_price']

    # 근처 FVG 찾기
    fvg_info = find_nearby_fvg(l_idx, l_datetime, fvg_df, max_distance_bars=20)

    if fvg_info:
        l_with_fvg.append({
            'l_idx': l_idx,
            'l_datetime': l_datetime,
            'l_price': l_price,
            'has_fvg': True,
            **fvg_info
        })
    else:
        l_with_fvg.append({
            'l_idx': l_idx,
            'l_datetime': l_datetime,
            'l_price': l_price,
            'has_fvg': False,
        })

l_with_fvg_df = pd.DataFrame(l_with_fvg)

print(f"\nL + FVG: {l_with_fvg_df['has_fvg'].sum()}개 ({l_with_fvg_df['has_fvg'].sum() / len(l_with_fvg_df) * 100:.1f}%)")
print(f"L only: {(~l_with_fvg_df['has_fvg']).sum()}개")

# 3. 이후 가격 움직임 분석
print("\n" + "="*60)
print("L + FVG vs L only 비교")
print("="*60)

def analyze_price_movement(l_idx, l_price, df, lookforward=50):
    """
    L 이후 가격 움직임 분석
    """

    max_idx = min(l_idx + lookforward, len(df) - 1)
    future = df.iloc[l_idx:max_idx+1]

    if len(future) < 2:
        return None

    # 1. 최대 상승
    max_high = future['high'].max()
    max_gain = (max_high - l_price) / l_price * 100

    # 2. 최대 하락 (지지 테스트)
    min_low = future['low'].min()
    max_drop = (min_low - l_price) / l_price * 100

    # 3. 지지 횟수 (L 가격 ±0.5% 터치 후 반등)
    support_tests = 0
    for i in range(1, len(future)):
        if future.iloc[i]['low'] <= l_price * 1.005:  # L 근처 터치
            if future.iloc[i]['close'] > future.iloc[i]['open']:  # 반등
                support_tests += 1

    # 4. 첫 상승 속도 (10봉 내 최고점)
    first_10 = future.iloc[:min(10, len(future))]
    first_10_high = first_10['high'].max()
    early_gain = (first_10_high - l_price) / l_price * 100

    return {
        'max_gain': max_gain,
        'max_drop': max_drop,
        'support_tests': support_tests,
        'early_gain': early_gain,
    }

print("\n분석 중...")

results_with_fvg = []
results_without_fvg = []

for idx, row in l_with_fvg_df.iterrows():
    analysis = analyze_price_movement(row['l_idx'], row['l_price'], df, lookforward=50)

    if analysis:
        if row['has_fvg']:
            results_with_fvg.append(analysis)
        else:
            results_without_fvg.append(analysis)

with_fvg_df = pd.DataFrame(results_with_fvg)
without_fvg_df = pd.DataFrame(results_without_fvg)

print(f"\nL + FVG: {len(with_fvg_df)}개 분석")
print(f"L only: {len(without_fvg_df)}개 분석")

# 4. 결과 비교
print("\n" + "="*60)
print("📊 결과 비교")
print("="*60)

comparison = pd.DataFrame({
    '지표': ['최대 상승', '초기 상승 (10봉)', '최대 하락', '지지 테스트 횟수'],
    'L + FVG': [
        f"{with_fvg_df['max_gain'].mean():.2f}%",
        f"{with_fvg_df['early_gain'].mean():.2f}%",
        f"{with_fvg_df['max_drop'].mean():.2f}%",
        f"{with_fvg_df['support_tests'].mean():.2f}회",
    ],
    'L only': [
        f"{without_fvg_df['max_gain'].mean():.2f}%",
        f"{without_fvg_df['early_gain'].mean():.2f}%",
        f"{without_fvg_df['max_drop'].mean():.2f}%",
        f"{without_fvg_df['support_tests'].mean():.2f}회",
    ],
    '차이': [
        f"{with_fvg_df['max_gain'].mean() - without_fvg_df['max_gain'].mean():+.2f}%",
        f"{with_fvg_df['early_gain'].mean() - without_fvg_df['early_gain'].mean():+.2f}%",
        f"{with_fvg_df['max_drop'].mean() - without_fvg_df['max_drop'].mean():+.2f}%",
        f"{with_fvg_df['support_tests'].mean() - without_fvg_df['support_tests'].mean():+.2f}회",
    ],
})

print("\n" + comparison.to_string(index=False))

# 5. 해석
print("\n" + "="*60)
print("💡 해석")
print("="*60)

max_gain_diff = with_fvg_df['max_gain'].mean() - without_fvg_df['max_gain'].mean()
support_diff = with_fvg_df['support_tests'].mean() - without_fvg_df['support_tests'].mean()
early_gain_diff = with_fvg_df['early_gain'].mean() - without_fvg_df['early_gain'].mean()

print(f"\n1. 상승 효과:")
if max_gain_diff > 0.5:
    print(f"   ✅ L + FVG가 {max_gain_diff:.2f}% 더 강한 상승")
    print(f"   → FVG가 상승 모멘텀 강화!")
elif max_gain_diff < -0.5:
    print(f"   ⚠️ L only가 {abs(max_gain_diff):.2f}% 더 강한 상승")
else:
    print(f"   ≈ 차이 미미 ({max_gain_diff:.2f}%)")

print(f"\n2. 초기 상승 속도:")
if early_gain_diff > 0.3:
    print(f"   ✅ L + FVG가 {early_gain_diff:.2f}% 더 빠른 상승")
    print(f"   → FVG가 초기 모멘텀 제공!")
else:
    print(f"   ≈ 차이 미미 ({early_gain_diff:.2f}%)")

print(f"\n3. 지지 효과:")
if support_diff > 0.2:
    print(f"   ✅ L + FVG가 {support_diff:.2f}회 더 많은 지지")
    print(f"   → FVG가 지지선 강화!")
elif support_diff < -0.2:
    print(f"   ⚠️ L only가 {abs(support_diff):.2f}회 더 많은 지지")
else:
    print(f"   ≈ 차이 미미 ({support_diff:.2f}회)")

# 6. 결론
print("\n" + "="*60)
print("🎯 결론")
print("="*60)

if max_gain_diff > 0.5 and support_diff > 0.2:
    print("\n✅ L + FVG는 '상승'과 '지지' 모두 강함!")
    print("   → FVG는 L의 효과를 증폭시키는 촉매")
    print("   → 진입 신뢰도 높음")
elif max_gain_diff > 0.5:
    print("\n✅ L + FVG는 '상승' 효과가 더 강함!")
    print("   → FVG가 상승 모멘텀 제공")
    print("   → 공격적 진입 유리")
elif support_diff > 0.2:
    print("\n✅ L + FVG는 '지지' 효과가 더 강함!")
    print("   → FVG가 안정적 바닥 형성")
    print("   → 보수적 진입 유리")
else:
    print("\n≈ FVG 효과 미미")
    print("   → L 단독으로 충분")

# 7. 실전 활용
print("\n" + "="*60)
print("💼 실전 활용")
print("="*60)

print(f"""
1. L + FVG 발견 시:
   - FVG 바닥 근처 진입 (추가 지지)
   - TP: {with_fvg_df['max_gain'].mean():.1f}% (평균 상승)
   - SL: L 아래 -1.0% (FVG 이탈)

2. L only:
   - L 가격 진입
   - TP: {without_fvg_df['max_gain'].mean():.1f}%
   - SL: -1.5%

3. FVG 크기별:
   - 큰 FVG (>0.3%): 더 강한 효과
   - 작은 FVG (<0.1%): 효과 미미
""")

# FVG 크기별 분석
if len(with_fvg_df) > 0:
    l_with_fvg_analyzed = l_with_fvg_df[l_with_fvg_df['has_fvg']].copy()
    l_with_fvg_analyzed['result'] = [r['max_gain'] for r in results_with_fvg]

    large_fvg = l_with_fvg_analyzed[l_with_fvg_analyzed['gap_size'] > 0.3]
    small_fvg = l_with_fvg_analyzed[l_with_fvg_analyzed['gap_size'] <= 0.3]

    if len(large_fvg) > 0 and len(small_fvg) > 0:
        print(f"\n4. FVG 크기별 효과:")
        print(f"   큰 FVG (>0.3%): 평균 상승 {large_fvg['result'].mean():.2f}%")
        print(f"   작은 FVG (≤0.3%): 평균 상승 {small_fvg['result'].mean():.2f}%")
        print(f"   차이: {large_fvg['result'].mean() - small_fvg['result'].mean():+.2f}%")

print("\n분석 완료!")
