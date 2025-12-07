import pandas as pd
import numpy as np

print("=" * 80)
print("🔄 역방향 분석: 승리 케이스의 변곡 캔들 위치 찾기")
print("=" * 80)
print()

# Load data
print("Loading data...")
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

df_hlhlhl = pd.read_csv('hlhlhl_full_labeling_analysis.csv')
df_hlhlhl['ll_time'] = pd.to_datetime(df_hlhlhl['ll_time'])
df_hlhlhl['h3_breakout_time'] = pd.to_datetime(df_hlhlhl['h3_breakout_time'])

print(f"Total opportunities: {len(df_hlhlhl)}")
print()

# Filter by success criteria
print("=" * 80)
print("1단계: 승리 케이스 필터링")
print("=" * 80)
print()

# Success = H1 reached (최종 목표 도달)
success_cases = df_hlhlhl[
    (df_hlhlhl['h3_reached'] == True) & 
    (df_hlhlhl['h2_reached'] == True) & 
    (df_hlhlhl['h1_reached'] == True)
].copy()

# Failure = H3 breakout but didn't reach H1
failure_cases = df_hlhlhl[
    (df_hlhlhl['h3_reached'] == True) & 
    (df_hlhlhl['h1_reached'] == False)
].copy()

print(f"✅ 승리 케이스 (H1 도달): {len(success_cases)}개 ({len(success_cases)/len(df_hlhlhl)*100:.1f}%)")
print(f"❌ 실패 케이스 (H3 돌파했지만 H1 실패): {len(failure_cases)}개")
print()

# Analyze candle patterns for success cases
print("=" * 80)
print("2단계: 승리 케이스의 변곡 캔들 패턴 분석")
print("=" * 80)
print()

confirmation_patterns = []

for idx, case in success_cases.iterrows():
    h3_breakout_time = case['h3_breakout_time']
    h3_price = case['h3_price']
    
    if pd.isna(h3_breakout_time):
        continue
    
    # Get candles after H3 breakout
    breakout_idx = df_15m[df_15m['datetime'] >= h3_breakout_time].index[0]
    
    # Look at next 20 candles after breakout
    candles_after = df_15m.loc[breakout_idx:breakout_idx+20].copy()
    
    if len(candles_after) < 5:
        continue
    
    # Find retest candle (price comes back to H3)
    retest_candle_idx = None
    retest_distance = None
    
    for i, (cidx, candle) in enumerate(candles_after.iterrows()):
        if i == 0:  # Skip breakout candle itself
            continue
        
        # Check if candle retests H3 (within ±2%)
        distance_to_h3 = abs(candle['low'] - h3_price) / h3_price * 100
        
        if distance_to_h3 <= 2.0:  # Within 2% of H3
            retest_candle_idx = i
            retest_distance = (candle['low'] - h3_price) / h3_price * 100
            break
    
    # If no retest found, skip
    if retest_candle_idx is None:
        continue
    
    # Check confirmation candles after retest
    retest_candle = candles_after.iloc[retest_candle_idx]
    confirmation_candles = candles_after.iloc[retest_candle_idx+1:retest_candle_idx+6]
    
    if len(confirmation_candles) == 0:
        continue
    
    # Count candles that stay above H3
    candles_above_h3 = 0
    candles_below_h3 = 0
    first_candle_above = None
    consecutive_above = 0
    max_consecutive = 0
    
    for i, (cidx, candle) in enumerate(confirmation_candles.iterrows()):
        if candle['close'] > h3_price:
            candles_above_h3 += 1
            consecutive_above += 1
            if first_candle_above is None:
                first_candle_above = i + 1  # 1-indexed
            max_consecutive = max(max_consecutive, consecutive_above)
        else:
            candles_below_h3 += 1
            consecutive_above = 0
    
    # Check if it's a bullish candle at retest
    retest_bullish = retest_candle['close'] > retest_candle['open']
    
    # Check first confirmation candle
    first_conf_candle = confirmation_candles.iloc[0] if len(confirmation_candles) > 0 else None
    first_conf_bullish = first_conf_candle['close'] > first_conf_candle['open'] if first_conf_candle is not None else False
    first_conf_above_h3 = first_conf_candle['close'] > h3_price if first_conf_candle is not None else False
    
    # Volume spike check (compared to previous 20 candles)
    volume_before = df_15m.loc[breakout_idx-20:breakout_idx-1]['volume'].mean()
    volume_at_retest = retest_candle['volume']
    volume_ratio = volume_at_retest / volume_before if volume_before > 0 else 1.0
    
    confirmation_patterns.append({
        'll_time': case['ll_time'],
        'h3_breakout_time': h3_breakout_time,
        'h3_price': h3_price,
        'retest_candle_idx': retest_candle_idx,
        'retest_distance_pct': retest_distance,
        'retest_bullish': retest_bullish,
        'first_conf_bullish': first_conf_bullish,
        'first_conf_above_h3': first_conf_above_h3,
        'candles_above_h3_next_5': candles_above_h3,
        'candles_below_h3_next_5': candles_below_h3,
        'max_consecutive_above': max_consecutive,
        'volume_ratio': volume_ratio,
        'power_score': case['power_score'],
        'result': 'SUCCESS'
    })

df_success_patterns = pd.DataFrame(confirmation_patterns)

print(f"✅ 분석 완료: {len(df_success_patterns)}개 승리 케이스")
print()

# Analyze failure cases
print("=" * 80)
print("3단계: 실패 케이스의 변곡 캔들 패턴 분석")
print("=" * 80)
print()

failure_patterns = []

for idx, case in failure_cases.head(len(success_cases)).iterrows():  # Same number as success
    h3_breakout_time = case['h3_breakout_time']
    h3_price = case['h3_price']
    
    if pd.isna(h3_breakout_time):
        continue
    
    breakout_idx = df_15m[df_15m['datetime'] >= h3_breakout_time].index[0]
    candles_after = df_15m.loc[breakout_idx:breakout_idx+20].copy()
    
    if len(candles_after) < 5:
        continue
    
    retest_candle_idx = None
    retest_distance = None
    
    for i, (cidx, candle) in enumerate(candles_after.iterrows()):
        if i == 0:
            continue
        
        distance_to_h3 = abs(candle['low'] - h3_price) / h3_price * 100
        
        if distance_to_h3 <= 2.0:
            retest_candle_idx = i
            retest_distance = (candle['low'] - h3_price) / h3_price * 100
            break
    
    if retest_candle_idx is None:
        continue
    
    retest_candle = candles_after.iloc[retest_candle_idx]
    confirmation_candles = candles_after.iloc[retest_candle_idx+1:retest_candle_idx+6]
    
    if len(confirmation_candles) == 0:
        continue
    
    candles_above_h3 = 0
    candles_below_h3 = 0
    first_candle_above = None
    consecutive_above = 0
    max_consecutive = 0
    
    for i, (cidx, candle) in enumerate(confirmation_candles.iterrows()):
        if candle['close'] > h3_price:
            candles_above_h3 += 1
            consecutive_above += 1
            if first_candle_above is None:
                first_candle_above = i + 1
            max_consecutive = max(max_consecutive, consecutive_above)
        else:
            candles_below_h3 += 1
            consecutive_above = 0
    
    retest_bullish = retest_candle['close'] > retest_candle['open']
    
    first_conf_candle = confirmation_candles.iloc[0] if len(confirmation_candles) > 0 else None
    first_conf_bullish = first_conf_candle['close'] > first_conf_candle['open'] if first_conf_candle is not None else False
    first_conf_above_h3 = first_conf_candle['close'] > h3_price if first_conf_candle is not None else False
    
    volume_before = df_15m.loc[breakout_idx-20:breakout_idx-1]['volume'].mean()
    volume_at_retest = retest_candle['volume']
    volume_ratio = volume_at_retest / volume_before if volume_before > 0 else 1.0
    
    failure_patterns.append({
        'll_time': case['ll_time'],
        'h3_breakout_time': h3_breakout_time,
        'h3_price': h3_price,
        'retest_candle_idx': retest_candle_idx,
        'retest_distance_pct': retest_distance,
        'retest_bullish': retest_bullish,
        'first_conf_bullish': first_conf_bullish,
        'first_conf_above_h3': first_conf_above_h3,
        'candles_above_h3_next_5': candles_above_h3,
        'candles_below_h3_next_5': candles_below_h3,
        'max_consecutive_above': max_consecutive,
        'volume_ratio': volume_ratio,
        'power_score': case['power_score'],
        'result': 'FAILURE'
    })

df_failure_patterns = pd.DataFrame(failure_patterns)

print(f"❌ 분석 완료: {len(df_failure_patterns)}개 실패 케이스")
print()

# Compare patterns
print("=" * 80)
print("4단계: 승리 vs 실패 패턴 비교")
print("=" * 80)
print()

if len(df_success_patterns) > 0 and len(df_failure_patterns) > 0:
    print("📊 통계 비교:")
    print()
    
    metrics = [
        ('리테스트 캔들 위치 (평균)', 'retest_candle_idx'),
        ('리테스트 거리 % (평균)', 'retest_distance_pct'),
        ('리테스트 캔들 양봉 비율 %', 'retest_bullish'),
        ('첫 확정 캔들 양봉 비율 %', 'first_conf_bullish'),
        ('첫 확정 캔들 H3 위 비율 %', 'first_conf_above_h3'),
        ('다음 5캔들 중 H3 위 개수 (평균)', 'candles_above_h3_next_5'),
        ('최대 연속 H3 위 개수 (평균)', 'max_consecutive_above'),
        ('볼륨 비율 (평균)', 'volume_ratio'),
        ('Power Score (평균)', 'power_score')
    ]
    
    for label, col in metrics:
        if col in ['retest_bullish', 'first_conf_bullish', 'first_conf_above_h3']:
            success_val = df_success_patterns[col].sum() / len(df_success_patterns) * 100
            failure_val = df_failure_patterns[col].sum() / len(df_failure_patterns) * 100
            print(f"{label}:")
            print(f"  ✅ 승리: {success_val:.1f}%")
            print(f"  ❌ 실패: {failure_val:.1f}%")
            print(f"  차이: {success_val - failure_val:+.1f}%p")
        else:
            success_val = df_success_patterns[col].mean()
            failure_val = df_failure_patterns[col].mean()
            print(f"{label}:")
            print(f"  ✅ 승리: {success_val:.2f}")
            print(f"  ❌ 실패: {failure_val:.2f}")
            print(f"  차이: {success_val - failure_val:+.2f}")
        print()

# Find "confirmation space" rules
print("=" * 80)
print("5단계: 확정 공간 규칙 도출")
print("=" * 80)
print()

if len(df_success_patterns) > 0:
    # Rule 1: Retest timing
    retest_timing_success = df_success_patterns['retest_candle_idx'].median()
    print(f"📍 규칙 1: 리테스트 타이밍")
    print(f"   - 승리 케이스 중앙값: H3 돌파 후 {retest_timing_success:.0f}캔들 이내")
    print()
    
    # Rule 2: First confirmation candle
    first_conf_above_rate = df_success_patterns['first_conf_above_h3'].sum() / len(df_success_patterns) * 100
    first_conf_bullish_rate = df_success_patterns['first_conf_bullish'].sum() / len(df_success_patterns) * 100
    print(f"📍 규칙 2: 첫 확정 캔들")
    print(f"   - H3 위에서 종가: {first_conf_above_rate:.1f}%")
    print(f"   - 양봉: {first_conf_bullish_rate:.1f}%")
    print()
    
    # Rule 3: Consecutive candles above H3
    max_consec_success = df_success_patterns['max_consecutive_above'].median()
    print(f"📍 규칙 3: 연속 확정")
    print(f"   - 최소 {max_consec_success:.0f}캔들 연속 H3 위 유지")
    print()
    
    # Rule 4: 5-candle test
    candles_above_5_success = df_success_patterns['candles_above_h3_next_5'].median()
    print(f"📍 규칙 4: 5캔들 테스트")
    print(f"   - 다음 5캔들 중 최소 {candles_above_5_success:.0f}개 H3 위")
    print()
    
    # Rule 5: Volume
    volume_ratio_success = df_success_patterns['volume_ratio'].median()
    print(f"📍 규칙 5: 볼륨")
    print(f"   - 리테스트 시 볼륨: 평균 대비 {volume_ratio_success:.2f}배")
    print()

# Create final rule set
print("=" * 80)
print("🎯 최종 확정 공간 규칙")
print("=" * 80)
print()

if len(df_success_patterns) > 0:
    print("✅ 진입 확정 조건:")
    print()
    print("1. H3 돌파 확인")
    print(f"2. {retest_timing_success:.0f}캔들 이내 H3 리테스트 (±2%)")
    print(f"3. 리테스트 후 첫 캔들:")
    print(f"   - H3 위에서 종가 ({first_conf_above_rate:.0f}% 필수)")
    print(f"   - 양봉 선호 ({first_conf_bullish_rate:.0f}%)")
    print(f"4. 다음 5캔들 중 최소 {candles_above_5_success:.0f}개 H3 위")
    print(f"5. 또는 최소 {max_consec_success:.0f}캔들 연속 H3 위")
    print()
    print("이 조건을 만족하면 → **확정 공간 진입** → 롱 진입 ✅")
    print()

# Save results
df_all_patterns = pd.concat([df_success_patterns, df_failure_patterns])
df_all_patterns.to_csv('confirmation_space_analysis.csv', index=False)
print("✅ 결과 저장: confirmation_space_analysis.csv")

