import pandas as pd
import numpy as np

# Load all L and H values
all_l = pd.read_csv('all_L_values_inference.csv')
all_l['datetime'] = pd.to_datetime(all_l['datetime'])
all_l = all_l.sort_values('datetime').reset_index(drop=True)

# Need to infer H values similarly
ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])

print("=" * 100)
print("HLHLHLHL 패턴 분석: 하락 가속/감속 체크")
print("=" * 100)

# Infer H values (Swing Highs) - mirror of L values
all_h = []
for i in range(10, len(ohlcv) - 10):
    current_high = ohlcv.iloc[i]['high']
    
    # Check if this is a swing high (higher than 10 bars left and right)
    is_swing_high = True
    for j in range(i - 10, i):
        if ohlcv.iloc[j]['high'] >= current_high:
            is_swing_high = False
            break
    
    if is_swing_high:
        for j in range(i + 1, i + 11):
            if ohlcv.iloc[j]['high'] >= current_high:
                is_swing_high = False
                break
    
    if is_swing_high:
        all_h.append({
            'datetime': ohlcv.iloc[i]['datetime'],
            'h_price': current_high
        })

all_h_df = pd.DataFrame(all_h)
print(f"\n총 H값 (Swing High): {len(all_h_df):,}개")
print(f"총 L값 (Swing Low): {len(all_l):,}개")

# Load LLLL patterns
llll_patterns = pd.read_csv('LLLL_chart_patterns_analysis.csv')
llll_patterns['l4_datetime'] = pd.to_datetime(llll_patterns['l4_datetime'])

print(f"총 LLLL 패턴: {len(llll_patterns)}개")

# Analyze each LLLL pattern
results = []

for idx, pattern in llll_patterns.iterrows():
    l4_dt = pattern['l4_datetime']
    l4_price = pattern['l4_price']
    l3_price = pattern['l3_price']
    l2_price = pattern['l2_price']
    l1_price = pattern['l1_price']
    
    # Calculate LL drops
    l1_l2_drop = ((l2_price - l1_price) / l1_price) * 100
    l2_l3_drop = ((l3_price - l2_price) / l2_price) * 100
    l3_l4_drop = ((l4_price - l3_price) / l3_price) * 100
    
    # Check acceleration/deceleration
    # More negative = bigger drop
    drop1 = abs(l1_l2_drop)
    drop2 = abs(l2_l3_drop)
    drop3 = abs(l3_l4_drop)
    
    # Pattern classification
    if drop3 < drop2 < drop1:
        acceleration_pattern = "Deceleration"  # 감속 (좋은 신호)
    elif drop3 > drop2 > drop1:
        acceleration_pattern = "Acceleration"  # 가속 (나쁜 신호)
    elif drop3 < drop2 and drop2 > drop1:
        acceleration_pattern = "Peak-Middle"  # 중간이 제일 큼
    elif drop3 > drop2 and drop2 < drop1:
        acceleration_pattern = "Valley-Middle"  # 중간이 제일 작음
    else:
        acceleration_pattern = "Mixed"
    
    # Get next L value
    l4_idx = all_l[all_l['datetime'] == l4_dt].index
    if len(l4_idx) > 0 and l4_idx[0] + 1 < len(all_l):
        next_l = all_l.iloc[l4_idx[0] + 1]
        next_l_price = next_l['l_price']
        next_l_higher = next_l_price > l4_price
        l4_to_l5_change = ((next_l_price - l4_price) / l4_price) * 100
    else:
        next_l_higher = None
        l4_to_l5_change = None
    
    results.append({
        'l4_datetime': l4_dt,
        'l1_price': l1_price,
        'l2_price': l2_price,
        'l3_price': l3_price,
        'l4_price': l4_price,
        'l1_l2_drop_pct': l1_l2_drop,
        'l2_l3_drop_pct': l2_l3_drop,
        'l3_l4_drop_pct': l3_l4_drop,
        'drop1_abs': drop1,
        'drop2_abs': drop2,
        'drop3_abs': drop3,
        'acceleration_pattern': acceleration_pattern,
        'next_l_higher': next_l_higher,
        'l4_to_l5_change': l4_to_l5_change,
        'total_drop_pct': pattern['total_drop_pct'],
        'l4_rsi': pattern['l4_rsi'],
        'conditions_met': (
            (1 if pattern['l4_rsi'] < 30 else 0) +
            (1 if pattern['l4_macd_hist'] < -50 else 0) +
            (1 if pattern.get('l4_bb_position', 0) < 0.1 else 0) +
            (1 if pattern.get('l4_volume_ratio', 0) > 3.0 else 0) +
            (1 if pattern.get('l4_atr_pct', 0) > 0.5 else 0)
        )
    })

results_df = pd.DataFrame(results)

print("\n" + "="*100)
print("하락 패턴 분류")
print("="*100)

for pattern_type in ['Deceleration', 'Acceleration', 'Peak-Middle', 'Valley-Middle', 'Mixed']:
    subset = results_df[results_df['acceleration_pattern'] == pattern_type]
    if len(subset) > 0:
        validated = subset[subset['next_l_higher'].notna()]
        if len(validated) > 0:
            success_count = validated[validated['next_l_higher'] == True].shape[0]
            success_rate = (success_count / len(validated)) * 100
            avg_change = validated[validated['next_l_higher'] == True]['l4_to_l5_change'].mean() if success_count > 0 else 0
            
            print(f"\n{pattern_type}: {len(subset)}건")
            print(f"  진입 성공률 (L5 > L4): {success_rate:.1f}% ({success_count}/{len(validated)})")
            if success_count > 0:
                print(f"  성공시 평균 저점 상승: {avg_change:+.2f}%")

# Detailed deceleration analysis
print("\n" + "="*100)
print("Deceleration (감속) 패턴 상세 분석")
print("="*100)

decel = results_df[results_df['acceleration_pattern'] == 'Deceleration']
if len(decel) > 0:
    print(f"\n총 케이스: {len(decel)}건")
    print(f"\n평균 하락 폭:")
    print(f"  L1→L2: {decel['drop1_abs'].mean():.2f}%")
    print(f"  L2→L3: {decel['drop2_abs'].mean():.2f}%")
    print(f"  L3→L4: {decel['drop3_abs'].mean():.2f}%")
    
    validated = decel[decel['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        print(f"\n진입 성공: {len(success)}/{len(validated)}건 ({len(success)/len(validated)*100:.1f}%)")
        if len(success) > 0:
            print(f"성공시 평균 저점 상승: {success['l4_to_l5_change'].mean():+.2f}%")

# Detailed acceleration analysis
print("\n" + "="*100)
print("Acceleration (가속) 패턴 상세 분석")
print("="*100)

accel = results_df[results_df['acceleration_pattern'] == 'Acceleration']
if len(accel) > 0:
    print(f"\n총 케이스: {len(accel)}건")
    print(f"\n평균 하락 폭:")
    print(f"  L1→L2: {accel['drop1_abs'].mean():.2f}%")
    print(f"  L2→L3: {accel['drop2_abs'].mean():.2f}%")
    print(f"  L3→L4: {accel['drop3_abs'].mean():.2f}%")
    
    validated = accel[accel['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        print(f"\n진입 성공: {len(success)}/{len(validated)}건 ({len(success)/len(validated)*100:.1f}%)")
        if len(success) > 0:
            print(f"성공시 평균 저점 상승: {success['l4_to_l5_change'].mean():+.2f}%")

# Combined analysis: Deceleration + Strong conditions
print("\n" + "="*100)
print("최적 조합: Deceleration + 4+ 조건")
print("="*100)

optimal = results_df[(results_df['acceleration_pattern'] == 'Deceleration') & 
                     (results_df['conditions_met'] >= 4)]
if len(optimal) > 0:
    print(f"\n총 케이스: {len(optimal)}건")
    validated = optimal[optimal['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        print(f"진입 성공률: {len(success)/len(validated)*100:.1f}% ({len(success)}/{len(validated)})")
        if len(success) > 0:
            print(f"성공시 평균 저점 상승: {success['l4_to_l5_change'].mean():+.2f}%")
        
        # Calculate monthly
        months = 69  # From previous analysis
        monthly_trades = len(optimal) / months
        print(f"\n예상 월간 거래: {monthly_trades:.2f}건")

# TOP 10 Deceleration success cases
print("\n" + "="*100)
print("TOP 10 Deceleration 성공 케이스")
print("="*100)

decel_success = results_df[(results_df['acceleration_pattern'] == 'Deceleration') & 
                           (results_df['next_l_higher'] == True)]
if len(decel_success) > 0:
    top10 = decel_success.nlargest(10, 'l4_to_l5_change')
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   하락폭: L1→L2: {row['drop1_abs']:.2f}% → L2→L3: {row['drop2_abs']:.2f}% → L3→L4: {row['drop3_abs']:.2f}%")
        print(f"   L4 가격: ${row['l4_price']:.2f}")
        print(f"   저점 상승: {row['l4_to_l5_change']:+.2f}%")
        print(f"   조건: {row['conditions_met']}/5")

# Save results
results_df.to_csv('hlhl_acceleration_analysis.csv', index=False)
print(f"\n\n✅ 분석 결과 저장: hlhl_acceleration_analysis.csv")

print("\n" + "="*100)
print("최종 결론")
print("="*100)

decel_validated = results_df[(results_df['acceleration_pattern'] == 'Deceleration') & 
                             (results_df['next_l_higher'].notna())]
if len(decel_validated) > 0:
    decel_success_rate = (decel_validated['next_l_higher'] == True).sum() / len(decel_validated) * 100
    
    accel_validated = results_df[(results_df['acceleration_pattern'] == 'Acceleration') & 
                                 (results_df['next_l_higher'].notna())]
    if len(accel_validated) > 0:
        accel_success_rate = (accel_validated['next_l_higher'] == True).sum() / len(accel_validated) * 100
        
        print(f"\nDeceleration (감속) 성공률: {decel_success_rate:.1f}%")
        print(f"Acceleration (가속) 성공률: {accel_success_rate:.1f}%")
        print(f"\n차이: {decel_success_rate - accel_success_rate:+.1f}%p")
        
        if decel_success_rate > accel_success_rate + 10:
            print("\n✅ Deceleration 패턴이 유의미하게 더 좋음!")
        elif decel_success_rate > accel_success_rate:
            print("\n⚠️  Deceleration 패턴이 약간 더 좋음")
        else:
            print("\n❌ 패턴 차이 없음")

print("\n" + "="*100)

