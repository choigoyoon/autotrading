import pandas as pd
import numpy as np

# Load data
llll_patterns = pd.read_csv('LLLL_chart_patterns_analysis.csv')
llll_patterns['l4_datetime'] = pd.to_datetime(llll_patterns['l4_datetime'])

all_l = pd.read_csv('all_L_values_inference.csv')
all_l['datetime'] = pd.to_datetime(all_l['datetime'])
all_l = all_l.sort_values('datetime').reset_index(drop=True)

print("=" * 100)
print("가격 하락 vs MACD 하락 비율 분석 (Divergence)")
print("=" * 100)

results = []

for idx, pattern in llll_patterns.iterrows():
    l1_price = pattern['l1_price']
    l2_price = pattern['l2_price']
    l3_price = pattern['l3_price']
    l4_price = pattern['l4_price']
    
    # Price drops
    l1_l2_price_drop = ((l2_price - l1_price) / l1_price) * 100
    l2_l3_price_drop = ((l3_price - l2_price) / l2_price) * 100
    l3_l4_price_drop = ((l4_price - l3_price) / l3_price) * 100
    
    # Need MACD at each L point - use from pattern data
    # Assuming we have MACD values at L1, L2, L3, L4
    l4_macd = pattern['l4_macd_hist']
    
    # Calculate MACD change relative to price change
    # "캔들값에 비해 MACD 감소폭"
    
    # For L3 → L4
    price_drop_abs = abs(l3_l4_price_drop)
    macd_at_l4 = abs(l4_macd)
    
    # Check if MACD is getting less negative (bullish divergence)
    # Price going down but MACD going up = divergence
    
    # Get next L for validation
    l4_dt = pattern['l4_datetime']
    l4_idx = all_l[all_l['datetime'] == l4_dt].index
    if len(l4_idx) > 0 and l4_idx[0] + 1 < len(all_l):
        next_l = all_l.iloc[l4_idx[0] + 1]
        next_l_higher = next_l['l_price'] > l4_price
        l4_to_l5_change = ((next_l['l_price'] - l4_price) / l4_price) * 100
    else:
        next_l_higher = None
        l4_to_l5_change = None
    
    # MACD depth classification
    if macd_at_l4 > 100:
        macd_depth = "Deep (>100)"
    elif macd_at_l4 > 50:
        macd_depth = "Medium (50-100)"
    else:
        macd_depth = "Shallow (<50)"
    
    # Price/MACD ratio
    if price_drop_abs > 0:
        price_to_macd_ratio = macd_at_l4 / price_drop_abs
    else:
        price_to_macd_ratio = 0
    
    results.append({
        'l4_datetime': l4_dt,
        'l4_price': l4_price,
        'total_drop_pct': pattern['total_drop_pct'],
        'l3_l4_price_drop': l3_l4_price_drop,
        'l4_macd_hist': l4_macd,
        'macd_depth': macd_depth,
        'price_to_macd_ratio': price_to_macd_ratio,
        'next_l_higher': next_l_higher,
        'l4_to_l5_change': l4_to_l5_change,
        'l4_rsi': pattern['l4_rsi']
    })

results_df = pd.DataFrame(results)

print(f"\n총 LLLL 패턴: {len(results_df)}개")

# Analyze by MACD depth
print("\n" + "="*100)
print("MACD 깊이별 성공률")
print("="*100)

for depth in ["Deep (>100)", "Medium (50-100)", "Shallow (<50)"]:
    subset = results_df[results_df['macd_depth'] == depth]
    if len(subset) > 0:
        validated = subset[subset['next_l_higher'].notna()]
        if len(validated) > 0:
            success = validated[validated['next_l_higher'] == True]
            success_rate = len(success) / len(validated) * 100
            avg_change = success['l4_to_l5_change'].mean() if len(success) > 0 else 0
            
            print(f"\n{depth}: {len(subset)}건")
            print(f"  성공률: {success_rate:.1f}% ({len(success)}/{len(validated)})")
            print(f"  평균 MACD: {subset['l4_macd_hist'].mean():.1f}")
            if len(success) > 0:
                print(f"  성공시 저점 상승: {avg_change:+.2f}%")

# Analyze by Price/MACD ratio
print("\n" + "="*100)
print("가격하락 대비 MACD 비율")
print("="*100)

print("\n설명: 비율이 높을수록 = 가격 하락에 비해 MACD가 깊음 = 강한 과매도")
print("     비율이 낮을수록 = 가격 하락에 비해 MACD가 얕음 = Divergence 가능성")

ratio_ranges = [
    ("매우 높음 (>50)", 50, 1000),
    ("높음 (30-50)", 30, 50),
    ("중간 (15-30)", 15, 30),
    ("낮음 (5-15)", 5, 15),
    ("매우 낮음 (<5)", 0, 5)
]

for label, min_r, max_r in ratio_ranges:
    subset = results_df[(results_df['price_to_macd_ratio'] >= min_r) & 
                        (results_df['price_to_macd_ratio'] < max_r)]
    if len(subset) > 0:
        validated = subset[subset['next_l_higher'].notna()]
        if len(validated) > 0:
            success = validated[validated['next_l_higher'] == True]
            success_rate = len(success) / len(validated) * 100
            avg_ratio = subset['price_to_macd_ratio'].mean()
            avg_change = success['l4_to_l5_change'].mean() if len(success) > 0 else 0
            
            print(f"\n{label}: {len(subset)}건")
            print(f"  평균 비율: {avg_ratio:.1f}")
            print(f"  성공률: {success_rate:.1f}% ({len(success)}/{len(validated)})")
            if len(success) > 0:
                print(f"  성공시 저점 상승: {avg_change:+.2f}%")

# Top divergence cases (low ratio but deep MACD)
print("\n" + "="*100)
print("Divergence 후보 (깊은 MACD + 낮은 비율)")
print("="*100)

# Deep MACD but low ratio = divergence signal
divergence_candidates = results_df[
    (results_df['l4_macd_hist'] < -100) &  # Deep MACD
    (results_df['price_to_macd_ratio'] < 30)  # But ratio not too high
]

if len(divergence_candidates) > 0:
    print(f"\n후보: {len(divergence_candidates)}건")
    validated = divergence_candidates[divergence_candidates['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        success_rate = len(success) / len(validated) * 100
        print(f"성공률: {success_rate:.1f}% ({len(success)}/{len(validated)})")
        
        if len(success) > 0:
            print(f"성공시 평균 저점 상승: {success['l4_to_l5_change'].mean():+.2f}%")
            
            print("\n\nTOP 5 Divergence 성공 케이스:")
            top5 = success.nlargest(5, 'l4_to_l5_change')
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
                print(f"   가격 하락: {row['l3_l4_price_drop']:.2f}%")
                print(f"   MACD: {row['l4_macd_hist']:.1f}")
                print(f"   비율: {row['price_to_macd_ratio']:.1f}")
                print(f"   저점 상승: {row['l4_to_l5_change']:+.2f}%")

# Extreme MACD cases
print("\n" + "="*100)
print("극한 MACD (< -200)")
print("="*100)

extreme_macd = results_df[results_df['l4_macd_hist'] < -200]
if len(extreme_macd) > 0:
    print(f"\n케이스: {len(extreme_macd)}건")
    validated = extreme_macd[extreme_macd['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        success_rate = len(success) / len(validated) * 100
        print(f"성공률: {success_rate:.1f}% ({len(success)}/{len(validated)})")
        if len(success) > 0:
            print(f"성공시 평균 저점 상승: {success['l4_to_l5_change'].mean():+.2f}%")

results_df.to_csv('macd_divergence_analysis.csv', index=False)
print(f"\n\n✅ 분석 저장: macd_divergence_analysis.csv")

print("\n" + "="*100)
print("최종 요약")
print("="*100)

print("\n핵심 발견:")
print("1. MACD 깊이가 클수록 성공률이 높은가?")
print("2. 가격 하락 대비 MACD 비율이 낮을수록 (Divergence) 성공률이 높은가?")
print("3. 극한 MACD (<-200)에서의 성공률은?")

print("\n" + "="*100)

