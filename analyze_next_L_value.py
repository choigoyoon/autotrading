import pandas as pd
import numpy as np

# Load all L-values
all_l_values = pd.read_csv('all_L_values_inference.csv')
all_l_values['datetime'] = pd.to_datetime(all_l_values['datetime'])
all_l_values = all_l_values.sort_values('datetime').reset_index(drop=True)

# Load LLLL patterns
llll_patterns = pd.read_csv('LLLL_chart_patterns_analysis.csv')
llll_patterns['l4_datetime'] = pd.to_datetime(llll_patterns['l4_datetime'])

print("=" * 100)
print("올바른 검증: L4 이후 다음 L값이 상승했는가?")
print("=" * 100)

print(f"\n총 L-값: {len(all_l_values):,}개")
print(f"총 LLLL 패턴: {len(llll_patterns)}개")

results = []

for idx, pattern in llll_patterns.iterrows():
    l4_dt = pattern['l4_datetime']
    l4_price = pattern['l4_price']
    
    # Find L4 in all L-values
    l4_idx = all_l_values[all_l_values['datetime'] == l4_dt].index
    if len(l4_idx) == 0:
        continue
    l4_idx = l4_idx[0]
    
    # Find NEXT L-value after L4
    next_l_idx = l4_idx + 1
    if next_l_idx >= len(all_l_values):
        continue  # No next L-value
    
    next_l = all_l_values.iloc[next_l_idx]
    next_l_price = next_l['l_price']
    next_l_datetime = next_l['datetime']
    
    # Calculate price change
    l_to_l_change = ((next_l_price - l4_price) / l4_price) * 100
    
    # Time between L4 and next L
    time_diff = (next_l_datetime - l4_dt).total_seconds() / 3600  # hours
    
    # Success = Next L is HIGHER than L4
    success = next_l_price > l4_price
    
    # Get pattern details
    total_drop = pattern['total_drop_pct']
    l4_rsi = pattern['l4_rsi']
    l4_macd = pattern['l4_macd_hist']
    
    # Count extreme conditions
    conditions_met = 0
    if l4_rsi < 30: conditions_met += 1
    if l4_macd < -50: conditions_met += 1
    if pattern.get('l4_bb_position', 0) < 0.1: conditions_met += 1
    if pattern.get('l4_volume_ratio', 0) > 3.0: conditions_met += 1
    if pattern.get('l4_atr_pct', 0) > 0.5: conditions_met += 1
    
    results.append({
        'l4_datetime': l4_dt,
        'l4_price': l4_price,
        'next_l_datetime': next_l_datetime,
        'next_l_price': next_l_price,
        'l_to_l_change_pct': l_to_l_change,
        'time_to_next_l_hours': time_diff,
        'success': success,
        'total_drop_pct': total_drop,
        'conditions_met': conditions_met,
        'l4_rsi': l4_rsi,
        'l4_macd_hist': l4_macd
    })

results_df = pd.DataFrame(results)

print("\n" + "="*100)
print("핵심 결과: L4가 진짜 바닥이었는가?")
print("="*100)

total = len(results_df)
success_count = results_df['success'].sum()
fail_count = total - success_count
success_rate = (success_count / total * 100) if total > 0 else 0

print(f"\n총 분석 케이스: {total}개")
print(f"\n✅ 성공 (L5 > L4): {success_count}건 ({success_rate:.1f}%)")
print(f"❌ 실패 (L5 <= L4): {fail_count}건 ({100-success_rate:.1f}%)")

# Success cases analysis
success_df = results_df[results_df['success'] == True]
fail_df = results_df[results_df['success'] == False]

if len(success_df) > 0:
    print("\n" + "="*100)
    print("성공 케이스 분석 (L5 > L4)")
    print("="*100)
    print(f"\n평균 저점 상승: {success_df['l_to_l_change_pct'].mean():+.2f}%")
    print(f"최소 저점 상승: {success_df['l_to_l_change_pct'].min():+.2f}%")
    print(f"최대 저점 상승: {success_df['l_to_l_change_pct'].max():+.2f}%")
    print(f"평균 시간 간격: {success_df['time_to_next_l_hours'].mean():.1f} 시간")

if len(fail_df) > 0:
    print("\n" + "="*100)
    print("실패 케이스 분석 (L5 <= L4)")
    print("="*100)
    print(f"\n평균 저점 하락: {fail_df['l_to_l_change_pct'].mean():+.2f}%")
    print(f"최소 저점 하락: {fail_df['l_to_l_change_pct'].min():+.2f}%")
    print(f"최대 저점 하락: {fail_df['l_to_l_change_pct'].max():+.2f}%")
    print(f"평균 시간 간격: {fail_df['time_to_next_l_hours'].mean():.1f} 시간")

# Analyze by conditions
print("\n" + "="*100)
print("극한 과매도 조건에 따른 성공률")
print("="*100)

for cond in range(0, 6):
    subset = results_df[results_df['conditions_met'] == cond]
    if len(subset) > 0:
        success = subset['success'].sum()
        rate = (success / len(subset) * 100)
        avg_change = subset[subset['success']]['l_to_l_change_pct'].mean() if success > 0 else 0
        print(f"\n{cond}개 조건 충족: {len(subset)}건")
        print(f"  성공률: {rate:.1f}% ({success}/{len(subset)})")
        if success > 0:
            print(f"  평균 저점 상승: {avg_change:+.2f}%")

# Analyze by total drop
print("\n" + "="*100)
print("L1-L4 총 하락폭에 따른 성공률")
print("="*100)

drop_ranges = [
    ("약한 하락 (<3%)", -3, 0),
    ("중간 하락 (3-5%)", -5, -3),
    ("강한 하락 (5-7%)", -7, -5),
    ("매우 강한 하락 (7%+)", -100, -7)
]

for label, min_drop, max_drop in drop_ranges:
    subset = results_df[(results_df['total_drop_pct'] >= min_drop) & 
                        (results_df['total_drop_pct'] < max_drop)]
    if len(subset) > 0:
        success = subset['success'].sum()
        rate = (success / len(subset) * 100)
        avg_change = subset[subset['success']]['l_to_l_change_pct'].mean() if success > 0 else 0
        print(f"\n{label}: {len(subset)}건")
        print(f"  성공률: {rate:.1f}% ({success}/{len(subset)})")
        if success > 0:
            print(f"  평균 저점 상승: {avg_change:+.2f}%")

# Best success cases
print("\n" + "="*100)
print("TOP 10 최고 성공 케이스 (저점 상승폭)")
print("="*100)

top_success = success_df.nlargest(10, 'l_to_l_change_pct') if len(success_df) > 0 else pd.DataFrame()
for i, (idx, row) in enumerate(top_success.iterrows(), 1):
    print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
    print(f"   L4 가격: ${row['l4_price']:.2f}")
    print(f"   L5 가격: ${row['next_l_price']:.2f}")
    print(f"   저점 상승: {row['l_to_l_change_pct']:+.2f}%")
    print(f"   시간 간격: {row['time_to_next_l_hours']:.1f}시간")
    print(f"   조건: {row['conditions_met']}/5")

# Worst failure cases
print("\n" + "="*100)
print("WORST 10 실패 케이스 (저점 하락폭)")
print("="*100)

worst_fail = fail_df.nsmallest(10, 'l_to_l_change_pct') if len(fail_df) > 0 else pd.DataFrame()
for i, (idx, row) in enumerate(worst_fail.iterrows(), 1):
    print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
    print(f"   L4 가격: ${row['l4_price']:.2f}")
    print(f"   L5 가격: ${row['next_l_price']:.2f}")
    print(f"   저점 하락: {row['l_to_l_change_pct']:+.2f}%")
    print(f"   시간 간격: {row['time_to_next_l_hours']:.1f}시간")
    print(f"   조건: {row['conditions_met']}/5")

# Calculate realistic trading performance
print("\n" + "="*100)
print("실제 거래 수익성 분석")
print("="*100)

if len(success_df) > 0:
    # For successful cases, we could enter at L4 and exit at higher price before next L
    # Assume we capture 50% of the move from L4 to next L
    avg_l_to_l_gain = success_df['l_to_l_change_pct'].mean() * 0.5
    
    # Monthly trades
    ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
    ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
    months = (ohlcv['datetime'].max() - ohlcv['datetime'].min()).days / 30
    
    monthly_success_trades = len(success_df) / months
    monthly_profit = monthly_success_trades * avg_l_to_l_gain
    
    print(f"\n성공 케이스만 거래한다면:")
    print(f"  월간 거래: {monthly_success_trades:.2f}건")
    print(f"  거래당 평균 수익: {avg_l_to_l_gain:.2f}%")
    print(f"  예상 월간 수익: {monthly_profit:.2f}%")
    
    if monthly_profit > 2.0:
        print(f"\n  ✅ 전략 실행 가능! (월 2% 이상)")
    elif monthly_profit > 1.0:
        print(f"\n  ⚠️  전략 수익성 보통 (월 1-2%)")
    else:
        print(f"\n  ❌ 전략 수익성 낮음 (월 1% 미만)")

# Save results
results_df.to_csv('next_L_value_analysis.csv', index=False)
print(f"\n\n✅ 상세 분석 결과 저장: next_L_value_analysis.csv")

print("\n" + "="*100)
print("최종 결론")
print("="*100)

print(f"\nLLLL 패턴이 진짜 바닥이었는가?")
print(f"  성공률: {success_rate:.1f}% ({success_count}/{total})")

if success_rate > 60:
    print(f"\n  ✅ LLLL 패턴은 신뢰할 만한 바닥 신호!")
elif success_rate > 50:
    print(f"\n  ⚠️  LLLL 패턴은 보통 수준의 바닥 신호")
else:
    print(f"\n  ❌ LLLL 패턴은 신뢰하기 어려운 신호")

print("\n" + "="*100)

