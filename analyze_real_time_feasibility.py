import pandas as pd
import numpy as np

# Load LLLL patterns analysis
llll_df = pd.read_csv('LLLL_chart_patterns_analysis.csv')
llll_df['l4_datetime'] = pd.to_datetime(llll_df['l4_datetime'])

# Load OHLCV data
ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])

print("=" * 100)
print("실시간 거래 가능성 분석 - Look-Ahead Bias 제거 후 실제 수익성")
print("=" * 100)

print(f"\n총 LLLL 패턴: {len(llll_df)}개")
print(f"데이터 기간: {ohlcv['datetime'].min()} ~ {ohlcv['datetime'].max()}")

# Key insight: L-value needs 10 bars to confirm
# So when we see L4, we need to wait 10 bars to confirm it's actually an L-value
# Then wait for entry signal

results = []

for idx, pattern in llll_df.iterrows():
    l4_dt = pattern['l4_datetime']
    l4_price = pattern['l4_price']
    
    # Find L4 in OHLCV
    l4_candle = ohlcv[ohlcv['datetime'] == l4_dt]
    if len(l4_candle) == 0:
        continue
    
    l4_idx = l4_candle.index[0]
    
    # REALITY 1: Need 10 bars AFTER L4 to confirm it's an L-value
    confirm_idx = l4_idx + 10
    if confirm_idx >= len(ohlcv):
        continue
    
    confirm_candle = ohlcv.iloc[confirm_idx]
    confirm_price = confirm_candle['close']
    price_change_at_confirm = ((confirm_price - l4_price) / l4_price) * 100
    
    # REALITY 2: After confirmation, wait for entry signal (first bullish candle)
    entry_idx = None
    for j in range(confirm_idx, min(confirm_idx + 20, len(ohlcv))):
        candle = ohlcv.iloc[j]
        if candle['close'] > candle['open']:  # Bullish candle
            entry_idx = j
            break
    
    if entry_idx is None:
        continue
    
    entry_candle = ohlcv.iloc[entry_idx]
    entry_price = entry_candle['close']
    
    # Total delay from L4 to actual entry
    total_delay_bars = entry_idx - l4_idx
    total_delay_hours = total_delay_bars * 0.25
    entry_slippage = ((entry_price - l4_price) / l4_price) * 100
    
    # Calculate REALISTIC max gain from ENTRY price (not L4!)
    max_gain_20 = 0
    max_gain_40 = 0
    max_gain_60 = 0
    
    for j in range(entry_idx, min(entry_idx + 20, len(ohlcv))):
        gain = ((ohlcv.iloc[j]['high'] - entry_price) / entry_price) * 100
        max_gain_20 = max(max_gain_20, gain)
    
    for j in range(entry_idx, min(entry_idx + 40, len(ohlcv))):
        gain = ((ohlcv.iloc[j]['high'] - entry_price) / entry_price) * 100
        max_gain_40 = max(max_gain_40, gain)
    
    for j in range(entry_idx, min(entry_idx + 60, len(ohlcv))):
        gain = ((ohlcv.iloc[j]['high'] - entry_price) / entry_price) * 100
        max_gain_60 = max(max_gain_60, gain)
    
    # Get pattern metrics
    total_drop_pct = pattern.get('total_drop_pct', 0)
    l4_rsi = pattern.get('l4_rsi', 0)
    l4_macd_hist = pattern.get('l4_macd_hist', 0)
    conditions_met = 0
    if l4_rsi < 30: conditions_met += 1
    if l4_macd_hist < -50: conditions_met += 1
    if pattern.get('l4_bb_position', 0) < 0.1: conditions_met += 1
    if pattern.get('l4_volume_ratio', 0) > 3.0: conditions_met += 1
    if pattern.get('l4_atr_pct', 0) > 0.5: conditions_met += 1
    
    results.append({
        'l4_datetime': l4_dt,
        'l4_price': l4_price,
        'entry_datetime': entry_candle['datetime'],
        'entry_price': entry_price,
        'delay_bars': total_delay_bars,
        'delay_hours': total_delay_hours,
        'entry_slippage_pct': entry_slippage,
        'max_gain_20bars': max_gain_20,
        'max_gain_40bars': max_gain_40,
        'max_gain_60bars': max_gain_60,
        'total_drop_pct': total_drop_pct,
        'conditions_met': conditions_met,
        'l4_rsi': l4_rsi,
        'l4_macd_hist': l4_macd_hist
    })

results_df = pd.DataFrame(results)

print("\n" + "="*100)
print("핵심 발견: 실제 진입 지연 및 가격 변화")
print("="*100)

print(f"\n분석 가능한 패턴: {len(results_df)}개")
print(f"\n평균 진입 지연: {results_df['delay_bars'].mean():.1f} 봉 ({results_df['delay_hours'].mean():.1f} 시간)")
print(f"평균 진입 슬리피지: {results_df['entry_slippage_pct'].mean():+.2f}% (L4 대비)")

print("\n" + "="*100)
print("진입 슬리피지 분포 (L4 가격 → 실제 진입 가격)")
print("="*100)

slippage_ranges = [
    ("큰 상승으로 진입 불리", 5, 100),
    ("상당한 상승", 3, 5),
    ("중간 상승", 1.5, 3),
    ("소폭 상승", 0.5, 1.5),
    ("거의 변화없음", -0.5, 0.5),
    ("오히려 하락 (진입 유리!)", -100, -0.5)
]

for label, min_val, max_val in slippage_ranges:
    subset = results_df[(results_df['entry_slippage_pct'] >= min_val) & 
                        (results_df['entry_slippage_pct'] < max_val)]
    if len(subset) > 0:
        success_2pct = (subset['max_gain_60bars'] >= 2.0).sum() / len(subset) * 100
        avg_gain = subset['max_gain_60bars'].mean()
        print(f"\n{label}: {len(subset)}건 ({len(subset)/len(results_df)*100:.1f}%)")
        print(f"  슬리피지: {subset['entry_slippage_pct'].mean():+.2f}%")
        print(f"  평균 최대 반등 (진입가 기준): {avg_gain:.2f}%")
        print(f"  성공률 (2%+): {success_2pct:.1f}%")

print("\n" + "="*100)
print("백테스트 vs 실제 거래 수익 비교")
print("="*100)

# Compare backtest (L4 price entry) vs real (delayed entry)
backtest_gain = llll_df['max_gain_60bars'].mean()
real_gain = results_df['max_gain_60bars'].mean()
degradation = ((real_gain - backtest_gain) / backtest_gain) * 100

print(f"\n백테스트 (L4 가격 진입): {backtest_gain:.2f}%")
print(f"실제 거래 (지연 진입):   {real_gain:.2f}%")
print(f"수익 감소:               {degradation:+.1f}%")

print("\n" + "="*100)
print("실제 수익 가능한 케이스 분석")
print("="*100)

# Profitable cases: entry slippage < 2% AND max gain > 2%
profitable = results_df[(results_df['entry_slippage_pct'] < 2.0) & 
                        (results_df['max_gain_60bars'] >= 2.0)]

print(f"\n✅ 수익 가능 케이스 (진입슬리피지 < 2%, 최대반등 > 2%)")
print(f"   {len(profitable)}건 / {len(results_df)}건 ({len(profitable)/len(results_df)*100:.1f}%)")

if len(profitable) > 0:
    print(f"\n   평균 진입 슬리피지: {profitable['entry_slippage_pct'].mean():+.2f}%")
    print(f"   평균 최대 반등: {profitable['max_gain_60bars'].mean():.2f}%")
    print(f"   평균 순수익: {profitable['max_gain_60bars'].mean() - profitable['entry_slippage_pct'].mean():.2f}%")
    
    # Calculate realistic performance
    start_date = ohlcv['datetime'].min()
    end_date = ohlcv['datetime'].max()
    months = (end_date - start_date).days / 30
    
    monthly_trades = len(profitable) / months
    # Assume we capture 60% of max gain (realistic TP)
    avg_profit_per_trade = (profitable['max_gain_60bars'].mean() - profitable['entry_slippage_pct'].mean()) * 0.6
    monthly_profit = monthly_trades * avg_profit_per_trade
    
    print(f"\n📊 예상 월간 성과:")
    print(f"   월간 거래 횟수: {monthly_trades:.2f}건")
    print(f"   평균 거래당 수익: {avg_profit_per_trade:.2f}%")
    print(f"   예상 월간 수익: {monthly_profit:.2f}%")
    
    # Strong signals
    strong = profitable[profitable['conditions_met'] >= 4]
    if len(strong) > 0:
        strong_monthly_trades = len(strong) / months
        strong_avg_profit = (strong['max_gain_60bars'].mean() - strong['entry_slippage_pct'].mean()) * 0.6
        strong_monthly_profit = strong_monthly_trades * strong_avg_profit
        
        print(f"\n🔥 강한 신호만 (4+ 조건):")
        print(f"   월간 거래 횟수: {strong_monthly_trades:.2f}건")
        print(f"   평균 거래당 수익: {strong_avg_profit:.2f}%")
        print(f"   예상 월간 수익: {strong_monthly_profit:.2f}%")

print("\n" + "="*100)
print("TOP 10 최고 수익 케이스 (실제 진입 기준)")
print("="*100)

top10 = results_df.nlargest(10, 'max_gain_60bars')
for i, (idx, row) in enumerate(top10.iterrows(), 1):
    net_profit = row['max_gain_60bars'] - row['entry_slippage_pct']
    print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
    print(f"   L4 가격: ${row['l4_price']:.2f}")
    print(f"   진입 지연: {row['delay_hours']:.1f}시간")
    print(f"   진입 슬리피지: {row['entry_slippage_pct']:+.2f}%")
    print(f"   최대 반등: {row['max_gain_60bars']:.2f}%")
    print(f"   순수익: {net_profit:.2f}%")
    print(f"   조건: {row['conditions_met']}/5")

# Worst cases
print("\n" + "="*100)
print("WORST 5 케이스 (큰 슬리피지로 수익 감소)")
print("="*100)

worst5 = results_df.nlargest(5, 'entry_slippage_pct')
for i, (idx, row) in enumerate(worst5.iterrows(), 1):
    net_profit = row['max_gain_60bars'] - row['entry_slippage_pct']
    print(f"\n#{i}. {row['l4_datetime'].strftime('%Y-%m-%d %H:%M')}")
    print(f"   진입 슬리피지: {row['entry_slippage_pct']:+.2f}% ⚠️")
    print(f"   최대 반등: {row['max_gain_60bars']:.2f}%")
    print(f"   순수익: {net_profit:.2f}%")

# Save results
results_df.to_csv('real_time_feasibility_analysis.csv', index=False)
print(f"\n\n✅ 상세 분석 결과 저장: real_time_feasibility_analysis.csv")

print("\n" + "="*100)
print("최종 결론: 실시간 거래 가능한가?")
print("="*100)

print(f"\n⚠️  백테스트 오류 확인됨:")
print(f"    • 백테스트는 L4 가격({backtest_gain:.2f}%)에서 진입 가정")
print(f"    • 실제는 평균 {results_df['delay_hours'].mean():.1f}시간 지연 후 진입")
print(f"    • 진입 슬리피지: 평균 {results_df['entry_slippage_pct'].mean():+.2f}%")
print(f"    • 실제 수익: {real_gain:.2f}% (백테스트 대비 {degradation:+.1f}%)")

if len(profitable) > 0 and monthly_profit > 2.0:
    print(f"\n✅ 그럼에도 전략 실행 가능:")
    print(f"    • 수익 가능 케이스: {len(profitable)}건 ({len(profitable)/len(results_df)*100:.1f}%)")
    print(f"    • 월간 거래: {monthly_trades:.2f}건")
    print(f"    • 예상 월수익: {monthly_profit:.2f}%")
    print(f"\n    ✅ 권장: 실시간 거래 가능 (월 2% 이상)")
elif len(profitable) > 0 and monthly_profit > 1.0:
    print(f"\n⚠️  전략 수익성 보통:")
    print(f"    • 수익 가능 케이스: {len(profitable)}건 ({len(profitable)/len(results_df)*100:.1f}%)")
    print(f"    • 월간 거래: {monthly_trades:.2f}건")
    print(f"    • 예상 월수익: {monthly_profit:.2f}%")
    print(f"\n    ⚠️  주의: 전략 개선 필요 (월 1-2%)")
else:
    if len(profitable) > 0:
        print(f"\n❌ 전략 실행 불가:")
        print(f"    • 수익 가능 케이스: {len(profitable)}건 ({len(profitable)/len(results_df)*100:.1f}%)")
        print(f"    • 월간 거래: {monthly_trades:.2f}건")
        print(f"    • 예상 월수익: {monthly_profit:.2f}%")
    else:
        print(f"\n❌ 전략 실행 불가:")
        print(f"    • 수익 가능 케이스: 0건")
    print(f"\n    ❌ 권장: 전략 폐기 또는 재설계 (월 1% 미만)")

print("\n" + "="*100)

