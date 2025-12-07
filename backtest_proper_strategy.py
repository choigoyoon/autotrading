import pandas as pd
import numpy as np

# Load data
ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])

llll_patterns = pd.read_csv('LLLL_chart_patterns_analysis.csv')
llll_patterns['l4_datetime'] = pd.to_datetime(llll_patterns['l4_datetime'])

all_l_values = pd.read_csv('all_L_values_inference.csv')
all_l_values['datetime'] = pd.to_datetime(all_l_values['datetime'])
all_l_values = all_l_values.sort_values('datetime').reset_index(drop=True)

print("=" * 100)
print("올바른 백테스트: 진입에 미래 데이터 사용 안함")
print("=" * 100)

results = []

for idx, pattern in llll_patterns.iterrows():
    l4_dt = pattern['l4_datetime']
    l4_price = pattern['l4_price']
    
    # Entry conditions (NO FUTURE DATA!)
    conditions_met = 0
    if pattern['l4_rsi'] < 30: conditions_met += 1
    if pattern['l4_macd_hist'] < -50: conditions_met += 1
    if pattern.get('l4_bb_position', 0) < 0.1: conditions_met += 1
    if pattern.get('l4_volume_ratio', 0) > 3.0: conditions_met += 1
    if pattern.get('l4_atr_pct', 0) > 0.5: conditions_met += 1
    
    # Filter: only enter if 4+ conditions met
    if conditions_met < 4:
        continue
    
    # Find L4 candle
    l4_candle = ohlcv[ohlcv['datetime'] == l4_dt]
    if len(l4_candle) == 0:
        continue
    l4_idx = l4_candle.index[0]
    
    # Entry: L4 confirmed (10 bars after L4 actual low)
    entry_idx = l4_idx + 10
    if entry_idx >= len(ohlcv):
        continue
    
    entry_candle = ohlcv.iloc[entry_idx]
    entry_price = entry_candle['close']
    entry_dt = entry_candle['datetime']
    
    # Exit conditions
    tp1 = entry_price * 1.02  # +2%
    tp2 = entry_price * 1.035  # +3.5%
    sl = l4_price * 0.985  # L4 -1.5%
    time_stop_idx = entry_idx + 192  # 48 hours (192 bars)
    
    # Simulate forward
    exit_type = None
    exit_price = None
    exit_dt = None
    exit_idx = None
    pnl_pct = 0
    
    for i in range(entry_idx + 1, min(time_stop_idx + 1, len(ohlcv))):
        candle = ohlcv.iloc[i]
        
        # Check SL
        if candle['low'] <= sl:
            exit_type = 'SL'
            exit_price = sl
            exit_dt = candle['datetime']
            exit_idx = i
            pnl_pct = ((sl - entry_price) / entry_price) * 100
            break
        
        # Check TP2
        if candle['high'] >= tp2:
            exit_type = 'TP2'
            exit_price = tp2
            exit_dt = candle['datetime']
            exit_idx = i
            pnl_pct = ((tp2 - entry_price) / entry_price) * 100
            break
        
        # Check TP1
        if candle['high'] >= tp1 and exit_type is None:
            exit_type = 'TP1'
            exit_price = tp1
            exit_dt = candle['datetime']
            exit_idx = i
            pnl_pct = ((tp1 - entry_price) / entry_price) * 100
            # Don't break, continue to TP2
        
        # Time stop
        if i == time_stop_idx:
            exit_type = exit_type or 'TIME'
            exit_price = candle['close']
            exit_dt = candle['datetime']
            exit_idx = i
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            break
    
    if exit_type is None:
        continue
    
    # Find next L-value for validation
    l4_in_all_l = all_l_values[all_l_values['datetime'] == l4_dt].index
    if len(l4_in_all_l) == 0:
        next_l_higher = None
    else:
        l4_in_all_l_idx = l4_in_all_l[0]
        if l4_in_all_l_idx + 1 < len(all_l_values):
            next_l = all_l_values.iloc[l4_in_all_l_idx + 1]
            next_l_higher = next_l['l_price'] > l4_price
        else:
            next_l_higher = None
    
    results.append({
        'entry_dt': entry_dt,
        'entry_price': entry_price,
        'l4_price': l4_price,
        'exit_dt': exit_dt,
        'exit_price': exit_price,
        'exit_type': exit_type,
        'pnl_pct': pnl_pct,
        'conditions_met': conditions_met,
        'next_l_higher': next_l_higher,
        'total_drop_pct': pattern['total_drop_pct']
    })

results_df = pd.DataFrame(results)

print(f"\n진입 조건: LLLL + 4+ 극한 과매도 조건")
print(f"총 거래: {len(results_df)}건")

if len(results_df) == 0:
    print("\n거래 없음!")
else:
    # Overall performance
    print("\n" + "="*100)
    print("전체 성과")
    print("="*100)
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['pnl_pct'] > 0])
    win_rate = (winning_trades / total_trades * 100)
    avg_pnl = results_df['pnl_pct'].mean()
    total_pnl = results_df['pnl_pct'].sum()
    
    print(f"\n총 거래: {total_trades}건")
    print(f"승률: {win_rate:.1f}% ({winning_trades}/{total_trades})")
    print(f"평균 수익: {avg_pnl:.2f}%")
    print(f"총 누적 수익: {total_pnl:.2f}%")
    
    # Calculate monthly
    months = (ohlcv['datetime'].max() - ohlcv['datetime'].min()).days / 30
    monthly_trades = total_trades / months
    monthly_pnl = total_pnl / months
    
    print(f"\n월간 거래: {monthly_trades:.2f}건")
    print(f"월간 수익: {monthly_pnl:.2f}%")
    
    # Exit type breakdown
    print("\n" + "="*100)
    print("청산 유형별 분석")
    print("="*100)
    
    for exit_type in ['SL', 'TP1', 'TP2', 'TIME']:
        subset = results_df[results_df['exit_type'] == exit_type]
        if len(subset) > 0:
            count = len(subset)
            pct = (count / total_trades * 100)
            avg_pnl = subset['pnl_pct'].mean()
            print(f"\n{exit_type}: {count}건 ({pct:.1f}%)")
            print(f"  평균 수익: {avg_pnl:.2f}%")
    
    # Next L validation
    print("\n" + "="*100)
    print("다음 L값 검증 (진입 성공 여부)")
    print("="*100)
    
    validated = results_df[results_df['next_l_higher'].notna()]
    if len(validated) > 0:
        success = validated[validated['next_l_higher'] == True]
        fail = validated[validated['next_l_higher'] == False]
        
        print(f"\n검증 가능: {len(validated)}건")
        print(f"진입 성공 (L5 > L4): {len(success)}건 ({len(success)/len(validated)*100:.1f}%)")
        print(f"진입 실패 (L5 <= L4): {len(fail)}건 ({len(fail)/len(validated)*100:.1f}%)")
        
        if len(success) > 0:
            print(f"\n성공 케이스 평균 PNL: {success['pnl_pct'].mean():.2f}%")
        if len(fail) > 0:
            print(f"실패 케이스 평균 PNL: {fail['pnl_pct'].mean():.2f}%")
    
    # Strong drop analysis
    print("\n" + "="*100)
    print("강한 하락 (7%+) 케이스")
    print("="*100)
    
    strong_drop = results_df[results_df['total_drop_pct'] <= -7]
    if len(strong_drop) > 0:
        winning = len(strong_drop[strong_drop['pnl_pct'] > 0])
        print(f"\n케이스: {len(strong_drop)}건")
        print(f"승률: {winning/len(strong_drop)*100:.1f}%")
        print(f"평균 수익: {strong_drop['pnl_pct'].mean():.2f}%")
        
        validated = strong_drop[strong_drop['next_l_higher'].notna()]
        if len(validated) > 0:
            success = len(validated[validated['next_l_higher'] == True])
            print(f"진입 성공률 (L5 > L4): {success/len(validated)*100:.1f}%")
    
    # Save results
    results_df.to_csv('proper_backtest_results.csv', index=False)
    print(f"\n\n✅ 백테스트 결과 저장: proper_backtest_results.csv")
    
    # Final verdict
    print("\n" + "="*100)
    print("최종 평가")
    print("="*100)
    
    if monthly_pnl > 2.0:
        print(f"\n✅ 전략 실행 가능! (월 {monthly_pnl:.2f}%)")
    elif monthly_pnl > 1.0:
        print(f"\n⚠️  전략 수익성 보통 (월 {monthly_pnl:.2f}%)")
    else:
        print(f"\n❌ 전략 수익성 낮음 (월 {monthly_pnl:.2f}%)")
    
    print(f"\n거래 빈도: {'적절함' if monthly_trades > 0.5 else '너무 낮음'} (월 {monthly_trades:.2f}건)")
    print(f"승률: {'양호' if win_rate > 55 else '부족'} ({win_rate:.1f}%)")

print("\n" + "="*100)

