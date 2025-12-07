import pandas as pd
import numpy as np

print("=" * 80)
print("🚀 확정 공간 규칙 백테스트")
print("=" * 80)
print()

# Load data
print("Loading data...")
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

df_hlhlhl = pd.read_csv('hlhlhl_full_labeling_analysis.csv')
df_hlhlhl['l_time'] = pd.to_datetime(df_hlhlhl['l_time'])
df_hlhlhl['trendline_break_time'] = pd.to_datetime(df_hlhlhl['trendline_break_time'])

print(f"Total opportunities: {len(df_hlhlhl)}")
print()

# Filter: H3 break cases only
h3_break_cases = df_hlhlhl[df_hlhlhl['h3_break'] == True].copy()
print(f"H3 breakout cases: {len(h3_break_cases)}")
print()

# Backtest with confirmation space rules
def backtest_with_confirmation_space(df_opportunities, df_candles):
    """
    확정 공간 규칙:
    1. H3 돌파 확인
    2. 1캔들 이내 H3 리테스트 (±2%)
    3. 리테스트 후 첫 캔들 H3 위 종가 (MUST)
    4. 다음 5캔들 중 최소 4개 H3 위 (완화: 5->4)
    5. 진입: 첫 확정 캔들 종가
    6. SL: H3 - 0.5%
    7. TP1: H2 (50% 익절)
    8. TP2: H1 (50% 익절)
    """
    trades = []
    
    for idx, opp in df_opportunities.iterrows():
        breakout_time = opp['trendline_break_time']
        h3_price = opp['h3_price']
        h2_price = opp['h2_price']
        h1_price = opp['h1_price']
        power_score = opp['power_score']
        
        if pd.isna(breakout_time):
            continue
        
        # Get candles after breakout
        breakout_candles = df_candles[df_candles['datetime'] >= breakout_time]
        if len(breakout_candles) == 0:
            continue
            
        breakout_idx = breakout_candles.index[0]
        candles_after = df_candles.loc[breakout_idx:breakout_idx+20].copy()
        
        if len(candles_after) < 7:  # Need at least 7 candles (breakout + retest + 5 confirmation)
            continue
        
        # Rule 1: H3 breakout (already filtered)
        
        # Rule 2: Find retest within 1 candle (±2%)
        retest_candle_idx = None
        for i in range(1, min(3, len(candles_after))):  # Check first 2 candles after breakout
            candle = candles_after.iloc[i]
            distance_to_h3 = abs(candle['low'] - h3_price) / h3_price * 100
            
            if distance_to_h3 <= 2.0:
                retest_candle_idx = i
                break
        
        if retest_candle_idx is None:
            continue
        
        # Rule 3: First confirmation candle MUST close above H3
        if retest_candle_idx + 1 >= len(candles_after):
            continue
            
        first_conf_candle = candles_after.iloc[retest_candle_idx + 1]
        
        if first_conf_candle['close'] <= h3_price:
            # REJECT: First confirmation candle below H3
            continue
        
        # Rule 4: Next 5 candles - at least 4 must close above H3
        confirmation_candles = candles_after.iloc[retest_candle_idx+1:retest_candle_idx+6]
        
        if len(confirmation_candles) < 5:
            continue
        
        candles_above_h3 = sum(1 for _, c in confirmation_candles.iterrows() if c['close'] > h3_price)
        
        if candles_above_h3 < 4:  # At least 4 out of 5
            # REJECT: Not enough candles above H3
            continue
        
        # ACCEPTED: Enter trade at first confirmation candle close
        entry_price = first_conf_candle['close']
        entry_time = first_conf_candle['datetime']
        
        # SL and TP
        sl_price = h3_price * 0.995  # H3 - 0.5%
        tp1_price = h2_price
        tp2_price = h1_price
        
        # Risk:Reward
        risk = entry_price - sl_price
        if risk <= 0:
            continue
            
        reward_tp1 = tp1_price - entry_price
        reward_tp2 = tp2_price - entry_price
        
        rr_tp1 = reward_tp1 / risk
        rr_tp2 = reward_tp2 / risk
        
        # Track outcome
        future_candles = df_candles[df_candles['datetime'] > entry_time].head(200)
        
        exit_reason = None
        pnl_pct = 0
        exit_time = None
        
        tp1_hit = False
        sl_moved_to_breakeven = False
        
        for _, candle in future_candles.iterrows():
            # Check SL
            if not sl_moved_to_breakeven:
                if candle['low'] <= sl_price:
                    exit_reason = 'SL'
                    pnl_pct = (sl_price - entry_price) / entry_price * 100
                    exit_time = candle['datetime']
                    break
            else:
                # SL at breakeven
                if candle['low'] <= entry_price:
                    exit_reason = 'TP1_Breakeven'
                    pnl_pct = (tp1_price - entry_price) / entry_price * 100 * 0.5
                    exit_time = candle['datetime']
                    break
            
            # Check TP1
            if not tp1_hit and candle['high'] >= tp1_price:
                tp1_hit = True
                sl_moved_to_breakeven = True
            
            # Check TP2
            if tp1_hit and candle['high'] >= tp2_price:
                exit_reason = 'TP2_Full'
                pnl_tp1 = (tp1_price - entry_price) / entry_price * 100 * 0.5
                pnl_tp2 = (tp2_price - entry_price) / entry_price * 100 * 0.5
                pnl_pct = pnl_tp1 + pnl_tp2
                exit_time = candle['datetime']
                break
        
        if exit_reason is None:
            exit_reason = 'Open'
            exit_time = future_candles.iloc[-1]['datetime'] if len(future_candles) > 0 else entry_time
            pnl_pct = 0
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'h1_price': h1_price,
            'h2_price': h2_price,
            'h3_price': h3_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'power_score': power_score,
            'exit_reason': exit_reason,
            'exit_time': exit_time,
            'pnl_pct': pnl_pct,
            'rr_tp1': rr_tp1,
            'rr_tp2': rr_tp2,
            'year': entry_time.year
        })
    
    return pd.DataFrame(trades)

# Run backtest
print("=" * 80)
print("백테스트 실행 중...")
print("=" * 80)
print()

df_trades = backtest_with_confirmation_space(h3_break_cases, df_15m)

if len(df_trades) > 0:
    print(f"✅ 총 거래: {len(df_trades)}개")
    print(f"📊 총 PnL: {df_trades['pnl_pct'].sum():.2f}%")
    print(f"📈 평균 PnL: {df_trades['pnl_pct'].mean():.2f}%")
    print(f"🎯 승률: {len(df_trades[df_trades['pnl_pct'] > 0])/len(df_trades)*100:.1f}%")
    print()
    
    print("결과 분포:")
    print(df_trades['exit_reason'].value_counts())
    print()
    
    # SL/TP stats
    sl_count = len(df_trades[df_trades['exit_reason'] == 'SL'])
    tp1_count = len(df_trades[df_trades['exit_reason'] == 'TP1_Breakeven'])
    tp2_count = len(df_trades[df_trades['exit_reason'] == 'TP2_Full'])
    
    print("=" * 80)
    print("📊 상세 통계")
    print("=" * 80)
    print()
    
    print(f"SL 손절: {sl_count}개 ({sl_count/len(df_trades)*100:.1f}%)")
    print(f"TP1 Breakeven: {tp1_count}개 ({tp1_count/len(df_trades)*100:.1f}%)")
    print(f"TP2 Full: {tp2_count}개 ({tp2_count/len(df_trades)*100:.1f}%)")
    print()
    
    # By power score
    print("Power Score별 성과:")
    for score in sorted(df_trades['power_score'].unique()):
        score_df = df_trades[df_trades['power_score'] == score]
        avg_pnl = score_df['pnl_pct'].mean()
        win_rate = len(score_df[score_df['pnl_pct'] > 0]) / len(score_df) * 100
        tp2_rate = len(score_df[score_df['exit_reason'] == 'TP2_Full']) / len(score_df) * 100
        print(f"  Score {score}점: {len(score_df)}개 | 평균 {avg_pnl:+.2f}% | 승률 {win_rate:.1f}% | TP2 {tp2_rate:.1f}%")
    print()
    
    # By year
    print("연도별 성과:")
    for year in sorted(df_trades['year'].unique()):
        year_df = df_trades[df_trades['year'] == year]
        total_pnl = year_df['pnl_pct'].sum()
        avg_pnl = year_df['pnl_pct'].mean()
        win_rate = len(year_df[year_df['pnl_pct'] > 0]) / len(year_df) * 100
        print(f"  {year}년: {len(year_df)}개 | 총 {total_pnl:+.2f}% | 평균 {avg_pnl:+.2f}% | 승률 {win_rate:.1f}%")
    print()
    
    # Compare with previous results
    print("=" * 80)
    print("📈 이전 백테스트 vs 확정 공간 규칙")
    print("=" * 80)
    print()
    
    print("이전 백테스트 (파라미터 기반):")
    print("  - 총 거래: 203개")
    print("  - 총 PnL: -16.73%")
    print("  - 평균 PnL: -0.08%")
    print("  - 승률: 43.8%")
    print("  - SL 비율: 40.9%")
    print()
    
    print("확정 공간 규칙 (역방향 분석 기반):")
    print(f"  - 총 거래: {len(df_trades)}개")
    print(f"  - 총 PnL: {df_trades['pnl_pct'].sum():.2f}%")
    print(f"  - 평균 PnL: {df_trades['pnl_pct'].mean():.2f}%")
    print(f"  - 승률: {len(df_trades[df_trades['pnl_pct'] > 0])/len(df_trades)*100:.1f}%")
    print(f"  - SL 비율: {sl_count/len(df_trades)*100:.1f}%")
    print()
    
    # Improvement calculation
    improvement_pnl = df_trades['pnl_pct'].sum() - (-16.73)
    improvement_win_rate = (len(df_trades[df_trades['pnl_pct'] > 0])/len(df_trades)*100) - 43.8
    improvement_sl = 40.9 - (sl_count/len(df_trades)*100)
    
    print(f"🎯 개선 효과:")
    print(f"  - 총 PnL: {improvement_pnl:+.2f}%p")
    print(f"  - 승률: {improvement_win_rate:+.1f}%p")
    print(f"  - SL 비율 감소: {improvement_sl:+.1f}%p")
    print()
    
    # Save results
    df_trades.to_csv('backtest_confirmation_space_results.csv', index=False)
    print("✅ 결과 저장: backtest_confirmation_space_results.csv")
    print()
    
    # Final verdict
    print("=" * 80)
    print("🎯 최종 평가")
    print("=" * 80)
    print()
    
    if df_trades['pnl_pct'].sum() > 0:
        print("🎉🎉🎉 성공! 확정 공간 규칙이 수익을 냅니다!")
        print(f"   총 PnL: {df_trades['pnl_pct'].sum():.2f}%")
        print(f"   평균 PnL: {df_trades['pnl_pct'].mean():.2f}%")
        print(f"   승률: {len(df_trades[df_trades['pnl_pct'] > 0])/len(df_trades)*100:.1f}%")
        print()
        print("사용자님이 말씀하신 '확정 공간'이 정답이었습니다! 🔥")
    else:
        print("⚠️ 아직 손실입니다.")
        print(f"   총 PnL: {df_trades['pnl_pct'].sum():.2f}%")
        print(f"   하지만 이전 대비 {improvement_pnl:+.2f}%p 개선!")
        print()
        print("추가 개선 필요:")
        print("  - Power score 필터 강화?")
        print("  - TP 전략 조정?")
        print("  - 2021년 필터링?")
else:
    print("⚠️ 확정 공간 조건을 만족하는 거래가 없습니다!")
    print("규칙이 너무 엄격할 수 있습니다.")

