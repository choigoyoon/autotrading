import pandas as pd
import numpy as np

# Load the proper HLHLHL analysis data
print("Loading HLHLHL full labeling analysis...")
df_hlhlhl = pd.read_csv('hlhlhl_full_labeling_analysis.csv')

print(f"Total counter-trend opportunities: {len(df_hlhlhl)}")
print()

# Load 15m candle data for price tracking
print("Loading 15m candle data...")
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print(f"Loaded {len(df_15m)} candles")
print()

# Filter by tradability
print("Filtering by tradability...")
tradable = df_hlhlhl[df_hlhlhl['tradability'] == 'Tradable'].copy()
conditional = df_hlhlhl[df_hlhlhl['tradability'] == 'Conditional'].copy()
not_tradable = df_hlhlhl[df_hlhlhl['tradability'] == 'Not Tradable'].copy()

print(f"Tradable: {len(tradable)} ({len(tradable)/len(df_hlhlhl)*100:.1f}%)")
print(f"Conditional: {len(conditional)} ({len(conditional)/len(df_hlhlhl)*100:.1f}%)")
print(f"Not Tradable: {len(not_tradable)} ({len(not_tradable)/len(df_hlhlhl)*100:.1f}%)")
print()

# Convert datetime columns
df_hlhlhl['ll_time'] = pd.to_datetime(df_hlhlhl['ll_time'])
df_hlhlhl['h3_breakout_time'] = pd.to_datetime(df_hlhlhl['h3_breakout_time'])

# Backtest function
def backtest_trades(df_opportunities, df_candles, sl_pct=0.5, tp1_pct=0.5, tp2_pct=0.5):
    """
    Backtest using the HLHLHL analysis data
    
    Entry: After H3 breakout (already identified in the data)
    SL: H3 - sl_pct%
    TP1: H2 (50% exit by default)
    TP2: H1 (50% exit by default)
    """
    trades = []
    
    for idx, opp in df_opportunities.iterrows():
        # Skip if no H3 breakout
        if pd.isna(opp['h3_breakout_time']):
            continue
        
        # Entry parameters from the data
        entry_time = opp['h3_breakout_time']
        h1_price = opp['h1_price']
        h2_price = opp['h2_price']
        h3_price = opp['h3_price']
        power_score = opp['power_score']
        
        # Find entry candle
        entry_candles = df_candles[df_candles['datetime'] >= entry_time]
        if len(entry_candles) == 0:
            continue
        
        entry_candle = entry_candles.iloc[0]
        entry_price = entry_candle['close']
        
        # Calculate SL and TP
        sl_price = h3_price * (1 - sl_pct/100)
        tp1_price = h2_price
        tp2_price = h1_price
        
        # Calculate R:R
        risk = entry_price - sl_price
        if risk <= 0:
            continue
            
        reward_tp1 = tp1_price - entry_price
        reward_tp2 = tp2_price - entry_price
        
        rr_tp1 = reward_tp1 / risk
        rr_tp2 = reward_tp2 / risk
        
        # Track trade outcome
        future_candles = df_candles[df_candles['datetime'] > entry_time].head(200)
        
        exit_reason = None
        pnl_pct = 0
        exit_time = None
        
        # Strategy: tp1_pct at TP1, move SL to breakeven, (1-tp1_pct) at TP2
        tp1_hit = False
        sl_moved_to_breakeven = False
        
        for _, candle in future_candles.iterrows():
            # Check SL first
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
                    # Already took tp1_pct profit at TP1
                    pnl_pct = (tp1_price - entry_price) / entry_price * 100 * tp1_pct
                    exit_time = candle['datetime']
                    break
            
            # Check TP1
            if not tp1_hit and candle['high'] >= tp1_price:
                tp1_hit = True
                sl_moved_to_breakeven = True
                # Don't exit yet, continue to TP2
            
            # Check TP2
            if tp1_hit and candle['high'] >= tp2_price:
                exit_reason = 'TP2_Full'
                # tp1_pct at TP1 + (1-tp1_pct) at TP2
                pnl_tp1 = (tp1_price - entry_price) / entry_price * 100 * tp1_pct
                pnl_tp2 = (tp2_price - entry_price) / entry_price * 100 * (1 - tp1_pct)
                pnl_pct = pnl_tp1 + pnl_tp2
                exit_time = candle['datetime']
                break
        
        # If no exit within 200 candles, mark as Open
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
            'year': entry_time.year,
            'tradability': opp['tradability']
        })
    
    return pd.DataFrame(trades)

# Run backtests
print("=" * 80)
print("백테스트 1: Tradable only (power_score >= 5, H3 breakout)")
print("=" * 80)
print()

df_trades_tradable = backtest_trades(tradable, df_15m, sl_pct=0.5, tp1_pct=0.5)

if len(df_trades_tradable) > 0:
    print(f"총 거래: {len(df_trades_tradable)}개")
    print(f"총 PnL: {df_trades_tradable['pnl_pct'].sum():.2f}%")
    print(f"평균 PnL: {df_trades_tradable['pnl_pct'].mean():.2f}%")
    print(f"승률: {len(df_trades_tradable[df_trades_tradable['pnl_pct'] > 0])/len(df_trades_tradable)*100:.1f}%")
    print("\n결과 분포:")
    print(df_trades_tradable['exit_reason'].value_counts())
    print()
    
    # Power score analysis
    print("Power Score별 성과:")
    for score in sorted(df_trades_tradable['power_score'].unique()):
        score_df = df_trades_tradable[df_trades_tradable['power_score'] == score]
        avg_pnl = score_df['pnl_pct'].mean()
        win_rate = len(score_df[score_df['pnl_pct'] > 0]) / len(score_df) * 100
        print(f"  Score {score}점: {len(score_df)}개 | 평균 {avg_pnl:+.2f}% | 승률 {win_rate:.1f}%")
    print()
else:
    print("⚠️ No tradable opportunities found!")
    print()

# Backtest 2: Tradable + Conditional
print("=" * 80)
print("백테스트 2: Tradable + Conditional (더 많은 거래)")
print("=" * 80)
print()

df_all = pd.concat([tradable, conditional])
df_trades_all = backtest_trades(df_all, df_15m, sl_pct=0.5, tp1_pct=0.5)

if len(df_trades_all) > 0:
    print(f"총 거래: {len(df_trades_all)}개")
    print(f"총 PnL: {df_trades_all['pnl_pct'].sum():.2f}%")
    print(f"평균 PnL: {df_trades_all['pnl_pct'].mean():.2f}%")
    print(f"승률: {len(df_trades_all[df_trades_all['pnl_pct'] > 0])/len(df_trades_all)*100:.1f}%")
    print("\n결과 분포:")
    print(df_trades_all['exit_reason'].value_counts())
    print()
else:
    print("⚠️ No opportunities found!")
    print()

# Backtest 3: TP1 70% exit (더 공격적 익절)
print("=" * 80)
print("백테스트 3: Tradable + TP1 70% 익절 전략")
print("=" * 80)
print()

df_trades_tp70 = backtest_trades(tradable, df_15m, sl_pct=0.5, tp1_pct=0.7)

if len(df_trades_tp70) > 0:
    print(f"총 거래: {len(df_trades_tp70)}개")
    print(f"총 PnL: {df_trades_tp70['pnl_pct'].sum():.2f}%")
    print(f"평균 PnL: {df_trades_tp70['pnl_pct'].mean():.2f}%")
    print(f"승률: {len(df_trades_tp70[df_trades_tp70['pnl_pct'] > 0])/len(df_trades_tp70)*100:.1f}%")
    print("\n결과 분포:")
    print(df_trades_tp70['exit_reason'].value_counts())
    print()

# Backtest 4: Power score 10 only
print("=" * 80)
print("백테스트 4: Power Score 10점만 (최고 품질)")
print("=" * 80)
print()

tradable_10 = tradable[tradable['power_score'] == 10].copy()
df_trades_10 = backtest_trades(tradable_10, df_15m, sl_pct=0.5, tp1_pct=0.5)

if len(df_trades_10) > 0:
    print(f"총 거래: {len(df_trades_10)}개")
    print(f"총 PnL: {df_trades_10['pnl_pct'].sum():.2f}%")
    print(f"평균 PnL: {df_trades_10['pnl_pct'].mean():.2f}%")
    print(f"승률: {len(df_trades_10[df_trades_10['pnl_pct'] > 0])/len(df_trades_10)*100:.1f}%")
    print("\n결과 분포:")
    print(df_trades_10['exit_reason'].value_counts())
    print()
else:
    print("⚠️ No 10-point opportunities found!")
    print()

# Comparison
print("=" * 80)
print("📊 전략 비교")
print("=" * 80)
print()

comparison_data = []

if len(df_trades_tradable) > 0:
    comparison_data.append({
        '전략': 'Tradable (50/50)',
        '거래수': len(df_trades_tradable),
        '총 PnL': f"{df_trades_tradable['pnl_pct'].sum():.2f}%",
        '평균 PnL': f"{df_trades_tradable['pnl_pct'].mean():.2f}%",
        '승률': f"{len(df_trades_tradable[df_trades_tradable['pnl_pct'] > 0])/len(df_trades_tradable)*100:.1f}%",
        'SL 비율': f"{len(df_trades_tradable[df_trades_tradable['exit_reason'] == 'SL'])/len(df_trades_tradable)*100:.1f}%"
    })

if len(df_trades_all) > 0:
    comparison_data.append({
        '전략': 'Tradable+Conditional',
        '거래수': len(df_trades_all),
        '총 PnL': f"{df_trades_all['pnl_pct'].sum():.2f}%",
        '평균 PnL': f"{df_trades_all['pnl_pct'].mean():.2f}%",
        '승률': f"{len(df_trades_all[df_trades_all['pnl_pct'] > 0])/len(df_trades_all)*100:.1f}%",
        'SL 비율': f"{len(df_trades_all[df_trades_all['exit_reason'] == 'SL'])/len(df_trades_all)*100:.1f}%"
    })

if len(df_trades_tp70) > 0:
    comparison_data.append({
        '전략': 'Tradable (70/30)',
        '거래수': len(df_trades_tp70),
        '총 PnL': f"{df_trades_tp70['pnl_pct'].sum():.2f}%",
        '평균 PnL': f"{df_trades_tp70['pnl_pct'].mean():.2f}%",
        '승률': f"{len(df_trades_tp70[df_trades_tp70['pnl_pct'] > 0])/len(df_trades_tp70)*100:.1f}%",
        'SL 비율': f"{len(df_trades_tp70[df_trades_tp70['exit_reason'] == 'SL'])/len(df_trades_tp70)*100:.1f}%"
    })

if len(df_trades_10) > 0:
    comparison_data.append({
        '전략': '10점만 (50/50)',
        '거래수': len(df_trades_10),
        '총 PnL': f"{df_trades_10['pnl_pct'].sum():.2f}%",
        '평균 PnL': f"{df_trades_10['pnl_pct'].mean():.2f}%",
        '승률': f"{len(df_trades_10[df_trades_10['pnl_pct'] > 0])/len(df_trades_10)*100:.1f}%",
        'SL 비율': f"{len(df_trades_10[df_trades_10['exit_reason'] == 'SL'])/len(df_trades_10)*100:.1f}%"
    })

if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print()

# Save results
if len(df_trades_tradable) > 0:
    df_trades_tradable.to_csv('backtest_hlhlhl_proper.csv', index=False)
    print("✅ 백테스트 결과 저장: backtest_hlhlhl_proper.csv")

print()
print("=" * 80)
print("🎯 결론")
print("=" * 80)
print()

if len(df_trades_tradable) > 0:
    if df_trades_tradable['pnl_pct'].sum() > 0:
        print("✅ 이 전략은 수익성이 있습니다!")
        print(f"   총 PnL: {df_trades_tradable['pnl_pct'].sum():.2f}%")
        print(f"   평균 PnL: {df_trades_tradable['pnl_pct'].mean():.2f}%")
    else:
        print("⚠️ 이 전략은 손실입니다.")
        print(f"   총 PnL: {df_trades_tradable['pnl_pct'].sum():.2f}%")
        print("   추가 개선 필요!")

