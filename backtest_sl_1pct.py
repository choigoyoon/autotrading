import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data
print("Loading data...")
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print(f"Loaded {len(df_15m)} candles")
print()

# Find swing highs and lows
def find_swing_points(df, window=20):
    highs = []
    lows = []
    
    for i in range(window, len(df) - window):
        # Swing High
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({
                'datetime': df.loc[i, 'datetime'],
                'price': df.loc[i, 'high'],
                'type': 'H',
                'index': i
            })
        
        # Swing Low
        if df.loc[i, 'low'] == df.loc[i-window:i+window+1, 'low'].min():
            lows.append({
                'datetime': df.loc[i, 'datetime'],
                'price': df.loc[i, 'low'],
                'type': 'L',
                'index': i
            })
    
    return highs, lows

print("Finding swing points...")
swing_highs, swing_lows = find_swing_points(df_15m)
print(f"Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
print()

# Find downtrend lines (H1 > H2 > H3)
def find_downtrend_lines(highs):
    lines = []
    for i in range(len(highs) - 2):
        h1 = highs[i]
        h2 = highs[i + 1]
        h3 = highs[i + 2]
        
        if h1['price'] > h2['price'] > h3['price']:
            lines.append({
                'h1': h1,
                'h2': h2,
                'h3': h3
            })
    return lines

print("Finding downtrend lines...")
downtrend_lines = find_downtrend_lines(swing_highs)
print(f"Found {len(downtrend_lines)} downtrend lines")
print()

# Backtest with SL -1.0%
def backtest_strategy(df, downtrend_lines, sl_pct=1.0):
    trades = []
    
    for line in downtrend_lines:
        h1_time = line['h1']['datetime']
        h2_time = line['h2']['datetime']
        h3_time = line['h3']['datetime']
        h3_price = line['h3']['price']
        
        # Check for breakout
        breakout_df = df[df['datetime'] > h3_time].copy()
        if len(breakout_df) == 0:
            continue
        
        # Find H3 trendline breakout
        breakout_candle = None
        for idx, row in breakout_df.iterrows():
            if row['close'] > h3_price:
                breakout_candle = row
                break
        
        if breakout_candle is None:
            continue
        
        # Wait for retest (price comes back to H3 ±1%)
        retest_df = df[df['datetime'] > breakout_candle['datetime']].head(20)
        retest_found = False
        entry_candle = None
        
        for idx, row in retest_df.iterrows():
            # Check if price retests H3
            if abs(row['low'] - h3_price) / h3_price <= 0.01:
                # Check if next 5 candles stay above H3
                future_df = df[df['datetime'] > row['datetime']].head(5)
                if len(future_df) >= 5:
                    avg_close = future_df['close'].mean()
                    if avg_close > h3_price:
                        retest_found = True
                        entry_candle = future_df.iloc[0]
                        break
        
        if not retest_found or entry_candle is None:
            continue
        
        # Calculate power score
        power_score = 5  # Base score
        
        # Check BB tear (price near upper band)
        # Simplified: if entry close > entry open * 1.01
        if entry_candle['close'] > entry_candle['open'] * 1.01:
            power_score += 3
        
        # Skip if power_score < 5
        if power_score < 5:
            continue
        
        # Entry parameters
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']
        
        # Stop Loss: H3 - sl_pct%
        sl_price = h3_price * (1 - sl_pct/100)
        
        # Take Profit 1: H2
        tp1_price = line['h2']['price']
        
        # Take Profit 2: H1
        tp2_price = line['h1']['price']
        
        # Calculate R:R
        risk = entry_price - sl_price
        reward_tp1 = tp1_price - entry_price
        reward_tp2 = tp2_price - entry_price
        
        rr_tp1 = reward_tp1 / risk if risk > 0 else 0
        rr_tp2 = reward_tp2 / risk if risk > 0 else 0
        
        # Track trade outcome
        future_candles = df[df['datetime'] > entry_time].head(200)  # Check next 200 candles
        
        exit_reason = None
        pnl_pct = 0
        exit_time = None
        
        # Balanced strategy: 50% at TP1, move SL to breakeven, 50% at TP2
        tp1_hit = False
        sl_moved_to_breakeven = False
        
        for idx, candle in future_candles.iterrows():
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
                    pnl_pct = (tp1_price - entry_price) / entry_price * 100 * 0.5  # 50% at TP1
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
                # 50% at TP1 + 50% at TP2
                pnl_tp1 = (tp1_price - entry_price) / entry_price * 100 * 0.5
                pnl_tp2 = (tp2_price - entry_price) / entry_price * 100 * 0.5
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
            'h1_price': line['h1']['price'],
            'h2_price': line['h2']['price'],
            'h3_price': h3_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'power_score': power_score,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct,
            'rr_tp1': rr_tp1,
            'rr_tp2': rr_tp2,
            'year': entry_time.year
        })
    
    return trades

# Run backtest with SL -0.5% (original)
print("=" * 80)
print("Backtest 1: SL = H3 - 0.5% (Original)")
print("=" * 80)
trades_05 = backtest_strategy(df_15m, downtrend_lines, sl_pct=0.5)
df_trades_05 = pd.DataFrame(trades_05)

print(f"\n총 거래: {len(trades_05)}개")
print(f"총 PnL: {df_trades_05['pnl_pct'].sum():.2f}%")
print(f"평균 PnL: {df_trades_05['pnl_pct'].mean():.2f}%")
print(f"승률: {len(df_trades_05[df_trades_05['pnl_pct'] > 0])/len(df_trades_05)*100:.1f}%")
print("\n결과 분포:")
print(df_trades_05['exit_reason'].value_counts())
print()

# Run backtest with SL -1.0% (new)
print("=" * 80)
print("Backtest 2: SL = H3 - 1.0% (New)")
print("=" * 80)
trades_10 = backtest_strategy(df_15m, downtrend_lines, sl_pct=1.0)
df_trades_10 = pd.DataFrame(trades_10)

print(f"\n총 거래: {len(trades_10)}개")
print(f"총 PnL: {df_trades_10['pnl_pct'].sum():.2f}%")
print(f"평균 PnL: {df_trades_10['pnl_pct'].mean():.2f}%")
print(f"승률: {len(df_trades_10[df_trades_10['pnl_pct'] > 0])/len(df_trades_10)*100:.1f}%")
print("\n결과 분포:")
print(df_trades_10['exit_reason'].value_counts())
print()

# Run backtest with SL -1.5%
print("=" * 80)
print("Backtest 3: SL = H3 - 1.5% (Extra Safe)")
print("=" * 80)
trades_15 = backtest_strategy(df_15m, downtrend_lines, sl_pct=1.5)
df_trades_15 = pd.DataFrame(trades_15)

print(f"\n총 거래: {len(trades_15)}개")
print(f"총 PnL: {df_trades_15['pnl_pct'].sum():.2f}%")
print(f"평균 PnL: {df_trades_15['pnl_pct'].mean():.2f}%")
print(f"승률: {len(df_trades_15[df_trades_15['pnl_pct'] > 0])/len(df_trades_15)*100:.1f}%")
print("\n결과 분포:")
print(df_trades_15['exit_reason'].value_counts())
print()

# Comparison
print("=" * 80)
print("📊 비교 분석")
print("=" * 80)
print()

comparison = pd.DataFrame([
    {
        'SL Setting': 'H3 - 0.5%',
        'Total Trades': len(trades_05),
        'Total PnL': f"{df_trades_05['pnl_pct'].sum():.2f}%",
        'Avg PnL': f"{df_trades_05['pnl_pct'].mean():.2f}%",
        'Win Rate': f"{len(df_trades_05[df_trades_05['pnl_pct'] > 0])/len(df_trades_05)*100:.1f}%",
        'SL Rate': f"{len(df_trades_05[df_trades_05['exit_reason'] == 'SL'])/len(df_trades_05)*100:.1f}%",
        'TP2 Rate': f"{len(df_trades_05[df_trades_05['exit_reason'] == 'TP2_Full'])/len(df_trades_05)*100:.1f}%"
    },
    {
        'SL Setting': 'H3 - 1.0%',
        'Total Trades': len(trades_10),
        'Total PnL': f"{df_trades_10['pnl_pct'].sum():.2f}%",
        'Avg PnL': f"{df_trades_10['pnl_pct'].mean():.2f}%",
        'Win Rate': f"{len(df_trades_10[df_trades_10['pnl_pct'] > 0])/len(df_trades_10)*100:.1f}%",
        'SL Rate': f"{len(df_trades_10[df_trades_10['exit_reason'] == 'SL'])/len(df_trades_10)*100:.1f}%",
        'TP2 Rate': f"{len(df_trades_10[df_trades_10['exit_reason'] == 'TP2_Full'])/len(df_trades_10)*100:.1f}%"
    },
    {
        'SL Setting': 'H3 - 1.5%',
        'Total Trades': len(trades_15),
        'Total PnL': f"{df_trades_15['pnl_pct'].sum():.2f}%",
        'Avg PnL': f"{df_trades_15['pnl_pct'].mean():.2f}%",
        'Win Rate': f"{len(df_trades_15[df_trades_15['pnl_pct'] > 0])/len(df_trades_15)*100:.1f}%",
        'SL Rate': f"{len(df_trades_15[df_trades_15['exit_reason'] == 'SL'])/len(df_trades_15)*100:.1f}%",
        'TP2 Rate': f"{len(df_trades_15[df_trades_15['exit_reason'] == 'TP2_Full'])/len(df_trades_15)*100:.1f}%"
    }
])

print(comparison.to_string(index=False))
print()

# Calculate improvements
print("=" * 80)
print("📈 개선 효과")
print("=" * 80)
print()

original_total_pnl = df_trades_05['pnl_pct'].sum()
new_10_total_pnl = df_trades_10['pnl_pct'].sum()
new_15_total_pnl = df_trades_15['pnl_pct'].sum()

original_avg_pnl = df_trades_05['pnl_pct'].mean()
new_10_avg_pnl = df_trades_10['pnl_pct'].mean()
new_15_avg_pnl = df_trades_15['pnl_pct'].mean()

original_win_rate = len(df_trades_05[df_trades_05['pnl_pct'] > 0])/len(df_trades_05)*100
new_10_win_rate = len(df_trades_10[df_trades_10['pnl_pct'] > 0])/len(df_trades_10)*100
new_15_win_rate = len(df_trades_15[df_trades_15['pnl_pct'] > 0])/len(df_trades_15)*100

print("SL -0.5% → -1.0%:")
print(f"  총 PnL: {original_total_pnl:.2f}% → {new_10_total_pnl:.2f}% ({new_10_total_pnl - original_total_pnl:+.2f}%p)")
print(f"  평균 PnL: {original_avg_pnl:.2f}% → {new_10_avg_pnl:.2f}% ({new_10_avg_pnl - original_avg_pnl:+.2f}%p)")
print(f"  승률: {original_win_rate:.1f}% → {new_10_win_rate:.1f}% ({new_10_win_rate - original_win_rate:+.1f}%p)")
print()

print("SL -0.5% → -1.5%:")
print(f"  총 PnL: {original_total_pnl:.2f}% → {new_15_total_pnl:.2f}% ({new_15_total_pnl - original_total_pnl:+.2f}%p)")
print(f"  평균 PnL: {original_avg_pnl:.2f}% → {new_15_avg_pnl:.2f}% ({new_15_avg_pnl - original_avg_pnl:+.2f}%p)")
print(f"  승률: {original_win_rate:.1f}% → {new_15_win_rate:.1f}% ({new_15_win_rate - original_win_rate:+.1f}%p)")
print()

# Save results
df_trades_10.to_csv('backtest_sl_1pct_results.csv', index=False)
print("✅ SL -1.0% 백테스트 결과 저장: backtest_sl_1pct_results.csv")

df_trades_15.to_csv('backtest_sl_1_5pct_results.csv', index=False)
print("✅ SL -1.5% 백테스트 결과 저장: backtest_sl_1_5pct_results.csv")

# Best recommendation
print()
print("=" * 80)
print("🎯 최종 권장사항")
print("=" * 80)
print()

best_sl = None
if new_10_total_pnl > original_total_pnl and new_10_total_pnl >= new_15_total_pnl:
    best_sl = "H3 - 1.0%"
elif new_15_total_pnl > original_total_pnl:
    best_sl = "H3 - 1.5%"
else:
    best_sl = "H3 - 0.5% (original)"

print(f"✅ 최적 SL 설정: {best_sl}")
print()

