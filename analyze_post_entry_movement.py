import pandas as pd
import numpy as np
import json

# Load data
signals_df = pd.read_csv('valid_signals.csv')
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

# Load backtest results to get actual trades
with open('backtest_results.json', 'r') as f:
    backtest = json.load(f)

trades = backtest['trades']

def analyze_extended_movement(trade_data):
    """Analyze what happens AFTER exit - does it keep going up?"""
    exit_time = pd.to_datetime(trade_data['exit_time'])
    exit_price = trade_data['exit_price']
    
    # Get candles for next 72 hours (288 candles) after exit
    future_candles = candles_df[candles_df['datetime'] > exit_time].head(288)
    
    if len(future_candles) == 0:
        return None
    
    # Calculate movements after exit
    max_gain_after_exit = ((future_candles['high'].max() - exit_price) / exit_price * 100)
    max_loss_after_exit = ((future_candles['low'].min() - exit_price) / exit_price * 100)
    
    # Check if price continued upward significantly
    continued_up_5pct = max_gain_after_exit >= 5.0
    continued_up_10pct = max_gain_after_exit >= 10.0
    
    return {
        'max_gain_after_exit': max_gain_after_exit,
        'max_loss_after_exit': max_loss_after_exit,
        'continued_up_5pct': continued_up_5pct,
        'continued_up_10pct': continued_up_10pct,
    }

def check_if_in_uptrend(entry_time, entry_price):
    """Check if entry was made during an existing uptrend"""
    entry_dt = pd.to_datetime(entry_time)
    
    # Get previous 96 candles (24 hours before entry)
    prev_candles = candles_df[candles_df['datetime'] < entry_dt].tail(96)
    
    if len(prev_candles) < 20:
        return None
    
    # Calculate if price was already in uptrend
    first_price = prev_candles.iloc[0]['close']
    last_price = prev_candles.iloc[-1]['close']
    trend_pct = (last_price - first_price) / first_price * 100
    
    # Calculate moving averages
    ma_short = prev_candles.tail(20)['close'].mean()
    ma_long = prev_candles['close'].mean()
    
    in_uptrend = (trend_pct > 2.0) and (ma_short > ma_long)
    
    return {
        'pre_entry_trend_pct': trend_pct,
        'ma_short': ma_short,
        'ma_long': ma_long,
        'in_uptrend': in_uptrend
    }

# Analyze all trades
results_by_exit = {
    'TP': [],
    'RETEST': [],
    'TIME': [],
    'SL': []
}

for trade in trades:
    exit_reason = trade['exit_reason']
    
    # Check post-exit movement
    post_move = analyze_extended_movement(trade)
    
    # Check pre-entry trend
    pre_trend = check_if_in_uptrend(trade['entry_time'], trade['entry_price'])
    
    if post_move and pre_trend:
        results_by_exit[exit_reason].append({
            'trade_id': trade.get('trade_id', 0),
            'entry_time': trade['entry_time'],
            'exit_reason': exit_reason,
            'pnl': trade['pnl'],
            **post_move,
            **pre_trend
        })

# Aggregate statistics
print("=" * 80)
print("POST-EXIT MOVEMENT ANALYSIS")
print("=" * 80)

for exit_reason, data in results_by_exit.items():
    if len(data) == 0:
        continue
    
    df = pd.DataFrame(data)
    
    print(f"\n{exit_reason} ({len(df)} trades):")
    print(f"  Pre-entry in uptrend: {df['in_uptrend'].sum()} ({df['in_uptrend'].mean()*100:.1f}%)")
    print(f"  Avg pre-entry trend: {df['pre_entry_trend_pct'].mean():.2f}%")
    print(f"  ")
    print(f"  Post-exit continued +5%: {df['continued_up_5pct'].sum()} ({df['continued_up_5pct'].mean()*100:.1f}%)")
    print(f"  Post-exit continued +10%: {df['continued_up_10pct'].sum()} ({df['continued_up_10pct'].mean()*100:.1f}%)")
    print(f"  Avg max gain after exit: {df['max_gain_after_exit'].mean():.2f}%")
    print(f"  Avg max loss after exit: {df['max_loss_after_exit'].mean():.2f}%")

# Special analysis for RETEST and TIME (early exits)
print("\n" + "=" * 80)
print("MISSED OPPORTUNITY ANALYSIS (RETEST + TIME exits)")
print("=" * 80)

early_exits = results_by_exit['RETEST'] + results_by_exit['TIME']
if len(early_exits) > 0:
    df_early = pd.DataFrame(early_exits)
    
    print(f"\nTotal early exits: {len(df_early)}")
    print(f"Pre-entry already in uptrend: {df_early['in_uptrend'].sum()} ({df_early['in_uptrend'].mean()*100:.1f}%)")
    print(f"")
    print(f"If held longer (not exited early):")
    print(f"  Could have gained +5%:  {df_early['continued_up_5pct'].sum()} ({df_early['continued_up_5pct'].mean()*100:.1f}%)")
    print(f"  Could have gained +10%: {df_early['continued_up_10pct'].sum()} ({df_early['continued_up_10pct'].mean()*100:.1f}%)")
    print(f"  Avg potential gain: {df_early['max_gain_after_exit'].mean():.2f}%")
    print(f"  Actual avg PnL: {df_early['pnl'].mean():.2f}%")
    print(f"  Opportunity cost: {df_early['max_gain_after_exit'].mean() - df_early['pnl'].mean():.2f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

all_trades = []
for data_list in results_by_exit.values():
    all_trades.extend(data_list)

if len(all_trades) > 0:
    df_all = pd.DataFrame(all_trades)
    
    pct_in_uptrend = df_all['in_uptrend'].mean() * 100
    pct_continued = df_all['continued_up_5pct'].mean() * 100
    
    print(f"\nOverall Statistics:")
    print(f"  Entries made during existing uptrend: {pct_in_uptrend:.1f}%")
    print(f"  Price continued up +5% after exit: {pct_continued:.1f}%")
    print(f"  Average pre-entry trend: {df_all['pre_entry_trend_pct'].mean():.2f}%")
    print(f"  Average post-exit potential: {df_all['max_gain_after_exit'].mean():.2f}%")
    
    if pct_in_uptrend > 70:
        print(f"\n⚠️  WARNING: {pct_in_uptrend:.1f}% of entries are in existing uptrends!")
        print(f"    → Not catching reversals, just riding existing momentum")
    
    if pct_continued > 60:
        print(f"\n⚠️  WARNING: {pct_continued:.1f}% continued up significantly after exit!")
        print(f"    → Exiting too early, missing major gains")

