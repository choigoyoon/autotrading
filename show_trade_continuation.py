import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta

# Load data
signals_df = pd.read_csv('valid_signals.csv')
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

# Backtest parameters
TP = 5.0  # Take profit 5%
SL = 2.0  # Stop loss 2%
MAX_HOLD_HOURS = 24

def backtest_trade(signal):
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['price']
    trendline_price = signal['trendline_price']
    
    # Get future candles
    future = candles_df[candles_df['datetime'] > entry_time].head(MAX_HOLD_HOURS * 4)  # 15분봉
    
    if len(future) == 0:
        return None
    
    exit_reason = None
    exit_time = None
    exit_price = None
    pnl = 0
    
    for i, (idx, candle) in enumerate(future.iterrows()):
        # Check retest
        if candle['low'] <= trendline_price:
            exit_reason = 'RETEST'
            exit_time = candle['datetime']
            exit_price = trendline_price
            pnl = ((exit_price - entry_price) / entry_price) * 100
            break
        
        # Check TP
        if candle['high'] >= entry_price * (1 + TP/100):
            exit_reason = 'TP'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 + TP/100)
            pnl = TP
            break
        
        # Check SL
        if candle['low'] <= entry_price * (1 - SL/100):
            exit_reason = 'SL'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 - SL/100)
            pnl = -SL
            break
    
    # Time limit
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
        pnl = ((exit_price - entry_price) / entry_price) * 100
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'trendline_price': trendline_price
    }

# Run backtest
trades = []
for idx, signal in signals_df.iterrows():
    result = backtest_trade(signal)
    if result:
        trades.append({**signal.to_dict(), **result})

trades_df = pd.DataFrame(trades)

# Select examples
tp_trades = trades_df[trades_df['exit_reason'] == 'TP'].head(2)
retest_trades = trades_df[trades_df['exit_reason'] == 'RETEST'].head(2)
time_trades = trades_df[trades_df['exit_reason'] == 'TIME'].head(2)

example_trades = pd.concat([tp_trades, retest_trades, time_trades])

# Visualize with extended view
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
axes = axes.flatten()

for i, (idx, trade) in enumerate(example_trades.iterrows()):
    ax = axes[i]
    
    h1_time = pd.to_datetime(trade['h1_time'])
    h2_time = pd.to_datetime(trade['h2_time'])
    entry_time = pd.to_datetime(trade['entry_time'])
    exit_time = pd.to_datetime(trade['exit_time'])
    
    # Get extended view: 24h before entry + 72h after exit
    start_time = entry_time - timedelta(hours=24)
    end_time = exit_time + timedelta(hours=72)  # Extended to see what happened after
    
    view_candles = candles_df[
        (candles_df['datetime'] >= start_time) & 
        (candles_df['datetime'] <= end_time)
    ]
    
    if len(view_candles) == 0:
        continue
    
    # Plot candles
    for j, (_, candle) in enumerate(view_candles.iterrows()):
        color = 'green' if candle['close'] > candle['open'] else 'red'
        ax.plot([j, j], [candle['low'], candle['high']], color=color, linewidth=0.5)
        ax.plot([j, j], [candle['open'], candle['close']], color=color, linewidth=2)
    
    # Plot trendline
    h1_idx = view_candles[view_candles['datetime'] == h1_time].index
    h2_idx = view_candles[view_candles['datetime'] == h2_time].index
    entry_idx = view_candles[view_candles['datetime'] == entry_time].index
    exit_idx = view_candles[view_candles['datetime'] == exit_time].index
    
    if len(h1_idx) > 0 and len(h2_idx) > 0:
        h1_pos = view_candles.index.get_loc(h1_idx[0])
        h2_pos = view_candles.index.get_loc(h2_idx[0])
        
        h1_price = trade['h1_price']
        h2_price = trade['h2_price']
        
        # Extend trendline
        slope = (h2_price - h1_price) / (h2_pos - h1_pos)
        x_line = np.arange(h1_pos, len(view_candles))
        y_line = h1_price + slope * (x_line - h1_pos)
        ax.plot(x_line, y_line, 'b--', linewidth=1.5, label='Trendline')
    
    # Mark key points
    if len(entry_idx) > 0:
        entry_pos = view_candles.index.get_loc(entry_idx[0])
        ax.scatter([entry_pos], [trade['entry_price']], color='blue', s=100, zorder=5, label='Entry')
        ax.axvline(entry_pos, color='blue', linestyle=':', alpha=0.5)
    
    if len(exit_idx) > 0:
        exit_pos = view_candles.index.get_loc(exit_idx[0])
        exit_color = 'green' if trade['pnl'] > 0 else 'red'
        ax.scatter([exit_pos], [trade['exit_price']], color=exit_color, s=100, zorder=5, label='Exit')
        ax.axvline(exit_pos, color=exit_color, linestyle=':', alpha=0.5)
        
        # Show what happened AFTER exit
        post_exit_candles = view_candles.iloc[exit_pos+1:]
        if len(post_exit_candles) > 0:
            max_after = post_exit_candles['high'].max()
            min_after = post_exit_candles['low'].min()
            
            # Calculate potential missed gain
            potential_gain = ((max_after - trade['exit_price']) / trade['exit_price']) * 100
            potential_loss = ((min_after - trade['exit_price']) / trade['exit_price']) * 100
            
            # Highlight post-exit area
            rect = patches.Rectangle((exit_pos, view_candles['low'].min()), 
                                     len(post_exit_candles), 
                                     view_candles['high'].max() - view_candles['low'].min(),
                                     linewidth=0, edgecolor='none', facecolor='yellow', alpha=0.1)
            ax.add_patch(rect)
            
            ax.text(exit_pos + len(post_exit_candles)//2, view_candles['high'].max() * 0.98,
                   f'After: +{potential_gain:.1f}% / {potential_loss:.1f}%',
                   ha='center', fontsize=9, color='orange', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Title with extended info
    title = f"{trade['exit_reason']} - Entry: {trade['entry_price']:.0f}, Exit: {trade['exit_price']:.0f}, PnL: {trade['pnl']:.2f}%\n"
    title += f"Entry Time: {entry_time.strftime('%m-%d %H:%M')}, Exit: {exit_time.strftime('%m-%d %H:%M')}"
    ax.set_title(title, fontsize=10)
    
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (candles)', fontsize=8)
    ax.set_ylabel('Price', fontsize=8)

plt.tight_layout()
plt.savefig('trade_examples_extended.png', dpi=150, bbox_inches='tight')
print("Saved: trade_examples_extended.png")

# Statistics
print("\n" + "=" * 80)
print("POST-EXIT MOVEMENT STATISTICS")
print("=" * 80)

for exit_reason in ['TP', 'RETEST', 'TIME']:
    subset = trades_df[trades_df['exit_reason'] == exit_reason]
    if len(subset) == 0:
        continue
    
    # Analyze post-exit movement
    post_gains = []
    post_losses = []
    continued_5pct = 0
    continued_10pct = 0
    
    for idx, trade in subset.iterrows():
        exit_time = pd.to_datetime(trade['exit_time'])
        exit_price = trade['exit_price']
        
        future = candles_df[candles_df['datetime'] > exit_time].head(288)  # 72 hours
        if len(future) > 0:
            max_gain = ((future['high'].max() - exit_price) / exit_price * 100)
            max_loss = ((future['low'].min() - exit_price) / exit_price * 100)
            
            post_gains.append(max_gain)
            post_losses.append(max_loss)
            
            if max_gain >= 5.0:
                continued_5pct += 1
            if max_gain >= 10.0:
                continued_10pct += 1
    
    print(f"\n{exit_reason} ({len(subset)} trades):")
    print(f"  Avg PnL at exit: {subset['pnl'].mean():.2f}%")
    if post_gains:
        print(f"  Avg max gain after exit (72h): {np.mean(post_gains):.2f}%")
        print(f"  Avg max loss after exit (72h): {np.mean(post_losses):.2f}%")
        print(f"  Continued to +5% after exit: {continued_5pct} ({continued_5pct/len(post_gains)*100:.1f}%)")
        print(f"  Continued to +10% after exit: {continued_10pct} ({continued_10pct/len(post_gains)*100:.1f}%)")

