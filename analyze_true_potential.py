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
TP = 5.0
SL = 2.0
MAX_HOLD_HOURS = 24

def backtest_trade(signal):
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    trendline_price = signal['trendline_price']
    
    future = candles_df[candles_df['datetime'] > entry_time].head(MAX_HOLD_HOURS * 4)
    
    if len(future) == 0:
        return None
    
    exit_reason = None
    exit_time = None
    exit_price = None
    pnl = 0
    
    for i, (idx, candle) in enumerate(future.iterrows()):
        if candle['low'] <= trendline_price:
            exit_reason = 'RETEST'
            exit_time = candle['datetime']
            exit_price = trendline_price
            pnl = ((exit_price - entry_price) / entry_price) * 100
            break
        
        if candle['high'] >= entry_price * (1 + TP/100):
            exit_reason = 'TP'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 + TP/100)
            pnl = TP
            break
        
        if candle['low'] <= entry_price * (1 - SL/100):
            exit_reason = 'SL'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 - SL/100)
            pnl = -SL
            break
    
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
        pnl = ((exit_price - entry_price) / entry_price) * 100
    
    # Calculate what happened in next 72 hours AFTER exit
    post_exit_candles = candles_df[candles_df['datetime'] > exit_time].head(288)  # 72시간
    
    if len(post_exit_candles) > 0:
        max_after = post_exit_candles['high'].max()
        min_after = post_exit_candles['low'].min()
        potential_gain = ((max_after - exit_price) / exit_price) * 100
        potential_loss = ((min_after - exit_price) / exit_price) * 100
        continued_5pct = potential_gain >= 5.0
        continued_10pct = potential_gain >= 10.0
    else:
        potential_gain = 0
        potential_loss = 0
        continued_5pct = False
        continued_10pct = False
    
    # Check if we were in uptrend before entry
    pre_entry_candles = candles_df[candles_df['datetime'] < entry_time].tail(96)  # 24시간 전
    if len(pre_entry_candles) >= 20:
        first_price = pre_entry_candles.iloc[0]['close']
        last_price = pre_entry_candles.iloc[-1]['close']
        pre_trend_pct = (last_price - first_price) / first_price * 100
        ma_short = pre_entry_candles.tail(20)['close'].mean()
        ma_long = pre_entry_candles['close'].mean()
        in_uptrend = (pre_trend_pct > 2.0) and (ma_short > ma_long)
    else:
        pre_trend_pct = 0
        in_uptrend = False
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'trendline_price': trendline_price,
        'potential_gain_after': potential_gain,
        'potential_loss_after': potential_loss,
        'continued_5pct': continued_5pct,
        'continued_10pct': continued_10pct,
        'pre_trend_pct': pre_trend_pct,
        'in_uptrend': in_uptrend
    }

# Run backtest
trades = []
for idx, signal in signals_df.iterrows():
    result = backtest_trade(signal)
    if result:
        trades.append({**signal.to_dict(), **result})

trades_df = pd.DataFrame(trades)

# Print statistics
print("=" * 80)
print("진실의 순간: 출구 이후에 실제로 무슨 일이 벌어졌나?")
print("=" * 80)

for exit_reason in ['TP', 'RETEST', 'TIME', 'SL']:
    subset = trades_df[trades_df['exit_reason'] == exit_reason]
    if len(subset) == 0:
        continue
    
    print(f"\n{exit_reason} ({len(subset)}건, {len(subset)/len(trades_df)*100:.1f}%):")
    print(f"  청산 시 평균 수익: {subset['pnl'].mean():.2f}%")
    print(f"  ")
    print(f"  진입 전 이미 상승추세였음: {subset['in_uptrend'].sum()}건 ({subset['in_uptrend'].mean()*100:.1f}%)")
    print(f"  평균 진입 전 추세: {subset['pre_trend_pct'].mean():.2f}%")
    print(f"  ")
    print(f"  청산 후 72시간 동안:")
    print(f"    평균 최대 상승 가능성: {subset['potential_gain_after'].mean():.2f}%")
    print(f"    평균 최대 하락 가능성: {subset['potential_loss_after'].mean():.2f}%")
    print(f"    +5% 이상 더 상승: {subset['continued_5pct'].sum()}건 ({subset['continued_5pct'].mean()*100:.1f}%)")
    print(f"    +10% 이상 더 상승: {subset['continued_10pct'].sum()}건 ({subset['continued_10pct'].mean()*100:.1f}%)")
    
    # 조기 청산으로 인한 기회비용
    if exit_reason in ['RETEST', 'TIME']:
        opportunity_cost = subset['potential_gain_after'].mean() - subset['pnl'].mean()
        print(f"  ")
        print(f"  ⚠️ 기회비용: {opportunity_cost:.2f}% (더 기다렸으면 벌 수 있었던 평균 수익)")

print("\n" + "=" * 80)
print("전체 통계")
print("=" * 80)

print(f"\n총 거래: {len(trades_df)}건")
print(f"진입 시 이미 상승추세: {trades_df['in_uptrend'].sum()}건 ({trades_df['in_uptrend'].mean()*100:.1f}%)")
print(f"평균 진입 전 추세: {trades_df['pre_trend_pct'].mean():.2f}%")
print(f"")
print(f"청산 후 계속 상승:")
print(f"  +5% 이상: {trades_df['continued_5pct'].sum()}건 ({trades_df['continued_5pct'].mean()*100:.1f}%)")
print(f"  +10% 이상: {trades_df['continued_10pct'].sum()}건 ({trades_df['continued_10pct'].mean()*100:.1f}%)")
print(f"")
print(f"평균 청산 시 수익: {trades_df['pnl'].mean():.2f}%")
print(f"평균 청산 후 추가 상승 가능성: {trades_df['potential_gain_after'].mean():.2f}%")
print(f"평균 총 잠재 수익 (청산 수익 + 청산 후 상승): {(trades_df['pnl'] + trades_df['potential_gain_after']).mean():.2f}%")

# 경고
pct_in_uptrend = trades_df['in_uptrend'].mean() * 100
pct_continued = trades_df['continued_5pct'].mean() * 100

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if pct_in_uptrend > 70:
    print(f"\n⚠️  경고 1: 진입의 {pct_in_uptrend:.1f}%가 이미 상승추세 중!")
    print(f"    → 추세 반전을 잡는 게 아니라, 그냥 상승 모멘텀을 타는 것")
    print(f"    → 우상향 자리에서 진입하고 있다는 지적이 맞습니다")

if pct_continued > 60:
    print(f"\n⚠️  경고 2: 청산 후 {pct_continued:.1f}%가 +5% 이상 더 상승!")
    print(f"    → 너무 일찍 나가고 있음")
    print(f"    → Case #3처럼 더 올라갈 여지가 많음")

# 시각화: 가장 극단적인 케이스들
print("\n" + "=" * 80)
print("차트 생성 중...")
print("=" * 80)

# 3가지 케이스 선택
# 1. 조기 청산 후 폭등한 케이스
early_exit_missed = trades_df[
    (trades_df['exit_reason'].isin(['RETEST', 'TIME'])) & 
    (trades_df['potential_gain_after'] > 10)
].nlargest(2, 'potential_gain_after')

# 2. TP 달성한 케이스
tp_cases = trades_df[trades_df['exit_reason'] == 'TP'].head(2)

example_trades = pd.concat([tp_cases, early_exit_missed])

if len(example_trades) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (idx, trade) in enumerate(example_trades.head(4).iterrows()):
        ax = axes[i]
        
        h1_time = pd.to_datetime(trade['h1_time'])
        h2_time = pd.to_datetime(trade['h2_time'])
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        
        start_time = entry_time - timedelta(hours=24)
        end_time = exit_time + timedelta(hours=72)
        
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
            
            slope = (h2_price - h1_price) / (h2_pos - h1_pos)
            x_line = np.arange(h1_pos, len(view_candles))
            y_line = h1_price + slope * (x_line - h1_pos)
            ax.plot(x_line, y_line, 'b--', linewidth=1.5, label='Trendline', alpha=0.7)
        
        # Mark entry and exit
        if len(entry_idx) > 0:
            entry_pos = view_candles.index.get_loc(entry_idx[0])
            ax.scatter([entry_pos], [trade['entry_price']], color='blue', s=150, zorder=5, label='Entry', marker='^')
            ax.axvline(entry_pos, color='blue', linestyle=':', alpha=0.5)
        
        if len(exit_idx) > 0:
            exit_pos = view_candles.index.get_loc(exit_idx[0])
            exit_color = 'green' if trade['pnl'] > 0 else 'red'
            ax.scatter([exit_pos], [trade['exit_price']], color=exit_color, s=150, zorder=5, label=f'Exit ({trade["exit_reason"]})', marker='v')
            ax.axvline(exit_pos, color=exit_color, linestyle=':', alpha=0.5)
            
            # Highlight post-exit zone
            post_exit_candles = view_candles.iloc[exit_pos+1:]
            if len(post_exit_candles) > 0:
                rect = patches.Rectangle((exit_pos, view_candles['low'].min()), 
                                         len(post_exit_candles), 
                                         view_candles['high'].max() - view_candles['low'].min(),
                                         linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.15, linestyle='--')
                ax.add_patch(rect)
                
                ax.text(exit_pos + len(post_exit_candles)//2, view_candles['high'].max() * 0.96,
                       f'청산 후: +{trade["potential_gain_after"]:.1f}% 더 상승 가능했음',
                       ha='center', fontsize=10, color='red', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Title
        title = f"{trade['exit_reason']} - 실제수익: {trade['pnl']:.2f}%, 잠재수익: +{trade['potential_gain_after']:.1f}%\n"
        title += f"진입 전 추세: {trade['pre_trend_pct']:.1f}% ({'상승추세' if trade['in_uptrend'] else '정상'})"
        ax.set_title(title, fontsize=11, weight='bold')
        
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (15분봉)', fontsize=9)
        ax.set_ylabel('Price', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('truth_revealed.png', dpi=150, bbox_inches='tight')
    print("\n✅ 차트 저장: truth_revealed.png")

