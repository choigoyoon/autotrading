"""
백테스트 수익곡선 및 통계 시각화
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 80)
print("📈 백테스트 수익곡선 시각화")
print("=" * 80)

def detect_fvg_signals(df_4h):
    signals = []
    for i in range(2, len(df_4h)):
        if df_4h['low'].iloc[i] > df_4h['high'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i]
            gap_bottom = df_4h['high'].iloc[i-2]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bull', 'direction': 'long',
                    'signal_time': df_4h['timestamp'].iloc[i],
                    'entry_zone': gap_top, 'gap_size': gap_size
                })
        if df_4h['high'].iloc[i] < df_4h['low'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i-2]
            gap_bottom = df_4h['high'].iloc[i]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bear', 'direction': 'short',
                    'signal_time': df_4h['timestamp'].iloc[i],
                    'entry_zone': gap_bottom, 'gap_size': gap_size
                })
    return signals

def detect_orderblock_signals(df_4h):
    signals = []
    for i in range(3, len(df_4h)):
        if (df_4h['close'].iloc[i-2] < df_4h['open'].iloc[i-2] and
            df_4h['close'].iloc[i-1] > df_4h['open'].iloc[i-1] and
            df_4h['close'].iloc[i-1] > df_4h['high'].iloc[i-2]):
            signals.append({
                'type': 'orderblock', 'direction': 'long',
                'signal_time': df_4h['timestamp'].iloc[i-1],
                'entry_zone': df_4h['high'].iloc[i-2],
            })
    return signals

def simulate(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False):
    trades = []
    pending_signals = []
    current_position = None
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    for i, candle in df_15m.iterrows():
        candle_time = candle['timestamp']
        
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=5)
            pending_signals.append({**sig, 'expire_time': expire_time})
            signal_idx += 1
        
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        if current_position is not None:
            pos = current_position
            bars_elapsed = (candle_time - pos['entry_time']).total_seconds() / 900
            closed = False
            
            if pos['direction'] == 'long':
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5
                
                tp2_price = pos['entry_price'] * (1 + tp2)
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    if use_partial:
                        if pos['tp1_hit']:
                            pos['pnl'] = pos.get('partial_pnl', 0)
                        else:
                            pos['pnl'] = sl * 100
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    if use_partial:
                        if pos['tp1_hit']:
                            pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
                        else:
                            pos['pnl'] = time_pnl
                    else:
                        pos['pnl'] = time_pnl
                    closed = True
            else:
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 - tp1)
                    if candle['low'] <= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5
                
                tp2_price = pos['entry_price'] * (1 - tp2)
                if candle['low'] <= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    if use_partial:
                        if pos['tp1_hit']:
                            pos['pnl'] = pos.get('partial_pnl', 0)
                        else:
                            pos['pnl'] = sl * 100
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    if use_partial:
                        if pos['tp1_hit']:
                            pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
                        else:
                            pos['pnl'] = time_pnl
                    else:
                        pos['pnl'] = time_pnl
                    closed = True
            
            if closed:
                trades.append(pos)
                current_position = None
        
        if current_position is None and pending_signals:
            for sig in pending_signals[:]:
                entry_zone = sig['entry_zone']
                direction = sig['direction']
                touched = (direction == 'long' and candle['low'] <= entry_zone) or \
                          (direction == 'short' and candle['high'] >= entry_zone)
                
                if touched:
                    current_position = {
                        'strategy': sig['type'], 'direction': direction,
                        'signal_time': sig['signal_time'], 'entry_time': candle_time,
                        'entry_price': entry_zone,
                        'sl_price': entry_zone * (1 + sl) if direction == 'long' else entry_zone * (1 - sl),
                        'tp1_hit': False
                    }
                    pending_signals.remove(sig)
                    break
    
    return trades

# 시그널 감지
print("\n📊 시그널 감지...")
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals
print(f"총 시그널: {len(all_signals)}개")

# 최적 설정으로 백테스트
print("\n⏳ 백테스트 실행...")
trades = simulate(all_signals, df_15m, tp1=0.015, tp2=0.035, sl=-0.015, time_stop=48, use_partial=False)
df_trades = pd.DataFrame(trades)
print(f"총 거래: {len(df_trades)}회")

# 수익곡선 계산
df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
df_trades['peak'] = df_trades['cumulative_pnl'].expanding().max()
df_trades['drawdown'] = df_trades['cumulative_pnl'] - df_trades['peak']

# 시각화
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. 수익곡선
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_trades['exit_time'], df_trades['cumulative_pnl'], linewidth=2, color='blue', label='Cumulative PNL')
ax1.fill_between(df_trades['exit_time'], 0, df_trades['cumulative_pnl'], alpha=0.3, color='blue')
ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
ax1.set_title('Cumulative PNL Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative PNL (%)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 최종 수익률 표시
final_pnl = df_trades['cumulative_pnl'].iloc[-1]
ax1.text(0.02, 0.95, f'Final PNL: {final_pnl:.1f}%', transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(df_trades['exit_time'], 0, df_trades['drawdown'], alpha=0.5, color='red', label='Drawdown')
ax2.plot(df_trades['exit_time'], df_trades['drawdown'], linewidth=1.5, color='darkred')
ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# MDD 표시
mdd = df_trades['drawdown'].min()
mdd_idx = df_trades['drawdown'].idxmin()
mdd_time = df_trades.loc[mdd_idx, 'exit_time']
ax2.scatter([mdd_time], [mdd], color='red', s=100, zorder=5, marker='v')
ax2.text(0.02, 0.05, f'Max Drawdown: {mdd:.1f}%', transform=ax2.transAxes,
         fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# 3. 결과 분포
ax3 = fig.add_subplot(gs[2, 0])
result_counts = df_trades['result'].value_counts()
colors_map = {'TP2': 'darkgreen', 'BE': 'orange', 'TIME': 'purple', 'SL': 'red'}
colors = [colors_map.get(r, 'gray') for r in result_counts.index]
ax3.bar(result_counts.index, result_counts.values, color=colors, alpha=0.7)
ax3.set_title('Trade Results Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Result', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')

# 비율 표시
for i, (result, count) in enumerate(result_counts.items()):
    percentage = count / len(df_trades) * 100
    ax3.text(i, count + 10, f'{count}\n({percentage:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# 4. 전략별 성과
ax4 = fig.add_subplot(gs[2, 1])
strategy_pnl = df_trades.groupby('strategy')['pnl'].sum().sort_values(ascending=False)
colors_strat = ['green' if x > 0 else 'red' for x in strategy_pnl.values]
ax4.barh(strategy_pnl.index, strategy_pnl.values, color=colors_strat, alpha=0.7)
ax4.set_title('PNL by Strategy', fontsize=14, fontweight='bold')
ax4.set_xlabel('Total PNL (%)', fontsize=12)
ax4.set_ylabel('Strategy', fontsize=12)
ax4.grid(True, alpha=0.3, axis='x')

# PNL 값 표시
for i, (strategy, pnl) in enumerate(strategy_pnl.items()):
    ax4.text(pnl + (50 if pnl > 0 else -50), i, f'{pnl:.1f}%', 
             va='center', ha='left' if pnl > 0 else 'right', fontsize=10, fontweight='bold')

plt.savefig('pnl_curve.png', dpi=150, bbox_inches='tight')
print("\n✅ 수익곡선 차트 저장: pnl_curve.png")

# 통계 출력
print("\n" + "=" * 80)
print("📊 백테스트 통계")
print("=" * 80)

total_trades = len(df_trades)
win_trades = len(df_trades[df_trades['result'] != 'SL'])
loss_trades = len(df_trades[df_trades['result'] == 'SL'])
win_rate = win_trades / total_trades * 100

first_time = df_trades['entry_time'].min()
last_time = df_trades['exit_time'].max()
months = (last_time - first_time).days / 30
monthly_avg = final_pnl / months

print(f"""
기간: {first_time.date()} ~ {last_time.date()} ({months:.1f}개월)
총 거래: {total_trades}회
승률: {win_rate:.1f}% ({win_trades}승 {loss_trades}패)
총 수익: {final_pnl:.1f}%
월평균: {monthly_avg:.2f}%
MDD: {mdd:.1f}%

결과 분포:
  - TP2: {result_counts.get('TP2', 0)}회 ({result_counts.get('TP2', 0)/total_trades*100:.1f}%)
  - BE:  {result_counts.get('BE', 0)}회 ({result_counts.get('BE', 0)/total_trades*100:.1f}%)
  - TIME: {result_counts.get('TIME', 0)}회 ({result_counts.get('TIME', 0)/total_trades*100:.1f}%)
  - SL:  {result_counts.get('SL', 0)}회 ({result_counts.get('SL', 0)/total_trades*100:.1f}%)

전략별 성과:
""")

for strategy in df_trades['strategy'].unique():
    strat_trades = df_trades[df_trades['strategy'] == strategy]
    strat_pnl = strat_trades['pnl'].sum()
    strat_count = len(strat_trades)
    strat_win = len(strat_trades[strat_trades['result'] != 'SL'])
    print(f"  - {strategy}: {strat_pnl:.1f}% ({strat_count}회, 승률 {strat_win/strat_count*100:.1f}%)")

