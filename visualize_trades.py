"""
백테스트 매매 시각화
- 실제 거래 예시를 차트로 표시
- 진입/청산 포인트, TP1/TP2/SL 레벨 표시
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 80)
print("📊 백테스트 매매 시각화")
print("=" * 80)

# FVG 감지
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

# 수정된 시뮬레이션 (샘플 추출용)
def simulate_with_details(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False, max_trades=10):
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
            
            # 가격 업데이트 (시각화용)
            pos['price_history'].append({
                'time': candle_time,
                'price': candle['close'],
                'high': candle['high'],
                'low': candle['low']
            })
            
            if pos['direction'] == 'long':
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['tp1_time'] = candle_time
                        pos['sl_price'] = pos['entry_price']
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5
                
                tp2_price = pos['entry_price'] * (1 + tp2)
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = tp2_price
                    pos['result'] = 'TP2'
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = pos['sl_price']
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
                    pos['exit_price'] = candle['close']
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
            else:  # short
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 - tp1)
                    if candle['low'] <= tp1_price:
                        pos['tp1_hit'] = True
                        pos['tp1_time'] = candle_time
                        pos['sl_price'] = pos['entry_price']
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5
                
                tp2_price = pos['entry_price'] * (1 - tp2)
                if candle['low'] <= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = tp2_price
                    pos['result'] = 'TP2'
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = pos['sl_price']
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
                    pos['exit_price'] = candle['close']
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
                if len(trades) >= max_trades:
                    return trades
        
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
                        'tp1_hit': False,
                        'price_history': []
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

# 최적 설정으로 샘플 거래 추출
print("\n⏳ 샘플 거래 시뮬레이션...")
sample_trades = simulate_with_details(
    all_signals, df_15m,
    tp1=0.015, tp2=0.035, sl=-0.015, time_stop=48,
    use_partial=False, max_trades=50
)

print(f"샘플 거래 수: {len(sample_trades)}개")

# 각 결과 타입별로 1개씩 선택
trade_examples = {}
for result_type in ['TP2', 'BE', 'TIME', 'SL']:
    for trade in sample_trades:
        if trade['result'] == result_type and result_type not in trade_examples:
            trade_examples[result_type] = trade
            break

print(f"\n선택된 예시: {list(trade_examples.keys())}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for idx, (result_type, trade) in enumerate(sorted(trade_examples.items())):
    ax = axes[idx]
    
    # 거래 기간의 15분봉 데이터
    start_time = trade['entry_time'] - timedelta(hours=12)
    end_time = trade['exit_time'] + timedelta(hours=6)
    
    mask = (df_15m['timestamp'] >= start_time) & (df_15m['timestamp'] <= end_time)
    chart_data = df_15m[mask].copy()
    
    if len(chart_data) == 0:
        continue
    
    # 캔들스틱 그리기
    for i, row in chart_data.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([row['timestamp'], row['timestamp']], [row['low'], row['high']], 
                color=color, linewidth=0.5, alpha=0.3)
        
        body_bottom = min(row['open'], row['close'])
        body_top = max(row['open'], row['close'])
        body_height = body_top - body_bottom
        
        rect = Rectangle((mdates.date2num(row['timestamp']) - 0.0002, body_bottom),
                        0.0004, body_height if body_height > 0 else row['close']*0.0001,
                        facecolor=color, edgecolor=color, alpha=0.6)
        ax.add_patch(rect)
    
    # 가격 레벨
    entry_price = trade['entry_price']
    direction = trade['direction']
    
    if direction == 'long':
        tp1_price = entry_price * 1.015
        tp2_price = entry_price * 1.035
        sl_price = entry_price * 0.985
    else:
        tp1_price = entry_price * 0.985
        tp2_price = entry_price * 0.965
        sl_price = entry_price * 1.015
    
    # 레벨 라인
    ax.axhline(entry_price, color='blue', linewidth=2, label='Entry', linestyle='--', alpha=0.7)
    ax.axhline(tp1_price, color='green', linewidth=1.5, label='TP1 (1.5%)', linestyle='--', alpha=0.6)
    ax.axhline(tp2_price, color='darkgreen', linewidth=1.5, label='TP2 (3.5%)', linestyle='--', alpha=0.6)
    ax.axhline(sl_price, color='red', linewidth=1.5, label='SL (-1.5%)', linestyle='--', alpha=0.6)
    
    # 진입/청산 포인트
    ax.scatter(trade['entry_time'], entry_price, color='blue', s=200, marker='^' if direction=='long' else 'v', 
               zorder=5, edgecolors='black', linewidths=2, label='Entry')
    
    exit_color = {'TP2': 'darkgreen', 'BE': 'orange', 'TIME': 'purple', 'SL': 'red'}[result_type]
    ax.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, s=200, marker='X',
               zorder=5, edgecolors='black', linewidths=2, label=f'Exit ({result_type})')
    
    # TP1 도달 표시
    if trade.get('tp1_hit'):
        ax.scatter(trade.get('tp1_time'), tp1_price, color='green', s=150, marker='*',
                  zorder=5, edgecolors='black', linewidths=1, label='TP1 Hit')
        ax.axvline(trade.get('tp1_time'), color='green', linewidth=1, alpha=0.3, linestyle=':')
    
    # 진입/청산 시간 라인
    ax.axvline(trade['entry_time'], color='blue', linewidth=1, alpha=0.3, linestyle=':')
    ax.axvline(trade['exit_time'], color=exit_color, linewidth=1, alpha=0.3, linestyle=':')
    
    # 제목 및 정보
    pnl_color = 'green' if trade['pnl'] > 0 else 'red'
    title = f"{result_type} | {trade['strategy']} | {direction.upper()}\n"
    title += f"Entry: ${entry_price:.0f} → Exit: ${trade['exit_price']:.0f} | "
    title += f"PNL: {trade['pnl']:+.2f}%"
    ax.set_title(title, fontsize=12, fontweight='bold', color=pnl_color)
    
    # 레이블
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Price (USD)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # x축 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('trade_examples.png', dpi=150, bbox_inches='tight')
print("\n✅ 차트 저장: trade_examples.png")

# 통계 출력
print("\n" + "=" * 80)
print("📊 샘플 거래 상세 정보")
print("=" * 80)

for result_type, trade in sorted(trade_examples.items()):
    print(f"\n[{result_type}] {trade['strategy']} - {trade['direction'].upper()}")
    print(f"  Signal: {trade['signal_time']}")
    print(f"  Entry:  {trade['entry_time']} @ ${trade['entry_price']:.2f}")
    print(f"  Exit:   {trade['exit_time']} @ ${trade['exit_price']:.2f}")
    print(f"  TP1 Hit: {'Yes' if trade.get('tp1_hit') else 'No'}")
    if trade.get('tp1_hit'):
        print(f"  TP1 Time: {trade.get('tp1_time')}")
    duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
    print(f"  Duration: {duration:.1f} hours")
    print(f"  PNL: {trade['pnl']:+.2f}%")

