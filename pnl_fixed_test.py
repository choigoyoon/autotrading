import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("🔧 PNL 계산 버그 수정 후 재테스트")
print("=" * 70)

df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

# FVG 감지
def detect_fvg_signals(df_4h):
    signals = []
    for i in range(2, len(df_4h)):
        if df_4h['low'].iloc[i] > df_4h['high'].iloc[i-2]:
            gap_size = (df_4h['low'].iloc[i] - df_4h['high'].iloc[i-2]) / df_4h['high'].iloc[i-2] * 100
            if gap_size >= 0.3:
                signals.append({'type': 'FVG_bull', 'direction': 'long',
                    'signal_time': df_4h['timestamp'].iloc[i], 'entry_zone': df_4h['low'].iloc[i]})
        if df_4h['high'].iloc[i] < df_4h['low'].iloc[i-2]:
            gap_size = (df_4h['low'].iloc[i-2] - df_4h['high'].iloc[i]) / df_4h['high'].iloc[i] * 100
            if gap_size >= 0.3:
                signals.append({'type': 'FVG_bear', 'direction': 'short',
                    'signal_time': df_4h['timestamp'].iloc[i], 'entry_zone': df_4h['high'].iloc[i]})
    return signals

def detect_orderblock_signals(df_4h):
    signals = []
    for i in range(3, len(df_4h)):
        if (df_4h['close'].iloc[i-2] < df_4h['open'].iloc[i-2] and
            df_4h['close'].iloc[i-1] > df_4h['open'].iloc[i-1] and
            df_4h['close'].iloc[i-1] > df_4h['high'].iloc[i-2]):
            signals.append({'type': 'orderblock', 'direction': 'long',
                'signal_time': df_4h['timestamp'].iloc[i-1], 'entry_zone': df_4h['high'].iloc[i-2]})
    return signals

# 🔧 수정된 시뮬레이션 함수
def simulate_fixed(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False):
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
                # TP1 체크
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']  # 본절 이동
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5  # 50% 익절
                
                tp2_price = pos['entry_price'] * (1 + tp2)
                
                # TP2 도달
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                
                # SL/BE 도달
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    
                    if use_partial:
                        if pos['tp1_hit']:
                            # TP1 찍었으면: 50% 이미 익절 + 50% 본절
                            pos['pnl'] = pos.get('partial_pnl', 0) + 0
                        else:
                            # 🔧 수정: TP1 안 찍었으면 100% 손실!
                            pos['pnl'] = sl * 100  # 전체 손실
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                
                # 시간스탑
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    
                    if use_partial:
                        if pos['tp1_hit']:
                            # TP1 찍었으면: 50% 이미 익절 + 50% 시간스탑
                            pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
                        else:
                            # 🔧 수정: TP1 안 찍었으면 100%
                            pos['pnl'] = time_pnl
                    else:
                        pos['pnl'] = time_pnl
                    closed = True
            
            else:  # short
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
                            pos['pnl'] = pos.get('partial_pnl', 0) + 0
                        else:
                            pos['pnl'] = sl * 100  # 🔧 수정
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
                            pos['pnl'] = time_pnl  # 🔧 수정
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

def analyze(trades):
    if not trades:
        return {'trades': 0, 'monthly': 0, 'mdd': 0, 'sl_rate': 0, 'win_rate': 0}
    
    df = pd.DataFrame(trades)
    total = len(df)
    sl_count = len(df[df['result'] == 'SL'])
    total_pnl = df['pnl'].sum()
    
    first = df['entry_time'].min()
    last = df['exit_time'].max()
    months = (last - first).days / 30
    monthly = total_pnl / months if months > 0 else 0
    
    cumulative = df['pnl'].cumsum()
    peak = cumulative.expanding().max()
    mdd = (cumulative - peak).min()
    
    return {
        'trades': total,
        'monthly': monthly,
        'mdd': mdd,
        'sl_rate': sl_count / total * 100,
        'win_rate': (total - sl_count) / total * 100,
        'total_pnl': total_pnl
    }

# 시그널 감지
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals

print(f"\n총 시그널: {len(all_signals)}개")

# 비교 테스트
configs = [
    {"name": "① 기준 (분할X)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": False},
    {"name": "② 분할익절 (버그)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": True, "fixed": False},
    {"name": "③ 분할익절 (수정)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": True, "fixed": True},
    {"name": "④ 분할+시간48 (버그)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": True, "fixed": False},
    {"name": "⑤ 분할+시간48 (수정)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": True, "fixed": True},
]

# 기존 버그 있는 함수
def simulate_buggy(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False):
    # 기존 mdd_optimization.py의 simulate 함수와 동일
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
                        pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5)  # 버그!
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5  # 버그!
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
                        pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5)
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    if use_partial:
                        pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
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

print("\n⏳ 버그/수정 비교 테스트 중...")

print(f"\n{'='*80}")
print(f"{'설정':<30} {'거래':>6} {'SL%':>8} {'월평균':>10} {'MDD':>10}")
print("-" * 80)

for cfg in configs:
    if cfg.get('fixed', True):
        trades = simulate_fixed(all_signals, df_15m, cfg['tp1'], cfg['tp2'], cfg['sl'], cfg['time'], cfg['partial'])
    else:
        trades = simulate_buggy(all_signals, df_15m, cfg['tp1'], cfg['tp2'], cfg['sl'], cfg['time'], cfg['partial'])
    
    stats = analyze(trades)
    print(f"{cfg['name']:<30} {stats['trades']:>6} {stats['sl_rate']:>7.1f}% {stats['monthly']:>9.2f}% {stats['mdd']:>9.1f}%")

print(f"\n{'='*80}")
print("📊 버그 영향 분석")
print("=" * 80)

