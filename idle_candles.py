import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 70)
print("🎯 포지션 없는 캔들 (놀고 있는 시간) 분석")
print("=" * 70)

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
                    'entry_zone': gap_top
                })
        if df_4h['high'].iloc[i] < df_4h['low'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i-2]
            gap_bottom = df_4h['high'].iloc[i]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bear', 'direction': 'short',
                    'signal_time': df_4h['timestamp'].iloc[i],
                    'entry_zone': gap_bottom
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

# 시뮬레이션 (포지션 상태 추적)
def simulate_with_tracking(signals, df_15m, tp1=0.02, tp2=0.035, sl=-0.015, time_stop=48, use_partial=True):
    trades = []
    pending_signals = []
    current_position = None
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    # 캔들별 상태 추적
    candle_states = []  # 'idle', 'waiting', 'in_position'
    
    for i, candle in df_15m.iterrows():
        candle_time = candle['timestamp']
        state = 'idle'
        
        # 새 시그널 추가
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=5)
            pending_signals.append({**sig, 'expire_time': expire_time})
            signal_idx += 1
        
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        # 포지션 관리
        if current_position is not None:
            state = 'in_position'
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
                    pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5 if use_partial else tp2 * 100
                    closed = True
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5) if use_partial else (0 if pos['tp1_hit'] else sl * 100)
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5 if use_partial else time_pnl
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
                    pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5 if use_partial else tp2 * 100
                    closed = True
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5) if use_partial else (0 if pos['tp1_hit'] else sl * 100)
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5 if use_partial else time_pnl
                    closed = True
            
            if closed:
                trades.append(pos)
                current_position = None
                state = 'idle'
        
        # 대기 중 시그널 있으면 waiting
        if current_position is None and pending_signals:
            state = 'waiting'
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
                    state = 'in_position'
                    break
        
        candle_states.append({'timestamp': candle_time, 'state': state})
    
    return trades, candle_states

# 시그널 감지
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals

print(f"\n총 15분봉 캔들: {len(df_15m):,}개")
print(f"총 시그널: {len(all_signals)}개")

# 시뮬레이션 실행
print("\n⏳ 분석 중...")
trades, candle_states = simulate_with_tracking(all_signals, df_15m)

# 상태 분석
df_states = pd.DataFrame(candle_states)
state_counts = df_states['state'].value_counts()

total_candles = len(df_states)
idle_count = state_counts.get('idle', 0)
waiting_count = state_counts.get('waiting', 0)
position_count = state_counts.get('in_position', 0)

print(f"\n{'='*70}")
print("📊 캔들 상태 분석 결과")
print(f"{'='*70}")

print(f"\n총 캔들: {total_candles:,}개 (약 {total_candles * 15 / 60 / 24:.0f}일)")

print(f"\n📋 상태별 분포:")
print(f"   🟢 포지션 보유 (in_position): {position_count:,}개 ({position_count/total_candles*100:.1f}%)")
print(f"   🟡 시그널 대기 (waiting):     {waiting_count:,}개 ({waiting_count/total_candles*100:.1f}%)")
print(f"   🔴 놀고 있음 (idle):          {idle_count:,}개 ({idle_count/total_candles*100:.1f}%)")

# 시간으로 환산
idle_hours = idle_count * 15 / 60
idle_days = idle_hours / 24
total_days = total_candles * 15 / 60 / 24

print(f"\n⏰ 시간 환산:")
print(f"   총 기간: {total_days:.0f}일")
print(f"   포지션 보유: {position_count * 15 / 60 / 24:.0f}일 ({position_count/total_candles*100:.1f}%)")
print(f"   놀고 있는 시간: {idle_days:.0f}일 ({idle_count/total_candles*100:.1f}%)")

# 연속 idle 구간 분석
df_states['is_idle'] = df_states['state'] == 'idle'
df_states['idle_group'] = (df_states['is_idle'] != df_states['is_idle'].shift()).cumsum()

idle_streaks = []
for group_id, group in df_states[df_states['is_idle']].groupby('idle_group'):
    idle_streaks.append(len(group))

if idle_streaks:
    print(f"\n📈 연속 idle 구간 분석:")
    print(f"   최대 연속 idle: {max(idle_streaks)}캔들 ({max(idle_streaks) * 15 / 60:.1f}시간)")
    print(f"   평균 연속 idle: {np.mean(idle_streaks):.1f}캔들 ({np.mean(idle_streaks) * 15 / 60:.1f}시간)")
    print(f"   idle 구간 수: {len(idle_streaks)}회")

print(f"\n{'='*70}")
print("💡 결론")
print(f"{'='*70}")
print(f"\n   🎯 현재 전략으로 전체 시간의 {position_count/total_candles*100:.1f}%만 포지션 보유")
print(f"   🔴 {idle_count/total_candles*100:.1f}%는 완전히 놀고 있음 (시그널도 없음)")
print(f"   🟡 {waiting_count/total_candles*100:.1f}%는 시그널 대기 중 (터치 대기)")

