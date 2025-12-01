import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 70)
print("🎯 MDD 최적화 테스트")
print("=" * 70)

# FVG 감지
def detect_fvg_signals(df_4h):
    signals = []
    for i in range(2, len(df_4h)):
        # Bullish FVG
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
        # Bearish FVG
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

# Order Block 감지
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

# 시뮬레이션 함수
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
                            pos['pnl'] = pos.get('partial_pnl', 0)  # 본절
                        else:
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 SL = 100% 손실
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
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 시간스탑 = 100% time_pnl
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
                            pos['pnl'] = pos.get('partial_pnl', 0)  # 본절
                        else:
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 SL = 100% 손실
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
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 시간스탑 = 100% time_pnl
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

def analyze(trades):
    if not trades:
        return {'trades': 0, 'monthly': 0, 'mdd': 0, 'sl_rate': 0, 'win_rate': 0}
    
    df = pd.DataFrame(trades)
    total = len(df)
    sl_count = len(df[df['result'] == 'SL'])
    win_count = len(df[df['result'] != 'SL'])
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
        'win_rate': win_count / total * 100,
        'total_pnl': total_pnl
    }

# 시그널 감지
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals

print(f"\n총 시그널: {len(all_signals)}개")

# 테스트할 설정들
configs = [
    # 기준
    {"name": "① 기준 (TP1 2%, TP2 3.5%, SL -1.5%)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": False},
    
    # SL 줄이기
    {"name": "② SL -1.0% (타이트)", "tp1": 0.02, "tp2": 0.035, "sl": -0.01, "time": 96, "partial": False},
    {"name": "③ SL -0.75% (매우 타이트)", "tp1": 0.02, "tp2": 0.035, "sl": -0.0075, "time": 96, "partial": False},
    
    # TP1 빨리 (본절 빨리 이동)
    {"name": "④ TP1 1.5% (빠른 본절)", "tp1": 0.015, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": False},
    {"name": "⑤ TP1 1.0% (초빠른 본절)", "tp1": 0.01, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": False},
    
    # 시간스탑 줄이기
    {"name": "⑥ 시간스탑 48바 (12시간)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": False},
    {"name": "⑦ 시간스탑 32바 (8시간)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 32, "partial": False},
    
    # 분할익절
    {"name": "⑧ 분할익절 50%+50%", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": True},
    
    # 조합
    {"name": "⑨ SL -1% + TP1 1.5%", "tp1": 0.015, "tp2": 0.035, "sl": -0.01, "time": 96, "partial": False},
    {"name": "⑩ SL -1% + 시간스탑 48바", "tp1": 0.02, "tp2": 0.035, "sl": -0.01, "time": 48, "partial": False},
    {"name": "⑪ TP1 1.5% + 시간스탑 48바", "tp1": 0.015, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": False},
    {"name": "⑫ SL -1% + TP1 1.5% + 시간48바", "tp1": 0.015, "tp2": 0.035, "sl": -0.01, "time": 48, "partial": False},
    {"name": "⑬ 분할익절 + SL -1%", "tp1": 0.02, "tp2": 0.035, "sl": -0.01, "time": 96, "partial": True},
    {"name": "⑭ 분할익절 + 시간48바", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": True},
    {"name": "⑮ 올인원: SL-1% + TP1 1.5% + 시간48 + 분할", "tp1": 0.015, "tp2": 0.035, "sl": -0.01, "time": 48, "partial": True},
]

results = []
print(f"\n⏳ {len(configs)}개 설정 테스트 중...\n")

for cfg in configs:
    trades = simulate(all_signals, df_15m, cfg['tp1'], cfg['tp2'], cfg['sl'], cfg['time'], cfg['partial'])
    stats = analyze(trades)
    stats['name'] = cfg['name']
    results.append(stats)

# 결과 정렬 (MDD 기준)
results_sorted = sorted(results, key=lambda x: x['mdd'], reverse=True)

print("=" * 90)
print("📊 MDD 최적화 결과 (MDD 좋은 순)")
print("=" * 90)
print(f"{'설정':<45} {'거래':>6} {'SL%':>6} {'승률':>6} {'월평균':>8} {'MDD':>8}")
print("-" * 90)

for r in results_sorted:
    print(f"{r['name']:<45} {r['trades']:>6} {r['sl_rate']:>5.1f}% {r['win_rate']:>5.1f}% {r['monthly']:>7.2f}% {r['mdd']:>7.1f}%")

# Top 3 추천
print(f"\n{'='*90}")
print("🏆 MDD 최소화 TOP 3 추천")
print("=" * 90)

for i, r in enumerate(results_sorted[:3], 1):
    print(f"\n{i}위: {r['name']}")
    print(f"    거래: {r['trades']}회 | SL: {r['sl_rate']:.1f}% | 승률: {r['win_rate']:.1f}%")
    print(f"    월평균: {r['monthly']:.2f}% | MDD: {r['mdd']:.1f}%")

# 수익 대비 MDD 효율 (월평균/|MDD|)
print(f"\n{'='*90}")
print("📈 수익/MDD 효율 TOP 3 (월평균 ÷ |MDD|)")
print("=" * 90)

for r in results:
    r['efficiency'] = r['monthly'] / abs(r['mdd']) if r['mdd'] != 0 else 0

eff_sorted = sorted(results, key=lambda x: x['efficiency'], reverse=True)

for i, r in enumerate(eff_sorted[:3], 1):
    print(f"\n{i}위: {r['name']}")
    print(f"    효율: {r['efficiency']:.2f} (월평균 {r['monthly']:.2f}% ÷ MDD {abs(r['mdd']):.1f}%)")
    print(f"    거래: {r['trades']}회 | 승률: {r['win_rate']:.1f}%")

