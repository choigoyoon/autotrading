"""
=============================================================================
PNL 계산 버그 분석 및 수정
=============================================================================

발견된 버그들:
-------------

## 버그 1: partial profit 모드에서 TP1 미도달 시 SL/TIME에서 50% PNL만 계산
   
   문제 코드 (line 102-103, 137-138):
   ```
   if use_partial:
       pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5)
   ```
   
   문제점:
   - use_partial=True 인데 TP1에 도달하지 않은 경우 (partial_pnl=0)
   - SL이 발동되면 sl * 100 * 0.5 = -1.5% * 0.5 = -0.75% 만 계산됨
   - 실제로는 전체 포지션이 SL에 걸리므로 100% 손실 = -1.5% 이어야 함

   올바른 계산:
   - TP1 도달 O: partial_pnl(1%) + 나머지 50%의 BE(0%) = 1% ✓
   - TP1 도달 X: 전체 포지션 SL = -1.5% (100%)
   
## 버그 2: TIME stop에서도 동일한 문제
   
   문제 코드 (line 111-112, 146-147):
   ```
   if use_partial:
       pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
   ```
   
   문제점:
   - TP1 미도달 시 partial_pnl=0, time_pnl * 0.5만 계산
   - 실제로는 전체 포지션이 time stop이므로 100% = time_pnl 이어야 함

## 버그 3: TP2에서도 잠재적 문제
   
   문제 코드 (line 94-95, 129-130):
   ```
   if use_partial:
       pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
   ```
   
   상황 분석:
   - TP1 도달 후 TP2 도달: partial_pnl(1%) + tp2*0.5(1.75%) = 2.75% ✓ 올바름
   - TP1 미도달 후 바로 TP2 도달: 이론적으로 TP1을 먼저 거쳐야 하지만
     만약 같은 캔들에서 TP2까지 도달하면 TP1 체크가 먼저 되므로 문제 없음

=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 80)
print("🔍 PNL 계산 버그 분석 및 수정 검증")
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


# ============================================================================
# 버그가 있는 원래 시뮬레이션 (비교용)
# ============================================================================
def simulate_buggy(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False):
    """원래 버그가 있는 코드"""
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
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5  # 버그: TP1 미도달시 문제없음 (TP1 먼저 체크됨)
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    if use_partial:
                        # 🐛 버그: TP1 미도달시 sl * 100 * 0.5 = 50%만 계산
                        pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5)
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    if use_partial:
                        # 🐛 버그: TP1 미도달시 time_pnl * 0.5 = 50%만 계산
                        pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
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
                        # 🐛 버그
                        pos['pnl'] = pos.get('partial_pnl', 0) + (0 if pos['tp1_hit'] else sl * 100 * 0.5)
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    if use_partial:
                        # 🐛 버그
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


# ============================================================================
# 수정된 시뮬레이션
# ============================================================================
def simulate_fixed(signals, df_15m, tp1, tp2, sl, time_stop, use_partial=False):
    """버그가 수정된 코드"""
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
                # TP1 체크 (50% 익절 + 본절 이동)
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']  # 본절로 이동
                        if use_partial:
                            pos['partial_pnl'] = tp1 * 100 * 0.5  # 50% 포지션의 TP1 수익
                
                # TP2 체크
                tp2_price = pos['entry_price'] * (1 + tp2)
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    if use_partial:
                        # TP1 이미 도달했으므로 partial_pnl + 나머지 50%의 TP2
                        pos['pnl'] = pos.get('partial_pnl', 0) + tp2 * 100 * 0.5
                    else:
                        pos['pnl'] = tp2 * 100
                    closed = True
                # SL 체크
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    if use_partial:
                        if pos['tp1_hit']:
                            # TP1 도달 후 본절: 50% 수익 + 50% 본절(0%) = partial_pnl
                            pos['pnl'] = pos.get('partial_pnl', 0)
                        else:
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 SL = 100% 손실
                            pos['pnl'] = sl * 100  # 0.5 곱하지 않음!
                    else:
                        pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                # 시간 스탑 체크
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    time_pnl = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    if use_partial:
                        if pos['tp1_hit']:
                            # TP1 도달 후 시간스탑: 50% 수익 + 나머지 50%의 time_pnl
                            pos['pnl'] = pos.get('partial_pnl', 0) + time_pnl * 0.5
                        else:
                            # ⚡ 수정: TP1 미도달 시 전체 포지션 시간스탑 = 100% time_pnl
                            pos['pnl'] = time_pnl  # 0.5 곱하지 않음!
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
                            pos['pnl'] = pos.get('partial_pnl', 0)
                        else:
                            # ⚡ 수정
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
                            # ⚡ 수정
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


def analyze(trades, label=""):
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
    
    # 결과별 분포
    result_dist = df['result'].value_counts()
    tp2_count = result_dist.get('TP2', 0)
    be_count = result_dist.get('BE', 0)
    time_count = result_dist.get('TIME', 0)
    
    return {
        'trades': total,
        'monthly': monthly,
        'mdd': mdd,
        'sl_rate': sl_count / total * 100,
        'win_rate': win_count / total * 100,
        'total_pnl': total_pnl,
        'tp2': tp2_count,
        'be': be_count,
        'time': time_count,
        'sl': sl_count
    }


# 시그널 감지
print("\n📊 시그널 감지 중...")
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals
print(f"총 시그널: {len(all_signals)}개")


# ============================================================================
# 버그 검증: 동일 설정으로 buggy vs fixed 비교
# ============================================================================
print("\n" + "=" * 80)
print("🐛 버그 vs 수정 비교 테스트")
print("=" * 80)

test_configs = [
    {"name": "① 기준 (partial=False)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": False},
    {"name": "② 분할익절 (partial=True)", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 96, "partial": True},
    {"name": "③ 분할+시간48바", "tp1": 0.02, "tp2": 0.035, "sl": -0.015, "time": 48, "partial": True},
    {"name": "④ 분할+SL-1%", "tp1": 0.02, "tp2": 0.035, "sl": -0.01, "time": 96, "partial": True},
    {"name": "⑤ 분할+SL-1%+시간48바", "tp1": 0.02, "tp2": 0.035, "sl": -0.01, "time": 48, "partial": True},
]

print(f"\n{'설정':<30} {'버전':>8} {'거래':>6} {'SL%':>6} {'월평균':>10} {'MDD':>8} {'차이':>8}")
print("-" * 90)

for cfg in test_configs:
    # 버그 버전
    trades_buggy = simulate_buggy(all_signals, df_15m, cfg['tp1'], cfg['tp2'], cfg['sl'], cfg['time'], cfg['partial'])
    stats_buggy = analyze(trades_buggy)
    
    # 수정 버전
    trades_fixed = simulate_fixed(all_signals, df_15m, cfg['tp1'], cfg['tp2'], cfg['sl'], cfg['time'], cfg['partial'])
    stats_fixed = analyze(trades_fixed)
    
    diff_monthly = stats_fixed['monthly'] - stats_buggy['monthly']
    diff_mdd = stats_fixed['mdd'] - stats_buggy['mdd']
    
    print(f"{cfg['name']:<30} {'BUGGY':>8} {stats_buggy['trades']:>6} {stats_buggy['sl_rate']:>5.1f}% {stats_buggy['monthly']:>9.2f}% {stats_buggy['mdd']:>7.1f}%")
    print(f"{'':<30} {'FIXED':>8} {stats_fixed['trades']:>6} {stats_fixed['sl_rate']:>5.1f}% {stats_fixed['monthly']:>9.2f}% {stats_fixed['mdd']:>7.1f}% {diff_monthly:>+7.2f}%")
    print("-" * 90)

# 상세 분석
print("\n" + "=" * 80)
print("📊 분할익절 모드 상세 분석")
print("=" * 80)

# 분할익절 + 시간스탑 48바 설정으로 상세 분석
trades_buggy = simulate_buggy(all_signals, df_15m, 0.02, 0.035, -0.015, 48, True)
trades_fixed = simulate_fixed(all_signals, df_15m, 0.02, 0.035, -0.015, 48, True)

df_buggy = pd.DataFrame(trades_buggy)
df_fixed = pd.DataFrame(trades_fixed)

print(f"\n설정: 분할익절 + 시간스탑 48바 (TP1:2%, TP2:3.5%, SL:-1.5%)")
print(f"\n결과 분포:")
print(f"{'결과':>10} {'버그':>10} {'수정':>10} {'비율차이':>10}")
print("-" * 50)

for result in ['TP2', 'BE', 'TIME', 'SL']:
    buggy_count = len(df_buggy[df_buggy['result'] == result])
    fixed_count = len(df_fixed[df_fixed['result'] == result])
    print(f"{result:>10} {buggy_count:>10} {fixed_count:>10} {buggy_count-fixed_count:>+10}")

print(f"\n🔍 TP1 도달 여부별 PNL 분석 (버그 영향 범위):")

# TP1 미도달 거래만 분석
for version, df, label in [(df_buggy, df_buggy, 'BUGGY'), (df_fixed, df_fixed, 'FIXED')]:
    tp1_not_hit = df[df.apply(lambda x: not x.get('tp1_hit', False), axis=1)]
    
    print(f"\n[{label}] TP1 미도달 거래 (버그 영향 범위):")
    if len(tp1_not_hit) > 0:
        print(f"  - 총 {len(tp1_not_hit)}건")
        for result in ['SL', 'TIME']:
            subset = tp1_not_hit[tp1_not_hit['result'] == result]
            if len(subset) > 0:
                avg_pnl = subset['pnl'].mean()
                print(f"  - {result}: {len(subset)}건, 평균 PNL: {avg_pnl:.3f}%")

# 결론
print("\n" + "=" * 80)
print("📋 버그 수정 결론")
print("=" * 80)

stats_buggy = analyze(trades_buggy)
stats_fixed = analyze(trades_fixed)

print(f"""
발견된 버그:
-----------
분할익절(use_partial=True) 모드에서 TP1에 도달하지 못하고 SL/TIME에 걸린 경우
- 버그: PNL을 50%만 계산 (sl * 100 * 0.5 또는 time_pnl * 0.5)
- 수정: PNL을 100% 계산 (sl * 100 또는 time_pnl)

영향:
-----
- 월평균 수익률: {stats_buggy['monthly']:.2f}% → {stats_fixed['monthly']:.2f}% ({stats_fixed['monthly'] - stats_buggy['monthly']:+.2f}%)
- MDD: {stats_buggy['mdd']:.1f}% → {stats_fixed['mdd']:.1f}% ({stats_fixed['mdd'] - stats_buggy['mdd']:+.1f}%)

버그가 수익률을 과대평가했음: 손실을 절반만 반영하여 실제보다 높은 수익률 보고
MDD도 과소평가됨: 실제 최대 손실이 더 컸음
""")

