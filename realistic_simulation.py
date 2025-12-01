import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 시그널 감지 함수들
def detect_bull_fvg(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['high'] < df.iloc[i]['low']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['low'], 'type': 'fvg_bull', 'dir': 'long'})
    return signals

def detect_bear_fvg(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['high'], 'type': 'fvg_bear', 'dir': 'short'})
    return signals

def detect_orderblock(df):
    signals = []
    for i in range(3, len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['open']:
            if df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                move = (df.iloc[i]['close'] - df.iloc[i-1]['low']) / df.iloc[i-1]['low'] * 100
                if move > 2:
                    signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i-1]['open'], 'type': 'orderblock', 'dir': 'long'})
    return signals

def detect_breaker_fixed(df):
    """수정된 브레이커 - 이전 캔들 종가로 돌파 확인"""
    signals = []
    for i in range(16, len(df)):
        window = df.iloc[i-16:i-1]
        highs_idx = argrelextrema(window['high'].values, np.greater, order=4)[0]
        if len(highs_idx) >= 1:
            high_price = window.iloc[highs_idx[-1]]['high']
            if df.iloc[i-1]['close'] > high_price * 1.005:
                signals.append({'time': df.iloc[i]['datetime'], 'level': high_price, 'type': 'breaker', 'dir': 'long'})
    return signals

# 동시 포지션 제한 시뮬레이션
def simulate_with_position_limit(all_signals, df_15m, max_positions=1, max_per_direction=None):
    """
    max_positions: 최대 동시 포지션 수
    max_per_direction: 방향별 최대 포지션 (None이면 제한 없음)
    """
    # 설정
    configs = {
        'fvg_bull': {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96},
        'fvg_bear': {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96},
        'orderblock': {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96},
        'breaker': {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96},
    }
    
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    # 모든 시그널 시간순 정렬
    all_signals = sorted(all_signals, key=lambda x: x['time'])
    
    trades = []
    active_positions = []  # [(exit_time, direction, type)]
    
    for sig in all_signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        config = configs.get(sig['type'], configs['fvg_bull'])
        direction = sig['dir']
        
        # 터치 대기
        touch_idx = None
        for i in range(start+1, min(start+50, len(times))):
            # 이미 만료된 포지션 제거
            active_positions = [(et, d, t) for et, d, t in active_positions if et > times[i]]
            
            if direction == 'long' and lows[i] <= sig['level']:
                touch_idx = i
                break
            elif direction == 'short' and highs[i] >= sig['level']:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        entry_time = times[entry_idx]
        
        # 만료된 포지션 다시 제거
        active_positions = [(et, d, t) for et, d, t in active_positions if et > entry_time]
        
        # 포지션 제한 체크
        current_count = len(active_positions)
        if current_count >= max_positions:
            continue
        
        # 방향별 제한 체크
        if max_per_direction:
            dir_count = sum(1 for _, d, _ in active_positions if d == direction)
            if dir_count >= max_per_direction:
                continue
        
        # 같은 전략 중복 체크
        same_type = sum(1 for _, _, t in active_positions if t == sig['type'])
        if same_type >= 1:
            continue
        
        # 진입
        ep = opens[entry_idx]
        tp1, tp2, sl = config['tp1'], config['tp2'], config['sl']
        time_stop = config['time_stop']
        
        if direction == 'long':
            tp1_p, tp2_p, sl_p = ep*(1+tp1/100), ep*(1+tp2/100), ep*(1+sl/100)
        else:
            tp1_p, tp2_p, sl_p = ep*(1-tp1/100), ep*(1-tp2/100), ep*(1-sl/100)
        
        result, pnl, tp1_hit = None, 0, False
        exit_idx = None
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                if direction == 'long':
                    pnl = (closes[i] - ep) / ep * 100 - 0.06
                else:
                    pnl = (ep - closes[i]) / ep * 100 - 0.06
                result = 'TIME'
                exit_idx = i
                break
            
            if not tp1_hit:
                if direction == 'long':
                    if lows[i] <= sl_p:
                        result, pnl = 'SL', sl - 0.06
                        exit_idx = i
                        break
                    if highs[i] >= tp1_p:
                        tp1_hit = True
                        continue
                else:
                    if highs[i] >= sl_p:
                        result, pnl = 'SL', sl - 0.06
                        exit_idx = i
                        break
                    if lows[i] <= tp1_p:
                        tp1_hit = True
                        continue
            else:
                if direction == 'long':
                    if lows[i] <= ep:
                        result, pnl = 'BE', -0.06
                        exit_idx = i
                        break
                    if highs[i] >= tp2_p:
                        result, pnl = 'TP2', tp2 - 0.06
                        exit_idx = i
                        break
                else:
                    if highs[i] >= ep:
                        result, pnl = 'BE', -0.06
                        exit_idx = i
                        break
                    if lows[i] <= tp2_p:
                        result, pnl = 'TP2', tp2 - 0.06
                        exit_idx = i
                        break
        
        if result is None and tp1_hit:
            result, pnl = 'BE', -0.06
            exit_idx = min(entry_idx + 200, len(times) - 1)
        
        if result and exit_idx:
            exit_time = times[exit_idx]
            active_positions.append((exit_time, direction, sig['type']))
            trades.append({
                'time': entry_time,
                'result': result,
                'pnl': pnl,
                'type': sig['type'],
                'direction': direction
            })
    
    return trades

def calc_stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    sl = (df['result'] == 'SL').sum()
    months = len(pd.to_datetime(df['time']).dt.to_period('M').unique())
    pnl = df['pnl'].sum()
    return {
        'n': len(trades),
        'sl_rate': sl/len(trades)*100,
        'win_rate': (len(trades)-sl)/len(trades)*100,
        'pnl': pnl,
        'mavg': pnl/months,
        'months': months
    }

print("=" * 95)
print("🎯 현실적 시뮬레이션 (동시 포지션 제한 적용)")
print("=" * 95)

# 모든 시그널 수집
all_signals = []
all_signals.extend(detect_bull_fvg(df_4h))
all_signals.extend(detect_bear_fvg(df_4h))
all_signals.extend(detect_orderblock(df_4h))
all_signals.extend(detect_breaker_fixed(df_4h))

print(f"\n총 시그널: {len(all_signals)}개")

# 다양한 포지션 제한 테스트
scenarios = [
    ("무제한", 999, None),
    ("최대 1포지션", 1, None),
    ("최대 2포지션", 2, None),
    ("최대 3포지션", 3, None),
    ("최대 4포지션 (전략당 1개)", 4, None),
    ("롱2 + 숏2", 4, 2),
]

print(f"\n{'시나리오':<25} {'거래':>6} {'SL%':>7} {'승률':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 75)

for name, max_pos, max_dir in scenarios:
    trades = simulate_with_position_limit(all_signals, df_15m, max_pos, max_dir)
    s = calc_stats(trades)
    if s:
        print(f"{name:<25} {s['n']:>6} {s['sl_rate']:>6.1f}% {s['win_rate']:>6.1f}% {s['pnl']:>8.1f}% {s['mavg']:>7.2f}%")

# 최적 시나리오 상세 분석
print("\n" + "=" * 95)
print("📊 최적 시나리오 상세 분석 (최대 3포지션)")
print("=" * 95)

trades = simulate_with_position_limit(all_signals, df_15m, max_positions=3)
df_trades = pd.DataFrame(trades)
df_trades['time'] = pd.to_datetime(df_trades['time'])

# 전략별 성과
print("\n[전략별 성과]")
print(f"{'전략':<15} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 55)

for stype in df_trades['type'].unique():
    subset = df_trades[df_trades['type'] == stype]
    sl = (subset['result'] == 'SL').sum()
    months = len(subset['time'].dt.to_period('M').unique())
    pnl = subset['pnl'].sum()
    print(f"{stype:<15} {len(subset):>6} {sl/len(subset)*100:>6.1f}% {pnl:>8.1f}% {pnl/months:>7.2f}%")

# 연도별 성과
print("\n[연도별 성과]")
df_trades['year'] = df_trades['time'].dt.year
print(f"{'연도':<6} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 45)

for year in sorted(df_trades['year'].unique()):
    yearly = df_trades[df_trades['year'] == year]
    sl = (yearly['result'] == 'SL').sum()
    months = len(yearly['time'].dt.to_period('M').unique())
    pnl = yearly['pnl'].sum()
    print(f"{year:<6} {len(yearly):>6} {sl/len(yearly)*100:>6.1f}% {pnl:>8.1f}% {pnl/months:>7.2f}%")

# 최종 비교
print("\n" + "=" * 95)
print("🏆 최종 비교: 단일 전략 vs 다중 전략")
print("=" * 95)

# 단일 전략 (FVG 롱만, 최대 1포지션)
single_signals = detect_bull_fvg(df_4h)
single_trades = simulate_with_position_limit(single_signals, df_15m, max_positions=1)
single_s = calc_stats(single_trades)

# 다중 전략 (3포지션)
multi_trades = simulate_with_position_limit(all_signals, df_15m, max_positions=3)
multi_s = calc_stats(multi_trades)

print(f"\n{'구분':<25} {'거래':>7} {'월거래':>7} {'승률':>7} {'월평균':>9}")
print("-" * 65)
print(f"{'단일 (FVG 롱, 1포지션)':<25} {single_s['n']:>7} {single_s['n']/single_s['months']:>6.1f} {single_s['win_rate']:>6.1f}% {single_s['mavg']:>8.2f}%")
print(f"{'다중 (4전략, 3포지션)':<25} {multi_s['n']:>7} {multi_s['n']/multi_s['months']:>6.1f} {multi_s['win_rate']:>6.1f}% {multi_s['mavg']:>8.2f}%")

improvement = multi_s['mavg'] - single_s['mavg']
trade_increase = multi_s['n'] / single_s['n']
print(f"\n  → 월평균 수익: +{improvement:.2f}% 증가")
print(f"  → 거래 빈도: {trade_increase:.1f}배 증가")

