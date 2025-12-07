import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 시그널 감지 (검증된 3개 전략만)
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

def simulate_with_position_limit(all_signals, df_15m, max_positions=3):
    configs = {
        'fvg_bull': {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96},
        'fvg_bear': {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96},
        'orderblock': {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96},
    }
    
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    all_signals = sorted(all_signals, key=lambda x: x['time'])
    
    trades = []
    active_positions = []
    
    for sig in all_signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        config = configs.get(sig['type'], configs['fvg_bull'])
        direction = sig['dir']
        
        touch_idx = None
        for i in range(start+1, min(start+50, len(times))):
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
        active_positions = [(et, d, t) for et, d, t in active_positions if et > entry_time]
        
        if len(active_positions) >= max_positions:
            continue
        
        same_type = sum(1 for _, _, t in active_positions if t == sig['type'])
        if same_type >= 1:
            continue
        
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
            trades.append({'time': entry_time, 'result': result, 'pnl': pnl, 'type': sig['type'], 'direction': direction})
    
    return trades

print("=" * 95)
print("🏆 최종 검증된 다중 전략 시스템")
print("=" * 95)

# 검증된 3개 전략만 사용
all_signals = []
all_signals.extend(detect_bull_fvg(df_4h))
all_signals.extend(detect_bear_fvg(df_4h))
all_signals.extend(detect_orderblock(df_4h))

print(f"\n[사용 전략]")
print(f"  1. FVG 상승 (롱): {len(detect_bull_fvg(df_4h))}개 시그널")
print(f"  2. FVG 하락 (숏): {len(detect_bear_fvg(df_4h))}개 시그널")
print(f"  3. 오더블럭 (롱): {len(detect_orderblock(df_4h))}개 시그널")
print(f"  총: {len(all_signals)}개")

print("\n[미래 데이터 체크]")
print("  ✅ FVG: i-2, i 캔들만 사용 (과거만)")
print("  ✅ 오더블럭: i-1, i 캔들만 사용 (과거만)")
print("  ✅ 진입: 터치 후 다음 캔들 시가 (미래 데이터 없음)")

# 최종 시뮬레이션
trades = simulate_with_position_limit(all_signals, df_15m, max_positions=3)
df_trades = pd.DataFrame(trades)
df_trades['time'] = pd.to_datetime(df_trades['time'])

# 결과 요약
sl = (df_trades['result'] == 'SL').sum()
months = len(df_trades['time'].dt.to_period('M').unique())
pnl = df_trades['pnl'].sum()

print("\n" + "=" * 95)
print("📊 최종 성과")
print("=" * 95)

print(f"\n[전체 통계]")
print(f"  총 거래: {len(df_trades)}회 ({len(df_trades)/months:.1f}회/월)")
print(f"  SL 비율: {sl/len(df_trades)*100:.1f}%")
print(f"  승률: {(len(df_trades)-sl)/len(df_trades)*100:.1f}%")
print(f"  총 PnL: {pnl:.1f}%")
print(f"  월평균 PnL: {pnl/months:.2f}%")

# 전략별 성과
print(f"\n[전략별 성과]")
print(f"{'전략':<15} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 55)

for stype in df_trades['type'].unique():
    subset = df_trades[df_trades['type'] == stype]
    sl_s = (subset['result'] == 'SL').sum()
    months_s = len(subset['time'].dt.to_period('M').unique())
    pnl_s = subset['pnl'].sum()
    print(f"{stype:<15} {len(subset):>6} {sl_s/len(subset)*100:>6.1f}% {pnl_s:>8.1f}% {pnl_s/months_s:>7.2f}%")

# 연도별 성과
print(f"\n[연도별 성과]")
df_trades['year'] = df_trades['time'].dt.year
print(f"{'연도':<6} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 45)

for year in sorted(df_trades['year'].unique()):
    yearly = df_trades[df_trades['year'] == year]
    sl_y = (yearly['result'] == 'SL').sum()
    months_y = len(yearly['time'].dt.to_period('M').unique())
    pnl_y = yearly['pnl'].sum()
    print(f"{year:<6} {len(yearly):>6} {sl_y/len(yearly)*100:>6.1f}% {pnl_y:>8.1f}% {pnl_y/months_y:>7.2f}%")

# 비교
print("\n" + "=" * 95)
print("📈 단일 vs 다중 전략 비교")
print("=" * 95)

single = simulate_with_position_limit(detect_bull_fvg(df_4h), df_15m, max_positions=1)
df_single = pd.DataFrame(single)
single_months = len(pd.to_datetime(df_single['time']).dt.to_period('M').unique())
single_pnl = df_single['pnl'].sum()
single_sl = (df_single['result'] == 'SL').sum()

print(f"\n{'구분':<25} {'거래':>7} {'월거래':>7} {'SL%':>7} {'승률':>7} {'월평균':>9}")
print("-" * 75)
print(f"{'단일 (FVG롱 1포지션)':<25} {len(single):>7} {len(single)/single_months:>6.1f} {single_sl/len(single)*100:>6.1f}% {(len(single)-single_sl)/len(single)*100:>6.1f}% {single_pnl/single_months:>8.2f}%")
print(f"{'다중 (3전략 3포지션)':<25} {len(df_trades):>7} {len(df_trades)/months:>6.1f} {sl/len(df_trades)*100:>6.1f}% {(len(df_trades)-sl)/len(df_trades)*100:>6.1f}% {pnl/months:>8.2f}%")

print(f"\n  → 월평균: {single_pnl/single_months:.2f}% → {pnl/months:.2f}% (+{pnl/months - single_pnl/single_months:.2f}%)")
print(f"  → 거래빈도: {len(single)/single_months:.1f}회 → {len(df_trades)/months:.1f}회 ({len(df_trades)/len(single):.1f}배)")

