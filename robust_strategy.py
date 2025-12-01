import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

def calc_macd(df):
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    return exp1 - exp2

df_4h['macd'] = calc_macd(df_4h)

def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        p2, curr = df.iloc[i-2], df.iloc[i]
        if p2['high'] < curr['low']:
            fvgs.append({
                'idx': i, 'time': curr['datetime'], 'top': curr['low'],
                'body': abs(curr['close'] - curr['open']) / curr['open'] * 100,
            })
    return fvgs

fvgs = detect_fvg(df_4h)

def classify_fvg(df, fvg, lookback=20):
    idx = fvg['idx']
    if idx < lookback:
        return {'diver': False, 'hh': False}
    window = df.iloc[idx-lookback:idx]
    lows_idx = argrelextrema(window['low'].values, np.less, order=3)[0]
    diver = False
    if len(lows_idx) >= 2:
        l1, l2 = lows_idx[-2], lows_idx[-1]
        if window.iloc[l2]['low'] < window.iloc[l1]['low'] and window.iloc[l2]['macd'] > window.iloc[l1]['macd']:
            diver = True
    highs_idx = argrelextrema(window['high'].values, np.greater, order=3)[0]
    hh = len(highs_idx) >= 2 and window.iloc[highs_idx[-1]]['high'] > window.iloc[highs_idx[-2]]['high']
    return {'diver': diver, 'hh': hh}

def simulate(fvgs, df_4h, df_15m, config):
    trades = []
    times, lows, highs, opens, closes = df_15m['datetime'].values, df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    for fvg in fvgs:
        cond = classify_fvg(df_4h, fvg)
        if cond['diver'] and not cond['hh']:
            continue
        
        start = np.searchsorted(times, np.datetime64(fvg['time']))
        if start >= len(times) - 200:
            continue
        
        touch_idx = None
        for i in range(start+1, min(start+200, len(times))):
            if lows[i] <= fvg['top']:
                touch_idx = i
                break
        if touch_idx is None or (touch_idx - start) > config.get('max_touch_bars', 999):
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        tp1_p, tp2_p, sl_p = ep*(1+config['tp1']/100), ep*(1+config['tp2']/100), ep*(1+config['sl']/100)
        
        result, pnl, tp1_hit = None, 0, False
        time_stop = config.get('time_stop', 0)
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                pnl = (closes[i] - ep) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            if not tp1_hit:
                if lows[i] <= sl_p:
                    result, pnl = 'SL', config['sl'] - 0.06
                    break
                if highs[i] >= tp1_p:
                    tp1_hit = True
                    continue
            else:
                if lows[i] <= ep:
                    result, pnl = 'BE', -0.06
                    break
                if highs[i] >= tp2_p:
                    result, pnl = 'TP2', config['tp2'] - 0.06
                    break
        
        if result is None and tp1_hit:
            result, pnl = 'BE', -0.06
        
        if result:
            trades.append({'time': times[entry_idx], 'result': result, 'pnl': pnl})
    
    return trades

def analyze(trades, label=""):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    df['time'] = pd.to_datetime(df['time'])
    
    # Split by year
    df['year'] = df['time'].dt.year
    years = sorted(df['year'].unique())
    
    stats = {}
    for year in years:
        yearly = df[df['year'] == year]
        sl = (yearly['result'] == 'SL').sum()
        stats[year] = {
            'n': len(yearly),
            'sl_rate': sl/len(yearly)*100,
            'pnl': yearly['pnl'].sum(),
            'mavg': yearly['pnl'].sum() / len(yearly['time'].dt.to_period('M').unique())
        }
    
    total_sl = (df['result'] == 'SL').sum()
    total_months = len(df['time'].dt.to_period('M').unique())
    stats['total'] = {
        'n': len(df),
        'sl_rate': total_sl/len(df)*100,
        'pnl': df['pnl'].sum(),
        'mavg': df['pnl'].sum() / total_months
    }
    
    return stats

print("=" * 100)
print("🔍 연도별 안정성 분석")
print("=" * 100)

# 다양한 설정 테스트
configs = [
    ("A: 기본 본절스탑", {'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5, 'max_touch_bars': 20}),
    ("B: 시간스탑 24h", {'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5, 'max_touch_bars': 20, 'time_stop': 96}),
    ("C: 시간스탑 + SL-2%", {'tp1': 1.5, 'tp2': 4.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}),
    ("D: TP2 3% (낮은 목표)", {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'max_touch_bars': 20, 'time_stop': 96}),
    ("E: TP1 1% TP2 3% SL-1%", {'tp1': 1.0, 'tp2': 3.0, 'sl': -1.0, 'max_touch_bars': 20, 'time_stop': 96}),
    ("F: TP1 2% TP2 4% SL-2%", {'tp1': 2.0, 'tp2': 4.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}),
]

for name, config in configs:
    trades = simulate(fvgs, df_4h, df_15m, config)
    stats = analyze(trades)
    
    if stats:
        print(f"\n[{name}]")
        print(f"{'연도':<8} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
        print("-" * 45)
        
        years = [k for k in stats.keys() if k != 'total']
        for year in sorted(years):
            s = stats[year]
            print(f"{year:<8} {s['n']:>6} {s['sl_rate']:>6.1f}% {s['pnl']:>8.1f}% {s['mavg']:>7.2f}%")
        
        s = stats['total']
        print("-" * 45)
        print(f"{'전체':<8} {s['n']:>6} {s['sl_rate']:>6.1f}% {s['pnl']:>8.1f}% {s['mavg']:>7.2f}%")

# 2025 성과 집중 분석
print("\n" + "=" * 100)
print("📊 2025년 성과 비교")
print("=" * 100)

print(f"\n{'전략':<25} {'2025 거래':>8} {'2025 SL%':>9} {'2025 월평균':>12} {'전체 월평균':>12}")
print("-" * 75)

for name, config in configs:
    trades = simulate(fvgs, df_4h, df_15m, config)
    stats = analyze(trades)
    if stats and 2025 in stats:
        s25 = stats[2025]
        st = stats['total']
        print(f"{name:<25} {s25['n']:>8} {s25['sl_rate']:>8.1f}% {s25['mavg']:>11.2f}% {st['mavg']:>11.2f}%")

# 가장 안정적인 전략 찾기
print("\n" + "=" * 100)
print("🎯 연도별 편차가 적은 안정적 전략 탐색")
print("=" * 100)

best_strategies = []

for tp1 in [1.0, 1.5, 2.0]:
    for tp2 in [2.5, 3.0, 3.5, 4.0]:
        for sl in [-1.0, -1.5, -2.0]:
            for time_stop in [0, 48, 96]:
                config = {'tp1': tp1, 'tp2': tp2, 'sl': sl, 'max_touch_bars': 20, 'time_stop': time_stop}
                trades = simulate(fvgs, df_4h, df_15m, config)
                stats = analyze(trades)
                
                if stats and 2025 in stats:
                    years = [k for k in stats.keys() if k != 'total']
                    mavgs = [stats[y]['mavg'] for y in years if stats[y]['n'] >= 10]
                    
                    if len(mavgs) >= 3:
                        # 안정성 = 최소 월평균 (최악의 해 기준)
                        min_mavg = min(mavgs)
                        avg_mavg = np.mean(mavgs)
                        std_mavg = np.std(mavgs)
                        
                        best_strategies.append({
                            'config': config,
                            'min_mavg': min_mavg,
                            'avg_mavg': avg_mavg,
                            'std_mavg': std_mavg,
                            'mavg_2025': stats[2025]['mavg'],
                            'total_mavg': stats['total']['mavg'],
                            'n': stats['total']['n']
                        })

# 최소 월평균 기준 상위 10개
df_best = pd.DataFrame(best_strategies)
df_sorted = df_best.sort_values('min_mavg', ascending=False).head(15)

print(f"\n[최악의 해 월평균 기준 TOP 15]")
print(f"{'TP1':>4} {'TP2':>4} {'SL':>5} {'시간':>4} {'최소월평균':>10} {'평균월평균':>10} {'2025월평균':>10} {'거래':>5}")
print("-" * 70)

for _, row in df_sorted.iterrows():
    c = row['config']
    ts = c['time_stop'] if c['time_stop'] > 0 else '-'
    print(f"{c['tp1']:>4} {c['tp2']:>4} {c['sl']:>5} {str(ts):>4} {row['min_mavg']:>9.2f}% {row['avg_mavg']:>9.2f}% {row['mavg_2025']:>9.2f}% {row['n']:>5}")

# 최종 추천
print("\n" + "=" * 100)
print("🏆 최종 추천 전략")
print("=" * 100)

# 상위 3개 상세 분석
top3 = df_sorted.head(3)
for i, (_, row) in enumerate(top3.iterrows(), 1):
    c = row['config']
    print(f"\n[추천 {i}] TP1 {c['tp1']}% → 본절 → TP2 {c['tp2']}%, SL {c['sl']}%, 시간스탑 {c['time_stop']}바")
    
    trades = simulate(fvgs, df_4h, df_15m, c)
    stats = analyze(trades)
    
    print(f"  - 전체 월평균: {row['total_mavg']:.2f}%")
    print(f"  - 최악의 해: {row['min_mavg']:.2f}%")
    print(f"  - 2025 월평균: {row['mavg_2025']:.2f}%")
    print(f"  - 연도별 편차: {row['std_mavg']:.2f}%")

