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
    macd = exp1 - exp2
    return macd, macd.ewm(span=9).mean()

df_4h['macd'], df_4h['macd_sig'] = calc_macd(df_4h)

def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

df_4h['atr'] = calc_atr(df_4h)

def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        p2, curr = df.iloc[i-2], df.iloc[i]
        if p2['high'] < curr['low']:
            fvgs.append({
                'idx': i,
                'time': curr['datetime'],
                'top': curr['low'],
                'bottom': p2['high'],
                'gap_size': (curr['low'] - p2['high']) / p2['high'] * 100,
                'body': abs(curr['close'] - curr['open']) / curr['open'] * 100,
                'atr': curr['atr'] if pd.notna(curr['atr']) else 0,
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
        l1_idx, l2_idx = lows_idx[-2], lows_idx[-1]
        l1_price, l2_price = window.iloc[l1_idx]['low'], window.iloc[l2_idx]['low']
        l1_macd, l2_macd = window.iloc[l1_idx]['macd'], window.iloc[l2_idx]['macd']
        if l2_price < l1_price and l2_macd > l1_macd:
            diver = True
    
    highs_idx = argrelextrema(window['high'].values, np.greater, order=3)[0]
    hh = False
    if len(highs_idx) >= 2:
        h1 = window.iloc[highs_idx[-2]]['high']
        h2 = window.iloc[highs_idx[-1]]['high']
        if h2 > h1:
            hh = True
    
    return {'diver': diver, 'hh': hh}

def simulate_final(fvgs, df_4h, df_15m, config):
    trades = []
    times = df_15m['datetime'].values
    lows = df_15m['low'].values
    highs = df_15m['high'].values
    opens = df_15m['open'].values
    closes = df_15m['close'].values
    
    for fvg in fvgs:
        cond = classify_fvg(df_4h, fvg)
        
        # 필터: 다이버만 O 제외
        if cond['diver'] and not cond['hh']:
            continue
        
        if fvg['body'] < config.get('min_body', 0):
            continue
        
        start = np.searchsorted(times, np.datetime64(fvg['time']))
        if start >= len(times) - 200:
            continue
        
        touch_idx = None
        for i in range(start+1, min(start+200, len(times))):
            if lows[i] <= fvg['top']:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        bars_to_touch = touch_idx - start
        if bars_to_touch > config.get('max_touch_bars', 999):
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        
        tp1_pct = config['tp1']
        tp2_pct = config['tp2']
        sl_pct = config['sl']
        
        tp1_p = ep * (1 + tp1_pct/100)
        tp2_p = ep * (1 + tp2_pct/100)
        sl_p = ep * (1 + sl_pct/100)
        
        result = None
        pnl = 0
        tp1_hit = False
        time_stop = config.get('time_stop', 0)
        partial_tp = config.get('partial_tp', False)
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            bars_in_trade = i - entry_idx
            
            # 시간 스탑 (TP1 전에만)
            if time_stop > 0 and bars_in_trade >= time_stop and not tp1_hit:
                exit_price = closes[i]
                pnl = (exit_price - ep) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            if not tp1_hit:
                if lows[i] <= sl_p:
                    result = 'SL'
                    pnl = sl_pct - 0.06
                    break
                
                if highs[i] >= tp1_p:
                    tp1_hit = True
                    if partial_tp:
                        pnl += tp1_pct / 2
                    continue
            
            if tp1_hit:
                if lows[i] <= ep:
                    result = 'BE'
                    pnl = pnl - 0.06 if partial_tp else -0.06
                    break
                
                if highs[i] >= tp2_p:
                    result = 'TP2'
                    pnl = (pnl + tp2_pct / 2 - 0.06) if partial_tp else (tp2_pct - 0.06)
                    break
        
        if result is None and tp1_hit:
            result = 'BE'
            pnl = pnl - 0.06 if partial_tp else -0.06
        
        if result:
            trades.append({
                'time': times[entry_idx],
                'result': result,
                'pnl': pnl,
            })
    
    return trades

def calc_stats(trades, label=""):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    
    results = df['result'].value_counts().to_dict()
    pnl = df['pnl'].sum()
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    months = len(df['month'].unique())
    
    cum_pnl = df['pnl'].cumsum()
    peak = cum_pnl.expanding().max()
    mdd = (cum_pnl - peak).min()
    
    sl_cnt = results.get('SL', 0)
    
    return {
        'n': len(trades),
        'results': results,
        'sl_rate': sl_cnt / len(trades) * 100,
        'win_rate': (len(trades) - sl_cnt) / len(trades) * 100,
        'pnl': pnl,
        'mdd': mdd,
        'mavg': pnl / months
    }

print("=" * 95)
print("🎯 최적 조합 탐색")
print("=" * 95)

# 시간스탑 최적화
print("\n[1] 시간스탑 최적화 (TP1 1.5% → 본절 → TP2 4%, SL -1.5%)")
print(f"{'시간스탑(바)':>12} {'거래':>6} {'SL%':>6} {'승률':>6} {'MDD':>7} {'월평균':>7}")
print("-" * 55)

for time_stop in [0, 24, 48, 72, 96, 120, 144]:
    config = {'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5, 'max_touch_bars': 20, 'time_stop': time_stop}
    trades = simulate_final(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        label = "없음" if time_stop == 0 else f"{time_stop}바({time_stop*15//60}h)"
        print(f"{label:>12} {s['n']:>6} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['mavg']:>6.2f}%")

# SL 최적화 (시간스탑 96바 적용)
print("\n[2] SL 최적화 (시간스탑 96바 적용)")
print(f"{'SL':>8} {'거래':>6} {'SL%':>6} {'승률':>6} {'MDD':>7} {'월평균':>7}")
print("-" * 50)

for sl in [-1.0, -1.5, -2.0, -2.5, -3.0]:
    config = {'tp1': 1.5, 'tp2': 4.0, 'sl': sl, 'max_touch_bars': 20, 'time_stop': 96}
    trades = simulate_final(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        print(f"{sl:>7.1f}% {s['n']:>6} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['mavg']:>6.2f}%")

# TP2 최적화 (시간스탑 96바 + SL -2% 적용)
print("\n[3] TP2 최적화 (시간스탑 96바, SL -2%)")
print(f"{'TP2':>8} {'거래':>6} {'SL%':>6} {'승률':>6} {'MDD':>7} {'월평균':>7}")
print("-" * 50)

for tp2 in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    config = {'tp1': 1.5, 'tp2': tp2, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}
    trades = simulate_final(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        print(f"{tp2:>7.1f}% {s['n']:>6} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['mavg']:>6.2f}%")

# 분할익절 비교
print("\n[4] 분할익절 효과 (시간스탑 96바, SL -2%, TP2 4%)")
print(f"{'모드':>15} {'거래':>6} {'SL%':>6} {'승률':>6} {'MDD':>7} {'월평균':>7}")
print("-" * 60)

for partial in [False, True]:
    config = {'tp1': 1.5, 'tp2': 4.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96, 'partial_tp': partial}
    trades = simulate_final(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        label = "전량 TP2" if not partial else "분할 (50%+50%)"
        print(f"{label:>15} {s['n']:>6} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['mavg']:>6.2f}%")

# 최종 비교
print("\n" + "=" * 95)
print("🏆 최종 전략 비교")
print("=" * 95)

strategies = [
    ("기존: 본절스탑만", {'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5, 'max_touch_bars': 20}),
    ("개선1: + 시간스탑 96바", {'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5, 'max_touch_bars': 20, 'time_stop': 96}),
    ("개선2: + SL -2%", {'tp1': 1.5, 'tp2': 4.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}),
    ("개선3: + 분할익절", {'tp1': 1.5, 'tp2': 4.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96, 'partial_tp': True}),
    ("최적: 시간96 + SL-2 + TP5", {'tp1': 1.5, 'tp2': 5.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}),
]

print(f"\n{'전략':>25} {'거래':>6} {'SL%':>6} {'승률':>6} {'MDD':>7} {'총PnL':>8} {'월평균':>7}")
print("-" * 80)

for name, config in strategies:
    trades = simulate_final(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        print(f"{name:>25} {s['n']:>6} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['pnl']:>7.1f}% {s['mavg']:>6.2f}%")

# Train/Test 검증
print("\n" + "=" * 95)
print("📊 Train/Test 검증 (최적 전략)")
print("=" * 95)

optimal_config = {'tp1': 1.5, 'tp2': 5.0, 'sl': -2.0, 'max_touch_bars': 20, 'time_stop': 96}

all_trades = simulate_final(fvgs, df_4h, df_15m, optimal_config)
df_trades = pd.DataFrame(all_trades)
df_trades['time'] = pd.to_datetime(df_trades['time'])

# Train: 2020-2024, Test: 2025
train = df_trades[df_trades['time'] < '2025-01-01']
test = df_trades[df_trades['time'] >= '2025-01-01']

print(f"\n[최적 전략: TP1 1.5% → 본절 → TP2 5%, SL -2%, 시간스탑 24시간]")
print(f"{'기간':>12} {'거래':>6} {'SL%':>6} {'승률':>6} {'총PnL':>8} {'월평균':>7}")
print("-" * 55)

for name, data in [("Train 2020-24", train), ("Test 2025", test), ("전체", df_trades)]:
    if len(data) < 5:
        continue
    results = data['result'].value_counts().to_dict()
    sl_cnt = results.get('SL', 0)
    months = len(data['time'].dt.to_period('M').unique())
    pnl = data['pnl'].sum()
    print(f"{name:>12} {len(data):>6} {sl_cnt/len(data)*100:>5.1f}% {(len(data)-sl_cnt)/len(data)*100:>5.1f}% {pnl:>7.1f}% {pnl/months:>6.2f}%")

# 결과 상세
print("\n[Test 2025 상세 결과]")
if len(test) > 0:
    results = test['result'].value_counts()
    for res, cnt in results.items():
        print(f"  {res}: {cnt}회 ({cnt/len(test)*100:.1f}%)")

