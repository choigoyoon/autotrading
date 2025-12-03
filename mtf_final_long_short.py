#!/usr/bin/env python3
"""
최종 MTF 롱/숏 전략

이전 잘 나왔던 롱 전략 로직 그대로 + 숏 추가
핵심: Entry-SL Gap 1~3% 범위에서 진입
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("최종 MTF 롱/숏 전략")
print("=" * 80)

# 데이터
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

df_1h = df_15m.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

print(f"15M: {len(df_15m):,}개, 1H: {len(df_1h):,}개")

# RSI, MACD
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df_1h['rsi'] = calc_rsi(df_1h['close'], 14)

def calc_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

df_1h = calc_macd(df_1h)
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# H/L 추출
def extract_hl_points(df):
    hist = df['macd_hist'].values
    high, low = df['high'].values, df['low'].values
    timestamps, rsi = df['datetime'].values, df['rsi'].values
    
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': pd.Timestamp(timestamps[max_idx]), 
                          'idx': max_idx, 'rsi': rsi[max_idx] if not np.isnan(rsi[max_idx]) else 50})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': pd.Timestamp(timestamps[min_idx]), 
                          'idx': min_idx, 'rsi': rsi[min_idx] if not np.isnan(rsi[min_idx]) else 50})
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
print(f"1H H/L points: {len(points_1h)}")

# 15M 스윙
def get_swing_points(df, window=5):
    highs, lows = [], []
    h, l = df['high'].values, df['low'].values
    t = df['datetime'].values
    for i in range(window, len(df) - window):
        if h[i] == max(h[i-window:i+window+1]):
            highs.append({'price': h[i], 'time': pd.Timestamp(t[i]), 'idx': i})
        if l[i] == min(l[i-window:i+window+1]):
            lows.append({'price': l[i], 'time': pd.Timestamp(t[i]), 'idx': i})
    return highs, lows

swing_h, swing_l = get_swing_points(df_15m)

# 다이버전스
def detect_bullish_div(df, start_idx, end_idx):
    if end_idx - start_idx < 20:
        return False
    window = df.iloc[start_idx:end_idx+1]
    lows, rsi = window['low'].values, window['rsi'].values
    n = len(window)
    if n < 20:
        return False
    first_idx = np.argmin(lows[:n//2])
    last_idx = n//2 + np.argmin(lows[n//2:])
    first_p, last_p = lows[first_idx], lows[last_idx]
    first_r = rsi[first_idx] if not np.isnan(rsi[first_idx]) else 50
    last_r = rsi[last_idx] if not np.isnan(rsi[last_idx]) else 50
    return last_p < first_p * 0.99 and last_r > first_r + 2

def detect_bearish_div(df, start_idx, end_idx):
    if end_idx - start_idx < 20:
        return False
    window = df.iloc[start_idx:end_idx+1]
    highs, rsi = window['high'].values, window['rsi'].values
    n = len(window)
    if n < 20:
        return False
    first_idx = np.argmax(highs[:n//2])
    last_idx = n//2 + np.argmax(highs[n//2:])
    first_p, last_p = highs[first_idx], highs[last_idx]
    first_r = rsi[first_idx] if not np.isnan(rsi[first_idx]) else 50
    last_r = rsi[last_idx] if not np.isnan(rsi[last_idx]) else 50
    return last_p > first_p * 1.01 and last_r < first_r - 2

# 15M 진입 신호 (이전 잘 나왔던 버전)
def detect_15m_entry_optimal(df_15m, swing_h, swing_l, pattern_time, sl_price, direction):
    search_start = pattern_time - pd.Timedelta(hours=6)
    search_end = pattern_time + pd.Timedelta(hours=24)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    window = df_15m[mask]
    
    if len(window) < 20:
        return None
    
    w_highs = [h for h in swing_h if search_start <= h['time'] <= search_end]
    w_lows = [l for l in swing_l if search_start <= l['time'] <= search_end]
    
    signals = {'trendline_break': False, 'hh_ll': False, 'hl_lh': False, 'volume': False}
    entry_candidates = []
    
    if direction == 'LONG':
        # 하락추세선 돌파
        if len(w_highs) >= 2:
            for i in range(len(w_highs) - 1):
                h1, h2 = w_highs[i], w_highs[i+1]
                if h2['price'] < h1['price'] * 0.998:
                    h2_idx = h2['idx']
                    after = df_15m[(df_15m.index > h2_idx) & (df_15m['datetime'] <= search_end)]
                    time_diff = h2['idx'] - h1['idx']
                    if time_diff > 0:
                        slope = (h2['price'] - h1['price']) / time_diff
                        for idx, row in after.iterrows():
                            tl_price = h2['price'] + slope * (idx - h2_idx)
                            if row['close'] > tl_price * 1.001:
                                signals['trendline_break'] = True
                                entry_candidates.append({'time': row['datetime'], 'price': row['close'], 'score': 3})
                                break
                    if signals['trendline_break']:
                        break
        
        # LH → HH
        if len(w_highs) >= 3:
            for i in range(len(w_highs) - 2):
                h1, h2, h3 = w_highs[i], w_highs[i+1], w_highs[i+2]
                if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                    signals['hh_ll'] = True
                    entry_candidates.append({'time': h3['time'], 'price': h3['price'], 'score': 2})
                    break
        
        # HL
        if len(w_lows) >= 3:
            for i in range(len(w_lows) - 2):
                l1, l2, l3 = w_lows[i], w_lows[i+1], w_lows[i+2]
                if l2['price'] < l1['price'] and l3['price'] > l2['price']:
                    signals['hl_lh'] = True
                    idx = l3['idx']
                    if idx < len(df_15m):
                        entry_candidates.append({'time': l3['time'], 'price': df_15m.iloc[idx]['close'], 'score': 2})
                    break
        
        # 양봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] > row['open'] and row['low'] <= sl_price * 1.02:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['volume'] = True
                    entry_candidates.append({'time': row['datetime'], 'price': row['close'], 'score': 1})
                    break
    
    else:  # SHORT
        # 상승추세선 돌파 (하향)
        if len(w_lows) >= 2:
            for i in range(len(w_lows) - 1):
                l1, l2 = w_lows[i], w_lows[i+1]
                if l2['price'] > l1['price'] * 1.002:  # 상승추세
                    l2_idx = l2['idx']
                    after = df_15m[(df_15m.index > l2_idx) & (df_15m['datetime'] <= search_end)]
                    time_diff = l2['idx'] - l1['idx']
                    if time_diff > 0:
                        slope = (l2['price'] - l1['price']) / time_diff
                        for idx, row in after.iterrows():
                            tl_price = l2['price'] + slope * (idx - l2_idx)
                            if row['close'] < tl_price * 0.999:  # 하향 돌파
                                signals['trendline_break'] = True
                                entry_candidates.append({'time': row['datetime'], 'price': row['close'], 'score': 3})
                                break
                    if signals['trendline_break']:
                        break
        
        # HL → LL
        if len(w_lows) >= 3:
            for i in range(len(w_lows) - 2):
                l1, l2, l3 = w_lows[i], w_lows[i+1], w_lows[i+2]
                if l2['price'] > l1['price'] and l3['price'] < l2['price']:
                    signals['hh_ll'] = True
                    entry_candidates.append({'time': l3['time'], 'price': l3['price'], 'score': 2})
                    break
        
        # LH
        if len(w_highs) >= 3:
            for i in range(len(w_highs) - 2):
                h1, h2, h3 = w_highs[i], w_highs[i+1], w_highs[i+2]
                if h2['price'] > h1['price'] and h3['price'] < h2['price']:
                    signals['hl_lh'] = True
                    idx = h3['idx']
                    if idx < len(df_15m):
                        entry_candidates.append({'time': h3['time'], 'price': df_15m.iloc[idx]['close'], 'score': 2})
                    break
        
        # 음봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] < row['open'] and row['high'] >= sl_price * 0.98:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['volume'] = True
                    entry_candidates.append({'time': row['datetime'], 'price': row['close'], 'score': 1})
                    break
    
    signal_count = sum([signals['trendline_break'], signals['hh_ll'], signals['hl_lh'], signals['volume']])
    
    # Entry-SL Gap 0.5~4% 범위에서 최적 진입점
    if len(entry_candidates) > 0:
        valid = []
        for c in entry_candidates:
            if direction == 'LONG':
                gap = (c['price'] - sl_price) / sl_price * 100
            else:
                gap = (sl_price - c['price']) / c['price'] * 100
            if 0.5 <= gap <= 4:
                c['entry_sl_gap'] = gap
                valid.append(c)
        
        if valid:
            valid.sort(key=lambda x: (-x['score'], x['time']))
            best = valid[0]
            return {
                'entry_time': best['time'],
                'entry_price': best['price'],
                'entry_sl_gap': best['entry_sl_gap'],
                'signal_count': signal_count,
                **signals
            }
    
    return None

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("백테스트")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # LONG: W 패턴
    if (points_1h[i]['type'] == 'L' and 
        points_1h[i+1]['type'] == 'H' and 
        points_1h[i+2]['type'] == 'L'):
        
        L1, H, L2 = points_1h[i], points_1h[i+1], points_1h[i+2]
        l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
        if abs(l_ratio) > 3:
            continue
        
        gap = (H['price'] - L2['price']) / L2['price'] * 100
        if gap < 2:
            continue
        
        sl_price = min(L1['price'], L2['price'])
        
        entry = detect_15m_entry_optimal(df_15m, swing_h, swing_l, L2['time'], sl_price, 'LONG')
        if entry is None or entry['signal_count'] < 1:
            continue
        
        # 다이버전스
        prev_h = [p for p in points_1h[:i] if p['type'] == 'H']
        div_1h = False
        if len(prev_h) > 0:
            trend_start = max(prev_h[-10:], key=lambda x: x['price'])
            div_1h = detect_bullish_div(df_1h, trend_start['idx'], L2['idx'])
        
        entry_time = entry['entry_time']
        entry_price = entry['entry_price']
        entry_sl_gap = entry['entry_sl_gap']
        
        # Entry-SL Gap 필터
        if not (1 <= entry_sl_gap <= 3):
            continue
        
        # 백테스트
        entry_idx = df_15m[df_15m['datetime'] >= entry_time].index
        if len(entry_idx) == 0:
            continue
        entry_idx = entry_idx[0]
        
        tp_pct = max(entry_sl_gap, gap * 0.5, 2)
        tp_pct = min(tp_pct, 5)
        tp_price = entry_price * (1 + tp_pct / 100)
        
        post = df_15m.iloc[entry_idx+1:entry_idx+500]
        pnl, exit_type, mfe = 0, 'TIMEOUT', 0
        
        for _, row in post.iterrows():
            mfe = max(mfe, (row['high'] - entry_price) / entry_price * 100)
            if row['high'] >= tp_price:
                pnl, exit_type = tp_pct, 'TP'
                break
            if row['low'] <= sl_price:
                pnl = (sl_price - entry_price) / entry_price * 100
                exit_type = 'SL'
                break
        
        results.append({
            'time': entry_time, 'direction': 'LONG', 'pattern': 'W',
            'entry': entry_price, 'sl': sl_price, 'entry_sl_gap': entry_sl_gap,
            'gap': gap, 'div_1h': div_1h, 'signal_count': entry['signal_count'],
            'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
        })
    
    # SHORT: M 패턴
    if (points_1h[i]['type'] == 'H' and 
        points_1h[i+1]['type'] == 'L' and 
        points_1h[i+2]['type'] == 'H'):
        
        H1, L, H2 = points_1h[i], points_1h[i+1], points_1h[i+2]
        h_ratio = (H2['price'] - H1['price']) / H1['price'] * 100
        if abs(h_ratio) > 3:
            continue
        
        gap = (H2['price'] - L['price']) / L['price'] * 100
        if gap < 2:
            continue
        
        sl_price = max(H1['price'], H2['price'])
        
        entry = detect_15m_entry_optimal(df_15m, swing_h, swing_l, H2['time'], sl_price, 'SHORT')
        if entry is None or entry['signal_count'] < 1:
            continue
        
        # 다이버전스
        prev_l = [p for p in points_1h[:i] if p['type'] == 'L']
        div_1h = False
        if len(prev_l) > 0:
            trend_start = min(prev_l[-10:], key=lambda x: x['price'])
            div_1h = detect_bearish_div(df_1h, trend_start['idx'], H2['idx'])
        
        entry_time = entry['entry_time']
        entry_price = entry['entry_price']
        entry_sl_gap = entry['entry_sl_gap']
        
        if not (1 <= entry_sl_gap <= 3):
            continue
        
        entry_idx = df_15m[df_15m['datetime'] >= entry_time].index
        if len(entry_idx) == 0:
            continue
        entry_idx = entry_idx[0]
        
        tp_pct = max(entry_sl_gap, gap * 0.5, 2)
        tp_pct = min(tp_pct, 5)
        tp_price = entry_price * (1 - tp_pct / 100)
        
        post = df_15m.iloc[entry_idx+1:entry_idx+500]
        pnl, exit_type, mfe = 0, 'TIMEOUT', 0
        
        for _, row in post.iterrows():
            mfe = max(mfe, (entry_price - row['low']) / entry_price * 100)
            if row['low'] <= tp_price:
                pnl, exit_type = tp_pct, 'TP'
                break
            if row['high'] >= sl_price:
                pnl = (entry_price - sl_price) / entry_price * 100
                exit_type = 'SL'
                break
        
        results.append({
            'time': entry_time, 'direction': 'SHORT', 'pattern': 'M',
            'entry': entry_price, 'sl': sl_price, 'entry_sl_gap': entry_sl_gap,
            'gap': gap, 'div_1h': div_1h, 'signal_count': entry['signal_count'],
            'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
        })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호 없음")
    exit()

# ============================================================
# 결과
# ============================================================
print("\n" + "=" * 80)
print("방향별 성과")
print("=" * 80)

print(f"\n{'방향':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 55)

for d in ['LONG', 'SHORT']:
    s = df_results[df_results['direction'] == d]
    if len(s) > 0:
        print(f"{d:>10} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f} {s['pnl'].sum():>10.1f}")

print("\n" + "=" * 80)
print("Gap 범위별 성과")
print("=" * 80)

print(f"\n{'Gap':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 45)

for low, high in [(2, 4), (4, 6), (6, 8), (8, 100)]:
    s = df_results[(df_results['gap'] >= low) & (df_results['gap'] < high)]
    if len(s) >= 5:
        print(f"{f'{low}-{high}%':>10} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f}")

print("\n" + "=" * 80)
print("Entry-SL Gap별 성과")
print("=" * 80)

print(f"\n{'Entry-SL':>12} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 50)

for low, high in [(1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3)]:
    s = df_results[(df_results['entry_sl_gap'] >= low) & (df_results['entry_sl_gap'] < high)]
    if len(s) >= 5:
        print(f"{f'{low}-{high}%':>12} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f}")

print("\n" + "=" * 80)
print("다이버전스별 성과")
print("=" * 80)

print(f"\n{'조건':>20} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 65)

conds = [
    ('전체', df_results),
    ('1H 다이버 있음', df_results[df_results['div_1h'] == True]),
    ('1H 다이버 없음', df_results[df_results['div_1h'] == False]),
    ('LONG + 다이버', df_results[(df_results['direction']=='LONG') & (df_results['div_1h']==True)]),
    ('SHORT + 다이버', df_results[(df_results['direction']=='SHORT') & (df_results['div_1h']==True)]),
]

for name, s in conds:
    if len(s) > 0:
        print(f"{name:>20} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f} {s['pnl'].sum():>10.1f}")

# 월별
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month')['pnl'].agg(['sum', 'count'])
monthly.columns = ['월PnL', '거래수']

long_c = len(df_results[df_results['direction']=='LONG'])
short_c = len(df_results[df_results['direction']=='SHORT'])

print(f"\n■ 전체:")
print(f"  총 거래: {len(df_results)}건 (롱 {long_c} + 숏 {short_c})")
print(f"  승률: {(df_results['pnl'] > 0).mean() * 100:.1f}%")
print(f"  평균 PnL: {df_results['pnl'].mean():.2f}%")
print(f"  월 평균: {monthly['월PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['월PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['월PnL'] > 0).sum()}/{len(monthly)}")

# 다이버 조건
div = df_results[df_results['div_1h'] == True]
if len(div) > 0:
    div = div.copy()
    div['month'] = pd.to_datetime(div['time']).dt.to_period('M')
    m_div = div.groupby('month')['pnl'].sum()
    print(f"\n■ 1H 다이버전스:")
    print(f"  총 거래: {len(div)}건")
    print(f"  승률: {(div['pnl'] > 0).mean() * 100:.1f}%")
    print(f"  평균 PnL: {div['pnl'].mean():.2f}%")
    print(f"  월 평균: {m_div.mean():.2f}%")
    print(f"  3x 레버리지: {m_div.mean()*3:.1f}%/월")

# 최종
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

total_monthly = monthly['월PnL'].mean()
div_monthly = m_div.mean() if len(div) > 0 else 0

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         최종 MTF 롱/숏 전략 결과                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 전체: {len(df_results)}건 (롱 {long_c} + 숏 {short_c})                                            │
│    승률: {(df_results['pnl']>0).mean()*100:.1f}%, 평균: {df_results['pnl'].mean():.2f}%                                          │
│    월 평균: {total_monthly:.2f}%, 3x: {total_monthly*3:.1f}%/월                                       │
│                                                                                  │
│  ■ 1H 다이버전스: {len(div)}건                                                     │
│    승률: {(div['pnl']>0).mean()*100 if len(div)>0 else 0:.1f}%, 평균: {div['pnl'].mean() if len(div)>0 else 0:.2f}%                                          │
│    월 평균: {div_monthly:.2f}%, 3x: {div_monthly*3:.1f}%/월                                       │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

df_results.to_csv('mtf_final_long_short_results.csv', index=False)
print("저장: mtf_final_long_short_results.csv")
