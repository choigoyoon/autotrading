#!/usr/bin/env python3
"""
MTF 롱/숏 통합 전략

기존 잘 나왔던 롱 전략 + 숏 전략 추가
+ MTF 다이버전스 필터
+ 리테스트 옵션

핵심:
- 롱: W패턴 + 상승 다이버전스
- 숏: M패턴 + 하락 다이버전스
- MTF 일치 필터로 승률 향상
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 롱/숏 통합 전략")
print("=" * 80)

# 데이터 로드
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
df_15m['rsi'] = calc_rsi(df_15m['close'], 14)

def calc_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

df_1h = calc_macd(df_1h)
df_15m = calc_macd(df_15m)
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

# 15M 스윙 포인트
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
print(f"15M Swing: H={len(swing_h)}, L={len(swing_l)}")

# ============================================================
# 다이버전스 감지
# ============================================================
def detect_bullish_div(df, start_idx, end_idx):
    """상승 다이버: 가격 LL + RSI HL"""
    if end_idx - start_idx < 20:
        return False
    window = df.iloc[start_idx:end_idx+1]
    lows, rsi = window['low'].values, window['rsi'].values
    
    n = len(window)
    first_low_idx = np.argmin(lows[:n//2])
    last_low_idx = n//2 + np.argmin(lows[n//2:])
    
    first_p, last_p = lows[first_low_idx], lows[last_low_idx]
    first_r = rsi[first_low_idx] if not np.isnan(rsi[first_low_idx]) else 50
    last_r = rsi[last_low_idx] if not np.isnan(rsi[last_low_idx]) else 50
    
    return last_p < first_p * 0.99 and last_r > first_r + 2

def detect_bearish_div(df, start_idx, end_idx):
    """하락 다이버: 가격 HH + RSI LH"""
    if end_idx - start_idx < 20:
        return False
    window = df.iloc[start_idx:end_idx+1]
    highs, rsi = window['high'].values, window['rsi'].values
    
    n = len(window)
    first_high_idx = np.argmax(highs[:n//2])
    last_high_idx = n//2 + np.argmax(highs[n//2:])
    
    first_p, last_p = highs[first_high_idx], highs[last_high_idx]
    first_r = rsi[first_high_idx] if not np.isnan(rsi[first_high_idx]) else 50
    last_r = rsi[last_high_idx] if not np.isnan(rsi[last_high_idx]) else 50
    
    return last_p > first_p * 1.01 and last_r < first_r - 2

# ============================================================
# 15M 진입 신호
# ============================================================
def detect_15m_signals(df_15m, swing_h, swing_l, pattern_time, sl_price, direction):
    search_start = pattern_time - pd.Timedelta(hours=6)
    search_end = pattern_time + pd.Timedelta(hours=12)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    window = df_15m[mask]
    
    if len(window) < 20:
        return {'count': 0, 'entry_time': None, 'entry_price': None}
    
    w_highs = [h for h in swing_h if search_start <= h['time'] <= search_end]
    w_lows = [l for l in swing_l if search_start <= l['time'] <= search_end]
    
    signals = {'hh_ll': False, 'hl_lh': False, 'volume': False, 'count': 0,
               'entry_time': None, 'entry_price': None}
    
    if direction == 'LONG':
        # LH → HH
        if len(w_highs) >= 3:
            for i in range(len(w_highs) - 2):
                h1, h2, h3 = w_highs[i], w_highs[i+1], w_highs[i+2]
                if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                    signals['hh_ll'] = True
                    signals['entry_time'] = h3['time']
                    signals['entry_price'] = h3['price']
                    break
        
        # HL
        if len(w_lows) >= 3:
            for i in range(len(w_lows) - 2):
                l1, l2, l3 = w_lows[i], w_lows[i+1], w_lows[i+2]
                if l2['price'] < l1['price'] and l3['price'] > l2['price']:
                    signals['hl_lh'] = True
                    break
        
        # 양봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] > row['open'] and row['low'] <= sl_price * 1.02:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['volume'] = True
                    if signals['entry_time'] is None:
                        signals['entry_time'] = row['datetime']
                        signals['entry_price'] = row['close']
                    break
    else:  # SHORT
        # HL → LL
        if len(w_lows) >= 3:
            for i in range(len(w_lows) - 2):
                l1, l2, l3 = w_lows[i], w_lows[i+1], w_lows[i+2]
                if l2['price'] > l1['price'] and l3['price'] < l2['price']:
                    signals['hh_ll'] = True
                    signals['entry_time'] = l3['time']
                    signals['entry_price'] = l3['price']
                    break
        
        # LH
        if len(w_highs) >= 3:
            for i in range(len(w_highs) - 2):
                h1, h2, h3 = w_highs[i], w_highs[i+1], w_highs[i+2]
                if h2['price'] > h1['price'] and h3['price'] < h2['price']:
                    signals['hl_lh'] = True
                    break
        
        # 음봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] < row['open'] and row['high'] >= sl_price * 0.98:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['volume'] = True
                    if signals['entry_time'] is None:
                        signals['entry_time'] = row['datetime']
                        signals['entry_price'] = row['close']
                    break
    
    signals['count'] = sum([signals['hh_ll'], signals['hl_lh'], signals['volume']])
    return signals

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # ================== LONG: W 패턴 ==================
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
        
        # 15M 신호
        signals = detect_15m_signals(df_15m, swing_h, swing_l, L2['time'], sl_price, 'LONG')
        
        if signals['count'] < 1 or signals['entry_time'] is None:
            continue
        
        entry_time = signals['entry_time']
        entry_price = signals['entry_price']
        entry_sl_gap = (entry_price - sl_price) / sl_price * 100
        
        if not (0.5 <= entry_sl_gap <= 4):
            continue
        
        # 다이버전스
        prev_h = [p for p in points_1h[:i] if p['type'] == 'H']
        if len(prev_h) > 0:
            trend_start = max(prev_h[-10:], key=lambda x: x['price'])
            div_1h = detect_bullish_div(df_1h, trend_start['idx'], L2['idx'])
            
            s15 = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
            e15 = L2['idx'] * 4 if L2['idx'] * 4 < len(df_15m) else len(df_15m) - 1
            div_15m = detect_bullish_div(df_15m, s15, e15)
        else:
            div_1h, div_15m = False, False
        
        mtf_div = div_1h and div_15m
        
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
            'gap': gap, 'div_1h': div_1h, 'div_15m': div_15m, 'mtf_div': mtf_div,
            'signal_count': signals['count'], 'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
        })
    
    # ================== SHORT: M 패턴 ==================
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
        
        # 15M 신호
        signals = detect_15m_signals(df_15m, swing_h, swing_l, H2['time'], sl_price, 'SHORT')
        
        if signals['count'] < 1 or signals['entry_time'] is None:
            continue
        
        entry_time = signals['entry_time']
        entry_price = signals['entry_price']
        entry_sl_gap = (sl_price - entry_price) / entry_price * 100
        
        if not (0.5 <= entry_sl_gap <= 4):
            continue
        
        # 다이버전스
        prev_l = [p for p in points_1h[:i] if p['type'] == 'L']
        if len(prev_l) > 0:
            trend_start = min(prev_l[-10:], key=lambda x: x['price'])
            div_1h = detect_bearish_div(df_1h, trend_start['idx'], H2['idx'])
            
            s15 = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
            e15 = H2['idx'] * 4 if H2['idx'] * 4 < len(df_15m) else len(df_15m) - 1
            div_15m = detect_bearish_div(df_15m, s15, e15)
        else:
            div_1h, div_15m = False, False
        
        mtf_div = div_1h and div_15m
        
        # 백테스트
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
            'gap': gap, 'div_1h': div_1h, 'div_15m': div_15m, 'mtf_div': mtf_div,
            'signal_count': signals['count'], 'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
        })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호 없음")
    exit()

# ============================================================
# 결과 분석
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
print("MTF 다이버전스별 성과")
print("=" * 80)

print(f"\n{'조건':>25} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 70)

conditions = [
    ('전체', df_results),
    ('1H 다이버', df_results[df_results['div_1h'] == True]),
    ('MTF 다이버 일치', df_results[df_results['mtf_div'] == True]),
    ('LONG + MTF 다이버', df_results[(df_results['direction']=='LONG') & (df_results['mtf_div']==True)]),
    ('SHORT + MTF 다이버', df_results[(df_results['direction']=='SHORT') & (df_results['mtf_div']==True)]),
]

for name, s in conditions:
    if len(s) > 0:
        print(f"{name:>25} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f} {s['pnl'].sum():>10.1f}")

print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>40} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 85)

optimal = [
    ('Gap>=4%', df_results[df_results['gap'] >= 4]),
    ('Gap>=4% + MTF다이버', df_results[(df_results['gap'] >= 4) & (df_results['mtf_div'] == True)]),
    ('Gap>=4% + Entry-SL 1-3%', df_results[(df_results['gap'] >= 4) & (df_results['entry_sl_gap'] >= 1) & (df_results['entry_sl_gap'] <= 3)]),
    ('Gap>=4% + MTF다이버 + Entry-SL 1-3%', df_results[(df_results['gap'] >= 4) & (df_results['mtf_div'] == True) & (df_results['entry_sl_gap'] >= 1) & (df_results['entry_sl_gap'] <= 3)]),
    ('LONG + Gap>=4% + MTF', df_results[(df_results['direction']=='LONG') & (df_results['gap'] >= 4) & (df_results['mtf_div'] == True)]),
    ('SHORT + Gap>=4% + MTF', df_results[(df_results['direction']=='SHORT') & (df_results['gap'] >= 4) & (df_results['mtf_div'] == True)]),
]

for name, s in optimal:
    if len(s) >= 5:
        print(f"{name:>40} {len(s):>8} {(s['pnl']>0).mean()*100:>10.1f} {s['pnl'].mean():>10.2f} {s['pnl'].sum():>10.1f}")

# ============================================================
# 월별 성과
# ============================================================
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month')['pnl'].agg(['sum', 'count'])
monthly.columns = ['월PnL', '거래수']

print(f"\n■ 전체:")
print(f"  총 거래: {len(df_results)}건 (롱 {len(df_results[df_results['direction']=='LONG'])} + 숏 {len(df_results[df_results['direction']=='SHORT'])})")
print(f"  승률: {(df_results['pnl'] > 0).mean() * 100:.1f}%")
print(f"  평균 PnL: {df_results['pnl'].mean():.2f}%")
print(f"  월 평균: {monthly['월PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['월PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['월PnL'] > 0).sum()}/{len(monthly)} ({(monthly['월PnL'] > 0).mean()*100:.1f}%)")

# MTF 다이버
mtf = df_results[df_results['mtf_div'] == True]
if len(mtf) > 0:
    mtf = mtf.copy()
    mtf['month'] = pd.to_datetime(mtf['time']).dt.to_period('M')
    m_mtf = mtf.groupby('month')['pnl'].sum()
    
    print(f"\n■ MTF 다이버전스:")
    print(f"  총 거래: {len(mtf)}건")
    print(f"  승률: {(mtf['pnl'] > 0).mean() * 100:.1f}%")
    print(f"  평균 PnL: {mtf['pnl'].mean():.2f}%")
    print(f"  월 평균: {m_mtf.mean():.2f}%")
    print(f"  3x 레버리지: {m_mtf.mean()*3:.1f}%/월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

long_c = len(df_results[df_results['direction']=='LONG'])
short_c = len(df_results[df_results['direction']=='SHORT'])
total_win = (df_results['pnl'] > 0).mean() * 100
total_avg = df_results['pnl'].mean()
total_monthly = monthly['월PnL'].mean()

mtf_c = len(mtf) if len(mtf) > 0 else 0
mtf_win = (mtf['pnl'] > 0).mean() * 100 if len(mtf) > 0 else 0
mtf_avg = mtf['pnl'].mean() if len(mtf) > 0 else 0
mtf_monthly = m_mtf.mean() if len(mtf) > 0 else 0

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       MTF 롱/숏 통합 전략 결과                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 전체:                                                                         │
│    총 거래: {len(df_results)}건 (롱 {long_c} + 숏 {short_c})                                            │
│    승률: {total_win:.1f}%, 평균 PnL: {total_avg:.2f}%                                          │
│    월 평균: {total_monthly:.2f}%, 3x: {total_monthly*3:.1f}%/월                                       │
│                                                                                  │
│  ■ MTF 다이버전스:                                                               │
│    총 거래: {mtf_c}건                                                              │
│    승률: {mtf_win:.1f}%, 평균 PnL: {mtf_avg:.2f}%                                          │
│    월 평균: {mtf_monthly:.2f}%, 3x: {mtf_monthly*3:.1f}%/월                                       │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 핵심 원칙:                                                                    │
│    1. 패턴 인식 (W/M)                                                            │
│    2. Gap 확인 (에너지 축적)                                                     │
│    3. MTF 다이버전스 (하락/상승 터진 곳부터)                                     │
│    4. 15M 진입 신호 (HH/LL 전환, HL/LH, 캔들+거래량)                             │
│    5. 돌파 → 조절 → 1차 → 2차                                                   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

df_results.to_csv('mtf_all_positions_results.csv', index=False)
print("결과 저장: mtf_all_positions_results.csv")
