#!/usr/bin/env python3
"""
완전한 MTF 전략: 롱 + 숏

롱 패턴:
- W (더블바텀)
- Inverse Head & Shoulders (역헤드앤숄더)
- 상승 다이버전스

숏 패턴:
- M (더블탑)
- Head & Shoulders (헤드앤숄더)
- 하락 다이버전스

핵심:
- 하락/상승 추세 시작 ~ 추세 돌파까지 전체 그림
- MTF 다이버전스 (1H + 15M)
- 돌파 → 조절 → 진입
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("완전한 MTF 롱/숏 전략")
print("=" * 80)

# 데이터 로드
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

# 1시간봉 생성
df_1h = df_15m.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

print(f"15M: {len(df_15m):,}개")
print(f"1H: {len(df_1h):,}개")

# RSI 계산
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_1h['rsi'] = calc_rsi(df_1h['close'], 14)
df_15m['rsi'] = calc_rsi(df_15m['close'], 14)

# MACD
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
df_1h['vol_ma20'] = df_1h['volume'].rolling(20).mean()

# H/L 포인트 추출
def extract_hl_points(df):
    hist = df['macd_hist'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    timestamps = df['datetime'].values
    rsi = df['rsi'].values
    
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({
                'type': 'H', 
                'price': high[max_idx], 
                'close': close[max_idx],
                'time': pd.Timestamp(timestamps[max_idx]), 
                'idx': max_idx,
                'rsi': rsi[max_idx] if not np.isnan(rsi[max_idx]) else 50
            })
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({
                'type': 'L', 
                'price': low[min_idx],
                'close': close[min_idx], 
                'time': pd.Timestamp(timestamps[min_idx]), 
                'idx': min_idx,
                'rsi': rsi[min_idx] if not np.isnan(rsi[min_idx]) else 50
            })
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
points_15m = extract_hl_points(df_15m)

print(f"1H H/L points: {len(points_1h)}")
print(f"15M H/L points: {len(points_15m)}")

# ============================================================
# 다이버전스 감지 (개선)
# ============================================================
def detect_bullish_divergence(df, start_idx, end_idx):
    """상승 다이버전스: 가격 LL + RSI/MACD HL"""
    if start_idx >= end_idx or end_idx - start_idx < 20:
        return False, None
    
    window = df.iloc[start_idx:end_idx+1]
    lows = window['low'].values
    rsi = window['rsi'].values
    
    # 구간 나누기
    third = len(window) // 3
    if third < 5:
        return False, None
    
    # 첫 구간 vs 마지막 구간
    first_low_idx = np.argmin(lows[:third*2])
    last_low_idx = third*2 + np.argmin(lows[third*2:])
    
    first_low_price = lows[first_low_idx]
    last_low_price = lows[last_low_idx]
    first_low_rsi = rsi[first_low_idx] if not np.isnan(rsi[first_low_idx]) else 50
    last_low_rsi = rsi[last_low_idx] if not np.isnan(rsi[last_low_idx]) else 50
    
    # 상승 다이버: 가격 LL + RSI HL
    if last_low_price < first_low_price * 0.995 and last_low_rsi > first_low_rsi + 1:
        return True, {
            'type': 'bullish',
            'price_diff': (last_low_price - first_low_price) / first_low_price * 100,
            'rsi_diff': last_low_rsi - first_low_rsi
        }
    
    return False, None

def detect_bearish_divergence(df, start_idx, end_idx):
    """하락 다이버전스: 가격 HH + RSI LH"""
    if start_idx >= end_idx or end_idx - start_idx < 20:
        return False, None
    
    window = df.iloc[start_idx:end_idx+1]
    highs = window['high'].values
    rsi = window['rsi'].values
    
    third = len(window) // 3
    if third < 5:
        return False, None
    
    first_high_idx = np.argmax(highs[:third*2])
    last_high_idx = third*2 + np.argmax(highs[third*2:])
    
    first_high_price = highs[first_high_idx]
    last_high_price = highs[last_high_idx]
    first_high_rsi = rsi[first_high_idx] if not np.isnan(rsi[first_high_idx]) else 50
    last_high_rsi = rsi[last_high_idx] if not np.isnan(rsi[last_high_idx]) else 50
    
    # 하락 다이버: 가격 HH + RSI LH
    if last_high_price > first_high_price * 1.005 and last_high_rsi < first_high_rsi - 1:
        return True, {
            'type': 'bearish',
            'price_diff': (last_high_price - first_high_price) / first_high_price * 100,
            'rsi_diff': last_high_rsi - first_high_rsi
        }
    
    return False, None

# ============================================================
# 패턴 감지
# ============================================================
def detect_patterns(points):
    """
    롱 패턴: W (L-H-L), IHS (L-H-L-H-L 중앙 L이 가장 낮음)
    숏 패턴: M (H-L-H), HS (H-L-H-L-H 중앙 H가 가장 높음)
    """
    patterns = []
    
    for i in range(len(points) - 4):
        # W 패턴 (더블바텀): L1-H-L2
        if (points[i]['type'] == 'L' and 
            points[i+1]['type'] == 'H' and 
            points[i+2]['type'] == 'L'):
            
            L1, H, L2 = points[i], points[i+1], points[i+2]
            l_ratio = abs(L2['price'] - L1['price']) / L1['price'] * 100
            gap = (H['price'] - min(L1['price'], L2['price'])) / min(L1['price'], L2['price']) * 100
            
            if l_ratio <= 5 and gap >= 2:  # L1 ≈ L2, Gap >= 2%
                patterns.append({
                    'type': 'W',
                    'direction': 'LONG',
                    'points': [L1, H, L2],
                    'neckline': H['price'],
                    'sl': min(L1['price'], L2['price']),
                    'gap': gap,
                    'time': L2['time'],
                    'idx': i
                })
        
        # M 패턴 (더블탑): H1-L-H2
        if (points[i]['type'] == 'H' and 
            points[i+1]['type'] == 'L' and 
            points[i+2]['type'] == 'H'):
            
            H1, L, H2 = points[i], points[i+1], points[i+2]
            h_ratio = abs(H2['price'] - H1['price']) / H1['price'] * 100
            gap = (max(H1['price'], H2['price']) - L['price']) / L['price'] * 100
            
            if h_ratio <= 5 and gap >= 2:  # H1 ≈ H2, Gap >= 2%
                patterns.append({
                    'type': 'M',
                    'direction': 'SHORT',
                    'points': [H1, L, H2],
                    'neckline': L['price'],
                    'sl': max(H1['price'], H2['price']),
                    'gap': gap,
                    'time': H2['time'],
                    'idx': i
                })
        
        # IHS (역헤드앤숄더): L1-H1-L2-H2-L3 (L2가 가장 낮음)
        if i + 4 < len(points):
            if (points[i]['type'] == 'L' and 
                points[i+1]['type'] == 'H' and 
                points[i+2]['type'] == 'L' and
                points[i+3]['type'] == 'H' and
                points[i+4]['type'] == 'L'):
                
                L1, H1, L2, H2, L3 = points[i], points[i+1], points[i+2], points[i+3], points[i+4]
                
                # L2가 가장 낮고, L1/L3가 비슷
                if (L2['price'] < L1['price'] * 0.98 and 
                    L2['price'] < L3['price'] * 0.98 and
                    abs(L1['price'] - L3['price']) / L1['price'] < 0.03):
                    
                    neckline = min(H1['price'], H2['price'])
                    gap = (neckline - L2['price']) / L2['price'] * 100
                    
                    if gap >= 3:
                        patterns.append({
                            'type': 'IHS',
                            'direction': 'LONG',
                            'points': [L1, H1, L2, H2, L3],
                            'neckline': neckline,
                            'sl': L2['price'],
                            'gap': gap,
                            'time': L3['time'],
                            'idx': i
                        })
        
        # HS (헤드앤숄더): H1-L1-H2-L2-H3 (H2가 가장 높음)
        if i + 4 < len(points):
            if (points[i]['type'] == 'H' and 
                points[i+1]['type'] == 'L' and 
                points[i+2]['type'] == 'H' and
                points[i+3]['type'] == 'L' and
                points[i+4]['type'] == 'H'):
                
                H1, L1, H2, L2, H3 = points[i], points[i+1], points[i+2], points[i+3], points[i+4]
                
                # H2가 가장 높고, H1/H3가 비슷
                if (H2['price'] > H1['price'] * 1.02 and 
                    H2['price'] > H3['price'] * 1.02 and
                    abs(H1['price'] - H3['price']) / H1['price'] < 0.03):
                    
                    neckline = max(L1['price'], L2['price'])
                    gap = (H2['price'] - neckline) / neckline * 100
                    
                    if gap >= 3:
                        patterns.append({
                            'type': 'HS',
                            'direction': 'SHORT',
                            'points': [H1, L1, H2, L2, H3],
                            'neckline': neckline,
                            'sl': H2['price'],
                            'gap': gap,
                            'time': H3['time'],
                            'idx': i
                        })
    
    return patterns

patterns_1h = detect_patterns(points_1h)
print(f"\n패턴 감지 완료:")
print(f"  W (더블바텀): {len([p for p in patterns_1h if p['type'] == 'W'])}개")
print(f"  M (더블탑): {len([p for p in patterns_1h if p['type'] == 'M'])}개")
print(f"  IHS (역헤숄): {len([p for p in patterns_1h if p['type'] == 'IHS'])}개")
print(f"  HS (헤숄): {len([p for p in patterns_1h if p['type'] == 'HS'])}개")

# ============================================================
# 15M 진입 신호 (롱/숏 공통)
# ============================================================
def detect_15m_entry_signals(df_15m, pattern, direction):
    """15M 진입 신호 감지"""
    
    pattern_time = pattern['time']
    search_start = pattern_time - pd.Timedelta(hours=6)
    search_end = pattern_time + pd.Timedelta(hours=24)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    window = df_15m[mask].copy()
    
    if len(window) < 20:
        return {'signal_count': 0, 'entry_time': None, 'entry_price': None}
    
    signals = {
        'trendline_break': False,
        'hh_or_ll': False,  # 롱: LH→HH, 숏: HL→LL
        'hl_or_lh': False,  # 롱: HL, 숏: LH
        'candle_volume': False,
        'signal_count': 0,
        'entry_time': None,
        'entry_price': None
    }
    
    # 스윙 포인트
    highs, lows = [], []
    for i in range(5, len(window) - 5):
        idx = window.index[i]
        if window.loc[idx, 'high'] == window.iloc[i-5:i+6]['high'].max():
            highs.append({'price': window.loc[idx, 'high'], 'idx': idx, 'time': window.loc[idx, 'datetime']})
        if window.loc[idx, 'low'] == window.iloc[i-5:i+6]['low'].min():
            lows.append({'price': window.loc[idx, 'low'], 'idx': idx, 'time': window.loc[idx, 'datetime']})
    
    if direction == 'LONG':
        # LH → HH
        if len(highs) >= 3:
            for i in range(len(highs) - 2):
                h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
                if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                    signals['hh_or_ll'] = True
                    signals['entry_time'] = h3['time']
                    signals['entry_price'] = h3['price']
                    break
        
        # HL (저점 올림)
        if len(lows) >= 3:
            for i in range(len(lows) - 2):
                l1, l2, l3 = lows[i], lows[i+1], lows[i+2]
                if l2['price'] < l1['price'] and l3['price'] > l2['price']:
                    signals['hl_or_lh'] = True
                    break
        
        # 양봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] > row['open'] and row['low'] <= pattern['sl'] * 1.02:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['candle_volume'] = True
                    if signals['entry_time'] is None:
                        signals['entry_time'] = row['datetime']
                        signals['entry_price'] = row['close']
                    break
    
    else:  # SHORT
        # HL → LL
        if len(lows) >= 3:
            for i in range(len(lows) - 2):
                l1, l2, l3 = lows[i], lows[i+1], lows[i+2]
                if l2['price'] > l1['price'] and l3['price'] < l2['price']:
                    signals['hh_or_ll'] = True
                    signals['entry_time'] = l3['time']
                    signals['entry_price'] = l3['price']
                    break
        
        # LH (고점 낮춤)
        if len(highs) >= 3:
            for i in range(len(highs) - 2):
                h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
                if h2['price'] > h1['price'] and h3['price'] < h2['price']:
                    signals['hl_or_lh'] = True
                    break
        
        # 음봉 + 거래량
        for idx in window.index:
            row = window.loc[idx]
            if row['close'] < row['open'] and row['high'] >= pattern['sl'] * 0.98:
                if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                    signals['candle_volume'] = True
                    if signals['entry_time'] is None:
                        signals['entry_time'] = row['datetime']
                        signals['entry_price'] = row['close']
                    break
    
    signals['signal_count'] = sum([
        signals['trendline_break'],
        signals['hh_or_ll'],
        signals['hl_or_lh'],
        signals['candle_volume']
    ])
    
    return signals

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("롱/숏 백테스트 실행")
print("=" * 80)

results = []

for pattern in patterns_1h:
    direction = pattern['direction']
    gap = pattern['gap']
    
    # Gap 필터
    if gap < 4:
        continue
    
    # 추세 시작점 찾기
    pattern_idx = pattern['idx']
    
    if direction == 'LONG':
        # 이전 H 중 가장 높은 것
        prev_h = [p for p in points_1h[:pattern_idx] if p['type'] == 'H']
        if len(prev_h) < 1:
            continue
        trend_start = max(prev_h[-10:], key=lambda x: x['price']) if len(prev_h) >= 10 else prev_h[-1]
        
        # 다이버전스
        last_l = pattern['points'][-1]  # 마지막 L
        div_1h, _ = detect_bullish_divergence(df_1h, trend_start['idx'], last_l['idx'])
        
        start_15m = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
        end_15m = last_l['idx'] * 4 if last_l['idx'] * 4 < len(df_15m) else len(df_15m) - 1
        div_15m, _ = detect_bullish_divergence(df_15m, start_15m, end_15m)
        
    else:  # SHORT
        # 이전 L 중 가장 낮은 것
        prev_l = [p for p in points_1h[:pattern_idx] if p['type'] == 'L']
        if len(prev_l) < 1:
            continue
        trend_start = min(prev_l[-10:], key=lambda x: x['price']) if len(prev_l) >= 10 else prev_l[-1]
        
        # 다이버전스
        last_h = pattern['points'][-1]  # 마지막 H
        div_1h, _ = detect_bearish_divergence(df_1h, trend_start['idx'], last_h['idx'])
        
        start_15m = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
        end_15m = last_h['idx'] * 4 if last_h['idx'] * 4 < len(df_15m) else len(df_15m) - 1
        div_15m, _ = detect_bearish_divergence(df_15m, start_15m, end_15m)
    
    mtf_divergence = div_1h and div_15m
    
    # 15M 신호
    signals = detect_15m_entry_signals(df_15m, pattern, direction)
    
    if signals['signal_count'] < 1:
        continue
    
    # 진입 가격 결정
    if signals['entry_time'] is not None:
        entry_time = signals['entry_time']
        entry_price = signals['entry_price']
    else:
        # 패턴 완성 후 진입
        if direction == 'LONG':
            entry_price = pattern['sl'] * 1.015
        else:
            entry_price = pattern['sl'] * 0.985
        entry_time = pattern['time'] + pd.Timedelta(hours=6)
    
    sl_price = pattern['sl']
    
    # Entry-SL Gap
    if direction == 'LONG':
        entry_sl_gap = (entry_price - sl_price) / sl_price * 100
    else:
        entry_sl_gap = (sl_price - entry_price) / entry_price * 100
    
    if not (1 <= entry_sl_gap <= 4):
        continue
    
    # 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    tp_pct = 5.0
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    
    for _, row in post.iterrows():
        if direction == 'LONG':
            current_pnl = (row['high'] - entry_price) / entry_price * 100
            mfe = max(mfe, current_pnl)
            
            if row['high'] >= entry_price * (1 + tp_pct / 100):
                pnl = tp_pct
                exit_type = 'TP'
                break
            if row['low'] <= sl_price:
                pnl = (sl_price - entry_price) / entry_price * 100
                exit_type = 'SL'
                break
        else:  # SHORT
            current_pnl = (entry_price - row['low']) / entry_price * 100
            mfe = max(mfe, current_pnl)
            
            if row['low'] <= entry_price * (1 - tp_pct / 100):
                pnl = tp_pct
                exit_type = 'TP'
                break
            if row['high'] >= sl_price:
                pnl = (entry_price - sl_price) / entry_price * 100
                exit_type = 'SL'
                break
    
    results.append({
        'time': entry_time,
        'direction': direction,
        'pattern': pattern['type'],
        'entry': entry_price,
        'sl': sl_price,
        'entry_sl_gap': entry_sl_gap,
        'gap': gap,
        'div_1h': div_1h,
        'div_15m': div_15m,
        'mtf_divergence': mtf_divergence,
        'signal_count': signals['signal_count'],
        'pnl': pnl,
        'mfe': mfe,
        'exit_type': exit_type
    })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호가 없습니다.")
    exit()

# ============================================================
# 결과 분석
# ============================================================
print("\n" + "=" * 80)
print("방향별 성과")
print("=" * 80)

print(f"\n{'방향':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 55)

for direction in ['LONG', 'SHORT']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{direction:>10} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

print("\n" + "=" * 80)
print("패턴별 성과")
print("=" * 80)

print(f"\n{'패턴':>10} {'방향':>8} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 55)

for pattern_type in ['W', 'M', 'IHS', 'HS']:
    subset = df_results[df_results['pattern'] == pattern_type]
    if len(subset) > 0:
        direction = subset['direction'].iloc[0]
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        print(f"{pattern_type:>10} {direction:>8} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f}")

print("\n" + "=" * 80)
print("MTF 다이버전스별 성과")
print("=" * 80)

print(f"\n{'조건':>25} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 70)

conditions = [
    ('전체', df_results),
    ('MTF 다이버 있음', df_results[df_results['mtf_divergence'] == True]),
    ('MTF 다이버 없음', df_results[df_results['mtf_divergence'] == False]),
    ('LONG + MTF 다이버', df_results[(df_results['direction'] == 'LONG') & (df_results['mtf_divergence'] == True)]),
    ('SHORT + MTF 다이버', df_results[(df_results['direction'] == 'SHORT') & (df_results['mtf_divergence'] == True)]),
]

for name, subset in conditions:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{name:>25} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>45} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 95)

optimal_conditions = [
    ('Gap>=5%', df_results[df_results['gap'] >= 5]),
    ('Gap>=5% + MTF다이버', df_results[(df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True)]),
    ('Gap>=5% + 신호2+', df_results[(df_results['gap'] >= 5) & (df_results['signal_count'] >= 2)]),
    ('Gap>=5% + MTF다이버 + 신호2+', df_results[(df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True) & (df_results['signal_count'] >= 2)]),
    ('LONG + Gap>=5% + MTF다이버', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True)]),
    ('SHORT + Gap>=5% + MTF다이버', df_results[(df_results['direction'] == 'SHORT') & (df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True)]),
]

for name, subset in optimal_conditions:
    if len(subset) >= 3:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{name:>45} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 월별 성과
# ============================================================
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month').agg({
    'pnl': ['sum', 'count']
}).round(2)
monthly.columns = ['월PnL', '거래수']

print(f"\n전체 전략:")
print(f"  총 거래: {len(df_results)}건")
print(f"  승률: {(df_results['pnl'] > 0).mean() * 100:.1f}%")
print(f"  평균 PnL: {df_results['pnl'].mean():.2f}%")
print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['월PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['월PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['월PnL'] > 0).sum()}/{len(monthly)} ({(monthly['월PnL'] > 0).mean()*100:.1f}%)")

# MTF 다이버 조건
mtf_div = df_results[df_results['mtf_divergence'] == True]
if len(mtf_div) > 0:
    mtf_div = mtf_div.copy()
    mtf_div['month'] = pd.to_datetime(mtf_div['time']).dt.to_period('M')
    monthly_mtf = mtf_div.groupby('month')['pnl'].agg(['sum', 'count']).round(2)
    monthly_mtf.columns = ['월PnL', '거래수']
    
    print(f"\nMTF 다이버전스 조건:")
    print(f"  총 거래: {len(mtf_div)}건")
    print(f"  승률: {(mtf_div['pnl'] > 0).mean() * 100:.1f}%")
    print(f"  평균 PnL: {mtf_div['pnl'].mean():.2f}%")
    print(f"  평균 거래수: {monthly_mtf['거래수'].mean():.1f}건/월")
    print(f"  월 평균 수익: {monthly_mtf['월PnL'].mean():.2f}%")
    print(f"  3x 레버리지: {monthly_mtf['월PnL'].mean()*3:.1f}%/월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

total_win = (df_results['pnl'] > 0).mean() * 100
total_avg = df_results['pnl'].mean()
total_monthly = monthly['월PnL'].mean()

mtf_win = (mtf_div['pnl'] > 0).mean() * 100 if len(mtf_div) > 0 else 0
mtf_avg = mtf_div['pnl'].mean() if len(mtf_div) > 0 else 0
mtf_monthly = monthly_mtf['월PnL'].mean() if len(mtf_div) > 0 else 0

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       완전한 MTF 롱/숏 전략 결과                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 패턴:                                                                         │
│    롱: W (더블바텀), IHS (역헤숄)                                                │
│    숏: M (더블탑), HS (헤숄)                                                     │
│                                                                                  │
│  ■ 전체 성과:                                                                    │
│    - 총 거래: {len(df_results)}건 (롱 {len(df_results[df_results['direction']=='LONG'])} + 숏 {len(df_results[df_results['direction']=='SHORT'])})                                          │
│    - 승률: {total_win:.1f}%                                                              │
│    - 평균 PnL: {total_avg:.2f}%                                                        │
│    - 월 평균: {total_monthly:.2f}%                                                      │
│    - 3x 레버리지: {total_monthly*3:.1f}%/월                                            │
│                                                                                  │
│  ■ MTF 다이버전스 적용 시:                                                       │
│    - 총 거래: {len(mtf_div)}건                                                          │
│    - 승률: {mtf_win:.1f}%                                                              │
│    - 평균 PnL: {mtf_avg:.2f}%                                                        │
│    - 월 평균: {mtf_monthly:.2f}%                                                      │
│    - 3x 레버리지: {mtf_monthly*3:.1f}%/월                                            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 핵심 원칙:                                                                    │
│    "하락/상승 추세 시작 ~ 추세 돌파까지 전체 그림을 봐야 한다"                   │
│    "MTF 다이버전스 (1H + 15M) 일치해야 진짜"                                     │
│    "돌파 → 조절 → 1차 → 2차"                                                    │
│                                                                                  │
│  ■ 진입 체크리스트:                                                              │
│    ✓ 패턴 인식 (W, M, IHS, HS)                                                   │
│    ✓ Gap >= 4~5% (에너지 축적)                                                   │
│    ✓ MTF 다이버전스 확인                                                         │
│    ✓ 15M 진입 신호 (HH/LL 전환, HL/LH, 캔들+거래량)                              │
│    ✓ Entry-SL Gap 1~4%                                                           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_long_short_results.csv', index=False)
print("\n결과 저장: mtf_long_short_results.csv")
