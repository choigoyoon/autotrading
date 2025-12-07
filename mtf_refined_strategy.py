#!/usr/bin/env python3
"""
개선된 MTF 롱/숏 전략

문제점 수정:
1. 리테스트 진입 로직 추가
2. 다이버전스 기준점: "하락/상승 터진 곳"부터
3. 넥라인 돌파 후 조절 대기

핵심:
"돌파 → 조절 → 1차 → 2차" 흐름
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("개선된 MTF 롱/숏 전략")
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
    high, low, close = df['high'].values, df['low'].values, df['close'].values
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

# ============================================================
# 다이버전스 (하락/상승 터진 곳부터)
# ============================================================
def find_trend_start_for_long(points, current_idx):
    """롱: 하락 시작점 (RSI 고점 + 가격 고점) 찾기"""
    prev_h = [p for p in points[:current_idx] if p['type'] == 'H']
    if len(prev_h) < 2:
        return None
    
    # 최근 15개 H 중 가장 높은 RSI를 가진 H (하락 시작점)
    recent_h = prev_h[-15:] if len(prev_h) >= 15 else prev_h
    best_h = max(recent_h, key=lambda x: x['rsi'])
    
    # 그 H가 상당히 높아야 함 (RSI > 60)
    if best_h['rsi'] > 55:
        return best_h
    return None

def find_trend_start_for_short(points, current_idx):
    """숏: 상승 시작점 (RSI 저점 + 가격 저점) 찾기"""
    prev_l = [p for p in points[:current_idx] if p['type'] == 'L']
    if len(prev_l) < 2:
        return None
    
    recent_l = prev_l[-15:] if len(prev_l) >= 15 else prev_l
    best_l = min(recent_l, key=lambda x: x['rsi'])
    
    if best_l['rsi'] < 45:
        return best_l
    return None

def detect_bullish_divergence_v2(df, start_idx, end_idx):
    """상승 다이버: 가격 LL + RSI HL (하락 터진 곳부터)"""
    if end_idx - start_idx < 30:
        return False
    
    window = df.iloc[start_idx:end_idx+1]
    lows, rsi = window['low'].values, window['rsi'].values
    
    # 시작점 근처 최저점 vs 끝점 근처 최저점
    first_quarter = len(window) // 4
    last_quarter = len(window) - first_quarter
    
    if first_quarter < 10:
        return False
    
    first_low_idx = np.argmin(lows[:first_quarter*2])
    last_low_idx = last_quarter + np.argmin(lows[last_quarter:])
    
    first_price, last_price = lows[first_low_idx], lows[last_low_idx]
    first_rsi = rsi[first_low_idx] if not np.isnan(rsi[first_low_idx]) else 50
    last_rsi = rsi[last_low_idx] if not np.isnan(rsi[last_low_idx]) else 50
    
    # 가격 LL + RSI HL
    if last_price < first_price * 0.99 and last_rsi > first_rsi + 3:
        return True
    return False

def detect_bearish_divergence_v2(df, start_idx, end_idx):
    """하락 다이버: 가격 HH + RSI LH (상승 터진 곳부터)"""
    if end_idx - start_idx < 30:
        return False
    
    window = df.iloc[start_idx:end_idx+1]
    highs, rsi = window['high'].values, window['rsi'].values
    
    first_quarter = len(window) // 4
    last_quarter = len(window) - first_quarter
    
    if first_quarter < 10:
        return False
    
    first_high_idx = np.argmax(highs[:first_quarter*2])
    last_high_idx = last_quarter + np.argmax(highs[last_quarter:])
    
    first_price, last_price = highs[first_high_idx], highs[last_high_idx]
    first_rsi = rsi[first_high_idx] if not np.isnan(rsi[first_high_idx]) else 50
    last_rsi = rsi[last_high_idx] if not np.isnan(rsi[last_high_idx]) else 50
    
    # 가격 HH + RSI LH
    if last_price > first_price * 1.01 and last_rsi < first_rsi - 3:
        return True
    return False

# ============================================================
# 리테스트 진입 감지
# ============================================================
def find_retest_entry(df_15m, neckline_price, breakout_time, direction, sl_price):
    """
    넥라인 돌파 후 리테스트 진입
    - 돌파 후 넥라인 근처로 복귀
    - 지지/저항 확인 후 진입
    """
    breakout_idx = df_15m[df_15m['datetime'] >= breakout_time].index
    if len(breakout_idx) == 0:
        return None
    
    start_idx = breakout_idx[0]
    search_window = df_15m.iloc[start_idx:start_idx+150]  # 약 37시간
    
    retest_found = False
    entry_time, entry_price = None, None
    
    for idx, row in search_window.iterrows():
        if direction == 'LONG':
            # 넥라인 근처로 하락 (넥라인 -1% ~ +2%)
            near_neckline = neckline_price * 0.99 <= row['low'] <= neckline_price * 1.02
            
            if near_neckline:
                # 양봉 확인 (지지)
                if row['close'] > row['open']:
                    # 다음 봉에서 상승 확인
                    next_idx = idx + 1
                    if next_idx < len(df_15m):
                        next_row = df_15m.iloc[next_idx]
                        if next_row['close'] > row['close']:
                            entry_time = next_row['datetime']
                            entry_price = next_row['close']
                            retest_found = True
                            break
        else:  # SHORT
            # 넥라인 근처로 상승 (넥라인 -2% ~ +1%)
            near_neckline = neckline_price * 0.98 <= row['high'] <= neckline_price * 1.01
            
            if near_neckline:
                # 음봉 확인 (저항)
                if row['close'] < row['open']:
                    next_idx = idx + 1
                    if next_idx < len(df_15m):
                        next_row = df_15m.iloc[next_idx]
                        if next_row['close'] < row['close']:
                            entry_time = next_row['datetime']
                            entry_price = next_row['close']
                            retest_found = True
                            break
    
    if retest_found:
        # Entry-SL Gap 체크
        if direction == 'LONG':
            entry_sl_gap = (entry_price - sl_price) / sl_price * 100
        else:
            entry_sl_gap = (sl_price - entry_price) / entry_price * 100
        
        if 1 <= entry_sl_gap <= 4:
            return {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'entry_sl_gap': entry_sl_gap,
                'entry_type': 'RETEST'
            }
    
    return None

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # ============================================================
    # 롱 패턴: W (L1-H-L2)
    # ============================================================
    if (points_1h[i]['type'] == 'L' and 
        points_1h[i+1]['type'] == 'H' and 
        points_1h[i+2]['type'] == 'L'):
        
        L1, H, L2 = points_1h[i], points_1h[i+1], points_1h[i+2]
        
        # W 조건
        l_ratio = abs(L2['price'] - L1['price']) / L1['price'] * 100
        gap = (H['price'] - min(L1['price'], L2['price'])) / min(L1['price'], L2['price']) * 100
        
        if l_ratio <= 4 and gap >= 4:  # Gap >= 5%
            # 하락 시작점 찾기
            trend_start = find_trend_start_for_long(points_1h, i)
            
            # 다이버전스
            div_1h = False
            if trend_start:
                div_1h = detect_bullish_divergence_v2(df_1h, trend_start['idx'], L2['idx'])
            
            # 15M 다이버전스
            div_15m = False
            if trend_start:
                start_15m = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
                end_15m = L2['idx'] * 4 if L2['idx'] * 4 < len(df_15m) else len(df_15m) - 1
                div_15m = detect_bullish_divergence_v2(df_15m, start_15m, end_15m)
            
            mtf_div = div_1h and div_15m
            
            # 넥라인 돌파 확인
            l2_idx = L2['idx']
            neckline = H['price']
            sl_price = min(L1['price'], L2['price'])
            
            breakout_found = False
            breakout_time = None
            
            for j in range(l2_idx + 1, min(l2_idx + 50, len(df_1h))):
                if df_1h.iloc[j]['close'] > neckline:
                    breakout_found = True
                    breakout_time = df_1h.iloc[j]['datetime']
                    break
            
            if not breakout_found:
                continue
            
            # 리테스트 진입
            retest = find_retest_entry(df_15m, neckline, breakout_time, 'LONG', sl_price)
            
            if retest:
                entry_time = retest['entry_time']
                entry_price = retest['entry_price']
                entry_sl_gap = retest['entry_sl_gap']
                entry_type = 'RETEST'
            else:
                # 리테스트 없으면 돌파 직후 진입
                entry_time = breakout_time
                entry_price = neckline * 1.005
                entry_sl_gap = (entry_price - sl_price) / sl_price * 100
                entry_type = 'BREAKOUT'
                
                if not (1 <= entry_sl_gap <= 4):
                    continue
            
            # 백테스트
            entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
            if len(entry_15m_idx) == 0:
                continue
            entry_15m_idx = entry_15m_idx[0]
            
            tp_price = entry_price * 1.05
            post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
            
            pnl, exit_type, mfe = 0, 'TIMEOUT', 0
            
            for _, row in post.iterrows():
                mfe = max(mfe, (row['high'] - entry_price) / entry_price * 100)
                if row['high'] >= tp_price:
                    pnl, exit_type = 5.0, 'TP'
                    break
                if row['low'] <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price * 100
                    exit_type = 'SL'
                    break
            
            results.append({
                'time': entry_time, 'direction': 'LONG', 'pattern': 'W',
                'entry': entry_price, 'sl': sl_price, 'entry_sl_gap': entry_sl_gap,
                'gap': gap, 'div_1h': div_1h, 'div_15m': div_15m, 'mtf_div': mtf_div,
                'entry_type': entry_type, 'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
            })
    
    # ============================================================
    # 숏 패턴: M (H1-L-H2)
    # ============================================================
    if (points_1h[i]['type'] == 'H' and 
        points_1h[i+1]['type'] == 'L' and 
        points_1h[i+2]['type'] == 'H'):
        
        H1, L, H2 = points_1h[i], points_1h[i+1], points_1h[i+2]
        
        h_ratio = abs(H2['price'] - H1['price']) / H1['price'] * 100
        gap = (max(H1['price'], H2['price']) - L['price']) / L['price'] * 100
        
        if h_ratio <= 4 and gap >= 4:
            # 상승 시작점 찾기
            trend_start = find_trend_start_for_short(points_1h, i)
            
            # 다이버전스
            div_1h = False
            if trend_start:
                div_1h = detect_bearish_divergence_v2(df_1h, trend_start['idx'], H2['idx'])
            
            div_15m = False
            if trend_start:
                start_15m = trend_start['idx'] * 4 if trend_start['idx'] * 4 < len(df_15m) else 0
                end_15m = H2['idx'] * 4 if H2['idx'] * 4 < len(df_15m) else len(df_15m) - 1
                div_15m = detect_bearish_divergence_v2(df_15m, start_15m, end_15m)
            
            mtf_div = div_1h and div_15m
            
            # 넥라인 돌파
            h2_idx = H2['idx']
            neckline = L['price']
            sl_price = max(H1['price'], H2['price'])
            
            breakout_found = False
            breakout_time = None
            
            for j in range(h2_idx + 1, min(h2_idx + 50, len(df_1h))):
                if df_1h.iloc[j]['close'] < neckline:
                    breakout_found = True
                    breakout_time = df_1h.iloc[j]['datetime']
                    break
            
            if not breakout_found:
                continue
            
            # 리테스트 진입
            retest = find_retest_entry(df_15m, neckline, breakout_time, 'SHORT', sl_price)
            
            if retest:
                entry_time = retest['entry_time']
                entry_price = retest['entry_price']
                entry_sl_gap = retest['entry_sl_gap']
                entry_type = 'RETEST'
            else:
                entry_time = breakout_time
                entry_price = neckline * 0.995
                entry_sl_gap = (sl_price - entry_price) / entry_price * 100
                entry_type = 'BREAKOUT'
                
                if not (1 <= entry_sl_gap <= 4):
                    continue
            
            # 백테스트
            entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
            if len(entry_15m_idx) == 0:
                continue
            entry_15m_idx = entry_15m_idx[0]
            
            tp_price = entry_price * 0.95
            post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
            
            pnl, exit_type, mfe = 0, 'TIMEOUT', 0
            
            for _, row in post.iterrows():
                mfe = max(mfe, (entry_price - row['low']) / entry_price * 100)
                if row['low'] <= tp_price:
                    pnl, exit_type = 5.0, 'TP'
                    break
                if row['high'] >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price * 100
                    exit_type = 'SL'
                    break
            
            results.append({
                'time': entry_time, 'direction': 'SHORT', 'pattern': 'M',
                'entry': entry_price, 'sl': sl_price, 'entry_sl_gap': entry_sl_gap,
                'gap': gap, 'div_1h': div_1h, 'div_15m': div_15m, 'mtf_div': mtf_div,
                'entry_type': entry_type, 'pnl': pnl, 'mfe': mfe, 'exit_type': exit_type
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
print("진입 타입별 성과")
print("=" * 80)

print(f"\n{'진입타입':>15} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 60)

for entry_type in ['RETEST', 'BREAKOUT']:
    subset = df_results[df_results['entry_type'] == entry_type]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{entry_type:>15} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

print("\n" + "=" * 80)
print("MTF 다이버전스별 성과")
print("=" * 80)

print(f"\n{'조건':>30} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 75)

conditions = [
    ('전체', df_results),
    ('1H 다이버 있음', df_results[df_results['div_1h'] == True]),
    ('MTF 다이버 일치', df_results[df_results['mtf_div'] == True]),
    ('LONG + MTF 다이버', df_results[(df_results['direction'] == 'LONG') & (df_results['mtf_div'] == True)]),
    ('SHORT + MTF 다이버', df_results[(df_results['direction'] == 'SHORT') & (df_results['mtf_div'] == True)]),
    ('리테스트 + MTF 다이버', df_results[(df_results['entry_type'] == 'RETEST') & (df_results['mtf_div'] == True)]),
]

for name, subset in conditions:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{name:>30} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 최적 조건
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>45} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 85)

optimal = [
    ('전체', df_results),
    ('Gap>=6%', df_results[df_results['gap'] >= 6]),
    ('Gap>=6% + MTF다이버', df_results[(df_results['gap'] >= 6) & (df_results['mtf_div'] == True)]),
    ('Gap>=6% + 리테스트', df_results[(df_results['gap'] >= 6) & (df_results['entry_type'] == 'RETEST')]),
    ('Gap>=6% + 리테스트 + MTF다이버', df_results[(df_results['gap'] >= 6) & (df_results['entry_type'] == 'RETEST') & (df_results['mtf_div'] == True)]),
    ('LONG + Gap>=6% + MTF다이버', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 6) & (df_results['mtf_div'] == True)]),
    ('SHORT + Gap>=6% + MTF다이버', df_results[(df_results['direction'] == 'SHORT') & (df_results['gap'] >= 6) & (df_results['mtf_div'] == True)]),
]

for name, subset in optimal:
    if len(subset) >= 3:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        print(f"{name:>45} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f}")

# ============================================================
# 월별 성과
# ============================================================
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month')['pnl'].agg(['sum', 'count']).round(2)
monthly.columns = ['월PnL', '거래수']

print(f"\n전체:")
print(f"  총 거래: {len(df_results)}건")
print(f"  승률: {(df_results['pnl'] > 0).mean() * 100:.1f}%")
print(f"  평균 PnL: {df_results['pnl'].mean():.2f}%")
print(f"  월 평균: {monthly['월PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['월PnL'].mean()*3:.1f}%/월")

# MTF 다이버
mtf = df_results[df_results['mtf_div'] == True]
if len(mtf) > 0:
    mtf = mtf.copy()
    mtf['month'] = pd.to_datetime(mtf['time']).dt.to_period('M')
    m_mtf = mtf.groupby('month')['pnl'].sum()
    
    print(f"\nMTF 다이버전스:")
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

total_win = (df_results['pnl'] > 0).mean() * 100
total_avg = df_results['pnl'].mean()
total_monthly = monthly['월PnL'].mean()

mtf_win = (mtf['pnl'] > 0).mean() * 100 if len(mtf) > 0 else 0
mtf_avg = mtf['pnl'].mean() if len(mtf) > 0 else 0
mtf_monthly = m_mtf.mean() if len(mtf) > 0 else 0

long_count = len(df_results[df_results['direction'] == 'LONG'])
short_count = len(df_results[df_results['direction'] == 'SHORT'])

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       개선된 MTF 롱/숏 전략 결과                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 전체 성과:                                                                    │
│    - 총 거래: {len(df_results)}건 (롱 {long_count} + 숏 {short_count})                                          │
│    - 승률: {total_win:.1f}%                                                              │
│    - 평균 PnL: {total_avg:.2f}%                                                        │
│    - 월 평균: {total_monthly:.2f}%                                                      │
│    - 3x 레버리지: {total_monthly*3:.1f}%/월                                            │
│                                                                                  │
│  ■ MTF 다이버전스 적용 시:                                                       │
│    - 총 거래: {len(mtf)}건                                                          │
│    - 승률: {mtf_win:.1f}%                                                              │
│    - 평균 PnL: {mtf_avg:.2f}%                                                        │
│    - 월 평균: {mtf_monthly:.2f}%                                                      │
│    - 3x 레버리지: {mtf_monthly*3:.1f}%/월                                            │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 핵심:                                                                         │
│    "하락/상승 터진 곳부터 다이버 봐야 한다"                                       │
│    "돌파 → 조절(리테스트) → 1차 → 2차"                                           │
│    "MTF 일치해야 진짜"                                                           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

# 저장
df_results.to_csv('mtf_refined_results.csv', index=False)
print("결과 저장: mtf_refined_results.csv")
