#!/usr/bin/env python3
"""
MTF 리테스트 진입 전략

사용자 핵심 원칙:
"Breakout → Adjustment (Retest) → 1차 상승 → 2차 상승"
"돌파한 85669자리가 자리잡기 좋지 않았어?"

돌파 후 리테스트 진입:
1. 추세선/넥라인 돌파
2. 돌파 후 되돌림 (retest)
3. 지지 확인 (양봉 + 거래량)
4. 진입
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("MTF 리테스트 진입 전략")
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
    return 100 - (100 / (1 + rs))

# 1H MACD & RSI
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']
df_1h['rsi'] = calc_rsi(df_1h['close'])

# 15M MACD & RSI
df_15m['rsi'] = calc_rsi(df_15m['close'])
exp1_15m = df_15m['close'].ewm(span=12, adjust=False).mean()
exp2_15m = df_15m['close'].ewm(span=26, adjust=False).mean()
df_15m['macd'] = exp1_15m - exp2_15m
df_15m['signal_line'] = df_15m['macd'].ewm(span=9, adjust=False).mean()
df_15m['hist'] = df_15m['macd'] - df_15m['signal_line']
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# H/L 추출
def extract_hl_points(df):
    hist = df['hist'].values
    high = df['high'].values
    low = df['low'].values
    timestamps = df['datetime'].values
    rsi = df['rsi'].values if 'rsi' in df.columns else np.zeros(len(df))
    
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({
                'type': 'H', 'price': high[max_idx], 
                'time': pd.Timestamp(timestamps[max_idx]), 
                'idx': max_idx, 'rsi': rsi[max_idx] if max_idx < len(rsi) else 50
            })
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({
                'type': 'L', 'price': low[min_idx], 
                'time': pd.Timestamp(timestamps[min_idx]), 
                'idx': min_idx, 'rsi': rsi[min_idx] if min_idx < len(rsi) else 50
            })
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
print(f"1H H/L points: {len(points_1h)}")

# 15M 스윙 포인트
def get_15m_swing_points(df_15m, window=5):
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    times = df_15m['datetime'].values
    rsi = df_15m['rsi'].values
    
    swing_highs, swing_lows = [], []
    
    for i in range(window, len(df_15m) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append({'price': highs[i], 'time': pd.Timestamp(times[i]), 'idx': i, 'rsi': rsi[i]})
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append({'price': lows[i], 'time': pd.Timestamp(times[i]), 'idx': i, 'rsi': rsi[i]})
    
    return swing_highs, swing_lows

swing_highs_15m, swing_lows_15m = get_15m_swing_points(df_15m)
print(f"15M Swing Highs: {len(swing_highs_15m)}, Swing Lows: {len(swing_lows_15m)}")

# ============================================================
# 추세선 감지 함수
# ============================================================
def detect_trendline(swing_points, direction='down', min_points=2):
    """
    추세선 감지
    direction: 'down' = 고점들을 이은 하락 추세선
               'up' = 저점들을 이은 상승 추세선
    """
    if len(swing_points) < min_points:
        return None
    
    # 최근 N개 포인트에서 추세선 찾기
    recent_points = swing_points[-10:]  # 최근 10개
    
    best_trendline = None
    best_score = 0
    
    for i in range(len(recent_points) - 1):
        for j in range(i + 1, len(recent_points)):
            p1, p2 = recent_points[i], recent_points[j]
            
            if p2['time'] <= p1['time']:
                continue
            
            # 기울기 계산
            time_diff = (p2['idx'] - p1['idx'])
            if time_diff <= 0:
                continue
            
            slope = (p2['price'] - p1['price']) / time_diff
            
            # 방향 확인
            if direction == 'down' and slope >= 0:
                continue
            if direction == 'up' and slope <= 0:
                continue
            
            # 추세선 품질 점수 (터치 횟수)
            touches = 0
            for k, p in enumerate(recent_points):
                if k == i or k == j:
                    continue
                expected_price = p1['price'] + slope * (p['idx'] - p1['idx'])
                tolerance = abs(expected_price * 0.005)  # 0.5% 허용
                if abs(p['price'] - expected_price) <= tolerance:
                    touches += 1
            
            score = touches + 2  # 기본 2점 (두 포인트)
            
            if score > best_score:
                best_score = score
                best_trendline = {
                    'p1': p1, 'p2': p2, 'slope': slope,
                    'touches': score, 'direction': direction
                }
    
    return best_trendline

# ============================================================
# 리테스트 감지 함수
# ============================================================
def detect_retest_entry(df_15m, breakout_time, breakout_price, breakout_type, sl_price, direction='LONG'):
    """
    돌파 후 리테스트 진입 감지
    
    breakout_type: 'trendline' or 'neckline'
    direction: 'LONG' or 'SHORT'
    
    리테스트 조건:
    1. 돌파 후 가격이 돌파 레벨 근처로 되돌아옴 (1~2% 이내)
    2. 지지/저항 확인 (양봉/음봉 + 거래량)
    3. Entry Gap 1~3%
    """
    # 돌파 후 48시간 내 리테스트 탐색
    search_start = breakout_time
    search_end = breakout_time + pd.Timedelta(hours=48)
    
    mask = (df_15m['datetime'] > search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 10:
        return None
    
    retest_entries = []
    
    if direction == 'LONG':
        # LONG: 돌파 후 가격이 돌파 레벨 근처로 하락 후 반등
        for idx, row in df_window.iterrows():
            # 돌파 레벨 근처로 되돌아왔는지 (1~3% 이내)
            pullback_pct = (row['low'] - breakout_price) / breakout_price * 100
            
            if -3 <= pullback_pct <= 1:  # 돌파 레벨 근처
                # 양봉 확인
                is_bullish = row['close'] > row['open']
                body_size = abs(row['close'] - row['open']) / row['open'] * 100
                high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.2
                
                # 꼬리 비율 (아래 꼬리가 길면 지지 확인)
                total_range = row['high'] - row['low']
                if total_range > 0:
                    lower_wick = min(row['open'], row['close']) - row['low']
                    lower_wick_ratio = lower_wick / total_range
                else:
                    lower_wick_ratio = 0
                
                # Entry Gap 계산
                entry_price = row['close']
                entry_gap = (entry_price - sl_price) / sl_price * 100
                
                # 리테스트 조건
                if is_bullish and body_size > 0.2 and 0.5 <= entry_gap <= 4:
                    score = 0
                    if high_volume: score += 2
                    if lower_wick_ratio > 0.3: score += 1
                    if body_size > 0.5: score += 1
                    
                    retest_entries.append({
                        'time': row['datetime'],
                        'price': entry_price,
                        'entry_gap': entry_gap,
                        'score': score,
                        'pullback_pct': pullback_pct,
                        'has_volume': high_volume,
                        'lower_wick_ratio': lower_wick_ratio
                    })
    
    else:  # SHORT
        # SHORT: 돌파 후 가격이 돌파 레벨 근처로 상승 후 하락
        for idx, row in df_window.iterrows():
            # 돌파 레벨 근처로 되돌아왔는지
            pullback_pct = (breakout_price - row['high']) / breakout_price * 100
            
            if -3 <= pullback_pct <= 1:  # 돌파 레벨 근처
                # 음봉 확인
                is_bearish = row['close'] < row['open']
                body_size = abs(row['close'] - row['open']) / row['open'] * 100
                high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.2
                
                # 꼬리 비율 (위 꼬리가 길면 저항 확인)
                total_range = row['high'] - row['low']
                if total_range > 0:
                    upper_wick = row['high'] - max(row['open'], row['close'])
                    upper_wick_ratio = upper_wick / total_range
                else:
                    upper_wick_ratio = 0
                
                # Entry Gap 계산
                entry_price = row['close']
                entry_gap = (sl_price - entry_price) / entry_price * 100
                
                # 리테스트 조건
                if is_bearish and body_size > 0.2 and 0.5 <= entry_gap <= 4:
                    score = 0
                    if high_volume: score += 2
                    if upper_wick_ratio > 0.3: score += 1
                    if body_size > 0.5: score += 1
                    
                    retest_entries.append({
                        'time': row['datetime'],
                        'price': entry_price,
                        'entry_gap': entry_gap,
                        'score': score,
                        'pullback_pct': pullback_pct,
                        'has_volume': high_volume,
                        'upper_wick_ratio': upper_wick_ratio
                    })
    
    if len(retest_entries) == 0:
        return None
    
    # 최적 Entry Gap (1~3%) 우선, 점수 높은 순
    optimal_entries = [e for e in retest_entries if 1 <= e['entry_gap'] <= 3]
    if len(optimal_entries) > 0:
        optimal_entries.sort(key=lambda x: (-x['score'], x['time']))
        return optimal_entries[0]
    
    # 없으면 점수 높은 순
    retest_entries.sort(key=lambda x: (-x['score'], x['time']))
    return retest_entries[0]

# ============================================================
# 백테스트 실행
# ============================================================
print("\n" + "=" * 80)
print("리테스트 백테스트 실행")
print("=" * 80)

results = []
tp_pct = 5

# LONG - W 패턴 + 리테스트
print("\nLONG 리테스트 스캔...")
long_count = 0

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴 조건
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 3:  # 최소 3% Gap
        continue
    
    sl_price = min(L1['price'], L2['price'])
    
    # L2 이후 15M 데이터에서 돌파 감지
    l2_15m_idx = df_15m[df_15m['datetime'] >= L2['time']].index
    if len(l2_15m_idx) == 0:
        continue
    l2_15m_idx = l2_15m_idx[0]
    
    # 15M 스윙 하이에서 하락 추세선 찾기
    window_highs = [h for h in swing_highs_15m if L2['time'] - pd.Timedelta(hours=24) <= h['time'] <= L2['time'] + pd.Timedelta(hours=24)]
    
    if len(window_highs) < 2:
        continue
    
    trendline = detect_trendline(window_highs, 'down')
    
    if trendline is None:
        continue
    
    # 돌파 감지
    search_df = df_15m.iloc[l2_15m_idx:l2_15m_idx+200]
    breakout_detected = False
    breakout_time = None
    breakout_price = None
    
    for idx, row in search_df.iterrows():
        bars_from_p2 = idx - trendline['p2']['idx']
        trendline_price = trendline['p2']['price'] + trendline['slope'] * bars_from_p2
        
        if row['close'] > trendline_price * 1.002:
            breakout_detected = True
            breakout_time = row['datetime']
            breakout_price = trendline_price
            break
    
    if not breakout_detected:
        continue
    
    # 리테스트 진입 감지
    retest = detect_retest_entry(df_15m, breakout_time, breakout_price, 'trendline', sl_price, 'LONG')
    
    if retest is None:
        continue
    
    entry_time = retest['time']
    entry_price = retest['price']
    entry_gap = retest['entry_gap']
    
    # TP
    tp_price = entry_price * (1 + tp_pct / 100)
    
    # 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    
    for j, (_, row) in enumerate(post.iterrows()):
        mfe = max(mfe, (row['high'] - entry_price) / entry_price * 100)
        
        if row['high'] >= tp_price:
            pnl = tp_pct
            exit_type = 'TP'
            break
        if row['low'] <= sl_price:
            pnl = (sl_price - entry_price) / entry_price * 100
            exit_type = 'SL'
            break
    
    long_count += 1
    results.append({
        'time': entry_time,
        'direction': 'LONG',
        'entry_type': 'RETEST',
        'entry': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'entry_gap': entry_gap,
        'gap': gap,
        'has_retest_volume': retest['has_volume'],
        'retest_score': retest['score'],
        'pullback_pct': retest['pullback_pct'],
        'pnl': pnl,
        'mfe': mfe,
        'exit_type': exit_type
    })

print(f"LONG 리테스트 신호: {long_count}건")

# SHORT - M 패턴 + 리테스트
print("\nSHORT 리테스트 스캔...")
short_count = 0

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'H' and 
            points_1h[i+1]['type'] == 'L' and 
            points_1h[i+2]['type'] == 'H'):
        continue
    
    H1 = points_1h[i]
    L = points_1h[i+1]
    H2 = points_1h[i+2]
    
    # M 패턴 조건
    h_ratio = (H2['price'] - H1['price']) / H1['price'] * 100
    if h_ratio < -3:
        continue
    
    gap = (H2['price'] - L['price']) / L['price'] * 100
    if gap < 3:
        continue
    
    sl_price = max(H1['price'], H2['price'])
    
    # H2 이후 15M 데이터에서 돌파 감지
    h2_15m_idx = df_15m[df_15m['datetime'] >= H2['time']].index
    if len(h2_15m_idx) == 0:
        continue
    h2_15m_idx = h2_15m_idx[0]
    
    # 15M 스윙 로우에서 상승 추세선 찾기
    window_lows = [l for l in swing_lows_15m if H2['time'] - pd.Timedelta(hours=24) <= l['time'] <= H2['time'] + pd.Timedelta(hours=24)]
    
    if len(window_lows) < 2:
        continue
    
    trendline = detect_trendline(window_lows, 'up')
    
    if trendline is None:
        continue
    
    # 돌파 감지
    search_df = df_15m.iloc[h2_15m_idx:h2_15m_idx+200]
    breakout_detected = False
    breakout_time = None
    breakout_price = None
    
    for idx, row in search_df.iterrows():
        bars_from_p2 = idx - trendline['p2']['idx']
        trendline_price = trendline['p2']['price'] + trendline['slope'] * bars_from_p2
        
        if row['close'] < trendline_price * 0.998:
            breakout_detected = True
            breakout_time = row['datetime']
            breakout_price = trendline_price
            break
    
    if not breakout_detected:
        continue
    
    # 리테스트 진입 감지
    retest = detect_retest_entry(df_15m, breakout_time, breakout_price, 'trendline', sl_price, 'SHORT')
    
    if retest is None:
        continue
    
    entry_time = retest['time']
    entry_price = retest['price']
    entry_gap = retest['entry_gap']
    
    # TP
    tp_price = entry_price * (1 - tp_pct / 100)
    
    # 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    
    for j, (_, row) in enumerate(post.iterrows()):
        mfe = max(mfe, (entry_price - row['low']) / entry_price * 100)
        
        if row['low'] <= tp_price:
            pnl = tp_pct
            exit_type = 'TP'
            break
        if row['high'] >= sl_price:
            pnl = (entry_price - sl_price) / entry_price * 100
            exit_type = 'SL'
            break
    
    short_count += 1
    results.append({
        'time': entry_time,
        'direction': 'SHORT',
        'entry_type': 'RETEST',
        'entry': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'entry_gap': entry_gap,
        'gap': gap,
        'has_retest_volume': retest['has_volume'],
        'retest_score': retest['score'],
        'pullback_pct': retest['pullback_pct'],
        'pnl': pnl,
        'mfe': mfe,
        'exit_type': exit_type
    })

print(f"SHORT 리테스트 신호: {short_count}건")

# ============================================================
# 결과 분석
# ============================================================
df_results = pd.DataFrame(results)
print(f"\n총 리테스트 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호가 없습니다. 조건 완화 필요.")
    exit()

# 전체 성과
print("\n" + "=" * 80)
print("리테스트 전략 성과")
print("=" * 80)

win_rate = (df_results['pnl'] > 0).mean() * 100
avg_pnl = df_results['pnl'].mean()
total_pnl = df_results['pnl'].sum()

print(f"  총 거래: {len(df_results)}건")
print(f"  승률: {win_rate:.1f}%")
print(f"  평균 PnL: {avg_pnl:.2f}%")
print(f"  총 PnL: {total_pnl:.1f}%")

# 방향별 성과
print("\n방향별 성과:")
for direction in ['LONG', 'SHORT']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"  [{direction}] {len(subset)}건, 승률: {wr:.1f}%, 평균 PnL: {ap:.2f}%, 총 PnL: {tp:.1f}%")

# Entry Gap 범위별
print("\nEntry Gap 범위별 성과:")
print(f"{'Entry Gap':>15} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 50)

for low, high in [(0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4)]:
    subset = df_results[(df_results['entry_gap'] >= low) & (df_results['entry_gap'] < high)]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {wr:>10.1f} {ap:>10.2f}")

# 리테스트 점수별
print("\n리테스트 점수별 성과:")
for score in sorted(df_results['retest_score'].unique()):
    subset = df_results[df_results['retest_score'] == score]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        print(f"  점수 {score}: {len(subset)}건, 승률: {wr:.1f}%, 평균 PnL: {ap:.2f}%")

# 볼륨 유무별
print("\n리테스트 볼륨 유무별 성과:")
for has_vol in [True, False]:
    subset = df_results[df_results['has_retest_volume'] == has_vol]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        status = "있음" if has_vol else "없음"
        print(f"  볼륨 {status}: {len(subset)}건, 승률: {wr:.1f}%, 평균 PnL: {ap:.2f}%")

# 월별 성과
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month').agg({
    'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
}).round(2)
monthly.columns = ['총PnL', '거래수', '승률']

print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['총PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['총PnL'] > 0).sum()}개월 / {len(monthly)}개월")

# 최적 조건
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

conditions = [
    ('전체', df_results),
    ('LONG만', df_results[df_results['direction'] == 'LONG']),
    ('SHORT만', df_results[df_results['direction'] == 'SHORT']),
    ('Entry Gap 1-3%', df_results[(df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('Gap>=4% + Entry Gap 1-3%', df_results[(df_results['gap'] >= 4) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('볼륨 있음', df_results[df_results['has_retest_volume'] == True]),
    ('점수>=2', df_results[df_results['retest_score'] >= 2]),
    ('LONG + Entry Gap 1-3%', df_results[(df_results['direction'] == 'LONG') & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
]

print(f"\n{'조건':>35} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 80)

for name, subset in conditions:
    if len(subset) >= 5:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"{name:>35} {len(subset):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f}")

# 최종 결론
print("\n" + "=" * 80)
print("최종 결론: 리테스트 진입 전략")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                       리테스트 진입 전략 결론                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 핵심 원칙:                                                               │
│    "Breakout → Adjustment (Retest) → 1차 상승 → 2차 상승"                  │
│                                                                            │
│  ■ 리테스트 진입 조건:                                                       │
│    1. 추세선 돌파 감지                                                       │
│    2. 돌파 후 1~3% 되돌림 (Retest)                                          │
│    3. 지지/저항 확인 (양봉/음봉 + 거래량 + 꼬리)                              │
│    4. Entry Gap 1~3%                                                        │
│                                                                            │
│  ■ 결과:                                                                    │
│    - 총 거래: {len(df_results)}건                                                          │
│    - 승률: {win_rate:.1f}%                                                          │
│    - 평균 PnL: {avg_pnl:.2f}%                                                    │
│    - 월 평균: {monthly['총PnL'].mean():.2f}%                                                     │
│    - 3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_retest_results.csv', index=False)
print("\n결과 저장: mtf_retest_results.csv")
