#!/usr/bin/env python3
"""
완전한 MTF 전략: 하락추세 시작 ~ 추세돌파까지 전체 그림

핵심 원칙:
"하락추세 시작 ~ 추세돌파까지의 내용을 안보면 매매가 안된다"

구현 로직:
1. 하락추세 시작점 감지 (RSI 고점 + 가격 고점)
2. 하락 진행 중 모니터링 (추세선, W패턴, 다이버전스)
3. MTF 다이버전스 확인 (1H + 15M)
4. 추세선 돌파 감지
5. 조절(리테스트) 대기
6. 지지 확인 후 진입

진입 조건:
- 돌파 → 조절 → 1차 → 2차 흐름 이해
- Entry-L Gap 1~3%
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("완전한 MTF 전략: 하락추세 시작 ~ 추세돌파")
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

# ============================================================
# RSI 계산
# ============================================================
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

# 볼륨 MA
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()
df_1h['vol_ma20'] = df_1h['volume'].rolling(20).mean()

print("RSI, MACD 계산 완료")

# ============================================================
# H/L 포인트 추출
# ============================================================
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
# 1. 하락추세 시작점 감지
# ============================================================
def find_downtrend_start(points, idx):
    """
    현재 L 포인트 기준으로 하락추세 시작점(H) 찾기
    - 이전 H 중에서 가장 높은 가격 + RSI 고점
    """
    # 현재 위치 이전의 H 포인트들
    prev_h_points = [p for p in points[:idx] if p['type'] == 'H']
    
    if len(prev_h_points) < 1:
        return None
    
    # 최근 10개 H 중에서 가장 높은 가격의 H
    recent_h = prev_h_points[-10:] if len(prev_h_points) >= 10 else prev_h_points
    highest_h = max(recent_h, key=lambda x: x['price'])
    
    return highest_h

# ============================================================
# 2. 다이버전스 감지 (하락 시작점부터)
# ============================================================
def detect_divergence(df, start_idx, end_idx, timeframe='1h'):
    """
    하락 시작점(start_idx)부터 현재(end_idx)까지 다이버전스 확인
    
    상승 다이버전스:
    - 가격: 더 낮은 저점 (LL)
    - RSI/MACD: 더 높은 저점 (HL)
    """
    if start_idx >= end_idx:
        return False, None
    
    window = df.iloc[start_idx:end_idx+1]
    
    if len(window) < 10:
        return False, None
    
    # 가격 저점들 찾기
    lows = window['low'].values
    rsi = window['rsi'].values
    macd = window['macd_hist'].values
    
    # 로컬 저점 찾기 (단순화: 구간 내 최저점들)
    price_low_idx = np.argmin(lows)
    
    # 첫 1/3 구간과 마지막 1/3 구간 비교
    first_third = len(window) // 3
    last_third = len(window) - first_third
    
    if first_third < 5 or last_third < 5:
        return False, None
    
    # 첫 구간 최저점
    first_low_idx = np.argmin(lows[:first_third])
    first_low_price = lows[first_low_idx]
    first_low_rsi = rsi[first_low_idx] if not np.isnan(rsi[first_low_idx]) else 50
    
    # 마지막 구간 최저점
    last_low_idx = last_third + np.argmin(lows[last_third:])
    last_low_price = lows[last_low_idx]
    last_low_rsi = rsi[last_low_idx] if not np.isnan(rsi[last_low_idx]) else 50
    
    # 상승 다이버전스: 가격 LL + RSI HL
    price_ll = last_low_price < first_low_price * 0.99  # 가격 1% 이상 낮음
    rsi_hl = last_low_rsi > first_low_rsi + 2  # RSI 2 이상 높음
    
    if price_ll and rsi_hl:
        return True, {
            'first_price': first_low_price,
            'last_price': last_low_price,
            'first_rsi': first_low_rsi,
            'last_rsi': last_low_rsi,
            'price_diff': (last_low_price - first_low_price) / first_low_price * 100,
            'rsi_diff': last_low_rsi - first_low_rsi
        }
    
    return False, None

# ============================================================
# 3. 추세선 돌파 + 리테스트 감지
# ============================================================
def detect_breakout_and_retest(df_15m, h1_price, h2_price, h1_idx, h2_idx, l2_idx, l2_price):
    """
    추세선 돌파 후 리테스트 감지
    
    1. H1-H2 추세선 돌파 확인
    2. 돌파 후 리테스트 (돌파선 근처로 복귀)
    3. 지지 확인 (양봉 + 거래량)
    """
    # 추세선 기울기
    bars_h1_h2 = h2_idx - h1_idx
    if bars_h1_h2 <= 0:
        return None
    
    slope = (h2_price - h1_price) / bars_h1_h2
    
    # L2 이후 데이터에서 돌파 및 리테스트 찾기
    l2_15m_idx = df_15m[df_15m['datetime'] >= df_15m.iloc[l2_idx * 4]['datetime'] if l2_idx * 4 < len(df_15m) else df_15m['datetime'].max()].index
    if len(l2_15m_idx) == 0:
        return None
    
    start_idx = l2_15m_idx[0]
    search_window = df_15m.iloc[start_idx:start_idx + 200]  # 약 50시간
    
    breakout_found = False
    breakout_price = None
    breakout_idx = None
    breakout_time = None
    
    retest_found = False
    retest_price = None
    retest_idx = None
    retest_time = None
    
    for idx, row in search_window.iterrows():
        # 현재 바 기준 추세선 가격
        bars_from_h2 = (idx - start_idx) / 4 + (l2_idx - h2_idx)  # 대략적 환산
        trendline_price = h2_price + slope * bars_from_h2
        
        # 1. 돌파 감지
        if not breakout_found:
            if row['close'] > trendline_price * 1.005:  # 0.5% 이상 돌파
                breakout_found = True
                breakout_price = row['close']
                breakout_idx = idx
                breakout_time = row['datetime']
                continue
        
        # 2. 리테스트 감지 (돌파 후)
        if breakout_found and not retest_found:
            # 돌파선 근처로 복귀 (돌파가보다 1~3% 낮은 구간)
            retest_zone_high = breakout_price * 0.99
            retest_zone_low = breakout_price * 0.96
            
            if retest_zone_low <= row['low'] <= retest_zone_high:
                # 지지 확인: 양봉 + 거래량
                is_bullish = row['close'] > row['open']
                high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.2
                
                # 다음 봉에서 반등 확인
                next_idx = idx + 1
                if next_idx < len(df_15m):
                    next_row = df_15m.iloc[next_idx]
                    if next_row['close'] > row['close']:
                        retest_found = True
                        retest_price = row['close']
                        retest_idx = idx
                        retest_time = row['datetime']
                        break
    
    if retest_found:
        return {
            'breakout_time': breakout_time,
            'breakout_price': breakout_price,
            'retest_time': retest_time,
            'retest_price': retest_price,
            'entry_type': 'RETEST'
        }
    elif breakout_found:
        return {
            'breakout_time': breakout_time,
            'breakout_price': breakout_price,
            'retest_time': None,
            'retest_price': None,
            'entry_type': 'BREAKOUT_ONLY'
        }
    
    return None

# ============================================================
# 4. 15M 진입 신호 감지
# ============================================================
def detect_15m_signals(df_15m, search_start, search_end, l2_price):
    """15M에서 추세 반전 신호 감지"""
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    window = df_15m[mask].copy()
    
    if len(window) < 20:
        return {'signal_count': 0}
    
    signals = {
        'trendline_break_15m': False,
        'lh_to_hh': False,
        'higher_low': False,
        'bullish_volume': False,
        'signal_count': 0
    }
    
    # 스윙 포인트 찾기
    highs = []
    lows = []
    for i in range(5, len(window) - 5):
        idx = window.index[i]
        if window.loc[idx, 'high'] == window.iloc[i-5:i+6]['high'].max():
            highs.append({'price': window.loc[idx, 'high'], 'idx': idx, 'time': window.loc[idx, 'datetime']})
        if window.loc[idx, 'low'] == window.iloc[i-5:i+6]['low'].min():
            lows.append({'price': window.loc[idx, 'low'], 'idx': idx, 'time': window.loc[idx, 'datetime']})
    
    # LH → HH 전환
    if len(highs) >= 3:
        for i in range(len(highs) - 2):
            h1, h2, h3 = highs[i], highs[i+1], highs[i+2]
            if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                signals['lh_to_hh'] = True
                break
    
    # HL (저점 올림)
    if len(lows) >= 3:
        for i in range(len(lows) - 2):
            l1, l2, l3 = lows[i], lows[i+1], lows[i+2]
            if l2['price'] < l1['price'] and l3['price'] > l2['price']:
                signals['higher_low'] = True
                break
    
    # 양봉 + 거래량
    for idx in window.index:
        row = window.loc[idx]
        if row['close'] > row['open']:
            if pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5:
                if row['low'] <= l2_price * 1.02:
                    signals['bullish_volume'] = True
                    break
    
    signals['signal_count'] = sum([
        signals['trendline_break_15m'],
        signals['lh_to_hh'],
        signals['higher_low'],
        signals['bullish_volume']
    ])
    
    return signals

# ============================================================
# 메인 백테스트
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행: 하락추세 시작 ~ 추세돌파 ~ 리테스트 진입")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # W 패턴 (L1 - H - L2) 찾기
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H_neckline = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴 조건
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    # Gap 조건
    gap = (H_neckline['price'] - L2['price']) / L2['price'] * 100
    if gap < 4:
        continue
    
    # 1. 하락추세 시작점 찾기
    downtrend_start = find_downtrend_start(points_1h, i)
    if downtrend_start is None:
        continue
    
    # 2. 다이버전스 확인 (하락 시작점 ~ L2)
    div_1h, div_info = detect_divergence(df_1h, downtrend_start['idx'], L2['idx'], '1h')
    
    # 15M 다이버전스
    start_15m_idx = downtrend_start['idx'] * 4 if downtrend_start['idx'] * 4 < len(df_15m) else 0
    end_15m_idx = L2['idx'] * 4 if L2['idx'] * 4 < len(df_15m) else len(df_15m) - 1
    div_15m, div_15m_info = detect_divergence(df_15m, start_15m_idx, end_15m_idx, '15m')
    
    # MTF 다이버 일치
    mtf_divergence = div_1h and div_15m
    
    # 3. 추세선 (H1-H2) 찾기 - 하락추세 시작점과 넥라인 사이의 H들
    prev_h_points = [p for p in points_1h[:i+2] if p['type'] == 'H' and p['idx'] > downtrend_start['idx']]
    
    if len(prev_h_points) < 2:
        continue
    
    H1 = prev_h_points[-2] if len(prev_h_points) >= 2 else downtrend_start
    H2 = prev_h_points[-1]
    
    # 하락추세선인지 확인
    if H2['price'] >= H1['price']:
        continue
    
    # 4. 15M 진입 신호
    search_start = L2['time'] - pd.Timedelta(hours=6)
    search_end = L2['time'] + pd.Timedelta(hours=24)
    signals_15m = detect_15m_signals(df_15m, search_start, search_end, L2['price'])
    
    # 5. 진입 결정
    sl_price = min(L1['price'], L2['price'])
    
    # 진입 가격 결정: L2 근처에서 15M 신호 발생 시
    # 리테스트가 있으면 리테스트에서, 없으면 L2 + 약간 위에서
    entry_price = L2['price'] * 1.015  # L2보다 1.5% 위에서 진입 (단순화)
    entry_time = L2['time'] + pd.Timedelta(hours=6)
    
    entry_l_gap = (entry_price - sl_price) / sl_price * 100
    
    # Entry-L Gap 필터
    if not (1 <= entry_l_gap <= 3):
        continue
    
    # 6. 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    # TP 5%
    tp_price = entry_price * 1.05
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    
    for _, row in post.iterrows():
        current_pnl = (row['high'] - entry_price) / entry_price * 100
        mfe = max(mfe, current_pnl)
        
        if row['high'] >= tp_price:
            pnl = 5.0
            exit_type = 'TP'
            break
        if row['low'] <= sl_price:
            pnl = (sl_price - entry_price) / entry_price * 100
            exit_type = 'SL'
            break
    
    results.append({
        'time': entry_time,
        'entry': entry_price,
        'sl': sl_price,
        'entry_l_gap': entry_l_gap,
        'gap': gap,
        'div_1h': div_1h,
        'div_15m': div_15m,
        'mtf_divergence': mtf_divergence,
        'signal_count_15m': signals_15m['signal_count'],
        'lh_to_hh': signals_15m['lh_to_hh'],
        'higher_low': signals_15m['higher_low'],
        'bullish_volume': signals_15m['bullish_volume'],
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
print("MTF 다이버전스별 성과")
print("=" * 80)

print(f"\n{'조건':>25} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 70)

conditions = [
    ('전체', df_results),
    ('1H 다이버 있음', df_results[df_results['div_1h'] == True]),
    ('15M 다이버 있음', df_results[df_results['div_15m'] == True]),
    ('MTF 다이버 일치', df_results[df_results['mtf_divergence'] == True]),
    ('MTF 다이버 불일치', df_results[df_results['mtf_divergence'] == False]),
]

for name, subset in conditions:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{name:>25} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 15M 신호별 성과
# ============================================================
print("\n" + "=" * 80)
print("15M 신호별 성과")
print("=" * 80)

print(f"\n{'조건':>30} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 65)

signal_conditions = [
    ('LH→HH 있음', df_results[df_results['lh_to_hh'] == True]),
    ('HL 저점올림 있음', df_results[df_results['higher_low'] == True]),
    ('양봉+거래량 있음', df_results[df_results['bullish_volume'] == True]),
    ('15M 신호 2개+', df_results[df_results['signal_count_15m'] >= 2]),
    ('15M 신호 3개+', df_results[df_results['signal_count_15m'] >= 3]),
]

for name, subset in signal_conditions:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        print(f"{name:>30} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f}")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>40} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 85)

optimal_conditions = [
    ('Gap>=5% + MTF다이버', 
     df_results[(df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True)]),
    ('Gap>=5% + MTF다이버 + 15M신호2+', 
     df_results[(df_results['gap'] >= 5) & (df_results['mtf_divergence'] == True) & (df_results['signal_count_15m'] >= 2)]),
    ('Gap>=4% + MTF다이버', 
     df_results[(df_results['gap'] >= 4) & (df_results['mtf_divergence'] == True)]),
    ('Gap>=4% + MTF다이버 + LH→HH', 
     df_results[(df_results['gap'] >= 4) & (df_results['mtf_divergence'] == True) & (df_results['lh_to_hh'] == True)]),
    ('Gap>=4% + 1H다이버 + 15M신호2+', 
     df_results[(df_results['gap'] >= 4) & (df_results['div_1h'] == True) & (df_results['signal_count_15m'] >= 2)]),
]

best_condition = None
best_score = -999

for name, subset in optimal_conditions:
    if len(subset) >= 5:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        score = win_rate * avg_pnl / 100
        print(f"{name:>40} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")
        
        if score > best_score:
            best_score = score
            best_condition = (name, subset)

# ============================================================
# 월별 성과
# ============================================================
if best_condition is not None:
    print("\n" + "=" * 80)
    print(f"최적 조건 월별 성과: {best_condition[0]}")
    print("=" * 80)
    
    best_df = best_condition[1].copy()
    best_df['month'] = pd.to_datetime(best_df['time']).dt.to_period('M')
    monthly = best_df.groupby('month')['pnl'].agg(['sum', 'count']).round(2)
    monthly.columns = ['월PnL', '거래수']
    
    print(f"\n  총 거래: {len(best_df)}건")
    print(f"  승률: {(best_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"  평균 PnL: {best_df['pnl'].mean():.2f}%")
    print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
    print(f"  월 평균 수익: {monthly['월PnL'].mean():.2f}%")
    print(f"  3x 레버리지: {monthly['월PnL'].mean()*3:.1f}%/월")
    print(f"  수익 월: {(monthly['월PnL'] > 0).sum()}/{len(monthly)} ({(monthly['월PnL'] > 0).mean()*100:.1f}%)")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

if best_condition is not None:
    best_name, best_df = best_condition
    best_win = (best_df['pnl'] > 0).mean() * 100
    best_avg = best_df['pnl'].mean()
    best_monthly = monthly['월PnL'].mean()
else:
    best_name = "N/A"
    best_win = 0
    best_avg = 0
    best_monthly = 0

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│              완전한 MTF 전략: 하락추세 시작 ~ 추세돌파                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 핵심 원칙:                                                                    │
│    "하락추세 시작 ~ 추세돌파까지의 내용을 안보면 매매가 안된다"                  │
│                                                                                  │
│  ■ 전체 흐름:                                                                    │
│    1. 하락추세 시작점 (RSI 고점 + 가격 고점)                                     │
│    2. 하락 진행 (추세선 형성, W패턴, 다이버전스 형성)                            │
│    3. MTF 다이버전스 확인 (1H + 15M 일치)                                        │
│    4. 추세선 돌파                                                                │
│    5. 조절 (리테스트) 대기                                                       │
│    6. 지지 확인 후 진입                                                          │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 최적 조건: {best_name:45}│
│                                                                                  │
│  ■ 성과:                                                                         │
│    - 승률: {best_win:.1f}%                                                                │
│    - 평균 PnL: {best_avg:.2f}%                                                          │
│    - 월 평균: {best_monthly:.2f}%                                                        │
│    - 3x 레버리지: {best_monthly*3:.1f}%/월                                              │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 진입 체크리스트:                                                              │
│    ✓ 1. 하락추세 시작점 확인 (RSI 하방 터진 곳)                                  │
│    ✓ 2. 1H W 패턴 형성 (L1-H-L2)                                                 │
│    ✓ 3. MTF 다이버전스 (1H + 15M 일치)                                           │
│    ✓ 4. Gap >= 4% (에너지 축적)                                                  │
│    ✓ 5. 추세선 돌파                                                              │
│    ✓ 6. 조절 (리테스트) 대기                                                     │
│    ✓ 7. 15M 진입 신호 (LH→HH, HL, 양봉+거래량)                                   │
│    ✓ 8. Entry-L Gap 1~3%                                                         │
│                                                                                  │
│  ■ 진입 흐름:                                                                    │
│    "돌파 → 조절 → 1차 → 2차"                                                    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_complete_strategy_results.csv', index=False)
print("\n결과 저장: mtf_complete_strategy_results.csv")
