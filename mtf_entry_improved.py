#!/usr/bin/env python3
"""
개선된 MTF (Multi-Time Frame) 진입 전략

핵심 개념:
- 1H: W 패턴 형성 중 (큰 그림) - L1, H(넥라인), L2 형성
- 15M: 진입 타이밍 (L2 형성 구간에서 추세 반전 신호)

15M 진입 신호 (순서대로 확인):
1. 하락추세선 돌파 (15M에서 H1-H2 하락추세선 상향 돌파)
2. LH → HH 전환 (더 낮은 고점 → 더 높은 고점)
3. HL (저점 올림) - LL → HL 전환
4. 양봉 + 거래량 (평균 대비 1.5배 이상)

핵심:
- 1H L2 형성 시점 "근처"에서 15M 신호가 나와야 함
- 15M 신호 = L2에서 반등 시작의 "확인"
- 너무 이른 진입 (L2 전) 또는 너무 늦은 진입 (넥라인 돌파 후) 모두 비효율적
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("개선된 MTF 진입 전략: 1H W패턴 + 15M 진입 신호")
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

# MACD for 1H
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

# 15M 볼륨 평균 (20봉)
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# ============================================================
# 1H H/L 추출
# ============================================================
def extract_hl_points(df, price_col_high='high', price_col_low='low'):
    hist = df['hist'].values
    high = df[price_col_high].values
    low = df[price_col_low].values
    timestamps = df['datetime'].values
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': pd.Timestamp(timestamps[max_idx]), 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': pd.Timestamp(timestamps[min_idx]), 'idx': min_idx})
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
print(f"1H H/L points: {len(points_1h)}")

# ============================================================
# 15M Swing High/Low 추출 (간단한 방식)
# ============================================================
def get_15m_swing_points(df_15m, window=5):
    """15M 스윙 고/저점 추출"""
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    times = df_15m['datetime'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df_15m) - window):
        # 스윙 하이: 양쪽 window 기간 내 최고점
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append({
                'price': highs[i],
                'time': pd.Timestamp(times[i]),
                'idx': i
            })
        # 스윙 로우: 양쪽 window 기간 내 최저점
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append({
                'price': lows[i],
                'time': pd.Timestamp(times[i]),
                'idx': i
            })
    
    return swing_highs, swing_lows

swing_highs_15m, swing_lows_15m = get_15m_swing_points(df_15m)
print(f"15M Swing Highs: {len(swing_highs_15m)}, Swing Lows: {len(swing_lows_15m)}")

# ============================================================
# 15M 진입 신호 감지 (개선된 로직)
# ============================================================
def detect_15m_entry_signals_v2(df_15m, swing_highs, swing_lows, l2_time, l2_price, h_neckline_price):
    """
    L2 형성 시점 근처에서 15M 진입 신호 감지
    
    Parameters:
    - l2_time: 1H L2 형성 시점
    - l2_price: 1H L2 가격
    - h_neckline_price: 1H 넥라인(H) 가격
    
    Returns:
    - 신호 정보 딕셔너리
    """
    
    # L2 시점 기준 앞뒤 구간 설정
    # L2 형성 전 6시간 ~ L2 형성 후 12시간 구간에서 신호 탐색
    search_start = l2_time - pd.Timedelta(hours=6)
    search_end = l2_time + pd.Timedelta(hours=12)
    
    # 해당 구간의 15M 데이터
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 20:
        return None
    
    # 해당 구간의 스윙 포인트
    window_highs = [h for h in swing_highs if search_start <= h['time'] <= search_end]
    window_lows = [l for l in swing_lows if search_start <= l['time'] <= search_end]
    
    signals = {
        'trendline_break': False,
        'trendline_break_time': None,
        'lh_to_hh': False,
        'lh_to_hh_time': None,
        'higher_low': False,
        'higher_low_time': None,
        'bullish_candle_volume': False,
        'bullish_volume_time': None,
        'entry_time': None,
        'entry_price': None,
        'signal_count': 0
    }
    
    # 1. 하락추세선 돌파 체크
    # 15M에서 연속 2개의 하락하는 고점 (H1 > H2) 후, 그 추세선을 상향 돌파
    if len(window_highs) >= 2:
        for i in range(len(window_highs) - 1):
            h1 = window_highs[i]
            h2 = window_highs[i + 1]
            
            # 하락추세 (H1 > H2)
            if h2['price'] < h1['price'] * 0.998:  # 최소 0.2% 하락
                # H2 이후 데이터에서 추세선 돌파 확인
                h2_idx = h2['idx']
                after_h2 = df_15m[(df_15m.index > h2_idx) & 
                                  (df_15m['datetime'] <= search_end)]
                
                # 추세선 기울기 계산
                time_diff = (h2['idx'] - h1['idx'])
                if time_diff > 0:
                    slope = (h2['price'] - h1['price']) / time_diff
                    
                    for idx, row in after_h2.iterrows():
                        bars_from_h2 = idx - h2_idx
                        trendline_price = h2['price'] + slope * bars_from_h2
                        
                        # 종가가 추세선 상향 돌파
                        if row['close'] > trendline_price * 1.001:  # 0.1% 이상 돌파
                            signals['trendline_break'] = True
                            signals['trendline_break_time'] = row['datetime']
                            if signals['entry_time'] is None or row['datetime'] < signals['entry_time']:
                                signals['entry_time'] = row['datetime']
                                signals['entry_price'] = row['close']
                            break
                    
                if signals['trendline_break']:
                    break
    
    # 2. LH → HH 전환 체크
    # 더 낮은 고점(LH)이 나온 후, 그보다 더 높은 고점(HH)이 형성
    if len(window_highs) >= 3:
        for i in range(len(window_highs) - 2):
            h1, h2, h3 = window_highs[i], window_highs[i+1], window_highs[i+2]
            
            # H1 > H2 (LH 형성) && H3 > H2 (HH 전환)
            if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                signals['lh_to_hh'] = True
                signals['lh_to_hh_time'] = h3['time']
                if signals['entry_time'] is None or h3['time'] < signals['entry_time']:
                    signals['entry_time'] = h3['time']
                    signals['entry_price'] = h3['price']
                break
    
    # 3. HL (저점 올림) 체크
    # 더 낮은 저점(LL)이 나온 후, 그보다 더 높은 저점(HL)이 형성
    if len(window_lows) >= 3:
        for i in range(len(window_lows) - 2):
            l1, l2_15m, l3 = window_lows[i], window_lows[i+1], window_lows[i+2]
            
            # L1 > L2 (LL 형성) && L3 > L2 (HL 전환)
            if l2_15m['price'] < l1['price'] and l3['price'] > l2_15m['price']:
                signals['higher_low'] = True
                signals['higher_low_time'] = l3['time']
                break
    
    # 4. 양봉 + 거래량 체크
    # L2 가격 근처에서 평균 이상 거래량을 동반한 양봉
    for idx, row in df_window.iterrows():
        is_bullish = row['close'] > row['open']
        body_size = abs(row['close'] - row['open']) / row['open'] * 100
        near_l2 = row['low'] <= l2_price * 1.01  # L2 가격의 1% 이내에 저점
        high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5
        
        if is_bullish and body_size > 0.3 and high_volume and near_l2:
            signals['bullish_candle_volume'] = True
            signals['bullish_volume_time'] = row['datetime']
            break
    
    # 신호 개수 계산
    signal_count = sum([
        signals['trendline_break'],
        signals['lh_to_hh'],
        signals['higher_low'],
        signals['bullish_candle_volume']
    ])
    signals['signal_count'] = signal_count
    
    return signals

# ============================================================
# 1H W 패턴 + 15M 진입 신호 백테스트 (개선)
# ============================================================
print("\n" + "=" * 80)
print("개선된 MTF 백테스트: 1H W패턴 + 15M 진입신호")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # 1H W 패턴 (L1 - H - L2)
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]  # 넥라인
    L2 = points_1h[i+2]
    
    # W 패턴 조건: L2 <= L1 * 1.03 (L2가 L1보다 크게 높지 않아야 함)
    # 더블바텀은 L1 ≈ L2 또는 L2 < L1
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:  # L2가 L1보다 3% 이상 높으면 W 패턴 아님
        continue
    
    # Gap 조건 (넥라인과 L2 사이 거리)
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 2:  # 최소 2% Gap
        continue
    
    # 15M 진입 신호 감지
    signals = detect_15m_entry_signals_v2(df_15m, swing_highs_15m, swing_lows_15m, 
                                          L2['time'], L2['price'], H['price'])
    
    if signals is None:
        continue
    
    signal_count = signals['signal_count']
    
    # 진입 시점 결정
    # 1) 15M 신호로 진입 (MTF 일치 시)
    # 2) 신호 없으면 넥라인 돌파 대기
    
    if signals['entry_time'] is not None and signal_count >= 1:
        entry_time = signals['entry_time']
        entry_price = signals['entry_price']
        entry_type = '15M_SIGNAL'
    else:
        # 넥라인 돌파 대기
        l2_idx = L2['idx']
        future_1h = df_1h.iloc[l2_idx+1:l2_idx+50]
        breakout_idx = None
        for j, (_, row) in enumerate(future_1h.iterrows()):
            if row['close'] > H['price']:
                breakout_idx = l2_idx + 1 + j
                break
        if breakout_idx is None:
            continue
        entry_time = df_1h.iloc[breakout_idx]['datetime']
        entry_price = df_1h.iloc[breakout_idx]['close']
        entry_type = 'NECKLINE_BREAK'
    
    # SL = min(L1, L2)
    sl_price = min(L1['price'], L2['price'])
    sl_gap = (entry_price - sl_price) / sl_price * 100
    
    # 백테스트 (15M 기준)
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    # TP = Gap 목표 (Entry + Gap%)
    tp_pct = min(gap * 0.8, 5)  # Gap의 80% 또는 최대 5%
    tp_price = entry_price * (1 + tp_pct / 100)
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0  # Maximum Favorable Excursion
    
    for _, row in post.iterrows():
        # MFE 기록
        current_pnl = (row['high'] - entry_price) / entry_price * 100
        mfe = max(mfe, current_pnl)
        
        if row['high'] >= tp_price:
            pnl = tp_pct
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
        'sl_gap': sl_gap,
        'tp_pct': tp_pct,
        'gap': gap,
        'l_ratio': l_ratio,  # (L2-L1)/L1
        'signal_count': signal_count,
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals['lh_to_hh'],
        'higher_low': signals['higher_low'],
        'bullish_volume': signals['bullish_candle_volume'],
        'entry_type': entry_type,
        'pnl': pnl,
        'mfe': mfe,
        'exit_type': exit_type
    })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호가 없습니다. 조건을 완화해야 합니다.")
    exit()

# ============================================================
# Gap 범위별 성과
# ============================================================
print("\n" + "=" * 80)
print("Gap 범위별 성과")
print("=" * 80)

print(f"\n{'Gap 범위':>12} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'평균MFE':>10}")
print("-" * 55)

gap_bins = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 100)]
for low, high in gap_bins:
    subset = df_results[(df_results['gap'] >= low) & (df_results['gap'] < high)]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        avg_mfe = subset['mfe'].mean()
        print(f"{f'{low}-{high}%':>12} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {avg_mfe:>10.2f}")

# ============================================================
# 신호 개수별 성과
# ============================================================
print("\n" + "=" * 80)
print("15M 신호 개수별 성과")
print("=" * 80)

print(f"\n{'신호개수':>8} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'평균MFE':>10}")
print("-" * 50)

for sig_count in [0, 1, 2, 3, 4]:
    subset = df_results[df_results['signal_count'] == sig_count]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        avg_mfe = subset['mfe'].mean()
        print(f"{sig_count:>8} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {avg_mfe:>10.2f}")

# ============================================================
# 진입 타입별 성과 (15M 신호 vs 넥라인 돌파)
# ============================================================
print("\n" + "=" * 80)
print("진입 타입별 성과")
print("=" * 80)

for entry_type in df_results['entry_type'].unique():
    subset = df_results[df_results['entry_type'] == entry_type]
    win_rate = (subset['pnl'] > 0).mean() * 100
    avg_pnl = subset['pnl'].mean()
    avg_mfe = subset['mfe'].mean()
    print(f"\n[{entry_type}]")
    print(f"  건수: {len(subset)}, 승률: {win_rate:.1f}%, 평균PnL: {avg_pnl:.2f}%, 평균MFE: {avg_mfe:.2f}%")

# ============================================================
# 개별 15M 신호별 성과
# ============================================================
print("\n" + "=" * 80)
print("개별 15M 신호별 성과")
print("=" * 80)

for signal_name in ['trendline_break', 'lh_to_hh', 'higher_low', 'bullish_volume']:
    # 해당 신호 있음
    with_signal = df_results[df_results[signal_name] == True]
    # 해당 신호 없음
    without_signal = df_results[df_results[signal_name] == False]
    
    print(f"\n[{signal_name}]")
    if len(with_signal) > 0:
        win_w = (with_signal['pnl'] > 0).mean() * 100
        pnl_w = with_signal['pnl'].mean()
        mfe_w = with_signal['mfe'].mean()
        print(f"  있음: {len(with_signal)}건, 승률: {win_w:.1f}%, 평균PnL: {pnl_w:.2f}%, MFE: {mfe_w:.2f}%")
    if len(without_signal) > 0:
        win_wo = (without_signal['pnl'] > 0).mean() * 100
        pnl_wo = without_signal['pnl'].mean()
        mfe_wo = without_signal['mfe'].mean()
        print(f"  없음: {len(without_signal)}건, 승률: {win_wo:.1f}%, 평균PnL: {pnl_wo:.2f}%, MFE: {mfe_wo:.2f}%")

# ============================================================
# 최적 조건 조합 분석
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 조합 분석")
print("=" * 80)

# Gap >= 4% + 신호 1개 이상
cond1 = df_results[(df_results['gap'] >= 4) & (df_results['signal_count'] >= 1)]
# Gap >= 4% + 신호 2개 이상
cond2 = df_results[(df_results['gap'] >= 4) & (df_results['signal_count'] >= 2)]
# Gap >= 5% + 신호 1개 이상
cond3 = df_results[(df_results['gap'] >= 5) & (df_results['signal_count'] >= 1)]
# Gap >= 5% + 하락추세선 돌파
cond4 = df_results[(df_results['gap'] >= 5) & (df_results['trendline_break'] == True)]
# Gap >= 4% + HL (저점올림)
cond5 = df_results[(df_results['gap'] >= 4) & (df_results['higher_low'] == True)]

conditions = [
    ('Gap>=4% + 신호1+', cond1),
    ('Gap>=4% + 신호2+', cond2),
    ('Gap>=5% + 신호1+', cond3),
    ('Gap>=5% + 추세돌파', cond4),
    ('Gap>=4% + HL', cond5)
]

print(f"\n{'조건':>20} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 65)

for name, subset in conditions:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{name:>20} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# MTF 일치 vs 불일치 (개선된 기준)
# ============================================================
print("\n" + "=" * 80)
print("MTF 일치 vs 불일치 (Gap>=4% 필터)")
print("=" * 80)

# Gap >= 4%만 대상
df_gap4 = df_results[df_results['gap'] >= 4]

# 15M 진입 (신호 1개 이상)
mtf_match = df_gap4[(df_gap4['entry_type'] == '15M_SIGNAL') & (df_gap4['signal_count'] >= 1)]
# 넥라인 돌파 대기
mtf_nomatch = df_gap4[df_gap4['entry_type'] == 'NECKLINE_BREAK']

if len(mtf_match) > 0 and len(mtf_nomatch) > 0:
    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    MTF 일치 vs 넥라인 돌파 대기 비교                        │
│                         (Gap >= 4% 필터 적용)                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        넥라인돌파대기           15M신호진입                 │
│  ──────────────────────────────────────────────────────────────────────── │
│  건수                  {len(mtf_nomatch):>10}건       {len(mtf_match):>10}건                     │
│  승률                  {(mtf_nomatch['pnl']>0).mean()*100:>10.1f}%       {(mtf_match['pnl']>0).mean()*100:>10.1f}%                     │
│  평균 PnL             {mtf_nomatch['pnl'].mean():>10.2f}%       {mtf_match['pnl'].mean():>10.2f}%                     │
│  평균 MFE             {mtf_nomatch['mfe'].mean():>10.2f}%       {mtf_match['mfe'].mean():>10.2f}%                     │
│  총 PnL               {mtf_nomatch['pnl'].sum():>10.1f}%       {mtf_match['pnl'].sum():>10.1f}%                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 월별 성과
# ============================================================
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')

# 최적 조건: Gap >= 4% + 신호 1개 이상
optimal = df_results[(df_results['gap'] >= 4) & (df_results['signal_count'] >= 1)]

if len(optimal) > 0:
    monthly = optimal.groupby('month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
    }).round(2)
    monthly.columns = ['총PnL', '거래수', '승률']
    
    print(f"\n[최적 조건: Gap>=4% + 신호1+]")
    print(f"  총 거래: {len(optimal)}건")
    print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
    print(f"  월 평균 수익: {monthly['총PnL'].mean():.2f}%")
    print(f"  수익 월: {(monthly['총PnL'] > 0).sum()}개월 / {len(monthly)}개월")
    print(f"  손실 월: {(monthly['총PnL'] < 0).sum()}개월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

# 최적 조건 통계
if len(optimal) > 0:
    opt_winrate = (optimal['pnl'] > 0).mean() * 100
    opt_avg_pnl = optimal['pnl'].mean()
    opt_monthly = monthly['총PnL'].mean()
else:
    opt_winrate = 0
    opt_avg_pnl = 0
    opt_monthly = 0

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         MTF 전략 최종 결론                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 1H 조건: W 패턴 (L1-H-L2), Gap >= 4%                                    │
│  ■ 15M 진입 신호 (1개 이상 필요):                                          │
│    ✓ 하락추세선 돌파                                                       │
│    ✓ LH → HH 전환                                                          │
│    ✓ HL (저점 올림)                                                        │
│    ✓ 양봉 + 거래량                                                         │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ■ 최적 조건 (Gap>=4% + 신호1+) 성과:                                      │
│    - 총 거래: {len(optimal) if len(optimal)>0 else 0}건                                                     │
│    - 승률: {opt_winrate:.1f}%                                                         │
│    - 평균 PnL: {opt_avg_pnl:.2f}%                                                   │
│    - 월 평균: {opt_monthly:.2f}%                                                     │
│    - 3x 레버리지: {opt_monthly*3:.1f}%/월                                           │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ■ 핵심 원칙:                                                              │
│    "1H 큰 그림 (W패턴) + 15M 진입 타이밍 (추세 반전) = MTF 일치"           │
│    "Gap >= 4% = 충분한 에너지 축적"                                         │
│    "15M 신호 = L2에서 반등 확인"                                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_entry_improved_results.csv', index=False)
print("\n결과 저장: mtf_entry_improved_results.csv")
