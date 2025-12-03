#!/usr/bin/env python3
"""
최적화된 MTF 진입 전략

핵심 문제 해결:
1. Gap이 크면 SL도 커짐 → 손절 많음
2. 진입 타이밍이 너무 이르거나 늦음

해결책:
1. Entry-L Gap (진입가격 - L값) 1~3% 유지
2. 15M 신호 중 가장 효과적인 것 선별
3. 더 정교한 진입 타이밍

사용자 원칙:
- 1H W 패턴 형성 중
- 15M에서: 하락추세선 돌파 또는 LH→HH 전환 또는 HL 저점올림 + 양봉+거래량
- MTF 일치해야 진짜
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("최적화된 MTF 진입 전략")
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

# 15M 볼륨 평균
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# H/L 추출 함수
def extract_hl_points(df):
    hist = df['hist'].values
    high = df['high'].values
    low = df['low'].values
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

# 15M 스윙 포인트
def get_15m_swing_points(df_15m, window=5):
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    times = df_15m['datetime'].values
    
    swing_highs, swing_lows = [], []
    
    for i in range(window, len(df_15m) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append({'price': highs[i], 'time': pd.Timestamp(times[i]), 'idx': i})
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append({'price': lows[i], 'time': pd.Timestamp(times[i]), 'idx': i})
    
    return swing_highs, swing_lows

swing_highs_15m, swing_lows_15m = get_15m_swing_points(df_15m)
print(f"15M Swing Highs: {len(swing_highs_15m)}, Swing Lows: {len(swing_lows_15m)}")

# ============================================================
# 15M 진입 신호 감지 (최적화)
# ============================================================
def detect_15m_entry_optimal(df_15m, swing_highs, swing_lows, l2_time, l2_price, h_neckline_price, l1_price):
    """
    L2 형성 근처에서 15M 진입 신호 감지 (최적화 버전)
    
    핵심 변경:
    1. 진입가가 L2보다 너무 높지 않도록 (Entry-L Gap 제한)
    2. 가장 빠른 신호 시점에서 진입
    3. 신호 품질 점수화
    """
    
    # L2 형성 전 6시간 ~ L2 형성 후 24시간
    search_start = l2_time - pd.Timedelta(hours=6)
    search_end = l2_time + pd.Timedelta(hours=24)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 20:
        return None
    
    window_highs = [h for h in swing_highs if search_start <= h['time'] <= search_end]
    window_lows = [l for l in swing_lows if search_start <= l['time'] <= search_end]
    
    signals = {
        'trendline_break': False, 'trendline_break_time': None, 'trendline_break_price': None,
        'lh_to_hh': False, 'lh_to_hh_time': None, 'lh_to_hh_price': None,
        'higher_low': False, 'higher_low_time': None, 'higher_low_price': None,
        'bullish_volume': False, 'bullish_volume_time': None, 'bullish_volume_price': None,
        'entry_time': None, 'entry_price': None,
        'signal_count': 0, 'signal_score': 0
    }
    
    entry_candidates = []
    sl_price = min(l1_price, l2_price)
    
    # 1. 하락추세선 돌파
    if len(window_highs) >= 2:
        for i in range(len(window_highs) - 1):
            h1, h2 = window_highs[i], window_highs[i + 1]
            if h2['price'] < h1['price'] * 0.998:
                h2_idx = h2['idx']
                after_h2 = df_15m[(df_15m.index > h2_idx) & (df_15m['datetime'] <= search_end)]
                
                time_diff = h2['idx'] - h1['idx']
                if time_diff > 0:
                    slope = (h2['price'] - h1['price']) / time_diff
                    
                    for idx, row in after_h2.iterrows():
                        bars_from_h2 = idx - h2_idx
                        trendline_price = h2['price'] + slope * bars_from_h2
                        
                        if row['close'] > trendline_price * 1.001:
                            signals['trendline_break'] = True
                            signals['trendline_break_time'] = row['datetime']
                            signals['trendline_break_price'] = row['close']
                            entry_candidates.append({
                                'time': row['datetime'],
                                'price': row['close'],
                                'signal': 'trendline_break',
                                'score': 3  # 추세돌파 가중치 높음
                            })
                            break
                    
                if signals['trendline_break']:
                    break
    
    # 2. LH → HH 전환
    if len(window_highs) >= 3:
        for i in range(len(window_highs) - 2):
            h1, h2, h3 = window_highs[i], window_highs[i+1], window_highs[i+2]
            if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                signals['lh_to_hh'] = True
                signals['lh_to_hh_time'] = h3['time']
                signals['lh_to_hh_price'] = h3['price']
                entry_candidates.append({
                    'time': h3['time'],
                    'price': h3['price'],
                    'signal': 'lh_to_hh',
                    'score': 2
                })
                break
    
    # 3. HL (저점 올림)
    if len(window_lows) >= 3:
        for i in range(len(window_lows) - 2):
            l1_15m, l2_15m, l3_15m = window_lows[i], window_lows[i+1], window_lows[i+2]
            if l2_15m['price'] < l1_15m['price'] and l3_15m['price'] > l2_15m['price']:
                signals['higher_low'] = True
                signals['higher_low_time'] = l3_15m['time']
                signals['higher_low_price'] = l3_15m['price']
                # HL 형성 시점의 종가로 진입
                hl_idx = l3_15m['idx']
                if hl_idx < len(df_15m):
                    hl_candle = df_15m.iloc[hl_idx]
                    entry_candidates.append({
                        'time': l3_15m['time'],
                        'price': hl_candle['close'],
                        'signal': 'higher_low',
                        'score': 2
                    })
                break
    
    # 4. 양봉 + 거래량 (L2 근처에서)
    for idx, row in df_window.iterrows():
        is_bullish = row['close'] > row['open']
        body_size = abs(row['close'] - row['open']) / row['open'] * 100
        near_l2 = row['low'] <= l2_price * 1.02  # L2의 2% 이내
        high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5
        
        if is_bullish and body_size > 0.3 and high_volume and near_l2:
            signals['bullish_volume'] = True
            signals['bullish_volume_time'] = row['datetime']
            signals['bullish_volume_price'] = row['close']
            entry_candidates.append({
                'time': row['datetime'],
                'price': row['close'],
                'signal': 'bullish_volume',
                'score': 1
            })
            break
    
    # 신호 개수
    signal_count = sum([
        signals['trendline_break'],
        signals['lh_to_hh'],
        signals['higher_low'],
        signals['bullish_volume']
    ])
    signals['signal_count'] = signal_count
    
    # 최적 진입점 선택
    # 기준: Entry-L Gap이 1~3% 범위 내에서 가장 빠른 신호
    if len(entry_candidates) > 0:
        valid_entries = []
        for cand in entry_candidates:
            entry_l_gap = (cand['price'] - sl_price) / sl_price * 100
            # Entry-L Gap 0.5~4% 범위만 허용
            if 0.5 <= entry_l_gap <= 4:
                cand['entry_l_gap'] = entry_l_gap
                valid_entries.append(cand)
        
        if len(valid_entries) > 0:
            # 점수 높고, 시간 빠른 순
            valid_entries.sort(key=lambda x: (-x['score'], x['time']))
            best = valid_entries[0]
            signals['entry_time'] = best['time']
            signals['entry_price'] = best['price']
            signals['signal_score'] = best['score']
    
    return signals

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("최적화된 MTF 백테스트")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴: L2 <= L1 * 1.03
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    # Gap (넥라인-L2)
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 2:
        continue
    
    # 15M 신호 감지
    signals = detect_15m_entry_optimal(df_15m, swing_highs_15m, swing_lows_15m, 
                                       L2['time'], L2['price'], H['price'], L1['price'])
    
    if signals is None:
        continue
    
    signal_count = signals['signal_count']
    
    # 진입
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
    
    # SL
    sl_price = min(L1['price'], L2['price'])
    entry_l_gap = (entry_price - sl_price) / sl_price * 100
    
    # 15M 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    # TP: Gap의 60% 또는 Entry-L Gap과 동일 (리스크리워드 1:1 이상)
    tp_pct = max(entry_l_gap, gap * 0.5, 2)  # 최소 2%
    tp_pct = min(tp_pct, 5)  # 최대 5%
    tp_price = entry_price * (1 + tp_pct / 100)
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    hold_time = 0
    
    for j, (_, row) in enumerate(post.iterrows()):
        current_pnl = (row['high'] - entry_price) / entry_price * 100
        mfe = max(mfe, current_pnl)
        
        if row['high'] >= tp_price:
            pnl = tp_pct
            exit_type = 'TP'
            hold_time = j * 0.25  # 시간 (15분 = 0.25시간)
            break
        if row['low'] <= sl_price:
            pnl = (sl_price - entry_price) / entry_price * 100
            exit_type = 'SL'
            hold_time = j * 0.25
            break
    
    results.append({
        'time': entry_time,
        'entry': entry_price,
        'sl': sl_price,
        'entry_l_gap': entry_l_gap,
        'tp_pct': tp_pct,
        'gap': gap,
        'l_ratio': l_ratio,
        'signal_count': signal_count,
        'signal_score': signals['signal_score'],
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals['lh_to_hh'],
        'higher_low': signals['higher_low'],
        'bullish_volume': signals['bullish_volume'],
        'entry_type': entry_type,
        'pnl': pnl,
        'mfe': mfe,
        'exit_type': exit_type,
        'hold_time': hold_time
    })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호가 없습니다.")
    exit()

# ============================================================
# Entry-L Gap 범위별 성과 (핵심!)
# ============================================================
print("\n" + "=" * 80)
print("Entry-L Gap 범위별 성과 (핵심)")
print("=" * 80)

print(f"\n{'Entry-L Gap':>15} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 60)

gap_bins = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6)]
for low, high in gap_bins:
    subset = df_results[(df_results['entry_l_gap'] >= low) & (df_results['entry_l_gap'] < high)]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# Gap(에너지) 범위별 성과
# ============================================================
print("\n" + "=" * 80)
print("Gap(에너지) 범위별 성과")
print("=" * 80)

print(f"\n{'Gap':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 55)

for low, high in [(2, 4), (4, 6), (6, 8), (8, 10), (10, 100)]:
    subset = df_results[(df_results['gap'] >= low) & (df_results['gap'] < high)]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>10} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 신호 조합별 성과
# ============================================================
print("\n" + "=" * 80)
print("신호 조합별 성과")
print("=" * 80)

print(f"\n{'신호 조합':>25} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 60)

# 추세돌파만
tl_only = df_results[(df_results['trendline_break']==True) & (df_results['lh_to_hh']==False) & (df_results['higher_low']==False)]
# 추세돌파 + HH
tl_hh = df_results[(df_results['trendline_break']==True) & (df_results['lh_to_hh']==True)]
# 추세돌파 + HL
tl_hl = df_results[(df_results['trendline_break']==True) & (df_results['higher_low']==True)]
# 추세돌파 + HH + HL (트리플 신호)
triple = df_results[(df_results['trendline_break']==True) & (df_results['lh_to_hh']==True) & (df_results['higher_low']==True)]

combos = [
    ('추세돌파만', tl_only),
    ('추세돌파 + HH전환', tl_hh),
    ('추세돌파 + HL', tl_hl),
    ('추세돌파 + HH + HL', triple)
]

for name, subset in combos:
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        print(f"{name:>25} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f}")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>35} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 80)

conditions = [
    ('전체', df_results),
    ('Entry-L Gap 1-2%', df_results[(df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 2)]),
    ('Entry-L Gap 1-3%', df_results[(df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 3)]),
    ('Gap>=4% + Entry-L 1-3%', df_results[(df_results['gap'] >= 4) & (df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 3)]),
    ('Gap>=5% + Entry-L 1-3%', df_results[(df_results['gap'] >= 5) & (df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 3)]),
    ('추세돌파 + Entry-L 1-3%', df_results[(df_results['trendline_break']==True) & (df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 3)]),
    ('추세돌파+HH + Entry-L 1-3%', df_results[(df_results['trendline_break']==True) & (df_results['lh_to_hh']==True) & (df_results['entry_l_gap'] >= 1) & (df_results['entry_l_gap'] < 3)]),
    ('Gap>=4% + 추세돌파', df_results[(df_results['gap'] >= 4) & (df_results['trendline_break']==True)]),
    ('Gap>=5% + 추세돌파', df_results[(df_results['gap'] >= 5) & (df_results['trendline_break']==True)]),
    ('Gap>=5% + 추세돌파 + HH', df_results[(df_results['gap'] >= 5) & (df_results['trendline_break']==True) & (df_results['lh_to_hh']==True)]),
]

best_condition = None
best_metric = -999

for name, subset in conditions:
    if len(subset) >= 10:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        # 평가 지표: 승률 * 평균PnL (균형잡힌 지표)
        metric = win_rate * avg_pnl / 100
        print(f"{name:>35} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")
        
        if metric > best_metric:
            best_metric = metric
            best_condition = (name, subset)

# ============================================================
# 최적 조건 월별 성과
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 월별 성과")
print("=" * 80)

if best_condition is not None:
    best_name, best_subset = best_condition
    print(f"\n[최적 조건: {best_name}]")
    
    best_subset = best_subset.copy()
    best_subset['month'] = pd.to_datetime(best_subset['time']).dt.to_period('M')
    monthly = best_subset.groupby('month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
    }).round(2)
    monthly.columns = ['총PnL', '거래수', '승률']
    
    print(f"  총 거래: {len(best_subset)}건")
    print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
    print(f"  월 평균 수익: {monthly['총PnL'].mean():.2f}%")
    print(f"  수익 월: {(monthly['총PnL'] > 0).sum()}개월 / {len(monthly)}개월")
    print(f"  손실 월: {(monthly['총PnL'] < 0).sum()}개월")
    print(f"  최대 월 수익: {monthly['총PnL'].max():.2f}%")
    print(f"  최대 월 손실: {monthly['총PnL'].min():.2f}%")

# ============================================================
# 15M 진입 vs 넥라인 돌파 비교
# ============================================================
print("\n" + "=" * 80)
print("15M 진입 vs 넥라인 돌파 비교")
print("=" * 80)

signal_entry = df_results[df_results['entry_type'] == '15M_SIGNAL']
neckline_entry = df_results[df_results['entry_type'] == 'NECKLINE_BREAK']

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    진입 타이밍 비교                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        넥라인돌파대기           15M신호진입                 │
│  ──────────────────────────────────────────────────────────────────────── │
│  건수                  {len(neckline_entry):>10}건       {len(signal_entry):>10}건                     │
│  승률                  {(neckline_entry['pnl']>0).mean()*100 if len(neckline_entry)>0 else 0:>10.1f}%       {(signal_entry['pnl']>0).mean()*100 if len(signal_entry)>0 else 0:>10.1f}%                     │
│  평균 PnL             {neckline_entry['pnl'].mean() if len(neckline_entry)>0 else 0:>10.2f}%       {signal_entry['pnl'].mean() if len(signal_entry)>0 else 0:>10.2f}%                     │
│  평균 Entry-L Gap     {neckline_entry['entry_l_gap'].mean() if len(neckline_entry)>0 else 0:>10.2f}%       {signal_entry['entry_l_gap'].mean() if len(signal_entry)>0 else 0:>10.2f}%                     │
│  총 PnL               {neckline_entry['pnl'].sum() if len(neckline_entry)>0 else 0:>10.1f}%       {signal_entry['pnl'].sum() if len(signal_entry)>0 else 0:>10.1f}%                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

# 가장 좋은 조건 통계
if best_condition is not None:
    opt = best_subset
    opt_winrate = (opt['pnl'] > 0).mean() * 100
    opt_avg_pnl = opt['pnl'].mean()
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
│  ■ 최적 조건: {best_name if best_condition else 'N/A':50}│
│                                                                            │
│  ■ 성과:                                                                   │
│    - 총 거래: {len(best_subset) if best_condition else 0}건                                                     │
│    - 승률: {opt_winrate:.1f}%                                                         │
│    - 평균 PnL: {opt_avg_pnl:.2f}%                                                   │
│    - 월 평균: {opt_monthly:.2f}%                                                     │
│    - 3x 레버리지: {opt_monthly*3:.1f}%/월                                           │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ■ 핵심 진입 조건:                                                          │
│    1H: W 패턴 (L1-H-L2), Gap >= 4~5%                                       │
│    15M: 하락추세선 돌파 (+ LH→HH 전환)                                      │
│    Entry-L Gap: 1~3% (너무 크지도 작지도 않게)                               │
│                                                                            │
│  ■ 사용자 원칙 정리:                                                        │
│    "1H 큰 그림 (W패턴) + 15M 진입 타이밍 = MTF 일치해야 진짜"               │
│    "Gap >= 4% = 충분한 에너지"                                              │
│    "15M 하락추세선 돌파 = 반전 확인"                                         │
│    "Entry-L Gap 1~3% = 적정 손절폭"                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_optimal_results.csv', index=False)
print("\n결과 저장: mtf_optimal_results.csv")

# ============================================================
# 추가: 실패 사례 분석
# ============================================================
print("\n" + "=" * 80)
print("실패 사례 분석 (SL 케이스)")
print("=" * 80)

sl_cases = df_results[df_results['exit_type'] == 'SL']
tp_cases = df_results[df_results['exit_type'] == 'TP']

print(f"\n[SL 케이스: {len(sl_cases)}건]")
if len(sl_cases) > 0:
    print(f"  평균 Entry-L Gap: {sl_cases['entry_l_gap'].mean():.2f}%")
    print(f"  평균 Gap: {sl_cases['gap'].mean():.2f}%")
    print(f"  평균 MFE: {sl_cases['mfe'].mean():.2f}%")
    print(f"  평균 손실: {sl_cases['pnl'].mean():.2f}%")
    
    # MFE가 1% 이상이었던 경우 (수익권 진입 후 손절)
    sl_with_profit = sl_cases[sl_cases['mfe'] >= 1]
    print(f"  수익권(MFE>=1%) 진입 후 손절: {len(sl_with_profit)}건 ({len(sl_with_profit)/len(sl_cases)*100:.1f}%)")

print(f"\n[TP 케이스: {len(tp_cases)}건]")
if len(tp_cases) > 0:
    print(f"  평균 Entry-L Gap: {tp_cases['entry_l_gap'].mean():.2f}%")
    print(f"  평균 Gap: {tp_cases['gap'].mean():.2f}%")
    print(f"  평균 MFE: {tp_cases['mfe'].mean():.2f}%")
    print(f"  평균 수익: {tp_cases['pnl'].mean():.2f}%")
