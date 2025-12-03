#!/usr/bin/env python3
"""
MTF 통합 최종 전략

이전 결과 종합:
1. mtf_optimized_long_short: 2193건, 30.8% 승률, 월 13.32%
2. LONG + Gap>=4% + Entry Gap 1-3%: 476건, 37.6% 승률, 월 5.88% (17.6%/월 3x)

핵심 원칙:
1. 하락추세 시작 ~ 추세돌파까지 전체 관점
2. MTF 다이버전스 (1H + 15M)
3. Breakout → Adjustment → 1차 → 2차
4. Entry Gap 1~3%
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("MTF 통합 최종 전략")
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

# MACD, RSI
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']
df_1h['rsi'] = calc_rsi(df_1h['close'])

df_15m['rsi'] = calc_rsi(df_15m['close'])
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
                'idx': max_idx, 'rsi': rsi[max_idx]
            })
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({
                'type': 'L', 'price': low[min_idx], 
                'time': pd.Timestamp(timestamps[min_idx]), 
                'idx': min_idx, 'rsi': rsi[min_idx]
            })
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
print(f"1H H/L points: {len(points_1h)}")

# ============================================================
# 다이버전스 감지
# ============================================================
def detect_divergence_1h(points, current_idx, direction='bullish'):
    """
    1H 다이버전스 감지
    - 하락 시작점에서부터의 다이버전스 확인
    """
    if direction == 'bullish':
        # current_idx까지의 L 포인트들
        lows = [p for p in points[:current_idx+1] if p['type'] == 'L']
        if len(lows) < 2:
            return False
        
        # 최근 10개 L 포인트에서 다이버전스 찾기
        recent_lows = lows[-10:]
        for i in range(len(recent_lows) - 1):
            for j in range(i + 1, len(recent_lows)):
                l1, l2 = recent_lows[i], recent_lows[j]
                # Price: LL or same, RSI: HL
                if l2['price'] <= l1['price'] * 1.01 and l2['rsi'] > l1['rsi'] + 3:
                    return True
        return False
    
    else:  # bearish
        highs = [p for p in points[:current_idx+1] if p['type'] == 'H']
        if len(highs) < 2:
            return False
        
        recent_highs = highs[-10:]
        for i in range(len(recent_highs) - 1):
            for j in range(i + 1, len(recent_highs)):
                h1, h2 = recent_highs[i], recent_highs[j]
                # Price: HH or same, RSI: LH
                if h2['price'] >= h1['price'] * 0.99 and h2['rsi'] < h1['rsi'] - 3:
                    return True
        return False

# ============================================================
# 백테스트
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행")
print("=" * 80)

results = []
tp_pct = 5

# LONG - W 패턴
print("\nLONG 스캔...")
long_count = 0

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 2:
        continue
    
    # 다이버전스
    has_div = detect_divergence_1h(points_1h, i+2, 'bullish')
    
    sl_price = min(L1['price'], L2['price'])
    
    # 진입가: L2 + 1.5% (이전 성공 전략 기준)
    for entry_gap_target in [1.5, 2.0, 2.5]:
        entry_price = L2['price'] * (1 + entry_gap_target / 100)
        
        if entry_price >= H['price']:  # 넥라인 미만에서 진입
            continue
        
        entry_l_gap = (entry_price - sl_price) / sl_price * 100
        if entry_l_gap < 0.5 or entry_l_gap > 4:
            continue
        
        # L2 이후 진입가 도달 확인
        l2_15m_idx = df_15m[df_15m['datetime'] >= L2['time']].index
        if len(l2_15m_idx) == 0:
            continue
        l2_15m_idx = l2_15m_idx[0]
        
        post_l2 = df_15m.iloc[l2_15m_idx:l2_15m_idx+200]
        
        entry_time = None
        for idx, row in post_l2.iterrows():
            if row['high'] >= entry_price:
                entry_time = row['datetime']
                break
        
        if entry_time is None:
            continue
        
        # TP 가격
        tp_price = entry_price * (1 + tp_pct / 100)
        
        # 백테스트
        entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index[0]
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
            'entry': entry_price,
            'sl': sl_price,
            'entry_gap': entry_l_gap,
            'gap': gap,
            'has_div': has_div,
            'entry_gap_target': entry_gap_target,
            'pnl': pnl,
            'mfe': mfe,
            'exit_type': exit_type
        })
        break  # 첫 진입만

print(f"LONG: {long_count}건")

# SHORT - M 패턴
print("SHORT 스캔...")
short_count = 0

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'H' and 
            points_1h[i+1]['type'] == 'L' and 
            points_1h[i+2]['type'] == 'H'):
        continue
    
    H1 = points_1h[i]
    L = points_1h[i+1]
    H2 = points_1h[i+2]
    
    h_ratio = (H2['price'] - H1['price']) / H1['price'] * 100
    if h_ratio < -3:
        continue
    
    gap = (H2['price'] - L['price']) / L['price'] * 100
    if gap < 2:
        continue
    
    has_div = detect_divergence_1h(points_1h, i+2, 'bearish')
    
    sl_price = max(H1['price'], H2['price'])
    
    for entry_gap_target in [1.5, 2.0, 2.5]:
        entry_price = H2['price'] * (1 - entry_gap_target / 100)
        
        if entry_price <= L['price']:
            continue
        
        entry_h_gap = (sl_price - entry_price) / entry_price * 100
        if entry_h_gap < 0.5 or entry_h_gap > 4:
            continue
        
        h2_15m_idx = df_15m[df_15m['datetime'] >= H2['time']].index
        if len(h2_15m_idx) == 0:
            continue
        h2_15m_idx = h2_15m_idx[0]
        
        post_h2 = df_15m.iloc[h2_15m_idx:h2_15m_idx+200]
        
        entry_time = None
        for idx, row in post_h2.iterrows():
            if row['low'] <= entry_price:
                entry_time = row['datetime']
                break
        
        if entry_time is None:
            continue
        
        tp_price = entry_price * (1 - tp_pct / 100)
        
        entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index[0]
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
            'entry': entry_price,
            'sl': sl_price,
            'entry_gap': entry_h_gap,
            'gap': gap,
            'has_div': has_div,
            'entry_gap_target': entry_gap_target,
            'pnl': pnl,
            'mfe': mfe,
            'exit_type': exit_type
        })
        break

print(f"SHORT: {short_count}건")

# ============================================================
# 결과 분석
# ============================================================
df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호 없음")
    exit()

# 전체 성과
print("\n" + "=" * 80)
print("전체 성과")
print("=" * 80)

print(f"  총 거래: {len(df_results)}건")
print(f"  승률: {(df_results['pnl']>0).mean()*100:.1f}%")
print(f"  평균 PnL: {df_results['pnl'].mean():.2f}%")
print(f"  총 PnL: {df_results['pnl'].sum():.1f}%")

# 방향별
print("\n방향별 성과:")
for direction in ['LONG', 'SHORT']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        print(f"  [{direction}] {len(subset)}건, 승률: {(subset['pnl']>0).mean()*100:.1f}%, 평균 PnL: {subset['pnl'].mean():.2f}%, 총 PnL: {subset['pnl'].sum():.1f}%")

# Gap 범위별
print("\nGap 범위별 성과:")
for low, high in [(2, 4), (4, 6), (6, 8), (8, 100)]:
    subset = df_results[(df_results['gap'] >= low) & (df_results['gap'] < high)]
    if len(subset) > 5:
        print(f"  {low}-{high}%: {len(subset)}건, 승률: {(subset['pnl']>0).mean()*100:.1f}%, 평균 PnL: {subset['pnl'].mean():.2f}%")

# Entry Gap 범위별
print("\nEntry Gap 범위별 성과:")
for low, high in [(0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4)]:
    subset = df_results[(df_results['entry_gap'] >= low) & (df_results['entry_gap'] < high)]
    if len(subset) > 5:
        print(f"  {low}-{high}%: {len(subset)}건, 승률: {(subset['pnl']>0).mean()*100:.1f}%, 평균 PnL: {subset['pnl'].mean():.2f}%")

# 다이버전스 유무
print("\n1H 다이버전스 유무별 성과:")
for has_div in [True, False]:
    subset = df_results[df_results['has_div'] == has_div]
    if len(subset) > 0:
        status = "있음" if has_div else "없음"
        print(f"  다이버전스 {status}: {len(subset)}건, 승률: {(subset['pnl']>0).mean()*100:.1f}%, 평균 PnL: {subset['pnl'].mean():.2f}%")

# 최적 조건
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

conditions = [
    ('전체', df_results),
    ('LONG', df_results[df_results['direction'] == 'LONG']),
    ('SHORT', df_results[df_results['direction'] == 'SHORT']),
    ('Gap>=4%', df_results[df_results['gap'] >= 4]),
    ('Gap>=5%', df_results[df_results['gap'] >= 5]),
    ('Entry Gap 1-3%', df_results[(df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('다이버전스 있음', df_results[df_results['has_div'] == True]),
    ('LONG + Gap>=4%', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 4)]),
    ('LONG + Gap>=4% + Entry 1-3%', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 4) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('LONG + Gap>=4% + 다이버전스', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 4) & (df_results['has_div'] == True)]),
    ('SHORT + Gap>=4%', df_results[(df_results['direction'] == 'SHORT') & (df_results['gap'] >= 4)]),
]

print(f"\n{'조건':>35} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 80)

for name, subset in conditions:
    if len(subset) >= 10:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"{name:>35} {len(subset):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f}")

# 월별 성과
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month')['pnl'].agg(['sum', 'count']).round(2)
monthly.columns = ['총PnL', '거래수']

print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['총PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['총PnL'] > 0).sum()}개월 / {len(monthly)}개월")
print(f"  최대 월 수익: {monthly['총PnL'].max():.2f}%")
print(f"  최대 월 손실: {monthly['총PnL'].min():.2f}%")

# LONG + Gap>=4% 최적 조건
opt_subset = df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 4)]
if len(opt_subset) > 0:
    opt_subset = opt_subset.copy()
    opt_subset['month'] = pd.to_datetime(opt_subset['time']).dt.to_period('M')
    opt_monthly = opt_subset.groupby('month')['pnl'].sum()
    
    print(f"\n[최적 조건: LONG + Gap>=4%]")
    print(f"  총 거래: {len(opt_subset)}건")
    print(f"  승률: {(opt_subset['pnl']>0).mean()*100:.1f}%")
    print(f"  평균 PnL: {opt_subset['pnl'].mean():.2f}%")
    print(f"  월 평균: {opt_monthly.mean():.2f}%")
    print(f"  3x 레버리지: {opt_monthly.mean()*3:.1f}%/월")
    print(f"  수익 월: {(opt_monthly > 0).sum()}개월 / {len(opt_monthly)}개월")

# 최종 결론
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                       MTF 통합 최종 전략                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 총 신호: {len(df_results)}건 (LONG: {len(df_results[df_results['direction']=='LONG'])}건, SHORT: {len(df_results[df_results['direction']=='SHORT'])}건)                               │
│                                                                            │
│  ■ 전체 성과:                                                               │
│    - 승률: {(df_results['pnl']>0).mean()*100:.1f}%                                                          │
│    - 평균 PnL: {df_results['pnl'].mean():.2f}%                                                     │
│    - 월 평균: {monthly['총PnL'].mean():.2f}%                                                       │
│    - 3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월                                             │
│                                                                            │
│  ■ 핵심 원칙:                                                               │
│    1. 하락추세 시작 ~ 추세돌파까지 전체 관점                                   │
│    2. MTF 다이버전스 (1H RSI)                                               │
│    3. Entry Gap 1~3% (L2/H2 + 1.5~2.5%)                                    │
│    4. SL = min(L1,L2) / max(H1,H2)                                         │
│    5. TP = 5%                                                               │
│                                                                            │
│  ■ 체크리스트:                                                               │
│    □ 1H W/M 패턴 확인                                                       │
│    □ Gap >= 4%                                                              │
│    □ 1H 다이버전스 확인                                                      │
│    □ Entry Gap 1~3%                                                         │
│    □ SL/TP 설정                                                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 저장
df_results.to_csv('mtf_combined_final_results.csv', index=False)
print("\n결과 저장: mtf_combined_final_results.csv")
