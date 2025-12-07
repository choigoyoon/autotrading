#!/usr/bin/env python3
"""
패턴 + 에너지(Gap) 분석

1. 패턴 인식: W(더블바텀), M(더블탑), 역헤숄, 헤숄
2. 레벨 확인: 추세선, 저항, 지지
3. Gap 측정: 얼마나 눌렸다가/올랐다가 돌파하는지 = 에너지
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("패턴 + 에너지(Gap) 분석")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

# 1시간봉
df_1h = df.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

# H/L 추출
def extract_hl_points(df_1h):
    hist = df_1h['hist'].values
    high = df_1h['high'].values
    low = df_1h['low'].values
    timestamps = df_1h['datetime'].values
    
    points = []
    i = 0
    n = len(hist)
    
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0:
                i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': timestamps[max_idx], 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0:
                i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': timestamps[min_idx], 'idx': min_idx})
        else:
            i += 1
    return points

points = extract_hl_points(df_1h)
print(f"데이터: {len(df_1h):,} 1H candles")
print(f"H/L points: {len(points)}")

# ============================================================
# 1. 패턴 인식
# ============================================================
print("\n" + "=" * 80)
print("1. 패턴 인식 (W, M, 역헤숄, 헤숄)")
print("=" * 80)

patterns = []

for i in range(len(points) - 4):
    p0, p1, p2, p3, p4 = points[i:i+5]
    
    # W 패턴 (더블바텀): L-H-L-H-L 또는 L-H-L
    if p0['type'] == 'L' and p1['type'] == 'H' and p2['type'] == 'L':
        L1, H, L2 = p0['price'], p1['price'], p2['price']
        
        # 두 저점이 비슷 (3% 이내)
        if abs(L2 - L1) / L1 < 0.03:
            # Gap 계산: 넥라인(H)에서 저점(L2)까지 거리
            gap = (H - L2) / L2 * 100
            
            patterns.append({
                'type': 'W',
                'time': p2['time'],
                'idx': p2['idx'],
                'L1': L1, 'H': H, 'L2': L2,
                'neckline': H,
                'sl_level': min(L1, L2),
                'gap': gap,  # 에너지 측정
                'gap_pct': gap
            })
    
    # M 패턴 (더블탑): H-L-H-L-H 또는 H-L-H
    if p0['type'] == 'H' and p1['type'] == 'L' and p2['type'] == 'H':
        H1, L, H2 = p0['price'], p1['price'], p2['price']
        
        # 두 고점이 비슷 (3% 이내)
        if abs(H2 - H1) / H1 < 0.03:
            # Gap 계산: 고점(H2)에서 넥라인(L)까지 거리
            gap = (H2 - L) / L * 100
            
            patterns.append({
                'type': 'M',
                'time': p2['time'],
                'idx': p2['idx'],
                'H1': H1, 'L': L, 'H2': H2,
                'neckline': L,
                'sl_level': max(H1, H2),
                'gap': gap,
                'gap_pct': gap
            })
    
    # 역헤숄 (Inverse Head & Shoulders): L-H-L(더깊음)-H-L
    if (p0['type'] == 'L' and p1['type'] == 'H' and p2['type'] == 'L' and 
        p3['type'] == 'H' and p4['type'] == 'L'):
        LS, H1, Head, H2, RS = p0['price'], p1['price'], p2['price'], p3['price'], p4['price']
        
        # Head가 가장 낮고, 양 어깨가 비슷
        if Head < LS and Head < RS and abs(LS - RS) / LS < 0.03:
            neckline = (H1 + H2) / 2
            gap = (neckline - Head) / Head * 100
            
            patterns.append({
                'type': 'IHS',  # Inverse Head & Shoulders
                'time': p4['time'],
                'idx': p4['idx'],
                'LS': LS, 'Head': Head, 'RS': RS,
                'neckline': neckline,
                'sl_level': Head,
                'gap': gap,
                'gap_pct': gap
            })
    
    # 헤숄 (Head & Shoulders): H-L-H(더높음)-L-H
    if (p0['type'] == 'H' and p1['type'] == 'L' and p2['type'] == 'H' and 
        p3['type'] == 'L' and p4['type'] == 'H'):
        LS, L1, Head, L2, RS = p0['price'], p1['price'], p2['price'], p3['price'], p4['price']
        
        # Head가 가장 높고, 양 어깨가 비슷
        if Head > LS and Head > RS and abs(LS - RS) / LS < 0.03:
            neckline = (L1 + L2) / 2
            gap = (Head - neckline) / neckline * 100
            
            patterns.append({
                'type': 'HS',  # Head & Shoulders
                'time': p4['time'],
                'idx': p4['idx'],
                'LS': LS, 'Head': Head, 'RS': RS,
                'neckline': neckline,
                'sl_level': Head,
                'gap': gap,
                'gap_pct': gap
            })

df_patterns = pd.DataFrame(patterns)
print(f"\n인식된 패턴:")
for ptype in ['W', 'M', 'IHS', 'HS']:
    count = len(df_patterns[df_patterns['type'] == ptype])
    print(f"  {ptype}: {count}개")

# ============================================================
# 2. 패턴별 백테스트
# ============================================================
print("\n" + "=" * 80)
print("2. 패턴별 성과 (넥라인 돌파 후)")
print("=" * 80)

def backtest_pattern(df_1h, pattern, max_bars=200, tp_pct=3.5):
    """패턴 완성 후 넥라인 돌파 시 백테스트"""
    idx = pattern['idx']
    neckline = pattern['neckline']
    sl_level = pattern['sl_level']
    ptype = pattern['type']
    
    # 롱 패턴 (W, IHS) vs 숏 패턴 (M, HS)
    is_long = ptype in ['W', 'IHS']
    
    # 넥라인 돌파 감지
    future = df_1h.iloc[idx+1:idx+max_bars]
    breakout_idx = None
    
    for j, (_, row) in enumerate(future.iterrows()):
        if is_long and row['close'] > neckline:
            breakout_idx = idx + 1 + j
            break
        elif not is_long and row['close'] < neckline:
            breakout_idx = idx + 1 + j
            break
    
    if breakout_idx is None:
        return {'result': 'NO_BREAKOUT', 'pnl': 0}
    
    # 진입
    entry = df_1h.iloc[breakout_idx]['close']
    
    # TP/SL 설정
    if is_long:
        tp_price = entry * (1 + tp_pct/100)
        sl_price = sl_level
    else:
        tp_price = entry * (1 - tp_pct/100)
        sl_price = sl_level
    
    # 결과 확인
    post_breakout = df_1h.iloc[breakout_idx+1:breakout_idx+max_bars]
    
    for _, row in post_breakout.iterrows():
        if is_long:
            if row['high'] >= tp_price:
                return {'result': 'TP', 'pnl': tp_pct}
            if row['low'] <= sl_price:
                return {'result': 'SL', 'pnl': (sl_price - entry) / entry * 100}
        else:
            if row['low'] <= tp_price:
                return {'result': 'TP', 'pnl': tp_pct}
            if row['high'] >= sl_price:
                return {'result': 'SL', 'pnl': (entry - sl_price) / entry * 100}
    
    if len(post_breakout) > 0:
        last = post_breakout.iloc[-1]['close']
        pnl = (last - entry) / entry * 100 if is_long else (entry - last) / entry * 100
        return {'result': 'TIMEOUT', 'pnl': pnl}
    
    return {'result': 'NO_DATA', 'pnl': 0}

# 백테스트 실행
results = []
for _, pattern in df_patterns.iterrows():
    bt = backtest_pattern(df_1h, pattern)
    results.append({
        **pattern.to_dict(),
        'bt_result': bt['result'],
        'bt_pnl': bt['pnl']
    })

df_results = pd.DataFrame(results)

# 패턴별 성과
print("\n[패턴별 성과]")
print("-" * 70)
for ptype in ['W', 'M', 'IHS', 'HS']:
    subset = df_results[df_results['type'] == ptype]
    valid = subset[subset['bt_result'].isin(['TP', 'SL', 'TIMEOUT'])]
    if len(valid) > 0:
        tp_rate = (valid['bt_result'] == 'TP').mean() * 100
        avg_pnl = valid['bt_pnl'].mean()
        print(f"{ptype:>5}: {len(valid):>4}건, TP율: {tp_rate:>5.1f}%, 평균PnL: {avg_pnl:>6.2f}%")

# ============================================================
# 3. GAP(에너지)별 성과 분석 - 핵심!
# ============================================================
print("\n" + "=" * 80)
print("3. GAP(에너지)별 성과 분석 ★ 핵심")
print("=" * 80)

print("""
Gap의 의미:
- 넥라인에서 저점(W)/고점(M)까지의 거리
- Gap이 클수록 = 더 많이 눌림/올림 = 축적된 에너지 큼
- Gap이 작으면 = 얕은 조정 = 돌파해도 힘 약함
""")

# Gap 구간별 분석
valid_results = df_results[df_results['bt_result'].isin(['TP', 'SL', 'TIMEOUT'])]

gap_bins = [0, 2, 4, 6, 8, 10, np.inf]
gap_labels = ['0-2%', '2-4%', '4-6%', '6-8%', '8-10%', '10%+']
valid_results = valid_results.copy()
valid_results['gap_bin'] = pd.cut(valid_results['gap_pct'], bins=gap_bins, labels=gap_labels)

print("\n[Gap(에너지) 구간별 성과]")
print("-" * 80)
print(f"{'Gap 구간':<12} {'거래수':>8} {'TP율%':>8} {'SL율%':>8} {'평균PnL%':>10} {'총PnL%':>10}")
print("-" * 80)

for gap_label in gap_labels:
    subset = valid_results[valid_results['gap_bin'] == gap_label]
    if len(subset) > 0:
        tp_rate = (subset['bt_result'] == 'TP').mean() * 100
        sl_rate = (subset['bt_result'] == 'SL').mean() * 100
        avg_pnl = subset['bt_pnl'].mean()
        total_pnl = subset['bt_pnl'].sum()
        marker = " ★" if tp_rate >= 55 else ""
        print(f"{gap_label:<12} {len(subset):>8} {tp_rate:>8.1f} {sl_rate:>8.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}{marker}")

# ============================================================
# 4. 패턴 + Gap 교차 분석
# ============================================================
print("\n" + "=" * 80)
print("4. 패턴 × Gap 교차 분석")
print("=" * 80)

print("\n[패턴별 Gap 구간 성과 (TP율%)]")
print("-" * 70)
print(f"{'패턴':>6}", end="")
for gap_label in gap_labels:
    print(f"{gap_label:>10}", end="")
print()
print("-" * 70)

for ptype in ['W', 'M', 'IHS', 'HS']:
    print(f"{ptype:>6}", end="")
    for gap_label in gap_labels:
        subset = valid_results[(valid_results['type'] == ptype) & (valid_results['gap_bin'] == gap_label)]
        if len(subset) >= 3:
            tp_rate = (subset['bt_result'] == 'TP').mean() * 100
            print(f"{tp_rate:>8.0f}%({len(subset):>2})", end="")
        else:
            print(f"{'N/A':>10}", end="")
    print()

# ============================================================
# 5. 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("5. 최적 조건 탐색")
print("=" * 80)

# 롱 패턴 (W, IHS) - Gap >= 4%
long_patterns = valid_results[valid_results['type'].isin(['W', 'IHS'])]
long_high_gap = long_patterns[long_patterns['gap_pct'] >= 4]

print(f"\n[롱 패턴 (W, IHS) + Gap >= 4%]")
print(f"  거래수: {len(long_high_gap)}")
print(f"  TP율: {(long_high_gap['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {long_high_gap['bt_pnl'].mean():.2f}%")

# 숏 패턴 (M, HS) - Gap >= 4%
short_patterns = valid_results[valid_results['type'].isin(['M', 'HS'])]
short_high_gap = short_patterns[short_patterns['gap_pct'] >= 4]

print(f"\n[숏 패턴 (M, HS) + Gap >= 4%]")
print(f"  거래수: {len(short_high_gap)}")
print(f"  TP율: {(short_high_gap['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {short_high_gap['bt_pnl'].mean():.2f}%")

# ============================================================
# 6. 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("6. 최종 결론")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                    패턴 + 에너지(Gap) 분석 결론                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1️⃣  패턴 인식 (우선)                                                      │
│      W (더블바텀), M (더블탑), 역헤숄(IHS), 헤숄(HS)                        │
│                                                                            │
│  2️⃣  레벨 확인                                                             │
│      - 넥라인 (돌파 기준)                                                  │
│      - 지지/저항 (손절 기준)                                               │
│                                                                            │
│  3️⃣  Gap(에너지) 측정 ★ 핵심                                              │
│      - Gap = 넥라인 ~ 저점(or 고점) 거리                                   │
│      - Gap 클수록 = 더 많이 눌림 = 반등 에너지 큼                          │
│                                                                            │
│      Gap < 2%  : 얕은 조정 → 힘 약함 → 페이크 많음                         │
│      Gap 2-4%  : 보통 조정 → 보통 힘                                       │
│      Gap 4-6%  : 깊은 조정 → 힘 강함 ★                                     │
│      Gap > 6%  : 매우 깊은 조정 → 힘 매우 강함 ★★                          │
│                                                                            │
│  4️⃣  진입 전략                                                             │
│      - 패턴 인식 → 대기                                                    │
│      - 넥라인 돌파 + Gap >= 4% → 진입                                      │
│      - 손절: 패턴의 저점(W)/고점(M) 이탈 시                                │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   시각화:                                                                  │
│                                                                            │
│      넥라인 ─────────────────────                                          │
│                    ↑                                                       │
│               Gap (에너지)                                                 │
│                    ↓                                                       │
│         ●─────────●                                                        │
│        L1         L2 (더블바텀)                                            │
│                                                                            │
│   Gap이 클수록 → 더 많이 눌림 → 반등 에너지 축적 → 돌파 성공률 ↑          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('pattern_energy_results.csv', index=False)
print("\n결과 저장: pattern_energy_results.csv")
