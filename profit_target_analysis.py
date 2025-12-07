#!/usr/bin/env python3
"""
월 15% 목표 - 수익 실현 전략 분석
추세 변곡 후 얼마나 끌고 가야 하는가?
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("월 15% 목표 - 수익 실현 전략")
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
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': timestamps[max_idx], 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': timestamps[min_idx], 'idx': min_idx})
        else:
            i += 1
    return points

points = extract_hl_points(df_1h)
print(f"데이터: {len(df_1h):,} 1H candles")

# ============================================================
# W 패턴 (더블바텀) 찾기 + 다양한 TP 테스트
# ============================================================
print("\n" + "=" * 80)
print("추세 변곡 후 최대 수익 분석")
print("=" * 80)

results = []

for i in range(len(points) - 2):
    if not (points[i]['type'] == 'L' and points[i+1]['type'] == 'H' and points[i+2]['type'] == 'L'):
        continue
    
    L1, H, L2 = points[i]['price'], points[i+1]['price'], points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    # 더블바텀 조건
    if abs(L2 - L1) / L1 > 0.03:
        continue
    
    gap = (H - L2) / L2 * 100
    if gap < 4:  # Gap >= 4% 필터
        continue
    
    # 넥라인 돌파 찾기
    future = df_1h.iloc[L2_idx+1:L2_idx+200]
    breakout_idx = None
    
    for j, (_, row) in enumerate(future.iterrows()):
        if row['close'] > H:
            breakout_idx = L2_idx + 1 + j
            break
    
    if breakout_idx is None:
        continue
    
    entry_price = df_1h.iloc[breakout_idx]['close']
    entry_time = df_1h.iloc[breakout_idx]['datetime']
    sl_price = min(L1, L2)
    
    # 돌파 후 최대 수익 & 다양한 TP 도달 시간 측정
    post_breakout = df_1h.iloc[breakout_idx+1:breakout_idx+500]
    
    max_price = entry_price
    max_pnl = 0
    max_time = None
    
    # TP 레벨별 도달 여부
    tp_levels = [2, 3, 5, 7, 10, 15, 20, 30]
    tp_results = {tp: {'reached': False, 'time': None, 'bars': None} for tp in tp_levels}
    
    sl_hit = False
    sl_time = None
    
    for bar_count, (_, row) in enumerate(post_breakout.iterrows()):
        # SL 체크
        if row['low'] <= sl_price:
            sl_hit = True
            sl_time = row['datetime']
            break
        
        # 최대 수익 갱신
        if row['high'] > max_price:
            max_price = row['high']
            max_pnl = (max_price - entry_price) / entry_price * 100
            max_time = row['datetime']
        
        # TP 도달 체크
        current_pnl = (row['high'] - entry_price) / entry_price * 100
        for tp in tp_levels:
            if not tp_results[tp]['reached'] and current_pnl >= tp:
                tp_results[tp]['reached'] = True
                tp_results[tp]['time'] = row['datetime']
                tp_results[tp]['bars'] = bar_count + 1
    
    results.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'gap': gap,
        'max_pnl': max_pnl,
        'max_time': max_time,
        'sl_hit': sl_hit,
        'sl_time': sl_time,
        **{f'tp{tp}_reached': tp_results[tp]['reached'] for tp in tp_levels},
        **{f'tp{tp}_bars': tp_results[tp]['bars'] for tp in tp_levels},
    })

df_results = pd.DataFrame(results)
print(f"\nGap >= 4% 더블바텀 패턴: {len(df_results)}건")

# ============================================================
# TP 레벨별 도달률 분석
# ============================================================
print("\n" + "=" * 80)
print("TP 레벨별 도달률")
print("=" * 80)

print(f"\n{'TP%':<8} {'도달건수':>10} {'도달률%':>10} {'평균봉수':>12} {'평균시간':>15}")
print("-" * 60)

for tp in [2, 3, 5, 7, 10, 15, 20, 30]:
    reached = df_results[f'tp{tp}_reached'].sum()
    rate = reached / len(df_results) * 100
    avg_bars = df_results[df_results[f'tp{tp}_reached']][f'tp{tp}_bars'].mean()
    avg_hours = avg_bars if pd.notna(avg_bars) else 0
    print(f"{tp}%{'':<5} {reached:>10} {rate:>10.1f} {avg_bars:>12.1f} {avg_hours:>12.1f}시간")

# ============================================================
# 최대 수익 분포
# ============================================================
print("\n" + "=" * 80)
print("최대 수익(MFE) 분포")
print("=" * 80)

# SL 안 맞은 케이스만
no_sl = df_results[~df_results['sl_hit']]
sl_hit = df_results[df_results['sl_hit']]

print(f"\n전체: {len(df_results)}건")
print(f"  - SL 안맞음: {len(no_sl)}건 ({len(no_sl)/len(df_results)*100:.1f}%)")
print(f"  - SL 맞음: {len(sl_hit)}건 ({len(sl_hit)/len(df_results)*100:.1f}%)")

print(f"\n[SL 안맞은 케이스의 최대 수익 분포]")
mfe_bins = [0, 3, 5, 10, 15, 20, 30, 50, np.inf]
mfe_labels = ['0-3%', '3-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50%+']

no_sl = no_sl.copy()
no_sl['mfe_bin'] = pd.cut(no_sl['max_pnl'], bins=mfe_bins, labels=mfe_labels)

for label in mfe_labels:
    count = (no_sl['mfe_bin'] == label).sum()
    pct = count / len(no_sl) * 100 if len(no_sl) > 0 else 0
    print(f"  {label}: {count}건 ({pct:.1f}%)")

# ============================================================
# 월 15% 달성 전략
# ============================================================
print("\n" + "=" * 80)
print("월 15% 달성 전략")
print("=" * 80)

# 월별 신호 수
df_results['month'] = pd.to_datetime(df_results['entry_time']).dt.to_period('M')
monthly_signals = df_results.groupby('month').size()

print(f"\n[월별 신호 수]")
print(f"  평균: {monthly_signals.mean():.1f}건/월")
print(f"  최소: {monthly_signals.min()}건")
print(f"  최대: {monthly_signals.max()}건")

# 시나리오 분석
print(f"\n" + "=" * 80)
print("시나리오별 월 수익률")
print("=" * 80)

scenarios = [
    {'name': 'TP 3% × 5회', 'tp': 3, 'trades': 5},
    {'name': 'TP 5% × 3회', 'tp': 5, 'trades': 3},
    {'name': 'TP 7% × 2회 + 3% × 1회', 'tp': 7, 'trades': 2, 'tp2': 3, 'trades2': 1},
    {'name': 'TP 10% × 2회', 'tp': 10, 'trades': 2},
    {'name': 'TP 15% × 1회', 'tp': 15, 'trades': 1},
]

print(f"\n{'시나리오':<25} {'필요승률':>10} {'실제도달률':>12} {'가능여부':>10}")
print("-" * 60)

for s in scenarios:
    tp = s['tp']
    reached_rate = df_results[f'tp{tp}_reached'].mean() * 100
    
    # 단순화: 필요 승률 계산 (SL -6% 가정)
    # 월 15% = tp% × trades × winrate - 6% × trades × (1-winrate)
    # 복합 시나리오는 단순화
    if 'tp2' in s:
        # TP7% × 2 + TP3% × 1 시나리오
        tp1_rate = df_results[f'tp{s["tp"]}_reached'].mean() * 100
        tp2_rate = df_results[f'tp{s["tp2"]}_reached'].mean() * 100
        possible = "△" if tp1_rate > 50 and tp2_rate > 60 else "✗"
        print(f"{s['name']:<25} {'복합':>10} {tp1_rate:.0f}%/{tp2_rate:.0f}%{'':<3} {possible:>10}")
    else:
        # 단순 시나리오
        possible = "✓" if reached_rate > 60 else ("△" if reached_rate > 40 else "✗")
        print(f"{s['name']:<25} {'-':>10} {reached_rate:>11.1f}% {possible:>10}")

# ============================================================
# 최적 전략 도출
# ============================================================
print("\n" + "=" * 80)
print("최적 전략 도출")
print("=" * 80)

# TP 5%의 성과
tp5_reached = df_results['tp5_reached'].sum()
tp5_rate = tp5_reached / len(df_results) * 100
tp5_avg_bars = df_results[df_results['tp5_reached']]['tp5_bars'].mean()

# TP 7%의 성과
tp7_reached = df_results['tp7_reached'].sum()
tp7_rate = tp7_reached / len(df_results) * 100
tp7_avg_bars = df_results[df_results['tp7_reached']]['tp7_bars'].mean()

# TP 10%의 성과
tp10_reached = df_results['tp10_reached'].sum()
tp10_rate = tp10_reached / len(df_results) * 100
tp10_avg_bars = df_results[df_results['tp10_reached']]['tp10_bars'].mean()

# SL률
sl_rate = df_results['sl_hit'].mean() * 100

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                      월 15% 달성 최적 전략                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  기본 조건: Gap >= 4% 더블바텀 패턴                                        │
│  월 평균 신호: {monthly_signals.mean():.1f}건                                              │
│  SL률: {sl_rate:.1f}% (평균손실 -6%)                                               │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  [TP 레벨별 도달률]                                                        │
│   TP 5%  : {tp5_rate:.1f}% 도달 (평균 {tp5_avg_bars:.0f}시간)                              │
│   TP 7%  : {tp7_rate:.1f}% 도달 (평균 {tp7_avg_bars:.0f}시간)                              │
│   TP 10% : {tp10_rate:.1f}% 도달 (평균 {tp10_avg_bars:.0f}시간)                             │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  [추천 전략: 분할 익절]                                                    │
│                                                                            │
│   1차 익절: +5%  (50% 물량) - 도달률 {tp5_rate:.0f}%                                │
│   2차 익절: +10% (30% 물량) - 도달률 {tp10_rate:.0f}%                               │
│   3차 익절: 추세 끝까지 (20% 물량)                                         │
│   손절: -6% (L값 이탈)                                                     │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  [월 15% 달성 시뮬레이션]                                                  │
│                                                                            │
│   월 3회 진입 가정:                                                        │
│   - 승률 70% (2.1승 0.9패)                                                 │
│   - 승: +5% × 0.5 + +10% × 0.3 + +15% × 0.2 = +8.5%                        │
│   - 패: -6%                                                                │
│   - 기대수익: 2.1 × 8.5% - 0.9 × 6% = 17.85% - 5.4% = +12.45%             │
│                                                                            │
│   월 4회 진입 시: +16.6%                                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 실제 백테스트: 분할익절 전략
# ============================================================
print("\n" + "=" * 80)
print("분할 익절 전략 백테스트")
print("=" * 80)

def backtest_partial_tp(df_1h, entry_idx, entry_price, sl_price, max_bars=500):
    """분할 익절 전략 백테스트"""
    tp1_pct, tp2_pct, tp3_pct = 5, 10, 20
    tp1_ratio, tp2_ratio, tp3_ratio = 0.5, 0.3, 0.2
    
    post = df_1h.iloc[entry_idx+1:entry_idx+max_bars]
    
    total_pnl = 0
    remaining = 1.0
    
    tp1_done = tp2_done = tp3_done = False
    
    for _, row in post.iterrows():
        # SL 체크
        if row['low'] <= sl_price:
            sl_pnl = (sl_price - entry_price) / entry_price * 100
            total_pnl += sl_pnl * remaining
            return total_pnl, 'SL'
        
        current_high_pnl = (row['high'] - entry_price) / entry_price * 100
        
        # TP1 체크
        if not tp1_done and current_high_pnl >= tp1_pct:
            total_pnl += tp1_pct * tp1_ratio
            remaining -= tp1_ratio
            tp1_done = True
        
        # TP2 체크
        if not tp2_done and current_high_pnl >= tp2_pct:
            total_pnl += tp2_pct * tp2_ratio
            remaining -= tp2_ratio
            tp2_done = True
        
        # TP3 체크
        if not tp3_done and current_high_pnl >= tp3_pct:
            total_pnl += tp3_pct * tp3_ratio
            remaining -= tp3_ratio
            tp3_done = True
            return total_pnl, 'FULL_TP'
    
    # 시간초과 - 남은 물량 현재가로 청산
    if len(post) > 0 and remaining > 0:
        last_pnl = (post.iloc[-1]['close'] - entry_price) / entry_price * 100
        total_pnl += last_pnl * remaining
    
    return total_pnl, 'TIMEOUT'

# 백테스트 실행
partial_results = []
for i in range(len(points) - 2):
    if not (points[i]['type'] == 'L' and points[i+1]['type'] == 'H' and points[i+2]['type'] == 'L'):
        continue
    
    L1, H, L2 = points[i]['price'], points[i+1]['price'], points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    if abs(L2 - L1) / L1 > 0.03:
        continue
    
    gap = (H - L2) / L2 * 100
    if gap < 4:
        continue
    
    # 넥라인 돌파 찾기
    future = df_1h.iloc[L2_idx+1:L2_idx+200]
    breakout_idx = None
    
    for j, (_, row) in enumerate(future.iterrows()):
        if row['close'] > H:
            breakout_idx = L2_idx + 1 + j
            break
    
    if breakout_idx is None:
        continue
    
    entry_price = df_1h.iloc[breakout_idx]['close']
    entry_time = df_1h.iloc[breakout_idx]['datetime']
    sl_price = min(L1, L2)
    
    pnl, exit_type = backtest_partial_tp(df_1h, breakout_idx, entry_price, sl_price)
    
    partial_results.append({
        'entry_time': entry_time,
        'gap': gap,
        'pnl': pnl,
        'exit_type': exit_type
    })

df_partial = pd.DataFrame(partial_results)

print(f"\n[분할 익절 전략 결과]")
print(f"  총 거래: {len(df_partial)}건")
print(f"  평균 수익: {df_partial['pnl'].mean():.2f}%")
print(f"  총 수익: {df_partial['pnl'].sum():.1f}%")
print(f"  승률: {(df_partial['pnl'] > 0).mean()*100:.1f}%")

# 월별 수익
df_partial['month'] = pd.to_datetime(df_partial['entry_time']).dt.to_period('M')
monthly_pnl = df_partial.groupby('month')['pnl'].sum()

print(f"\n[월별 수익]")
print(f"  평균: {monthly_pnl.mean():.2f}%")
print(f"  중앙값: {monthly_pnl.median():.2f}%")
print(f"  최소: {monthly_pnl.min():.2f}%")
print(f"  최대: {monthly_pnl.max():.2f}%")
print(f"  15% 이상 달성 월: {(monthly_pnl >= 15).sum()}개월 / {len(monthly_pnl)}개월")

print("\n분석 완료!")
