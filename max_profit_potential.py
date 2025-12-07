#!/usr/bin/env python3
"""
추세 변곡 후 실제로 얼마나 갈 수 있는가?
TP 3.5%가 아니라 진짜 최대 수익(MFE) 분석
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("추세 변곡 후 최대 수익 잠재력 분석")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

df_1h = df.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

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

# ============================================================
# W패턴 + Gap >= 5% → 최대 수익(MFE) 추적
# ============================================================
print("\n패턴 조건: W (더블바텀), Gap >= 5%")

results = []

for i in range(len(points) - 2):
    if not (points[i]['type'] == 'L' and points[i+1]['type'] == 'H' and points[i+2]['type'] == 'L'):
        continue
    
    L1, H, L2 = points[i]['price'], points[i+1]['price'], points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    if abs(L2 - L1) / L1 > 0.03:
        continue
    
    gap = (H - L2) / L2 * 100
    if gap < 5:
        continue
    
    # 넥라인 돌파
    future = df_1h.iloc[L2_idx+1:L2_idx+100]
    breakout_idx = None
    for j, (_, row) in enumerate(future.iterrows()):
        if row['close'] > H:
            breakout_idx = L2_idx + 1 + j
            break
    
    if breakout_idx is None:
        continue
    
    entry = df_1h.iloc[breakout_idx]['close']
    entry_time = df_1h.iloc[breakout_idx]['datetime']
    sl_price = min(L1, L2)
    
    # 이후 500봉 동안 추적
    post = df_1h.iloc[breakout_idx+1:breakout_idx+500]
    
    # MFE (Maximum Favorable Excursion) - 최대 수익
    max_price = entry
    max_pnl = 0
    max_bars = 0
    
    # MAE (Maximum Adverse Excursion) - 최대 손실 (SL 전)
    min_price = entry
    min_pnl = 0
    
    sl_hit = False
    sl_bar = 0
    
    # 다음 H 포인트 (저항선) 찾기
    next_H = None
    for p in points[i+3:]:
        if p['type'] == 'H' and p['idx'] > breakout_idx:
            next_H = p['price']
            break
    
    for bar, (_, row) in enumerate(post.iterrows()):
        # SL 체크
        if row['low'] <= sl_price:
            sl_hit = True
            sl_bar = bar
            break
        
        # MFE 갱신
        if row['high'] > max_price:
            max_price = row['high']
            max_pnl = (max_price - entry) / entry * 100
            max_bars = bar
        
        # MAE 갱신
        if row['low'] < min_price:
            min_price = row['low']
            min_pnl = (min_price - entry) / entry * 100
    
    results.append({
        'time': entry_time,
        'entry': entry,
        'sl': sl_price,
        'gap': gap,
        'next_H': next_H,
        'mfe': max_pnl,
        'mfe_bars': max_bars,
        'mae': min_pnl,
        'sl_hit': sl_hit,
        'sl_bar': sl_bar
    })

df_r = pd.DataFrame(results)
print(f"\n총 신호: {len(df_r)}건")
print(f"SL 발동: {df_r['sl_hit'].sum()}건 ({df_r['sl_hit'].mean()*100:.1f}%)")

# ============================================================
# MFE (최대 수익) 분석 - SL 안맞은 케이스
# ============================================================
print("\n" + "=" * 80)
print("SL 안맞은 케이스의 최대 수익(MFE)")
print("=" * 80)

no_sl = df_r[~df_r['sl_hit']]
print(f"\nSL 안맞은 케이스: {len(no_sl)}건 ({len(no_sl)/len(df_r)*100:.1f}%)")

print(f"\n[MFE 통계]")
print(f"  평균: {no_sl['mfe'].mean():.1f}%")
print(f"  중앙값: {no_sl['mfe'].median():.1f}%")
print(f"  최소: {no_sl['mfe'].min():.1f}%")
print(f"  최대: {no_sl['mfe'].max():.1f}%")

print(f"\n[MFE 분포]")
mfe_bins = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
mfe_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%', '100%+']
no_sl = no_sl.copy()
no_sl['mfe_bin'] = pd.cut(no_sl['mfe'], bins=mfe_bins, labels=mfe_labels)

for label in mfe_labels:
    count = (no_sl['mfe_bin'] == label).sum()
    pct = count / len(no_sl) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<10}: {count:>3}건 ({pct:>5.1f}%) {bar}")

# ============================================================
# SL 맞은 케이스도 MFE 분석
# ============================================================
print("\n" + "=" * 80)
print("SL 맞은 케이스의 MFE (손절 전 최대 수익)")
print("=" * 80)

sl_hit = df_r[df_r['sl_hit']]
print(f"\nSL 맞은 케이스: {len(sl_hit)}건")

print(f"\n[SL 전 MFE 통계]")
print(f"  평균: {sl_hit['mfe'].mean():.1f}%")
print(f"  중앙값: {sl_hit['mfe'].median():.1f}%")
print(f"  MFE >= 3.5% 후 SL: {(sl_hit['mfe'] >= 3.5).sum()}건 ({(sl_hit['mfe'] >= 3.5).mean()*100:.1f}%)")
print(f"  MFE >= 5% 후 SL: {(sl_hit['mfe'] >= 5).sum()}건 ({(sl_hit['mfe'] >= 5).mean()*100:.1f}%)")
print(f"  MFE >= 10% 후 SL: {(sl_hit['mfe'] >= 10).sum()}건 ({(sl_hit['mfe'] >= 10).mean()*100:.1f}%)")

# ============================================================
# 핵심 인사이트
# ============================================================
print("\n" + "=" * 80)
print("핵심 인사이트")
print("=" * 80)

# 전체 MFE
all_mfe_avg = df_r['mfe'].mean()
all_mfe_med = df_r['mfe'].median()

# TP 3.5%로 익절 vs 더 큰 TP
tp35_potential = len(df_r[df_r['mfe'] >= 3.5]) / len(df_r) * 100
tp10_potential = len(df_r[df_r['mfe'] >= 10]) / len(df_r) * 100
tp20_potential = len(df_r[df_r['mfe'] >= 20]) / len(df_r) * 100
tp30_potential = len(df_r[df_r['mfe'] >= 30]) / len(df_r) * 100

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                       진짜 수익 잠재력                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [전체 MFE (최대 도달 수익)]                                               │
│  - 평균: {all_mfe_avg:.1f}%                                                        │
│  - 중앙값: {all_mfe_med:.1f}%                                                      │
│                                                                            │
│  [TP 도달 가능성]                                                          │
│  - 3.5% 도달: {tp35_potential:.0f}% ← 현재 TP (너무 보수적)                        │
│  - 10% 도달: {tp10_potential:.0f}%                                                  │
│  - 20% 도달: {tp20_potential:.0f}%                                                  │
│  - 30% 도달: {tp30_potential:.0f}%                                                  │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  [SL 맞은 케이스 분석]                                                     │
│  - SL률: {df_r['sl_hit'].mean()*100:.0f}%                                                │
│  - SL 전 평균 MFE: {sl_hit['mfe'].mean():.1f}%                                        │
│  - SL 전 3.5%+ 도달: {(sl_hit['mfe'] >= 3.5).mean()*100:.0f}%                              │
│                                                                            │
│  → 문제: SL 맞기 전에 이미 수익권이었던 경우 많음!                         │
│  → 해결: 트레일링 스탑 or 분할 익절                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 개선 전략 시뮬레이션
# ============================================================
print("\n" + "=" * 80)
print("개선 전략 시뮬레이션")
print("=" * 80)

# 전략 1: 고정 TP (기존)
def strategy_fixed_tp(row, tp_pct=3.5):
    if row['mfe'] >= tp_pct:
        return tp_pct
    elif row['sl_hit']:
        return -6  # SL
    else:
        return row['mfe']  # 시간초과

# 전략 2: 큰 TP
def strategy_big_tp(row, tp_pct=10):
    if row['mfe'] >= tp_pct:
        return tp_pct
    elif row['sl_hit']:
        return -6
    else:
        return row['mfe']

# 전략 3: 분할 익절
def strategy_partial(row):
    pnl = 0
    if row['mfe'] >= 5:
        pnl += 5 * 0.5  # 50% 물량 @ 5%
    if row['mfe'] >= 10:
        pnl += 10 * 0.3  # 30% 물량 @ 10%
    if row['mfe'] >= 20:
        pnl += 20 * 0.2  # 20% 물량 @ 20%
    
    if pnl > 0:
        remaining = 1.0
        if row['mfe'] >= 5: remaining -= 0.5
        if row['mfe'] >= 10: remaining -= 0.3
        if row['mfe'] >= 20: remaining -= 0.2
        
        if remaining > 0:
            if row['sl_hit']:
                pnl += -6 * remaining
            else:
                pnl += row['mfe'] * 0.5 * remaining  # 절반 수익으로 마감 가정
        return pnl
    else:
        if row['sl_hit']:
            return -6
        return row['mfe'] * 0.5

# 전략 4: 트레일링 (MFE의 60% 확보)
def strategy_trailing(row, capture_rate=0.6):
    if row['sl_hit']:
        # SL 전에 얼마나 갔나
        if row['mfe'] >= 3:
            return row['mfe'] * capture_rate  # 트레일링으로 일부 확보
        return -6
    else:
        return row['mfe'] * capture_rate

df_r['pnl_fixed35'] = df_r.apply(lambda r: strategy_fixed_tp(r, 3.5), axis=1)
df_r['pnl_fixed10'] = df_r.apply(lambda r: strategy_fixed_tp(r, 10), axis=1)
df_r['pnl_partial'] = df_r.apply(strategy_partial, axis=1)
df_r['pnl_trailing'] = df_r.apply(strategy_trailing, axis=1)

print(f"\n{'전략':<25} {'평균PnL':>10} {'총PnL':>12} {'승률':>10}")
print("-" * 60)
print(f"{'고정 TP 3.5%':<25} {df_r['pnl_fixed35'].mean():>10.2f} {df_r['pnl_fixed35'].sum():>12.1f} {(df_r['pnl_fixed35']>0).mean()*100:>9.1f}%")
print(f"{'고정 TP 10%':<25} {df_r['pnl_fixed10'].mean():>10.2f} {df_r['pnl_fixed10'].sum():>12.1f} {(df_r['pnl_fixed10']>0).mean()*100:>9.1f}%")
print(f"{'분할익절 (5/10/20%)':<25} {df_r['pnl_partial'].mean():>10.2f} {df_r['pnl_partial'].sum():>12.1f} {(df_r['pnl_partial']>0).mean()*100:>9.1f}%")
print(f"{'트레일링 (60% 확보)':<25} {df_r['pnl_trailing'].mean():>10.2f} {df_r['pnl_trailing'].sum():>12.1f} {(df_r['pnl_trailing']>0).mean()*100:>9.1f}%")

# 월별 수익
df_r['month'] = pd.to_datetime(df_r['time']).dt.to_period('M')

print(f"\n[월별 평균 수익]")
for col in ['pnl_fixed35', 'pnl_fixed10', 'pnl_partial', 'pnl_trailing']:
    monthly = df_r.groupby('month')[col].sum().mean()
    print(f"  {col}: {monthly:.2f}%/월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

best_monthly = df_r.groupby('month')['pnl_trailing'].sum().mean()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         솔직한 분석 결과                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  문제점:                                                                   │
│  - 기존 TP 3.5% → 너무 작음 (MFE 평균 {all_mfe_avg:.0f}%인데 3.5%만 먹음)           │
│  - SL -6% 고정 → 수익 다 날림                                              │
│  - SL 전에 이미 {(sl_hit['mfe'] >= 3.5).mean()*100:.0f}%가 3.5%+ 수익권 도달                      │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  해결책: 트레일링 스탑 or 분할 익절                                        │
│                                                                            │
│  트레일링 스탑 (MFE의 60% 확보):                                           │
│  - 평균 수익: {df_r['pnl_trailing'].mean():.2f}%/거래                                   │
│  - 월 평균: {best_monthly:.2f}%                                                    │
│  - 3x 레버리지: {best_monthly*3:.1f}%/월                                         │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  핵심:                                                                     │
│  "추세 변곡 잡았으면, 끝까지 끌고 가라"                                    │
│  - 3.5%에서 익절하지 말고                                                  │
│  - 트레일링으로 추세가 끝날 때까지 보유                                    │
│  - MFE 평균 {all_mfe_avg:.0f}%의 60% = {all_mfe_avg*0.6:.0f}% 확보 가능                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")
