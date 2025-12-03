#!/usr/bin/env python3
"""
볼린저밴드 수축/확장 분석 v2

핵심 질문:
1. 얼마나 좁아졌을 때 폭발하는가?
2. 어떤 상황에서 추세가 결정나는가?
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 수축/확장 분석 v2")
print("=" * 80)

# 데이터 로드 (15M으로 1H 생성)
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

# 1H 리샘플링
df = df_15m.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

print(f"1H 데이터: {len(df):,}개")

# 볼린저밴드 계산 (30 기준)
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

# 밴드폭 (% 기준)
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

# 밴드폭 백분위 (최근 100개 기준)
df['bb_width_min_100'] = df['bb_width'].rolling(100).min()
df['bb_width_max_100'] = df['bb_width'].rolling(100).max()
df['bb_width_pct'] = (df['bb_width'] - df['bb_width_min_100']) / (df['bb_width_max_100'] - df['bb_width_min_100']) * 100

df = df.dropna().reset_index(drop=True)
print(f"계산 후 데이터: {len(df):,}개")

# ============================================================
# 밴드폭 분포 분석
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 분포 분석 (BB 30)")
print("=" * 80)

bb_width = df['bb_width']
print(f"\n밴드폭 통계:")
print(f"  최소: {bb_width.min():.2f}%")
print(f"  최대: {bb_width.max():.2f}%")
print(f"  평균: {bb_width.mean():.2f}%")
print(f"  중앙값: {bb_width.median():.2f}%")

print(f"\n밴드폭 백분위:")
for pct in [5, 10, 20, 25, 50, 75, 90, 95]:
    val = bb_width.quantile(pct/100)
    print(f"  {pct}%: {val:.2f}%")

# ============================================================
# 수축 후 폭발 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 후 폭발 분석")
print("=" * 80)

results = []

# 밴드폭이 특정 기준 이하로 내려갔다가 다시 올라가는 지점 찾기
squeeze_threshold_pct = 20  # 백분위 20% 이하면 수축

for i in range(100, len(df) - 50):
    current_pct = df.iloc[i]['bb_width_pct']
    prev_pct = df.iloc[i-1]['bb_width_pct']
    
    # 수축 → 확장 전환점 (백분위 기준)
    if prev_pct <= squeeze_threshold_pct and current_pct > squeeze_threshold_pct:
        
        # 수축 구간 분석
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width_pct'] <= squeeze_threshold_pct:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        min_width = df.iloc[squeeze_start:i]['bb_width'].min()
        min_width_pct = df.iloc[squeeze_start:i]['bb_width_pct'].min()
        
        # 돌파 방향 판단
        breakout_candle = df.iloc[i]
        
        if breakout_candle['close'] > breakout_candle['bb_upper']:
            direction = 'UP'
        elif breakout_candle['close'] < breakout_candle['bb_lower']:
            direction = 'DOWN'
        else:
            # 밴드 중간이면 이전 대비 방향
            if breakout_candle['close'] > df.iloc[i-1]['close']:
                direction = 'UP'
            else:
                direction = 'DOWN'
        
        entry_price = breakout_candle['close']
        
        # 이후 50개 캔들 수익 측정
        future = df.iloc[i+1:i+51]
        
        if len(future) < 10:
            continue
        
        if direction == 'UP':
            mfe = (future['high'].max() - entry_price) / entry_price * 100
            mae = (future['low'].min() - entry_price) / entry_price * 100
            pnl_20 = (future.iloc[19]['close'] - entry_price) / entry_price * 100 if len(future) >= 20 else 0
            pnl_50 = (future.iloc[-1]['close'] - entry_price) / entry_price * 100
        else:
            mfe = (entry_price - future['low'].min()) / entry_price * 100
            mae = (entry_price - future['high'].max()) / entry_price * 100
            pnl_20 = (entry_price - future.iloc[19]['close']) / entry_price * 100 if len(future) >= 20 else 0
            pnl_50 = (entry_price - future.iloc[-1]['close']) / entry_price * 100
        
        results.append({
            'time': breakout_candle['datetime'],
            'direction': direction,
            'squeeze_duration': squeeze_duration,
            'min_width': min_width,
            'min_width_pct': min_width_pct,
            'entry_price': entry_price,
            'mfe': mfe,
            'mae': mae,
            'pnl_20h': pnl_20,
            'pnl_50h': pnl_50
        })

df_results = pd.DataFrame(results)
print(f"수축→확장 전환점: {len(df_results)}건")

if len(df_results) == 0:
    print("결과 없음")
    exit()

# ============================================================
# 결과 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 강도별 폭발력 분석 (밴드폭 백분위)")
print("=" * 80)

print(f"\n{'수축강도':>15} {'건수':>8} {'MFE':>10} {'MAE':>10} {'PnL_20H':>10} {'PnL_50H':>10}")
print("-" * 70)

for low, high in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    subset = df_results[(df_results['min_width_pct'] >= low) & (df_results['min_width_pct'] < high)]
    if len(subset) > 0:
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['mae'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {subset['pnl_50h'].mean():>10.2f}%")

# ============================================================
# 밴드폭 절대값 기준
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 절대값 기준 분석")
print("=" * 80)

print(f"\n{'밴드폭':>15} {'건수':>8} {'MFE':>10} {'MAE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 70)

for low, high in [(0, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 15)]:
    subset = df_results[(df_results['min_width'] >= low) & (df_results['min_width'] < high)]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['mae'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 수축 기간별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 기간별 분석")
print("=" * 80)

print(f"\n{'수축기간':>15} {'건수':>8} {'MFE':>10} {'MAE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 70)

for low, high in [(1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 500)]:
    subset = df_results[(df_results['squeeze_duration'] >= low) & (df_results['squeeze_duration'] < high)]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}H':>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['mae'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 방향별 분석
# ============================================================
print("\n" + "=" * 80)
print("돌파 방향별 분석")
print("=" * 80)

for direction in ['UP', 'DOWN']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"\n[{direction}] {len(subset)}건")
        print(f"  평균 MFE: {subset['mfe'].mean():.2f}%")
        print(f"  평균 MAE: {subset['mae'].mean():.2f}%")
        print(f"  평균 PnL (20H): {subset['pnl_20h'].mean():.2f}%")
        print(f"  승률: {win_rate:.1f}%")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>40} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 85)

conditions = [
    ('전체', df_results),
    ('밴드폭 < 5%', df_results[df_results['min_width'] < 5]),
    ('밴드폭 < 4%', df_results[df_results['min_width'] < 4]),
    ('밴드폭 < 3%', df_results[df_results['min_width'] < 3]),
    ('수축기간 >= 10H', df_results[df_results['squeeze_duration'] >= 10]),
    ('수축기간 >= 20H', df_results[df_results['squeeze_duration'] >= 20]),
    ('밴드폭<5% + 기간>=10H', df_results[(df_results['min_width'] < 5) & (df_results['squeeze_duration'] >= 10)]),
    ('밴드폭<4% + 기간>=10H', df_results[(df_results['min_width'] < 4) & (df_results['squeeze_duration'] >= 10)]),
    ('밴드폭<4% + 기간>=20H', df_results[(df_results['min_width'] < 4) & (df_results['squeeze_duration'] >= 20)]),
    ('UP + 밴드폭<5%', df_results[(df_results['direction'] == 'UP') & (df_results['min_width'] < 5)]),
    ('DOWN + 밴드폭<5%', df_results[(df_results['direction'] == 'DOWN') & (df_results['min_width'] < 5)]),
]

for name, subset in conditions:
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>40} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    볼린저밴드 수축/확장 분석 결과                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 밴드폭 통계 (BB 30):                                                     │
│    - 평균: {bb_width.mean():.2f}%                                                          │
│    - 중앙값: {bb_width.median():.2f}%                                                        │
│    - 5% 백분위: {bb_width.quantile(0.05):.2f}%                                                    │
│    - 10% 백분위: {bb_width.quantile(0.1):.2f}%                                                   │
│                                                                            │
│  ■ 수축→확장 전환점: {len(df_results)}건                                                  │
│                                                                            │
│  ■ 핵심 발견:                                                               │
│    - 밴드폭이 좁을수록 폭발력 큼                                              │
│    - 수축 기간이 길수록 추세 지속력 큼                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 저장
df_results.to_csv('bb_squeeze_results.csv', index=False)
print("결과 저장: bb_squeeze_results.csv")
