#!/usr/bin/env python3
"""
볼린저밴드 수축/확장 분석

핵심 질문:
1. 얼마나 좁아졌을 때 폭발하는가?
2. 어떤 상황에서 추세가 결정나는가?
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 수축/확장 분석")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"1H 데이터: {len(df):,}개")

# 볼린저밴드 계산 (20, 30 기준)
for period in [20, 30]:
    df[f'bb_mid_{period}'] = df['close'].rolling(period).mean()
    df[f'bb_std_{period}'] = df['close'].rolling(period).std()
    df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + 2 * df[f'bb_std_{period}']
    df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - 2 * df[f'bb_std_{period}']
    
    # 밴드폭 (% 기준)
    df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_mid_{period}'] * 100
    
    # 밴드폭의 이동평균 (수축 판단용)
    df[f'bb_width_ma_{period}'] = df[f'bb_width_{period}'].rolling(20).mean()
    
    # 밴드폭 백분위 (최근 100개 기준)
    df[f'bb_width_pct_{period}'] = df[f'bb_width_{period}'].rolling(100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
    )

df = df.dropna().reset_index(drop=True)
print(f"계산 후 데이터: {len(df):,}개")

# ============================================================
# 밴드폭 분포 분석
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 분포 분석 (BB 30)")
print("=" * 80)

bb_width = df['bb_width_30']
print(f"\n밴드폭 통계:")
print(f"  최소: {bb_width.min():.2f}%")
print(f"  최대: {bb_width.max():.2f}%")
print(f"  평균: {bb_width.mean():.2f}%")
print(f"  중앙값: {bb_width.median():.2f}%")
print(f"  표준편차: {bb_width.std():.2f}%")

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

# 수축 기준: 밴드폭 백분위 20% 이하
squeeze_threshold = 20  # 백분위

df['is_squeeze'] = df['bb_width_pct_30'] <= squeeze_threshold

# 수축 구간 찾기
squeeze_starts = []
in_squeeze = False
squeeze_start_idx = None

for i in range(len(df)):
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start_idx = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_starts.append({
            'start_idx': squeeze_start_idx,
            'end_idx': i,
            'start_time': df.iloc[squeeze_start_idx]['datetime'],
            'end_time': df.iloc[i]['datetime'],
            'duration': i - squeeze_start_idx,
            'min_width': df.iloc[squeeze_start_idx:i]['bb_width_30'].min(),
            'min_width_pct': df.iloc[squeeze_start_idx:i]['bb_width_pct_30'].min(),
            'breakout_price': df.iloc[i]['close'],
            'bb_upper': df.iloc[i]['bb_upper_30'],
            'bb_lower': df.iloc[i]['bb_lower_30']
        })

print(f"\n총 수축 구간: {len(squeeze_starts)}개")

# ============================================================
# 수축 후 방향 및 수익 분석
# ============================================================
print("\n수축 후 방향 및 수익 분석...")

results = []

for sq in squeeze_starts:
    end_idx = sq['end_idx']
    
    if end_idx + 50 >= len(df):
        continue
    
    breakout_candle = df.iloc[end_idx]
    
    # 돌파 방향 판단
    if breakout_candle['close'] > breakout_candle['bb_upper_30']:
        direction = 'UP'
    elif breakout_candle['close'] < breakout_candle['bb_lower_30']:
        direction = 'DOWN'
    else:
        # 중간이면 이전 캔들 대비
        if breakout_candle['close'] > df.iloc[end_idx-1]['close']:
            direction = 'UP'
        else:
            direction = 'DOWN'
    
    entry_price = breakout_candle['close']
    
    # 이후 20, 50개 캔들 수익 측정
    future = df.iloc[end_idx+1:end_idx+51]
    
    if direction == 'UP':
        max_profit = (future['high'].max() - entry_price) / entry_price * 100
        max_loss = (future['low'].min() - entry_price) / entry_price * 100
        final_pnl = (future.iloc[-1]['close'] - entry_price) / entry_price * 100 if len(future) > 0 else 0
    else:
        max_profit = (entry_price - future['low'].min()) / entry_price * 100
        max_loss = (entry_price - future['high'].max()) / entry_price * 100
        final_pnl = (entry_price - future.iloc[-1]['close']) / entry_price * 100 if len(future) > 0 else 0
    
    results.append({
        'time': sq['end_time'],
        'direction': direction,
        'squeeze_duration': sq['duration'],
        'min_width': sq['min_width'],
        'min_width_pct': sq['min_width_pct'],
        'entry_price': entry_price,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_pnl': final_pnl,
        'mfe': max_profit,
        'mae': max_loss
    })

df_results = pd.DataFrame(results)
print(f"분석 완료: {len(df_results)}건")

# ============================================================
# 결과 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 강도별 폭발력 분석")
print("=" * 80)

print(f"\n{'수축강도(백분위)':>20} {'건수':>8} {'평균MFE':>10} {'평균MAE':>10} {'평균PnL':>10}")
print("-" * 65)

for low, high in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    subset = df_results[(df_results['min_width_pct'] >= low) & (df_results['min_width_pct'] < high)]
    if len(subset) > 0:
        avg_mfe = subset['mfe'].mean()
        avg_mae = subset['mae'].mean()
        avg_pnl = subset['final_pnl'].mean()
        print(f"{f'{low}-{high}%':>20} {len(subset):>8} {avg_mfe:>10.2f}% {avg_mae:>10.2f}% {avg_pnl:>10.2f}%")

# ============================================================
# 수축 기간별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 기간별 폭발력 분석")
print("=" * 80)

print(f"\n{'수축기간(시간)':>20} {'건수':>8} {'평균MFE':>10} {'평균MAE':>10} {'평균PnL':>10}")
print("-" * 65)

for low, high in [(1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 500)]:
    subset = df_results[(df_results['squeeze_duration'] >= low) & (df_results['squeeze_duration'] < high)]
    if len(subset) > 0:
        avg_mfe = subset['mfe'].mean()
        avg_mae = subset['mae'].mean()
        avg_pnl = subset['final_pnl'].mean()
        print(f"{f'{low}-{high}H':>20} {len(subset):>8} {avg_mfe:>10.2f}% {avg_mae:>10.2f}% {avg_pnl:>10.2f}%")

# ============================================================
# 방향별 분석
# ============================================================
print("\n" + "=" * 80)
print("돌파 방향별 분석")
print("=" * 80)

for direction in ['UP', 'DOWN']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        print(f"\n[{direction}] {len(subset)}건")
        print(f"  평균 MFE: {subset['mfe'].mean():.2f}%")
        print(f"  평균 MAE: {subset['mae'].mean():.2f}%")
        print(f"  평균 PnL: {subset['final_pnl'].mean():.2f}%")
        print(f"  승률 (PnL>0): {(subset['final_pnl']>0).mean()*100:.1f}%")

# ============================================================
# 밴드폭 절대값 기준 분석
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 절대값 기준 분석")
print("=" * 80)

print(f"\n{'밴드폭':>15} {'건수':>8} {'평균MFE':>10} {'평균MAE':>10} {'평균PnL':>10} {'승률':>10}")
print("-" * 70)

for low, high in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 20)]:
    subset = df_results[(df_results['min_width'] >= low) & (df_results['min_width'] < high)]
    if len(subset) > 5:
        avg_mfe = subset['mfe'].mean()
        avg_mae = subset['mae'].mean()
        avg_pnl = subset['final_pnl'].mean()
        win_rate = (subset['final_pnl'] > 0).mean() * 100
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {avg_mfe:>10.2f}% {avg_mae:>10.2f}% {avg_pnl:>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

# 수축 강도 + 기간 조합
print(f"\n{'조건':>40} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 85)

conditions = [
    ('전체', df_results),
    ('수축 백분위 < 10%', df_results[df_results['min_width_pct'] < 10]),
    ('수축 백분위 < 5%', df_results[df_results['min_width_pct'] < 5]),
    ('수축 기간 >= 10H', df_results[df_results['squeeze_duration'] >= 10]),
    ('수축 기간 >= 20H', df_results[df_results['squeeze_duration'] >= 20]),
    ('백분위<10% + 기간>=10H', df_results[(df_results['min_width_pct'] < 10) & (df_results['squeeze_duration'] >= 10)]),
    ('백분위<10% + 기간>=20H', df_results[(df_results['min_width_pct'] < 10) & (df_results['squeeze_duration'] >= 20)]),
    ('밴드폭 < 4%', df_results[df_results['min_width'] < 4]),
    ('밴드폭 < 3%', df_results[df_results['min_width'] < 3]),
    ('밴드폭<4% + 기간>=10H', df_results[(df_results['min_width'] < 4) & (df_results['squeeze_duration'] >= 10)]),
]

for name, subset in conditions:
    if len(subset) >= 10:
        avg_mfe = subset['mfe'].mean()
        avg_pnl = subset['final_pnl'].mean()
        win_rate = (subset['final_pnl'] > 0).mean() * 100
        print(f"{name:>40} {len(subset):>8} {avg_mfe:>10.2f}% {avg_pnl:>10.2f}% {win_rate:>10.1f}%")

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
│    - 10% 백분위: {bb_width.quantile(0.1):.2f}%                                                   │
│    - 5% 백분위: {bb_width.quantile(0.05):.2f}%                                                    │
│                                                                            │
│  ■ 총 수축 구간: {len(squeeze_starts)}개                                                     │
│  ■ 분석 완료: {len(df_results)}건                                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 저장
df_results.to_csv('bb_squeeze_results.csv', index=False)
print("결과 저장: bb_squeeze_results.csv")
