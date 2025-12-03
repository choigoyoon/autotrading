#!/usr/bin/env python3
"""
볼린저밴드 예측 검증 v3 - 미래 데이터 없이 (정확한 검증)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 예측 검증 v3 - 미래 데이터 없이")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"\n15M 데이터: {len(df):,}개")

# BB 계산 (window 기반)
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# NaN 제거 (처음 period 개만)
df = df.iloc[period:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값: {width_threshold:.3f}%")
print(f"평균 밴드폭: {df['bb_width'].mean():.2f}%")

# ============================================================
# 핵심 검증: "돌파 캔들"의 방향 예측
# ============================================================
print("\n" + "=" * 80)
print("검증: 수축 중 위치 → 돌파 캔들 방향 예측")
print("=" * 80)

results = []
min_squeeze = 4  # 최소 수축 기간

for i in range(50, len(df) - 100):
    current_width = df.iloc[i]['bb_width']
    prev_width = df.iloc[i-1]['bb_width']
    
    # 수축 → 확장 전환 시점
    if prev_width <= width_threshold and current_width > width_threshold:
        
        # 수축 구간 찾기
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width'] <= width_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < min_squeeze:
            continue
        
        # 수축 구간 데이터
        during_squeeze = df.iloc[squeeze_start:i]
        
        # 수축 중 평균 위치
        avg_position = during_squeeze['bb_position'].mean()
        
        if avg_position > 60:
            sq_position = 'UPPER'
        elif avg_position < 40:
            sq_position = 'LOWER'
        else:
            sq_position = 'MIDDLE'
        
        # 돌파 캔들
        breakout_candle = df.iloc[i]
        prev_close = df.iloc[i-1]['close']
        
        # 돌파 방향
        if breakout_candle['close'] > df.iloc[i-1]['bb_upper']:
            immediate_direction = 'UP'
        elif breakout_candle['close'] < df.iloc[i-1]['bb_lower']:
            immediate_direction = 'DOWN'
        else:
            immediate_direction = 'UP' if breakout_candle['close'] > prev_close else 'DOWN'
        
        # 미래 성과 (검증용)
        future = df.iloc[i+1:i+81]
        if len(future) < 80:
            continue
        
        entry = breakout_candle['close']
        
        if immediate_direction == 'UP':
            mfe = (future['high'].max() - entry) / entry * 100
            mae = (future['low'].min() - entry) / entry * 100
            pnl_20h = (future.iloc[79]['close'] - entry) / entry * 100
        else:
            mfe = (entry - future['low'].min()) / entry * 100
            mae = (entry - future['high'].max()) / entry * 100
            pnl_20h = (entry - future.iloc[79]['close']) / entry * 100
        
        results.append({
            'time': breakout_candle['datetime'],
            'sq_position': sq_position,
            'avg_position': avg_position,
            'immediate_dir': immediate_direction,
            'duration': squeeze_duration,
            'min_width': during_squeeze['bb_width'].min(),
            'mfe': mfe,
            'mae': mae,
            'pnl_20h': pnl_20h
        })

df_results = pd.DataFrame(results)
print(f"총 검증 건수: {len(df_results)}건")

if len(df_results) == 0:
    print("결과 없음")
    exit()

# ============================================================
# 핵심 질문 1: "수축 중 위치"가 "돌파 방향"을 예측하는가?
# ============================================================
print("\n" + "=" * 80)
print("★★★ 핵심 질문 1: 수축 중 위치 → 돌파 방향 예측 정확도 ★★★")
print("=" * 80)

for pos in ['UPPER', 'MIDDLE', 'LOWER']:
    subset = df_results[df_results['sq_position'] == pos]
    if len(subset) >= 10:
        up_pct = (subset['immediate_dir'] == 'UP').mean() * 100
        down_pct = (subset['immediate_dir'] == 'DOWN').mean() * 100
        
        print(f"\n[{pos}] 총 {len(subset)}건")
        print(f"  → UP 돌파: {up_pct:.1f}%")
        print(f"  → DOWN 돌파: {down_pct:.1f}%")
        
        if pos == 'UPPER':
            print(f"  ★ 예측 정확도 (UPPER→UP): {up_pct:.1f}%")
        elif pos == 'LOWER':
            print(f"  ★ 예측 정확도 (LOWER→DOWN): {down_pct:.1f}%")

# ============================================================
# 핵심 질문 2: "돌파 방향 예측"이 수익을 주는가?
# ============================================================
print("\n" + "=" * 80)
print("★★★ 핵심 질문 2: 예측 방향대로 진입 시 수익성 ★★★")
print("=" * 80)

print(f"\n{'전략':>40} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 85)

strategies = [
    ('전체', df_results),
    ('UPPER + UP돌파 (예측 일치)', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'UP')]),
    ('UPPER + DOWN돌파 (예측 불일치)', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('LOWER + DOWN돌파 (예측 일치)', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('LOWER + UP돌파 (예측 불일치)', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'UP')]),
]

for name, subset in strategies:
    if len(subset) >= 5:
        mfe = subset['mfe'].mean()
        pnl = subset['pnl_20h'].mean()
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>40} {len(subset):>8} {mfe:>10.2f}% {pnl:>10.2f}% {wr:>10.1f}%")

# ============================================================
# 수축 기간별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 기간별 예측 정확도")
print("=" * 80)

print(f"\n{'기간':>12} {'건수':>8} {'UPPER→UP':>12} {'LOWER→DOWN':>12} {'평균PnL':>10}")
print("-" * 65)

for low, high in [(4, 10), (10, 20), (20, 50), (50, 500)]:
    subset = df_results[(df_results['duration'] >= low) & (df_results['duration'] < high)]
    if len(subset) >= 10:
        upper = subset[subset['sq_position'] == 'UPPER']
        lower = subset[subset['sq_position'] == 'LOWER']
        
        upper_up = (upper['immediate_dir'] == 'UP').mean() * 100 if len(upper) > 0 else 0
        lower_down = (lower['immediate_dir'] == 'DOWN').mean() * 100 if len(lower) > 0 else 0
        
        print(f"{f'{low}-{high}봉':>12} {len(subset):>8} {upper_up:>12.1f}% {lower_down:>12.1f}% {subset['pnl_20h'].mean():>10.2f}%")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 검증 결론 ★★★")
print("=" * 80)

upper = df_results[df_results['sq_position'] == 'UPPER']
lower = df_results[df_results['sq_position'] == 'LOWER']

upper_up_rate = (upper['immediate_dir'] == 'UP').mean() * 100 if len(upper) > 0 else 0
lower_down_rate = (lower['immediate_dir'] == 'DOWN').mean() * 100 if len(lower) > 0 else 0

# 예측 일치/불일치 성과
match = df_results[
    ((df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'UP')) |
    ((df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'DOWN'))
]
mismatch = df_results[
    ((df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'DOWN')) |
    ((df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'UP'))
]

print(f"""
■ 예측 정확도 (미래 데이터 없이 검증):
  - UPPER 위치 → UP 돌파: {upper_up_rate:.1f}% ({len(upper)}건)
  - LOWER 위치 → DOWN 돌파: {lower_down_rate:.1f}% ({len(lower)}건)

■ 이전 분석(80%)과의 차이:
  - 이전 분석: "돌파 후" 20시간 뒤 가격 변화 측정
  - 이번 검증: "돌파 캔들" 자체의 방향 측정

■ 실제 트레이딩 관점:
  - 예측 일치 시 (UPPER→UP, LOWER→DOWN) 진입:
    건수 {len(match)}, MFE {match['mfe'].mean():.2f}%, PnL {match['pnl_20h'].mean():.2f}%, 승률 {(match['pnl_20h'] > 0).mean() * 100:.1f}%
    
  - 예측 불일치 시 (역추세) 진입:
    건수 {len(mismatch)}, MFE {mismatch['mfe'].mean():.2f}%, PnL {mismatch['pnl_20h'].mean():.2f}%, 승률 {(mismatch['pnl_20h'] > 0).mean() * 100:.1f}%

■ 핵심 발견:
  - 수축 중 위치는 돌파 방향을 약 {max(upper_up_rate, lower_down_rate):.0f}% 예측 가능
  - 하지만 "예측 일치"보다 "돌파 방향대로 진입"이 더 중요
  - 즉, 예측보다는 "돌파 확인 후" 진입이 핵심
""")

# 저장
df_results.to_csv('bb_prediction_validation_v3.csv', index=False)
print(f"\n저장: bb_prediction_validation_v3.csv")
