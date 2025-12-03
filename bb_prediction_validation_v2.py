#!/usr/bin/env python3
"""
볼린저밴드 예측 검증 v2 - 미래 데이터 없이 (정확한 검증)

핵심: "돌파 방향"을 예측하는 것 vs "20시간 후 결과" 예측
- 이전 분석: 수축 중 위치 → 돌파 방향 (이것은 맞음)
- 이번 검증: 수축 중 위치 → 돌파 방향 → 그 방향으로 수익 가능성
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 예측 검증 v2 - 미래 데이터 없이")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"\n15M 데이터: {len(df):,}개")

# BB 계산
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

df = df.dropna().reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값: {width_threshold:.3f}%")

# ============================================================
# 핵심 검증: "돌파 캔들"의 방향 예측
# ============================================================
print("\n" + "=" * 80)
print("검증 1: 수축 중 위치 → 돌파 캔들 방향 예측")
print("=" * 80)

results = []

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
        if squeeze_duration < 4:
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
        
        # ★★★ 돌파 캔들의 방향 (즉각적인 결과) ★★★
        breakout_candle = df.iloc[i]
        prev_close = df.iloc[i-1]['close']
        
        # 돌파 방향 판정 (더 정확한 방식)
        # 1. 상단 밴드 돌파
        if breakout_candle['close'] > df.iloc[i-1]['bb_upper']:
            immediate_direction = 'UP'
        # 2. 하단 밴드 돌파
        elif breakout_candle['close'] < df.iloc[i-1]['bb_lower']:
            immediate_direction = 'DOWN'
        # 3. 중간 밴드 내에서 방향
        else:
            if breakout_candle['close'] > prev_close:
                immediate_direction = 'UP'
            else:
                immediate_direction = 'DOWN'
        
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
    ('전체 (돌파 방향대로 진입)', df_results),
    ('UPPER + UP돌파', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'UP')]),
    ('UPPER + DOWN돌파', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('LOWER + UP돌파', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'UP')]),
    ('LOWER + DOWN돌파', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('예측 성공: UPPER→UP', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'UP')]),
    ('예측 성공: LOWER→DOWN', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('예측 실패: UPPER→DOWN', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'DOWN')]),
    ('예측 실패: LOWER→UP', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'UP')]),
]

for name, subset in strategies:
    if len(subset) >= 5:
        mfe = subset['mfe'].mean()
        pnl = subset['pnl_20h'].mean()
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>40} {len(subset):>8} {mfe:>10.2f}% {pnl:>10.2f}% {wr:>10.1f}%")

# ============================================================
# 핵심 질문 3: 실제 트레이딩 전략
# ============================================================
print("\n" + "=" * 80)
print("★★★ 실제 트레이딩 전략 제안 ★★★")
print("=" * 80)

# 전략 1: 위치 기반 예측 + 돌파 방향 일치
print("\n[전략 1: 위치-방향 일치 시 진입]")
position_match = df_results[
    ((df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'UP')) |
    ((df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'DOWN'))
]
if len(position_match) >= 5:
    print(f"  건수: {len(position_match)}")
    print(f"  평균 MFE: {position_match['mfe'].mean():.2f}%")
    print(f"  평균 PnL(20H): {position_match['pnl_20h'].mean():.2f}%")
    print(f"  승률: {(position_match['pnl_20h'] > 0).mean() * 100:.1f}%")

# 전략 2: 위치-방향 불일치 (역추세)
print("\n[전략 2: 위치-방향 불일치 시 진입 (역추세)]")
position_mismatch = df_results[
    ((df_results['sq_position'] == 'UPPER') & (df_results['immediate_dir'] == 'DOWN')) |
    ((df_results['sq_position'] == 'LOWER') & (df_results['immediate_dir'] == 'UP'))
]
if len(position_mismatch) >= 5:
    print(f"  건수: {len(position_mismatch)}")
    print(f"  평균 MFE: {position_mismatch['mfe'].mean():.2f}%")
    print(f"  평균 PnL(20H): {position_mismatch['pnl_20h'].mean():.2f}%")
    print(f"  승률: {(position_mismatch['pnl_20h'] > 0).mean() * 100:.1f}%")

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

print(f"""
■ 예측 정확도 (미래 데이터 없이):
  - UPPER 위치 → UP 돌파: {upper_up_rate:.1f}% (이전 분석: 80%)
  - LOWER 위치 → DOWN 돌파: {lower_down_rate:.1f}% (이전 분석: 76%)

■ 차이 분석:
  - 이전 분석은 "돌파 후 결과"를 측정 (lookback bias 가능성)
  - 이번 검증은 "돌파 시점"의 방향만 측정 (순수 예측)

■ 결론:
  - 수축 중 위치가 돌파 방향을 {max(upper_up_rate, lower_down_rate):.0f}% 예측
  - 하지만 돌파 방향이 결정된 후에는 그 방향으로 진입 시 수익 가능
""")

# 수익성 분석
if len(position_match) > 0 and len(position_mismatch) > 0:
    print(f"""
■ 수익성 분석:
  - 예측 일치 (UPPER→UP, LOWER→DOWN) 진입:
    MFE {position_match['mfe'].mean():.2f}%, PnL {position_match['pnl_20h'].mean():.2f}%, 승률 {(position_match['pnl_20h'] > 0).mean() * 100:.1f}%
    
  - 예측 불일치 (역추세) 진입:
    MFE {position_mismatch['mfe'].mean():.2f}%, PnL {position_mismatch['pnl_20h'].mean():.2f}%, 승률 {(position_mismatch['pnl_20h'] > 0).mean() * 100:.1f}%

■ 실제 트레이딩 적용:
  - 방법 1: "위치 + 방향 일치" 확인 후 진입 → 기대 PnL {position_match['pnl_20h'].mean():.2f}%
  - 방법 2: 돌파 방향만 확인하고 진입 → 기대 PnL {df_results['pnl_20h'].mean():.2f}%
""")

# 저장
df_results.to_csv('bb_prediction_validation_v2.csv', index=False)
print(f"\n저장: bb_prediction_validation_v2.csv")
