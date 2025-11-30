"""
Phase 1: L값 정의 (1H MACD Cross 기준)
======================================================================
정의: 1H MACD Dead Cross와 다음 Golden Cross 사이의 최저가

기존과 차이점:
- 기존: 15min MACD Hist 0교차 (7,257개)
- 신규: 1H MACD Dead/Golden Cross 사이 최저가 (더 큰 사이클)

제약: 없음 (MACD < 0 필터 제거)
출력: l_labels_1h_cross.csv
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("Phase 1: L값 정의 (1H MACD Cross 기준)")
print("=" * 70)
print()

# 15분 데이터 로드
df_15m = pd.read_csv('output_phase1_labeled.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

# 1시간 데이터 로드 (MTF에서)
try:
    df_1h = pd.read_csv('btcusdt_1h_raw.csv')
    df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    print("1H 데이터 로드 완료")
except:
    print("⚠️  1H 데이터 없음 - 15분 데이터에서 생성")
    # 15분 → 1시간 리샘플
    df_1h = df_15m.set_index('datetime').resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

# 최근 5년
cutoff = df_15m['datetime'].max() - timedelta(days=1825)
df_15m = df_15m[df_15m['datetime'] >= cutoff].reset_index(drop=True)
df_1h = df_1h[df_1h['datetime'] >= cutoff].reset_index(drop=True)

print(f"15분 데이터: {len(df_15m):,}개")
print(f"1시간 데이터: {len(df_1h):,}개\n")

# ═══════════════════════════════════════════════════════════════════
# 1H MACD 계산
# ═══════════════════════════════════════════════════════════════════

print("1H MACD 계산 중...")

# EMA
ema12 = df_1h['close'].ewm(span=12, adjust=False).mean()
ema26 = df_1h['close'].ewm(span=26, adjust=False).mean()

df_1h['macd'] = ema12 - ema26
df_1h['macd_signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['macd_hist'] = df_1h['macd'] - df_1h['macd_signal']

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# Dead Cross / Golden Cross 찾기
# ═══════════════════════════════════════════════════════════════════

print("1H MACD Cross 찾기...")

dead_crosses = []  # MACD가 Signal 아래로
golden_crosses = []  # MACD가 Signal 위로

for i in range(1, len(df_1h)):
    prev = df_1h.iloc[i-1]
    curr = df_1h.iloc[i]

    # Dead Cross: MACD가 Signal 아래로 교차
    if prev['macd'] >= prev['macd_signal'] and curr['macd'] < curr['macd_signal']:
        dead_crosses.append({
            'idx': i,
            'datetime': curr['datetime'],
            'price': curr['close']
        })

    # Golden Cross: MACD가 Signal 위로 교차
    if prev['macd'] <= prev['macd_signal'] and curr['macd'] > curr['macd_signal']:
        golden_crosses.append({
            'idx': i,
            'datetime': curr['datetime'],
            'price': curr['close']
        })

print(f"  Dead Cross: {len(dead_crosses)}개")
print(f"  Golden Cross: {len(golden_crosses)}개\n")

# ═══════════════════════════════════════════════════════════════════
# L값 찾기: Dead Cross ~ Golden Cross 사이 최저가
# ═══════════════════════════════════════════════════════════════════

print("L값 추출 중...")

l_values = []

for i, dead in enumerate(dead_crosses):
    dead_time = dead['datetime']
    dead_idx = dead['idx']

    # 다음 Golden Cross 찾기
    next_golden = None
    for golden in golden_crosses:
        if golden['datetime'] > dead_time:
            next_golden = golden
            break

    if next_golden is None:
        continue

    golden_time = next_golden['datetime']

    # Dead ~ Golden 사이 1H 데이터
    period_1h = df_1h[
        (df_1h['datetime'] >= dead_time) &
        (df_1h['datetime'] <= golden_time)
    ]

    if len(period_1h) == 0:
        continue

    # 최저가 찾기 (1H 기준)
    lowest_row = period_1h.loc[period_1h['low'].idxmin()]

    # 15분 데이터에서 정확한 L값 찾기
    l_time_1h = lowest_row['datetime']

    # 해당 1H 구간의 15분 캔들들
    period_15m = df_15m[
        (df_15m['datetime'] >= l_time_1h) &
        (df_15m['datetime'] < l_time_1h + timedelta(hours=1))
    ]

    if len(period_15m) == 0:
        continue

    # 15분 기준 최저가
    l_row_15m = period_15m.loc[period_15m['low'].idxmin()]

    l_values.append({
        'l_idx': l_row_15m.name,
        'datetime': l_row_15m['datetime'],
        'price': l_row_15m['low'],
        'dead_cross_time': dead_time,
        'golden_cross_time': golden_time,
        'cycle_duration_hours': (golden_time - dead_time).total_seconds() / 3600
    })

print(f"  L값 총 {len(l_values)}개 발견\n")

# ═══════════════════════════════════════════════════════════════════
# 저장
# ═══════════════════════════════════════════════════════════════════

df_l = pd.DataFrame(l_values)
df_l.to_csv('l_labels_1h_cross.csv', index=False)

print("=" * 70)
print("결과 요약")
print("=" * 70)
print()
print(f"L값 총 개수: {len(l_values)}개")
print(f"평균 사이클 길이: {df_l['cycle_duration_hours'].mean():.1f}시간")
print(f"최소 사이클: {df_l['cycle_duration_hours'].min():.1f}시간")
print(f"최대 사이클: {df_l['cycle_duration_hours'].max():.1f}시간")
print()

# 기존 15min L값과 비교
print("기존 15min L값과 비교:")
print(f"  15min MACD 0교차 L값: 7,257개 (추정)")
print(f"  1H Cross 기반 L값: {len(l_values)}개")
print(f"  → {7257/len(l_values) if len(l_values) > 0 else 0:.1f}배 차이")
print()

print("저장: l_labels_1h_cross.csv")
print()
print("✅ Phase 1 완료")
