"""
1시간 BTC 안정형 전략 생성
- 15분 시스템을 1H로 확장
- 더 안정적인 파라미터
- 승률 93%+ 목표
"""

import pandas as pd
import numpy as np
from itertools import product

print("="*60)
print("1시간 BTC 안정형 전략 생성")
print("="*60)

# 1. 기존 15분 데이터 로드
print("\n기존 15분 데이터 로드...")
df_15m = pd.read_csv('btcusdt_15m_raw.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.set_index('datetime')

print(f"15분 데이터: {len(df_15m):,}개")

# 2. 1H 데이터 생성 (리샘플링)
print("\n1H 데이터 생성 중...")

df_1h = df_15m.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"1H 데이터: {len(df_1h):,}개")
print(f"비율 확인: {len(df_15m) / len(df_1h):.1f}배 (4배 예상)")

# 3. MTF 데이터 생성 (4H, 1D, 1W)
print("\nMTF 데이터 생성 중...")

df_4h = df_1h.resample('4H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_1d = df_1h.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

df_1w = df_1h.resample('1W').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"4H 데이터: {len(df_4h):,}개 ({len(df_1h) / len(df_4h):.1f}배)")
print(f"1D 데이터: {len(df_1d):,}개 ({len(df_1h) / len(df_1d):.1f}배)")
print(f"1W 데이터: {len(df_1w):,}개 ({len(df_1h) / len(df_1w):.1f}배)")

# 4. 저장
print("\n데이터 저장 중...")

df_1h_reset = df_1h.reset_index()
df_4h_reset = df_4h.reset_index()
df_1d_reset = df_1d.reset_index()
df_1w_reset = df_1w.reset_index()

df_1h_reset.to_csv('btcusdt_1h_base.csv', index=False)
df_4h_reset.to_csv('btcusdt_4h_mtf.csv', index=False)
df_1d_reset.to_csv('btcusdt_1d_mtf.csv', index=False)
df_1w_reset.to_csv('btcusdt_1w_mtf.csv', index=False)

print("\n저장 완료:")
print("  - btcusdt_1h_base.csv (기준 타임프레임)")
print("  - btcusdt_4h_mtf.csv")
print("  - btcusdt_1d_mtf.csv")
print("  - btcusdt_1w_mtf.csv")

# 5. 기본 통계
print("\n" + "="*60)
print("1H 데이터 통계")
print("="*60)

print(f"\n기간: {df_1h_reset['datetime'].min()} ~ {df_1h_reset['datetime'].max()}")
print(f"총 캔들: {len(df_1h):,}개")
print(f"일수: {(df_1h_reset['datetime'].max() - df_1h_reset['datetime'].min()).days}일")

# 변동성 분석
df_1h_reset['range_pct'] = (df_1h_reset['high'] - df_1h_reset['low']) / df_1h_reset['close'] * 100
print(f"\n1H 평균 변동폭: {df_1h_reset['range_pct'].mean():.3f}%")
print(f"15분 평균 변동폭 (참고): ~0.3%")
print(f"비율: {df_1h_reset['range_pct'].mean() / 0.3:.1f}배")

print("\n다음 단계:")
print("  1. python label_1h_macd.py - MACD L/H 라벨링")
print("  2. python classify_1h_situations.py - 상황 분류")
print("  3. python optimize_1h_parameters.py - 파라미터 최적화")
print("  4. python backtest_1h_strategy.py - 백테스트")
