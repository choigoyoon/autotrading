#!/usr/bin/env python3
"""
L값(Swing Low) 이후 캔들 패턴 분석
- L값 이후 N개 캔들의 가격 움직임
- 캔들 크기, 방향, 연속성
- 지표 변화 추적
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("L값(Swing Low) 이후 캔들 패턴 분석")
print("=" * 60)

# ============================================================
# 데이터 로드 및 지표 계산
# ============================================================

def calculate_all_indicators(df):
    """모든 지표 계산"""
    df = df.copy()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 캔들 특성
    df['body'] = df['close'] - df['open']
    df['body_pct'] = (df['body'] / df['open']) * 100
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']
    df['candle_range_pct'] = (df['candle_range'] / df['open']) * 100
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # Volume
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    return df

print("\n📊 데이터 로드 및 지표 계산 중...")
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = calculate_all_indicators(df)
df = df.dropna().reset_index(drop=True)

print(f"✅ 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"✅ 총 15분봉 개수: {len(df):,}개")

# ============================================================
# Swing Low 탐지
# ============================================================

print("\n🔍 Swing Low 탐지 중...")

order = 10
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
swing_lows = df.iloc[low_indices].copy()

print(f"✅ Swing Low: {len(swing_lows):,}개")

# ============================================================
# L값 이후 캔들 패턴 분석
# ============================================================

print("\n🔍 L값 이후 캔들 패턴 수집 중...")

N_CANDLES = 20  # L값 이후 20개 캔들 추적

patterns = []

for idx, low_point in swing_lows.iterrows():
    low_idx = df[df['datetime'] == low_point['datetime']].index[0]
    
    # L값 이후 N개 캔들 추출
    if low_idx + N_CANDLES >= len(df):
        continue
    
    next_candles = df.iloc[low_idx + 1 : low_idx + 1 + N_CANDLES].copy()
    
    # 패턴 분석
    pattern_data = {
        'low_datetime': low_point['datetime'],
        'low_price': low_point['low'],
        'low_rsi': low_point['rsi'],
        'low_macd_hist': low_point['macd_hist'],
        'low_volume_ratio': low_point['volume_ratio']
    }
    
    # 각 캔들별 정보
    for i in range(min(10, len(next_candles))):  # 처음 10개만 상세 분석
        candle = next_candles.iloc[i]
        pattern_data[f'c{i+1}_body_pct'] = candle['body_pct']
        pattern_data[f'c{i+1}_range_pct'] = candle['candle_range_pct']
        pattern_data[f'c{i+1}_close_change_pct'] = ((candle['close'] - low_point['low']) / low_point['low']) * 100
    
    # 전체 통계 (20개 캔들)
    pattern_data['green_count'] = len(next_candles[next_candles['body'] > 0])
    pattern_data['red_count'] = len(next_candles[next_candles['body'] < 0])
    pattern_data['green_ratio'] = pattern_data['green_count'] / len(next_candles)
    
    pattern_data['avg_body_pct'] = next_candles['body_pct'].mean()
    pattern_data['max_high_pct'] = ((next_candles['high'].max() - low_point['low']) / low_point['low']) * 100
    pattern_data['min_low_pct'] = ((next_candles['low'].min() - low_point['low']) / low_point['low']) * 100
    pattern_data['final_close_pct'] = ((next_candles.iloc[-1]['close'] - low_point['low']) / low_point['low']) * 100
    
    # 연속 패턴
    consecutive_green = 0
    max_consecutive_green = 0
    for candle in next_candles.itertuples():
        if candle.body > 0:
            consecutive_green += 1
            max_consecutive_green = max(max_consecutive_green, consecutive_green)
        else:
            consecutive_green = 0
    pattern_data['max_consecutive_green'] = max_consecutive_green
    
    # RSI/MACD 변화
    pattern_data['rsi_change'] = next_candles.iloc[-1]['rsi'] - low_point['rsi']
    pattern_data['macd_hist_change'] = next_candles.iloc[-1]['macd_hist'] - low_point['macd_hist']
    
    # 평균 거래량
    pattern_data['avg_volume_ratio'] = next_candles['volume_ratio'].mean()
    
    patterns.append(pattern_data)

df_patterns = pd.DataFrame(patterns)

print(f"✅ 분석 완료: {len(df_patterns):,}개 L값 패턴")

# ============================================================
# 통계 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 L값 이후 캔들 패턴 통계")
print("=" * 60)

print(f"\n전체 평균:")
print(f"   양봉(초록) 비율:        {df_patterns['green_ratio'].mean() * 100:.1f}%")
print(f"   최대 연속 양봉:         {df_patterns['max_consecutive_green'].mean():.1f}개")
print(f"   평균 캔들 크기:         {df_patterns['avg_body_pct'].mean():+.2f}%")
print(f"   최고가 도달:            {df_patterns['max_high_pct'].mean():+.2f}%")
print(f"   최저가 도달:            {df_patterns['min_low_pct'].mean():+.2f}%")
print(f"   최종 종가 위치:         {df_patterns['final_close_pct'].mean():+.2f}%")
print(f"   RSI 변화:               {df_patterns['rsi_change'].mean():+.2f}")
print(f"   MACD Hist 변화:         {df_patterns['macd_hist_change'].mean():+.2f}")
print(f"   평균 거래량 비율:       {df_patterns['avg_volume_ratio'].mean():.2f}")

# ============================================================
# 처음 N개 캔들 상세 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 L값 이후 각 캔들별 평균 수익률")
print("=" * 60)

print(f"\n{'캔들':>6s} {'평균 수익률':>12s} {'중앙값':>10s} {'양봉비율':>10s}")
print("-" * 45)
for i in range(1, 11):
    col_name = f'c{i}_close_change_pct'
    body_col = f'c{i}_body_pct'
    
    if col_name in df_patterns.columns:
        avg_profit = df_patterns[col_name].mean()
        median_profit = df_patterns[col_name].median()
        green_pct = len(df_patterns[df_patterns[body_col] > 0]) / len(df_patterns) * 100
        print(f"{i:>6d} {avg_profit:>+11.2f}% {median_profit:>+9.2f}% {green_pct:>9.1f}%")

# ============================================================
# 성공 vs 실패 패턴 비교
# ============================================================

print("\n" + "=" * 60)
print("📊 성공 vs 실패 패턴 비교 (20개 캔들 후 기준)")
print("=" * 60)

# 성공: 최종 종가가 L값보다 2% 이상 상승
# 실패: 최종 종가가 L값보다 하락
success = df_patterns[df_patterns['final_close_pct'] > 2]
fail = df_patterns[df_patterns['final_close_pct'] < 0]
neutral = df_patterns[(df_patterns['final_close_pct'] >= 0) & (df_patterns['final_close_pct'] <= 2)]

print(f"\n성공 (>+2%): {len(success)}개 ({len(success)/len(df_patterns)*100:.1f}%)")
print(f"중립 (0~2%): {len(neutral)}개 ({len(neutral)/len(df_patterns)*100:.1f}%)")
print(f"실패 (<0%):  {len(fail)}개 ({len(fail)/len(df_patterns)*100:.1f}%)")

comparison_metrics = {
    '양봉 비율': 'green_ratio',
    '최대 연속 양봉': 'max_consecutive_green',
    '평균 캔들 크기': 'avg_body_pct',
    '최고가 도달': 'max_high_pct',
    '최저가 도달': 'min_low_pct',
    'RSI 변화': 'rsi_change',
    'MACD Hist 변화': 'macd_hist_change',
    '평균 거래량 비율': 'avg_volume_ratio',
    'L값 RSI': 'low_rsi',
    'L값 MACD Hist': 'low_macd_hist',
    'L값 Volume Ratio': 'low_volume_ratio'
}

print(f"\n{'지표':20s} {'성공':>12s} {'실패':>12s} {'차이':>12s}")
print("-" * 60)
for name, col in comparison_metrics.items():
    success_val = success[col].mean()
    fail_val = fail[col].mean()
    diff = success_val - fail_val
    print(f"{name:20s} {success_val:>12.2f} {fail_val:>12.2f} {diff:>+12.2f}")

# ============================================================
# 캔들 패턴 시퀀스 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 처음 3개 캔들 패턴 시퀀스")
print("=" * 60)

# 첫 3개 캔들의 방향 패턴
sequences = []
for idx, row in df_patterns.iterrows():
    seq = ""
    for i in range(1, 4):
        body_col = f'c{i}_body_pct'
        if body_col in row:
            seq += "G" if row[body_col] > 0 else "R"
    sequences.append(seq)

df_patterns['sequence_3'] = sequences

print("\n처음 3개 캔들 패턴 (G=양봉, R=음봉):")
seq_stats = df_patterns.groupby('sequence_3').agg({
    'final_close_pct': ['mean', 'count']
}).round(2)

seq_stats.columns = ['평균 수익률', '발생 횟수']
seq_stats = seq_stats.sort_values('평균 수익률', ascending=False)
print(seq_stats)

# ============================================================
# 초기 반등 강도별 분류
# ============================================================

print("\n" + "=" * 60)
print("📊 초기 반등 강도별 분류 (첫 3개 캔들)")
print("=" * 60)

df_patterns['first_3_gain'] = df_patterns['c3_close_change_pct']

# 강도별 분류
strong_bounce = df_patterns[df_patterns['first_3_gain'] > 1]  # 1% 이상
weak_bounce = df_patterns[(df_patterns['first_3_gain'] >= 0) & (df_patterns['first_3_gain'] <= 1)]
false_bounce = df_patterns[df_patterns['first_3_gain'] < 0]  # 추가 하락

print(f"\n강한 반등 (>1%):   {len(strong_bounce)}개 ({len(strong_bounce)/len(df_patterns)*100:.1f}%)")
print(f"   최종 평균 수익: {strong_bounce['final_close_pct'].mean():+.2f}%")

print(f"\n약한 반등 (0~1%):  {len(weak_bounce)}개 ({len(weak_bounce)/len(df_patterns)*100:.1f}%)")
print(f"   최종 평균 수익: {weak_bounce['final_close_pct'].mean():+.2f}%")

print(f"\n가짜 반등 (<0%):   {len(false_bounce)}개 ({len(false_bounce)/len(df_patterns)*100:.1f}%)")
print(f"   최종 평균 수익: {false_bounce['final_close_pct'].mean():+.2f}%")

# ============================================================
# 저장
# ============================================================

df_patterns.to_csv('candles_after_low_analysis.csv', index=False)

print("\n💾 결과 저장: candles_after_low_analysis.csv")

print("\n" + "=" * 60)
print("✅ L값 이후 캔들 패턴 분석 완료!")
print("=" * 60)
