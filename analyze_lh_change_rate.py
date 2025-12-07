#!/usr/bin/env python3
"""
각 LH 발생 시점의 이전 LH 대비 변화율 분석
- LH → LH 간의 가격 변화
- 지표 변화율 추적
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LH → LH 변화율 분석")
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
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    return df

print("\n📊 데이터 로드 및 지표 계산 중...")
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = calculate_all_indicators(df)
df = df.dropna().reset_index(drop=True)

print(f"✅ 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"✅ 총 15분봉 개수: {len(df):,}개")

# ============================================================
# Swing High 탐지 및 LH 분류
# ============================================================

print("\n🔍 Swing High 탐지 및 LH 분류 중...")

order = 10
high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
swing_highs = df.iloc[high_indices].copy()
swing_highs['swing_price'] = swing_highs['high']
swing_highs = swing_highs.reset_index(drop=True)

# 이전 Swing High와 비교
swing_highs['prev_swing_price'] = swing_highs['swing_price'].shift(1)
swing_highs['price_diff'] = swing_highs['swing_price'] - swing_highs['prev_swing_price']
swing_highs['price_diff_pct'] = (swing_highs['price_diff'] / swing_highs['prev_swing_price']) * 100
swing_highs['pattern'] = np.where(swing_highs['price_diff'] > 0, 'HH', 'LH')

swing_highs = swing_highs.dropna(subset=['prev_swing_price'])

# LH만 추출
lh_data = swing_highs[swing_highs['pattern'] == 'LH'].copy()
print(f"✅ LH 패턴: {len(lh_data):,}개")

# ============================================================
# LH → LH 간 변화율 계산
# ============================================================

print("\n🔍 LH → LH 간 변화율 계산 중...")

lh_data = lh_data.reset_index(drop=True)

# 이전 LH 값들
lh_data['prev_lh_price'] = lh_data['swing_price'].shift(1)
lh_data['prev_lh_macd'] = lh_data['macd'].shift(1)
lh_data['prev_lh_macd_hist'] = lh_data['macd_hist'].shift(1)
lh_data['prev_lh_rsi'] = lh_data['rsi'].shift(1)
lh_data['prev_lh_bb_position'] = lh_data['bb_position'].shift(1)
lh_data['prev_lh_atr_pct'] = lh_data['atr_pct'].shift(1)

# 변화율 계산
lh_data['lh_price_change'] = lh_data['swing_price'] - lh_data['prev_lh_price']
lh_data['lh_price_change_pct'] = (lh_data['lh_price_change'] / lh_data['prev_lh_price']) * 100

lh_data['lh_macd_change'] = lh_data['macd'] - lh_data['prev_lh_macd']
lh_data['lh_macd_hist_change'] = lh_data['macd_hist'] - lh_data['prev_lh_macd_hist']
lh_data['lh_rsi_change'] = lh_data['rsi'] - lh_data['prev_lh_rsi']
lh_data['lh_bb_position_change'] = lh_data['bb_position'] - lh_data['prev_lh_bb_position']
lh_data['lh_atr_pct_change'] = lh_data['atr_pct'] - lh_data['prev_lh_atr_pct']

# LH 간 시간 차이
lh_data['prev_lh_datetime'] = lh_data['datetime'].shift(1)
lh_data['lh_time_diff'] = lh_data['datetime'] - lh_data['prev_lh_datetime']
lh_data['lh_time_diff_hours'] = lh_data['lh_time_diff'].dt.total_seconds() / 3600

# NaN 제거
lh_changes = lh_data.dropna(subset=['prev_lh_price']).copy()

print(f"✅ LH → LH 변화 데이터: {len(lh_changes):,}개")

# ============================================================
# 통계 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 LH → LH 변화율 통계")
print("=" * 60)

change_indicators = {
    'LH 가격 변화 (USD)': 'lh_price_change',
    'LH 가격 변화 (%)': 'lh_price_change_pct',
    'MACD 변화': 'lh_macd_change',
    'MACD Hist 변화': 'lh_macd_hist_change',
    'RSI 변화': 'lh_rsi_change',
    'BB Position 변화': 'lh_bb_position_change',
    'ATR % 변화': 'lh_atr_pct_change',
    'LH 간 시간 (시간)': 'lh_time_diff_hours'
}

print("\n변화율 평균:")
for name, col in change_indicators.items():
    if col in lh_changes.columns:
        mean_val = lh_changes[col].mean()
        median_val = lh_changes[col].median()
        std_val = lh_changes[col].std()
        min_val = lh_changes[col].min()
        max_val = lh_changes[col].max()
        print(f"\n{name}:")
        print(f"   평균: {mean_val:+10.2f} | 중앙값: {median_val:+10.2f}")
        print(f"   표준편차: {std_val:10.2f} | 최소: {min_val:+10.2f} | 최대: {max_val:+10.2f}")

# ============================================================
# 변화율 구간별 분포
# ============================================================

print("\n" + "=" * 60)
print("📊 LH 가격 변화율 분포")
print("=" * 60)

# 가격 변화율(%)
print("\n[LH 가격 변화율(%) 분포]")
price_change_bins = [-float('inf'), -10, -5, -2, -1, 0, 1, 2, 5, 10, float('inf')]
price_change_labels = ['<-10%', '-10~-5%', '-5~-2%', '-2~-1%', '-1~0%', '0~1%', '1~2%', '2~5%', '5~10%', '>10%']
lh_changes['price_change_bin'] = pd.cut(lh_changes['lh_price_change_pct'], bins=price_change_bins, labels=price_change_labels)

for label in price_change_labels:
    count = len(lh_changes[lh_changes['price_change_bin'] == label])
    if count > 0:
        pct = count / len(lh_changes) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# RSI 변화
print("\n[LH RSI 변화 분포]")
rsi_change_bins = [-float('inf'), -20, -10, -5, 0, 5, 10, 20, float('inf')]
rsi_change_labels = ['<-20', '-20~-10', '-10~-5', '-5~0', '0~5', '5~10', '10~20', '>20']
lh_changes['rsi_change_bin'] = pd.cut(lh_changes['lh_rsi_change'], bins=rsi_change_bins, labels=rsi_change_labels)

for label in rsi_change_labels:
    count = len(lh_changes[lh_changes['rsi_change_bin'] == label])
    if count > 0:
        pct = count / len(lh_changes) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# MACD Histogram 변화
print("\n[MACD Histogram 변화 분포]")
macd_change_bins = [-float('inf'), -50, -20, -10, 0, 10, 20, 50, float('inf')]
macd_change_labels = ['<-50', '-50~-20', '-20~-10', '-10~0', '0~10', '10~20', '20~50', '>50']
lh_changes['macd_change_bin'] = pd.cut(lh_changes['lh_macd_hist_change'], bins=macd_change_bins, labels=macd_change_labels)

for label in macd_change_labels:
    count = len(lh_changes[lh_changes['macd_change_bin'] == label])
    if count > 0:
        pct = count / len(lh_changes) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# LH 간 시간 간격
print("\n[LH 간 시간 간격 분포]")
time_bins = [0, 6, 12, 24, 48, 72, 168, float('inf')]
time_labels = ['0~6h', '6~12h', '12~24h', '24~48h', '48~72h', '72~168h', '>168h']
lh_changes['time_bin'] = pd.cut(lh_changes['lh_time_diff_hours'], bins=time_bins, labels=time_labels)

for label in time_labels:
    count = len(lh_changes[lh_changes['time_bin'] == label])
    if count > 0:
        pct = count / len(lh_changes) * 100
        avg_price_change = lh_changes[lh_changes['time_bin'] == label]['lh_price_change_pct'].mean()
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%) | 평균 가격 변화: {avg_price_change:+6.2f}%")

# ============================================================
# 상승/하락 LH 비교
# ============================================================

print("\n" + "=" * 60)
print("📊 LH 가격 변화 방향별 지표 차이")
print("=" * 60)

rising_lh = lh_changes[lh_changes['lh_price_change_pct'] > 0]
falling_lh = lh_changes[lh_changes['lh_price_change_pct'] < 0]

print(f"\n상승 LH (이전 LH보다 높은 가격): {len(rising_lh)}회 ({len(rising_lh)/len(lh_changes)*100:.1f}%)")
print(f"하락 LH (이전 LH보다 낮은 가격): {len(falling_lh)}회 ({len(falling_lh)/len(lh_changes)*100:.1f}%)")

comparison_indicators = {
    'RSI 변화': 'lh_rsi_change',
    'MACD Hist 변화': 'lh_macd_hist_change',
    'BB Position 변화': 'lh_bb_position_change',
    'ATR % 변화': 'lh_atr_pct_change'
}

print("\n지표 변화 비교:")
print(f"{'지표':20s} {'상승 LH':>15s} {'하락 LH':>15s} {'차이':>15s}")
print("-" * 70)
for name, col in comparison_indicators.items():
    rising_mean = rising_lh[col].mean()
    falling_mean = falling_lh[col].mean()
    diff = rising_mean - falling_mean
    print(f"{name:20s} {rising_mean:+15.2f} {falling_mean:+15.2f} {diff:+15.2f}")

# ============================================================
# 저장
# ============================================================

output_cols = ['datetime', 'swing_price', 'prev_lh_price', 'lh_price_change', 'lh_price_change_pct',
               'lh_time_diff_hours', 'macd', 'prev_lh_macd', 'lh_macd_change',
               'macd_hist', 'prev_lh_macd_hist', 'lh_macd_hist_change',
               'rsi', 'prev_lh_rsi', 'lh_rsi_change',
               'bb_position', 'prev_lh_bb_position', 'lh_bb_position_change',
               'atr_pct', 'prev_lh_atr_pct', 'lh_atr_pct_change']

lh_changes[output_cols].to_csv('lh_change_rate_analysis.csv', index=False)

print("\n💾 결과 저장: lh_change_rate_analysis.csv")

print("\n" + "=" * 60)
