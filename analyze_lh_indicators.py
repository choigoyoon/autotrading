#!/usr/bin/env python3
"""
LH (Lower High) 발생 시점의 지표(인디케이터) 값 분석
- MACD, RSI, Bollinger Bands, ATR, Volume 등
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LH 발생 시점 지표 분석")
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
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['ma_distance_20'] = ((df['close'] - df['ma_20']) / df['ma_20']) * 100
    df['ma_distance_50'] = ((df['close'] - df['ma_50']) / df['ma_50']) * 100
    
    # Volume (정규화)
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
# Swing High 탐지
# ============================================================

print("\n🔍 Swing High 탐지 중...")

order = 10
high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
swing_highs = df.iloc[high_indices].copy()
swing_highs['swing_price'] = swing_highs['high']

print(f"✅ Swing High: {len(swing_highs):,}개")

# ============================================================
# LH 패턴 분류
# ============================================================

print("🔍 LH 패턴 분류 중...")

swing_highs = swing_highs.reset_index(drop=True)
swing_highs['prev_price'] = swing_highs['swing_price'].shift(1)
swing_highs['diff'] = swing_highs['swing_price'] - swing_highs['prev_price']
swing_highs['diff_pct'] = (swing_highs['diff'] / swing_highs['prev_price']) * 100
swing_highs['pattern'] = np.where(swing_highs['diff'] > 0, 'HH', 'LH')

swing_highs = swing_highs.dropna(subset=['prev_price'])

lh_data = swing_highs[swing_highs['pattern'] == 'LH'].copy()
print(f"✅ LH 패턴: {len(lh_data):,}개")

# ============================================================
# LH 발생 시점 지표 통계 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 LH 발생 시점의 지표 평균값")
print("=" * 60)

indicators = {
    'MACD': 'macd',
    'MACD Signal': 'macd_signal',
    'MACD Histogram': 'macd_hist',
    'RSI': 'rsi',
    'Stochastic %K': 'stoch_k',
    'Stochastic %D': 'stoch_d',
    'BB Position (0~1)': 'bb_position',
    'ATR %': 'atr_pct',
    'MA20 Distance %': 'ma_distance_20',
    'MA50 Distance %': 'ma_distance_50',
    'Volume Ratio': 'volume_ratio'
}

print("\n지표별 평균값:")
for name, col in indicators.items():
    if col in lh_data.columns:
        mean_val = lh_data[col].mean()
        median_val = lh_data[col].median()
        std_val = lh_data[col].std()
        print(f"   {name:20s}: 평균 {mean_val:8.2f} | 중앙값 {median_val:8.2f} | 표준편차 {std_val:8.2f}")

# ============================================================
# 구간별 분포 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 LH 발생 시점의 지표 분포")
print("=" * 60)

# MACD
print("\n[MACD Histogram 분포]")
macd_bins = [-float('inf'), -100, -50, -10, 0, 10, 50, 100, float('inf')]
macd_labels = ['<-100', '-100~-50', '-50~-10', '-10~0', '0~10', '10~50', '50~100', '>100']
lh_data['macd_bin'] = pd.cut(lh_data['macd_hist'], bins=macd_bins, labels=macd_labels)
for label in macd_labels:
    count = len(lh_data[lh_data['macd_bin'] == label])
    if count > 0:
        pct = count / len(lh_data) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# RSI
print("\n[RSI 분포]")
rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
rsi_labels = ['0~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~100']
lh_data['rsi_bin'] = pd.cut(lh_data['rsi'], bins=rsi_bins, labels=rsi_labels)
for label in rsi_labels:
    count = len(lh_data[lh_data['rsi_bin'] == label])
    if count > 0:
        pct = count / len(lh_data) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# Bollinger Band Position
print("\n[Bollinger Band Position 분포] (0=하단, 1=상단)")
bb_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bb_labels = ['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']
lh_data['bb_bin'] = pd.cut(lh_data['bb_position'], bins=bb_bins, labels=bb_labels)
for label in bb_labels:
    count = len(lh_data[lh_data['bb_bin'] == label])
    if count > 0:
        pct = count / len(lh_data) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# MA Distance
print("\n[MA20 거리(%) 분포]")
ma_bins = [-float('inf'), -5, -2, -1, 0, 1, 2, 5, float('inf')]
ma_labels = ['<-5%', '-5~-2%', '-2~-1%', '-1~0%', '0~1%', '1~2%', '2~5%', '>5%']
lh_data['ma_bin'] = pd.cut(lh_data['ma_distance_20'], bins=ma_bins, labels=ma_labels)
for label in ma_labels:
    count = len(lh_data[lh_data['ma_bin'] == label])
    if count > 0:
        pct = count / len(lh_data) * 100
        print(f"   {label:10s}: {count:4d}회 ({pct:5.1f}%)")

# ============================================================
# 조건별 필터링
# ============================================================

print("\n" + "=" * 60)
print("📊 LH 발생 조건별 필터")
print("=" * 60)

conditions = {
    'MACD < 0': lh_data['macd'] < 0,
    'MACD > 0': lh_data['macd'] > 0,
    'MACD Hist < 0': lh_data['macd_hist'] < 0,
    'RSI < 30': lh_data['rsi'] < 30,
    'RSI 30~50': (lh_data['rsi'] >= 30) & (lh_data['rsi'] < 50),
    'RSI 50~70': (lh_data['rsi'] >= 50) & (lh_data['rsi'] < 70),
    'RSI > 70': lh_data['rsi'] >= 70,
    'BB 상단(>0.8)': lh_data['bb_position'] > 0.8,
    'BB 중간(0.4~0.6)': (lh_data['bb_position'] >= 0.4) & (lh_data['bb_position'] <= 0.6),
    'BB 하단(<0.2)': lh_data['bb_position'] < 0.2,
    '가격 > MA20': lh_data['close'] > lh_data['ma_20'],
    '가격 < MA20': lh_data['close'] < lh_data['ma_20'],
}

print("\n조건별 LH 발생 빈도:")
for condition_name, condition_mask in conditions.items():
    count = condition_mask.sum()
    pct = count / len(lh_data) * 100
    print(f"   {condition_name:20s}: {count:4d}회 ({pct:5.1f}%)")

# ============================================================
# 저장
# ============================================================

output_cols = ['datetime', 'swing_price', 'prev_price', 'diff', 'diff_pct', 
               'macd', 'macd_signal', 'macd_hist', 'rsi', 'stoch_k', 'stoch_d',
               'bb_position', 'atr_pct', 'ma_distance_20', 'ma_distance_50', 'volume_ratio']

lh_data[output_cols].to_csv('lh_indicators_analysis.csv', index=False)

print("\n💾 결과 저장: lh_indicators_analysis.csv")

print("\n" + "=" * 60)
