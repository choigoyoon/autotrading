#!/usr/bin/env python3
"""
L값 변화 단계 분석

목표: 이전 L → 현재 L → 다음 L의 흐름 속에서
      현재 L이 어떤 단계(LL, HL)인지 파악하고
      각 단계별 인디케이터 패턴 분석
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("L값 변화 단계 분석 (이전 L → 현재 L → 다음 L)")
print("=" * 70)

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
    
    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # Volume
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    return df

print("\n📊 데이터 로드 및 지표 계산 중...")
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = calculate_all_indicators(df)
df = df.dropna().reset_index(drop=True)

print(f"✅ 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# ============================================================
# Swing Low 탐지
# ============================================================

print("\n🔍 Swing Low (L값) 탐지 중...")

order = 10
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]

print(f"✅ Swing Low: {len(low_indices):,}개")

# L값 DataFrame 생성
l_values = df.iloc[low_indices].copy()
l_values['l_price'] = l_values['low']
l_values = l_values.reset_index(drop=True)

# ============================================================
# L값 간 관계 분석 (이전 L → 현재 L → 다음 L)
# ============================================================

print("\n🔍 L값 변화 단계 분석 중...")

l_sequences = []

for i in range(1, len(l_values) - 1):
    prev_l = l_values.iloc[i - 1]
    curr_l = l_values.iloc[i]
    next_l = l_values.iloc[i + 1]
    
    # 현재 L의 패턴 분류
    if curr_l['l_price'] < prev_l['l_price']:
        curr_pattern = 'LL'  # Lower Low
    else:
        curr_pattern = 'HL'  # Higher Low
    
    # 다음 L의 패턴 예측
    if next_l['l_price'] < curr_l['l_price']:
        next_pattern = 'LL'
    else:
        next_pattern = 'HL'
    
    # 3개 L의 연속 패턴
    sequence_pattern = f"{prev_l['l_price'] > curr_l['l_price'] and 'LL' or 'HL'}→{curr_pattern}→{next_pattern}"
    
    # 가격 변화
    prev_to_curr_change = ((curr_l['l_price'] - prev_l['l_price']) / prev_l['l_price']) * 100
    curr_to_next_change = ((next_l['l_price'] - curr_l['l_price']) / curr_l['l_price']) * 100
    
    # 지표 변화
    rsi_change_prev = curr_l['rsi'] - prev_l['rsi']
    rsi_change_next = next_l['rsi'] - curr_l['rsi']
    
    macd_change_prev = curr_l['macd_hist'] - prev_l['macd_hist']
    macd_change_next = next_l['macd_hist'] - curr_l['macd_hist']
    
    stoch_change_prev = curr_l['stoch_k'] - prev_l['stoch_k']
    stoch_change_next = next_l['stoch_k'] - curr_l['stoch_k']
    
    volume_change_prev = curr_l['volume_ratio'] - prev_l['volume_ratio']
    volume_change_next = next_l['volume_ratio'] - curr_l['volume_ratio']
    
    # L 간 시간 간격
    time_to_curr = (curr_l['datetime'] - prev_l['datetime']).total_seconds() / 3600
    time_to_next = (next_l['datetime'] - curr_l['datetime']).total_seconds() / 3600
    
    l_sequences.append({
        'curr_datetime': curr_l['datetime'],
        'curr_pattern': curr_pattern,
        'next_pattern': next_pattern,
        'sequence_pattern': sequence_pattern,
        
        # 가격
        'prev_l_price': prev_l['l_price'],
        'curr_l_price': curr_l['l_price'],
        'next_l_price': next_l['l_price'],
        'prev_to_curr_change_pct': prev_to_curr_change,
        'curr_to_next_change_pct': curr_to_next_change,
        
        # 현재 L의 지표
        'curr_rsi': curr_l['rsi'],
        'curr_macd_hist': curr_l['macd_hist'],
        'curr_stoch_k': curr_l['stoch_k'],
        'curr_bb_position': curr_l['bb_position'],
        'curr_cci': curr_l['cci'],
        'curr_volume_ratio': curr_l['volume_ratio'],
        
        # 이전 L 대비 지표 변화
        'rsi_change_from_prev': rsi_change_prev,
        'macd_hist_change_from_prev': macd_change_prev,
        'stoch_k_change_from_prev': stoch_change_prev,
        'volume_ratio_change_from_prev': volume_change_prev,
        
        # 다음 L로의 지표 변화
        'rsi_change_to_next': rsi_change_next,
        'macd_hist_change_to_next': macd_change_next,
        'stoch_k_change_to_next': stoch_change_next,
        'volume_ratio_change_to_next': volume_change_next,
        
        # 시간 간격
        'time_from_prev_hours': time_to_curr,
        'time_to_next_hours': time_to_next,
    })

df_sequences = pd.DataFrame(l_sequences)

print(f"✅ 분석 완료: {len(df_sequences):,}개 L값 시퀀스")

# ============================================================
# 현재 L 패턴별 통계
# ============================================================

print("\n" + "=" * 70)
print("📊 현재 L 패턴별 통계 (LL vs HL)")
print("=" * 70)

for pattern in ['LL', 'HL']:
    subset = df_sequences[df_sequences['curr_pattern'] == pattern]
    print(f"\n[{pattern}] {'Lower Low' if pattern == 'LL' else 'Higher Low'} ({len(subset)}개)")
    print(f"   가격 변화 (이전 L 대비): {subset['prev_to_curr_change_pct'].mean():+.2f}%")
    print(f"   가격 변화 (다음 L로):    {subset['curr_to_next_change_pct'].mean():+.2f}%")
    print(f"   RSI:                    {subset['curr_rsi'].mean():.2f}")
    print(f"   MACD Hist:              {subset['curr_macd_hist'].mean():.2f}")
    print(f"   Stochastic K:           {subset['curr_stoch_k'].mean():.2f}")
    print(f"   BB Position:            {subset['curr_bb_position'].mean():.2f}")
    print(f"   CCI:                    {subset['curr_cci'].mean():.2f}")
    print(f"   Volume Ratio:           {subset['curr_volume_ratio'].mean():.2f}")

# ============================================================
# 다음 L 패턴 예측 (현재 LL일 때 vs HL일 때)
# ============================================================

print("\n" + "=" * 70)
print("📊 현재 L 패턴에 따른 다음 L 패턴 확률")
print("=" * 70)

for curr_pattern in ['LL', 'HL']:
    subset = df_sequences[df_sequences['curr_pattern'] == curr_pattern]
    
    print(f"\n[현재 L = {curr_pattern}]")
    next_ll_count = len(subset[subset['next_pattern'] == 'LL'])
    next_hl_count = len(subset[subset['next_pattern'] == 'HL'])
    
    print(f"   다음 L = LL: {next_ll_count}개 ({next_ll_count/len(subset)*100:.1f}%)")
    print(f"   다음 L = HL: {next_hl_count}개 ({next_hl_count/len(subset)*100:.1f}%)")

# ============================================================
# 3단계 시퀀스 패턴 분석
# ============================================================

print("\n" + "=" * 70)
print("📊 3단계 L 시퀀스 패턴 (이전→현재→다음)")
print("=" * 70)

sequence_stats = df_sequences.groupby('sequence_pattern').agg({
    'curr_to_next_change_pct': 'mean',
    'curr_datetime': 'count'
}).round(2)

sequence_stats.columns = ['다음 L까지 평균 변화(%)', '발생 횟수']
sequence_stats = sequence_stats.sort_values('발생 횟수', ascending=False)

print("\n시퀀스 패턴별 통계:")
print(sequence_stats)

# ============================================================
# LL → HL 전환 시점의 특징
# ============================================================

print("\n" + "=" * 70)
print("📊 반등 전환 시점 (LL → HL) 인디케이터 특징")
print("=" * 70)

ll_to_hl = df_sequences[(df_sequences['curr_pattern'] == 'LL') & (df_sequences['next_pattern'] == 'HL')]
ll_to_ll = df_sequences[(df_sequences['curr_pattern'] == 'LL') & (df_sequences['next_pattern'] == 'LL')]

print(f"\nLL → HL (반등 성공): {len(ll_to_hl)}개")
print(f"LL → LL (추가 하락): {len(ll_to_ll)}개")

comparison_indicators = [
    ('현재 RSI', 'curr_rsi'),
    ('현재 MACD Hist', 'curr_macd_hist'),
    ('현재 Stoch K', 'curr_stoch_k'),
    ('현재 BB Position', 'curr_bb_position'),
    ('현재 CCI', 'curr_cci'),
    ('현재 Volume Ratio', 'curr_volume_ratio'),
    ('RSI 변화 (이전 대비)', 'rsi_change_from_prev'),
    ('MACD Hist 변화 (이전 대비)', 'macd_hist_change_from_prev'),
    ('Volume 변화 (이전 대비)', 'volume_ratio_change_from_prev'),
]

print(f"\n{'지표':30s} {'LL→HL':>12s} {'LL→LL':>12s} {'차이':>12s}")
print("-" * 70)
for name, col in comparison_indicators:
    ll_to_hl_val = ll_to_hl[col].mean()
    ll_to_ll_val = ll_to_ll[col].mean()
    diff = ll_to_hl_val - ll_to_ll_val
    print(f"{name:30s} {ll_to_hl_val:>12.2f} {ll_to_ll_val:>12.2f} {diff:>+12.2f}")

# ============================================================
# HL → HL 지속 시점의 특징
# ============================================================

print("\n" + "=" * 70)
print("📊 상승 지속 시점 (HL → HL) 인디케이터 특징")
print("=" * 70)

hl_to_hl = df_sequences[(df_sequences['curr_pattern'] == 'HL') & (df_sequences['next_pattern'] == 'HL')]
hl_to_ll = df_sequences[(df_sequences['curr_pattern'] == 'HL') & (df_sequences['next_pattern'] == 'LL')]

print(f"\nHL → HL (상승 지속): {len(hl_to_hl)}개")
print(f"HL → LL (하락 전환): {len(hl_to_ll)}개")

print(f"\n{'지표':30s} {'HL→HL':>12s} {'HL→LL':>12s} {'차이':>12s}")
print("-" * 70)
for name, col in comparison_indicators:
    hl_to_hl_val = hl_to_hl[col].mean()
    hl_to_ll_val = hl_to_ll[col].mean()
    diff = hl_to_hl_val - hl_to_ll_val
    print(f"{name:30s} {hl_to_hl_val:>12.2f} {hl_to_ll_val:>12.2f} {diff:>+12.2f}")

# ============================================================
# 연속 LL 패턴 분석
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 패턴 분석")
print("=" * 70)

# 연속 LL 카운트
consecutive_ll_count = 0
max_consecutive_ll = 0
consecutive_patterns = []

for i in range(len(df_sequences)):
    if df_sequences.loc[i, 'curr_pattern'] == 'LL':
        consecutive_ll_count += 1
        max_consecutive_ll = max(max_consecutive_ll, consecutive_ll_count)
    else:
        if consecutive_ll_count >= 2:
            consecutive_patterns.append(consecutive_ll_count)
        consecutive_ll_count = 0

print(f"\n최대 연속 LL: {max_consecutive_ll}개")
print(f"평균 연속 LL: {np.mean(consecutive_patterns):.2f}개" if consecutive_patterns else "평균 연속 LL: N/A")

# ============================================================
# 실전 진입 조건 도출
# ============================================================

print("\n" + "=" * 70)
print("📊 실전 진입 조건 (LL에서 HL 전환 포착)")
print("=" * 70)

# LL → HL 전환 조건 분석
conditions = {
    'RSI < 30': (ll_to_hl['curr_rsi'] < 30).sum(),
    'RSI < 35': (ll_to_hl['curr_rsi'] < 35).sum(),
    'RSI < 40': (ll_to_hl['curr_rsi'] < 40).sum(),
    'MACD Hist < -30': (ll_to_hl['curr_macd_hist'] < -30).sum(),
    'MACD Hist < -50': (ll_to_hl['curr_macd_hist'] < -50).sum(),
    'Stoch K < 30': (ll_to_hl['curr_stoch_k'] < 30).sum(),
    'BB Position < 0.2': (ll_to_hl['curr_bb_position'] < 0.2).sum(),
    'CCI < -100': (ll_to_hl['curr_cci'] < -100).sum(),
    'Volume Ratio > 2.0': (ll_to_hl['curr_volume_ratio'] > 2.0).sum(),
    'RSI 상승 (이전 대비)': (ll_to_hl['rsi_change_from_prev'] > 0).sum(),
    'MACD Hist 상승': (ll_to_hl['macd_hist_change_from_prev'] > 0).sum(),
}

print("\nLL → HL 전환 시 조건 충족률:")
for condition_name, count in conditions.items():
    pct = (count / len(ll_to_hl)) * 100
    print(f"   {condition_name:25s}: {pct:5.1f}%")

# ============================================================
# 저장
# ============================================================

df_sequences.to_csv('L_sequence_stages_analysis.csv', index=False)

print("\n💾 결과 저장: L_sequence_stages_analysis.csv")

print("\n" + "=" * 70)
print("✅ L값 변화 단계 분석 완료!")
print("=" * 70)
