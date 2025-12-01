#!/usr/bin/env python3
"""
연속 LL 횟수별 패턴 분석

목표: LL이 1번, 2번, 3번, 4번... 연속될 때
      각 단계별 인디케이터 특징과 반등 확률 분석
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("연속 LL 횟수별 패턴 분석")
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
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
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
# 연속 LL 패턴 추적
# ============================================================

print("\n🔍 연속 LL 패턴 추적 중...")

consecutive_patterns = []
consecutive_ll_count = 0

for i in range(1, len(l_values)):
    prev_l = l_values.iloc[i - 1]
    curr_l = l_values.iloc[i]
    
    # 현재 L 패턴
    if curr_l['l_price'] < prev_l['l_price']:
        consecutive_ll_count += 1
        current_pattern = 'LL'
    else:
        # HL로 전환 = 연속 LL 종료
        if consecutive_ll_count > 0:
            # 이전 LL의 정보 저장 (마지막 LL)
            last_ll = l_values.iloc[i - 1]
            
            consecutive_patterns.append({
                'll_count': consecutive_ll_count,
                'last_ll_datetime': last_ll['datetime'],
                'last_ll_price': last_ll['l_price'],
                'last_ll_rsi': last_ll['rsi'],
                'last_ll_macd_hist': last_ll['macd_hist'],
                'last_ll_stoch_k': last_ll['stoch_k'],
                'last_ll_bb_position': last_ll['bb_position'],
                'last_ll_cci': last_ll['cci'],
                'last_ll_volume_ratio': last_ll['volume_ratio'],
                'last_ll_atr_pct': last_ll['atr_pct'],
                
                # HL로 전환된 현재 L의 정보
                'first_hl_datetime': curr_l['datetime'],
                'first_hl_price': curr_l['l_price'],
                'first_hl_rsi': curr_l['rsi'],
                'first_hl_macd_hist': curr_l['macd_hist'],
                'first_hl_stoch_k': curr_l['stoch_k'],
                'first_hl_bb_position': curr_l['bb_position'],
                'first_hl_cci': curr_l['cci'],
                'first_hl_volume_ratio': curr_l['volume_ratio'],
                
                # 반등 성과
                'bounce_pct': ((curr_l['l_price'] - last_ll['l_price']) / last_ll['l_price']) * 100,
                
                # 지표 변화
                'rsi_change': curr_l['rsi'] - last_ll['rsi'],
                'macd_hist_change': curr_l['macd_hist'] - last_ll['macd_hist'],
                'stoch_k_change': curr_l['stoch_k'] - last_ll['stoch_k'],
                'volume_ratio_change': curr_l['volume_ratio'] - last_ll['volume_ratio'],
            })
        
        consecutive_ll_count = 0
        current_pattern = 'HL'

df_consecutive = pd.DataFrame(consecutive_patterns)

print(f"✅ 분석 완료: {len(df_consecutive):,}개 연속 LL → HL 전환 패턴")

# ============================================================
# 연속 LL 횟수별 통계
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 횟수별 발생 빈도")
print("=" * 70)

ll_count_dist = df_consecutive['ll_count'].value_counts().sort_index()

print("\n연속 LL 횟수 분포:")
for ll_count, freq in ll_count_dist.items():
    pct = (freq / len(df_consecutive)) * 100
    print(f"   {ll_count}번 연속 LL: {freq:4d}회 ({pct:5.1f}%)")

# ============================================================
# 연속 LL 횟수별 마지막 LL 인디케이터
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 횟수별 마지막 LL의 인디케이터 평균")
print("=" * 70)

indicators = [
    ('RSI', 'last_ll_rsi'),
    ('MACD Hist', 'last_ll_macd_hist'),
    ('Stoch K', 'last_ll_stoch_k'),
    ('BB Position', 'last_ll_bb_position'),
    ('CCI', 'last_ll_cci'),
    ('Volume Ratio', 'last_ll_volume_ratio'),
    ('ATR %', 'last_ll_atr_pct'),
]

# 테이블 헤더
header = f"{'연속횟수':>8s}"
for name, _ in indicators:
    header += f" {name:>12s}"
print(f"\n{header}")
print("-" * (10 + 14 * len(indicators)))

# 연속 횟수별 평균
for ll_count in sorted(df_consecutive['ll_count'].unique()):
    subset = df_consecutive[df_consecutive['ll_count'] == ll_count]
    row = f"{ll_count:8d}"
    for _, col in indicators:
        avg_val = subset[col].mean()
        row += f" {avg_val:12.2f}"
    print(row)

# ============================================================
# 연속 LL 횟수별 반등 성과
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 횟수별 반등 성과")
print("=" * 70)

print(f"\n{'연속횟수':>8s} {'발생수':>8s} {'평균 반등':>12s} {'중앙값':>10s} {'최대':>10s} {'최소':>10s}")
print("-" * 70)

for ll_count in sorted(df_consecutive['ll_count'].unique()):
    subset = df_consecutive[df_consecutive['ll_count'] == ll_count]
    freq = len(subset)
    avg_bounce = subset['bounce_pct'].mean()
    median_bounce = subset['bounce_pct'].median()
    max_bounce = subset['bounce_pct'].max()
    min_bounce = subset['bounce_pct'].min()
    
    print(f"{ll_count:8d} {freq:8d} {avg_bounce:>+11.2f}% {median_bounce:>+9.2f}% {max_bounce:>+9.2f}% {min_bounce:>+9.2f}%")

# ============================================================
# 연속 LL 횟수별 지표 변화
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 횟수별 지표 변화 (마지막 LL → 첫 HL)")
print("=" * 70)

change_indicators = [
    ('RSI 변화', 'rsi_change'),
    ('MACD Hist 변화', 'macd_hist_change'),
    ('Stoch K 변화', 'stoch_k_change'),
    ('Volume 변화', 'volume_ratio_change'),
]

header = f"{'연속횟수':>8s}"
for name, _ in change_indicators:
    header += f" {name:>15s}"
print(f"\n{header}")
print("-" * (10 + 17 * len(change_indicators)))

for ll_count in sorted(df_consecutive['ll_count'].unique()):
    subset = df_consecutive[df_consecutive['ll_count'] == ll_count]
    row = f"{ll_count:8d}"
    for _, col in change_indicators:
        avg_change = subset[col].mean()
        row += f" {avg_change:>+15.2f}"
    print(row)

# ============================================================
# 3~4번 연속 LL 상세 분석
# ============================================================

print("\n" + "=" * 70)
print("📊 3~4번 연속 LL 상세 분석 (일반적인 하락 패턴)")
print("=" * 70)

ll_3_4 = df_consecutive[df_consecutive['ll_count'].isin([3, 4])]
ll_other = df_consecutive[~df_consecutive['ll_count'].isin([3, 4])]

print(f"\n3~4번 연속: {len(ll_3_4)}회 ({len(ll_3_4)/len(df_consecutive)*100:.1f}%)")
print(f"그 외:      {len(ll_other)}회 ({len(ll_other)/len(df_consecutive)*100:.1f}%)")

comparison = [
    ('마지막 LL RSI', 'last_ll_rsi'),
    ('마지막 LL MACD Hist', 'last_ll_macd_hist'),
    ('마지막 LL Stoch K', 'last_ll_stoch_k'),
    ('마지막 LL BB Pos', 'last_ll_bb_position'),
    ('마지막 LL CCI', 'last_ll_cci'),
    ('마지막 LL Volume', 'last_ll_volume_ratio'),
    ('마지막 LL ATR %', 'last_ll_atr_pct'),
    ('반등 성과', 'bounce_pct'),
    ('RSI 변화', 'rsi_change'),
    ('MACD Hist 변화', 'macd_hist_change'),
]

print(f"\n{'지표':25s} {'3~4번 LL':>15s} {'그 외':>15s} {'차이':>15s}")
print("-" * 75)
for name, col in comparison:
    val_3_4 = ll_3_4[col].mean()
    val_other = ll_other[col].mean()
    diff = val_3_4 - val_other
    print(f"{name:25s} {val_3_4:>15.2f} {val_other:>15.2f} {diff:>+15.2f}")

# ============================================================
# 실전 진입 조건 (3~4번 연속 LL 기준)
# ============================================================

print("\n" + "=" * 70)
print("📊 3~4번 연속 LL 후 반등 진입 조건")
print("=" * 70)

conditions = {
    'RSI < 25': (ll_3_4['last_ll_rsi'] < 25).sum(),
    'RSI < 30': (ll_3_4['last_ll_rsi'] < 30).sum(),
    'RSI < 35': (ll_3_4['last_ll_rsi'] < 35).sum(),
    'MACD Hist < -50': (ll_3_4['last_ll_macd_hist'] < -50).sum(),
    'MACD Hist < -60': (ll_3_4['last_ll_macd_hist'] < -60).sum(),
    'Stoch K < 20': (ll_3_4['last_ll_stoch_k'] < 20).sum(),
    'Stoch K < 25': (ll_3_4['last_ll_stoch_k'] < 25).sum(),
    'BB Position < 0.05': (ll_3_4['last_ll_bb_position'] < 0.05).sum(),
    'BB Position < 0.1': (ll_3_4['last_ll_bb_position'] < 0.1).sum(),
    'CCI < -150': (ll_3_4['last_ll_cci'] < -150).sum(),
    'CCI < -120': (ll_3_4['last_ll_cci'] < -120).sum(),
    'Volume Ratio > 2.5': (ll_3_4['last_ll_volume_ratio'] > 2.5).sum(),
    'Volume Ratio > 3.0': (ll_3_4['last_ll_volume_ratio'] > 3.0).sum(),
    'ATR % > 0.5': (ll_3_4['last_ll_atr_pct'] > 0.5).sum(),
}

print("\n3~4번 연속 LL 마지막 시점 조건 충족률:")
for condition_name, count in conditions.items():
    pct = (count / len(ll_3_4)) * 100
    print(f"   {condition_name:25s}: {pct:5.1f}%")

# ============================================================
# 연속 LL 구간별 성과 비교
# ============================================================

print("\n" + "=" * 70)
print("📊 연속 LL 구간별 그룹 비교")
print("=" * 70)

groups = {
    '1~2번 (약한 하락)': df_consecutive[df_consecutive['ll_count'].isin([1, 2])],
    '3~4번 (일반 하락)': df_consecutive[df_consecutive['ll_count'].isin([3, 4])],
    '5번+ (강한 하락)': df_consecutive[df_consecutive['ll_count'] >= 5],
}

print(f"\n{'그룹':20s} {'발생수':>10s} {'평균 반등':>12s} {'RSI':>10s} {'MACD Hist':>12s} {'Volume':>10s}")
print("-" * 80)

for group_name, group_data in groups.items():
    if len(group_data) > 0:
        freq = len(group_data)
        avg_bounce = group_data['bounce_pct'].mean()
        avg_rsi = group_data['last_ll_rsi'].mean()
        avg_macd = group_data['last_ll_macd_hist'].mean()
        avg_vol = group_data['last_ll_volume_ratio'].mean()
        
        print(f"{group_name:20s} {freq:10d} {avg_bounce:>+11.2f}% {avg_rsi:>10.2f} {avg_macd:>12.2f} {avg_vol:>10.2f}")

# ============================================================
# 저장
# ============================================================

df_consecutive.to_csv('consecutive_LL_analysis.csv', index=False)

print("\n💾 결과 저장: consecutive_LL_analysis.csv")

print("\n" + "=" * 70)
print("✅ 연속 LL 패턴 분석 완료!")
print("=" * 70)
