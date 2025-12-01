#!/usr/bin/env python3
"""
하락 후 반등 패턴 탐지
- LL (Lower Low) 연속 → HL (Higher Low) 전환 시점
- LH 연속 → HH 전환 시점
- 지표 다이버전스
- V자 반등 패턴
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("하락 → 반등 패턴 탐지")
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
# Swing Point 탐지
# ============================================================

print("\n🔍 Swing High/Low 탐지 중...")

order = 10
high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]

swing_highs = df.iloc[high_indices].copy()
swing_highs['swing_price'] = swing_highs['high']
swing_highs['swing_type'] = 'HIGH'

swing_lows = df.iloc[low_indices].copy()
swing_lows['swing_price'] = swing_lows['low']
swing_lows['swing_type'] = 'LOW'

# 모든 Swing Point 병합 및 시간순 정렬
all_swings = pd.concat([swing_highs, swing_lows]).sort_values('datetime').reset_index(drop=True)

print(f"✅ 총 Swing Point: {len(all_swings):,}개")

# ============================================================
# High/Low 패턴 분류
# ============================================================

print("\n🔍 HH/LH/HL/LL 패턴 분류 중...")

# High 패턴
highs = all_swings[all_swings['swing_type'] == 'HIGH'].copy()
highs['prev_price'] = highs['swing_price'].shift(1)
highs['pattern'] = np.where(highs['swing_price'] > highs['prev_price'], 'HH', 'LH')

# Low 패턴
lows = all_swings[all_swings['swing_type'] == 'LOW'].copy()
lows['prev_price'] = lows['swing_price'].shift(1)
lows['pattern'] = np.where(lows['swing_price'] > lows['prev_price'], 'HL', 'LL')

print(f"✅ High 패턴: {len(highs):,}개")
print(f"✅ Low 패턴: {len(lows):,}개")

# ============================================================
# 반등 패턴 탐지
# ============================================================

print("\n" + "=" * 60)
print("📊 반등 패턴 탐지")
print("=" * 60)

reversals = []

# 패턴 1: LL 연속 → HL 전환
print("\n[패턴 1] LL 연속 → HL 전환")
lows = lows.dropna(subset=['prev_price']).reset_index(drop=True)

ll_count = 0
for i in range(len(lows)):
    if lows.loc[i, 'pattern'] == 'LL':
        ll_count += 1
    elif lows.loc[i, 'pattern'] == 'HL' and ll_count >= 2:
        # 반등 시작점
        reversal_point = lows.loc[i]
        reversals.append({
            'datetime': reversal_point['datetime'],
            'price': reversal_point['swing_price'],
            'pattern_type': 'LL→HL',
            'll_count': ll_count,
            'rsi': reversal_point['rsi'],
            'macd_hist': reversal_point['macd_hist'],
            'stoch_k': reversal_point['stoch_k'],
            'bb_position': reversal_point['bb_position'],
            'volume_ratio': reversal_point['volume_ratio']
        })
        ll_count = 0
    else:
        ll_count = 0

print(f"✅ LL→HL 반등: {len([r for r in reversals if r['pattern_type'] == 'LL→HL'])}개")

# 패턴 2: LH 연속 → HH 전환
print("\n[패턴 2] LH 연속 → HH 전환")
highs = highs.dropna(subset=['prev_price']).reset_index(drop=True)

lh_count = 0
for i in range(len(highs)):
    if highs.loc[i, 'pattern'] == 'LH':
        lh_count += 1
    elif highs.loc[i, 'pattern'] == 'HH' and lh_count >= 2:
        # 상승 전환점
        reversal_point = highs.loc[i]
        reversals.append({
            'datetime': reversal_point['datetime'],
            'price': reversal_point['swing_price'],
            'pattern_type': 'LH→HH',
            'lh_count': lh_count,
            'rsi': reversal_point['rsi'],
            'macd_hist': reversal_point['macd_hist'],
            'stoch_k': reversal_point['stoch_k'],
            'bb_position': reversal_point['bb_position'],
            'volume_ratio': reversal_point['volume_ratio']
        })
        lh_count = 0
    else:
        lh_count = 0

print(f"✅ LH→HH 반등: {len([r for r in reversals if r['pattern_type'] == 'LH→HH'])}개")

# 패턴 3: RSI/MACD 다이버전스
print("\n[패턴 3] RSI 다이버전스 (가격↓ RSI↑)")

divergence_count = 0
for i in range(2, len(lows)):
    # 최근 3개의 Low 비교
    low1 = lows.loc[i-2]
    low2 = lows.loc[i-1]
    low3 = lows.loc[i]
    
    # 가격은 하락, RSI는 상승
    price_falling = low1['swing_price'] > low2['swing_price'] > low3['swing_price']
    rsi_rising = low1['rsi'] < low2['rsi'] < low3['rsi']
    
    if price_falling and rsi_rising:
        reversals.append({
            'datetime': low3['datetime'],
            'price': low3['swing_price'],
            'pattern_type': 'RSI_Divergence',
            'rsi': low3['rsi'],
            'macd_hist': low3['macd_hist'],
            'stoch_k': low3['stoch_k'],
            'bb_position': low3['bb_position'],
            'volume_ratio': low3['volume_ratio']
        })
        divergence_count += 1

print(f"✅ RSI 다이버전스: {divergence_count}개")

# 패턴 4: 과매도 + 급반등 (V자)
print("\n[패턴 4] 과매도 V자 반등 (RSI<30 → RSI>50)")

v_reversal_count = 0
for i in range(1, len(lows)):
    prev_low = lows.loc[i-1]
    curr_low = lows.loc[i]
    
    # RSI 30 이하에서 50 이상으로 급등
    if prev_low['rsi'] < 30 and curr_low['rsi'] > 50:
        reversals.append({
            'datetime': curr_low['datetime'],
            'price': curr_low['swing_price'],
            'pattern_type': 'V_Reversal',
            'rsi': curr_low['rsi'],
            'macd_hist': curr_low['macd_hist'],
            'stoch_k': curr_low['stoch_k'],
            'bb_position': curr_low['bb_position'],
            'volume_ratio': curr_low['volume_ratio']
        })
        v_reversal_count += 1

print(f"✅ V자 반등: {v_reversal_count}개")

# DataFrame으로 변환
df_reversals = pd.DataFrame(reversals)

print(f"\n📊 총 반등 패턴: {len(df_reversals)}개")

# ============================================================
# 반등 패턴별 지표 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 반등 패턴별 지표 평균")
print("=" * 60)

for pattern in df_reversals['pattern_type'].unique():
    subset = df_reversals[df_reversals['pattern_type'] == pattern]
    print(f"\n[{pattern}] ({len(subset)}개)")
    print(f"   RSI 평균:           {subset['rsi'].mean():.2f}")
    print(f"   MACD Hist 평균:     {subset['macd_hist'].mean():.2f}")
    print(f"   Stochastic K 평균:  {subset['stoch_k'].mean():.2f}")
    print(f"   BB Position 평균:   {subset['bb_position'].mean():.2f}")
    print(f"   Volume Ratio 평균:  {subset['volume_ratio'].mean():.2f}")

# ============================================================
# 반등 성공률 분석 (반등 후 N시간 수익률)
# ============================================================

print("\n" + "=" * 60)
print("📊 반등 후 수익률 분석")
print("=" * 60)

# 각 반등 패턴 이후 가격 변화 추적
for hours in [4, 12, 24, 48]:
    print(f"\n[반등 후 {hours}시간 수익률]")
    
    success_rates = {}
    
    for pattern in df_reversals['pattern_type'].unique():
        subset = df_reversals[df_reversals['pattern_type'] == pattern]
        
        profits = []
        for _, reversal in subset.iterrows():
            # 반등 시점 찾기
            reversal_idx = df[df['datetime'] == reversal['datetime']].index
            if len(reversal_idx) == 0:
                continue
            
            reversal_idx = reversal_idx[0]
            future_idx = min(reversal_idx + hours * 4, len(df) - 1)  # 15분봉이므로 *4
            
            entry_price = reversal['price']
            exit_price = df.loc[future_idx, 'close']
            profit_pct = (exit_price - entry_price) / entry_price * 100
            profits.append(profit_pct)
        
        if len(profits) > 0:
            avg_profit = np.mean(profits)
            win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
            success_rates[pattern] = {
                'avg_profit': avg_profit,
                'win_rate': win_rate,
                'count': len(profits)
            }
    
    # 결과 출력
    for pattern, stats in sorted(success_rates.items(), key=lambda x: x[1]['avg_profit'], reverse=True):
        print(f"   {pattern:20s}: 평균 {stats['avg_profit']:+6.2f}% | 승률 {stats['win_rate']:5.1f}% | {stats['count']}회")

# ============================================================
# 반등 패턴 조합 조건
# ============================================================

print("\n" + "=" * 60)
print("📊 최적 반등 진입 조건 (조합)")
print("=" * 60)

# 조건 조합 테스트
conditions = {
    'RSI < 40': df_reversals['rsi'] < 40,
    'MACD Hist < 0': df_reversals['macd_hist'] < 0,
    'Stoch K < 30': df_reversals['stoch_k'] < 30,
    'BB Position < 0.3': df_reversals['bb_position'] < 0.3,
    'Volume Ratio > 1.5': df_reversals['volume_ratio'] > 1.5,
}

print("\n조건별 반등 패턴 비율:")
for condition_name, condition_mask in conditions.items():
    count = condition_mask.sum()
    pct = count / len(df_reversals) * 100
    print(f"   {condition_name:25s}: {count:4d}개 ({pct:5.1f}%)")

# ============================================================
# 저장
# ============================================================

df_reversals.to_csv('reversal_patterns_analysis.csv', index=False)

print("\n💾 결과 저장: reversal_patterns_analysis.csv")

print("\n" + "=" * 60)
print("✅ 반등 패턴 탐지 완료!")
print("=" * 60)
