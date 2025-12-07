#!/usr/bin/env python3
"""
하락 종료 시점 실시간 탐지 - 인디케이터 패턴 분석

목표: L값이 확정되기 전에, 실시간으로 바닥을 포착할 수 있는
      인디케이터 조합 패턴을 찾는다.

방법: 
1. 과거 L값(Swing Low) 확정 시점을 찾는다
2. L값 확정 이전 N개 캔들의 인디케이터 패턴을 분석
3. 바닥 신호를 가장 빨리, 정확하게 포착하는 패턴 도출
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("하락 종료 시점 실시간 탐지 - 인디케이터 패턴 분석")
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
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # Williams %R
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
    # Moving Averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # Volume
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 변화율 (모멘텀)
    df['rsi_change'] = df['rsi'].diff()
    df['macd_hist_change'] = df['macd_hist'].diff()
    df['stoch_k_change'] = df['stoch_k'].diff()
    df['bb_position_change'] = df['bb_position'].diff()
    
    # 가격 변화율
    df['price_change_1'] = df['close'].pct_change(1) * 100
    df['price_change_3'] = df['close'].pct_change(3) * 100
    df['price_change_5'] = df['close'].pct_change(5) * 100
    
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

print("\n🔍 Swing Low (L값) 탐지 중...")

order = 10
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]

print(f"✅ Swing Low: {len(low_indices):,}개")

# ============================================================
# L값 형성 과정의 인디케이터 패턴 추출
# ============================================================

print("\n🔍 L값 형성 과정 인디케이터 패턴 분석 중...")

LOOKBACK = 15  # L값 이전 15개 캔들 분석

bottom_patterns = []

for low_idx in low_indices:
    if low_idx < LOOKBACK + 50:  # 충분한 과거 데이터 필요
        continue
    
    # L값 시점
    l_point = df.iloc[low_idx]
    
    # L값 이전 N개 캔들
    before_candles = df.iloc[low_idx - LOOKBACK : low_idx]
    
    # 각 캔들별 인디케이터 값 저장
    pattern = {
        'l_datetime': l_point['datetime'],
        'l_price': l_point['low'],
        'l_rsi': l_point['rsi'],
        'l_macd_hist': l_point['macd_hist'],
        'l_stoch_k': l_point['stoch_k'],
        'l_bb_position': l_point['bb_position'],
        'l_cci': l_point['cci'],
        'l_williams_r': l_point['williams_r'],
        'l_volume_ratio': l_point['volume_ratio'],
        'l_atr_pct': l_point['atr_pct']
    }
    
    # L값 이전 캔들들의 패턴
    for i in range(1, min(11, LOOKBACK + 1)):  # 최근 10개 캔들
        candle = before_candles.iloc[-i]
        pattern[f'b{i}_rsi'] = candle['rsi']
        pattern[f'b{i}_macd_hist'] = candle['macd_hist']
        pattern[f'b{i}_stoch_k'] = candle['stoch_k']
        pattern[f'b{i}_bb_position'] = candle['bb_position']
        pattern[f'b{i}_volume_ratio'] = candle['volume_ratio']
        pattern[f'b{i}_rsi_change'] = candle['rsi_change']
        pattern[f'b{i}_macd_hist_change'] = candle['macd_hist_change']
    
    # RSI 다이버전스 체크 (최근 5개 캔들)
    recent_5 = before_candles.tail(5)
    if len(recent_5) >= 3:
        price_trend = recent_5['low'].iloc[-1] < recent_5['low'].iloc[0]
        rsi_trend = recent_5['rsi'].iloc[-1] > recent_5['rsi'].iloc[0]
        pattern['rsi_divergence'] = 1 if (price_trend and rsi_trend) else 0
    else:
        pattern['rsi_divergence'] = 0
    
    # MACD Histogram 다이버전스
    if len(recent_5) >= 3:
        price_trend = recent_5['low'].iloc[-1] < recent_5['low'].iloc[0]
        macd_trend = recent_5['macd_hist'].iloc[-1] > recent_5['macd_hist'].iloc[0]
        pattern['macd_divergence'] = 1 if (price_trend and macd_trend) else 0
    else:
        pattern['macd_divergence'] = 0
    
    # 연속 하락 캔들 수
    consecutive_down = 0
    for candle in before_candles.iloc[::-1].itertuples():
        if candle.close < candle.open:
            consecutive_down += 1
        else:
            break
    pattern['consecutive_down'] = consecutive_down
    
    # RSI 최저점 도달 (30 이하)
    pattern['rsi_oversold_count'] = len(before_candles[before_candles['rsi'] < 30])
    
    # BB 하단 터치 횟수
    pattern['bb_lower_touch_count'] = len(before_candles[before_candles['bb_position'] < 0.1])
    
    # L값 이후 반등 성공 여부 (정답 레이블)
    if low_idx + 20 < len(df):
        future_20 = df.iloc[low_idx + 1 : low_idx + 21]
        max_gain = ((future_20['high'].max() - l_point['low']) / l_point['low']) * 100
        final_price = future_20.iloc[-1]['close']
        final_gain = ((final_price - l_point['low']) / l_point['low']) * 100
        
        pattern['future_max_gain'] = max_gain
        pattern['future_final_gain'] = final_gain
        pattern['success'] = 1 if final_gain > 1.5 else 0  # 1.5% 이상 반등하면 성공
    else:
        continue
    
    bottom_patterns.append(pattern)

df_patterns = pd.DataFrame(bottom_patterns)

print(f"✅ 분석 완료: {len(df_patterns):,}개 바닥 패턴")

# ============================================================
# 성공 vs 실패 패턴 비교
# ============================================================

print("\n" + "=" * 70)
print("📊 성공(반등) vs 실패(추가하락) 패턴 비교")
print("=" * 70)

success = df_patterns[df_patterns['success'] == 1]
fail = df_patterns[df_patterns['success'] == 0]

print(f"\n성공 (반등 >1.5%): {len(success)}개 ({len(success)/len(df_patterns)*100:.1f}%)")
print(f"실패 (반등 <1.5%): {len(fail)}개 ({len(fail)/len(df_patterns)*100:.1f}%)")

# L값 시점 지표 비교
l_indicators = [
    ('L값 RSI', 'l_rsi'),
    ('L값 MACD Hist', 'l_macd_hist'),
    ('L값 Stoch K', 'l_stoch_k'),
    ('L값 BB Position', 'l_bb_position'),
    ('L값 CCI', 'l_cci'),
    ('L값 Williams %R', 'l_williams_r'),
    ('L값 Volume Ratio', 'l_volume_ratio'),
    ('L값 ATR %', 'l_atr_pct'),
]

print(f"\n{'지표':25s} {'성공':>12s} {'실패':>12s} {'차이':>12s}")
print("-" * 70)
for name, col in l_indicators:
    success_val = success[col].mean()
    fail_val = fail[col].mean()
    diff = success_val - fail_val
    print(f"{name:25s} {success_val:>12.2f} {fail_val:>12.2f} {diff:>+12.2f}")

# L값 이전 캔들 패턴 비교
print(f"\n{'패턴 특성':25s} {'성공':>12s} {'실패':>12s} {'차이':>12s}")
print("-" * 70)

pattern_features = [
    ('RSI 다이버전스', 'rsi_divergence'),
    ('MACD 다이버전스', 'macd_divergence'),
    ('연속 하락 캔들', 'consecutive_down'),
    ('RSI 과매도 횟수', 'rsi_oversold_count'),
    ('BB 하단 터치', 'bb_lower_touch_count'),
]

for name, col in pattern_features:
    success_val = success[col].mean()
    fail_val = fail[col].mean()
    diff = success_val - fail_val
    print(f"{name:25s} {success_val:>12.2f} {fail_val:>12.2f} {diff:>+12.2f}")

# ============================================================
# 최근 N개 캔들의 지표 변화 패턴
# ============================================================

print("\n" + "=" * 70)
print("📊 L값 직전 캔들별 지표 차이 (성공 - 실패)")
print("=" * 70)

print(f"\n{'캔들':>6s} {'RSI':>10s} {'MACD Hist':>12s} {'Stoch K':>10s} {'BB Pos':>10s} {'Vol':>10s}")
print("-" * 70)

for i in range(1, 11):
    rsi_diff = success[f'b{i}_rsi'].mean() - fail[f'b{i}_rsi'].mean()
    macd_diff = success[f'b{i}_macd_hist'].mean() - fail[f'b{i}_macd_hist'].mean()
    stoch_diff = success[f'b{i}_stoch_k'].mean() - fail[f'b{i}_stoch_k'].mean()
    bb_diff = success[f'b{i}_bb_position'].mean() - fail[f'b{i}_bb_position'].mean()
    vol_diff = success[f'b{i}_volume_ratio'].mean() - fail[f'b{i}_volume_ratio'].mean()
    
    print(f"L-{i:>2d}봉 {rsi_diff:>+10.2f} {macd_diff:>+12.2f} {stoch_diff:>+10.2f} {bb_diff:>+10.2f} {vol_diff:>+10.2f}")

# ============================================================
# 실시간 진입 신호 조건 도출
# ============================================================

print("\n" + "=" * 70)
print("📊 실시간 바닥 포착 조건 (성공 패턴 특징)")
print("=" * 70)

# 성공 패턴의 조건별 비율
conditions = {
    'RSI < 30': success['l_rsi'] < 30,
    'RSI < 35': success['l_rsi'] < 35,
    'RSI < 40': success['l_rsi'] < 40,
    'MACD Hist < -30': success['l_macd_hist'] < -30,
    'MACD Hist < -50': success['l_macd_hist'] < -50,
    'Stoch K < 20': success['l_stoch_k'] < 20,
    'Stoch K < 30': success['l_stoch_k'] < 30,
    'BB Position < 0.2': success['l_bb_position'] < 0.2,
    'BB Position < 0.3': success['l_bb_position'] < 0.3,
    'CCI < -100': success['l_cci'] < -100,
    'Williams %R < -80': success['l_williams_r'] < -80,
    'Volume Ratio > 1.5': success['l_volume_ratio'] > 1.5,
    'Volume Ratio > 2.0': success['l_volume_ratio'] > 2.0,
    'RSI 다이버전스': success['rsi_divergence'] == 1,
    'MACD 다이버전스': success['macd_divergence'] == 1,
    '연속 하락 >= 3': success['consecutive_down'] >= 3,
}

print("\n성공 패턴에서 조건 충족 비율:")
for condition_name, condition_mask in conditions.items():
    pct = condition_mask.sum() / len(success) * 100
    print(f"   {condition_name:25s}: {pct:5.1f}%")

# ============================================================
# 최적 조합 조건 찾기
# ============================================================

print("\n" + "=" * 70)
print("📊 조건 조합별 성공률")
print("=" * 70)

# 여러 조건 조합 테스트
combinations = [
    {
        'name': '조합1: RSI+MACD+BB',
        'condition': lambda row: (row['l_rsi'] < 35) & (row['l_macd_hist'] < -30) & (row['l_bb_position'] < 0.3)
    },
    {
        'name': '조합2: RSI+Stoch+Volume',
        'condition': lambda row: (row['l_rsi'] < 35) & (row['l_stoch_k'] < 30) & (row['l_volume_ratio'] > 1.5)
    },
    {
        'name': '조합3: RSI다이버전스+Volume',
        'condition': lambda row: (row['rsi_divergence'] == 1) & (row['l_volume_ratio'] > 1.5)
    },
    {
        'name': '조합4: 과매도 종합',
        'condition': lambda row: (row['l_rsi'] < 35) & (row['l_stoch_k'] < 30) & (row['l_bb_position'] < 0.3) & (row['l_cci'] < -100)
    },
    {
        'name': '조합5: 다이버전스+과매도',
        'condition': lambda row: (row['rsi_divergence'] == 1) & (row['l_rsi'] < 40) & (row['l_volume_ratio'] > 1.5)
    },
]

print(f"\n{'조합':30s} {'발생수':>10s} {'성공률':>10s} {'평균수익':>12s}")
print("-" * 70)

for combo in combinations:
    mask = df_patterns.apply(combo['condition'], axis=1)
    subset = df_patterns[mask]
    
    if len(subset) > 0:
        success_rate = (subset['success'].sum() / len(subset)) * 100
        avg_gain = subset['future_final_gain'].mean()
        print(f"{combo['name']:30s} {len(subset):>10d} {success_rate:>9.1f}% {avg_gain:>+11.2f}%")
    else:
        print(f"{combo['name']:30s} {0:>10d} {'N/A':>10s} {'N/A':>12s}")

# ============================================================
# 저장
# ============================================================

df_patterns.to_csv('bottom_indicator_patterns.csv', index=False)

print("\n💾 결과 저장: bottom_indicator_patterns.csv")

print("\n" + "=" * 70)
print("✅ 하락 종료 시점 인디케이터 패턴 분석 완료!")
print("=" * 70)
