#!/usr/bin/env python3
"""
모든 L값 패턴 종합 추론

목표: 지금까지 분석한 모든 L값 특징을 종합하여
      현재 시점에서 다음 L의 위치, 시점, 패턴을 추론
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("모든 L값 패턴 종합 추론")
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
    
    # Williams %R
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
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
print(f"✅ 총 캔들 수: {len(df):,}개")

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
# 모든 L값에 대한 종합 분석
# ============================================================

print("\n🔍 모든 L값 패턴 종합 분석 중...")

all_l_analysis = []

for i in range(1, len(l_values)):
    prev_l = l_values.iloc[i - 1]
    curr_l = l_values.iloc[i]
    
    # 연속 LL 카운트
    consecutive_ll = 0
    for j in range(i, 0, -1):
        if j > 0 and l_values.iloc[j]['l_price'] < l_values.iloc[j-1]['l_price']:
            consecutive_ll += 1
        else:
            break
    
    # 현재 패턴
    if curr_l['l_price'] < prev_l['l_price']:
        curr_pattern = 'LL'
    else:
        curr_pattern = 'HL'
    
    # 가격 변화
    price_change_pct = ((curr_l['l_price'] - prev_l['l_price']) / prev_l['l_price']) * 100
    
    # 시간 간격
    time_diff_hours = (curr_l['datetime'] - prev_l['datetime']).total_seconds() / 3600
    
    # L값 이후 반등 분석 (다음 20개 캔들)
    curr_idx = df[df['datetime'] == curr_l['datetime']].index[0]
    if curr_idx + 20 < len(df):
        future_20 = df.iloc[curr_idx + 1 : curr_idx + 21]
        max_gain_pct = ((future_20['high'].max() - curr_l['l_price']) / curr_l['l_price']) * 100
        final_price = future_20.iloc[-1]['close']
        final_gain_pct = ((final_price - curr_l['l_price']) / curr_l['l_price']) * 100
        
        # 첫 양봉까지 시간
        first_green_idx = None
        for idx, candle in future_20.iterrows():
            if candle['close'] > candle['open']:
                first_green_idx = idx - curr_idx - 1
                break
        
        first_green_candles = first_green_idx if first_green_idx is not None else 20
    else:
        max_gain_pct = np.nan
        final_gain_pct = np.nan
        first_green_candles = np.nan
    
    # RSI 다이버전스 체크
    if i >= 3:
        last_3_l = l_values.iloc[i-2:i+1]
        price_declining = (last_3_l['l_price'].iloc[0] > last_3_l['l_price'].iloc[1] > 
                          last_3_l['l_price'].iloc[2])
        rsi_rising = (last_3_l['rsi'].iloc[0] < last_3_l['rsi'].iloc[1] < 
                     last_3_l['rsi'].iloc[2])
        rsi_divergence = 1 if (price_declining and rsi_rising) else 0
    else:
        rsi_divergence = 0
    
    # MACD 다이버전스
    if i >= 3:
        macd_rising = (last_3_l['macd_hist'].iloc[0] < last_3_l['macd_hist'].iloc[1] < 
                      last_3_l['macd_hist'].iloc[2])
        macd_divergence = 1 if (price_declining and macd_rising) else 0
    else:
        macd_divergence = 0
    
    all_l_analysis.append({
        # 기본 정보
        'datetime': curr_l['datetime'],
        'l_price': curr_l['l_price'],
        'pattern': curr_pattern,
        'consecutive_ll': consecutive_ll,
        
        # 이전 L 대비 변화
        'price_change_pct': price_change_pct,
        'time_from_prev_hours': time_diff_hours,
        
        # 현재 L의 지표
        'rsi': curr_l['rsi'],
        'macd_hist': curr_l['macd_hist'],
        'stoch_k': curr_l['stoch_k'],
        'bb_position': curr_l['bb_position'],
        'cci': curr_l['cci'],
        'williams_r': curr_l['williams_r'],
        'volume_ratio': curr_l['volume_ratio'],
        'atr_pct': curr_l['atr_pct'],
        
        # 다이버전스
        'rsi_divergence': rsi_divergence,
        'macd_divergence': macd_divergence,
        
        # 반등 성과
        'max_gain_20': max_gain_pct,
        'final_gain_20': final_gain_pct,
        'first_green_candles': first_green_candles,
    })

df_all_l = pd.DataFrame(all_l_analysis)

print(f"✅ 분석 완료: {len(df_all_l):,}개 L값")

# ============================================================
# L값 특징 요약
# ============================================================

print("\n" + "=" * 70)
print("📊 전체 L값 통계 요약")
print("=" * 70)

print(f"\n패턴 분포:")
pattern_dist = df_all_l['pattern'].value_counts()
for pattern, count in pattern_dist.items():
    pct = count / len(df_all_l) * 100
    print(f"   {pattern}: {count:,}개 ({pct:.1f}%)")

print(f"\n연속 LL 분포:")
ll_dist = df_all_l[df_all_l['pattern'] == 'LL']['consecutive_ll'].value_counts().sort_index()
for ll_count, count in ll_dist.items():
    pct = count / len(df_all_l[df_all_l['pattern'] == 'LL']) * 100
    print(f"   {ll_count}번 연속: {count:,}개 ({pct:.1f}%)")

# ============================================================
# L값 유형 분류 및 특징
# ============================================================

print("\n" + "=" * 70)
print("📊 L값 유형별 특징")
print("=" * 70)

# 유형 정의
def classify_l_type(row):
    """L값 유형 분류"""
    if row['pattern'] == 'HL':
        return 'HL_반등'
    elif row['consecutive_ll'] == 1:
        return 'LL_1번'
    elif row['consecutive_ll'] == 2:
        return 'LL_2번'
    elif row['consecutive_ll'] in [3, 4]:
        return 'LL_3~4번'
    else:
        return 'LL_5번+'

df_all_l['l_type'] = df_all_l.apply(classify_l_type, axis=1)

type_summary = df_all_l.groupby('l_type').agg({
    'datetime': 'count',
    'rsi': 'mean',
    'macd_hist': 'mean',
    'volume_ratio': 'mean',
    'atr_pct': 'mean',
    'max_gain_20': 'mean',
    'final_gain_20': 'mean',
}).round(2)

type_summary.columns = ['발생수', 'RSI평균', 'MACD Hist평균', 'Volume평균', 'ATR%평균', '최대반등', '최종반등']
print("\n" + str(type_summary))

# ============================================================
# 반등 확률 및 수익 기대치
# ============================================================

print("\n" + "=" * 70)
print("📊 L값 유형별 반등 확률 및 수익 기대치")
print("=" * 70)

success_threshold = 1.5  # 1.5% 이상 반등을 성공으로 정의

print(f"\n{'유형':15s} {'발생수':>8s} {'성공률':>10s} {'평균반등':>12s} {'첫양봉':>10s}")
print("-" * 60)

for l_type in df_all_l['l_type'].unique():
    subset = df_all_l[df_all_l['l_type'] == l_type]
    count = len(subset)
    success_rate = (subset['final_gain_20'] > success_threshold).sum() / count * 100
    avg_bounce = subset['final_gain_20'].mean()
    avg_first_green = subset['first_green_candles'].mean()
    
    print(f"{l_type:15s} {count:8d} {success_rate:9.1f}% {avg_bounce:>+11.2f}% {avg_first_green:9.1f}봉")

# ============================================================
# 진입 조건별 성과
# ============================================================

print("\n" + "=" * 70)
print("📊 진입 조건별 성과 (전체 L값 기준)")
print("=" * 70)

entry_conditions = [
    ('RSI < 30', df_all_l['rsi'] < 30),
    ('RSI < 35', df_all_l['rsi'] < 35),
    ('MACD Hist < -50', df_all_l['macd_hist'] < -50),
    ('Volume > 2.5배', df_all_l['volume_ratio'] > 2.5),
    ('Volume > 3.0배', df_all_l['volume_ratio'] > 3.0),
    ('BB Position < 0.1', df_all_l['bb_position'] < 0.1),
    ('CCI < -120', df_all_l['cci'] < -120),
    ('ATR % > 0.5', df_all_l['atr_pct'] > 0.5),
    ('RSI 다이버전스', df_all_l['rsi_divergence'] == 1),
    ('MACD 다이버전스', df_all_l['macd_divergence'] == 1),
    ('연속 LL 3~4번', df_all_l['consecutive_ll'].isin([3, 4])),
]

print(f"\n{'조건':25s} {'해당수':>8s} {'성공률':>10s} {'평균반등':>12s}")
print("-" * 60)

for condition_name, condition_mask in entry_conditions:
    subset = df_all_l[condition_mask]
    if len(subset) > 0:
        count = len(subset)
        success_rate = (subset['final_gain_20'] > success_threshold).sum() / count * 100
        avg_bounce = subset['final_gain_20'].mean()
        print(f"{condition_name:25s} {count:8d} {success_rate:9.1f}% {avg_bounce:>+11.2f}%")

# ============================================================
# 최적 조합 조건
# ============================================================

print("\n" + "=" * 70)
print("📊 최적 진입 조건 조합")
print("=" * 70)

# 조합 조건 테스트
combinations = [
    {
        'name': '조합1: 3~4번 LL + 과매도',
        'mask': (df_all_l['consecutive_ll'].isin([3, 4])) & 
                (df_all_l['rsi'] < 35) & 
                (df_all_l['volume_ratio'] > 2.5)
    },
    {
        'name': '조합2: 다이버전스 + Volume',
        'mask': (df_all_l['rsi_divergence'] == 1) & 
                (df_all_l['volume_ratio'] > 2.0)
    },
    {
        'name': '조합3: 극과매도 종합',
        'mask': (df_all_l['rsi'] < 30) & 
                (df_all_l['macd_hist'] < -50) & 
                (df_all_l['bb_position'] < 0.1) & 
                (df_all_l['volume_ratio'] > 3.0)
    },
    {
        'name': '조합4: 5번+ LL + 극한',
        'mask': (df_all_l['consecutive_ll'] >= 5) & 
                (df_all_l['rsi'] < 35)
    },
    {
        'name': '조합5: 종합 시그널',
        'mask': ((df_all_l['consecutive_ll'] >= 3) | (df_all_l['rsi_divergence'] == 1)) &
                (df_all_l['rsi'] < 35) & 
                (df_all_l['volume_ratio'] > 2.5) &
                (df_all_l['bb_position'] < 0.15)
    },
]

print(f"\n{'조합':30s} {'해당수':>8s} {'성공률':>10s} {'평균반등':>12s} {'최대반등':>12s}")
print("-" * 80)

for combo in combinations:
    subset = df_all_l[combo['mask']]
    if len(subset) > 0:
        count = len(subset)
        success_rate = (subset['final_gain_20'] > success_threshold).sum() / count * 100
        avg_bounce = subset['final_gain_20'].mean()
        max_bounce = subset['max_gain_20'].mean()
        print(f"{combo['name']:30s} {count:8d} {success_rate:9.1f}% {avg_bounce:>+11.2f}% {max_bounce:>+11.2f}%")

# ============================================================
# 실시간 적용 가능한 조건
# ============================================================

print("\n" + "=" * 70)
print("📊 실시간 적용 체크리스트 (Look-Ahead Bias 없음)")
print("=" * 70)

print("""
실시간 L값 판단 불가능 (10개 봉 확정 필요)
→ 대신 "바닥 형성 징후"를 실시간 체크!

실시간 체크 가능 지표:
✅ RSI < 35
✅ MACD Hist < -50
✅ Stochastic K < 30
✅ BB Position < 0.1
✅ CCI < -120
✅ Volume Ratio > 3.0
✅ ATR % > 0.5
✅ Williams %R < -80

실시간 패턴 추적:
✅ 최근 3개 L의 연속 하락 (LL 카운트)
✅ RSI 다이버전스 (가격↓ RSI↑)
✅ MACD 다이버전스 (가격↓ MACD↑)

진입 전략:
1. 위 조건 5개 이상 충족 시 "바닥 가능성" 경고
2. 첫 양봉 출현 시 진입
3. RSI > 35 회복 시 진입 확정
""")

# ============================================================
# 저장
# ============================================================

df_all_l.to_csv('all_L_values_inference.csv', index=False)

print("\n💾 결과 저장: all_L_values_inference.csv")

print("\n" + "=" * 70)
print("✅ 모든 L값 패턴 종합 추론 완료!")
print("=" * 70)
