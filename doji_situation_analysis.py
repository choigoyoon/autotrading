"""
하이킨아시 도지 상황 분석
도지가 나온 상황으로 이후 행동 예측
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("도지 상황 분석 - 이후 행동 예측")
print("=" * 80)
print()

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv('btcusdt_1h_raw.csv', parse_dates=['datetime'])
print(f"  1시간봉: {len(df):,}개")
print()

# 하이킨아시 변환
print("하이킨아시 변환...")
df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['ha_open'] = 0.0
df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

for i in range(1, len(df)):
    df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2

df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

# 도지 정의
df['ha_range'] = df['ha_high'] - df['ha_low']
df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
df['ha_body_pct'] = df['ha_body'] / df['ha_range']
df['is_doji'] = (df['ha_body_pct'] < 0.1) & (df['ha_range'] > 0)

print(f"  도지: {df['is_doji'].sum():,}개")
print()

# 기술 지표 계산
print("기술 지표 계산 중...")

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'], 14)

# EMA
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

# 거래량 이동평균
df['volume_ma20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma20']

# ATR (변동성)
df['tr'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
df['atr14'] = df['tr'].rolling(window=14).mean()
df['atr_pct'] = df['atr14'] / df['close'] * 100

print("  완료!")
print()

# 상황 분류
print("상황 분류 중...")

situations = []

for i in df[df['is_doji']].index:
    if i < 50 or i + 5 >= len(df):  # 충분한 이전/이후 데이터 필요
        continue

    curr = df.iloc[i]

    # 1. 추세 판단 (EMA 기준)
    if curr['close'] > curr['ema20'] and curr['ema20'] > curr['ema50']:
        trend = 'uptrend'
    elif curr['close'] < curr['ema20'] and curr['ema20'] < curr['ema50']:
        trend = 'downtrend'
    else:
        trend = 'sideways'

    # 2. 직전 움직임 (최근 5봉)
    prev_5 = df.iloc[i-5:i]
    price_change_5 = (curr['close'] - prev_5.iloc[0]['close']) / prev_5.iloc[0]['close'] * 100

    if price_change_5 > 5:
        recent_move = 'strong_rally'
    elif price_change_5 > 2:
        recent_move = 'rally'
    elif price_change_5 < -5:
        recent_move = 'strong_drop'
    elif price_change_5 < -2:
        recent_move = 'drop'
    else:
        recent_move = 'consolidation'

    # 3. RSI 상태
    if pd.isna(curr['rsi']):
        rsi_state = 'unknown'
    elif curr['rsi'] > 70:
        rsi_state = 'overbought'
    elif curr['rsi'] > 60:
        rsi_state = 'high'
    elif curr['rsi'] < 30:
        rsi_state = 'oversold'
    elif curr['rsi'] < 40:
        rsi_state = 'low'
    else:
        rsi_state = 'neutral'

    # 4. 거래량 상태
    if pd.isna(curr['volume_ratio']):
        volume_state = 'unknown'
    elif curr['volume_ratio'] > 2:
        volume_state = 'very_high'
    elif curr['volume_ratio'] > 1.5:
        volume_state = 'high'
    elif curr['volume_ratio'] < 0.5:
        volume_state = 'very_low'
    elif curr['volume_ratio'] < 0.8:
        volume_state = 'low'
    else:
        volume_state = 'normal'

    # 5. 변동성 상태
    if pd.isna(curr['atr_pct']):
        volatility = 'unknown'
    elif curr['atr_pct'] > 5:
        volatility = 'very_high'
    elif curr['atr_pct'] > 3:
        volatility = 'high'
    elif curr['atr_pct'] < 1:
        volatility = 'very_low'
    elif curr['atr_pct'] < 2:
        volatility = 'low'
    else:
        volatility = 'normal'

    # 이후 움직임 측정 (다음 5봉)
    next_5 = df.iloc[i+1:i+6]

    # 다음 1봉
    next_1 = df.iloc[i+1]
    next_1_gain = (next_1['ha_close'] - curr['ha_close']) / curr['ha_close'] * 100
    next_1_bullish = next_1['ha_close'] > next_1['ha_open']

    # 다음 5봉 최고가/최저가
    max_gain_5 = (next_5['high'].max() - curr['close']) / curr['close'] * 100
    max_loss_5 = (curr['close'] - next_5['low'].min()) / curr['close'] * 100
    final_gain_5 = (next_5.iloc[-1]['close'] - curr['close']) / curr['close'] * 100

    # 다음 5봉 중 상승봉 개수
    bullish_count_5 = sum(next_5['ha_close'] > next_5['ha_open'])

    situations.append({
        'datetime': curr['datetime'],
        'price': curr['close'],
        # 상황
        'trend': trend,
        'recent_move': recent_move,
        'rsi_state': rsi_state,
        'rsi_value': curr['rsi'],
        'volume_state': volume_state,
        'volume_ratio': curr['volume_ratio'],
        'volatility': volatility,
        'atr_pct': curr['atr_pct'],
        # 이후 움직임
        'next_1_gain': next_1_gain,
        'next_1_bullish': next_1_bullish,
        'max_gain_5': max_gain_5,
        'max_loss_5': max_loss_5,
        'final_gain_5': final_gain_5,
        'bullish_count_5': bullish_count_5,
        'next_5_net_bullish': bullish_count_5 >= 3,  # 5개중 3개 이상 상승
    })

sit_df = pd.DataFrame(situations)
print(f"  분석된 도지: {len(sit_df):,}개")
print()

# 상황별 성과 분석
print("=" * 80)
print("상황별 이후 행동 분석")
print("=" * 80)
print()

def analyze_situation(df, condition, name):
    """특정 상황의 성과 분석"""
    subset = df[condition]

    if len(subset) == 0:
        return None

    next_1_bullish_rate = subset['next_1_bullish'].sum() / len(subset) * 100
    next_5_bullish_rate = subset['next_5_net_bullish'].sum() / len(subset) * 100
    avg_next_1_gain = subset['next_1_gain'].mean()
    avg_final_5_gain = subset['final_gain_5'].mean()
    avg_max_gain_5 = subset['max_gain_5'].mean()

    return {
        'name': name,
        'count': len(subset),
        'next_1_bullish_rate': next_1_bullish_rate,
        'next_5_bullish_rate': next_5_bullish_rate,
        'avg_next_1_gain': avg_next_1_gain,
        'avg_final_5_gain': avg_final_5_gain,
        'avg_max_gain_5': avg_max_gain_5,
    }

results = []

# 1. 추세별
print("【1. 추세 맥락】")
for trend_type in ['uptrend', 'downtrend', 'sideways']:
    result = analyze_situation(sit_df, sit_df['trend'] == trend_type, trend_type)
    if result:
        results.append(result)
        print(f"  {result['name']:15s} | {result['count']:4d}개 | "
              f"다음1봉 상승: {result['next_1_bullish_rate']:5.1f}% | "
              f"다음5봉 순상승: {result['next_5_bullish_rate']:5.1f}% | "
              f"평균 gain: {result['avg_next_1_gain']:+.3f}%")
print()

# 2. 직전 움직임별
print("【2. 직전 움직임】")
for move_type in ['strong_rally', 'rally', 'consolidation', 'drop', 'strong_drop']:
    result = analyze_situation(sit_df, sit_df['recent_move'] == move_type, move_type)
    if result:
        results.append(result)
        print(f"  {result['name']:15s} | {result['count']:4d}개 | "
              f"다음1봉 상승: {result['next_1_bullish_rate']:5.1f}% | "
              f"다음5봉 순상승: {result['next_5_bullish_rate']:5.1f}% | "
              f"평균 gain: {result['avg_next_1_gain']:+.3f}%")
print()

# 3. RSI 상태별
print("【3. RSI 상태】")
for rsi_type in ['overbought', 'high', 'neutral', 'low', 'oversold']:
    result = analyze_situation(sit_df, sit_df['rsi_state'] == rsi_type, rsi_type)
    if result:
        results.append(result)
        print(f"  {result['name']:15s} | {result['count']:4d}개 | "
              f"다음1봉 상승: {result['next_1_bullish_rate']:5.1f}% | "
              f"다음5봉 순상승: {result['next_5_bullish_rate']:5.1f}% | "
              f"평균 gain: {result['avg_next_1_gain']:+.3f}%")
print()

# 4. 거래량 상태별
print("【4. 거래량】")
for vol_type in ['very_high', 'high', 'normal', 'low', 'very_low']:
    result = analyze_situation(sit_df, sit_df['volume_state'] == vol_type, vol_type)
    if result:
        results.append(result)
        print(f"  {result['name']:15s} | {result['count']:4d}개 | "
              f"다음1봉 상승: {result['next_1_bullish_rate']:5.1f}% | "
              f"다음5봉 순상승: {result['next_5_bullish_rate']:5.1f}% | "
              f"평균 gain: {result['avg_next_1_gain']:+.3f}%")
print()

# 5. 변동성 상태별
print("【5. 변동성】")
for vol_type in ['very_high', 'high', 'normal', 'low', 'very_low']:
    result = analyze_situation(sit_df, sit_df['volatility'] == vol_type, vol_type)
    if result:
        results.append(result)
        print(f"  {result['name']:15s} | {result['count']:4d}개 | "
              f"다음1봉 상승: {result['next_1_bullish_rate']:5.1f}% | "
              f"다음5봉 순상승: {result['next_5_bullish_rate']:5.1f}% | "
              f"평균 gain: {result['avg_next_1_gain']:+.3f}%")
print()

# 조합 분석
print("=" * 80)
print("고성과 상황 조합 찾기")
print("=" * 80)
print()

# 다음 1봉 상승률 60%+ 찾기
print("【다음 1봉 상승률 60%+ 상황】")
high_performers = []
for trend_type in sit_df['trend'].unique():
    for move_type in sit_df['recent_move'].unique():
        for rsi_type in sit_df['rsi_state'].unique():
            condition = (sit_df['trend'] == trend_type) & \
                       (sit_df['recent_move'] == move_type) & \
                       (sit_df['rsi_state'] == rsi_type)

            result = analyze_situation(sit_df, condition,
                                     f"{trend_type} + {move_type} + {rsi_type}")
            if result and result['count'] >= 10:  # 최소 10개 이상
                if result['next_1_bullish_rate'] >= 60:
                    high_performers.append(result)

# 정렬
high_performers.sort(key=lambda x: x['next_1_bullish_rate'], reverse=True)

for i, result in enumerate(high_performers[:10], 1):
    print(f"{i:2d}. {result['name']}")
    print(f"    건수: {result['count']:4d}개 | "
          f"다음1봉 상승: {result['next_1_bullish_rate']:.1f}% | "
          f"평균 gain: {result['avg_next_1_gain']:+.3f}%")

print()

# 저장
sit_df.to_csv('doji_situation_analysis.csv', index=False)
print("💾 저장: doji_situation_analysis.csv")

print()
print("=" * 80)
print("결론")
print("=" * 80)
print()

# 전체 평균
overall_next_1_rate = sit_df['next_1_bullish'].sum() / len(sit_df) * 100
overall_next_5_rate = sit_df['next_5_net_bullish'].sum() / len(sit_df) * 100

print(f"전체 도지 ({len(sit_df)}개):")
print(f"  다음 1봉 상승률: {overall_next_1_rate:.1f}%")
print(f"  다음 5봉 순상승률: {overall_next_5_rate:.1f}%")
print()

print("💡 핵심 인사이트:")
print("  - 도지 자체는 약 50:50 (랜덤)")
print("  - 상황 맥락이 방향성 결정")
print("  - 특정 상황 조합 = 60%+ 예측 가능")
print()
print("=" * 80)
