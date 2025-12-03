import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('btc_1h_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"전체 데이터: {len(df)} 캔들")
print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"시작가: ${df['close'].iloc[0]:,.0f} → 종가: ${df['close'].iloc[-1]:,.0f}")
print()

# 상위 추세 파악을 위한 지표
# 1. 주간 단위로 고점/저점 추적
df['week'] = df['timestamp'].dt.to_period('W')

weekly = df.groupby('week').agg({
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'open': 'first',
    'timestamp': 'first'
}).reset_index()

print("=== 주간 데이터 ===")
print(f"총 {len(weekly)} 주")
print()

# 계단식 상승/하락 판단
# - 계단식 상승: 고점 갱신 + 저점도 상승 (Higher High + Higher Low)
# - 계단식 하락: 저점 갱신 + 고점도 하락 (Lower Low + Lower High)

weekly['prev_high'] = weekly['high'].shift(1)
weekly['prev_low'] = weekly['low'].shift(1)

weekly['HH'] = weekly['high'] > weekly['prev_high']  # Higher High
weekly['HL'] = weekly['low'] > weekly['prev_low']    # Higher Low
weekly['LH'] = weekly['high'] < weekly['prev_high']  # Lower High
weekly['LL'] = weekly['low'] < weekly['prev_low']    # Lower Low

# 추세 판단
def determine_trend(row):
    if pd.isna(row['prev_high']):
        return 'UNKNOWN'
    if row['HH'] and row['HL']:
        return 'UPTREND'      # 계단식 상승
    elif row['LH'] and row['LL']:
        return 'DOWNTREND'    # 계단식 하락
    elif row['HH'] and row['LL']:
        return 'EXPANSION'    # 확장 (변동성 증가)
    elif row['LH'] and row['HL']:
        return 'CONTRACTION'  # 수렴 (변동성 감소)
    else:
        return 'MIXED'

weekly['trend'] = weekly.apply(determine_trend, axis=1)

print("=== 주간 추세 분포 ===")
print(weekly['trend'].value_counts())
print()

# 연속 추세 구간 찾기
def find_trend_phases(weekly_df):
    phases = []
    current_trend = None
    start_idx = 0
    
    for i, row in weekly_df.iterrows():
        if row['trend'] != current_trend:
            if current_trend is not None and current_trend in ['UPTREND', 'DOWNTREND']:
                phases.append({
                    'trend': current_trend,
                    'start_week': weekly_df.iloc[start_idx]['week'],
                    'end_week': weekly_df.iloc[i-1]['week'] if i > 0 else weekly_df.iloc[start_idx]['week'],
                    'start_date': weekly_df.iloc[start_idx]['timestamp'],
                    'end_date': weekly_df.iloc[min(i, len(weekly_df)-1)]['timestamp'],
                    'duration_weeks': i - start_idx,
                    'start_price': weekly_df.iloc[start_idx]['open'],
                    'end_price': weekly_df.iloc[min(i-1, len(weekly_df)-1)]['close']
                })
            current_trend = row['trend']
            start_idx = i
    
    return phases

phases = find_trend_phases(weekly)

print("=== 주요 추세 구간 (2주 이상 지속) ===")
for p in phases:
    if p['duration_weeks'] >= 2:
        change = ((p['end_price'] - p['start_price']) / p['start_price']) * 100
        print(f"{p['trend']}: {p['start_date'].strftime('%Y-%m-%d')} ~ {p['end_date'].strftime('%Y-%m-%d')}")
        print(f"  기간: {p['duration_weeks']}주, 가격: ${p['start_price']:,.0f} → ${p['end_price']:,.0f} ({change:+.1f}%)")
        print()

# EMA200 기반 장기 추세 판단
df['EMA200'] = df['close'].ewm(span=200).mean()
df['above_ema200'] = df['close'] > df['EMA200']

# EMA200 위/아래 구간 분석
df['ema_trend'] = df['above_ema200'].map({True: 'ABOVE_EMA200', False: 'BELOW_EMA200'})

print("=== EMA200 기준 추세 분포 ===")
ema_dist = df['ema_trend'].value_counts()
print(f"EMA200 위: {ema_dist.get('ABOVE_EMA200', 0)} 캔들 ({ema_dist.get('ABOVE_EMA200', 0)/len(df)*100:.1f}%)")
print(f"EMA200 아래: {ema_dist.get('BELOW_EMA200', 0)} 캔들 ({ema_dist.get('BELOW_EMA200', 0)/len(df)*100:.1f}%)")
print()

# 장기 추세 구간 매핑 (일 단위)
df['date'] = df['timestamp'].dt.date

# 가격 움직임으로 추세 구간 자동 감지
# Swing High/Low 기반 추세 판단
def detect_swing_points(prices, window=48):  # 48시간 = 2일
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(prices) - window):
        # Swing High: 양쪽 window 내에서 최고점
        if prices[i] == max(prices[i-window:i+window+1]):
            swing_highs.append((i, prices[i]))
        # Swing Low: 양쪽 window 내에서 최저점
        if prices[i] == min(prices[i-window:i+window+1]):
            swing_lows.append((i, prices[i]))
    
    return swing_highs, swing_lows

swing_highs, swing_lows = detect_swing_points(df['close'].values)

print(f"=== Swing Point 감지 (48h window) ===")
print(f"Swing Highs: {len(swing_highs)}개")
print(f"Swing Lows: {len(swing_lows)}개")
print()

# 최근 10개 스윙 포인트 출력
print("=== 최근 Swing Highs ===")
for idx, price in swing_highs[-5:]:
    print(f"  {df.iloc[idx]['timestamp']}: ${price:,.0f}")

print()
print("=== 최근 Swing Lows ===")
for idx, price in swing_lows[-5:]:
    print(f"  {df.iloc[idx]['timestamp']}: ${price:,.0f}")

# 추세 구간 라벨링
# HH+HL 연속 = 상승추세, LL+LH 연속 = 하락추세
def label_trend_phases(df, swing_highs, swing_lows):
    # 모든 스윙 포인트를 시간순 정렬
    all_swings = []
    for idx, price in swing_highs:
        all_swings.append({'idx': idx, 'price': price, 'type': 'HIGH'})
    for idx, price in swing_lows:
        all_swings.append({'idx': idx, 'price': price, 'type': 'LOW'})
    
    all_swings = sorted(all_swings, key=lambda x: x['idx'])
    
    # 추세 라벨 초기화
    trend_labels = ['UNKNOWN'] * len(df)
    
    # 연속된 HH/HL 또는 LL/LH 패턴 찾기
    prev_high = None
    prev_low = None
    
    for i, swing in enumerate(all_swings):
        if swing['type'] == 'HIGH':
            if prev_high is not None:
                if swing['price'] > prev_high:  # Higher High
                    pattern_hh = True
                else:  # Lower High
                    pattern_hh = False
            prev_high = swing['price']
        else:  # LOW
            if prev_low is not None:
                if swing['price'] > prev_low:  # Higher Low
                    pattern_hl = True
                else:  # Lower Low
                    pattern_hl = False
            prev_low = swing['price']
    
    return trend_labels

# 간단한 추세 라벨링: EMA20 > EMA50 > EMA200 = 상승, 반대 = 하락
df['EMA20'] = df['close'].ewm(span=20).mean()
df['EMA50'] = df['close'].ewm(span=50).mean()

df['macro_trend'] = 'SIDEWAYS'
df.loc[(df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200']), 'macro_trend'] = 'STRONG_UP'
df.loc[(df['EMA20'] < df['EMA50']) & (df['EMA50'] < df['EMA200']), 'macro_trend'] = 'STRONG_DOWN'
df.loc[(df['EMA20'] > df['EMA50']) & (df['EMA50'] < df['EMA200']), 'macro_trend'] = 'RECOVERING'
df.loc[(df['EMA20'] < df['EMA50']) & (df['EMA50'] > df['EMA200']), 'macro_trend'] = 'WEAKENING'

print()
print("=== 매크로 추세 분포 (EMA 정렬 기준) ===")
macro_dist = df['macro_trend'].value_counts()
for trend, count in macro_dist.items():
    print(f"{trend}: {count} 캔들 ({count/len(df)*100:.1f}%)")

# 추세 데이터 저장
df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'EMA20', 'EMA50', 'EMA200', 'macro_trend']].to_csv('btc_with_macro_trend.csv', index=False)
print()
print("✅ 매크로 추세 데이터 저장: btc_with_macro_trend.csv")

