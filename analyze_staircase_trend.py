import pandas as pd
import numpy as np

# 1시간봉 데이터 로드
df = pd.read_csv('btc_1h_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print("=" * 60)
print("1단계: 전체 데이터 개요")
print("=" * 60)
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"캔들 수: {len(df):,}")
print(f"시작가: ${df['close'].iloc[0]:,.0f} → 종가: ${df['close'].iloc[-1]:,.0f}")
print()

# EMA 계산
df['EMA20'] = df['close'].ewm(span=20).mean()
df['EMA50'] = df['close'].ewm(span=50).mean()
df['EMA200'] = df['close'].ewm(span=200).mean()

# Swing High/Low 감지 (일봉 기준 = 24시간 윈도우)
def detect_swing_points(df, window=72):  # 72시간 = 3일
    swing_highs = []
    swing_lows = []
    
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(window, len(df) - window):
        left_high = max(highs[i-window:i])
        right_high = max(highs[i+1:i+window+1])
        if highs[i] >= left_high and highs[i] >= right_high:
            swing_highs.append({
                'idx': i,
                'datetime': df.iloc[i]['datetime'],
                'price': highs[i]
            })
        
        left_low = min(lows[i-window:i])
        right_low = min(lows[i+1:i+window+1])
        if lows[i] <= left_low and lows[i] <= right_low:
            swing_lows.append({
                'idx': i,
                'datetime': df.iloc[i]['datetime'],
                'price': lows[i]
            })
    
    return swing_highs, swing_lows

swing_highs, swing_lows = detect_swing_points(df, window=72)

print("=" * 60)
print("2단계: Swing Points 감지 (72h window)")
print("=" * 60)
print(f"Swing Highs: {len(swing_highs)}개")
print(f"Swing Lows: {len(swing_lows)}개")
print()

# 계단식 상승/하락 판단
# 모든 스윙 포인트 병합 및 시간순 정렬
all_swings = []
for sh in swing_highs:
    all_swings.append({**sh, 'type': 'HIGH'})
for sl in swing_lows:
    all_swings.append({**sl, 'type': 'LOW'})

all_swings = sorted(all_swings, key=lambda x: x['idx'])

print("=" * 60)
print("3단계: 추세 구간 식별")
print("=" * 60)

# HH/HL vs LH/LL 패턴으로 추세 판단
prev_high = None
prev_low = None
trend_changes = []

for swing in all_swings:
    if swing['type'] == 'HIGH':
        if prev_high is not None:
            if swing['price'] > prev_high:
                pattern = 'HH'  # Higher High
            else:
                pattern = 'LH'  # Lower High
            trend_changes.append({
                'idx': swing['idx'],
                'datetime': swing['datetime'],
                'type': 'HIGH',
                'price': swing['price'],
                'pattern': pattern,
                'prev_price': prev_high
            })
        prev_high = swing['price']
    else:  # LOW
        if prev_low is not None:
            if swing['price'] > prev_low:
                pattern = 'HL'  # Higher Low
            else:
                pattern = 'LL'  # Lower Low
            trend_changes.append({
                'idx': swing['idx'],
                'datetime': swing['datetime'],
                'type': 'LOW',
                'price': swing['price'],
                'pattern': pattern,
                'prev_price': prev_low
            })
        prev_low = swing['price']

trend_df = pd.DataFrame(trend_changes)
print(f"추세 변화 포인트: {len(trend_df)}개")
print()

# 계단식 상승 구간: HH와 HL이 연속
# 계단식 하락 구간: LH와 LL이 연속
def identify_staircase_trend(trend_df):
    """연속된 HH+HL 또는 LH+LL 패턴으로 추세 구간 식별"""
    
    trends = []
    
    # 패턴 2개씩 확인
    for i in range(1, len(trend_df)):
        curr = trend_df.iloc[i]
        prev = trend_df.iloc[i-1]
        
        # HH + HL = 상승 추세
        if (curr['pattern'] == 'HL' and prev['pattern'] == 'HH') or \
           (curr['pattern'] == 'HH' and prev['pattern'] == 'HL'):
            trend_type = 'UPTREND'
        # LH + LL = 하락 추세
        elif (curr['pattern'] == 'LL' and prev['pattern'] == 'LH') or \
             (curr['pattern'] == 'LH' and prev['pattern'] == 'LL'):
            trend_type = 'DOWNTREND'
        else:
            trend_type = 'TRANSITION'
        
        trends.append({
            'idx': curr['idx'],
            'datetime': curr['datetime'],
            'pattern': f"{prev['pattern']}->{curr['pattern']}",
            'trend': trend_type
        })
    
    return pd.DataFrame(trends)

staircase = identify_staircase_trend(trend_df)

print("=== 패턴 분포 ===")
print(staircase['trend'].value_counts())
print()

print("=== 패턴 상세 ===")
print(staircase['pattern'].value_counts())
print()

# 각 캔들에 추세 라벨 부여
df['staircase_trend'] = 'UNKNOWN'

# 추세 구간 매핑
for i in range(len(staircase)):
    row = staircase.iloc[i]
    start_idx = row['idx']
    end_idx = staircase.iloc[i+1]['idx'] if i < len(staircase)-1 else len(df)-1
    
    df.loc[start_idx:end_idx, 'staircase_trend'] = row['trend']

print("=== 캔들별 추세 분포 ===")
trend_dist = df['staircase_trend'].value_counts()
for trend, count in trend_dist.items():
    pct = count / len(df) * 100
    print(f"{trend}: {count:,} 캔들 ({pct:.1f}%)")
print()

# 주요 계단식 상승/하락 구간 출력
print("=" * 60)
print("4단계: 주요 추세 구간")
print("=" * 60)

# 연속된 같은 추세 구간 찾기
def find_continuous_trends(staircase_df):
    phases = []
    current_trend = None
    start_idx = 0
    start_dt = None
    
    for i, row in staircase_df.iterrows():
        if row['trend'] != current_trend:
            if current_trend is not None and current_trend in ['UPTREND', 'DOWNTREND']:
                phases.append({
                    'trend': current_trend,
                    'start_idx': start_idx,
                    'end_idx': row['idx'],
                    'start_dt': start_dt,
                    'end_dt': row['datetime'],
                    'duration_hours': row['idx'] - start_idx
                })
            current_trend = row['trend']
            start_idx = row['idx']
            start_dt = row['datetime']
    
    # 마지막 구간
    if current_trend in ['UPTREND', 'DOWNTREND']:
        phases.append({
            'trend': current_trend,
            'start_idx': start_idx,
            'end_idx': len(df)-1,
            'start_dt': start_dt,
            'end_dt': df.iloc[-1]['datetime'],
            'duration_hours': len(df) - start_idx
        })
    
    return phases

phases = find_continuous_trends(staircase)

# 가격 변화 계산
for p in phases:
    p['start_price'] = df.iloc[p['start_idx']]['close']
    p['end_price'] = df.iloc[p['end_idx']]['close']
    p['change_pct'] = (p['end_price'] - p['start_price']) / p['start_price'] * 100

# 100시간 이상 지속된 구간만 출력
print("\n=== 100시간 이상 지속된 추세 구간 ===")
long_phases = [p for p in phases if p['duration_hours'] >= 100]
print(f"총 {len(long_phases)}개 구간")
print()

for p in long_phases[:15]:  # 최근 15개
    days = p['duration_hours'] / 24
    print(f"[{p['trend']}] {p['start_dt'].strftime('%Y-%m-%d')} ~ {p['end_dt'].strftime('%Y-%m-%d')}")
    print(f"  기간: {p['duration_hours']}h ({days:.1f}일)")
    print(f"  가격: ${p['start_price']:,.0f} → ${p['end_price']:,.0f} ({p['change_pct']:+.1f}%)")
    print()

# 저장
df.to_csv('btc_with_staircase_trend.csv', index=False)
print("✅ 저장: btc_with_staircase_trend.csv")

# 추세 구간 요약 저장
phases_df = pd.DataFrame(phases)
phases_df.to_csv('staircase_trend_phases.csv', index=False)
print("✅ 저장: staircase_trend_phases.csv")

