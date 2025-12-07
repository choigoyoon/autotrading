import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('btc_with_staircase_trend.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

trades = pd.read_csv('extended_holding_results.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print("=" * 70)
print("발산 시점에 추세를 어떻게 판단할 수 있는가?")
print("=" * 70)
print()

# 각 거래 시점의 지표들 수집
results = []

for idx, trade in trades.iterrows():
    entry_time = trade['datetime']
    entry_idx = df[df['datetime'] <= entry_time].index
    if len(entry_idx) == 0:
        continue
    entry_idx = entry_idx[-1]
    
    # 진입 시점 데이터
    row = df.loc[entry_idx]
    
    # 1. EMA 배열 상태 (진입 시점 기준)
    ema20 = row['EMA20']
    ema50 = row['EMA50']
    ema200 = row['EMA200']
    close = row['close']
    
    ema_bullish = (ema20 > ema50) and (ema50 > ema200)  # 정배열
    ema_bearish = (ema20 < ema50) and (ema50 < ema200)  # 역배열
    above_ema200 = close > ema200
    
    # 2. 최근 스윙 포인트 패턴 (이전 데이터로 HH/HL vs LH/LL 판단)
    # 최근 50봉의 고점/저점
    lookback = 200
    start_idx = max(0, entry_idx - lookback)
    recent = df.loc[start_idx:entry_idx]
    
    # 단순화: 최근 고점/저점 비교
    half = len(recent) // 2
    if half > 10:
        first_half = recent.iloc[:half]
        second_half = recent.iloc[half:]
        
        first_high = first_half['high'].max()
        second_high = second_half['high'].max()
        first_low = first_half['low'].min()
        second_low = second_half['low'].min()
        
        # HH + HL = 상승 추세 신호
        hh = second_high > first_high
        hl = second_low > first_low
        # LH + LL = 하락 추세 신호
        lh = second_high < first_high
        ll = second_low < first_low
        
        if hh and hl:
            swing_pattern = 'BULLISH'  # 계단식 상승
        elif lh and ll:
            swing_pattern = 'BEARISH'  # 계단식 하락
        else:
            swing_pattern = 'MIXED'
    else:
        swing_pattern = 'UNKNOWN'
    
    # 3. 가격 위치 (최근 range 대비)
    recent_high = recent['high'].max()
    recent_low = recent['low'].min()
    price_position = (close - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    
    results.append({
        'datetime': entry_time,
        'direction': trade['direction'],
        'actual_trend': trade['trend'],  # 실제 추세 (정답)
        'extended_pnl': trade['extended_pnl'],
        # 판단 지표들
        'ema_bullish': ema_bullish,
        'ema_bearish': ema_bearish,
        'above_ema200': above_ema200,
        'swing_pattern': swing_pattern,
        'price_position': price_position,
    })

result_df = pd.DataFrame(results)

print(f"분석 대상: {len(result_df)}건")
print()

# 각 지표의 추세 예측 정확도 분석
print("=" * 70)
print("1. EMA 정배열 → 실제 UPTREND 적중률")
print("=" * 70)

bullish_ema = result_df[result_df['ema_bullish'] == True]
print(f"EMA 정배열 케이스: {len(bullish_ema)}건")
print(f"  실제 UPTREND: {(bullish_ema['actual_trend'] == 'UPTREND').sum()}건 ({(bullish_ema['actual_trend'] == 'UPTREND').mean()*100:.1f}%)")
print(f"  실제 DOWNTREND: {(bullish_ema['actual_trend'] == 'DOWNTREND').sum()}건")
print(f"  실제 TRANSITION: {(bullish_ema['actual_trend'] == 'TRANSITION').sum()}건")

# 정배열일 때 LONG 성과
bullish_long_pnl = bullish_ema.apply(
    lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'],
    axis=1
)
print(f"  → LONG 진입 시: 승률 {(bullish_long_pnl > 0).mean()*100:.1f}%, 평균 {bullish_long_pnl.mean():.2f}%")
print()

print("=" * 70)
print("2. EMA 역배열 → 실제 DOWNTREND 적중률")
print("=" * 70)

bearish_ema = result_df[result_df['ema_bearish'] == True]
print(f"EMA 역배열 케이스: {len(bearish_ema)}건")
print(f"  실제 UPTREND: {(bearish_ema['actual_trend'] == 'UPTREND').sum()}건")
print(f"  실제 DOWNTREND: {(bearish_ema['actual_trend'] == 'DOWNTREND').sum()}건 ({(bearish_ema['actual_trend'] == 'DOWNTREND').mean()*100:.1f}%)")
print(f"  실제 TRANSITION: {(bearish_ema['actual_trend'] == 'TRANSITION').sum()}건")

# 역배열일 때 SHORT 성과
bearish_short_pnl = bearish_ema.apply(
    lambda x: x['extended_pnl'] if x['direction'] == 'SHORT' else -x['extended_pnl'],
    axis=1
)
print(f"  → SHORT 진입 시: 승률 {(bearish_short_pnl > 0).mean()*100:.1f}%, 평균 {bearish_short_pnl.mean():.2f}%")
print()

print("=" * 70)
print("3. 스윙 패턴 → 실제 추세 적중률")
print("=" * 70)

for pattern in ['BULLISH', 'BEARISH', 'MIXED']:
    subset = result_df[result_df['swing_pattern'] == pattern]
    if len(subset) == 0:
        continue
    
    print(f"\n[스윙 패턴: {pattern}] ({len(subset)}건)")
    print(f"  실제 UPTREND: {(subset['actual_trend'] == 'UPTREND').mean()*100:.1f}%")
    print(f"  실제 DOWNTREND: {(subset['actual_trend'] == 'DOWNTREND').mean()*100:.1f}%")
    
    if pattern == 'BULLISH':
        pnl = subset.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
        print(f"  → LONG 시: 승률 {(pnl > 0).mean()*100:.1f}%, 평균 {pnl.mean():.2f}%")
    elif pattern == 'BEARISH':
        pnl = subset.apply(lambda x: x['extended_pnl'] if x['direction'] == 'SHORT' else -x['extended_pnl'], axis=1)
        print(f"  → SHORT 시: 승률 {(pnl > 0).mean()*100:.1f}%, 평균 {pnl.mean():.2f}%")

print()
print("=" * 70)
print("4. 복합 조건 테스트")
print("=" * 70)

# 정배열 + 스윙 BULLISH
combo1 = result_df[(result_df['ema_bullish'] == True) & (result_df['swing_pattern'] == 'BULLISH')]
if len(combo1) > 0:
    pnl = combo1.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    print(f"\n[정배열 + 스윙 BULLISH] ({len(combo1)}건)")
    print(f"  실제 UPTREND: {(combo1['actual_trend'] == 'UPTREND').mean()*100:.1f}%")
    print(f"  → LONG 시: 승률 {(pnl > 0).mean()*100:.1f}%, 평균 {pnl.mean():.2f}%")

# 역배열 + 스윙 BEARISH
combo2 = result_df[(result_df['ema_bearish'] == True) & (result_df['swing_pattern'] == 'BEARISH')]
if len(combo2) > 0:
    pnl = combo2.apply(lambda x: x['extended_pnl'] if x['direction'] == 'SHORT' else -x['extended_pnl'], axis=1)
    print(f"\n[역배열 + 스윙 BEARISH] ({len(combo2)}건)")
    print(f"  실제 DOWNTREND: {(combo2['actual_trend'] == 'DOWNTREND').mean()*100:.1f}%")
    print(f"  → SHORT 시: 승률 {(pnl > 0).mean()*100:.1f}%, 평균 {pnl.mean():.2f}%")

# EMA200 위 + 정배열
combo3 = result_df[(result_df['above_ema200'] == True) & (result_df['ema_bullish'] == True)]
if len(combo3) > 0:
    pnl = combo3.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    print(f"\n[EMA200↑ + 정배열] ({len(combo3)}건)")
    print(f"  실제 UPTREND: {(combo3['actual_trend'] == 'UPTREND').mean()*100:.1f}%")
    print(f"  → LONG 시: 승률 {(pnl > 0).mean()*100:.1f}%, 평균 {pnl.mean():.2f}%")

# 저장
result_df.to_csv('trend_identification_analysis.csv', index=False)
print()
print("✅ 저장: trend_identification_analysis.csv")

