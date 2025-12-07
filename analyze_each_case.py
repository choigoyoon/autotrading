import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("🔍 각 상황별 실패 사유 상세 분석")
print("=" * 80)

# Load data
df = pd.read_csv('backtest_confirmation_space_results.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df.get('exit_time', df['entry_time']))

ohlcv = pd.read_csv('btc_15m_ohlcv.csv')
ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
ohlcv = ohlcv.sort_values('datetime').reset_index(drop=True)

print(f"\n총 거래: {len(df)}개")
print(f"SL: {len(df[df['exit_reason'] == 'SL'])}개")
print(f"TP1 BE: {len(df[df['exit_reason'] == 'TP1_Breakeven'])}개")
print(f"TP2 Full: {len(df[df['exit_reason'] == 'TP2_Full'])}개")

# Analyze each case type
print("\n" + "=" * 80)
print("1단계: SL 케이스 상세 분석")
print("=" * 80)

sl_cases = df[df['exit_reason'] == 'SL'].copy()
sl_analysis = []

for idx, trade in sl_cases.iterrows():
    # Get candles after entry
    entry_idx = ohlcv[ohlcv['datetime'] == trade['entry_time']].index
    if len(entry_idx) == 0:
        continue
    entry_idx = entry_idx[0]
    
    # Next 20 candles
    next_candles = ohlcv.iloc[entry_idx:entry_idx+20].copy()
    
    if len(next_candles) < 2:
        continue
    
    # Calculate key metrics
    entry_price = trade['entry_price']
    h3_price = trade['h3_price']
    sl_price = trade['sl_price']
    
    # 진입 후 즉시 하락 여부
    first_candle = next_candles.iloc[1] if len(next_candles) > 1 else next_candles.iloc[0]
    immediate_drop = first_candle['low'] < entry_price
    
    # H3 지지 실패 여부
    h3_support_fail = next_candles['low'].min() < h3_price
    
    # 진입 후 최대 상승 도달
    max_high = next_candles['high'].max()
    max_gain_pct = ((max_high - entry_price) / entry_price) * 100
    
    # H2/H1 도달 여부
    reached_h2 = max_high >= trade['h2_price']
    reached_h1 = max_high >= trade['h1_price']
    
    # 변동성 체크
    volatility = next_candles['high'] - next_candles['low']
    avg_volatility = volatility.mean()
    volatility_pct = (avg_volatility / entry_price) * 100
    
    # 실패 유형 분류
    if immediate_drop:
        failure_type = "진입 직후 하락"
    elif not h3_support_fail:
        failure_type = "H3 지지 유지했으나 상승 실패"
    elif max_gain_pct > 1.0:
        failure_type = "상승 후 되돌림 (페이크)"
    elif volatility_pct > 2.0:
        failure_type = "고변동성 급락"
    else:
        failure_type = "약한 상승 동력"
    
    sl_analysis.append({
        'entry_time': trade['entry_time'],
        'power_score': trade['power_score'],
        'year': trade['year'],
        'immediate_drop': immediate_drop,
        'h3_support_fail': h3_support_fail,
        'max_gain_pct': max_gain_pct,
        'reached_h2': reached_h2,
        'reached_h1': reached_h1,
        'volatility_pct': volatility_pct,
        'failure_type': failure_type
    })

sl_df = pd.DataFrame(sl_analysis)

print(f"\nSL 케이스 실패 유형 분포:")
print(sl_df['failure_type'].value_counts())
print(f"\n각 유형별 상세:")
for ftype in sl_df['failure_type'].unique():
    subset = sl_df[sl_df['failure_type'] == ftype]
    print(f"\n  [{ftype}]: {len(subset)}개")
    print(f"    - 평균 Power Score: {subset['power_score'].mean():.1f}")
    print(f"    - 평균 최대 상승: {subset['max_gain_pct'].mean():.2f}%")
    print(f"    - H2 도달률: {(subset['reached_h2'].sum() / len(subset) * 100):.1f}%")
    print(f"    - 평균 변동성: {subset['volatility_pct'].mean():.2f}%")

print("\n" + "=" * 80)
print("2단계: TP1 Breakeven 케이스 상세 분석")
print("=" * 80)

tp1_cases = df[df['exit_reason'] == 'TP1_Breakeven'].copy()
tp1_analysis = []

for idx, trade in tp1_cases.iterrows():
    entry_idx = ohlcv[ohlcv['datetime'] == trade['entry_time']].index
    if len(entry_idx) == 0:
        continue
    entry_idx = entry_idx[0]
    
    # Next 30 candles
    next_candles = ohlcv.iloc[entry_idx:entry_idx+30].copy()
    
    if len(next_candles) < 2:
        continue
    
    entry_price = trade['entry_price']
    h2_price = trade['h2_price']
    h1_price = trade['h1_price']
    
    # H2 도달 여부
    max_high = next_candles['high'].max()
    reached_h2 = max_high >= h2_price
    h2_progress = ((max_high - entry_price) / (h2_price - entry_price)) * 100
    
    # H1 근접도
    h1_progress = ((max_high - entry_price) / (h1_price - entry_price)) * 100
    
    # TP2 근접도 (90% 이상이면 아쉬운 케이스)
    almost_tp2 = h1_progress >= 90
    
    # 되돌림 강도
    min_low_after_peak = next_candles['low'].min()
    retracement_pct = ((max_high - min_low_after_peak) / (max_high - entry_price)) * 100 if max_high > entry_price else 0
    
    # 실패 유형 분류
    if almost_tp2:
        failure_type = "TP2 직전 되돌림 (90%+)"
    elif reached_h2:
        failure_type = "H2 돌파 후 되돌림"
    elif h2_progress >= 70:
        failure_type = "H2 70% 근접 후 되돌림"
    elif h2_progress >= 50:
        failure_type = "H2 50% 근접 후 되돌림"
    else:
        failure_type = "약한 상승 (50% 미달)"
    
    tp1_analysis.append({
        'entry_time': trade['entry_time'],
        'power_score': trade['power_score'],
        'year': trade['year'],
        'reached_h2': reached_h2,
        'h2_progress': h2_progress,
        'h1_progress': h1_progress,
        'almost_tp2': almost_tp2,
        'retracement_pct': retracement_pct,
        'failure_type': failure_type
    })

tp1_df = pd.DataFrame(tp1_analysis)

print(f"\nTP1 Breakeven 케이스 실패 유형 분포:")
print(tp1_df['failure_type'].value_counts())
print(f"\n각 유형별 상세:")
for ftype in tp1_df['failure_type'].unique():
    subset = tp1_df[tp1_df['failure_type'] == ftype]
    print(f"\n  [{ftype}]: {len(subset)}개")
    print(f"    - 평균 Power Score: {subset['power_score'].mean():.1f}")
    print(f"    - 평균 H2 진행률: {subset['h2_progress'].mean():.1f}%")
    print(f"    - 평균 H1 진행률: {subset['h1_progress'].mean():.1f}%")
    print(f"    - 평균 되돌림: {subset['retracement_pct'].mean():.1f}%")

print("\n" + "=" * 80)
print("3단계: 성공 케이스 vs 실패 케이스 비교")
print("=" * 80)

tp2_cases = df[df['exit_reason'] == 'TP2_Full'].copy()
tp2_analysis = []

for idx, trade in tp2_cases.iterrows():
    entry_idx = ohlcv[ohlcv['datetime'] == trade['entry_time']].index
    if len(entry_idx) == 0:
        continue
    entry_idx = entry_idx[0]
    
    next_candles = ohlcv.iloc[entry_idx:entry_idx+20].copy()
    
    if len(next_candles) < 2:
        continue
    
    entry_price = trade['entry_price']
    h1_price = trade['h1_price']
    h3_price = trade['h3_price']
    
    # H1 도달까지 캔들 수
    h1_candles = 0
    for i, candle in next_candles.iterrows():
        h1_candles += 1
        if candle['high'] >= h1_price:
            break
    
    # H3 지지 유지
    h3_support_maintained = next_candles['low'].min() >= h3_price * 0.998
    
    # 연속 상승
    consecutive_up = 0
    max_consecutive_up = 0
    for i in range(1, len(next_candles)):
        if next_candles.iloc[i]['close'] > next_candles.iloc[i-1]['close']:
            consecutive_up += 1
            max_consecutive_up = max(max_consecutive_up, consecutive_up)
        else:
            consecutive_up = 0
    
    tp2_analysis.append({
        'entry_time': trade['entry_time'],
        'power_score': trade['power_score'],
        'h1_candles': h1_candles,
        'h3_support_maintained': h3_support_maintained,
        'max_consecutive_up': max_consecutive_up
    })

tp2_df = pd.DataFrame(tp2_analysis)

print(f"\n성공 케이스 (TP2 Full) 특징:")
print(f"  - 평균 Power Score: {tp2_df['power_score'].mean():.1f}")
print(f"  - 평균 H1 도달 캔들: {tp2_df['h1_candles'].mean():.1f}개")
print(f"  - H3 지지 유지율: {(tp2_df['h3_support_maintained'].sum() / len(tp2_df) * 100):.1f}%")
print(f"  - 평균 최대 연속 상승: {tp2_df['max_consecutive_up'].mean():.1f}개")

print(f"\n실패 케이스 (SL) 특징:")
print(f"  - 평균 Power Score: {sl_df['power_score'].mean():.1f}")
print(f"  - H3 지지 실패율: {(sl_df['h3_support_fail'].sum() / len(sl_df) * 100):.1f}%")
print(f"  - 즉시 하락율: {(sl_df['immediate_drop'].sum() / len(sl_df) * 100):.1f}%")

print("\n" + "=" * 80)
print("4단계: 핵심 차이점")
print("=" * 80)

print(f"\n✅ 성공 케이스는:")
print(f"  1. H3 지지 유지: {(tp2_df['h3_support_maintained'].sum() / len(tp2_df) * 100):.1f}%")
print(f"  2. 빠른 H1 도달: 평균 {tp2_df['h1_candles'].mean():.1f}캔들")
print(f"  3. 연속 상승: 평균 {tp2_df['max_consecutive_up'].mean():.1f}개")

print(f"\n❌ 실패 케이스는:")
print(f"  1. 진입 직후 하락: {len(sl_df[sl_df['failure_type'] == '진입 직후 하락'])}개")
print(f"  2. 상승 후 되돌림: {len(sl_df[sl_df['failure_type'] == '상승 후 되돌림 (페이크)'])}개")
print(f"  3. 약한 상승 동력: {len(sl_df[sl_df['failure_type'] == '약한 상승 동력'])}개")
print(f"  4. 고변동성 급락: {len(sl_df[sl_df['failure_type'] == '고변동성 급락'])}개")

# Save detailed analysis
sl_df.to_csv('sl_cases_detailed.csv', index=False)
tp1_df.to_csv('tp1_cases_detailed.csv', index=False)
tp2_df.to_csv('tp2_cases_detailed.csv', index=False)

print(f"\n✅ 상세 분석 저장:")
print(f"  - sl_cases_detailed.csv")
print(f"  - tp1_cases_detailed.csv")
print(f"  - tp2_cases_detailed.csv")
