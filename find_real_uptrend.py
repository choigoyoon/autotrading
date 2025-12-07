import pandas as pd
import numpy as np

# Load data
signals_df = pd.read_csv('valid_signals.csv')
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

def analyze_entry_strength(signal):
    """진입 시점의 '힘'을 측정 - 쫄지 말고 강한 자리만"""
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    gap_pct = signal['gap_pct']
    
    # 진입 직전 24시간 추세
    pre_candles = candles_df[candles_df['datetime'] < entry_time].tail(96)
    if len(pre_candles) < 20:
        return None
    
    first_price = pre_candles.iloc[0]['close']
    last_price = pre_candles.iloc[-1]['close']
    pre_trend = (last_price - first_price) / first_price * 100
    
    # 진입 직전 4시간 (16캔들) 추세
    recent_candles = pre_candles.tail(16)
    recent_first = recent_candles.iloc[0]['close']
    recent_last = recent_candles.iloc[-1]['close']
    recent_trend = (recent_last - recent_first) / recent_first * 100
    
    # 진입 시점 거래량
    entry_candle = candles_df[candles_df['datetime'] == entry_time]
    if len(entry_candle) > 0:
        entry_volume = entry_candle.iloc[0]['volume']
        avg_volume = pre_candles['volume'].mean()
        volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 1
    else:
        volume_ratio = 1
    
    # 양봉 비율 (최근 16캔들)
    bullish_count = (recent_candles['close'] > recent_candles['open']).sum()
    bullish_ratio = bullish_count / len(recent_candles)
    
    return {
        'pre_trend_24h': pre_trend,
        'pre_trend_4h': recent_trend,
        'volume_ratio': volume_ratio,
        'bullish_ratio': bullish_ratio,
        'gap_pct': gap_pct
    }

def backtest_with_strength(signal, strength):
    """백테스트"""
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    trendline_price = signal['trendline_price']
    
    future = candles_df[candles_df['datetime'] > entry_time].head(96)  # 24시간
    
    if len(future) == 0:
        return None
    
    # 재하락 체크
    retest = False
    for idx, candle in future.iterrows():
        if candle['low'] <= trendline_price:
            retest = True
            break
    
    # 최대 수익
    max_price = future['high'].max()
    max_gain = (max_price - entry_price) / entry_price * 100
    
    return {
        **strength,
        'retest': retest,
        'max_gain': max_gain,
        'reached_5pct': max_gain >= 5.0,
        'reached_10pct': max_gain >= 10.0
    }

# 분석
results = []
for idx, signal in signals_df.iterrows():
    strength = analyze_entry_strength(signal)
    if strength:
        result = backtest_with_strength(signal, strength)
        if result:
            results.append(result)

df = pd.DataFrame(results)

print("=" * 80)
print("진실: 쫄지 말고 진짜 강한 자리를 찾아야")
print("=" * 80)

# 재하락 vs 비재하락 비교
retest_group = df[df['retest'] == True]
no_retest_group = df[df['retest'] == False]

print(f"\n재하락한 경우 ({len(retest_group)}건, {len(retest_group)/len(df)*100:.1f}%):")
print(f"  평균 24h 전 추세: {retest_group['pre_trend_24h'].mean():.2f}%")
print(f"  평균 4h 전 추세: {retest_group['pre_trend_4h'].mean():.2f}%")
print(f"  평균 갭: {retest_group['gap_pct'].mean():.2f}%")
print(f"  평균 거래량 배율: {retest_group['volume_ratio'].mean():.2f}x")
print(f"  평균 양봉 비율: {retest_group['bullish_ratio'].mean()*100:.1f}%")
print(f"  평균 최대 수익: {retest_group['max_gain'].mean():.2f}%")

print(f"\n재하락 안 한 경우 ({len(no_retest_group)}건, {len(no_retest_group)/len(df)*100:.1f}%):")
print(f"  평균 24h 전 추세: {no_retest_group['pre_trend_24h'].mean():.2f}%")
print(f"  평균 4h 전 추세: {no_retest_group['pre_trend_4h'].mean():.2f}%")
print(f"  평균 갭: {no_retest_group['gap_pct'].mean():.2f}%")
print(f"  평균 거래량 배율: {no_retest_group['volume_ratio'].mean():.2f}x")
print(f"  평균 양봉 비율: {no_retest_group['bullish_ratio'].mean()*100:.1f}%")
print(f"  평균 최대 수익: {no_retest_group['max_gain'].mean():.2f}%")
print(f"  5% 이상: {no_retest_group['reached_5pct'].sum()}건 ({no_retest_group['reached_5pct'].mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("강한 진입 조건 테스트")
print("=" * 80)

# 조건별 테스트
conditions = [
    ("4h 추세 > 0%", df['pre_trend_4h'] > 0),
    ("4h 추세 > 1%", df['pre_trend_4h'] > 1),
    ("4h 추세 > 2%", df['pre_trend_4h'] > 2),
    ("양봉 비율 > 60%", df['bullish_ratio'] > 0.6),
    ("양봉 비율 > 70%", df['bullish_ratio'] > 0.7),
    ("갭 > 0.5%", df['gap_pct'] > 0.5),
    ("갭 > 1.0%", df['gap_pct'] > 1.0),
]

for name, condition in conditions:
    subset = df[condition]
    if len(subset) > 0:
        retest_rate = subset['retest'].mean() * 100
        success_5pct = (subset['retest'] == False).sum()
        if success_5pct > 0:
            actual_reached = subset[subset['retest'] == False]['reached_5pct'].sum()
            success_rate = actual_reached / success_5pct * 100 if success_5pct > 0 else 0
        else:
            success_rate = 0
        
        print(f"\n{name}:")
        print(f"  시그널 수: {len(subset)}건")
        print(f"  재하락률: {retest_rate:.1f}%")
        print(f"  재하락 안 함: {success_5pct}건 ({success_5pct/len(subset)*100:.1f}%)")
        print(f"  5% 달성: {success_rate:.1f}%")

# 최적 조합
print("\n" + "=" * 80)
print("🔥 쫄지 말고 이 조건으로 가자!")
print("=" * 80)

optimal = df[
    (df['pre_trend_4h'] > 1.0) &  # 4시간 상승 중
    (df['bullish_ratio'] > 0.6) &  # 양봉 60% 이상
    (df['gap_pct'] > 0.5)  # 갭 0.5% 이상
]

if len(optimal) > 0:
    print(f"\n조건: 4h 추세 > 1% + 양봉 > 60% + 갭 > 0.5%")
    print(f"  시그널 수: {len(optimal)}건 (연 {len(optimal)/5.7:.1f}건)")
    print(f"  재하락률: {optimal['retest'].mean()*100:.1f}%")
    no_retest = optimal[optimal['retest'] == False]
    print(f"  재하락 안 함: {len(no_retest)}건 ({len(no_retest)/len(optimal)*100:.1f}%)")
    if len(no_retest) > 0:
        print(f"  평균 최대 수익: {no_retest['max_gain'].mean():.2f}%")
        print(f"  5% 달성: {no_retest['reached_5pct'].sum()}건 ({no_retest['reached_5pct'].mean()*100:.1f}%)")
        print(f"  10% 달성: {no_retest['reached_10pct'].sum()}건 ({no_retest['reached_10pct'].mean()*100:.1f}%)")

