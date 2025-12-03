import pandas as pd
import numpy as np

print("=" * 70)
print("상위 타임프레임 추세 분석")
print("=" * 70)

# 4시간봉, 일봉 데이터 로드
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

df_1d = pd.read_csv('btc_1d_ohlcv.csv')
df_1d['datetime'] = pd.to_datetime(df_1d['datetime'])

# 1시간봉 거래 데이터
trades = pd.read_csv('extended_holding_results.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print(f"4시간봉: {len(df_4h)}개")
print(f"일봉: {len(df_1d)}개")
print(f"거래: {len(trades)}건")
print()

# 상위 타임프레임 지표 계산
def add_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    
    df['ema_bullish'] = (df['EMA20'] > df['EMA50']) & (df['EMA50'] > df['EMA200'])
    df['ema_bearish'] = (df['EMA20'] < df['EMA50']) & (df['EMA50'] < df['EMA200'])
    df['above_ema200'] = df['close'] > df['EMA200']
    
    # 추세 기울기
    df['ema200_slope'] = df['EMA200'].diff(5) / df['EMA200'].shift(5) * 100
    
    # 모멘텀
    df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
    
    # 고점/저점 갱신
    df['highest_20'] = df['high'].rolling(20).max()
    df['lowest_20'] = df['low'].rolling(20).min()
    df['near_high'] = df['close'] >= df['highest_20'] * 0.97  # 고점 3% 이내
    df['near_low'] = df['close'] <= df['lowest_20'] * 1.03   # 저점 3% 이내
    
    # HH/HL 패턴 (최근 vs 이전)
    window = 20
    df['prev_high'] = df['high'].shift(window).rolling(window).max()
    df['prev_low'] = df['low'].shift(window).rolling(window).min()
    df['curr_high'] = df['high'].rolling(window).max()
    df['curr_low'] = df['low'].rolling(window).min()
    df['hh'] = df['curr_high'] > df['prev_high']
    df['hl'] = df['curr_low'] > df['prev_low']
    df['lh'] = df['curr_high'] < df['prev_high']
    df['ll'] = df['curr_low'] < df['prev_low']
    df['swing_bullish'] = df['hh'] & df['hl']
    df['swing_bearish'] = df['lh'] & df['ll']
    
    return df

df_4h = add_indicators(df_4h)
df_1d = add_indicators(df_1d)

# 거래 시점에 상위 TF 지표 매핑
def get_htf_data(trade_time, htf_df):
    """상위 타임프레임에서 해당 시점 데이터 가져오기"""
    idx = htf_df[htf_df['datetime'] <= trade_time].index
    if len(idx) == 0:
        return None
    return htf_df.loc[idx[-1]]

results = []
for idx, trade in trades.iterrows():
    entry_time = trade['datetime']
    
    # 4시간봉 데이터
    h4 = get_htf_data(entry_time, df_4h)
    # 일봉 데이터
    d1 = get_htf_data(entry_time, df_1d)
    
    if h4 is None or d1 is None:
        continue
    
    results.append({
        'datetime': entry_time,
        'direction': trade['direction'],
        'actual_trend': trade['trend'],
        'extended_pnl': trade['extended_pnl'],
        # 4시간봉 지표
        'h4_bullish': h4['ema_bullish'],
        'h4_bearish': h4['ema_bearish'],
        'h4_above_ema200': h4['above_ema200'],
        'h4_ema200_slope': h4['ema200_slope'],
        'h4_momentum_10': h4['momentum_10'],
        'h4_near_high': h4['near_high'],
        'h4_near_low': h4['near_low'],
        'h4_swing_bullish': h4['swing_bullish'],
        'h4_swing_bearish': h4['swing_bearish'],
        # 일봉 지표
        'd1_bullish': d1['ema_bullish'],
        'd1_bearish': d1['ema_bearish'],
        'd1_above_ema200': d1['above_ema200'],
        'd1_ema200_slope': d1['ema200_slope'],
        'd1_momentum_10': d1['momentum_10'],
        'd1_near_high': d1['near_high'],
        'd1_near_low': d1['near_low'],
        'd1_swing_bullish': d1['swing_bullish'],
        'd1_swing_bearish': d1['swing_bearish'],
    })

result_df = pd.DataFrame(results)
result_df = result_df.dropna()

print(f"분석 대상: {len(result_df)}건")
print()

# 상위 TF 지표별 성과
print("=" * 70)
print("4시간봉 기준")
print("=" * 70)

# 4H 정배열
h4_bull = result_df[result_df['h4_bullish'] == True]
pnl = h4_bull.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (h4_bull['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[4H 정배열 → LONG] ({len(h4_bull)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 4H swing bullish
h4_swing = result_df[result_df['h4_swing_bullish'] == True]
pnl = h4_swing.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (h4_swing['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[4H HH+HL 패턴 → LONG] ({len(h4_swing)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 4H 고점 근처
h4_high = result_df[result_df['h4_near_high'] == True]
pnl = h4_high.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (h4_high['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[4H 20봉 고점 근처 → LONG] ({len(h4_high)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

print()
print("=" * 70)
print("일봉 기준")
print("=" * 70)

# 1D 정배열
d1_bull = result_df[result_df['d1_bullish'] == True]
pnl = d1_bull.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (d1_bull['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[일봉 정배열 → LONG] ({len(d1_bull)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 1D swing bullish
d1_swing = result_df[result_df['d1_swing_bullish'] == True]
pnl = d1_swing.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (d1_swing['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[일봉 HH+HL 패턴 → LONG] ({len(d1_swing)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 1D 고점 근처
d1_high = result_df[result_df['d1_near_high'] == True]
pnl = d1_high.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
trend_acc = (d1_high['actual_trend'] == 'UPTREND').mean() * 100
print(f"\n[일봉 20봉 고점 근처 → LONG] ({len(d1_high)}건)")
print(f"  UPTREND 적중률: {trend_acc:.1f}%")
print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

print()
print("=" * 70)
print("복합 조건 (1H + 4H + 1D)")
print("=" * 70)

# 모든 TF 정배열
all_bull = result_df[(result_df['h4_bullish'] == True) & (result_df['d1_bullish'] == True)]
if len(all_bull) >= 20:
    pnl = all_bull.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    trend_acc = (all_bull['actual_trend'] == 'UPTREND').mean() * 100
    print(f"\n[4H+일봉 모두 정배열 → LONG] ({len(all_bull)}건)")
    print(f"  UPTREND 적중률: {trend_acc:.1f}%")
    print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 일봉 정배열 + 4H HH/HL
combo1 = result_df[(result_df['d1_bullish'] == True) & (result_df['h4_swing_bullish'] == True)]
if len(combo1) >= 20:
    pnl = combo1.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    trend_acc = (combo1['actual_trend'] == 'UPTREND').mean() * 100
    print(f"\n[일봉 정배열 + 4H HH/HL → LONG] ({len(combo1)}건)")
    print(f"  UPTREND 적중률: {trend_acc:.1f}%")
    print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 일봉 HH/HL + 일봉 EMA200 위
combo2 = result_df[(result_df['d1_swing_bullish'] == True) & (result_df['d1_above_ema200'] == True)]
if len(combo2) >= 20:
    pnl = combo2.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    trend_acc = (combo2['actual_trend'] == 'UPTREND').mean() * 100
    print(f"\n[일봉 HH/HL + EMA200↑ → LONG] ({len(combo2)}건)")
    print(f"  UPTREND 적중률: {trend_acc:.1f}%")
    print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 일봉 + 4H 모두 고점 근처
combo3 = result_df[(result_df['d1_near_high'] == True) & (result_df['h4_near_high'] == True)]
if len(combo3) >= 20:
    pnl = combo3.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    trend_acc = (combo3['actual_trend'] == 'UPTREND').mean() * 100
    print(f"\n[일봉+4H 모두 고점 근처 → LONG] ({len(combo3)}건)")
    print(f"  UPTREND 적중률: {trend_acc:.1f}%")
    print(f"  승률: {(pnl > 0).mean()*100:.1f}%, 평균: {pnl.mean():.2f}%")

# 저장
result_df.to_csv('htf_trend_analysis.csv', index=False)
print()
print("✅ 저장: htf_trend_analysis.csv")

