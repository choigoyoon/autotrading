import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('btc_with_staircase_trend.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

trades = pd.read_csv('extended_holding_results.csv')
trades['datetime'] = pd.to_datetime(trades['datetime'])

print("=" * 70)
print("방향성 판단 지표 탐색")
print("=" * 70)

# 추가 지표 계산
df['EMA20'] = df['close'].ewm(span=20).mean()
df['EMA50'] = df['close'].ewm(span=50).mean()
df['EMA200'] = df['close'].ewm(span=200).mean()

# 1. EMA 기울기
df['EMA20_slope'] = df['EMA20'].diff(5) / df['EMA20'].shift(5) * 100
df['EMA50_slope'] = df['EMA50'].diff(10) / df['EMA50'].shift(10) * 100
df['EMA200_slope'] = df['EMA200'].diff(20) / df['EMA200'].shift(20) * 100

# 2. 가격 모멘텀
df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
df['momentum_50'] = (df['close'] - df['close'].shift(50)) / df['close'].shift(50) * 100
df['momentum_100'] = (df['close'] - df['close'].shift(100)) / df['close'].shift(100) * 100

# 3. ATR (변동성)
df['TR'] = np.maximum(df['high'] - df['low'], 
                      np.maximum(abs(df['high'] - df['close'].shift(1)),
                                abs(df['low'] - df['close'].shift(1))))
df['ATR'] = df['TR'].rolling(14).mean()
df['ATR_pct'] = df['ATR'] / df['close'] * 100

# 4. 고점/저점 갱신 여부
df['highest_20'] = df['high'].rolling(20).max()
df['lowest_20'] = df['low'].rolling(20).min()
df['highest_50'] = df['high'].rolling(50).max()
df['lowest_50'] = df['low'].rolling(50).min()
df['at_20_high'] = df['high'] >= df['highest_20']
df['at_20_low'] = df['low'] <= df['lowest_20']
df['at_50_high'] = df['high'] >= df['highest_50']
df['at_50_low'] = df['low'] <= df['lowest_50']

# 5. 이전 N개 캔들 상승/하락 비율
df['up_candle'] = (df['close'] > df['open']).astype(int)
df['up_ratio_20'] = df['up_candle'].rolling(20).mean()
df['up_ratio_50'] = df['up_candle'].rolling(50).mean()

# 6. RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 7. MACD
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# 8. 볼린저밴드 위치
df['BB_mid'] = df['close'].rolling(20).mean()
df['BB_std'] = df['close'].rolling(20).std()
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# 거래에 지표 매핑
results = []
for idx, trade in trades.iterrows():
    entry_time = trade['datetime']
    entry_idx = df[df['datetime'] <= entry_time].index
    if len(entry_idx) == 0:
        continue
    entry_idx = entry_idx[-1]
    
    row = df.loc[entry_idx]
    
    results.append({
        'datetime': entry_time,
        'direction': trade['direction'],
        'actual_trend': trade['trend'],
        'extended_pnl': trade['extended_pnl'],
        # 지표들
        'ema20_slope': row['EMA20_slope'],
        'ema50_slope': row['EMA50_slope'],
        'ema200_slope': row['EMA200_slope'],
        'momentum_20': row['momentum_20'],
        'momentum_50': row['momentum_50'],
        'momentum_100': row['momentum_100'],
        'atr_pct': row['ATR_pct'],
        'at_20_high': row['at_20_high'],
        'at_20_low': row['at_20_low'],
        'at_50_high': row['at_50_high'],
        'at_50_low': row['at_50_low'],
        'up_ratio_20': row['up_ratio_20'],
        'up_ratio_50': row['up_ratio_50'],
        'rsi': row['RSI'],
        'macd_hist': row['MACD_hist'],
        'bb_position': row['BB_position'],
        # EMA 관련
        'above_ema20': row['close'] > row['EMA20'],
        'above_ema50': row['close'] > row['EMA50'],
        'above_ema200': row['close'] > row['EMA200'],
        'ema_bullish': (row['EMA20'] > row['EMA50']) and (row['EMA50'] > row['EMA200']),
        'ema_bearish': (row['EMA20'] < row['EMA50']) and (row['EMA50'] < row['EMA200']),
    })

result_df = pd.DataFrame(results)
result_df = result_df.dropna()

print(f"분석 대상: {len(result_df)}건")
print()

# 각 지표별 추세 예측 정확도 테스트
def test_indicator(df, indicator, threshold, direction, trend_type):
    """지표 기준으로 필터링 후 추세 적중률과 수익 계산"""
    if direction == 'above':
        subset = df[df[indicator] > threshold]
    else:
        subset = df[df[indicator] < threshold]
    
    if len(subset) < 20:
        return None
    
    trend_accuracy = (subset['actual_trend'] == trend_type).mean() * 100
    
    if trend_type == 'UPTREND':
        pnl = subset.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    else:
        pnl = subset.apply(lambda x: x['extended_pnl'] if x['direction'] == 'SHORT' else -x['extended_pnl'], axis=1)
    
    win_rate = (pnl > 0).mean() * 100
    avg_pnl = pnl.mean()
    
    return {
        'indicator': indicator,
        'condition': f'{direction} {threshold}',
        'count': len(subset),
        'trend_accuracy': trend_accuracy,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    }

print("=" * 70)
print("1. UPTREND 예측 지표 테스트")
print("=" * 70)

uptrend_tests = []

# EMA 기울기
for thresh in [0, 0.5, 1, 2]:
    r = test_indicator(result_df, 'ema200_slope', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)

# 모멘텀
for thresh in [0, 5, 10, 15, 20]:
    r = test_indicator(result_df, 'momentum_50', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)
    r = test_indicator(result_df, 'momentum_100', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)

# RSI
for thresh in [50, 55, 60]:
    r = test_indicator(result_df, 'rsi', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)

# 양봉 비율
for thresh in [0.5, 0.55, 0.6]:
    r = test_indicator(result_df, 'up_ratio_50', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)

# MACD
r = test_indicator(result_df, 'macd_hist', 0, 'above', 'UPTREND')
if r: uptrend_tests.append(r)

# BB position
for thresh in [0.5, 0.6, 0.7]:
    r = test_indicator(result_df, 'bb_position', thresh, 'above', 'UPTREND')
    if r: uptrend_tests.append(r)

# 정렬 (추세 적중률 기준)
uptrend_tests = sorted(uptrend_tests, key=lambda x: x['trend_accuracy'], reverse=True)

print("\n[추세 적중률 TOP 10]")
for r in uptrend_tests[:10]:
    marker = "⭐" if r['win_rate'] >= 60 and r['avg_pnl'] >= 2 else ""
    print(f"{r['indicator']} {r['condition']}: {r['count']}건, "
          f"적중 {r['trend_accuracy']:.1f}%, 승률 {r['win_rate']:.1f}%, 평균 {r['avg_pnl']:.2f}% {marker}")

# 정렬 (평균수익 기준)
uptrend_tests_pnl = sorted(uptrend_tests, key=lambda x: x['avg_pnl'], reverse=True)

print("\n[평균수익 TOP 10]")
for r in uptrend_tests_pnl[:10]:
    marker = "⭐" if r['win_rate'] >= 60 and r['avg_pnl'] >= 2 else ""
    print(f"{r['indicator']} {r['condition']}: {r['count']}건, "
          f"적중 {r['trend_accuracy']:.1f}%, 승률 {r['win_rate']:.1f}%, 평균 {r['avg_pnl']:.2f}% {marker}")

print()
print("=" * 70)
print("2. 복합 조건 테스트")
print("=" * 70)

# 복합 조건들
conditions = [
    ('EMA200 기울기↑ + 모멘텀50↑', 
     (result_df['ema200_slope'] > 0) & (result_df['momentum_50'] > 0)),
    ('EMA200 기울기↑ + 모멘텀100↑', 
     (result_df['ema200_slope'] > 0) & (result_df['momentum_100'] > 0)),
    ('모멘텀50 > 10% + RSI > 50', 
     (result_df['momentum_50'] > 10) & (result_df['rsi'] > 50)),
    ('모멘텀100 > 10% + 정배열', 
     (result_df['momentum_100'] > 10) & (result_df['ema_bullish'] == True)),
    ('EMA200↑ + 기울기↑ + 모멘텀↑', 
     (result_df['above_ema200'] == True) & (result_df['ema200_slope'] > 0) & (result_df['momentum_50'] > 0)),
    ('양봉비율 > 55% + EMA200↑', 
     (result_df['up_ratio_50'] > 0.55) & (result_df['above_ema200'] == True)),
    ('모멘텀50 > 15% + 모멘텀100 > 20%', 
     (result_df['momentum_50'] > 15) & (result_df['momentum_100'] > 20)),
    ('MACD↑ + RSI > 55 + EMA200↑', 
     (result_df['macd_hist'] > 0) & (result_df['rsi'] > 55) & (result_df['above_ema200'] == True)),
]

print("\n[복합 조건 → LONG 성과]")
for name, cond in conditions:
    subset = result_df[cond]
    if len(subset) < 20:
        continue
    
    trend_acc = (subset['actual_trend'] == 'UPTREND').mean() * 100
    pnl = subset.apply(lambda x: x['extended_pnl'] if x['direction'] == 'LONG' else -x['extended_pnl'], axis=1)
    win_rate = (pnl > 0).mean() * 100
    avg_pnl = pnl.mean()
    
    marker = "⭐" if trend_acc >= 50 or (win_rate >= 65 and avg_pnl >= 3) else ""
    print(f"{name}")
    print(f"  {len(subset)}건, 적중 {trend_acc:.1f}%, 승률 {win_rate:.1f}%, 평균 {avg_pnl:.2f}% {marker}")
    print()

# 저장
result_df.to_csv('trend_indicators_full.csv', index=False)
print("✅ 저장: trend_indicators_full.csv")

