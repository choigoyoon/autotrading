import pandas as pd
import numpy as np

"""
MDD 원인 분석
- MDD 발생 구간 찾기
- 연속 손실 케이스 분석
- 잘못된 진입 자리 패턴 식별
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("="*100)
print("📉 MDD 원인 분석")
print("="*100)

# 1. 누적 PNL 및 MDD 계산
trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']

# MDD 구간 찾기
mdd_value = trades_df['drawdown'].min()
mdd_idx = trades_df['drawdown'].idxmin()

print(f"\n전체 MDD: {mdd_value:.2f}%")
print(f"MDD 발생 거래 인덱스: {mdd_idx}")

# 2. MDD 구간 상세 분석
# Drawdown이 시작된 시점 찾기
dd_start = None
for i in range(mdd_idx, -1, -1):
    if trades_df.loc[i, 'drawdown'] >= 0:
        dd_start = i + 1
        break
    if i == 0:
        dd_start = 0

# Drawdown이 회복된 시점 찾기
dd_end = None
for i in range(mdd_idx, len(trades_df)):
    if trades_df.loc[i, 'drawdown'] >= 0:
        dd_end = i
        break
    if i == len(trades_df) - 1:
        dd_end = i

print(f"\nMDD 구간: 거래 #{dd_start} ~ #{dd_end}")

mdd_trades = trades_df.loc[dd_start:mdd_idx].copy()

print(f"\n📊 MDD 구간 거래 수: {len(mdd_trades)}건")
print(f"  시작: {mdd_trades.iloc[0]['entry_time']}")
print(f"  끝: {mdd_trades.iloc[-1]['exit_time']}")
print(f"  기간: {(mdd_trades.iloc[-1]['exit_time'] - mdd_trades.iloc[0]['entry_time']).days}일")

# 3. MDD 구간 거래 상세
print("\n" + "="*100)
print("📋 MDD 구간 거래 상세")
print("="*100)

sl_count = len(mdd_trades[mdd_trades['exit_reason'].str.contains('SL')])
tp_count = len(mdd_trades[mdd_trades['exit_reason'].str.contains('TP')])

print(f"\n손절(SL): {sl_count}건 ({sl_count/len(mdd_trades)*100:.1f}%)")
print(f"익절(TP): {tp_count}건 ({tp_count/len(mdd_trades)*100:.1f}%)")
print(f"평균 손실: {mdd_trades['pnl_pct'].mean():.2f}%")
print(f"총 손실: {mdd_trades['pnl_pct'].sum():.2f}%")

# 4. 연속 손실 분석
print("\n" + "="*100)
print("🔴 연속 손실 분석")
print("="*100)

# 연속 손실 streak 계산
trades_df['is_loss'] = trades_df['pnl_pct'] < 0
trades_df['loss_streak'] = 0

current_streak = 0
max_streak = 0
max_streak_start = 0
max_streak_end = 0

streaks = []
streak_start = None

for i, row in trades_df.iterrows():
    if row['is_loss']:
        if streak_start is None:
            streak_start = i
        current_streak += 1
    else:
        if current_streak > 0:
            streaks.append({
                'start_idx': streak_start,
                'end_idx': i - 1,
                'length': current_streak,
                'total_loss': trades_df.loc[streak_start:i-1, 'pnl_pct'].sum()
            })
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_start = streak_start
                max_streak_end = i - 1
        current_streak = 0
        streak_start = None

# 마지막 streak 처리
if current_streak > 0:
    streaks.append({
        'start_idx': streak_start,
        'end_idx': len(trades_df) - 1,
        'length': current_streak,
        'total_loss': trades_df.loc[streak_start:, 'pnl_pct'].sum()
    })

# 상위 10개 연속 손실
streaks_df = pd.DataFrame(streaks)
streaks_df = streaks_df.sort_values('total_loss').head(10)

print(f"\n최대 연속 손실: {max_streak}회 연속")
print(f"최대 연속 손실 구간: 거래 #{max_streak_start} ~ #{max_streak_end}")

print("\n상위 10개 연속 손실 구간:")
print(f"{'구간':<15} {'연속':>8} {'총손실':>10} {'시작시간':<25}")
print("-"*70)

for _, s in streaks_df.iterrows():
    start_time = trades_df.loc[s['start_idx'], 'entry_time']
    print(f"#{int(s['start_idx'])}-#{int(s['end_idx']):<5} {int(s['length']):>8}회 {s['total_loss']:>9.2f}% {str(start_time):<25}")

# 5. 연속 손실 구간 시장 상황 분석
print("\n" + "="*100)
print("📈 연속 손실 구간 시장 상황 분석")
print("="*100)

# 가장 큰 손실 연속 구간 분석
worst_streak = streaks_df.iloc[0]
worst_start = int(worst_streak['start_idx'])
worst_end = int(worst_streak['end_idx'])
worst_trades = trades_df.loc[worst_start:worst_end]

print(f"\n가장 큰 연속 손실 구간: #{worst_start} ~ #{worst_end}")
print(f"  연속: {int(worst_streak['length'])}회")
print(f"  총 손실: {worst_streak['total_loss']:.2f}%")

# 해당 기간 시장 상황
start_time = worst_trades.iloc[0]['entry_time']
end_time = worst_trades.iloc[-1]['exit_time']

market_data = candles_df[(candles_df['datetime'] >= start_time) & 
                         (candles_df['datetime'] <= end_time)]

if len(market_data) > 0:
    price_start = market_data.iloc[0]['close']
    price_end = market_data.iloc[-1]['close']
    price_change = ((price_end - price_start) / price_start) * 100
    
    print(f"\n시장 상황:")
    print(f"  기간: {start_time} ~ {end_time}")
    print(f"  BTC 가격: ${price_start:.0f} → ${price_end:.0f} ({price_change:+.1f}%)")
    
    # 최고/최저
    price_high = market_data['high'].max()
    price_low = market_data['low'].min()
    volatility = ((price_high - price_low) / price_low) * 100
    print(f"  최고: ${price_high:.0f}, 최저: ${price_low:.0f}")
    print(f"  변동폭: {volatility:.1f}%")

# 6. 진입 자리 문제점 분석
print("\n" + "="*100)
print("🎯 진입 자리 문제점 분석")
print("="*100)

# 기술적 지표 계산
candles_df['ema_20'] = candles_df['close'].ewm(span=20, adjust=False).mean()
candles_df['ema_50'] = candles_df['close'].ewm(span=50, adjust=False).mean()
candles_df['ema_200'] = candles_df['close'].ewm(span=200, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

candles_df['rsi'] = calculate_rsi(candles_df['close'], 14)
candles_df['change_4h'] = ((candles_df['close'] - candles_df['close'].shift(16)) / candles_df['close'].shift(16)) * 100

# 손실 거래 진입 시점 분석
candles_df.set_index('datetime', inplace=True)

loss_trades = trades_df[trades_df['pnl_pct'] < 0].copy()
win_trades = trades_df[trades_df['pnl_pct'] > 0].copy()

def get_entry_conditions(trade_df, candles):
    conditions = []
    for _, trade in trade_df.iterrows():
        entry_time = trade['entry_time']
        try:
            idx = candles.index.get_indexer([entry_time], method='nearest')[0]
            candle = candles.iloc[idx]
            
            conditions.append({
                'pnl': trade['pnl_pct'],
                'ema_trend': 'UP' if candle['ema_20'] > candle['ema_50'] > candle['ema_200'] else ('DOWN' if candle['ema_20'] < candle['ema_50'] else 'MIXED'),
                'rsi': candle['rsi'],
                'above_ema200': candle['close'] > candle['ema_200'],
                'above_ema50': candle['close'] > candle['ema_50'],
                'above_ema20': candle['close'] > candle['ema_20'],
                'change_4h': candle['change_4h'],
                'dist_from_ema200': ((candle['close'] - candle['ema_200']) / candle['ema_200']) * 100 if candle['ema_200'] > 0 else 0
            })
        except:
            pass
    return pd.DataFrame(conditions)

print("\n손실 거래 진입 조건 분석 중...")
loss_conditions = get_entry_conditions(loss_trades, candles_df)
win_conditions = get_entry_conditions(win_trades, candles_df)

print(f"\n{'조건':<25} {'손실 거래':>15} {'승리 거래':>15} {'차이':>15}")
print("-"*70)

# EMA 트렌드
for trend in ['UP', 'DOWN', 'MIXED']:
    loss_pct = (loss_conditions['ema_trend'] == trend).sum() / len(loss_conditions) * 100 if len(loss_conditions) > 0 else 0
    win_pct = (win_conditions['ema_trend'] == trend).sum() / len(win_conditions) * 100 if len(win_conditions) > 0 else 0
    diff = win_pct - loss_pct
    print(f"EMA {trend:<20} {loss_pct:>14.1f}% {win_pct:>14.1f}% {diff:>+14.1f}%p")

# RSI
loss_rsi = loss_conditions['rsi'].mean() if len(loss_conditions) > 0 else 0
win_rsi = win_conditions['rsi'].mean() if len(win_conditions) > 0 else 0
print(f"{'RSI 평균':<25} {loss_rsi:>14.1f} {win_rsi:>14.1f} {win_rsi-loss_rsi:>+14.1f}")

# EMA200 위
loss_above = loss_conditions['above_ema200'].sum() / len(loss_conditions) * 100 if len(loss_conditions) > 0 else 0
win_above = win_conditions['above_ema200'].sum() / len(win_conditions) * 100 if len(win_conditions) > 0 else 0
print(f"{'EMA200 위':<25} {loss_above:>14.1f}% {win_above:>14.1f}% {win_above-loss_above:>+14.1f}%p")

# 4시간 가격 변화
loss_change = loss_conditions['change_4h'].mean() if len(loss_conditions) > 0 else 0
win_change = win_conditions['change_4h'].mean() if len(win_conditions) > 0 else 0
print(f"{'4시간 가격변화':<25} {loss_change:>14.2f}% {win_change:>14.2f}% {win_change-loss_change:>+14.2f}%p")

# EMA200 거리
loss_dist = loss_conditions['dist_from_ema200'].mean() if len(loss_conditions) > 0 else 0
win_dist = win_conditions['dist_from_ema200'].mean() if len(win_conditions) > 0 else 0
print(f"{'EMA200 거리':<25} {loss_dist:>14.2f}% {win_dist:>14.2f}% {win_dist-loss_dist:>+14.2f}%p")

# 7. 문제 진입 패턴 식별
print("\n" + "="*100)
print("🚨 문제 진입 패턴 (손실 비율 높은 조건)")
print("="*100)

# RSI 구간별 분석
print("\nRSI 구간별 손실률:")
for rsi_min, rsi_max in [(0, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    loss_in_range = len(loss_conditions[(loss_conditions['rsi'] >= rsi_min) & (loss_conditions['rsi'] < rsi_max)])
    win_in_range = len(win_conditions[(win_conditions['rsi'] >= rsi_min) & (win_conditions['rsi'] < rsi_max)])
    total = loss_in_range + win_in_range
    if total > 10:
        loss_rate = loss_in_range / total * 100
        flag = "🔴 위험" if loss_rate > 50 else "✅ 안전" if loss_rate < 35 else ""
        print(f"  RSI {rsi_min}-{rsi_max}: 손실 {loss_in_range}건, 승리 {win_in_range}건, 손실률 {loss_rate:.1f}% {flag}")

# EMA 트렌드별 분석
print("\nEMA 트렌드별 손실률:")
for trend in ['UP', 'DOWN', 'MIXED']:
    loss_in_trend = len(loss_conditions[loss_conditions['ema_trend'] == trend])
    win_in_trend = len(win_conditions[win_conditions['ema_trend'] == trend])
    total = loss_in_trend + win_in_trend
    if total > 10:
        loss_rate = loss_in_trend / total * 100
        flag = "🔴 위험" if loss_rate > 50 else "✅ 안전" if loss_rate < 35 else ""
        print(f"  EMA {trend}: 손실 {loss_in_trend}건, 승리 {win_in_trend}건, 손실률 {loss_rate:.1f}% {flag}")

# 4시간 변화 구간별
print("\n4시간 가격변화별 손실률:")
for change_min, change_max in [(-100, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 100)]:
    loss_in_range = len(loss_conditions[(loss_conditions['change_4h'] >= change_min) & (loss_conditions['change_4h'] < change_max)])
    win_in_range = len(win_conditions[(win_conditions['change_4h'] >= change_min) & (win_conditions['change_4h'] < change_max)])
    total = loss_in_range + win_in_range
    if total > 10:
        loss_rate = loss_in_range / total * 100
        flag = "🔴 위험" if loss_rate > 50 else "✅ 안전" if loss_rate < 35 else ""
        print(f"  변화 {change_min}~{change_max}%: 손실 {loss_in_range}건, 승리 {win_in_range}건, 손실률 {loss_rate:.1f}% {flag}")

# 8. 결론 및 개선안
print("\n" + "="*100)
print("💡 결론 및 개선안")
print("="*100)

# 가장 위험한 조건 찾기
dangerous_conditions = []

# RSI 낮은 구간
low_rsi_loss = len(loss_conditions[loss_conditions['rsi'] < 50])
low_rsi_win = len(win_conditions[win_conditions['rsi'] < 50])
if low_rsi_loss + low_rsi_win > 0:
    low_rsi_rate = low_rsi_loss / (low_rsi_loss + low_rsi_win) * 100
    if low_rsi_rate > 45:
        dangerous_conditions.append(f"RSI < 50 진입 (손실률 {low_rsi_rate:.1f}%)")

# EMA 역배열
down_loss = len(loss_conditions[loss_conditions['ema_trend'] == 'DOWN'])
down_win = len(win_conditions[win_conditions['ema_trend'] == 'DOWN'])
if down_loss + down_win > 0:
    down_rate = down_loss / (down_loss + down_win) * 100
    if down_rate > 45:
        dangerous_conditions.append(f"EMA 역배열 진입 (손실률 {down_rate:.1f}%)")

# 하락 모멘텀
neg_loss = len(loss_conditions[loss_conditions['change_4h'] < -1])
neg_win = len(win_conditions[win_conditions['change_4h'] < -1])
if neg_loss + neg_win > 0:
    neg_rate = neg_loss / (neg_loss + neg_win) * 100
    if neg_rate > 45:
        dangerous_conditions.append(f"4시간 -1% 이상 하락 중 진입 (손실률 {neg_rate:.1f}%)")

if dangerous_conditions:
    print("\n🚨 위험한 진입 조건 (피해야 할 패턴):")
    for cond in dangerous_conditions:
        print(f"  • {cond}")
    
    print("\n✅ 개선안:")
    print("  1. RSI 50 이상에서만 진입")
    print("  2. EMA 정배열 또는 MIXED에서만 진입 (역배열 금지)")
    print("  3. 4시간 -1% 이상 하락 중 진입 금지")
    print("  4. 위 조건 적용 시 MDD 개선 예상")
else:
    print("\n명확한 위험 패턴 미발견. 추가 분석 필요.")

# MDD 구간 거래 저장
mdd_trades.to_csv('mdd_trades_analysis.csv', index=False)
print(f"\n✅ MDD 구간 거래 저장: mdd_trades_analysis.csv")
