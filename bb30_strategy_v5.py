import pandas as pd
import numpy as np

df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# BB 30
BB_PERIOD = 30
df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 수축 구간
squeeze_ranges = []
in_squeeze = False
squeeze_start = 0

for i in range(60, len(df)):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_ranges.append((squeeze_start, i))

print(f"수축 구간: {len(squeeze_ranges)}개")

# 스윙 포인트 찾기
def find_swing_points(data, window=2):
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        # 스윙 하이
        if all(data.iloc[i]['high'] >= data.iloc[i-j]['high'] for j in range(1, window+1)) and \
           all(data.iloc[i]['high'] >= data.iloc[i+j]['high'] for j in range(1, window+1)):
            highs.append({'idx': i, 'price': data.iloc[i]['high']})
        
        # 스윙 로우
        if all(data.iloc[i]['low'] <= data.iloc[i-j]['low'] for j in range(1, window+1)) and \
           all(data.iloc[i]['low'] <= data.iloc[i+j]['low'] for j in range(1, window+1)):
            lows.append({'idx': i, 'price': data.iloc[i]['low']})
    
    return highs, lows

trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 5:
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end].reset_index(drop=True)
    
    # 스윙 포인트 찾기
    highs, lows = find_swing_points(squeeze_data, window=2)
    
    # 수축 범위
    squeeze_high = squeeze_data['high'].max()
    squeeze_low = squeeze_data['low'].min()
    
    # === LH/HL 패턴 판단 ===
    has_lh = False  # Lower High
    has_hl = False  # Higher Low
    last_high = None
    last_low = None
    
    if len(highs) >= 2:
        # 마지막 2개 고점 비교
        last_high = highs[-1]['price']
        prev_high = highs[-2]['price']
        has_lh = last_high < prev_high  # Lower High
    
    if len(lows) >= 2:
        # 마지막 2개 저점 비교
        last_low = lows[-1]['price']
        prev_low = lows[-2]['price']
        has_hl = last_low > prev_low  # Higher Low
    
    # === 방향 결정 ===
    # LH면 → 상방 돌파 시 LONG (저항 돌파)
    # HL면 → 하방 이탈 시 SHORT (지지 이탈)
    
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    pattern = None
    
    for i in range(sq_end, min(sq_end + 10, len(df))):
        candle = df.iloc[i]
        
        # LH 패턴 + 상단 돌파 → LONG
        if has_lh and candle['close'] > squeeze_high:
            entry_idx = i
            direction = 'LONG'
            entry_price = candle['close']
            stop_loss = squeeze_low
            pattern = 'LH_BREAK_UP'
            break
        
        # HL 패턴 + 하단 이탈 → SHORT
        if has_hl and candle['close'] < squeeze_low:
            entry_idx = i
            direction = 'SHORT'
            entry_price = candle['close']
            stop_loss = squeeze_high
            pattern = 'HL_BREAK_DOWN'
            break
    
    if entry_idx is None:
        continue
    
    # 손절폭
    if direction == 'LONG':
        sl_pct = (entry_price - stop_loss) / entry_price * 100
    else:
        sl_pct = (stop_loss - entry_price) / entry_price * 100
    
    # 다음 수축까지
    next_squeeze_start = None
    for next_sq_start, _ in squeeze_ranges[sq_idx+1:]:
        if next_sq_start > entry_idx + 3:
            next_squeeze_start = next_sq_start
            break
    
    if next_squeeze_start is None:
        next_squeeze_start = min(entry_idx + 100, len(df) - 1)
    
    # 추적
    exit_price = None
    exit_reason = None
    exit_idx = None
    max_profit = 0
    
    for i in range(entry_idx + 1, min(next_squeeze_start + 1, len(df))):
        candle = df.iloc[i]
        
        if direction == 'LONG':
            if candle['close'] < stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            max_profit = max(max_profit, (candle['high'] - entry_price) / entry_price * 100)
        else:
            if candle['close'] > stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            max_profit = max(max_profit, (entry_price - candle['low']) / entry_price * 100)
    
    if exit_reason is None:
        exit_idx = min(next_squeeze_start, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        exit_reason = 'NEXT_SQUEEZE'
    
    if direction == 'LONG':
        final_pnl = (exit_price - entry_price) / entry_price * 100
    else:
        final_pnl = (entry_price - exit_price) / entry_price * 100
    
    trades.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'direction': direction,
        'pattern': pattern,
        'has_lh': has_lh,
        'has_hl': has_hl,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'max_profit': max_profit,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건")

print("\n" + "="*80)
print("📊 V5 결과 (LH→LONG, HL→SHORT 방향 필터)")
print("="*80)

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")
print(f"승리 시: +{wins['final_pnl'].mean():.2f}%, 패배 시: {losses['final_pnl'].mean():.2f}%")

print("\n### 청산 사유별")
for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = trades_df[trades_df['exit_reason'] == reason]
    if len(sub) > 0:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {reason}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 패턴별")
for p in trades_df['pattern'].unique():
    sub = trades_df[trades_df['pattern'] == p]
    if len(sub) >= 5:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {p}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 손절폭별")
for low, high in [(0, 2), (2, 4), (4, 6), (6, 10)]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {low}-{high}%: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 수축 길이별")
for low, high in [(5, 10), (10, 20), (20, 40), (40, 100)]:
    sub = trades_df[(trades_df['squeeze_length'] >= low) & (trades_df['squeeze_length'] < high)]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {low}-{high}h: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

trades_df.to_csv('bb30_strategy_v5_results.csv', index=False)
