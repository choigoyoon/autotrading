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

# 간단한 LH/HL 판단 (롤링 고점/저점)
def simple_lh_hl(data):
    """수축 구간 데이터에서 LH/HL 판단"""
    if len(data) < 6:
        return None, None
    
    # 전반부와 후반부로 나눠서 비교
    mid = len(data) // 2
    first_half = data.iloc[:mid]
    second_half = data.iloc[mid:]
    
    first_high = first_half['high'].max()
    second_high = second_half['high'].max()
    first_low = first_half['low'].min()
    second_low = second_half['low'].min()
    
    has_lh = second_high < first_high  # 후반부 고점 < 전반부 고점
    has_hl = second_low > first_low    # 후반부 저점 > 전반부 저점
    
    return has_lh, has_hl

trades = []
debug_info = {'no_pattern': 0, 'no_breakout': 0, 'traded': 0}

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 4:
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end]
    
    # 수축 범위
    squeeze_high = squeeze_data['high'].max()
    squeeze_low = squeeze_data['low'].min()
    
    # LH/HL 판단 (간단 버전)
    has_lh, has_hl = simple_lh_hl(squeeze_data)
    
    # 수축 종료 시점 BB 위치
    last_close = df.iloc[sq_end-1]['close']
    last_bb_mid = df.iloc[sq_end-1]['bb_mid']
    bb_position = 'UPPER' if last_close > last_bb_mid else 'LOWER'
    
    # 마지막 3봉 방향
    last_3 = squeeze_data.tail(3)
    up_count = (last_3['close'] > last_3['open']).sum()
    last_trend = 'UP' if up_count >= 2 else 'DOWN'
    
    # === 진입 조건 ===
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    pattern = None
    
    for i in range(sq_end, min(sq_end + 10, len(df))):
        candle = df.iloc[i]
        
        # 상단 돌파
        if candle['close'] > squeeze_high:
            # LH + 상단돌파 → LONG (역추세 돌파)
            if has_lh:
                entry_idx = i
                direction = 'LONG'
                entry_price = candle['close']
                stop_loss = squeeze_low
                pattern = 'LH_UP'
                break
        
        # 하단 이탈
        if candle['close'] < squeeze_low:
            # HL + 하단이탈 → SHORT (역추세 이탈)
            if has_hl:
                entry_idx = i
                direction = 'SHORT'
                entry_price = candle['close']
                stop_loss = squeeze_high
                pattern = 'HL_DOWN'
                break
    
    if entry_idx is None:
        debug_info['no_breakout'] += 1
        continue
    
    debug_info['traded'] += 1
    
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
    max_profit = 0
    
    for i in range(entry_idx + 1, min(next_squeeze_start + 1, len(df))):
        candle = df.iloc[i]
        
        if direction == 'LONG':
            if candle['close'] < stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                break
            max_profit = max(max_profit, (candle['high'] - entry_price) / entry_price * 100)
        else:
            if candle['close'] > stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                break
            max_profit = max(max_profit, (entry_price - candle['low']) / entry_price * 100)
    
    if exit_reason is None:
        exit_price = df.iloc[min(next_squeeze_start, len(df) - 1)]['close']
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
        'bb_position': bb_position,
        'last_trend': last_trend,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_reason': exit_reason,
        'max_profit': max_profit,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"\n디버그: 패턴없음={debug_info['no_pattern']}, 돌파없음={debug_info['no_breakout']}, 거래={debug_info['traded']}")
print(f"총 매매: {len(trades_df)}건")

if len(trades_df) == 0:
    exit()

print("\n" + "="*80)
print("📊 V6 결과 (간단 LH/HL 판단)")
print("="*80)

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")

print("\n### 청산 사유별")
for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = trades_df[trades_df['exit_reason'] == reason]
    if len(sub) > 0:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {reason}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 패턴별")
for p in trades_df['pattern'].unique():
    sub = trades_df[trades_df['pattern'] == p]
    wr = (sub['final_pnl'] > 0).mean() * 100
    print(f"  {p}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### BB 위치별")
for pos in ['UPPER', 'LOWER']:
    sub = trades_df[trades_df['bb_position'] == pos]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {pos}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 마지막추세별")
for trend in ['UP', 'DOWN']:
    sub = trades_df[trades_df['last_trend'] == trend]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {trend}: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 손절폭별")
for low, high in [(0, 2), (2, 4), (4, 6), (6, 15)]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {low}-{high}%: {len(sub)}건, 승률 {wr:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

trades_df.to_csv('bb30_strategy_v6_results.csv', index=False)
