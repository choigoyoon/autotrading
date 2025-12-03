import pandas as pd
import numpy as np

df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# 지표 계산
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()
df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

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

for i in range(200, len(df)):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_ranges.append((squeeze_start, i))

print(f"수축 구간: {len(squeeze_ranges)}개")

trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 3:
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end]
    squeeze_high = squeeze_data['high'].max()
    squeeze_low = squeeze_data['low'].min()
    
    # 수축 종료 시점 지표
    last_idx = sq_end - 1
    last_close = df.iloc[last_idx]['close']
    last_ema200 = df.iloc[last_idx]['ema200']
    last_ema20 = df.iloc[last_idx]['ema20']
    last_ema50 = df.iloc[last_idx]['ema50']
    last_vwap = df.iloc[last_idx]['vwap']
    
    above_ema200 = last_close > last_ema200
    above_vwap = last_close > last_vwap
    ema_bullish = last_ema20 > last_ema50 > last_ema200
    ema_bearish = last_ema20 < last_ema50 < last_ema200
    
    # === 방향 필터 적용 ===
    # EMA200 위 → LONG만
    # EMA200 아래 → SHORT만
    
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    
    for i in range(sq_end, min(sq_end + 10, len(df))):
        candle = df.iloc[i]
        
        # EMA200 위 + 상단 돌파 → LONG
        if above_ema200 and candle['close'] > squeeze_high:
            entry_idx = i
            direction = 'LONG'
            entry_price = candle['close']
            stop_loss = squeeze_low
            break
        
        # EMA200 아래 + 하단 이탈 → SHORT
        if not above_ema200 and candle['close'] < squeeze_low:
            entry_idx = i
            direction = 'SHORT'
            entry_price = candle['close']
            stop_loss = squeeze_high
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
        'above_ema200': above_ema200,
        'above_vwap': above_vwap,
        'ema_bullish': ema_bullish,
        'ema_bearish': ema_bearish,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_reason': exit_reason,
        'max_profit': max_profit,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건")

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print("\n" + "="*80)
print("📊 최적화 결과 (EMA200 위→LONG, EMA200 아래→SHORT)")
print("="*80)
print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")

print("\n### 방향별")
for d in ['LONG', 'SHORT']:
    sub = trades_df[trades_df['direction'] == d]
    if len(sub) > 0:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {d}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n### 청산 사유별")
for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = trades_df[trades_df['exit_reason'] == reason]
    if len(sub) > 0:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {reason}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 추가 필터 테스트
print("\n" + "="*80)
print("🔍 추가 필터 테스트")
print("="*80)

# LONG만 (EMA200 위)
print("\n### LONG만 (EMA200 위에서만 진입)")
long_only = trades_df[trades_df['direction'] == 'LONG']
if len(long_only) > 0:
    wr = (long_only['final_pnl'] > 0).mean() * 100
    print(f"  전체: {len(long_only)}건, 승률 {wr:.1f}%, 평균 {long_only['final_pnl'].mean():.2f}%, 총 {long_only['final_pnl'].sum():.1f}%")

# LONG + 정배열
print("\n### LONG + 정배열")
sub = trades_df[(trades_df['direction'] == 'LONG') & (trades_df['ema_bullish'] == True)]
if len(sub) > 0:
    wr = (sub['final_pnl'] > 0).mean() * 100
    print(f"  {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%, 총 {sub['final_pnl'].sum():.1f}%")

# LONG + VWAP 위
print("\n### LONG + VWAP 위")
sub = trades_df[(trades_df['direction'] == 'LONG') & (trades_df['above_vwap'] == True)]
if len(sub) > 0:
    wr = (sub['final_pnl'] > 0).mean() * 100
    print(f"  {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%, 총 {sub['final_pnl'].sum():.1f}%")

# LONG + 정배열 + VWAP 위
print("\n### LONG + 정배열 + VWAP 위")
sub = trades_df[(trades_df['direction'] == 'LONG') & 
                (trades_df['ema_bullish'] == True) & 
                (trades_df['above_vwap'] == True)]
if len(sub) > 0:
    wr = (sub['final_pnl'] > 0).mean() * 100
    print(f"  {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%, 총 {sub['final_pnl'].sum():.1f}%")

# 손절폭별 LONG
print("\n### LONG 손절폭별")
for low, high in [(0, 2), (2, 4), (4, 7)]:
    sub = trades_df[(trades_df['direction'] == 'LONG') & 
                    (trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 10:
        wr = (sub['final_pnl'] > 0).mean() * 100
        print(f"  SL {low}-{high}%: {len(sub)}건, 승률 {wr:.1f}%, 평균 {sub['final_pnl'].mean():.2f}%")

trades_df.to_csv('bb30_optimized_results.csv', index=False)
