import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# BB 30 계산
BB_PERIOD = 30
df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

# 수축 상태
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 수축 구간 찾기
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

# 매매 분석 - 필터 완화 버전
trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 3:  # 최소 3봉으로 완화
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end]
    
    # 수축 구간의 고점/저점 (단순화)
    squeeze_high = squeeze_data['high'].max()
    squeeze_low = squeeze_data['low'].min()
    squeeze_range_pct = (squeeze_high - squeeze_low) / squeeze_low * 100
    
    # 마지막 캔들 방향
    last_candle = df.iloc[sq_end - 1]
    last_direction = 'UP' if last_candle['close'] > last_candle['open'] else 'DOWN'
    
    # === 돌파 확인 ===
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    
    for i in range(sq_end, min(sq_end + 10, len(df))):
        candle = df.iloc[i]
        
        # 상단 돌파 → LONG
        if candle['close'] > squeeze_high:
            entry_idx = i
            direction = 'LONG'
            entry_price = candle['close']
            stop_loss = squeeze_low  # 수축 저점이 손절
            break
        
        # 하단 이탈 → SHORT
        if candle['close'] < squeeze_low:
            entry_idx = i
            direction = 'SHORT'
            entry_price = candle['close']
            stop_loss = squeeze_high  # 수축 고점이 손절
            break
    
    if entry_idx is None:
        continue
    
    # 손절폭 계산
    if direction == 'LONG':
        sl_pct = (entry_price - stop_loss) / entry_price * 100
    else:
        sl_pct = (stop_loss - entry_price) / entry_price * 100
    
    # 다음 수축까지 추적
    next_squeeze_start = None
    for next_sq_start, next_sq_end in squeeze_ranges[sq_idx+1:]:
        if next_sq_start > entry_idx + 3:
            next_squeeze_start = next_sq_start
            break
    
    if next_squeeze_start is None:
        next_squeeze_start = min(entry_idx + 100, len(df) - 1)
    
    # 진입 후 추적
    max_profit = 0
    max_loss = 0
    exit_price = None
    exit_reason = None
    exit_idx = None
    
    for i in range(entry_idx + 1, min(next_squeeze_start + 1, len(df))):
        candle = df.iloc[i]
        
        if direction == 'LONG':
            # 손절 체크
            if candle['close'] < stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            
            max_profit = max(max_profit, (candle['high'] - entry_price) / entry_price * 100)
            max_loss = min(max_loss, (candle['low'] - entry_price) / entry_price * 100)
            
        else:
            if candle['close'] > stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            
            max_profit = max(max_profit, (entry_price - candle['low']) / entry_price * 100)
            max_loss = min(max_loss, (entry_price - candle['high']) / entry_price * 100)
    
    if exit_reason is None:
        exit_idx = min(next_squeeze_start, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        exit_reason = 'NEXT_SQUEEZE'
    
    if direction == 'LONG':
        final_pnl = (exit_price - entry_price) / entry_price * 100
    else:
        final_pnl = (entry_price - exit_price) / entry_price * 100
    
    hold_hours = exit_idx - entry_idx
    
    trades.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'direction': direction,
        'last_direction': last_direction,
        'entry_price': entry_price,
        'squeeze_high': squeeze_high,
        'squeeze_low': squeeze_low,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_hours': hold_hours,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length,
        'squeeze_range_pct': squeeze_range_pct
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건 (수축 {len(squeeze_ranges)}개 중)")

print("\n" + "="*80)
print("📊 전략 결과 V4 (필터 완화 - 기본 돌파)")
print("="*80)

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")

if len(wins) > 0:
    print(f"승리 시 평균: +{wins['final_pnl'].mean():.2f}%")
if len(losses) > 0:
    print(f"패배 시 평균: {losses['final_pnl'].mean():.2f}%")

# 청산 사유별
print("\n### 청산 사유별")
for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = trades_df[trades_df['exit_reason'] == reason]
    if len(sub) > 0:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {reason}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 방향별
print("\n### 방향별")
for d in ['LONG', 'SHORT']:
    sub = trades_df[trades_df['direction'] == d]
    if len(sub) > 0:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {d}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 수축 길이별
print("\n### 수축 길이별")
for low, high, label in [(3, 10, '3-9h'), (10, 20, '10-19h'), (20, 40, '20-39h'), (40, 100, '40h+')]:
    sub = trades_df[(trades_df['squeeze_length'] >= low) & (trades_df['squeeze_length'] < high)]
    if len(sub) >= 10:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 손절폭별
print("\n### 손절폭별")
for low, high, label in [(0, 2, '0-2%'), (2, 4, '2-4%'), (4, 6, '4-6%'), (6, 10, '6-10%'), (10, 100, '10%+')]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 10:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 수축 범위별
print("\n### 수축 범위별 (수축 중 가격 변동폭)")
for low, high, label in [(0, 1, '0-1%'), (1, 2, '1-2%'), (2, 4, '2-4%'), (4, 8, '4-8%'), (8, 100, '8%+')]:
    sub = trades_df[(trades_df['squeeze_range_pct'] >= low) & (trades_df['squeeze_range_pct'] < high)]
    if len(sub) >= 10:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 누적 수익
trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적 수익: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

trades_df.to_csv('bb30_strategy_v4_results.csv', index=False)
print(f"\n결과 저장: bb30_strategy_v4_results.csv")
