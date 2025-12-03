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

# 스윙 포인트 찾기
def find_swings(data, window=2):
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        is_high = all(data.iloc[i]['high'] > data.iloc[i-j]['high'] and 
                      data.iloc[i]['high'] > data.iloc[i+j]['high'] 
                      for j in range(1, window+1))
        if is_high:
            highs.append({'idx': i, 'price': data.iloc[i]['high']})
        
        is_low = all(data.iloc[i]['low'] < data.iloc[i-j]['low'] and 
                     data.iloc[i]['low'] < data.iloc[i+j]['low'] 
                     for j in range(1, window+1))
        if is_low:
            lows.append({'idx': i, 'price': data.iloc[i]['low']})
    
    return highs, lows

# 수축 구간 찾기 (시작, 끝 인덱스)
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

# 매매 분석
trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 10:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 6:  # 최소 6봉
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end].reset_index(drop=True)
    
    # 스윙 포인트 찾기
    highs, lows = find_swings(squeeze_data, window=2)
    
    if len(highs) < 1 or len(lows) < 1:
        continue
    
    # 마지막 고점(저항), 마지막 저점(지지)
    last_high_price = highs[-1]['price']  # 저항선
    last_low_price = lows[-1]['price']    # 지지선
    
    # LH/HL 판단 (2개 이상일 때)
    last_high_pattern = None
    last_low_pattern = None
    
    if len(highs) >= 2:
        last_high_pattern = 'LH' if highs[-1]['price'] < highs[-2]['price'] else 'HH'
    if len(lows) >= 2:
        last_low_pattern = 'HL' if lows[-1]['price'] > lows[-2]['price'] else 'LL'
    
    # === 돌파 확인 ===
    # 확장 시작 후 LH 돌파 or HL 이탈 확인
    
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    
    for i in range(sq_end, min(sq_end + 20, len(df))):  # 20봉 내 돌파 확인
        candle = df.iloc[i]
        
        # LH 저항 돌파 → LONG
        if candle['close'] > last_high_price:
            entry_idx = i
            direction = 'LONG'
            entry_price = candle['close']
            stop_loss = last_low_price  # 지지선이 손절
            break
        
        # HL 지지 이탈 → SHORT
        if candle['close'] < last_low_price:
            entry_idx = i
            direction = 'SHORT'
            entry_price = candle['close']
            stop_loss = last_high_price  # 저항선이 손절
            break
    
    if entry_idx is None:
        continue
    
    # 손절폭 계산
    if direction == 'LONG':
        sl_pct = (entry_price - stop_loss) / entry_price * 100
    else:
        sl_pct = (stop_loss - entry_price) / entry_price * 100
    
    # === 다음 수축까지 추적 ===
    # 다음 수축 구간 찾기
    next_squeeze_start = None
    for next_sq_start, next_sq_end in squeeze_ranges[sq_idx+1:]:
        if next_sq_start > entry_idx:
            next_squeeze_start = next_sq_start
            break
    
    if next_squeeze_start is None:
        next_squeeze_start = len(df) - 1
    
    # 진입 후 다음 수축까지 추적
    max_profit = 0
    max_loss = 0
    exit_price = None
    exit_reason = None
    exit_idx = None
    
    for i in range(entry_idx + 1, min(next_squeeze_start + 1, len(df))):
        candle = df.iloc[i]
        
        if direction == 'LONG':
            # 손절 체크
            if candle['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'SL'
                exit_idx = i
                break
            
            # 현재 수익
            current_pnl = (candle['close'] - entry_price) / entry_price * 100
            max_profit = max(max_profit, (candle['high'] - entry_price) / entry_price * 100)
            max_loss = min(max_loss, (candle['low'] - entry_price) / entry_price * 100)
            
        else:  # SHORT
            # 손절 체크
            if candle['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'SL'
                exit_idx = i
                break
            
            # 현재 수익
            current_pnl = (entry_price - candle['close']) / entry_price * 100
            max_profit = max(max_profit, (entry_price - candle['low']) / entry_price * 100)
            max_loss = min(max_loss, (entry_price - candle['high']) / entry_price * 100)
    
    # 손절 안됐으면 다음 수축에서 청산
    if exit_reason is None:
        exit_idx = min(next_squeeze_start, len(df) - 1)
        exit_price = df.iloc[exit_idx]['close']
        exit_reason = 'NEXT_SQUEEZE'
        
        if direction == 'LONG':
            final_pnl = (exit_price - entry_price) / entry_price * 100
        else:
            final_pnl = (entry_price - exit_price) / entry_price * 100
    else:
        final_pnl = -sl_pct  # 손절
    
    # 보유 기간
    hold_hours = exit_idx - entry_idx
    
    trades.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'direction': direction,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_hours': hold_hours,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_pnl': final_pnl,
        'last_high_pattern': last_high_pattern,
        'last_low_pattern': last_low_pattern,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건")

print("\n" + "="*80)
print("📊 전략 결과")
print("="*80)

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")
print(f"평균 보유 시간: {trades_df['hold_hours'].mean():.1f}시간")

print(f"\n승리 시 평균: +{wins['final_pnl'].mean():.2f}%")
print(f"패배 시 평균: {losses['final_pnl'].mean():.2f}%")

# 청산 사유별
print("\n### 청산 사유별")
for reason in ['NEXT_SQUEEZE', 'SL']:
    sub = trades_df[trades_df['exit_reason'] == reason]
    if len(sub) > 0:
        print(f"  {reason}: {len(sub)}건, 평균 {sub['final_pnl'].mean():.2f}%")

# 방향별
print("\n### 방향별")
for d in ['LONG', 'SHORT']:
    sub = trades_df[trades_df['direction'] == d]
    if len(sub) > 0:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {d}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# LH/HL 패턴별
print("\n### 패턴별")
print("\nLONG (LH 돌파):")
for pattern in ['LH', 'HH', None]:
    sub = trades_df[(trades_df['direction'] == 'LONG') & (trades_df['last_high_pattern'] == pattern)]
    if len(sub) >= 2:
        label = pattern if pattern else '없음'
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\nSHORT (HL 이탈):")
for pattern in ['HL', 'LL', None]:
    sub = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['last_low_pattern'] == pattern)]
    if len(sub) >= 2:
        label = pattern if pattern else '없음'
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 손절폭별
print("\n### 손절폭별")
for low, high, label in [(0, 1, '0-1%'), (1, 2, '1-2%'), (2, 3, '2-3%'), (3, 5, '3-5%'), (5, 100, '5%+')]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 3:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 최근 20건 상세
print("\n" + "="*80)
print("📋 최근 20건 상세")
print("="*80)
print(f"{'날짜':^16} | {'방향':^5} | {'진입가':^8} | {'손절':^8} | {'SL%':^5} | {'청산':^12} | {'보유':^4} | {'수익':^7}")
print("-"*80)

for _, row in trades_df.tail(20).iterrows():
    date_str = row['datetime'].strftime('%Y-%m-%d %H:%M')
    print(f"{date_str} | {row['direction']:^5} | {row['entry_price']:>8.0f} | {row['stop_loss']:>8.0f} | {row['sl_pct']:>4.1f}% | {row['exit_reason']:^12} | {row['hold_hours']:>3}h | {row['final_pnl']:>+6.2f}%")

# 누적 수익 계산
trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적 수익: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

# 저장
trades_df.to_csv('bb30_real_strategy_results.csv', index=False)
print("\n결과 저장: bb30_real_strategy_results.csv")
