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

# 수축 상태 - 더 엄격하게
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 스윙 포인트 찾기
def find_swings(data, window=2):
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        is_high = all(data.iloc[i]['high'] >= data.iloc[i-j]['high'] and 
                      data.iloc[i]['high'] >= data.iloc[i+j]['high'] 
                      for j in range(1, window+1))
        if is_high:
            highs.append({'idx': i, 'price': data.iloc[i]['high'], 'global_idx': data.index[i]})
        
        is_low = all(data.iloc[i]['low'] <= data.iloc[i-j]['low'] and 
                     data.iloc[i]['low'] <= data.iloc[i+j]['low'] 
                     for j in range(1, window+1))
        if is_low:
            lows.append({'idx': i, 'price': data.iloc[i]['low'], 'global_idx': data.index[i]})
    
    return highs, lows

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

# 매매 분석
trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 8:  # 최소 8봉으로 늘림
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end].copy()
    
    # 스윙 포인트 찾기
    highs, lows = find_swings(squeeze_data.reset_index(drop=True), window=2)
    
    if len(highs) < 2 or len(lows) < 2:  # 최소 2개씩 필요 (패턴 판단용)
        continue
    
    # 마지막 고점들(저항), 마지막 저점들(지지)
    last_high_1 = highs[-1]['price']  # 가장 최근 고점
    last_high_2 = highs[-2]['price'] if len(highs) >= 2 else None  # 이전 고점
    
    last_low_1 = lows[-1]['price']   # 가장 최근 저점
    last_low_2 = lows[-2]['price'] if len(lows) >= 2 else None   # 이전 저점
    
    # LH/HL 판단
    is_lh = last_high_1 < last_high_2  # Lower High
    is_hh = last_high_1 >= last_high_2  # Higher High
    is_hl = last_low_1 > last_low_2    # Higher Low  
    is_ll = last_low_1 <= last_low_2   # Lower Low
    
    # 수축 중 고점/저점 레벨 (저항/지지)
    swing_high = max([h['price'] for h in highs])  # 수축 중 최고점 = 저항
    swing_low = min([l['price'] for l in lows])    # 수축 중 최저점 = 지지
    
    # 저항/지지 레인지
    resistance = last_high_1  # 마지막 고점이 저항
    support = last_low_1      # 마지막 저점이 지지
    
    # === 전략 조건 ===
    # 1) LH 상태에서 저항 돌파 → LONG (손절: 지지선)
    # 2) HL 상태에서 지지 이탈 → SHORT (손절: 저항선)
    
    # 확장 시작 후 돌파 확인 (3봉 연속 확인)
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    signal_type = None
    
    for i in range(sq_end, min(sq_end + 15, len(df) - 3)):  # 15봉 내 확인
        # 3봉 연속 상승/하락 확인
        c1 = df.iloc[i]
        c2 = df.iloc[i+1] if i+1 < len(df) else None
        c3 = df.iloc[i+2] if i+2 < len(df) else None
        
        if c2 is None or c3 is None:
            break
            
        # 3봉 연속 상승 + 저항 돌파 → LONG
        if (c1['close'] > c1['open'] and 
            c2['close'] > c2['open'] and 
            c3['close'] > c3['open'] and
            c3['close'] > resistance):
            
            # LH 패턴일 때만 진입 (역추세 돌파)
            if is_lh:
                entry_idx = i + 2
                direction = 'LONG'
                entry_price = c3['close']
                stop_loss = support  # 지지선 아래가 손절
                signal_type = 'LH_BREAK'
                break
        
        # 3봉 연속 하락 + 지지 이탈 → SHORT
        if (c1['close'] < c1['open'] and 
            c2['close'] < c2['open'] and 
            c3['close'] < c3['open'] and
            c3['close'] < support):
            
            # HL 패턴일 때만 진입 (역추세 이탈)
            if is_hl:
                entry_idx = i + 2
                direction = 'SHORT'
                entry_price = c3['close']
                stop_loss = resistance  # 저항선 위가 손절
                signal_type = 'HL_BREAK'
                break
    
    if entry_idx is None:
        continue
    
    # 손절폭 계산
    if direction == 'LONG':
        sl_pct = (entry_price - stop_loss) / entry_price * 100
    else:
        sl_pct = (stop_loss - entry_price) / entry_price * 100
    
    # 손절폭 필터 (너무 크면 제외)
    if sl_pct > 5 or sl_pct < 0.3:
        continue
    
    # === 다음 수축까지 추적 ===
    next_squeeze_start = None
    for next_sq_start, next_sq_end in squeeze_ranges[sq_idx+1:]:
        if next_sq_start > entry_idx + 5:  # 최소 5봉 후
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
            # 손절 체크 - 지지선 아래로 종가 이탈
            if candle['close'] < stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            
            current_pnl = (candle['close'] - entry_price) / entry_price * 100
            max_profit = max(max_profit, (candle['high'] - entry_price) / entry_price * 100)
            max_loss = min(max_loss, (candle['low'] - entry_price) / entry_price * 100)
            
        else:  # SHORT
            # 손절 체크 - 저항선 위로 종가 이탈
            if candle['close'] > stop_loss:
                exit_price = candle['close']
                exit_reason = 'SL'
                exit_idx = i
                break
            
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
        if direction == 'LONG':
            final_pnl = (exit_price - entry_price) / entry_price * 100
        else:
            final_pnl = (entry_price - exit_price) / entry_price * 100
    
    # 보유 기간
    hold_hours = exit_idx - entry_idx
    
    trades.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'direction': direction,
        'signal_type': signal_type,
        'is_lh': is_lh,
        'is_hl': is_hl,
        'entry_price': entry_price,
        'resistance': resistance,
        'support': support,
        'stop_loss': stop_loss,
        'sl_pct': sl_pct,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_hours': hold_hours,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건")

if len(trades_df) == 0:
    print("매매가 없습니다!")
    exit()

print("\n" + "="*80)
print("📊 전략 결과 (LH 돌파 LONG / HL 이탈 SHORT)")
print("="*80)

wins = trades_df[trades_df['final_pnl'] > 0]
losses = trades_df[trades_df['final_pnl'] <= 0]

print(f"\n승률: {len(wins)}/{len(trades_df)} = {len(wins)/len(trades_df)*100:.1f}%")
print(f"평균 수익: {trades_df['final_pnl'].mean():.2f}%")
print(f"총 수익: {trades_df['final_pnl'].sum():.1f}%")
print(f"평균 보유 시간: {trades_df['hold_hours'].mean():.1f}시간")

if len(wins) > 0:
    print(f"\n승리 시 평균: +{wins['final_pnl'].mean():.2f}%")
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

# 신호 타입별
print("\n### 신호 타입별")
for sig in ['LH_BREAK', 'HL_BREAK']:
    sub = trades_df[trades_df['signal_type'] == sig]
    if len(sub) > 0:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {sig}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 손절폭별
print("\n### 손절폭별")
for low, high, label in [(0.3, 1, '0.3-1%'), (1, 2, '1-2%'), (2, 3, '2-3%'), (3, 5, '3-5%')]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 2:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 수축 길이별
print("\n### 수축 길이별")
for low, high, label in [(8, 15, '8-14h'), (15, 25, '15-24h'), (25, 50, '25-49h'), (50, 200, '50h+')]:
    sub = trades_df[(trades_df['squeeze_length'] >= low) & (trades_df['squeeze_length'] < high)]
    if len(sub) >= 2:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 최근 20건 상세
print("\n" + "="*80)
print("📋 최근 20건 상세")
print("="*80)
print(f"{'날짜':^16} | {'방향':^5} | {'신호':^10} | {'진입가':^8} | {'손절':^8} | {'SL%':^5} | {'청산':^12} | {'수익':^7}")
print("-"*90)

for _, row in trades_df.tail(20).iterrows():
    date_str = row['datetime'].strftime('%Y-%m-%d %H:%M')
    print(f"{date_str} | {row['direction']:^5} | {row['signal_type']:^10} | {row['entry_price']:>8.0f} | {row['stop_loss']:>8.0f} | {row['sl_pct']:>4.1f}% | {row['exit_reason']:^12} | {row['final_pnl']:>+6.2f}%")

# 누적 수익
trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적 수익: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

# 저장
trades_df.to_csv('bb30_real_strategy_v2_results.csv', index=False)
print("\n결과 저장: bb30_real_strategy_v2_results.csv")
