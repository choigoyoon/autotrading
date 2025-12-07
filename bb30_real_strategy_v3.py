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

# 매매 분석 - 개선된 버전
trades = []

for sq_idx, (sq_start, sq_end) in enumerate(squeeze_ranges):
    if sq_end >= len(df) - 30:
        continue
    
    squeeze_length = sq_end - sq_start
    if squeeze_length < 10:  # 최소 10봉으로 늘림 (더 명확한 수축)
        continue
    
    squeeze_data = df.iloc[sq_start:sq_end].copy()
    
    # 스윙 포인트 찾기
    highs, lows = find_swings(squeeze_data.reset_index(drop=True), window=2)
    
    if len(highs) < 2 or len(lows) < 2:
        continue
    
    # 마지막 고점들, 저점들
    last_high_1 = highs[-1]['price']
    last_high_2 = highs[-2]['price']
    
    last_low_1 = lows[-1]['price']
    last_low_2 = lows[-2]['price']
    
    # LH/HL 판단
    is_lh = last_high_1 < last_high_2
    is_hl = last_low_1 > last_low_2
    
    # 수축 범위
    squeeze_high = max([h['price'] for h in highs])
    squeeze_low = min([l['price'] for l in lows])
    squeeze_range = (squeeze_high - squeeze_low) / squeeze_low * 100
    
    # 저항/지지
    resistance = last_high_1
    support = last_low_1
    
    # === 돌파 확인 (3봉 연속) ===
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    signal_type = None
    
    for i in range(sq_end, min(sq_end + 15, len(df) - 3)):
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
            
            if is_lh:
                entry_idx = i + 2
                direction = 'LONG'
                entry_price = c3['close']
                # 손절 개선: 지지선에서 추가 여유를 줌
                sl_buffer = (resistance - support) * 0.3  # 30% 추가 여유
                stop_loss = support - sl_buffer
                signal_type = 'LH_BREAK'
                break
        
        # 3봉 연속 하락 + 지지 이탈 → SHORT
        if (c1['close'] < c1['open'] and 
            c2['close'] < c2['open'] and 
            c3['close'] < c3['open'] and
            c3['close'] < support):
            
            if is_hl:
                entry_idx = i + 2
                direction = 'SHORT'
                entry_price = c3['close']
                # 손절 개선
                sl_buffer = (resistance - support) * 0.3
                stop_loss = resistance + sl_buffer
                signal_type = 'HL_BREAK'
                break
    
    if entry_idx is None:
        continue
    
    # 손절폭 계산
    if direction == 'LONG':
        sl_pct = (entry_price - stop_loss) / entry_price * 100
    else:
        sl_pct = (stop_loss - entry_price) / entry_price * 100
    
    # 손절폭 필터
    if sl_pct > 6 or sl_pct < 0.5:
        continue
    
    # 다음 수축까지 추적
    next_squeeze_start = None
    for next_sq_start, next_sq_end in squeeze_ranges[sq_idx+1:]:
        if next_sq_start > entry_idx + 5:
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
        'squeeze_length': squeeze_length,
        'squeeze_range': squeeze_range
    })

trades_df = pd.DataFrame(trades)
print(f"\n총 매매: {len(trades_df)}건")

if len(trades_df) == 0:
    print("매매가 없습니다!")
    exit()

print("\n" + "="*80)
print("📊 전략 결과 V3 (손절 여유 30% 추가)")
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

# 손절폭별
print("\n### 손절폭별")
for low, high, label in [(0.5, 1.5, '0.5-1.5%'), (1.5, 3, '1.5-3%'), (3, 6, '3-6%')]:
    sub = trades_df[(trades_df['sl_pct'] >= low) & (trades_df['sl_pct'] < high)]
    if len(sub) >= 2:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 수축 범위별
print("\n### 수축 범위별")
for low, high, label in [(0, 2, '0-2%'), (2, 4, '2-4%'), (4, 10, '4-10%')]:
    sub = trades_df[(trades_df['squeeze_range'] >= low) & (trades_df['squeeze_range'] < high)]
    if len(sub) >= 2:
        win_rate = (sub['final_pnl'] > 0).mean() * 100
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

print("\n" + "="*80)
print("📋 최근 20건 상세")
print("="*80)

for _, row in trades_df.tail(20).iterrows():
    date_str = row['datetime'].strftime('%Y-%m-%d %H:%M')
    print(f"{date_str} | {row['direction']:^5} | SL:{row['sl_pct']:>4.1f}% | {row['exit_reason']:^12} | {row['final_pnl']:>+6.2f}%")

# 누적 수익
trades_df['cumulative_pnl'] = trades_df['final_pnl'].cumsum()
print(f"\n최종 누적 수익: {trades_df['cumulative_pnl'].iloc[-1]:.1f}%")

# 저장
trades_df.to_csv('bb30_real_strategy_v3_results.csv', index=False)
print("\n결과 저장: bb30_real_strategy_v3_results.csv")

# === 추가 분석: 최적 조건 찾기 ===
print("\n" + "="*80)
print("🔍 최적 조건 탐색")
print("="*80)

# 수축 길이 + 방향 조합
print("\n### 수축 길이 + 방향 조합")
for d in ['LONG', 'SHORT']:
    for sl_low, sl_high in [(8, 20), (20, 40), (40, 100)]:
        sub = trades_df[(trades_df['direction'] == d) & 
                        (trades_df['squeeze_length'] >= sl_low) & 
                        (trades_df['squeeze_length'] < sl_high)]
        if len(sub) >= 3:
            win_rate = (sub['final_pnl'] > 0).mean() * 100
            print(f"  {d} + 수축{sl_low}-{sl_high}h: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {sub['final_pnl'].mean():.2f}%")

# 수익성 높은 조합 찾기
print("\n### 수익성 높은 조합 (승률 >= 55% 또는 평균수익 >= 0.5%)")
for d in ['LONG', 'SHORT']:
    for sl_range in [(0.5, 2), (2, 4), (4, 6)]:
        for sq_range in [(0, 3), (3, 6), (6, 15)]:
            sub = trades_df[(trades_df['direction'] == d) & 
                            (trades_df['sl_pct'] >= sl_range[0]) & 
                            (trades_df['sl_pct'] < sl_range[1]) &
                            (trades_df['squeeze_range'] >= sq_range[0]) & 
                            (trades_df['squeeze_range'] < sq_range[1])]
            if len(sub) >= 5:
                win_rate = (sub['final_pnl'] > 0).mean() * 100
                avg_pnl = sub['final_pnl'].mean()
                if win_rate >= 55 or avg_pnl >= 0.5:
                    print(f"  {d} + SL{sl_range[0]}-{sl_range[1]}% + 범위{sq_range[0]}-{sq_range[1]}%: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 {avg_pnl:.2f}%")
