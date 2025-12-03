import pandas as pd
import numpy as np

df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# EMA 계산
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema200'] = df['close'].ewm(span=200).mean()

# VWAP (일별 리셋 - 간단히 20봉 기준)
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
    
    # 수축 범위
    squeeze_high = squeeze_data['high'].max()
    squeeze_low = squeeze_data['low'].min()
    
    # 수축 종료 시점 지표값
    last_idx = sq_end - 1
    last_close = df.iloc[last_idx]['close']
    last_ema20 = df.iloc[last_idx]['ema20']
    last_ema50 = df.iloc[last_idx]['ema50']
    last_ema200 = df.iloc[last_idx]['ema200']
    last_vwap = df.iloc[last_idx]['vwap']
    last_bb_mid = df.iloc[last_idx]['bb_mid']
    
    # 지표 기준 위치
    above_ema20 = last_close > last_ema20
    above_ema50 = last_close > last_ema50
    above_ema200 = last_close > last_ema200
    above_vwap = last_close > last_vwap
    above_bb_mid = last_close > last_bb_mid
    
    # EMA 정배열/역배열
    ema_bullish = last_ema20 > last_ema50 > last_ema200  # 정배열
    ema_bearish = last_ema20 < last_ema50 < last_ema200  # 역배열
    
    # 마지막 캔들 방향
    last_candle = df.iloc[last_idx]
    last_direction = 'UP' if last_candle['close'] > last_candle['open'] else 'DOWN'
    
    # === 돌파 확인 ===
    entry_idx = None
    direction = None
    entry_price = None
    stop_loss = None
    
    for i in range(sq_end, min(sq_end + 10, len(df))):
        candle = df.iloc[i]
        
        if candle['close'] > squeeze_high:
            entry_idx = i
            direction = 'LONG'
            entry_price = candle['close']
            stop_loss = squeeze_low
            break
        
        if candle['close'] < squeeze_low:
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
        'last_direction': last_direction,
        'above_ema20': above_ema20,
        'above_ema50': above_ema50,
        'above_ema200': above_ema200,
        'above_vwap': above_vwap,
        'above_bb_mid': above_bb_mid,
        'ema_bullish': ema_bullish,
        'ema_bearish': ema_bearish,
        'sl_pct': sl_pct,
        'exit_reason': exit_reason,
        'max_profit': max_profit,
        'final_pnl': final_pnl,
        'squeeze_length': squeeze_length
    })

trades_df = pd.DataFrame(trades)
print(f"총 매매: {len(trades_df)}건")

print("\n" + "="*80)
print("📊 지표별 승/패 분석")
print("="*80)

# 1. EMA20 기준
print("\n### 1. EMA20 기준")
for above in [True, False]:
    sub = trades_df[trades_df['above_ema20'] == above]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        label = "EMA20 위" if above else "EMA20 아래"
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 2. EMA50 기준
print("\n### 2. EMA50 기준")
for above in [True, False]:
    sub = trades_df[trades_df['above_ema50'] == above]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        label = "EMA50 위" if above else "EMA50 아래"
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 3. EMA200 기준
print("\n### 3. EMA200 기준")
for above in [True, False]:
    sub = trades_df[trades_df['above_ema200'] == above]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        label = "EMA200 위" if above else "EMA200 아래"
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 4. VWAP 기준
print("\n### 4. VWAP 기준")
for above in [True, False]:
    sub = trades_df[trades_df['above_vwap'] == above]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        label = "VWAP 위" if above else "VWAP 아래"
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 5. BB MID 기준
print("\n### 5. BB MID 기준")
for above in [True, False]:
    sub = trades_df[trades_df['above_bb_mid'] == above]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        label = "BB중심 위" if above else "BB중심 아래"
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 6. EMA 정배열/역배열
print("\n### 6. EMA 배열 (20>50>200)")
for bull, bear, label in [(True, False, '정배열'), (False, True, '역배열'), (False, False, '혼조')]:
    if bull:
        sub = trades_df[trades_df['ema_bullish'] == True]
    elif bear:
        sub = trades_df[trades_df['ema_bearish'] == True]
    else:
        sub = trades_df[(trades_df['ema_bullish'] == False) & (trades_df['ema_bearish'] == False)]
    if len(sub) >= 20:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print("\n" + "="*80)
print("📊 방향 + 지표 조합")
print("="*80)

# 방향 + EMA200
print("\n### 방향 + EMA200")
for d in ['LONG', 'SHORT']:
    for above in [True, False]:
        sub = trades_df[(trades_df['direction'] == d) & (trades_df['above_ema200'] == above)]
        if len(sub) >= 20:
            wr = (sub['final_pnl'] > 0).mean() * 100
            avg = sub['final_pnl'].mean()
            label = f"{d} + EMA200{'위' if above else '아래'}"
            print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 방향 + VWAP
print("\n### 방향 + VWAP")
for d in ['LONG', 'SHORT']:
    for above in [True, False]:
        sub = trades_df[(trades_df['direction'] == d) & (trades_df['above_vwap'] == above)]
        if len(sub) >= 20:
            wr = (sub['final_pnl'] > 0).mean() * 100
            avg = sub['final_pnl'].mean()
            label = f"{d} + VWAP{'위' if above else '아래'}"
            print(f"  {label}: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

# 방향 + EMA 배열
print("\n### 방향 + EMA 배열")
for d in ['LONG', 'SHORT']:
    # 정배열
    sub = trades_df[(trades_df['direction'] == d) & (trades_df['ema_bullish'] == True)]
    if len(sub) >= 15:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {d} + 정배열: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")
    # 역배열
    sub = trades_df[(trades_df['direction'] == d) & (trades_df['ema_bearish'] == True)]
    if len(sub) >= 15:
        wr = (sub['final_pnl'] > 0).mean() * 100
        avg = sub['final_pnl'].mean()
        print(f"  {d} + 역배열: {len(sub)}건, 승률 {wr:.1f}%, 평균 {avg:.2f}%")

print("\n" + "="*80)
print("🔍 최적 조합 탐색")
print("="*80)

results = []
for d in ['LONG', 'SHORT']:
    for ema200 in [True, False]:
        for vwap in [True, False]:
            sub = trades_df[(trades_df['direction'] == d) & 
                           (trades_df['above_ema200'] == ema200) &
                           (trades_df['above_vwap'] == vwap)]
            if len(sub) >= 20:
                wr = (sub['final_pnl'] > 0).mean() * 100
                avg = sub['final_pnl'].mean()
                results.append({
                    'combo': f"{d}+EMA200{'↑' if ema200 else '↓'}+VWAP{'↑' if vwap else '↓'}",
                    'count': len(sub),
                    'win_rate': wr,
                    'avg_pnl': avg
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('win_rate', ascending=False)
print("\n### 승률순")
for _, row in results_df.iterrows():
    print(f"  {row['combo']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}%")

trades_df.to_csv('trades_with_indicators.csv', index=False)
