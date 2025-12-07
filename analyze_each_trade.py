import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# BB 30 계산
BB_PERIOD = 30
df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# 수축 상태
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 스윙 포인트 찾기
def find_swings(data, window=2):
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        # 스윙 고점
        is_high = all(data.iloc[i]['high'] > data.iloc[i-j]['high'] and 
                      data.iloc[i]['high'] > data.iloc[i+j]['high'] 
                      for j in range(1, window+1))
        if is_high:
            highs.append({'idx': i, 'price': data.iloc[i]['high']})
        
        # 스윙 저점
        is_low = all(data.iloc[i]['low'] < data.iloc[i-j]['low'] and 
                     data.iloc[i]['low'] < data.iloc[i+j]['low'] 
                     for j in range(1, window+1))
        if is_low:
            lows.append({'idx': i, 'price': data.iloc[i]['low']})
    
    return highs, lows

# 수축 → 확장 + 3봉 확인 매매 찾기
trades = []
in_squeeze = False
squeeze_start = 0

for i in range(60, len(df) - 35):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        break_idx = i
        
        break_candle = df.iloc[break_idx]
        break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
        
        # 3봉 연속 확인
        confirmed = True
        for j in range(1, 4):
            candle = df.iloc[break_idx + j]
            candle_dir = 1 if candle['close'] > candle['open'] else -1
            if candle_dir != break_dir:
                confirmed = False
                break
        
        if confirmed:
            entry_idx = break_idx + 3
            entry_price = df.iloc[entry_idx]['close']
            
            # 결과 계산
            max_profit = 0
            max_loss = 0
            for k in range(1, 31):
                if entry_idx + k >= len(df):
                    break
                price = df.iloc[entry_idx + k]['close']
                if break_dir == 1:
                    profit = (price - entry_price) / entry_price * 100
                else:
                    profit = (entry_price - price) / entry_price * 100
                max_profit = max(max_profit, profit)
                max_loss = min(max_loss, profit)
            
            trades.append({
                'squeeze_start': squeeze_start,
                'break_idx': break_idx,
                'entry_idx': entry_idx,
                'direction': break_dir,
                'entry_price': entry_price,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'win': max_profit > abs(max_loss)
            })

print(f"총 매매: {len(trades)}개\n")

# 각 매매의 패턴 분석
results = []

for trade_num, trade in enumerate(trades):
    sq_start = trade['squeeze_start']
    sq_end = trade['break_idx']
    
    squeeze_data = df.iloc[sq_start:sq_end].reset_index(drop=True)
    
    if len(squeeze_data) < 5:
        continue
    
    # 스윙 포인트 찾기
    highs, lows = find_swings(squeeze_data, window=2)
    
    # 패턴 분석
    high_patterns = []
    low_patterns = []
    
    if len(highs) >= 2:
        for i in range(1, len(highs)):
            if highs[i]['price'] < highs[i-1]['price']:
                high_patterns.append('LH')
            else:
                high_patterns.append('HH')
    
    if len(lows) >= 2:
        for i in range(1, len(lows)):
            if lows[i]['price'] > lows[i-1]['price']:
                low_patterns.append('HL')
            else:
                low_patterns.append('LL')
    
    # 마지막 패턴
    last_high = high_patterns[-1] if high_patterns else '-'
    last_low = low_patterns[-1] if low_patterns else '-'
    
    # 마지막 봉 BB 위치
    last_bb_pos = df.iloc[sq_end - 1]['bb_position']
    
    # 마지막 봉 방향
    last_candle = df.iloc[sq_end - 1]
    last_candle_dir = 'UP' if last_candle['close'] > last_candle['open'] else 'DOWN'
    
    # 수축 중 가격 변화
    price_change = (squeeze_data.iloc[-1]['close'] - squeeze_data.iloc[0]['close']) / squeeze_data.iloc[0]['close'] * 100
    
    results.append({
        'trade_num': trade_num + 1,
        'datetime': df.iloc[trade['entry_idx']]['datetime'],
        'direction': 'LONG' if trade['direction'] == 1 else 'SHORT',
        'squeeze_length': sq_end - sq_start,
        'num_highs': len(highs),
        'num_lows': len(lows),
        'high_pattern': '→'.join(high_patterns) if high_patterns else '-',
        'low_pattern': '→'.join(low_patterns) if low_patterns else '-',
        'last_high': last_high,
        'last_low': last_low,
        'last_candle': last_candle_dir,
        'last_bb_pos': last_bb_pos,
        'price_change': price_change,
        'max_profit': trade['max_profit'],
        'max_loss': trade['max_loss'],
        'result': 'WIN' if trade['win'] else 'LOSS'
    })

res_df = pd.DataFrame(results)

# 전체 출력
print("="*120)
print("📊 전체 매매 패턴 분석 (119개)")
print("="*120)

# 요약 통계
print("\n### 패턴별 승률")
print("\n마지막 고점 패턴:")
for p in ['LH', 'HH', '-']:
    sub = res_df[res_df['last_high'] == p]
    if len(sub) >= 5:
        win_rate = (sub['result'] == 'WIN').mean() * 100
        avg_profit = sub['max_profit'].mean()
        print(f"  {p}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 최대수익 +{avg_profit:.1f}%")

print("\n마지막 저점 패턴:")
for p in ['HL', 'LL', '-']:
    sub = res_df[res_df['last_low'] == p]
    if len(sub) >= 5:
        win_rate = (sub['result'] == 'WIN').mean() * 100
        avg_profit = sub['max_profit'].mean()
        print(f"  {p}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 최대수익 +{avg_profit:.1f}%")

print("\n마지막 봉 방향:")
for d in ['UP', 'DOWN']:
    sub = res_df[res_df['last_candle'] == d]
    win_rate = (sub['result'] == 'WIN').mean() * 100
    avg_profit = sub['max_profit'].mean()
    print(f"  {d}: {len(sub)}건, 승률 {win_rate:.0f}%, 평균 최대수익 +{avg_profit:.1f}%")

print("\n마지막 고점 + 저점 조합:")
for hp in ['LH', 'HH']:
    for lp in ['HL', 'LL']:
        sub = res_df[(res_df['last_high'] == hp) & (res_df['last_low'] == lp)]
        if len(sub) >= 5:
            win_rate = (sub['result'] == 'WIN').mean() * 100
            avg_profit = sub['max_profit'].mean()
            avg_loss = sub['max_loss'].mean()
            print(f"  {hp}+{lp}: {len(sub)}건, 승률 {win_rate:.0f}%, +{avg_profit:.1f}% / {avg_loss:.1f}%")

# 최근 20개 상세 출력
print("\n" + "="*120)
print("📋 최근 20개 매매 상세")
print("="*120)
print(f"{'#':>3} | {'날짜':^16} | {'방향':^5} | {'수축':^4} | {'고점패턴':^12} | {'저점패턴':^12} | {'마지막봉':^6} | {'BB위치':^6} | {'수익':^6} | {'손실':^6} | {'결과':^4}")
print("-"*120)

for _, row in res_df.tail(20).iterrows():
    date_str = row['datetime'].strftime('%Y-%m-%d %H:%M')
    print(f"{row['trade_num']:3} | {date_str:16} | {row['direction']:^5} | {row['squeeze_length']:^4} | {row['high_pattern']:^12} | {row['low_pattern']:^12} | {row['last_candle']:^6} | {row['last_bb_pos']:5.0f}% | +{row['max_profit']:4.1f}% | {row['max_loss']:5.1f}% | {row['result']:^4}")

# 승률 높은 패턴 조합
print("\n" + "="*120)
print("🎯 승률 높은 패턴 조합")
print("="*120)

# 방향 + 마지막 패턴 조합
print("\n### 진입방향 + 마지막 고점/저점")
for direction in ['LONG', 'SHORT']:
    for hp in ['LH', 'HH']:
        for lp in ['HL', 'LL']:
            sub = res_df[(res_df['direction'] == direction) & 
                         (res_df['last_high'] == hp) & 
                         (res_df['last_low'] == lp)]
            if len(sub) >= 3:
                win_rate = (sub['result'] == 'WIN').mean() * 100
                avg_profit = sub['max_profit'].mean()
                avg_loss = sub['max_loss'].mean()
                if win_rate >= 70 or len(sub) >= 10:
                    print(f"  {direction} + {hp}+{lp}: {len(sub)}건, 승률 {win_rate:.0f}%, +{avg_profit:.1f}%/{avg_loss:.1f}%")

# 저장
res_df.to_csv('trade_patterns.csv', index=False)
print("\n\n결과 저장: trade_patterns.csv")
