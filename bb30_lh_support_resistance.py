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

# 수축 구간 찾기
squeeze_periods = []
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
        squeeze_periods.append({
            'start': squeeze_start,
            'end': i,
            'length': i - squeeze_start
        })

print(f"수축 구간: {len(squeeze_periods)}개")

# 각 수축 구간에서 LH/HL 저항/지지 분석
results = []

for sq in squeeze_periods:
    start = sq['start']
    end = sq['end']
    length = sq['length']
    
    if length < 8 or end >= len(df) - 30:  # 최소 8봉 (스윙 포인트 필요)
        continue
    
    squeeze_data = df.iloc[start:end].reset_index(drop=True)
    
    # 스윙 포인트 찾기
    highs, lows = find_swings(squeeze_data, window=2)
    
    if len(highs) < 2 or len(lows) < 2:
        continue
    
    # === LH/HL/HH/LL 분석 ===
    
    # 고점 패턴
    high_patterns = []
    for i in range(1, len(highs)):
        prev_price = highs[i-1]['price']
        curr_price = highs[i]['price']
        pattern = 'LH' if curr_price < prev_price else 'HH'
        high_patterns.append({
            'type': pattern,
            'price': curr_price,
            'prev_price': prev_price,
            'idx': highs[i]['idx']
        })
    
    # 저점 패턴
    low_patterns = []
    for i in range(1, len(lows)):
        prev_price = lows[i-1]['price']
        curr_price = lows[i]['price']
        pattern = 'HL' if curr_price > prev_price else 'LL'
        low_patterns.append({
            'type': pattern,
            'price': curr_price,
            'prev_price': prev_price,
            'idx': lows[i]['idx']
        })
    
    # === 돌파 후 저항/지지 테스트 ===
    break_idx_global = end
    break_candle = df.iloc[break_idx_global]
    break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
    
    # 3봉 확인
    confirmed = True
    for j in range(1, 4):
        if break_idx_global + j >= len(df):
            confirmed = False
            break
        candle = df.iloc[break_idx_global + j]
        if (1 if candle['close'] > candle['open'] else -1) != break_dir:
            confirmed = False
            break
    
    if not confirmed:
        continue
    
    entry_idx = break_idx_global + 3
    entry_price = df.iloc[entry_idx]['close']
    
    # 수축 중 주요 레벨들
    all_high_prices = [h['price'] for h in highs]
    all_low_prices = [l['price'] for l in lows]
    
    # 마지막 LH/HH 가격 (저항)
    last_high_price = highs[-1]['price'] if highs else None
    second_last_high = highs[-2]['price'] if len(highs) >= 2 else None
    
    # 마지막 HL/LL 가격 (지지)
    last_low_price = lows[-1]['price'] if lows else None
    second_last_low = lows[-2]['price'] if len(lows) >= 2 else None
    
    # === 돌파 후 30봉 동안 저항/지지 테스트 ===
    
    # LONG인 경우: 이전 고점(LH)이 저항으로 작용하는지
    # SHORT인 경우: 이전 저점(HL)이 지지로 작용하는지
    
    resistance_tested = False
    resistance_held = False
    resistance_broken = False
    
    support_tested = False
    support_held = False
    support_broken = False
    
    max_profit = 0
    max_loss = 0
    
    for k in range(1, 31):
        if entry_idx + k >= len(df):
            break
        
        candle = df.iloc[entry_idx + k]
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        # 수익 계산
        if break_dir == 1:
            profit = (close - entry_price) / entry_price * 100
        else:
            profit = (entry_price - close) / entry_price * 100
        
        max_profit = max(max_profit, profit)
        max_loss = min(max_loss, profit)
        
        # LONG: 이전 고점(저항) 테스트
        if break_dir == 1 and last_high_price:
            if high >= last_high_price * 0.998:  # 0.2% 허용
                resistance_tested = True
                if close < last_high_price:  # 종가가 저항 아래
                    resistance_held = True
                else:
                    resistance_broken = True
        
        # SHORT: 이전 저점(지지) 테스트
        if break_dir == -1 and last_low_price:
            if low <= last_low_price * 1.002:  # 0.2% 허용
                support_tested = True
                if close > last_low_price:  # 종가가 지지 위
                    support_held = True
                else:
                    support_broken = True
    
    # 마지막 패턴 타입
    last_high_pattern = high_patterns[-1]['type'] if high_patterns else None
    last_low_pattern = low_patterns[-1]['type'] if low_patterns else None
    
    results.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'squeeze_length': length,
        'direction': 'LONG' if break_dir == 1 else 'SHORT',
        'last_high_pattern': last_high_pattern,
        'last_low_pattern': last_low_pattern,
        'last_high_price': last_high_price,
        'last_low_price': last_low_price,
        'entry_price': entry_price,
        'resistance_tested': resistance_tested,
        'resistance_held': resistance_held,
        'resistance_broken': resistance_broken,
        'support_tested': support_tested,
        'support_held': support_held,
        'support_broken': support_broken,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'win': max_profit > abs(max_loss)
    })

res_df = pd.DataFrame(results)
print(f"\n분석 케이스: {len(res_df)}개")
print(f"기본 승률: {res_df['win'].mean()*100:.1f}%")

print("\n" + "="*80)
print("📊 LH/HL 저항/지지 분석")
print("="*80)

# LONG 케이스 - 저항 분석
print("\n### LONG 진입 시 - 이전 고점(LH/HH) 저항 분석")
long_df = res_df[res_df['direction'] == 'LONG']
print(f"  총 LONG: {len(long_df)}건")

# 저항 테스트 여부
tested = long_df[long_df['resistance_tested'] == True]
not_tested = long_df[long_df['resistance_tested'] == False]
print(f"\n  저항 테스트됨: {len(tested)}건")
print(f"  저항 테스트 안됨: {len(not_tested)}건")

if len(tested) >= 3:
    held = tested[tested['resistance_held'] == True]
    broken = tested[tested['resistance_broken'] == True]
    print(f"\n  → 저항에서 막힘: {len(held)}건, 승률 {held['win'].mean()*100:.0f}%")
    print(f"  → 저항 돌파: {len(broken)}건, 승률 {broken['win'].mean()*100:.0f}%")

# 마지막 고점 패턴별
print("\n  마지막 고점 패턴별:")
for pattern in ['LH', 'HH']:
    sub = long_df[long_df['last_high_pattern'] == pattern]
    if len(sub) >= 3:
        tested_sub = sub[sub['resistance_tested'] == True]
        broken_sub = sub[sub['resistance_broken'] == True]
        print(f"    {pattern}: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%")
        if len(tested_sub) >= 2:
            print(f"      → 저항 테스트: {len(tested_sub)}건")
        if len(broken_sub) >= 2:
            print(f"      → 저항 돌파: {len(broken_sub)}건, 승률 {broken_sub['win'].mean()*100:.0f}%")

# SHORT 케이스 - 지지 분석
print("\n### SHORT 진입 시 - 이전 저점(HL/LL) 지지 분석")
short_df = res_df[res_df['direction'] == 'SHORT']
print(f"  총 SHORT: {len(short_df)}건")

tested = short_df[short_df['support_tested'] == True]
not_tested = short_df[short_df['support_tested'] == False]
print(f"\n  지지 테스트됨: {len(tested)}건")
print(f"  지지 테스트 안됨: {len(not_tested)}건")

if len(tested) >= 3:
    held = tested[tested['support_held'] == True]
    broken = tested[tested['support_broken'] == True]
    print(f"\n  → 지지에서 반등: {len(held)}건, 승률 {held['win'].mean()*100:.0f}%")
    print(f"  → 지지 이탈: {len(broken)}건, 승률 {broken['win'].mean()*100:.0f}%")

# 마지막 저점 패턴별
print("\n  마지막 저점 패턴별:")
for pattern in ['HL', 'LL']:
    sub = short_df[short_df['last_low_pattern'] == pattern]
    if len(sub) >= 3:
        tested_sub = sub[sub['support_tested'] == True]
        broken_sub = sub[sub['support_broken'] == True]
        print(f"    {pattern}: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%")
        if len(tested_sub) >= 2:
            print(f"      → 지지 테스트: {len(tested_sub)}건")
        if len(broken_sub) >= 2:
            print(f"      → 지지 이탈: {len(broken_sub)}건, 승률 {broken_sub['win'].mean()*100:.0f}%")

print("\n" + "="*80)
print("🎯 핵심 전략 인사이트")
print("="*80)

# LH 돌파 시 승률
print("\n### LONG + LH 저항 돌파")
sub = res_df[(res_df['direction'] == 'LONG') & 
             (res_df['last_high_pattern'] == 'LH') & 
             (res_df['resistance_broken'] == True)]
if len(sub) >= 2:
    print(f"  LH 저항 돌파 LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

# HH 돌파 시 승률
sub = res_df[(res_df['direction'] == 'LONG') & 
             (res_df['last_high_pattern'] == 'HH') & 
             (res_df['resistance_broken'] == True)]
if len(sub) >= 2:
    print(f"  HH 저항 돌파 LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

# HL 이탈 시 승률
print("\n### SHORT + HL 지지 이탈")
sub = res_df[(res_df['direction'] == 'SHORT') & 
             (res_df['last_low_pattern'] == 'HL') & 
             (res_df['support_broken'] == True)]
if len(sub) >= 2:
    print(f"  HL 지지 이탈 SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

# LL 이탈 시 승률
sub = res_df[(res_df['direction'] == 'SHORT') & 
             (res_df['last_low_pattern'] == 'LL') & 
             (res_df['support_broken'] == True)]
if len(sub) >= 2:
    print(f"  LL 지지 이탈 SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

# 저항/지지 미테스트 케이스 (강한 추세)
print("\n### 저항/지지 테스트 없이 바로 진행 (강한 추세)")
sub = res_df[(res_df['direction'] == 'LONG') & (res_df['resistance_tested'] == False)]
if len(sub) >= 3:
    print(f"  LONG - 저항 미테스트: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

sub = res_df[(res_df['direction'] == 'SHORT') & (res_df['support_tested'] == False)]
if len(sub) >= 3:
    print(f"  SHORT - 지지 미테스트: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 수익 +{sub['max_profit'].mean():.1f}%")

# 저장
res_df.to_csv('bb30_support_resistance.csv', index=False)
print("\n\n결과 저장: bb30_support_resistance.csv")
