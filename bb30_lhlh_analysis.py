import pandas as pd
import numpy as np

# 1시간봉 데이터 로드
df = pd.read_csv('analysis_1h.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"1시간봉 데이터: {len(df)}개")

# BB 30 계산
BB_PERIOD = 30
BB_STD = 2

df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
df['bb_std'] = df['close'].rolling(BB_PERIOD).std()
df['bb_upper'] = df['bb_mid'] + BB_STD * df['bb_std']
df['bb_lower'] = df['bb_mid'] - BB_STD * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# 수축 상태
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# 스윙 고점/저점 찾기 (3봉 기준)
def find_swing_points(data, window=3):
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        # 스윙 고점: 양쪽보다 높음
        is_high = True
        for j in range(1, window + 1):
            if data.iloc[i]['high'] <= data.iloc[i-j]['high'] or data.iloc[i]['high'] <= data.iloc[i+j]['high']:
                is_high = False
                break
        if is_high:
            highs.append({'idx': i, 'price': data.iloc[i]['high'], 'datetime': data.iloc[i]['datetime']})
        
        # 스윙 저점: 양쪽보다 낮음
        is_low = True
        for j in range(1, window + 1):
            if data.iloc[i]['low'] >= data.iloc[i-j]['low'] or data.iloc[i]['low'] >= data.iloc[i+j]['low']:
                is_low = False
                break
        if is_low:
            lows.append({'idx': i, 'price': data.iloc[i]['low'], 'datetime': data.iloc[i]['datetime']})
    
    return highs, lows

# 수축 구간 찾기
squeeze_periods = []
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
        squeeze_periods.append({
            'start': squeeze_start,
            'end': i,
            'length': i - squeeze_start
        })

print(f"수축 구간: {len(squeeze_periods)}개")

# 각 수축 구간에서 LH/HL 패턴 분석
results = []

for sq in squeeze_periods:
    start = sq['start']
    end = sq['end']
    length = sq['length']
    
    if length < 5 or end >= len(df) - 30:  # 최소 5봉 이상
        continue
    
    squeeze_data = df.iloc[start:end].reset_index(drop=True)
    
    # 스윙 포인트 찾기
    highs, lows = find_swing_points(squeeze_data, window=2)
    
    if len(highs) < 2 or len(lows) < 2:
        continue
    
    # === LH/HL 패턴 분석 ===
    
    # 고점들 분석 (LH = Lower High, HH = Higher High)
    high_prices = [h['price'] for h in highs]
    high_pattern = []
    for i in range(1, len(high_prices)):
        if high_prices[i] < high_prices[i-1]:
            high_pattern.append('LH')  # Lower High
        else:
            high_pattern.append('HH')  # Higher High
    
    # 저점들 분석 (HL = Higher Low, LL = Lower Low)
    low_prices = [l['price'] for l in lows]
    low_pattern = []
    for i in range(1, len(low_prices)):
        if low_prices[i] > low_prices[i-1]:
            low_pattern.append('HL')  # Higher Low
        else:
            low_pattern.append('LL')  # Lower Low
    
    # 마지막 패턴
    last_high_pattern = high_pattern[-1] if high_pattern else None
    last_low_pattern = low_pattern[-1] if low_pattern else None
    
    # 전체 패턴 요약
    lh_count = high_pattern.count('LH')
    hh_count = high_pattern.count('HH')
    hl_count = low_pattern.count('HL')
    ll_count = low_pattern.count('LL')
    
    # 구조 판단
    # 상승 구조: HH + HL
    # 하락 구조: LH + LL
    # 수렴 구조: LH + HL (삼각수렴)
    # 확산 구조: HH + LL
    
    if lh_count > hh_count and hl_count > ll_count:
        structure = 'CONVERGE'  # 수렴 (LH + HL)
    elif hh_count > lh_count and hl_count > ll_count:
        structure = 'UPTREND'   # 상승 (HH + HL)
    elif lh_count > hh_count and ll_count > hl_count:
        structure = 'DOWNTREND' # 하락 (LH + LL)
    elif hh_count > lh_count and ll_count > hl_count:
        structure = 'EXPAND'    # 확산 (HH + LL)
    else:
        structure = 'NEUTRAL'
    
    # 마지막 스윙 (돌파 직전)
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    
    # 마지막 고점/저점 중 어느 것이 더 최근인지
    if last_high and last_low:
        if last_high['idx'] > last_low['idx']:
            last_swing = 'HIGH'
            last_swing_price = last_high['price']
        else:
            last_swing = 'LOW'
            last_swing_price = last_low['price']
    else:
        last_swing = None
        last_swing_price = None
    
    # === 확장 결과 ===
    # 돌파 방향
    break_candle = df.iloc[end]
    break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
    
    # 확장 피크 찾기 (30봉 내)
    max_up = df.iloc[end:end+30]['high'].max()
    max_down = df.iloc[end:end+30]['low'].min()
    entry_price = break_candle['close']
    
    up_move = (max_up - entry_price) / entry_price * 100
    down_move = (entry_price - max_down) / entry_price * 100
    
    # 실제 확장 방향 (더 큰 움직임)
    if up_move > down_move:
        expansion_dir = 1
        expansion_size = up_move
    else:
        expansion_dir = -1
        expansion_size = down_move
    
    # 수익 계산
    if break_dir == 1:
        pnl = up_move if expansion_dir == 1 else -down_move
    else:
        pnl = down_move if expansion_dir == -1 else -up_move
    
    results.append({
        'datetime': df.iloc[end]['datetime'],
        'squeeze_length': length,
        'num_highs': len(highs),
        'num_lows': len(lows),
        'lh_count': lh_count,
        'hh_count': hh_count,
        'hl_count': hl_count,
        'll_count': ll_count,
        'last_high_pattern': last_high_pattern,
        'last_low_pattern': last_low_pattern,
        'structure': structure,
        'last_swing': last_swing,
        'break_dir': break_dir,
        'expansion_dir': expansion_dir,
        'expansion_size': expansion_size,
        'pnl': pnl,
        'win': 1 if pnl > 0 else 0
    })

res_df = pd.DataFrame(results)
print(f"\n분석 가능 케이스: {len(res_df)}개")
print(f"기본 승률: {res_df['win'].mean()*100:.1f}%")
print(f"평균 수익: {res_df['pnl'].mean():+.2f}%")

print("\n" + "="*80)
print("📊 수축 중 가격 구조별 분석")
print("="*80)

# 구조별 분석
print("\n### 구조별 확장 결과")
for struct in ['CONVERGE', 'UPTREND', 'DOWNTREND', 'EXPAND', 'NEUTRAL']:
    sub = res_df[res_df['structure'] == struct]
    if len(sub) >= 10:
        up_exp = (sub['expansion_dir'] == 1).sum()
        down_exp = (sub['expansion_dir'] == -1).sum()
        print(f"\n  {struct}: {len(sub)}건")
        print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
        print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")
        print(f"    → 평균 확장크기: {sub['expansion_size'].mean():.2f}%")

print("\n" + "="*80)
print("📊 마지막 패턴별 분석")
print("="*80)

# 마지막 고점 패턴별
print("\n### 마지막 고점 패턴 (LH vs HH)")
for pattern in ['LH', 'HH']:
    sub = res_df[res_df['last_high_pattern'] == pattern]
    if len(sub) >= 20:
        up_exp = (sub['expansion_dir'] == 1).sum()
        down_exp = (sub['expansion_dir'] == -1).sum()
        print(f"  {pattern}: {len(sub)}건")
        print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
        print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")

# 마지막 저점 패턴별
print("\n### 마지막 저점 패턴 (HL vs LL)")
for pattern in ['HL', 'LL']:
    sub = res_df[res_df['last_low_pattern'] == pattern]
    if len(sub) >= 20:
        up_exp = (sub['expansion_dir'] == 1).sum()
        down_exp = (sub['expansion_dir'] == -1).sum()
        print(f"  {pattern}: {len(sub)}건")
        print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
        print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")

# 마지막 스윙별
print("\n### 마지막 스윙 (돌파 직전)")
for swing in ['HIGH', 'LOW']:
    sub = res_df[res_df['last_swing'] == swing]
    if len(sub) >= 20:
        up_exp = (sub['expansion_dir'] == 1).sum()
        down_exp = (sub['expansion_dir'] == -1).sum()
        print(f"  마지막 {swing}: {len(sub)}건")
        print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
        print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")

print("\n" + "="*80)
print("🎯 조합 패턴")
print("="*80)

# 마지막 고점 + 마지막 저점 조합
print("\n### 마지막 고점 + 마지막 저점 조합")
for hp in ['LH', 'HH']:
    for lp in ['HL', 'LL']:
        sub = res_df[(res_df['last_high_pattern'] == hp) & (res_df['last_low_pattern'] == lp)]
        if len(sub) >= 15:
            up_exp = (sub['expansion_dir'] == 1).sum()
            down_exp = (sub['expansion_dir'] == -1).sum()
            print(f"  {hp} + {lp}: {len(sub)}건")
            print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
            print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")

# 구조 + 마지막 스윙 조합
print("\n### 구조 + 마지막 스윙")
for struct in ['CONVERGE', 'UPTREND', 'DOWNTREND']:
    for swing in ['HIGH', 'LOW']:
        sub = res_df[(res_df['structure'] == struct) & (res_df['last_swing'] == swing)]
        if len(sub) >= 10:
            up_exp = (sub['expansion_dir'] == 1).sum()
            down_exp = (sub['expansion_dir'] == -1).sum()
            print(f"  {struct} + 마지막{swing}: {len(sub)}건")
            print(f"    → UP확장 {up_exp/len(sub)*100:.0f}% / DOWN확장 {down_exp/len(sub)*100:.0f}%")
            print(f"    → 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl'].mean():+.2f}%")

# 저장
res_df.to_csv('bb30_lhlh_results.csv', index=False)
print("\n\n결과 저장: bb30_lhlh_results.csv")
