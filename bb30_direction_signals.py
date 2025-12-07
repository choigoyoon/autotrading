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
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# 수축 상태
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1

# === 시그널 계산 ===

# 1. 캔들 방향
df['candle_dir'] = np.where(df['close'] > df['open'], 1, -1)
df['candle_body'] = abs(df['close'] - df['open'])
df['candle_range'] = df['high'] - df['low']
df['body_ratio'] = df['candle_body'] / df['candle_range']  # 몸통 비율

# 2. 3연속 양봉/음봉
df['three_up'] = (df['candle_dir'] == 1) & (df['candle_dir'].shift(1) == 1) & (df['candle_dir'].shift(2) == 1)
df['three_down'] = (df['candle_dir'] == -1) & (df['candle_dir'].shift(1) == -1) & (df['candle_dir'].shift(2) == -1)

# 3. FVG (Fair Value Gap)
# Bullish FVG: 현재 low > 2봉전 high (갭 상승)
# Bearish FVG: 현재 high < 2봉전 low (갭 하락)
df['fvg_bull'] = df['low'] > df['high'].shift(2)
df['fvg_bear'] = df['high'] < df['low'].shift(2)

# 4. 강한 캔들 (평균 대비 1.5배 이상 몸통)
avg_body = df['candle_body'].rolling(20).mean()
df['strong_candle'] = df['candle_body'] > avg_body * 1.5

# 5. 밴드 터치
df['touch_upper'] = df['high'] >= df['bb_upper']
df['touch_lower'] = df['low'] <= df['bb_lower']

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

# 각 수축 구간에서 시그널 분석
results = []

for sq in squeeze_periods:
    start = sq['start']
    end = sq['end']
    length = sq['length']
    
    if length < 5 or end >= len(df) - 30:
        continue
    
    squeeze_data = df.iloc[start:end]
    
    # === 수축 중 시그널 찾기 ===
    
    # 3연속 양봉/음봉
    three_up_idx = squeeze_data[squeeze_data['three_up']].index.tolist()
    three_down_idx = squeeze_data[squeeze_data['three_down']].index.tolist()
    
    # 마지막 3연속 시그널
    last_three_signal = None
    last_three_pos = None  # 시그널 위치 (수축 내 %)
    
    if three_up_idx or three_down_idx:
        last_up = three_up_idx[-1] if three_up_idx else -1
        last_down = three_down_idx[-1] if three_down_idx else -1
        
        if last_up > last_down:
            last_three_signal = 'UP3'
            last_three_pos = (last_up - start) / length * 100
        else:
            last_three_signal = 'DOWN3'
            last_three_pos = (last_down - start) / length * 100
    
    # FVG
    fvg_bull_idx = squeeze_data[squeeze_data['fvg_bull']].index.tolist()
    fvg_bear_idx = squeeze_data[squeeze_data['fvg_bear']].index.tolist()
    
    last_fvg = None
    last_fvg_pos = None
    
    if fvg_bull_idx or fvg_bear_idx:
        last_bull = fvg_bull_idx[-1] if fvg_bull_idx else -1
        last_bear = fvg_bear_idx[-1] if fvg_bear_idx else -1
        
        if last_bull > last_bear:
            last_fvg = 'FVG_BULL'
            last_fvg_pos = (last_bull - start) / length * 100
        else:
            last_fvg = 'FVG_BEAR'
            last_fvg_pos = (last_bear - start) / length * 100
    
    # 강한 캔들
    strong_up_idx = squeeze_data[(squeeze_data['strong_candle']) & (squeeze_data['candle_dir'] == 1)].index.tolist()
    strong_down_idx = squeeze_data[(squeeze_data['strong_candle']) & (squeeze_data['candle_dir'] == -1)].index.tolist()
    
    last_strong = None
    last_strong_pos = None
    
    if strong_up_idx or strong_down_idx:
        last_up = strong_up_idx[-1] if strong_up_idx else -1
        last_down = strong_down_idx[-1] if strong_down_idx else -1
        
        if last_up > last_down:
            last_strong = 'STRONG_UP'
            last_strong_pos = (last_up - start) / length * 100
        else:
            last_strong = 'STRONG_DOWN'
            last_strong_pos = (last_down - start) / length * 100
    
    # 밴드 터치
    touch_upper_idx = squeeze_data[squeeze_data['touch_upper']].index.tolist()
    touch_lower_idx = squeeze_data[squeeze_data['touch_lower']].index.tolist()
    
    last_touch = None
    last_touch_pos = None
    
    if touch_upper_idx or touch_lower_idx:
        last_up = touch_upper_idx[-1] if touch_upper_idx else -1
        last_low = touch_lower_idx[-1] if touch_lower_idx else -1
        
        if last_up > last_low:
            last_touch = 'TOUCH_UPPER'
            last_touch_pos = (last_up - start) / length * 100
        else:
            last_touch = 'TOUCH_LOWER'
            last_touch_pos = (last_low - start) / length * 100
    
    # === 돌파 결과 ===
    break_candle = df.iloc[end]
    break_dir = 1 if break_candle['close'] > break_candle['open'] else -1
    
    # 3봉 확인
    confirmed = True
    for j in range(1, 4):
        if end + j >= len(df):
            confirmed = False
            break
        candle = df.iloc[end + j]
        if (1 if candle['close'] > candle['open'] else -1) != break_dir:
            confirmed = False
            break
    
    if not confirmed:
        continue
    
    # 결과 계산
    entry_idx = end + 3
    entry_price = df.iloc[entry_idx]['close']
    
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
    
    results.append({
        'datetime': df.iloc[entry_idx]['datetime'],
        'squeeze_length': length,
        'direction': 'LONG' if break_dir == 1 else 'SHORT',
        'last_three': last_three_signal,
        'last_three_pos': last_three_pos,
        'last_fvg': last_fvg,
        'last_fvg_pos': last_fvg_pos,
        'last_strong': last_strong,
        'last_strong_pos': last_strong_pos,
        'last_touch': last_touch,
        'last_touch_pos': last_touch_pos,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'win': max_profit > abs(max_loss)
    })

res_df = pd.DataFrame(results)
print(f"\n분석 케이스: {len(res_df)}개")
print(f"기본 승률: {res_df['win'].mean()*100:.1f}%")

print("\n" + "="*80)
print("📊 방향 결정 시그널 분석")
print("="*80)

# 1. 3연속 캔들
print("\n### 1. 마지막 3연속 캔들")
for sig in ['UP3', 'DOWN3', None]:
    sub = res_df[res_df['last_three'] == sig]
    if len(sub) >= 5:
        label = sig if sig else '없음'
        win_rate = sub['win'].mean() * 100
        
        # 방향 일치 (UP3 → LONG, DOWN3 → SHORT)
        if sig == 'UP3':
            match = (sub['direction'] == 'LONG').mean() * 100
        elif sig == 'DOWN3':
            match = (sub['direction'] == 'SHORT').mean() * 100
        else:
            match = 0
        
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 방향일치 {match:.0f}%")

# 2. FVG
print("\n### 2. 마지막 FVG")
for sig in ['FVG_BULL', 'FVG_BEAR', None]:
    sub = res_df[res_df['last_fvg'] == sig]
    if len(sub) >= 5:
        label = sig if sig else '없음'
        win_rate = sub['win'].mean() * 100
        
        if sig == 'FVG_BULL':
            match = (sub['direction'] == 'LONG').mean() * 100
        elif sig == 'FVG_BEAR':
            match = (sub['direction'] == 'SHORT').mean() * 100
        else:
            match = 0
        
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 방향일치 {match:.0f}%")

# 3. 강한 캔들
print("\n### 3. 마지막 강한 캔들")
for sig in ['STRONG_UP', 'STRONG_DOWN', None]:
    sub = res_df[res_df['last_strong'] == sig]
    if len(sub) >= 5:
        label = sig if sig else '없음'
        win_rate = sub['win'].mean() * 100
        
        if sig == 'STRONG_UP':
            match = (sub['direction'] == 'LONG').mean() * 100
        elif sig == 'STRONG_DOWN':
            match = (sub['direction'] == 'SHORT').mean() * 100
        else:
            match = 0
        
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 방향일치 {match:.0f}%")

# 4. 밴드 터치
print("\n### 4. 마지막 밴드 터치")
for sig in ['TOUCH_UPPER', 'TOUCH_LOWER', None]:
    sub = res_df[res_df['last_touch'] == sig]
    if len(sub) >= 5:
        label = sig if sig else '없음'
        win_rate = sub['win'].mean() * 100
        
        if sig == 'TOUCH_UPPER':
            match = (sub['direction'] == 'LONG').mean() * 100
        elif sig == 'TOUCH_LOWER':
            match = (sub['direction'] == 'SHORT').mean() * 100
        else:
            match = 0
        
        print(f"  {label}: {len(sub)}건, 승률 {win_rate:.0f}%, 방향일치 {match:.0f}%")

print("\n" + "="*80)
print("🎯 시그널 위치 분석 (수축 구간 내 어디서 발생?)")
print("="*80)

# 시그널 위치별 분석
print("\n### 3연속 캔들 발생 위치")
for sig in ['UP3', 'DOWN3']:
    sub = res_df[(res_df['last_three'] == sig) & (res_df['last_three_pos'].notna())]
    if len(sub) >= 3:
        avg_pos = sub['last_three_pos'].mean()
        print(f"  {sig}: 평균 {avg_pos:.0f}% 지점에서 발생")
        
        # 위치별 승률
        early = sub[sub['last_three_pos'] < 50]
        late = sub[sub['last_three_pos'] >= 50]
        if len(early) >= 2:
            print(f"    → 전반부(0-50%): {len(early)}건, 승률 {early['win'].mean()*100:.0f}%")
        if len(late) >= 2:
            print(f"    → 후반부(50-100%): {len(late)}건, 승률 {late['win'].mean()*100:.0f}%")

print("\n### 강한 캔들 발생 위치")
for sig in ['STRONG_UP', 'STRONG_DOWN']:
    sub = res_df[(res_df['last_strong'] == sig) & (res_df['last_strong_pos'].notna())]
    if len(sub) >= 3:
        avg_pos = sub['last_strong_pos'].mean()
        print(f"  {sig}: 평균 {avg_pos:.0f}% 지점에서 발생")
        
        late = sub[sub['last_strong_pos'] >= 70]
        if len(late) >= 2:
            print(f"    → 후반부(70%+): {len(late)}건, 승률 {late['win'].mean()*100:.0f}%")

print("\n" + "="*80)
print("🔥 조합 분석")
print("="*80)

# 시그널 + 방향 일치
print("\n### 시그널 방향 일치 시 승률")

# UP3 + LONG
sub = res_df[(res_df['last_three'] == 'UP3') & (res_df['direction'] == 'LONG')]
if len(sub) >= 3:
    print(f"  UP3 → LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

# DOWN3 + SHORT
sub = res_df[(res_df['last_three'] == 'DOWN3') & (res_df['direction'] == 'SHORT')]
if len(sub) >= 3:
    print(f"  DOWN3 → SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

# STRONG_UP + LONG
sub = res_df[(res_df['last_strong'] == 'STRONG_UP') & (res_df['direction'] == 'LONG')]
if len(sub) >= 3:
    print(f"  STRONG_UP → LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

# STRONG_DOWN + SHORT
sub = res_df[(res_df['last_strong'] == 'STRONG_DOWN') & (res_df['direction'] == 'SHORT')]
if len(sub) >= 3:
    print(f"  STRONG_DOWN → SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

# FVG
sub = res_df[(res_df['last_fvg'] == 'FVG_BULL') & (res_df['direction'] == 'LONG')]
if len(sub) >= 3:
    print(f"  FVG_BULL → LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

sub = res_df[(res_df['last_fvg'] == 'FVG_BEAR') & (res_df['direction'] == 'SHORT')]
if len(sub) >= 3:
    print(f"  FVG_BEAR → SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%, 평균수익 +{sub['max_profit'].mean():.1f}%")

# 후반부 강한 캔들 + 방향 일치
print("\n### 후반부(70%+) 강한 캔들 + 방향 일치")
sub = res_df[(res_df['last_strong'] == 'STRONG_UP') & 
             (res_df['last_strong_pos'] >= 70) & 
             (res_df['direction'] == 'LONG')]
if len(sub) >= 2:
    print(f"  후반부 STRONG_UP → LONG: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%")

sub = res_df[(res_df['last_strong'] == 'STRONG_DOWN') & 
             (res_df['last_strong_pos'] >= 70) & 
             (res_df['direction'] == 'SHORT')]
if len(sub) >= 2:
    print(f"  후반부 STRONG_DOWN → SHORT: {len(sub)}건, 승률 {sub['win'].mean()*100:.0f}%")

# 저장
res_df.to_csv('bb30_direction_signals.csv', index=False)
print("\n\n결과 저장: bb30_direction_signals.csv")
