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

# BB 내 위치 (0~100%)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# 캔들 방향
df['candle_dir'] = np.where(df['close'] > df['open'], 1, -1)

# 수축/확장 상태 판단
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['bb_width_max30'] = df['bb_width'].rolling(30).max()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.1  # 수축 상태

# 수축 → 확장 → 수축 전체 사이클 찾기
cycles = []
in_squeeze = False
squeeze_start = 0
expansion_start = 0
expansion_max_width = 0
expansion_max_idx = 0

for i in range(60, len(df)):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    
    # 수축 시작
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
        
    # 확장 시작 (수축 끝)
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        expansion_start = i
        expansion_max_width = df.iloc[i]['bb_width']
        expansion_max_idx = i
        
        # 다음 수축까지 확장 추적
        for j in range(i+1, min(i+100, len(df))):  # 최대 100봉 추적
            if df.iloc[j]['bb_width'] > expansion_max_width:
                expansion_max_width = df.iloc[j]['bb_width']
                expansion_max_idx = j
            
            # 다시 수축 시작
            if df.iloc[j]['is_squeeze']:
                cycles.append({
                    'squeeze_start': squeeze_start,
                    'squeeze_end': expansion_start,  # 확장 시작 = 수축 끝
                    'expansion_peak': expansion_max_idx,
                    'cycle_end': j,
                    'squeeze_length': expansion_start - squeeze_start,
                    'expansion_length': j - expansion_start,
                    'squeeze_min_width': df.iloc[squeeze_start:expansion_start]['bb_width'].min(),
                    'expansion_max_width': expansion_max_width,
                })
                break

print(f"\n전체 사이클 수: {len(cycles)}개")

# 각 사이클 분석
results = []
for c in cycles:
    sq_start = c['squeeze_start']
    sq_end = c['squeeze_end']
    exp_peak = c['expansion_peak']
    cycle_end = c['cycle_end']
    
    if exp_peak >= len(df) - 1:
        continue
    
    # === 수축 구간 특성 ===
    squeeze_data = df.iloc[sq_start:sq_end]
    if len(squeeze_data) < 2:
        continue
    
    # 수축 중 마지막 봉
    last_candle_dir = df.iloc[sq_end - 1]['candle_dir']
    last_bb_pos = df.iloc[sq_end - 1]['bb_position']
    
    # 수축 중 가격 이동
    squeeze_price_start = df.iloc[sq_start]['close']
    squeeze_price_end = df.iloc[sq_end - 1]['close']
    squeeze_price_change = (squeeze_price_end - squeeze_price_start) / squeeze_price_start * 100
    
    # 수축 중 BB 위치 이동
    squeeze_pos_start = df.iloc[sq_start]['bb_position']
    squeeze_pos_end = df.iloc[sq_end - 1]['bb_position']
    squeeze_pos_drift = squeeze_pos_end - squeeze_pos_start
    
    # === 확장 구간 특성 ===
    expansion_data = df.iloc[sq_end:cycle_end]
    
    # 확장 첫 봉 방향 (돌파 방향)
    break_dir = df.iloc[sq_end]['candle_dir']
    
    # 확장 중 가격 변화 (수축 끝 → 확장 피크)
    expansion_price_start = df.iloc[sq_end]['close']
    expansion_price_peak = df.iloc[exp_peak]['close']
    expansion_price_change = (expansion_price_peak - expansion_price_start) / expansion_price_start * 100
    
    # 확장 방향 (가격 기준)
    expansion_dir = 1 if expansion_price_change > 0 else -1
    
    # 밴드폭 확장률
    width_expansion_ratio = c['expansion_max_width'] / c['squeeze_min_width']
    
    # === 수익 계산 ===
    # 돌파 방향으로 진입했을 때 확장 피크까지의 수익
    entry_price = df.iloc[sq_end]['close']
    peak_price = df.iloc[exp_peak]['close']
    
    if break_dir == 1:  # UP 돌파 → LONG
        pnl_to_peak = (peak_price - entry_price) / entry_price * 100
    else:  # DOWN 돌파 → SHORT
        pnl_to_peak = (entry_price - peak_price) / entry_price * 100
    
    # 방향 일치 여부 (돌파 방향 = 확장 방향)
    dir_match = 1 if break_dir == expansion_dir else 0
    
    results.append({
        'datetime': df.iloc[sq_end]['datetime'],
        'squeeze_length': c['squeeze_length'],
        'expansion_length': c['expansion_length'],
        'squeeze_min_width': c['squeeze_min_width'],
        'expansion_max_width': c['expansion_max_width'],
        'width_expansion_ratio': width_expansion_ratio,
        'last_candle_dir': last_candle_dir,
        'last_bb_pos': last_bb_pos,
        'squeeze_price_change': squeeze_price_change,
        'squeeze_pos_drift': squeeze_pos_drift,
        'break_dir': break_dir,
        'expansion_dir': expansion_dir,
        'expansion_price_change': abs(expansion_price_change),
        'dir_match': dir_match,
        'pnl_to_peak': pnl_to_peak,
        'win': 1 if pnl_to_peak > 0 else 0
    })

res_df = pd.DataFrame(results)
print(f"분석 가능 사이클: {len(res_df)}개")
print(f"기본 승률: {res_df['win'].mean()*100:.1f}%")
print(f"평균 수익 (피크까지): {res_df['pnl_to_peak'].mean():+.2f}%")
print(f"방향 일치율: {res_df['dir_match'].mean()*100:.1f}%")

print("\n" + "="*80)
print("📊 사이클 통계")
print("="*80)

print(f"\n수축 기간: 평균 {res_df['squeeze_length'].mean():.1f}시간, 중앙값 {res_df['squeeze_length'].median():.0f}시간")
print(f"확장 기간: 평균 {res_df['expansion_length'].mean():.1f}시간, 중앙값 {res_df['expansion_length'].median():.0f}시간")
print(f"밴드폭 확장률: 평균 {res_df['width_expansion_ratio'].mean():.2f}배, 최대 {res_df['width_expansion_ratio'].max():.2f}배")
print(f"확장 시 가격변동: 평균 {res_df['expansion_price_change'].mean():.2f}%")

print("\n" + "="*80)
print("🔍 확장 크기별 분석")
print("="*80)

# 확장률별 분석
print("\n### 밴드폭 확장률별")
for low, high, label in [(1, 1.5, '약한확장(1-1.5배)'), (1.5, 2, '보통확장(1.5-2배)'), 
                          (2, 3, '강한확장(2-3배)'), (3, 10, '매우강한확장(3배+)')]:
    sub = res_df[(res_df['width_expansion_ratio'] >= low) & (res_df['width_expansion_ratio'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건")
        print(f"    → 방향일치: {sub['dir_match'].mean()*100:.0f}%, 승률: {sub['win'].mean()*100:.0f}%, 평균수익: {sub['pnl_to_peak'].mean():+.2f}%")
        print(f"    → 확장 가격변동: {sub['expansion_price_change'].mean():.2f}%")

print("\n" + "="*80)
print("🔍 수축 마지막 특성 → 확장 결과")
print("="*80)

# 마지막 봉 방향별
print("\n### 마지막 봉 방향 → 돌파/확장 결과")
for d in [1, -1]:
    label = 'UP' if d == 1 else 'DOWN'
    sub = res_df[res_df['last_candle_dir'] == d]
    
    # 돌파 방향 분포
    up_break = (sub['break_dir'] == 1).sum()
    down_break = (sub['break_dir'] == -1).sum()
    
    # 방향 일치율
    match_rate = sub['dir_match'].mean() * 100
    
    print(f"  마지막 {label}: {len(sub)}건")
    print(f"    → UP돌파 {up_break/len(sub)*100:.0f}% / DOWN돌파 {down_break/len(sub)*100:.0f}%")
    print(f"    → 방향일치율: {match_rate:.0f}%, 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl_to_peak'].mean():+.2f}%")

# BB 위치별
print("\n### 마지막 BB 위치 → 확장 결과")
for low, high, label in [(0, 30, '하단(0-30)'), (30, 50, '중하단(30-50)'), 
                          (50, 70, '중상단(50-70)'), (70, 100, '상단(70-100)')]:
    sub = res_df[(res_df['last_bb_pos'] >= low) & (res_df['last_bb_pos'] < high)]
    if len(sub) >= 20:
        up_break = (sub['break_dir'] == 1).sum()
        down_break = (sub['break_dir'] == -1).sum()
        print(f"  {label}: {len(sub)}건")
        print(f"    → UP돌파 {up_break/len(sub)*100:.0f}% / DOWN돌파 {down_break/len(sub)*100:.0f}%")
        print(f"    → 방향일치: {sub['dir_match'].mean()*100:.0f}%, 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl_to_peak'].mean():+.2f}%")

# 수축 중 가격 이동별
print("\n### 수축 중 가격 이동 → 확장 결과")
for low, high, label in [(-10, -1, '하락(-1%이상)'), (-1, 1, '횡보(-1~1%)'), (1, 10, '상승(+1%이상)')]:
    sub = res_df[(res_df['squeeze_price_change'] >= low) & (res_df['squeeze_price_change'] < high)]
    if len(sub) >= 20:
        up_break = (sub['break_dir'] == 1).sum()
        down_break = (sub['break_dir'] == -1).sum()
        print(f"  {label}: {len(sub)}건")
        print(f"    → UP돌파 {up_break/len(sub)*100:.0f}% / DOWN돌파 {down_break/len(sub)*100:.0f}%")
        print(f"    → 방향일치: {sub['dir_match'].mean()*100:.0f}%, 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl_to_peak'].mean():+.2f}%")

print("\n" + "="*80)
print("🎯 수익나는 조합 찾기")
print("="*80)

# 마지막방향 + BB위치 + 돌파방향 조합
print("\n### 마지막방향 + BB위치 + 돌파방향")
for last_d in [1, -1]:
    last_label = 'UP' if last_d == 1 else 'DOWN'
    for pos_low, pos_high, pos_label in [(0, 40, '하단'), (40, 60, '중간'), (60, 100, '상단')]:
        for break_d in [1, -1]:
            break_label = 'LONG' if break_d == 1 else 'SHORT'
            
            sub = res_df[(res_df['last_candle_dir'] == last_d) & 
                         (res_df['last_bb_pos'] >= pos_low) & 
                         (res_df['last_bb_pos'] < pos_high) &
                         (res_df['break_dir'] == break_d)]
            
            if len(sub) >= 15:
                print(f"  마지막{last_label} + {pos_label} + {break_label}: {len(sub)}건")
                print(f"    → 방향일치: {sub['dir_match'].mean()*100:.0f}%, 승률: {sub['win'].mean()*100:.0f}%, 수익: {sub['pnl_to_peak'].mean():+.2f}%")

# 저장
res_df.to_csv('bb30_cycle_results.csv', index=False)
print("\n\n결과 저장: bb30_cycle_results.csv")
