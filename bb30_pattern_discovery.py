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

# BB 기울기 (중심선 변화율)
df['bb_slope'] = (df['bb_mid'] - df['bb_mid'].shift(1)) / df['bb_mid'].shift(1) * 100

# 밴드폭 변화율
df['bb_width_change'] = df['bb_width'] - df['bb_width'].shift(1)

# 캔들 정보
df['candle_dir'] = np.where(df['close'] > df['open'], 1, -1)  # 1=UP, -1=DOWN
df['candle_body'] = abs(df['close'] - df['open']) / df['open'] * 100
df['candle_range'] = (df['high'] - df['low']) / df['open'] * 100

# 수축 감지
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.05

# 수축 구간 찾기
squeeze_periods = []
in_squeeze = False
squeeze_start = 0

for i in range(len(df)):
    if pd.isna(df.iloc[i]['is_squeeze']):
        continue
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        squeeze_start = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_periods.append({
            'start_idx': squeeze_start,
            'end_idx': i,
            'length': i - squeeze_start
        })

print(f"수축 구간: {len(squeeze_periods)}개")

# 각 수축 구간의 다양한 특성 추출
features = []

for s in squeeze_periods:
    start_idx = s['start_idx']
    end_idx = s['end_idx']
    
    if end_idx >= len(df) - 10 or start_idx < 5:
        continue
    
    squeeze_data = df.iloc[start_idx:end_idx]
    if len(squeeze_data) < 2:
        continue
    
    # === 수축 구간 특성 ===
    
    # 1. 마지막 N봉 방향 패턴
    last_1 = df.iloc[end_idx - 1]['candle_dir']
    last_2 = df.iloc[end_idx - 2]['candle_dir'] if end_idx >= 2 else 0
    last_3 = df.iloc[end_idx - 3]['candle_dir'] if end_idx >= 3 else 0
    last_3_sum = last_1 + last_2 + last_3  # -3~+3
    
    # 2. 마지막 봉 BB 위치
    last_bb_pos = df.iloc[end_idx - 1]['bb_position']
    
    # 3. 수축 중 평균 BB 위치
    avg_bb_pos = squeeze_data['bb_position'].mean()
    
    # 4. 수축 중 BB 위치 이동 방향 (시작 vs 끝)
    pos_start = squeeze_data.iloc[0]['bb_position'] if len(squeeze_data) > 0 else 50
    pos_end = squeeze_data.iloc[-1]['bb_position'] if len(squeeze_data) > 0 else 50
    pos_drift = pos_end - pos_start
    
    # 5. 수축 중 기울기 평균
    avg_slope = squeeze_data['bb_slope'].mean()
    
    # 6. 밴드폭 (수축 강도)
    min_width = squeeze_data['bb_width'].min()
    
    # 7. 수축 길이
    length = s['length']
    
    # 8. 마지막 봉 크기
    last_body = df.iloc[end_idx - 1]['candle_body']
    
    # 9. 마지막 3봉 평균 크기
    last_3_body = df.iloc[end_idx-3:end_idx]['candle_body'].mean() if end_idx >= 3 else 0
    
    # 10. 돌파 직전 밴드폭 변화 (확장 시작 신호)
    width_change_last = df.iloc[end_idx - 1]['bb_width_change']
    
    # === 결과 (돌파 후 10봉 수익) ===
    entry_price = df.iloc[end_idx]['close']
    future_price = df.iloc[end_idx + 10]['close']
    break_dir = 1 if df.iloc[end_idx]['close'] > df.iloc[end_idx]['open'] else -1
    
    if break_dir == 1:
        pnl = (future_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - future_price) / entry_price * 100
    
    features.append({
        'datetime': df.iloc[end_idx]['datetime'],
        'length': length,
        'last_1_dir': last_1,
        'last_3_sum': last_3_sum,
        'last_bb_pos': last_bb_pos,
        'avg_bb_pos': avg_bb_pos,
        'pos_drift': pos_drift,
        'avg_slope': avg_slope,
        'min_width': min_width,
        'last_body': last_body,
        'last_3_body': last_3_body,
        'width_change': width_change_last,
        'break_dir': break_dir,
        'pnl': pnl,
        'win': 1 if pnl > 0 else 0
    })

feat_df = pd.DataFrame(features)
print(f"\n분석 가능 케이스: {len(feat_df)}개")
print(f"기본 승률: {feat_df['win'].mean()*100:.1f}%")
print(f"기본 평균 수익: {feat_df['pnl'].mean():+.2f}%")

# === 패턴 탐색 ===
print("\n" + "="*80)
print("🔍 패턴 탐색 - 어떤 조건이 수익에 영향을 주는가?")
print("="*80)

# 1. 마지막 봉 방향별
print("\n### 1. 마지막 1봉 방향")
for d in [1, -1]:
    sub = feat_df[feat_df['last_1_dir'] == d]
    label = 'UP' if d == 1 else 'DOWN'
    print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 2. 마지막 3봉 방향 합계별
print("\n### 2. 마지막 3봉 방향 합계 (-3=모두DOWN, +3=모두UP)")
for s in [-3, -1, 1, 3]:
    sub = feat_df[feat_df['last_3_sum'] == s]
    if len(sub) >= 10:
        print(f"  합계 {s:+d}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 3. BB 위치별
print("\n### 3. 마지막 봉 BB 위치")
for low, high, label in [(0, 20, '하단(0-20)'), (20, 40, '중하단(20-40)'), 
                          (40, 60, '중간(40-60)'), (60, 80, '중상단(60-80)'), (80, 100, '상단(80-100)')]:
    sub = feat_df[(feat_df['last_bb_pos'] >= low) & (feat_df['last_bb_pos'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 4. 위치 이동 방향 (Drift)
print("\n### 4. 수축 중 BB 위치 이동 (시작→끝)")
for low, high, label in [(-100, -20, '하락이동(-20이하)'), (-20, -5, '약한하락'), 
                          (-5, 5, '횡보'), (5, 20, '약한상승'), (20, 100, '상승이동(+20이상)')]:
    sub = feat_df[(feat_df['pos_drift'] >= low) & (feat_df['pos_drift'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 5. 기울기별
print("\n### 5. 수축 중 평균 기울기")
slope_q = feat_df['avg_slope'].quantile([0.2, 0.4, 0.6, 0.8])
for low, high, label in [(-10, slope_q[0.2], '강한하락'), (slope_q[0.2], slope_q[0.4], '약한하락'),
                          (slope_q[0.4], slope_q[0.6], '횡보'), (slope_q[0.6], slope_q[0.8], '약한상승'),
                          (slope_q[0.8], 10, '강한상승')]:
    sub = feat_df[(feat_df['avg_slope'] >= low) & (feat_df['avg_slope'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 6. 수축 강도 (밴드폭)
print("\n### 6. 수축 강도 (최소 밴드폭)")
width_q = feat_df['min_width'].quantile([0.2, 0.4, 0.6, 0.8])
for low, high, label in [(0, width_q[0.2], '매우강한수축'), (width_q[0.2], width_q[0.4], '강한수축'),
                          (width_q[0.4], width_q[0.6], '보통수축'), (width_q[0.6], width_q[0.8], '약한수축'),
                          (width_q[0.8], 100, '매우약한수축')]:
    sub = feat_df[(feat_df['min_width'] >= low) & (feat_df['min_width'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 7. 수축 길이별
print("\n### 7. 수축 길이")
for low, high, label in [(1, 5, '1-5시간'), (6, 10, '6-10시간'), 
                          (11, 20, '11-20시간'), (21, 50, '21-50시간')]:
    sub = feat_df[(feat_df['length'] >= low) & (feat_df['length'] <= high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 8. 마지막 봉 크기
print("\n### 8. 마지막 봉 크기 (body %)")
body_q = feat_df['last_body'].quantile([0.2, 0.5, 0.8])
for low, high, label in [(0, body_q[0.2], '작은봉'), (body_q[0.2], body_q[0.5], '보통봉'),
                          (body_q[0.5], body_q[0.8], '큰봉'), (body_q[0.8], 100, '매우큰봉')]:
    sub = feat_df[(feat_df['last_body'] >= low) & (feat_df['last_body'] < high)]
    if len(sub) >= 10:
        print(f"  {label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# === 조합 패턴 탐색 ===
print("\n" + "="*80)
print("🎯 조합 패턴 탐색")
print("="*80)

# 방향 + 위치 조합
print("\n### 마지막 방향 + BB 위치 조합")
for d in [1, -1]:
    dir_label = 'UP' if d == 1 else 'DOWN'
    for pos_low, pos_high, pos_label in [(0, 30, '하단'), (30, 70, '중간'), (70, 100, '상단')]:
        sub = feat_df[(feat_df['last_1_dir'] == d) & 
                      (feat_df['last_bb_pos'] >= pos_low) & 
                      (feat_df['last_bb_pos'] < pos_high)]
        if len(sub) >= 20:
            print(f"  {dir_label} + {pos_label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 기울기 + 위치 조합
print("\n### 기울기 + BB 위치 조합")
slope_med = feat_df['avg_slope'].median()
for slope_cond, slope_label in [(feat_df['avg_slope'] > slope_med, '상승기울기'), 
                                 (feat_df['avg_slope'] <= slope_med, '하락기울기')]:
    for pos_low, pos_high, pos_label in [(0, 30, '하단'), (70, 100, '상단')]:
        sub = feat_df[slope_cond & 
                      (feat_df['last_bb_pos'] >= pos_low) & 
                      (feat_df['last_bb_pos'] < pos_high)]
        if len(sub) >= 20:
            print(f"  {slope_label} + {pos_label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 수축강도 + 위치 이동 조합
print("\n### 수축강도 + 위치이동 조합")
width_med = feat_df['min_width'].median()
for width_cond, width_label in [(feat_df['min_width'] < width_med, '강한수축'), 
                                 (feat_df['min_width'] >= width_med, '약한수축')]:
    for drift_cond, drift_label in [(feat_df['pos_drift'] > 10, '상승이동'), 
                                     (feat_df['pos_drift'] < -10, '하락이동')]:
        sub = feat_df[width_cond & drift_cond]
        if len(sub) >= 20:
            print(f"  {width_label} + {drift_label}: {len(sub)}건, 승률 {sub['win'].mean()*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")

# 저장
feat_df.to_csv('bb30_pattern_features.csv', index=False)
print("\n\n특성 데이터 저장: bb30_pattern_features.csv")
