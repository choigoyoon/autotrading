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
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # %

# BB 내 위치 (0~100%)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# 수축 감지: 최근 30봉 중 밴드폭 최저
df['bb_width_min30'] = df['bb_width'].rolling(30).min()
df['is_squeeze'] = df['bb_width'] <= df['bb_width_min30'] * 1.05  # 5% 여유

# 방향성: 현재 봉의 방향
df['candle_dir'] = np.where(df['close'] > df['open'], 'UP', 'DOWN')

# 확장 시작점 찾기
df['squeeze_end'] = df['is_squeeze'].shift(1) & ~df['is_squeeze']

# 수축 기간 계산
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
        squeeze_len = i - squeeze_start
        squeeze_periods.append({
            'end_idx': i,
            'length': squeeze_len,
            'datetime': df.iloc[i]['datetime']
        })

print(f"\n수축 기간 수: {len(squeeze_periods)}")
print(f"평균 수축 기간: {np.mean([s['length'] for s in squeeze_periods]):.1f}시간 (봉)")
print(f"중앙값: {np.median([s['length'] for s in squeeze_periods]):.1f}시간")
print(f"최소: {np.min([s['length'] for s in squeeze_periods])}시간")
print(f"최대: {np.max([s['length'] for s in squeeze_periods])}시간")

# 수축 기간별 분류
duration_groups = [
    (1, 5, "1-5시간"),
    (6, 10, "6-10시간"),
    (11, 20, "11-20시간"),
    (21, 50, "21-50시간"),
    (51, 100, "51-100시간")
]

print("\n" + "="*80)
print("📊 BB 30 - 1시간봉 수축 기간별 분석")
print("="*80)

for min_len, max_len, label in duration_groups:
    group = [s for s in squeeze_periods if min_len <= s['length'] <= max_len]
    if len(group) < 10:
        continue
    
    up_count = 0
    down_count = 0
    last_up_match = 0
    last_down_match = 0
    
    # 조건별 성과
    results = {
        'up_upper': [],  # 마지막 UP + 상단
        'down_lower': [],  # 마지막 DOWN + 하단
    }
    
    for s in group:
        idx = s['end_idx']
        if idx >= len(df) - 10:
            continue
        
        # 수축 마지막 봉 정보
        last_idx = idx - 1
        if last_idx < 0:
            continue
            
        last_dir = df.iloc[last_idx]['candle_dir']
        last_pos = df.iloc[last_idx]['bb_position']
        
        # 돌파 방향 (확장 첫 봉)
        break_dir = 'UP' if df.iloc[idx]['close'] > df.iloc[idx]['open'] else 'DOWN'
        
        if break_dir == 'UP':
            up_count += 1
        else:
            down_count += 1
        
        # 방향 일치 체크
        if last_dir == break_dir:
            if last_dir == 'UP':
                last_up_match += 1
            else:
                last_down_match += 1
        
        # 10봉 후 수익
        future_close = df.iloc[idx + 10]['close']
        entry_close = df.iloc[idx]['close']
        
        if break_dir == 'UP':
            pnl = (future_close - entry_close) / entry_close * 100
        else:
            pnl = (entry_close - future_close) / entry_close * 100
        
        # 조건별 분류
        if last_dir == 'UP' and last_pos > 50:  # 마지막 UP + 상단
            results['up_upper'].append(pnl)
        elif last_dir == 'DOWN' and last_pos < 50:  # 마지막 DOWN + 하단
            results['down_lower'].append(pnl)
    
    total = up_count + down_count
    if total == 0:
        continue
    
    print(f"\n### {label} ({len(group)}개)")
    print(f"돌파 방향: UP {up_count} ({up_count/total*100:.0f}%) / DOWN {down_count} ({down_count/total*100:.0f}%)")
    
    # 방향 일치율
    up_total = sum(1 for s in group if df.iloc[s['end_idx']-1]['candle_dir'] == 'UP')
    down_total = len(group) - up_total
    
    if up_total > 0:
        print(f"마지막 UP → UP돌파: {last_up_match}/{up_total} ({last_up_match/up_total*100:.0f}%)")
    if down_total > 0:
        print(f"마지막 DOWN → DOWN돌파: {last_down_match}/{down_total} ({last_down_match/down_total*100:.0f}%)")
    
    # 조건별 성과
    for key, pnls in results.items():
        if len(pnls) >= 5:
            avg_pnl = np.mean(pnls)
            win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
            print(f"  → {key}: {len(pnls)}건, 평균 {avg_pnl:+.2f}%, 승률 {win_rate:.0f}%")

# 최적 조건 종합
print("\n" + "="*80)
print("🎯 BB 30 최적 조건 종합")
print("="*80)

optimal_results = []
for s in squeeze_periods:
    idx = s['end_idx']
    if idx >= len(df) - 10 or idx < 1:
        continue
    
    last_idx = idx - 1
    last_dir = df.iloc[last_idx]['candle_dir']
    last_pos = df.iloc[last_idx]['bb_position']
    break_dir = 'UP' if df.iloc[idx]['close'] > df.iloc[idx]['open'] else 'DOWN'
    
    future_close = df.iloc[idx + 10]['close']
    entry_close = df.iloc[idx]['close']
    
    if break_dir == 'UP':
        pnl = (future_close - entry_close) / entry_close * 100
    else:
        pnl = (entry_close - future_close) / entry_close * 100
    
    # 최적 조건: 마지막 방향 + BB 위치 + 돌파 방향 일치
    if (last_dir == 'UP' and last_pos > 70 and break_dir == 'UP') or \
       (last_dir == 'DOWN' and last_pos < 30 and break_dir == 'DOWN'):
        optimal_results.append({
            'datetime': df.iloc[idx]['datetime'],
            'direction': break_dir,
            'pnl': pnl,
            'length': s['length']
        })

if optimal_results:
    print(f"\n최적 조건 (마지막 방향 + BB위치 70/30 + 돌파 일치)")
    print(f"총 {len(optimal_results)}건")
    print(f"평균 수익: {np.mean([r['pnl'] for r in optimal_results]):+.2f}%")
    print(f"승률: {len([r for r in optimal_results if r['pnl'] > 0])/len(optimal_results)*100:.1f}%")
    
    up_results = [r for r in optimal_results if r['direction'] == 'UP']
    down_results = [r for r in optimal_results if r['direction'] == 'DOWN']
    
    if up_results:
        print(f"\n  LONG: {len(up_results)}건, 평균 {np.mean([r['pnl'] for r in up_results]):+.2f}%, 승률 {len([r for r in up_results if r['pnl'] > 0])/len(up_results)*100:.0f}%")
    if down_results:
        print(f"  SHORT: {len(down_results)}건, 평균 {np.mean([r['pnl'] for r in down_results]):+.2f}%, 승률 {len([r for r in down_results if r['pnl'] > 0])/len(down_results)*100:.0f}%")

# 결과 저장
pd.DataFrame(optimal_results).to_csv('bb_1h_30_results.csv', index=False)
print("\n결과 저장: bb_1h_30_results.csv")
