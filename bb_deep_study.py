#!/usr/bin/env python3
"""
BB 하나만 깊게 판다
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 집중 분석")
print("=" * 80)

# 4시간봉 사용
df = pd.read_csv('analysis_4h.csv', parse_dates=['datetime'])
df = df.iloc[200:].reset_index(drop=True)

# BB 계산
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # 밴드폭 %

df = df.dropna().reset_index(drop=True)
print(f"데이터: {len(df)}개 (4시간봉)")

# ============================================================
# 1. 밴드폭 분포 분석
# ============================================================
print("\n" + "=" * 80)
print("1. 밴드폭(%) 분포")
print("=" * 80)

print(f"\n밴드폭 통계:")
print(f"  최소: {df['bb_width'].min():.2f}%")
print(f"  최대: {df['bb_width'].max():.2f}%")
print(f"  평균: {df['bb_width'].mean():.2f}%")
print(f"  중앙값: {df['bb_width'].median():.2f}%")

# 분위수
for q in [10, 25, 50, 75, 90]:
    val = df['bb_width'].quantile(q/100)
    print(f"  {q}%ile: {val:.2f}%")

# ============================================================
# 2. 수축 구간 정의 및 분석
# ============================================================
print("\n" + "=" * 80)
print("2. 수축 구간 분석")
print("=" * 80)

# 수축 = 밴드폭이 최근 N봉 중 최저
df['is_squeeze'] = df['bb_width'] == df['bb_width'].rolling(20).min()

# 수축 구간 찾기
squeeze_starts = []
in_squeeze = False
start_idx = 0

for i in range(len(df)):
    if df.iloc[i]['is_squeeze'] and not in_squeeze:
        in_squeeze = True
        start_idx = i
    elif not df.iloc[i]['is_squeeze'] and in_squeeze:
        in_squeeze = False
        squeeze_starts.append({
            'start_idx': start_idx,
            'end_idx': i - 1,
            'length': i - start_idx,
            'min_width': df.iloc[start_idx:i]['bb_width'].min(),
            'datetime': df.iloc[start_idx]['datetime']
        })

print(f"\n수축 구간: {len(squeeze_starts)}개 발견")

if squeeze_starts:
    lengths = [s['length'] for s in squeeze_starts]
    print(f"수축 길이 분포:")
    print(f"  평균: {np.mean(lengths):.1f}봉")
    print(f"  최소: {min(lengths)}봉")
    print(f"  최대: {max(lengths)}봉")

# ============================================================
# 3. 수축 후 움직임 분석 (핵심!)
# ============================================================
print("\n" + "=" * 80)
print("3. 수축 종료 후 가격 움직임")
print("=" * 80)

close = df['close'].values
high = df['high'].values
low = df['low'].values
bb_upper = df['bb_upper'].values
bb_lower = df['bb_lower'].values
bb_mid = df['bb_mid'].values

results = []

for sq in squeeze_starts:
    end_idx = sq['end_idx']
    if end_idx + 30 >= len(df):
        continue
    
    entry_price = close[end_idx]
    entry_bb_upper = bb_upper[end_idx]
    entry_bb_lower = bb_lower[end_idx]
    entry_bb_mid = bb_mid[end_idx]
    
    # 진입 시점 가격 위치
    if entry_price > entry_bb_mid:
        position = 'UPPER_HALF'
    else:
        position = 'LOWER_HALF'
    
    # 첫 돌파 방향
    first_break = None
    break_idx = None
    for j in range(end_idx + 1, min(end_idx + 20, len(df))):
        if high[j] > bb_upper[j] and first_break is None:
            first_break = 'UP'
            break_idx = j
            break
        elif low[j] < bb_lower[j] and first_break is None:
            first_break = 'DOWN'
            break_idx = j
            break
    
    if first_break is None:
        continue
    
    # 돌파 후 10봉, 20봉, 30봉 수익률
    break_price = close[break_idx]
    
    future_returns = {}
    for bars in [5, 10, 20, 30]:
        if break_idx + bars < len(df):
            future_price = close[break_idx + bars]
            ret = (future_price - break_price) / break_price * 100
            if first_break == 'DOWN':
                ret = -ret  # SHORT 기준으로 변환
            future_returns[f'ret_{bars}'] = ret
    
    # MFE (Maximum Favorable Excursion)
    mfe = 0
    for j in range(break_idx + 1, min(break_idx + 30, len(df))):
        if first_break == 'UP':
            mfe = max(mfe, (high[j] - break_price) / break_price * 100)
        else:
            mfe = max(mfe, (break_price - low[j]) / break_price * 100)
    
    results.append({
        'datetime': df.iloc[end_idx]['datetime'],
        'squeeze_length': sq['length'],
        'min_width': sq['min_width'],
        'position': position,
        'first_break': first_break,
        'mfe': mfe,
        **future_returns
    })

df_results = pd.DataFrame(results)
print(f"\n분석 가능한 수축: {len(df_results)}개")

# ============================================================
# 4. 패턴별 수익률 분석
# ============================================================
print("\n" + "=" * 80)
print("4. 돌파 방향별 수익률")
print("=" * 80)

for direction in ['UP', 'DOWN']:
    sub = df_results[df_results['first_break'] == direction]
    print(f"\n{direction} 돌파 ({len(sub)}건):")
    for col in ['ret_5', 'ret_10', 'ret_20', 'ret_30']:
        if col in sub.columns:
            avg = sub[col].mean()
            win = (sub[col] > 0).mean() * 100
            print(f"  {col.replace('ret_', '')}봉 후: 평균 {avg:+.2f}%, 양수 {win:.1f}%")
    print(f"  MFE 평균: {sub['mfe'].mean():.2f}%")

# ============================================================
# 5. 수축 길이별 분석
# ============================================================
print("\n" + "=" * 80)
print("5. 수축 길이별 수익률")
print("=" * 80)

df_results['sq_group'] = pd.cut(df_results['squeeze_length'], 
                                 bins=[0, 5, 10, 20, 50, 1000],
                                 labels=['1-5', '6-10', '11-20', '21-50', '50+'])

for group in ['1-5', '6-10', '11-20', '21-50', '50+']:
    sub = df_results[df_results['sq_group'] == group]
    if len(sub) >= 5:
        print(f"\n수축 {group}봉 ({len(sub)}건):")
        if 'ret_10' in sub.columns:
            avg = sub['ret_10'].mean()
            win = (sub['ret_10'] > 0).mean() * 100
            print(f"  10봉 후: 평균 {avg:+.2f}%, 양수 {win:.1f}%")
        print(f"  MFE: {sub['mfe'].mean():.2f}%")

# ============================================================
# 6. 밴드폭(수축 강도)별 분석
# ============================================================
print("\n" + "=" * 80)
print("6. 밴드폭(수축 강도)별 수익률")
print("=" * 80)

df_results['width_group'] = pd.cut(df_results['min_width'],
                                    bins=[0, 2, 3, 4, 5, 100],
                                    labels=['<2%', '2-3%', '3-4%', '4-5%', '5%+'])

for group in ['<2%', '2-3%', '3-4%', '4-5%', '5%+']:
    sub = df_results[df_results['width_group'] == group]
    if len(sub) >= 5:
        print(f"\n밴드폭 {group} ({len(sub)}건):")
        if 'ret_10' in sub.columns:
            avg = sub['ret_10'].mean()
            win = (sub['ret_10'] > 0).mean() * 100
            print(f"  10봉 후: 평균 {avg:+.2f}%, 양수 {win:.1f}%")
        print(f"  MFE: {sub['mfe'].mean():.2f}%")

# ============================================================
# 7. 진입 위치별 분석
# ============================================================
print("\n" + "=" * 80)
print("7. 수축 종료 시 가격 위치별")
print("=" * 80)

for pos in ['UPPER_HALF', 'LOWER_HALF']:
    sub = df_results[df_results['position'] == pos]
    print(f"\n{pos} ({len(sub)}건):")
    
    # 돌파 방향
    up_pct = (sub['first_break'] == 'UP').mean() * 100
    print(f"  → UP 돌파: {up_pct:.1f}%")
    print(f"  → DOWN 돌파: {100-up_pct:.1f}%")
    
    if 'ret_10' in sub.columns:
        avg = sub['ret_10'].mean()
        win = (sub['ret_10'] > 0).mean() * 100
        print(f"  10봉 후: 평균 {avg:+.2f}%, 양수 {win:.1f}%")

# ============================================================
# 8. 핵심 발견 정리
# ============================================================
print("\n" + "=" * 80)
print("★ 핵심 발견 ★")
print("=" * 80)

# 가장 좋은 조건 찾기
best_conditions = []

# 수축 길이별
for group in df_results['sq_group'].unique():
    sub = df_results[df_results['sq_group'] == group]
    if len(sub) >= 10 and 'ret_10' in sub.columns:
        avg = sub['ret_10'].mean()
        win = (sub['ret_10'] > 0).mean() * 100
        if avg > 0.5 and win > 55:
            best_conditions.append(f"수축 {group}봉: {avg:+.2f}%, 승률 {win:.0f}%")

# 밴드폭별
for group in df_results['width_group'].unique():
    sub = df_results[df_results['width_group'] == group]
    if len(sub) >= 10 and 'ret_10' in sub.columns:
        avg = sub['ret_10'].mean()
        win = (sub['ret_10'] > 0).mean() * 100
        if avg > 0.5 and win > 55:
            best_conditions.append(f"밴드폭 {group}: {avg:+.2f}%, 승률 {win:.0f}%")

if best_conditions:
    print("\n유망한 조건:")
    for cond in best_conditions:
        print(f"  ✓ {cond}")
else:
    print("\n뚜렷하게 유망한 조건 없음. 더 깊이 분석 필요.")

# 저장
df_results.to_csv('bb_deep_study_results.csv', index=False)
print(f"\n저장: bb_deep_study_results.csv")
