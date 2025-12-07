#!/usr/bin/env python3
"""
BB 하나만 깊게 판다 - v2
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 집중 분석")
print("=" * 80)

# 4시간봉
df = pd.read_csv('analysis_4h.csv', parse_dates=['datetime'])
print(f"데이터: {len(df)}개 (4시간봉)")
print(f"기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")

# BB 계산
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # %

# 밴드폭의 최근 20봉 최저값
df['bb_width_min20'] = df['bb_width'].rolling(20).min()

df = df.iloc[50:].reset_index(drop=True)  # 초반 NaN 제거
print(f"분석 대상: {len(df)}개")

# ============================================================
# 1. 밴드폭 분포
# ============================================================
print("\n" + "=" * 80)
print("1. 밴드폭 분포")
print("=" * 80)

print(f"평균: {df['bb_width'].mean():.2f}%")
print(f"중앙값: {df['bb_width'].median():.2f}%")
for q in [10, 25, 50, 75, 90]:
    print(f"  {q}%: {df['bb_width'].quantile(q/100):.2f}%")

# ============================================================
# 2. 수축 정의: 밴드폭이 최근 20봉 최저
# ============================================================
print("\n" + "=" * 80)
print("2. 수축 감지")
print("=" * 80)

df['is_squeeze'] = df['bb_width'] <= df['bb_width_min20'] * 1.05  # 5% 여유

squeeze_count = df['is_squeeze'].sum()
print(f"수축 상태 봉: {squeeze_count}개 ({squeeze_count/len(df)*100:.1f}%)")

# ============================================================
# 3. 수축 → 확장 전환점 찾기
# ============================================================
print("\n" + "=" * 80)
print("3. 수축→확장 전환점")
print("=" * 80)

# 전봉이 수축, 현재봉이 비수축 = 확장 시작
df['expansion_start'] = df['is_squeeze'].shift(1) & ~df['is_squeeze']

expansion_points = df[df['expansion_start']].index.tolist()
print(f"확장 시작점: {len(expansion_points)}개")

# ============================================================
# 4. 확장 시작 후 가격 움직임 분석
# ============================================================
print("\n" + "=" * 80)
print("4. 확장 후 가격 움직임")
print("=" * 80)

close = df['close'].values
high = df['high'].values
low = df['low'].values
bb_upper = df['bb_upper'].values
bb_lower = df['bb_lower'].values
bb_mid = df['bb_mid'].values
bb_width = df['bb_width'].values

results = []

for idx in expansion_points:
    if idx + 40 >= len(df):
        continue
    
    # 확장 시작 시점 정보
    entry_price = close[idx]
    entry_width = bb_width[idx]
    
    # 가격이 밴드 어디에 있는지
    bb_position = (entry_price - bb_lower[idx]) / (bb_upper[idx] - bb_lower[idx])
    
    # 첫 밴드 돌파 방향 (5봉 이내)
    first_break = None
    break_bar = None
    for j in range(idx, min(idx + 5, len(df))):
        if close[j] > bb_upper[j]:
            first_break = 'UP'
            break_bar = j - idx
            break
        elif close[j] < bb_lower[j]:
            first_break = 'DOWN'
            break_bar = j - idx
            break
    
    # 미래 수익률 (돌파 방향 기준)
    future = {}
    for bars in [5, 10, 20, 30]:
        if idx + bars < len(df):
            future_price = close[idx + bars]
            ret = (future_price - entry_price) / entry_price * 100
            future[f'ret_{bars}'] = ret
    
    # MFE (최대 유리 움직임)
    mfe_up = 0
    mfe_down = 0
    for j in range(idx + 1, min(idx + 30, len(df))):
        mfe_up = max(mfe_up, (high[j] - entry_price) / entry_price * 100)
        mfe_down = max(mfe_down, (entry_price - low[j]) / entry_price * 100)
    
    results.append({
        'idx': idx,
        'datetime': df.iloc[idx]['datetime'],
        'bb_width': entry_width,
        'bb_position': bb_position,  # 0=하단, 0.5=중간, 1=상단
        'first_break': first_break,
        'break_bar': break_bar,
        'mfe_up': mfe_up,
        'mfe_down': mfe_down,
        **future
    })

df_r = pd.DataFrame(results)
print(f"분석 완료: {len(df_r)}개 확장점")

# ============================================================
# 5. 핵심 분석: 돌파 방향 예측 가능성
# ============================================================
print("\n" + "=" * 80)
print("5. 돌파 방향 분석")
print("=" * 80)

# 돌파 방향 분포
break_dist = df_r['first_break'].value_counts()
print(f"\n돌파 방향:")
for k, v in break_dist.items():
    if k:
        print(f"  {k}: {v}건 ({v/len(df_r)*100:.1f}%)")

none_count = df_r['first_break'].isna().sum()
print(f"  돌파 없음: {none_count}건")

# ============================================================
# 6. BB 위치와 돌파 방향 관계
# ============================================================
print("\n" + "=" * 80)
print("6. BB 위치 → 돌파 방향 예측")
print("=" * 80)

df_r['position_group'] = pd.cut(df_r['bb_position'], 
                                 bins=[0, 0.3, 0.5, 0.7, 1.0],
                                 labels=['하단(0-30%)', '중하(30-50%)', '중상(50-70%)', '상단(70-100%)'])

for pos in ['하단(0-30%)', '중하(30-50%)', '중상(50-70%)', '상단(70-100%)']:
    sub = df_r[df_r['position_group'] == pos]
    if len(sub) >= 5:
        up_pct = (sub['first_break'] == 'UP').sum() / len(sub[sub['first_break'].notna()]) * 100 if sub['first_break'].notna().sum() > 0 else 0
        print(f"\n{pos} ({len(sub)}건):")
        print(f"  → UP 돌파: {up_pct:.1f}%")
        print(f"  → DOWN 돌파: {100-up_pct:.1f}%")

# ============================================================
# 7. 돌파 방향으로 진입 시 수익률
# ============================================================
print("\n" + "=" * 80)
print("7. 돌파 방향 추종 시 수익률")
print("=" * 80)

for direction in ['UP', 'DOWN']:
    sub = df_r[df_r['first_break'] == direction].copy()
    if len(sub) < 10:
        continue
    
    print(f"\n{direction} 돌파 추종 ({len(sub)}건):")
    
    # 수익률 계산 (방향에 맞게)
    for bars in [5, 10, 20, 30]:
        col = f'ret_{bars}'
        if col in sub.columns:
            if direction == 'UP':
                ret = sub[col]
            else:
                ret = -sub[col]  # SHORT은 반대
            
            avg = ret.mean()
            win = (ret > 0).mean() * 100
            print(f"  {bars}봉 후: 평균 {avg:+.2f}%, 승률 {win:.1f}%")
    
    mfe_col = 'mfe_up' if direction == 'UP' else 'mfe_down'
    print(f"  MFE: {sub[mfe_col].mean():.2f}%")

# ============================================================
# 8. 밴드폭 크기별 수익률
# ============================================================
print("\n" + "=" * 80)
print("8. 밴드폭(수축 강도)별 수익률")
print("=" * 80)

df_r['width_group'] = pd.cut(df_r['bb_width'],
                              bins=[0, 3, 5, 7, 10, 100],
                              labels=['<3%', '3-5%', '5-7%', '7-10%', '10%+'])

for group in ['<3%', '3-5%', '5-7%', '7-10%', '10%+']:
    sub = df_r[df_r['width_group'] == group]
    if len(sub) >= 10:
        print(f"\n밴드폭 {group} ({len(sub)}건):")
        
        # 돌파 방향 추종 수익
        sub_with_break = sub[sub['first_break'].notna()].copy()
        if len(sub_with_break) > 0:
            returns = []
            for _, row in sub_with_break.iterrows():
                if row['first_break'] == 'UP':
                    returns.append(row.get('ret_10', 0))
                else:
                    returns.append(-row.get('ret_10', 0))
            
            avg = np.mean(returns)
            win = sum(1 for r in returns if r > 0) / len(returns) * 100
            print(f"  10봉 후 (추종): 평균 {avg:+.2f}%, 승률 {win:.1f}%")
        
        print(f"  MFE(상): {sub['mfe_up'].mean():.2f}%, MFE(하): {sub['mfe_down'].mean():.2f}%")

# ============================================================
# 9. 핵심 발견
# ============================================================
print("\n" + "=" * 80)
print("★★★ 핵심 발견 ★★★")
print("=" * 80)

# 가장 좋은 조건 찾기
print("\n분석 결과:")

# BB 위치와 돌파 예측
df_upper = df_r[df_r['bb_position'] > 0.7]
df_lower = df_r[df_r['bb_position'] < 0.3]

if len(df_upper) > 0:
    up_breaks_upper = (df_upper['first_break'] == 'UP').sum()
    total_upper = df_upper['first_break'].notna().sum()
    if total_upper > 0:
        print(f"\n1. 상단(70%+)에서 확장 → UP 돌파 확률: {up_breaks_upper/total_upper*100:.1f}%")

if len(df_lower) > 0:
    down_breaks_lower = (df_lower['first_break'] == 'DOWN').sum()
    total_lower = df_lower['first_break'].notna().sum()
    if total_lower > 0:
        print(f"2. 하단(30%-)에서 확장 → DOWN 돌파 확률: {down_breaks_lower/total_lower*100:.1f}%")

# 돌파 추종 수익률
for direction in ['UP', 'DOWN']:
    sub = df_r[df_r['first_break'] == direction]
    if len(sub) >= 20 and 'ret_10' in sub.columns:
        if direction == 'UP':
            ret = sub['ret_10']
        else:
            ret = -sub['ret_10']
        
        avg = ret.mean()
        win = (ret > 0).mean() * 100
        print(f"\n3. {direction} 돌파 추종: 평균 {avg:+.2f}%, 승률 {win:.1f}% ({len(sub)}건)")

# 저장
df_r.to_csv('bb_deep_v2_results.csv', index=False)
print(f"\n저장: bb_deep_v2_results.csv")
