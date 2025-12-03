import pandas as pd
import numpy as np

result_df = pd.read_csv('expanded_squeeze_analysis.csv')

print("=" * 70)
print("핵심 발견: M200 20~30% 구간")
print("=" * 70)

# M200 20~30% 구간이 압도적
# 72h: 72% 승률, +2.81%
# 168h: 68% 승률, +6.42%
# 336h: 76% 승률, +8.34%

# 이 구간을 더 세밀하게 분석
print("\n[M200 ≥ 20% 상세 분석]")
m200_20 = result_df[result_df['momentum_200'] >= 20]
print(f"총 {len(m200_20)}건")

for period in [72, 168, 336]:
    col = f'long_{period}h'
    win = (m200_20[col] > 0).mean() * 100
    avg = m200_20[col].mean()
    total = m200_20[col].sum()
    max_col = f'max_profit_{period}h'
    max_avg = m200_20[max_col].mean()
    print(f"  {period}h: 승률 {win:.1f}%, 평균 {avg:.2f}%, 총 {total:.1f}%, max평균 {max_avg:.1f}%")

# M200 ≥ 15% 분석
print("\n[M200 ≥ 15% 상세 분석]")
m200_15 = result_df[result_df['momentum_200'] >= 15]
print(f"총 {len(m200_15)}건")

for period in [72, 168, 336]:
    col = f'long_{period}h'
    win = (m200_15[col] > 0).mean() * 100
    avg = m200_15[col].mean()
    total = m200_15[col].sum()
    print(f"  {period}h: 승률 {win:.1f}%, 평균 {avg:.2f}%, 총 {total:.1f}%")

# M200 ≥ 10% 분석
print("\n[M200 ≥ 10% 상세 분석]")
m200_10 = result_df[result_df['momentum_200'] >= 10]
print(f"총 {len(m200_10)}건")

for period in [72, 168, 336]:
    col = f'long_{period}h'
    win = (m200_10[col] > 0).mean() * 100
    avg = m200_10[col].mean()
    total = m200_10[col].sum()
    print(f"  {period}h: 승률 {win:.1f}%, 평균 {avg:.2f}%, 총 {total:.1f}%")

print()
print("=" * 70)
print("최적 조합 찾기: M200 + 추가 필터")
print("=" * 70)

# M200 ≥ 15% 기반 + 추가 필터
base = result_df['momentum_200'] >= 15

# 추가 필터 테스트
filters = [
    ('정배열', result_df['ema_bull'] == True),
    ('EMA200↑', result_df['above_ema200'] == True),
    ('HH', result_df['HH'] == True),
    ('HL', result_df['HL'] == True),
    ('HH+HL', (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('위치≥0.7', result_df['position_100'] >= 0.7),
    ('위치≥0.8', result_df['position_100'] >= 0.8),
    ('M100≥5', result_df['momentum_100'] >= 5),
    ('M100≥10', result_df['momentum_100'] >= 10),
    ('M50≥0', result_df['momentum_50'] >= 0),
    ('M50≥5', result_df['momentum_50'] >= 5),
]

print("\n[M200≥15% + 추가 필터]")
for name, filt in filters:
    cond = base & filt
    subset = result_df[cond]
    if len(subset) < 15:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    
    marker = "⭐" if win_168 >= 65 or win_336 >= 70 else ""
    print(f"+{name}: {len(subset)}건")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}% | 336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}% {marker}")

# M200 ≥ 10% 기반 + 추가 필터
print("\n[M200≥10% + 추가 필터]")
base10 = result_df['momentum_200'] >= 10

for name, filt in filters:
    cond = base10 & filt
    subset = result_df[cond]
    if len(subset) < 20:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    total_336 = subset['long_336h'].sum()
    
    marker = "⭐" if win_168 >= 60 and avg_168 >= 3 else ""
    print(f"+{name}: {len(subset)}건")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}% | 336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}%, 총 {total_336:.0f}% {marker}")

print()
print("=" * 70)
print("복합 필터 조합")
print("=" * 70)

# 2개 이상 필터 조합
combos = [
    ('M200≥15 + 정배열 + HH', (result_df['momentum_200'] >= 15) & (result_df['ema_bull'] == True) & (result_df['HH'] == True)),
    ('M200≥15 + 정배열 + 위치≥0.7', (result_df['momentum_200'] >= 15) & (result_df['ema_bull'] == True) & (result_df['position_100'] >= 0.7)),
    ('M200≥15 + HH+HL + M50≥0', (result_df['momentum_200'] >= 15) & (result_df['HH'] == True) & (result_df['HL'] == True) & (result_df['momentum_50'] >= 0)),
    ('M200≥10 + 정배열 + HH', (result_df['momentum_200'] >= 10) & (result_df['ema_bull'] == True) & (result_df['HH'] == True)),
    ('M200≥10 + 정배열 + HH+HL', (result_df['momentum_200'] >= 10) & (result_df['ema_bull'] == True) & (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('M200≥10 + M100≥5 + 정배열', (result_df['momentum_200'] >= 10) & (result_df['momentum_100'] >= 5) & (result_df['ema_bull'] == True)),
    ('M200≥10 + M100≥5 + HH+HL', (result_df['momentum_200'] >= 10) & (result_df['momentum_100'] >= 5) & (result_df['HH'] == True) & (result_df['HL'] == True)),
    ('M200≥20 + 정배열', (result_df['momentum_200'] >= 20) & (result_df['ema_bull'] == True)),
    ('M200≥20 + HH', (result_df['momentum_200'] >= 20) & (result_df['HH'] == True)),
]

for name, cond in combos:
    subset = result_df[cond]
    if len(subset) < 10:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    total_336 = subset['long_336h'].sum()
    
    marker = "⭐⭐" if win_336 >= 70 and len(subset) >= 30 else "⭐" if win_168 >= 65 else ""
    print(f"{name}")
    print(f"  {len(subset)}건, 168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}%")
    print(f"        336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}%, 총 {total_336:.0f}% {marker}")

print()
print("=" * 70)
print("최종 후보 전략")
print("=" * 70)

# 최종 후보들
candidates = [
    ('전략1: M200≥20%', result_df['momentum_200'] >= 20),
    ('전략2: M200≥15% + 정배열', (result_df['momentum_200'] >= 15) & (result_df['ema_bull'] == True)),
    ('전략3: M200≥10% + M100≥10%', (result_df['momentum_200'] >= 10) & (result_df['momentum_100'] >= 10)),
    ('전략4: M200≥10% + 정배열 + HH', (result_df['momentum_200'] >= 10) & (result_df['ema_bull'] == True) & (result_df['HH'] == True)),
]

for name, cond in candidates:
    subset = result_df[cond]
    
    print(f"\n{name}")
    print(f"건수: {len(subset)}건 (5년 기준 연 {len(subset)/5:.1f}건)")
    
    for period in [168, 336]:
        col = f'long_{period}h'
        win = (subset[col] > 0).mean() * 100
        avg = subset[col].mean()
        total = subset[col].sum()
        print(f"  {period}h: 승률 {win:.1f}%, 평균 {avg:.2f}%, 총 {total:.1f}%")

