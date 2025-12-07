import pandas as pd
import numpy as np

result_df = pd.read_csv('expanded_squeeze_analysis.csv')

print("=" * 70)
print("나머지 매매 기회 분석")
print("=" * 70)
print(f"전체 수축/확장: {len(result_df)}건")
print()

# 모멘텀 구간별 분포
print("=== 모멘텀200 구간별 분포 ===")
bins = [-100, 0, 5, 10, 15, 20, 100]
labels = ['음수', '0~5%', '5~10%', '10~15%', '15~20%', '20%+']
result_df['m200_range'] = pd.cut(result_df['momentum_200'], bins=bins, labels=labels)

for label in labels:
    subset = result_df[result_df['m200_range'] == label]
    if len(subset) == 0:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    
    print(f"\n{label}: {len(subset)}건")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}%")
    print(f"  336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}%")

print()
print("=" * 70)
print("M200 0% 미만 (하락 중) 분석")
print("=" * 70)

negative = result_df[result_df['momentum_200'] < 0]
print(f"총 {len(negative)}건")

# 하락 중에도 수익 나는 조건
print("\n[하락 중 + 추가 조건]")

conditions = [
    ('정배열', negative['ema_bull'] == True),
    ('EMA200 위', negative['above_ema200'] == True),
    ('HH', negative['HH'] == True),
    ('HL', negative['HL'] == True),
    ('HH+HL', (negative['HH'] == True) & (negative['HL'] == True)),
    ('위치≥0.6', negative['position_100'] >= 0.6),
    ('위치≥0.7', negative['position_100'] >= 0.7),
    ('M100≥0', negative['momentum_100'] >= 0),
    ('M100≥5', negative['momentum_100'] >= 5),
    ('M50≥0', negative['momentum_50'] >= 0),
    ('M50≥5', negative['momentum_50'] >= 5),
]

for name, cond in conditions:
    subset = negative[cond]
    if len(subset) < 20:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    
    marker = "⭐" if win_168 >= 55 or win_336 >= 60 else ""
    print(f"{name}: {len(subset)}건")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}% | 336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}% {marker}")

print()
print("=" * 70)
print("M200 0~10% (약한 상승) 분석")
print("=" * 70)

weak = result_df[(result_df['momentum_200'] >= 0) & (result_df['momentum_200'] < 10)]
print(f"총 {len(weak)}건")

print("\n[약한 상승 + 추가 조건]")
for name, filt in [
    ('정배열', weak['ema_bull'] == True),
    ('EMA200 위', weak['above_ema200'] == True),
    ('HH+HL', (weak['HH'] == True) & (weak['HL'] == True)),
    ('위치≥0.7', weak['position_100'] >= 0.7),
    ('M100≥5', weak['momentum_100'] >= 5),
    ('M100≥10', weak['momentum_100'] >= 10),
    ('M50≥5', weak['momentum_50'] >= 5),
    ('M50≥10', weak['momentum_50'] >= 10),
]:
    subset = weak[filt]
    if len(subset) < 20:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    
    marker = "⭐" if win_168 >= 55 or win_336 >= 60 else ""
    print(f"{name}: {len(subset)}건")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}% | 336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}% {marker}")

print()
print("=" * 70)
print("복합 조건 - M200 낮아도 다른 지표로 커버")
print("=" * 70)

# M200 < 10% 이지만 다른 조건으로 승률 높이기
low_m200 = result_df[result_df['momentum_200'] < 10]

combos = [
    ('M100≥10 + M50≥10', (low_m200['momentum_100'] >= 10) & (low_m200['momentum_50'] >= 10)),
    ('M100≥10 + HH+HL', (low_m200['momentum_100'] >= 10) & (low_m200['HH'] == True) & (low_m200['HL'] == True)),
    ('M100≥10 + 정배열', (low_m200['momentum_100'] >= 10) & (low_m200['ema_bull'] == True)),
    ('M100≥15', low_m200['momentum_100'] >= 15),
    ('M100≥15 + HH', (low_m200['momentum_100'] >= 15) & (low_m200['HH'] == True)),
    ('M100≥20', low_m200['momentum_100'] >= 20),
    ('정배열 + HH+HL + 위치≥0.7', (low_m200['ema_bull'] == True) & (low_m200['HH'] == True) & (low_m200['HL'] == True) & (low_m200['position_100'] >= 0.7)),
    ('정배열 + HH + M50≥10', (low_m200['ema_bull'] == True) & (low_m200['HH'] == True) & (low_m200['momentum_50'] >= 10)),
]

for name, cond in combos:
    subset = low_m200[cond]
    if len(subset) < 15:
        continue
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    total_336 = subset['long_336h'].sum()
    
    marker = "⭐⭐" if win_168 >= 60 and len(subset) >= 30 else "⭐" if win_168 >= 55 else ""
    print(f"{name}")
    print(f"  {len(subset)}건, 168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}%")
    print(f"        336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}%, 총 {total_336:.0f}% {marker}")

print()
print("=" * 70)
print("전체 통합 - 모든 조건 종합")
print("=" * 70)

# 지금까지 나온 모든 좋은 조건들
all_strategies = [
    # 기존
    ('전략1: M200≥20%', result_df['momentum_200'] >= 20),
    ('전략2: M200≥15% + 정배열', (result_df['momentum_200'] >= 15) & (result_df['ema_bull'] == True)),
    ('전략3: M200≥10% + M100≥10%', (result_df['momentum_200'] >= 10) & (result_df['momentum_100'] >= 10)),
    # 새로 추가
    ('전략5: M100≥20% (M200 무관)', result_df['momentum_100'] >= 20),
    ('전략6: M100≥15% + HH', (result_df['momentum_100'] >= 15) & (result_df['HH'] == True)),
    ('전략7: M100≥15% + 정배열', (result_df['momentum_100'] >= 15) & (result_df['ema_bull'] == True)),
    ('전략8: M100≥10% + M50≥10%', (result_df['momentum_100'] >= 10) & (result_df['momentum_50'] >= 10)),
]

for name, cond in all_strategies:
    subset = result_df[cond]
    
    win_168 = (subset['long_168h'] > 0).mean() * 100
    avg_168 = subset['long_168h'].mean()
    total_168 = subset['long_168h'].sum()
    win_336 = (subset['long_336h'] > 0).mean() * 100
    avg_336 = subset['long_336h'].mean()
    total_336 = subset['long_336h'].sum()
    
    print(f"\n{name}")
    print(f"  건수: {len(subset)}건 (연 {len(subset)/5:.1f}건)")
    print(f"  168h: 승률 {win_168:.1f}%, 평균 {avg_168:.2f}%, 총 {total_168:.0f}%")
    print(f"  336h: 승률 {win_336:.1f}%, 평균 {avg_336:.2f}%, 총 {total_336:.0f}%")

