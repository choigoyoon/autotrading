"""
추세 변곡 vs 단순 반등 구분 분석

바로 가는 31.9% vs 되돌림 후 68.1%의 차이는?
- 첫 N봉의 움직임
- Volume 차이
- Divergence 여부
- 최종 성과 차이
"""

import pandas as pd
import numpy as np

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv("output_phase1_labeled.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

breakouts_df = pd.read_csv("output_phase4_breakouts.csv")
trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

# 이전 분석 결과 로드
max_analysis = pd.read_csv("output/breakout_max_analysis.csv")

print(f"추세선 돌파: {len(trendline_breakouts):,}개")
print(f"분석 데이터: {len(max_analysis):,}개\n")

# 두 그룹 분리
group_direct = max_analysis[max_analysis['profit_first'] == True]  # 바로 가는 31.9%
group_pullback = max_analysis[max_analysis['profit_first'] == False]  # 되돌림 후 68.1%

print("=" * 60)
print("그룹 분리")
print("=" * 60)
print(f"바로 가는 그룹: {len(group_direct):,}개 ({len(group_direct)/len(max_analysis)*100:.1f}%)")
print(f"되돌림 후 그룹: {len(group_pullback):,}개 ({len(group_pullback)/len(max_analysis)*100:.1f}%)")

# 각 그룹의 첫 N봉 움직임 분석
def analyze_first_n_bars(breakout_indices, n_bars=[1, 3, 5, 10]):
    """첫 N봉의 움직임 분석"""
    results = {n: [] for n in n_bars}

    for idx in breakout_indices:
        break_idx = int(idx)
        break_price = df.iloc[break_idx]['close']

        for n in n_bars:
            if break_idx + n < len(df):
                future_price = df.iloc[break_idx + n]['close']
                ret = (future_price - break_price) / break_price * 100
                results[n].append(ret)
            else:
                results[n].append(np.nan)

    return {n: pd.Series(vals).mean() for n, vals in results.items()}

print("\n" + "=" * 60)
print("첫 N봉 움직임 비교")
print("=" * 60)

direct_first_n = analyze_first_n_bars(group_direct['break_idx'])
pullback_first_n = analyze_first_n_bars(group_pullback['break_idx'])

print("\n바로 가는 그룹 (31.9%):")
for n, ret in direct_first_n.items():
    print(f"  첫 {n:>2}봉: {ret:>6.3f}%")

print("\n되돌림 후 그룹 (68.1%):")
for n, ret in pullback_first_n.items():
    print(f"  첫 {n:>2}봉: {ret:>6.3f}%")

print("\n차이:")
for n in direct_first_n.keys():
    diff = direct_first_n[n] - pullback_first_n[n]
    print(f"  첫 {n:>2}봉: {diff:>+6.3f}%")

# 최종 성과 비교
print("\n" + "=" * 60)
print("최종 성과 비교")
print("=" * 60)

print("\n[바로 가는 그룹]")
print(f"  평균 최대 반등: {group_direct['max_profit'].mean():.3f}%")
print(f"  평균 최대 하락: {group_direct['max_drawdown'].mean():.3f}%")
print(f"  평균 최종 수익: {group_direct['final_return'].mean():.3f}%")
print(f"  반등 도달 시점: {group_direct['max_profit_bar'].mean():.1f}봉")
print(f"  하락 도달 시점: {group_direct['max_dd_bar'].mean():.1f}봉")

print("\n[되돌림 후 그룹]")
print(f"  평균 최대 반등: {group_pullback['max_profit'].mean():.3f}%")
print(f"  평균 최대 하락: {group_pullback['max_drawdown'].mean():.3f}%")
print(f"  평균 최종 수익: {group_pullback['final_return'].mean():.3f}%")
print(f"  반등 도달 시점: {group_pullback['max_profit_bar'].mean():.1f}봉")
print(f"  하락 도달 시점: {group_pullback['max_dd_bar'].mean():.1f}봉")

# 첫 5봉 기준 구분 가능한가?
print("\n" + "=" * 60)
print("구분 가능성 분석")
print("=" * 60)

# 첫 5봉에서 양수인 비율
direct_5bar_positive = sum(analyze_first_n_bars(group_direct['break_idx'], [5])[5] > 0 for _ in [1])
pullback_5bar_positive = sum(analyze_first_n_bars(group_pullback['break_idx'], [5])[5] > 0 for _ in [1])

# 실제 계산
direct_returns_5 = []
pullback_returns_5 = []

for idx in group_direct['break_idx']:
    break_idx = int(idx)
    if break_idx + 5 < len(df):
        break_price = df.iloc[break_idx]['close']
        future_price = df.iloc[break_idx + 5]['close']
        ret = (future_price - break_price) / break_price * 100
        direct_returns_5.append(ret)

for idx in group_pullback['break_idx']:
    break_idx = int(idx)
    if break_idx + 5 < len(df):
        break_price = df.iloc[break_idx]['close']
        future_price = df.iloc[break_idx + 5]['close']
        ret = (future_price - break_price) / break_price * 100
        pullback_returns_5.append(ret)

direct_positive_5 = sum(1 for r in direct_returns_5 if r > 0)
pullback_positive_5 = sum(1 for r in pullback_returns_5 if r > 0)

print(f"\n첫 5봉에서 양수 비율:")
print(f"  바로 가는 그룹: {direct_positive_5}/{len(direct_returns_5)} ({direct_positive_5/len(direct_returns_5)*100:.1f}%)")
print(f"  되돌림 후 그룹: {pullback_positive_5}/{len(pullback_returns_5)} ({pullback_positive_5/len(pullback_returns_5)*100:.1f}%)")

# 첫 5봉 > 0.2% 기준
direct_strong_5 = sum(1 for r in direct_returns_5 if r > 0.2)
pullback_strong_5 = sum(1 for r in pullback_returns_5 if r > 0.2)

print(f"\n첫 5봉에서 >0.2% 비율:")
print(f"  바로 가는 그룹: {direct_strong_5}/{len(direct_returns_5)} ({direct_strong_5/len(direct_returns_5)*100:.1f}%)")
print(f"  되돌림 후 그룹: {pullback_strong_5}/{len(pullback_returns_5)} ({pullback_strong_5/len(pullback_returns_5)*100:.1f}%)")

# Volume 분석 (첫 3봉 평균)
print("\n" + "=" * 60)
print("거래량 분석")
print("=" * 60)

direct_volumes = []
pullback_volumes = []

for idx in group_direct['break_idx']:
    break_idx = int(idx)
    if break_idx + 3 < len(df) and break_idx >= 20:
        # 첫 3봉 평균 거래량
        first_3_vol = df.iloc[break_idx:break_idx+3]['volume'].mean()
        # 이전 20봉 평균 거래량
        prev_20_vol = df.iloc[break_idx-20:break_idx]['volume'].mean()
        if prev_20_vol > 0:
            vol_ratio = first_3_vol / prev_20_vol
            direct_volumes.append(vol_ratio)

for idx in group_pullback['break_idx']:
    break_idx = int(idx)
    if break_idx + 3 < len(df) and break_idx >= 20:
        first_3_vol = df.iloc[break_idx:break_idx+3]['volume'].mean()
        prev_20_vol = df.iloc[break_idx-20:break_idx]['volume'].mean()
        if prev_20_vol > 0:
            vol_ratio = first_3_vol / prev_20_vol
            pullback_volumes.append(vol_ratio)

print(f"\n첫 3봉 거래량 비율 (vs 이전 20봉):")
print(f"  바로 가는 그룹: {np.mean(direct_volumes):.2f}x")
print(f"  되돌림 후 그룹: {np.mean(pullback_volumes):.2f}x")
print(f"  차이: {np.mean(direct_volumes) - np.mean(pullback_volumes):+.2f}x")

# 되돌림 후 그룹의 최종 성과
print("\n" + "=" * 60)
print("💡 핵심 인사이트")
print("=" * 60)

# 되돌림 크기별 최종 수익
pullback_by_dd = group_pullback.copy()
pullback_by_dd['dd_size'] = pullback_by_dd['max_drawdown'].abs()

dd_bins = [0, 0.5, 1.0, 1.5, 2.0, 10]
dd_labels = ['0-0.5%', '0.5-1%', '1-1.5%', '1.5-2%', '2%+']
pullback_by_dd['dd_bucket'] = pd.cut(pullback_by_dd['dd_size'], bins=dd_bins, labels=dd_labels)

print("\n[되돌림 크기별 최종 수익]")
for bucket in dd_labels:
    bucket_data = pullback_by_dd[pullback_by_dd['dd_bucket'] == bucket]
    if len(bucket_data) > 0:
        avg_final = bucket_data['final_return'].mean()
        avg_max = bucket_data['max_profit'].mean()
        count = len(bucket_data)
        print(f"  되돌림 {bucket}: {count:>4}개 | 최종 {avg_final:>6.3f}% | 최대 {avg_max:>6.3f}%")

# 결론
print("\n" + "=" * 60)
print("결론")
print("=" * 60)

if direct_first_n[5] > 0.3 and pullback_first_n[5] < 0.1:
    print("\n✅ 첫 5봉으로 구분 가능!")
    print(f"   바로 가는 그룹: 첫 5봉 {direct_first_n[5]:.3f}%")
    print(f"   되돌림 후 그룹: 첫 5봉 {pullback_first_n[5]:.3f}%")
    print(f"   → 첫 5봉 >0.2% 면 바로 가는 그룹!")
else:
    print("\n⚠️  첫 5봉으로 구분 어려움")
    print(f"   바로 가는 그룹: 첫 5봉 {direct_first_n[5]:.3f}%")
    print(f"   되돌림 후 그룹: 첫 5봉 {pullback_first_n[5]:.3f}%")

if group_direct['final_return'].mean() > group_pullback['final_return'].mean() * 1.3:
    print("\n✅ 바로 가는 그룹이 훨씬 우수!")
    print(f"   바로: {group_direct['final_return'].mean():.3f}%")
    print(f"   되돌림: {group_pullback['final_return'].mean():.3f}%")
    print("   → 바로 가는 케이스만 거래해야!")
else:
    print("\n⚠️  최종 수익은 비슷함")
    print(f"   바로: {group_direct['final_return'].mean():.3f}%")
    print(f"   되돌림: {group_pullback['final_return'].mean():.3f}%")
    print("   → 되돌림 후에도 결국 비슷하게 수익")

# 추세 변곡 판단
print("\n" + "=" * 60)
print("추세 변곡인가?")
print("=" * 60)

direct_pct = len(group_direct) / len(max_analysis) * 100

if direct_pct > 60:
    print(f"\n✅ 추세 변곡 맞음! (바로 {direct_pct:.1f}%)")
else:
    print(f"\n❌ 추세 변곡 아님! (바로 {direct_pct:.1f}% 뿐)")
    print(f"   되돌림 후: {100-direct_pct:.1f}%")
    print("\n   추세선 돌파 = 추세 변곡 아니라")
    print("   추세선 돌파 = 지지/저항 테스트 시작")
    print("                   ↓")
    print("                 리테스트 (68%)")
    print("                   ↓")
    print("              지지 확인 후 상승")

print("\n저장 중...")
comparison = pd.DataFrame({
    'metric': ['count', 'first_1bar', 'first_3bar', 'first_5bar', 'first_10bar',
               'max_profit', 'max_drawdown', 'final_return',
               'profit_time', 'dd_time', 'volume_ratio'],
    'direct': [len(group_direct), direct_first_n[1], direct_first_n[3], direct_first_n[5], direct_first_n[10],
               group_direct['max_profit'].mean(), group_direct['max_drawdown'].mean(),
               group_direct['final_return'].mean(),
               group_direct['max_profit_bar'].mean(), group_direct['max_dd_bar'].mean(),
               np.mean(direct_volumes)],
    'pullback': [len(group_pullback), pullback_first_n[1], pullback_first_n[3], pullback_first_n[5], pullback_first_n[10],
                 group_pullback['max_profit'].mean(), group_pullback['max_drawdown'].mean(),
                 group_pullback['final_return'].mean(),
                 group_pullback['max_profit_bar'].mean(), group_pullback['max_dd_bar'].mean(),
                 np.mean(pullback_volumes)]
})

comparison.to_csv("output/pattern_comparison.csv", index=False)
print("저장: output/pattern_comparison.csv")
