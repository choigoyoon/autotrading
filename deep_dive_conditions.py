import pandas as pd
import numpy as np

result_df = pd.read_csv('expanded_squeeze_analysis.csv')

print("=" * 70)
print("깊이 파기: 더 정밀한 조건 탐색")
print("=" * 70)
print(f"전체: {len(result_df)}건")
print()

# 홀딩 기간별로도 분석
print("=== 홀딩 기간별 전체 성과 ===")
for period in [24, 48, 72, 168, 336]:
    col = f'long_{period}h'
    win = (result_df[col] > 0).mean() * 100
    avg = result_df[col].mean()
    print(f"{period}h: 승률 {win:.1f}%, 평균 {avg:.2f}%")

print()

# 모멘텀 구간을 더 세분화
print("=" * 70)
print("모멘텀 구간 세분화")
print("=" * 70)

def analyze_condition(df, cond, name, periods=[72, 168, 336]):
    subset = df[cond]
    if len(subset) < 20:
        return None
    
    results = {'name': name, 'count': len(subset)}
    for p in periods:
        col = f'long_{p}h'
        results[f'win_{p}h'] = (subset[col] > 0).mean() * 100
        results[f'avg_{p}h'] = subset[col].mean()
        results[f'total_{p}h'] = subset[col].sum()
    return results

# 모멘텀200 구간별
print("\n[모멘텀200 구간별]")
for low, high in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]:
    cond = (result_df['momentum_200'] >= low) & (result_df['momentum_200'] < high)
    r = analyze_condition(result_df, cond, f"M200: {low}~{high}%")
    if r:
        print(f"M200 {low}~{high}%: {r['count']}건")
        print(f"  72h: 승률 {r['win_72h']:.1f}%, 평균 {r['avg_72h']:.2f}%")
        print(f"  168h: 승률 {r['win_168h']:.1f}%, 평균 {r['avg_168h']:.2f}%")
        print(f"  336h: 승률 {r['win_336h']:.1f}%, 평균 {r['avg_336h']:.2f}%")

# 모멘텀100 구간별  
print("\n[모멘텀100 구간별]")
for low, high in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)]:
    cond = (result_df['momentum_100'] >= low) & (result_df['momentum_100'] < high)
    r = analyze_condition(result_df, cond, f"M100: {low}~{high}%")
    if r:
        print(f"M100 {low}~{high}%: {r['count']}건")
        print(f"  168h: 승률 {r['win_168h']:.1f}%, 평균 {r['avg_168h']:.2f}%")

print()
print("=" * 70)
print("모멘텀 조합 + HH/HL 세분화")
print("=" * 70)

results_all = []

# 더 세밀한 모멘텀 조합
for m200_low in [10, 15, 20, 25, 30]:
    for m100_low in [0, 5, 10]:
        for m50_low in [-10, 0, 5]:
            cond = (result_df['momentum_200'] >= m200_low) & \
                   (result_df['momentum_100'] >= m100_low) & \
                   (result_df['momentum_50'] >= m50_low)
            
            subset = result_df[cond]
            if len(subset) < 20:
                continue
            
            win_168 = (subset['long_168h'] > 0).mean() * 100
            avg_168 = subset['long_168h'].mean()
            win_336 = (subset['long_336h'] > 0).mean() * 100
            avg_336 = subset['long_336h'].mean()
            
            results_all.append({
                'condition': f"M200≥{m200_low} + M100≥{m100_low} + M50≥{m50_low}",
                'count': len(subset),
                'win_168h': win_168,
                'avg_168h': avg_168,
                'win_336h': win_336,
                'avg_336h': avg_336,
                'total_168h': subset['long_168h'].sum(),
                'total_336h': subset['long_336h'].sum()
            })
            
            # + HH
            cond_hh = cond & (result_df['HH'] == True)
            subset_hh = result_df[cond_hh]
            if len(subset_hh) >= 20:
                win_168 = (subset_hh['long_168h'] > 0).mean() * 100
                avg_168 = subset_hh['long_168h'].mean()
                results_all.append({
                    'condition': f"M200≥{m200_low} + M100≥{m100_low} + M50≥{m50_low} + HH",
                    'count': len(subset_hh),
                    'win_168h': win_168,
                    'avg_168h': avg_168,
                    'win_336h': (subset_hh['long_336h'] > 0).mean() * 100,
                    'avg_336h': subset_hh['long_336h'].mean(),
                    'total_168h': subset_hh['long_168h'].sum(),
                    'total_336h': subset_hh['long_336h'].sum()
                })
            
            # + HH + HL
            cond_hhhl = cond & (result_df['HH'] == True) & (result_df['HL'] == True)
            subset_hhhl = result_df[cond_hhhl]
            if len(subset_hhhl) >= 20:
                win_168 = (subset_hhhl['long_168h'] > 0).mean() * 100
                avg_168 = subset_hhhl['long_168h'].mean()
                results_all.append({
                    'condition': f"M200≥{m200_low} + M100≥{m100_low} + M50≥{m50_low} + HH+HL",
                    'count': len(subset_hhhl),
                    'win_168h': win_168,
                    'avg_168h': avg_168,
                    'win_336h': (subset_hhhl['long_336h'] > 0).mean() * 100,
                    'avg_336h': subset_hhhl['long_336h'].mean(),
                    'total_168h': subset_hhhl['long_168h'].sum(),
                    'total_336h': subset_hhhl['long_336h'].sum()
                })

# 가격 위치 추가
for m200_low in [10, 15, 20]:
    for pos_low in [0.5, 0.6, 0.7, 0.8]:
        cond = (result_df['momentum_200'] >= m200_low) & \
               (result_df['position_100'] >= pos_low)
        
        subset = result_df[cond]
        if len(subset) >= 20:
            win_168 = (subset['long_168h'] > 0).mean() * 100
            avg_168 = subset['long_168h'].mean()
            results_all.append({
                'condition': f"M200≥{m200_low} + 위치≥{pos_low}",
                'count': len(subset),
                'win_168h': win_168,
                'avg_168h': avg_168,
                'win_336h': (subset['long_336h'] > 0).mean() * 100,
                'avg_336h': subset['long_336h'].mean(),
                'total_168h': subset['long_168h'].sum(),
                'total_336h': subset['long_336h'].sum()
            })

# EMA 조합
for m200_low in [10, 15, 20]:
    # EMA200 위
    cond = (result_df['momentum_200'] >= m200_low) & (result_df['above_ema200'] == True)
    subset = result_df[cond]
    if len(subset) >= 20:
        results_all.append({
            'condition': f"M200≥{m200_low} + EMA200↑",
            'count': len(subset),
            'win_168h': (subset['long_168h'] > 0).mean() * 100,
            'avg_168h': subset['long_168h'].mean(),
            'win_336h': (subset['long_336h'] > 0).mean() * 100,
            'avg_336h': subset['long_336h'].mean(),
            'total_168h': subset['long_168h'].sum(),
            'total_336h': subset['long_336h'].sum()
        })
    
    # 정배열
    cond = (result_df['momentum_200'] >= m200_low) & (result_df['ema_bull'] == True)
    subset = result_df[cond]
    if len(subset) >= 20:
        results_all.append({
            'condition': f"M200≥{m200_low} + 정배열",
            'count': len(subset),
            'win_168h': (subset['long_168h'] > 0).mean() * 100,
            'avg_168h': subset['long_168h'].mean(),
            'win_336h': (subset['long_336h'] > 0).mean() * 100,
            'avg_336h': subset['long_336h'].mean(),
            'total_168h': subset['long_168h'].sum(),
            'total_336h': subset['long_336h'].sum()
        })

# 결과 정리
results_df = pd.DataFrame(results_all)

# 승률 60% 이상 + 건수 30건 이상
print("\n[168h 기준 - 승률 60%↑, 30건↑]")
filtered = results_df[(results_df['win_168h'] >= 60) & (results_df['count'] >= 30)]
filtered = filtered.sort_values('win_168h', ascending=False)

for _, r in filtered.head(15).iterrows():
    marker = "⭐⭐" if r['count'] >= 80 else "⭐" if r['avg_168h'] >= 4 else ""
    print(f"{r['condition']}")
    print(f"  {r['count']}건, 승률 {r['win_168h']:.1f}%, 평균 {r['avg_168h']:.2f}%, 총 {r['total_168h']:.1f}% {marker}")

# 336h(2주) 기준도 확인
print("\n[336h 기준 - 승률 60%↑, 30건↑]")
filtered_336 = results_df[(results_df['win_336h'] >= 60) & (results_df['count'] >= 30)]
filtered_336 = filtered_336.sort_values('win_336h', ascending=False)

for _, r in filtered_336.head(15).iterrows():
    marker = "⭐⭐" if r['count'] >= 80 else "⭐" if r['avg_336h'] >= 5 else ""
    print(f"{r['condition']}")
    print(f"  {r['count']}건, 승률 {r['win_336h']:.1f}%, 평균 {r['avg_336h']:.2f}%, 총 {r['total_336h']:.1f}% {marker}")

# 총수익 기준 TOP
print("\n[168h 총수익 TOP 10]")
top_total = results_df.sort_values('total_168h', ascending=False).head(10)
for _, r in top_total.iterrows():
    print(f"{r['condition']}")
    print(f"  {r['count']}건, 승률 {r['win_168h']:.1f}%, 평균 {r['avg_168h']:.2f}%, 총 {r['total_168h']:.1f}%")

