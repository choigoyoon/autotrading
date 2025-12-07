import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('trades_with_indicators.csv')
df['optimal_pnl'] = df.apply(
    lambda x: x['final_pnl'] if x['direction'] == 'LONG' else -x['final_pnl'], 
    axis=1
)
print(f"총 매매: {len(df)}건")

# 승률 70% 이상 조건 찾기
print("\n" + "="*80)
print("🎯 승률 70% 이상 조건 찾기 (최소 20건)")
print("="*80)

results = []

# 모든 파라미터 조합 탐색
for ema200 in [True, False, None]:
    for ema50 in [True, False, None]:
        for vwap in [True, False, None]:
            for bull in [True, False, None]:  # 정배열/역배열
                for sq_range in [(3, 10), (10, 20), (20, 50), None]:
                    
                    sub = df.copy()
                    conditions = []
                    
                    if ema200 is not None:
                        sub = sub[sub['above_ema200'] == ema200]
                        conditions.append(f"EMA200{'↑' if ema200 else '↓'}")
                    
                    if ema50 is not None:
                        sub = sub[sub['above_ema50'] == ema50]
                        conditions.append(f"EMA50{'↑' if ema50 else '↓'}")
                    
                    if vwap is not None:
                        sub = sub[sub['above_vwap'] == vwap]
                        conditions.append(f"VWAP{'↑' if vwap else '↓'}")
                    
                    if bull is True:
                        sub = sub[sub['ema_bullish'] == True]
                        conditions.append("정배열")
                    elif bull is False:
                        sub = sub[sub['ema_bearish'] == True]
                        conditions.append("역배열")
                    
                    if sq_range is not None:
                        sub = sub[(sub['squeeze_length'] >= sq_range[0]) & 
                                  (sub['squeeze_length'] < sq_range[1])]
                        conditions.append(f"수축{sq_range[0]}-{sq_range[1]}h")
                    
                    if len(sub) >= 20 and len(conditions) >= 2:
                        wr = (sub['optimal_pnl'] > 0).mean() * 100
                        avg = sub['optimal_pnl'].mean()
                        
                        if wr >= 60:  # 60% 이상만 저장
                            results.append({
                                'conditions': '+'.join(conditions),
                                'count': len(sub),
                                'win_rate': wr,
                                'avg_pnl': avg,
                                'total_pnl': sub['optimal_pnl'].sum()
                            })

results_df = pd.DataFrame(results)
results_df = results_df.drop_duplicates(subset=['conditions'])
results_df = results_df.sort_values('win_rate', ascending=False)

print("\n### 승률 TOP 30:")
for i, row in results_df.head(30).iterrows():
    marker = "⭐" if row['win_rate'] >= 70 else ""
    print(f"  {row['conditions']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}% {marker}")

# 70% 이상만
print("\n" + "="*80)
print("⭐ 승률 70% 이상 조건")
print("="*80)
high_wr = results_df[results_df['win_rate'] >= 70]
if len(high_wr) > 0:
    for i, row in high_wr.iterrows():
        print(f"  {row['conditions']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}%, 총 {row['total_pnl']:.1f}%")
else:
    print("  70% 이상 조건 없음")

# 65% 이상 + 평균 1% 이상
print("\n### 승률 65%+ AND 평균 0.8%+:")
good = results_df[(results_df['win_rate'] >= 65) & (results_df['avg_pnl'] >= 0.8)]
for i, row in good.iterrows():
    print(f"  {row['conditions']}: {row['count']}건, 승률 {row['win_rate']:.1f}%, 평균 {row['avg_pnl']:.2f}%")
