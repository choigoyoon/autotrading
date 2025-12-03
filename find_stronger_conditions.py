import pandas as pd
import numpy as np

result_df = pd.read_csv('expanded_squeeze_analysis.csv')

print("=" * 70)
print("더 강한 조건 탐색")
print("=" * 70)
print(f"전체: {len(result_df)}건")
print()

# 현재 최고: 모멘텀200 > 20% + HH = 28건, 67.9%, +6.47%
# 문제: 건수가 너무 적음

# 목표: 건수 100건 이상, 승률 65% 이상, 평균 3% 이상

print("=== 목표: 100건↑, 승률 65%↑, 평균 3%↑ ===\n")

# 다양한 조합 테스트
def test_condition(df, cond, name):
    subset = df[cond]
    if len(subset) < 30:
        return None
    
    win_rate = (subset['long_168h'] > 0).mean() * 100
    avg_pnl = subset['long_168h'].mean()
    total_pnl = subset['long_168h'].sum()
    max_profit = subset['max_profit_168h'].mean()
    
    return {
        'name': name,
        'count': len(subset),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'max_profit': max_profit
    }

results = []

# 모멘텀 기반 조합
for m50 in [0, 5, 10]:
    for m100 in [0, 5, 10, 15]:
        for m200 in [0, 10, 15, 20]:
            cond = (result_df['momentum_50'] > m50) & \
                   (result_df['momentum_100'] > m100) & \
                   (result_df['momentum_200'] > m200)
            name = f"M50>{m50} + M100>{m100} + M200>{m200}"
            r = test_condition(result_df, cond, name)
            if r: results.append(r)

# 모멘텀 + HH/HL 조합
for m100 in [0, 5, 10]:
    for m200 in [0, 10, 15]:
        # HH만
        cond = (result_df['momentum_100'] > m100) & \
               (result_df['momentum_200'] > m200) & \
               (result_df['HH'] == True)
        name = f"M100>{m100} + M200>{m200} + HH"
        r = test_condition(result_df, cond, name)
        if r: results.append(r)
        
        # HH + HL
        cond = (result_df['momentum_100'] > m100) & \
               (result_df['momentum_200'] > m200) & \
               (result_df['HH'] == True) & (result_df['HL'] == True)
        name = f"M100>{m100} + M200>{m200} + HH + HL"
        r = test_condition(result_df, cond, name)
        if r: results.append(r)

# 가격위치 + 모멘텀 조합
for pos in [0.5, 0.6, 0.7]:
    for m100 in [0, 5, 10]:
        cond = (result_df['position_100'] > pos) & \
               (result_df['momentum_100'] > m100)
        name = f"위치>{pos} + M100>{m100}"
        r = test_condition(result_df, cond, name)
        if r: results.append(r)
        
        # + HH
        cond = (result_df['position_100'] > pos) & \
               (result_df['momentum_100'] > m100) & \
               (result_df['HH'] == True)
        name = f"위치>{pos} + M100>{m100} + HH"
        r = test_condition(result_df, cond, name)
        if r: results.append(r)

# EMA200 + 모멘텀 조합
for m100 in [0, 5, 10]:
    for m200 in [0, 10, 15]:
        cond = (result_df['above_ema200'] == True) & \
               (result_df['momentum_100'] > m100) & \
               (result_df['momentum_200'] > m200)
        name = f"EMA200↑ + M100>{m100} + M200>{m200}"
        r = test_condition(result_df, cond, name)
        if r: results.append(r)

# 정배열 + 모멘텀 조합
for m100 in [0, 5, 10]:
    cond = (result_df['ema_bull'] == True) & \
           (result_df['momentum_100'] > m100)
    name = f"정배열 + M100>{m100}"
    r = test_condition(result_df, cond, name)
    if r: results.append(r)
    
    # + HH
    cond = (result_df['ema_bull'] == True) & \
           (result_df['momentum_100'] > m100) & \
           (result_df['HH'] == True)
    name = f"정배열 + M100>{m100} + HH"
    r = test_condition(result_df, cond, name)
    if r: results.append(r)

# 결과 정렬 (승률 기준)
results_df = pd.DataFrame(results)
results_df = results_df.drop_duplicates()

# 승률 60% 이상만 필터
good_results = results_df[results_df['win_rate'] >= 58].sort_values('win_rate', ascending=False)

print("[승률 58% 이상 조건들]")
print()
for _, r in good_results.head(20).iterrows():
    marker = "⭐⭐" if r['count'] >= 100 and r['win_rate'] >= 65 and r['avg_pnl'] >= 3 else \
             "⭐" if r['win_rate'] >= 60 and r['avg_pnl'] >= 2 else ""
    print(f"{r['name']}")
    print(f"  {r['count']:.0f}건, 승률 {r['win_rate']:.1f}%, 평균 {r['avg_pnl']:.2f}%, 총 {r['total_pnl']:.1f}% {marker}")

# 총수익 기준 정렬
print()
print("[총수익 TOP 10]")
top_total = results_df.sort_values('total_pnl', ascending=False).head(10)
for _, r in top_total.iterrows():
    marker = "⭐⭐" if r['count'] >= 100 and r['win_rate'] >= 65 and r['avg_pnl'] >= 3 else \
             "⭐" if r['win_rate'] >= 60 and r['avg_pnl'] >= 2 else ""
    print(f"{r['name']}")
    print(f"  {r['count']:.0f}건, 승률 {r['win_rate']:.1f}%, 평균 {r['avg_pnl']:.2f}%, 총 {r['total_pnl']:.1f}% {marker}")

# 균형 잡힌 조건 (건수 × 평균수익)
print()
print("[건수 × 평균수익 TOP 10]")
results_df['score'] = results_df['count'] * results_df['avg_pnl']
top_score = results_df.sort_values('score', ascending=False).head(10)
for _, r in top_score.iterrows():
    marker = "⭐⭐" if r['count'] >= 100 and r['win_rate'] >= 65 and r['avg_pnl'] >= 3 else \
             "⭐" if r['win_rate'] >= 60 and r['avg_pnl'] >= 2 else ""
    print(f"{r['name']}")
    print(f"  {r['count']:.0f}건, 승률 {r['win_rate']:.1f}%, 평균 {r['avg_pnl']:.2f}%, 총 {r['total_pnl']:.1f}% {marker}")

