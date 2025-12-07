import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('expanded_squeeze_analysis.csv')

print("="*80)
print("SHORT 전략 탐색 - BB 수축→확장 시점 기반")
print("="*80)

# SHORT 방향으로 성과 분석
df['short_168h'] = -df['long_168h']  # SHORT은 가격 하락이 수익
df['short_336h'] = -df['long_336h']

print(f"\n총 분석 케이스: {len(df)}건")

# 1. 전체 SHORT 성과
print("\n" + "="*80)
print("1. 전체 SHORT 무조건 진입 시")
print("="*80)
win_168 = (df['short_168h'] > 0).sum()
win_336 = (df['short_336h'] > 0).sum()
print(f"168h: 승률 {win_168/len(df)*100:.1f}%, 평균 {df['short_168h'].mean():.2f}%")
print(f"336h: 승률 {win_336/len(df)*100:.1f}%, 평균 {df['short_336h'].mean():.2f}%")

# 2. 모멘텀 음수 구간
print("\n" + "="*80)
print("2. 모멘텀 음수 구간 SHORT")
print("="*80)

momentum_ranges = [
    ('M200 < -20%', df['momentum_200'] < -20),
    ('M200 < -15%', df['momentum_200'] < -15),
    ('M200 < -10%', df['momentum_200'] < -10),
    ('M100 < -20%', df['momentum_100'] < -20),
    ('M100 < -15%', df['momentum_100'] < -15),
    ('M100 < -10%', df['momentum_100'] < -10),
]

results = []
for name, cond in momentum_ranges:
    subset = df[cond]
    if len(subset) > 0:
        win_168 = (subset['short_168h'] > 0).sum()
        win_336 = (subset['short_336h'] > 0).sum()
        results.append({
            '조건': name,
            '건수': len(subset),
            '168h_승률': f"{win_168/len(subset)*100:.1f}%",
            '168h_평균': f"{subset['short_168h'].mean():.2f}%",
            '336h_승률': f"{win_336/len(subset)*100:.1f}%",
            '336h_평균': f"{subset['short_336h'].mean():.2f}%",
            '총수익': f"{subset['short_336h'].sum():.1f}%"
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 3. 역배열 조건
print("\n" + "="*80)
print("3. EMA 역배열 (하락추세) + SHORT")
print("="*80)

# 역배열 조건
reverse_conds = [
    ('역배열', df['ema_bear'] == True),
    ('역배열 + M100<0', (df['ema_bear'] == True) & (df['momentum_100'] < 0)),
    ('역배열 + M200<0', (df['ema_bear'] == True) & (df['momentum_200'] < 0)),
    ('역배열 + M100<-10', (df['ema_bear'] == True) & (df['momentum_100'] < -10)),
]

results = []
for name, cond in reverse_conds:
    subset = df[cond]
    if len(subset) > 0:
        win_168 = (subset['short_168h'] > 0).sum()
        win_336 = (subset['short_336h'] > 0).sum()
        results.append({
            '조건': name,
            '건수': len(subset),
            '168h_승률': f"{win_168/len(subset)*100:.1f}%",
            '168h_평균': f"{subset['short_168h'].mean():.2f}%",
            '336h_승률': f"{win_336/len(subset)*100:.1f}%",
            '336h_평균': f"{subset['short_336h'].mean():.2f}%",
            '총수익': f"{subset['short_336h'].sum():.1f}%"
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 4. 복합 조건
print("\n" + "="*80)
print("4. 복합 조건 SHORT 전략")
print("="*80)

complex_conds = [
    ('M100<-15 + 역배열', (df['momentum_100'] < -15) & (df['ema_bear'] == True)),
    ('M100<-15 + LL', (df['momentum_100'] < -15) & (df['LL'] == True)),
    ('M100<-20', df['momentum_100'] < -20),
    ('M200<-20 + 역배열', (df['momentum_200'] < -20) & (df['ema_bear'] == True)),
    ('M200<-15 + M100<-10', (df['momentum_200'] < -15) & (df['momentum_100'] < -10)),
    ('M200<-10 + M100<-10 + 역배열', 
     (df['momentum_200'] < -10) & (df['momentum_100'] < -10) & (df['ema_bear'] == True)),
]

results = []
for name, cond in complex_conds:
    subset = df[cond]
    if len(subset) >= 5:  # 최소 5건 이상
        win_168 = (subset['short_168h'] > 0).sum()
        win_336 = (subset['short_336h'] > 0).sum()
        results.append({
            '조건': name,
            '건수': len(subset),
            '연간': f"{len(subset)/5:.1f}",
            '168h_승률': f"{win_168/len(subset)*100:.1f}%",
            '168h_평균': f"{subset['short_168h'].mean():.2f}%",
            '336h_승률': f"{win_336/len(subset)*100:.1f}%",
            '336h_평균': f"{subset['short_336h'].mean():.2f}%",
            '총수익': f"{subset['short_336h'].sum():.1f}%"
        })

if results:
    results_df = pd.DataFrame(results)
    # 336h 평균 수익률로 정렬
    results_df['sort_key'] = results_df['336h_평균'].str.rstrip('%').astype(float)
    results_df = results_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    print(results_df.to_string(index=False))
else:
    print("조건 충족하는 케이스 없음")

# 5. 최종 후보 전략
print("\n" + "="*80)
print("5. SHORT 전략 최종 후보 (승률 55%+ 또는 평균 3%+)")
print("="*80)

all_conditions = [
    ('M100 < -20%', df['momentum_100'] < -20),
    ('M100 < -15% + 역배열', (df['momentum_100'] < -15) & (df['ema_bear'] == True)),
    ('M100 < -15% + LL', (df['momentum_100'] < -15) & (df['LL'] == True)),
    ('M200 < -20%', df['momentum_200'] < -20),
    ('M200 < -15% + M100 < -10%', (df['momentum_200'] < -15) & (df['momentum_100'] < -10)),
]

final_results = []
for name, cond in all_conditions:
    subset = df[cond]
    if len(subset) >= 3:  # 최소 3건 이상 (SHORT은 기회 적음)
        win_168 = (subset['short_168h'] > 0).sum()
        win_336 = (subset['short_336h'] > 0).sum()
        avg_336 = subset['short_336h'].mean()
        wr_336 = win_336/len(subset)*100
        
        # 승률 50% 이상 또는 평균 2% 이상 (SHORT은 기준 낮춤)
        if wr_336 >= 50 or avg_336 >= 2:
            final_results.append({
                '조건': name,
                '건수': len(subset),
                '연간': f"{len(subset)/5:.1f}",
                '168h_승률': f"{win_168/len(subset)*100:.1f}%",
                '168h_평균': f"{subset['short_168h'].mean():.2f}%",
                '336h_승률': f"{wr_336:.1f}%",
                '336h_평균': f"{avg_336:.2f}%",
                '총수익': f"{subset['short_336h'].sum():.1f}%"
            })

if final_results:
    final_df = pd.DataFrame(final_results)
    final_df['sort_key'] = final_df['336h_평균'].str.rstrip('%').astype(float)
    final_df = final_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
    print(final_df.to_string(index=False))
    
    # CSV 저장
    final_df.to_csv('short_strategies_final.csv', index=False, encoding='utf-8-sig')
    print("\n✅ short_strategies_final.csv 저장 완료")
else:
    print("⚠️ 조건 충족하는 SHORT 전략 없음")
    print("   - 승률 50% 이상 OR 평균 수익 2% 이상인 조건 없음")

# 6. 상세 분석 - 왜 SHORT이 안 되는가?
print("\n" + "="*80)
print("6. SHORT 실패 원인 분석")
print("="*80)

# 모멘텀 음수 구간에서도 가격이 상승하는 경우
m200_neg = df[df['momentum_200'] < -10]
if len(m200_neg) > 0:
    print(f"\nM200 < -10% 구간 ({len(m200_neg)}건):")
    print(f"  168h 후 가격 상승: {(m200_neg['long_168h'] > 0).sum()}건 ({(m200_neg['long_168h'] > 0).sum()/len(m200_neg)*100:.1f}%)")
    print(f"  336h 후 가격 상승: {(m200_neg['long_336h'] > 0).sum()}건 ({(m200_neg['long_336h'] > 0).sum()/len(m200_neg)*100:.1f}%)")
    print(f"  → SHORT 평균 수익: 168h {m200_neg['short_168h'].mean():.2f}%, 336h {m200_neg['short_336h'].mean():.2f}%")

print("\n" + "="*80)
print("분석 완료")
print("="*80)
