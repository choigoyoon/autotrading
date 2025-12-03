"""
사용자 매매법 최적화 분석

이전 결과에서 좋았던 조건들을 더 자세히 분석
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("사용자 매매법 최적화 분석")
print("=" * 80)

# 결과 로드
results_df = pd.read_csv('user_method_trend_first_results.csv', parse_dates=['entry_time', 'exit_time'])

print(f"총 결과: {len(results_df)}개")

# =============================================================================
# 1. Long 최적 조건 찾기
# =============================================================================
print("\n" + "=" * 80)
print("1. Long 최적 조건 분석")
print("=" * 80)

long_df = results_df[results_df['direction'] == 'long'].copy()
print(f"Long 거래: {len(long_df)}개")

# Gap 구간 세분화
print("\n[Gap 구간별 성과]")
gap_results = []
for gap_min in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    for gap_max in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if gap_max <= gap_min:
            continue
        
        subset = long_df[(long_df['gap_pct'] >= gap_min) & (long_df['gap_pct'] < gap_max)]
        if len(subset) >= 10:
            cum = subset.sort_values('entry_time')['pnl'].cumsum()
            mdd = (cum - cum.cummax()).min()
            mdd_ratio = subset['pnl'].sum() / abs(mdd) if mdd < 0 else 0
            
            gap_results.append({
                'condition': f'Gap {gap_min}-{gap_max}%',
                'trades': len(subset),
                'win_rate': (subset['pnl'] > 0).mean() * 100,
                'total_pnl': subset['pnl'].sum(),
                'avg_pnl': subset['pnl'].mean(),
                'mdd': mdd,
                'mdd_ratio': mdd_ratio
            })

gap_df = pd.DataFrame(gap_results).sort_values('mdd_ratio', ascending=False)
print("\n상위 10개 Gap 조건 (MDD Ratio 기준):")
for i, row in gap_df.head(10).iterrows():
    print(f"  {row['condition']}: {row['trades']}회, 승률 {row['win_rate']:.1f}%, "
          f"PnL {row['total_pnl']:.1f}%, MDD {row['mdd']:.1f}%, MDDRatio {row['mdd_ratio']:.1f}x")

# =============================================================================
# 2. Short 최적 조건 찾기
# =============================================================================
print("\n" + "=" * 80)
print("2. Short 최적 조건 분석")
print("=" * 80)

short_df = results_df[results_df['direction'] == 'short'].copy()
print(f"Short 거래: {len(short_df)}개")

# Gap 구간 분석
print("\n[Gap 구간별 성과]")
gap_results_short = []
for gap_min in [0, 0.1, 0.2, 0.3, 0.5, 0.7]:
    for gap_max in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        if gap_max <= gap_min:
            continue
        
        subset = short_df[(short_df['gap_pct'] >= gap_min) & (short_df['gap_pct'] < gap_max)]
        if len(subset) >= 10:
            cum = subset.sort_values('entry_time')['pnl'].cumsum()
            mdd = (cum - cum.cummax()).min()
            mdd_ratio = subset['pnl'].sum() / abs(mdd) if mdd < 0 else 0
            
            gap_results_short.append({
                'condition': f'Gap {gap_min}-{gap_max}%',
                'trades': len(subset),
                'win_rate': (subset['pnl'] > 0).mean() * 100,
                'total_pnl': subset['pnl'].sum(),
                'avg_pnl': subset['pnl'].mean(),
                'mdd': mdd,
                'mdd_ratio': mdd_ratio
            })

if gap_results_short:
    gap_df_short = pd.DataFrame(gap_results_short).sort_values('mdd_ratio', ascending=False)
    print("\n상위 10개 Gap 조건 (MDD Ratio 기준):")
    for i, row in gap_df_short.head(10).iterrows():
        print(f"  {row['condition']}: {row['trades']}회, 승률 {row['win_rate']:.1f}%, "
              f"PnL {row['total_pnl']:.1f}%, MDD {row['mdd']:.1f}%, MDDRatio {row['mdd_ratio']:.1f}x")

# =============================================================================
# 3. 청산 방식 최적화
# =============================================================================
print("\n" + "=" * 80)
print("3. 청산 방식 분석")
print("=" * 80)

print("\n[청산 이유별 분석]")
for direction in ['long', 'short']:
    print(f"\n{direction.upper()}:")
    sub = results_df[results_df['direction'] == direction]
    for reason in ['TRAIL', 'TIME', 'SL']:
        rsub = sub[sub['exit_reason'] == reason]
        if len(rsub) > 0:
            print(f"  {reason}: {len(rsub)}회, 승률 {(rsub['pnl']>0).mean()*100:.1f}%, "
                  f"평균 PnL {rsub['pnl'].mean():.2f}%")

# =============================================================================
# 4. 최적 조합 추천
# =============================================================================
print("\n" + "=" * 80)
print("4. 최적 전략 추천")
print("=" * 80)

# 최적 Long 조건: Gap 0.2-0.7%
best_long = long_df[(long_df['gap_pct'] >= 0.2) & (long_df['gap_pct'] < 0.7)]
# 최적 Short 조건: 전체 또는 Gap 0.1-0.5%
best_short = short_df[(short_df['gap_pct'] >= 0.1) & (short_df['gap_pct'] < 0.7)]

print("\n[최적 Long 조건: Gap 0.2-0.7%]")
if len(best_long) >= 5:
    cum = best_long.sort_values('entry_time')['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    print(f"  거래: {len(best_long)}회")
    print(f"  승률: {(best_long['pnl']>0).mean()*100:.1f}%")
    print(f"  총 PnL: {best_long['pnl'].sum():.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  MDD Ratio: {best_long['pnl'].sum()/abs(mdd):.1f}x")

print("\n[최적 Short 조건: Gap 0.1-0.7%]")
if len(best_short) >= 5:
    cum = best_short.sort_values('entry_time')['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    print(f"  거래: {len(best_short)}회")
    print(f"  승률: {(best_short['pnl']>0).mean()*100:.1f}%")
    print(f"  총 PnL: {best_short['pnl'].sum():.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  MDD Ratio: {best_short['pnl'].sum()/abs(mdd):.1f}x")

# 최적 조합 통합
print("\n[최적 조합 통합: Long Gap 0.2-0.7% + Short Gap 0.1-0.7%]")
combined = pd.concat([best_long, best_short]).sort_values('entry_time')
if len(combined) > 0:
    cum = combined['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    years = (combined['entry_time'].max() - combined['entry_time'].min()).days / 365
    print(f"  기간: {years:.1f}년")
    print(f"  총 거래: {len(combined)}회 (연간 {len(combined)/years:.0f}회)")
    print(f"  Long: {(combined['direction']=='long').sum()}회, Short: {(combined['direction']=='short').sum()}회")
    print(f"  승률: {(combined['pnl']>0).mean()*100:.1f}%")
    print(f"  총 PnL: {combined['pnl'].sum():.1f}%")
    print(f"  연평균 PnL: {combined['pnl'].sum()/years:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  MDD Ratio: {combined['pnl'].sum()/abs(mdd):.1f}x")

# =============================================================================
# 5. 연도별 성과
# =============================================================================
print("\n" + "=" * 80)
print("5. 연도별 성과")
print("=" * 80)

results_df['year'] = pd.to_datetime(results_df['entry_time']).dt.year

print("\n[전체 전략 연도별]")
for year in sorted(results_df['year'].unique()):
    yr = results_df[results_df['year'] == year]
    cum = yr.sort_values('entry_time')['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    print(f"  {year}: {len(yr)}회, 승률 {(yr['pnl']>0).mean()*100:.1f}%, "
          f"PnL {yr['pnl'].sum():.1f}%, MDD {mdd:.1f}%")

print("\n[최적 조합 연도별]")
combined['year'] = pd.to_datetime(combined['entry_time']).dt.year
for year in sorted(combined['year'].unique()):
    yr = combined[combined['year'] == year]
    cum = yr.sort_values('entry_time')['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    print(f"  {year}: {len(yr)}회, 승률 {(yr['pnl']>0).mean()*100:.1f}%, "
          f"PnL {yr['pnl'].sum():.1f}%, MDD {mdd:.1f}%")

# =============================================================================
# 결과 저장
# =============================================================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

combined.to_csv('user_method_optimal_results.csv', index=False)
print(f"최적 조합 결과 저장: user_method_optimal_results.csv ({len(combined)}개)")

# 최종 요약
print(f"""
╔══════════════════════════════════════════════════════════════════╗
║             사용자 매매법 최적화 최종 결과                           ║
╠══════════════════════════════════════════════════════════════════╣
║  전략: 추세 확인 후 추세선 돌파/이탈 진입                            ║
║  - Long: 하락추세(LH+LL) 내 추세선 돌파, Gap 0.2-0.7%               ║
║  - Short: 상승추세(HH+HL) 내 추세선 이탈, Gap 0.1-0.7%              ║
║  - 청산: 동적 트레일링 (2% 후 50% 되돌림) / 24시간 시간제한          ║
╠══════════════════════════════════════════════════════════════════╣
║  총 거래: {len(combined):,}회 (연간 {len(combined)/years:.0f}회)                              ║
║  승률: {(combined['pnl']>0).mean()*100:.1f}%                                                 ║
║  총 PnL: {combined['pnl'].sum():.1f}%                                               ║
║  연평균 PnL: {combined['pnl'].sum()/years:.1f}%                                           ║
║  MDD: {mdd:.1f}%                                                         ║
║  MDD Ratio: {combined['pnl'].sum()/abs(mdd):.1f}x                                          ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("\n완료!")
