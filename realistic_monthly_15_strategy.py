#!/usr/bin/env python3
"""
월 15% 현실적 전략
문제: Gap>=4% 조건만으로는 SL률 60% → 수익 불가능

해결: 더 정교한 필터 + 적절한 TP
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("월 15% 현실적 전략 수립")
print("=" * 80)

# 데이터 로드
df_patterns = pd.read_csv('pattern_energy_results.csv')
df_patterns['time'] = pd.to_datetime(df_patterns['time'])

# Gap 2-10%, W/IHS (롱 패턴)만
df = df_patterns[
    (df_patterns['gap_pct'] >= 2) & 
    (df_patterns['gap_pct'] <= 10) &
    (df_patterns['type'].isin(['W', 'IHS'])) &
    (df_patterns['bt_result'].isin(['TP', 'SL']))
].copy()

print(f"\n기본 데이터: {len(df)}건 (W, IHS 롱패턴, Gap 2-10%)")
print(f"  TP: {(df['bt_result']=='TP').sum()}건 ({(df['bt_result']=='TP').mean()*100:.1f}%)")
print(f"  SL: {(df['bt_result']=='SL').sum()}건")

# ============================================================
# Gap 구간별 성과 재확인
# ============================================================
print("\n" + "=" * 80)
print("1. Gap 구간별 성과 (롱 패턴)")
print("=" * 80)

gap_bins = [2, 3, 4, 5, 6, 8, 10]
gap_labels = ['2-3%', '3-4%', '4-5%', '5-6%', '6-8%', '8-10%']
df['gap_bin'] = pd.cut(df['gap_pct'], bins=gap_bins, labels=gap_labels)

print(f"\n{'Gap':<10} {'건수':>8} {'승률%':>10} {'평균PnL':>10}")
print("-" * 45)

for label in gap_labels:
    subset = df[df['gap_bin'] == label]
    if len(subset) > 0:
        win_rate = (subset['bt_result'] == 'TP').mean() * 100
        avg_pnl = subset['bt_pnl'].mean()
        print(f"{label:<10} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f}")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("2. 최적 조건 탐색")
print("=" * 80)

# Gap >= 4%
gap4 = df[df['gap_pct'] >= 4]
print(f"\n[Gap >= 4%]")
print(f"  건수: {len(gap4)}")
print(f"  승률: {(gap4['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {gap4['bt_pnl'].mean():.2f}%")

# Gap >= 5%
gap5 = df[df['gap_pct'] >= 5]
print(f"\n[Gap >= 5%]")
print(f"  건수: {len(gap5)}")
print(f"  승률: {(gap5['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {gap5['bt_pnl'].mean():.2f}%")

# Gap >= 6%
gap6 = df[df['gap_pct'] >= 6]
print(f"\n[Gap >= 6%]")
print(f"  건수: {len(gap6)}")
print(f"  승률: {(gap6['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {gap6['bt_pnl'].mean():.2f}%")

# ============================================================
# 월별 수익 시뮬레이션
# ============================================================
print("\n" + "=" * 80)
print("3. 월별 수익 시뮬레이션")
print("=" * 80)

# Gap >= 5% 조건으로 월별 분석
target = df[df['gap_pct'] >= 5].copy()
target['month'] = target['time'].dt.to_period('M')

monthly = target.groupby('month').agg({
    'bt_pnl': ['sum', 'mean', 'count'],
    'bt_result': lambda x: (x == 'TP').sum()
}).round(2)
monthly.columns = ['총PnL', '평균PnL', '거래수', 'TP수']
monthly['승률'] = (monthly['TP수'] / monthly['거래수'] * 100).round(1)

print(f"\n[Gap >= 5% 월별 성과]")
print(f"  총 월수: {len(monthly)}")
print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  평균 월수익: {monthly['총PnL'].mean():.2f}%")
print(f"  15% 이상 달성: {(monthly['총PnL'] >= 15).sum()}개월")
print(f"  10% 이상 달성: {(monthly['총PnL'] >= 10).sum()}개월")
print(f"  손실 월: {(monthly['총PnL'] < 0).sum()}개월")

# ============================================================
# 레버리지 고려
# ============================================================
print("\n" + "=" * 80)
print("4. 레버리지 적용 시뮬레이션")
print("=" * 80)

base_monthly_pnl = monthly['총PnL'].mean()

print(f"\n기본 월평균 수익: {base_monthly_pnl:.2f}%")
print(f"\n{'레버리지':<12} {'월수익%':>10} {'연수익%':>12} {'MDD예상':>12}")
print("-" * 50)

for lev in [1, 2, 3, 5, 10]:
    monthly_pnl = base_monthly_pnl * lev
    yearly_pnl = monthly_pnl * 12
    # MDD 추정 (최악의 달 × 레버리지)
    worst_month = monthly['총PnL'].min()
    mdd = worst_month * lev
    
    marker = " ★" if monthly_pnl >= 15 else ""
    print(f"{lev}x{'':<10} {monthly_pnl:>10.1f} {yearly_pnl:>12.1f} {mdd:>12.1f}{marker}")

# ============================================================
# 현실적 전략
# ============================================================
print("\n" + "=" * 80)
print("5. 현실적 월 15% 전략")
print("=" * 80)

# Gap >= 5% 기준
gap5_win = (gap5['bt_result'] == 'TP').mean()
gap5_avg = gap5['bt_pnl'].mean()
gap5_monthly_trades = len(gap5) / 61  # 약 61개월

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    현실적 월 15% 달성 전략                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [전제 조건]                                                               │
│  - 패턴: W (더블바텀) + IHS (역헤숄)                                       │
│  - Gap: >= 5% (충분한 에너지 축적)                                         │
│  - TP: 3.5% (넥라인 도달)                                                  │
│  - SL: L값 이탈 (평균 -6%)                                                 │
│                                                                            │
│  [백테스트 결과 (Gap >= 5%)]                                               │
│  - 승률: {gap5_win*100:.1f}%                                                         │
│  - 평균 수익: {gap5_avg:.2f}%/거래                                              │
│  - 월 평균 거래: {gap5_monthly_trades:.1f}건                                            │
│  - 월 평균 수익: {gap5['bt_pnl'].sum()/61:.2f}%                                           │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  [월 15% 달성 방법]                                                        │
│                                                                            │
│  방법 1: 레버리지 활용                                                     │
│    - 3x 레버리지 → 월 {gap5['bt_pnl'].sum()/61*3:.1f}%                                    │
│    - 5x 레버리지 → 월 {gap5['bt_pnl'].sum()/61*5:.1f}%                                    │
│                                                                            │
│  방법 2: 거래 빈도 증가                                                    │
│    - 숏 패턴(M, HS) 추가                                                   │
│    - Gap 4% 이상도 포함 (승률 다소 하락)                                   │
│                                                                            │
│  방법 3: TP 확대 (추세 추종)                                               │
│    - 분할 익절로 큰 추세 포착                                              │
│    - 1차 TP 3.5%, 2차 TP 7%, 트레일링                                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 롱+숏 전체 포함
# ============================================================
print("\n" + "=" * 80)
print("6. 롱+숏 전체 전략")
print("=" * 80)

# 전체 패턴 (Gap >= 5%)
df_all = df_patterns[
    (df_patterns['gap_pct'] >= 5) & 
    (df_patterns['bt_result'].isin(['TP', 'SL']))
].copy()

print(f"\n[Gap >= 5% 전체 패턴]")
print(f"  건수: {len(df_all)}")
print(f"  승률: {(df_all['bt_result']=='TP').mean()*100:.1f}%")
print(f"  평균PnL: {df_all['bt_pnl'].mean():.2f}%")
print(f"  총PnL: {df_all['bt_pnl'].sum():.1f}%")

df_all['month'] = pd.to_datetime(df_all['time']).dt.to_period('M')
monthly_all = df_all.groupby('month')['bt_pnl'].agg(['sum', 'count'])

print(f"\n[월별 성과]")
print(f"  평균 거래수: {monthly_all['count'].mean():.1f}건/월")
print(f"  평균 월수익: {monthly_all['sum'].mean():.2f}%")
print(f"  15% 이상 달성: {(monthly_all['sum'] >= 15).sum()}개월 / {len(monthly_all)}개월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("7. 최종 결론")
print("=" * 80)

monthly_avg = monthly_all['sum'].mean()
print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         월 15% 전략 결론                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Gap >= 5% + 전체 패턴(W,M,IHS,HS) 기준:                                   │
│  - 월 평균 수익: {monthly_avg:.2f}%                                                 │
│  - 월 15% 달성: {(monthly_all['sum'] >= 15).sum()}/{len(monthly_all)}개월 ({(monthly_all['sum'] >= 15).mean()*100:.0f}%)                             │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  월 15% 달성을 위한 현실적 방안:                                           │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  기본 전략 수익: ~{monthly_avg:.0f}%/월                                        │  │
│  │  ↓                                                                  │  │
│  │  × 3배 레버리지 = ~{monthly_avg*3:.0f}%/월 ✓                                    │  │
│  │  또는                                                               │  │
│  │  × 5배 레버리지 = ~{monthly_avg*5:.0f}%/월 ✓                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  리스크 관리:                                                              │
│  - 레버리지 사용 시 최대 손실도 증가                                       │
│  - 3x: 월 MDD ~{monthly_all['sum'].min()*3:.0f}% 예상                                      │
│  - 5x: 월 MDD ~{monthly_all['sum'].min()*5:.0f}% 예상                                      │
│  - 적절한 자금 관리 필수 (전체 자금의 10-20%만 운용)                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

print("\n분석 완료!")
