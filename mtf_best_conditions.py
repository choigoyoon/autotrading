#!/usr/bin/env python3
"""
MTF 최적 조건 도출

핵심 발견:
1. Entry Gap 2.5-3%: 43.9% 승률, 0.90% 평균 PnL
2. Entry Gap 3-4%: 45.3% 승률, 0.88% 평균 PnL
3. Gap >= 4%: 30.1% 승률, 0.44% 평균 PnL

최적 조건: Entry Gap 2~4% + Gap >= 4%
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 최적 조건 분석")
print("=" * 80)

# 이전 결과 로드
df1 = pd.read_csv('mtf_optimized_long_short_results.csv')
df2 = pd.read_csv('mtf_combined_final_results.csv')

print(f"\n1. mtf_optimized_long_short: {len(df1)}건")
print(f"2. mtf_combined_final: {len(df2)}건")

# ============================================================
# 최적 조건 분석
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 분석 (mtf_optimized_long_short)")
print("=" * 80)

# Entry Gap 세분화
print("\nEntry Gap 세분화:")
print(f"{'Entry Gap':>15} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 60)

for low, high in [(0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]:
    subset = df1[(df1['entry_gap'] >= low) & (df1['entry_gap'] < high)]
    if len(subset) > 5:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f}")

# Gap 세분화
print("\nGap 세분화:")
print(f"{'Gap':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 55)

for low, high in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 100)]:
    subset = df1[(df1['gap'] >= low) & (df1['gap'] < high)]
    if len(subset) > 5:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>10} {len(subset):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f}")

# 최적 조건 조합
print("\n" + "=" * 80)
print("최적 조건 조합")
print("=" * 80)

print(f"\n{'조건':>50} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 95)

conditions = [
    ('전체', df1),
    ('Entry Gap 2-4%', df1[(df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('Entry Gap 2.5-4%', df1[(df1['entry_gap'] >= 2.5) & (df1['entry_gap'] < 4)]),
    ('Entry Gap 3-4%', df1[(df1['entry_gap'] >= 3) & (df1['entry_gap'] < 4)]),
    ('Gap>=4% + Entry 2-4%', df1[(df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('Gap>=4% + Entry 2.5-4%', df1[(df1['gap'] >= 4) & (df1['entry_gap'] >= 2.5) & (df1['entry_gap'] < 4)]),
    ('Gap>=5% + Entry 2-4%', df1[(df1['gap'] >= 5) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('Gap>=5% + Entry 2.5-4%', df1[(df1['gap'] >= 5) & (df1['entry_gap'] >= 2.5) & (df1['entry_gap'] < 4)]),
    ('Gap>=6% + Entry 2-4%', df1[(df1['gap'] >= 6) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('LONG + Entry 2-4%', df1[(df1['direction'] == 'LONG') & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('LONG + Gap>=4% + Entry 2-4%', df1[(df1['direction'] == 'LONG') & (df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('LONG + Gap>=5% + Entry 2-4%', df1[(df1['direction'] == 'LONG') & (df1['gap'] >= 5) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('SHORT + Entry 2-4%', df1[(df1['direction'] == 'SHORT') & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
    ('SHORT + Gap>=4% + Entry 2-4%', df1[(df1['direction'] == 'SHORT') & (df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]),
]

best_condition = None
best_monthly = 0

for name, subset in conditions:
    if len(subset) >= 20:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        
        # 월별 계산
        subset = subset.copy()
        subset['month'] = pd.to_datetime(subset['time']).dt.to_period('M')
        monthly_avg = subset.groupby('month')['pnl'].sum().mean()
        
        print(f"{name:>50} {len(subset):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f}")
        
        if monthly_avg > best_monthly:
            best_monthly = monthly_avg
            best_condition = (name, subset)

# ============================================================
# 최적 조건 상세 분석
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 상세 분석")
print("=" * 80)

# Entry Gap 2-4%
optimal = df1[(df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]
if len(optimal) > 0:
    print(f"\n[Entry Gap 2-4% 조건]")
    print(f"  총 거래: {len(optimal)}건")
    print(f"  승률: {(optimal['pnl']>0).mean()*100:.1f}%")
    print(f"  평균 PnL: {optimal['pnl'].mean():.2f}%")
    print(f"  총 PnL: {optimal['pnl'].sum():.1f}%")
    
    optimal = optimal.copy()
    optimal['month'] = pd.to_datetime(optimal['time']).dt.to_period('M')
    monthly = optimal.groupby('month')['pnl'].sum()
    
    print(f"  월 평균: {monthly.mean():.2f}%")
    print(f"  3x 레버리지: {monthly.mean()*3:.1f}%/월")
    print(f"  수익 월: {(monthly > 0).sum()}/{len(monthly)} ({(monthly>0).mean()*100:.1f}%)")

# Gap>=4% + Entry Gap 2-4%
optimal2 = df1[(df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]
if len(optimal2) > 0:
    print(f"\n[Gap>=4% + Entry Gap 2-4% 조건]")
    print(f"  총 거래: {len(optimal2)}건")
    print(f"  승률: {(optimal2['pnl']>0).mean()*100:.1f}%")
    print(f"  평균 PnL: {optimal2['pnl'].mean():.2f}%")
    print(f"  총 PnL: {optimal2['pnl'].sum():.1f}%")
    
    optimal2 = optimal2.copy()
    optimal2['month'] = pd.to_datetime(optimal2['time']).dt.to_period('M')
    monthly2 = optimal2.groupby('month')['pnl'].sum()
    
    print(f"  월 평균: {monthly2.mean():.2f}%")
    print(f"  3x 레버리지: {monthly2.mean()*3:.1f}%/월")
    print(f"  수익 월: {(monthly2 > 0).sum()}/{len(monthly2)} ({(monthly2>0).mean()*100:.1f}%)")

# LONG + Gap>=4% + Entry Gap 2-4%
optimal3 = df1[(df1['direction'] == 'LONG') & (df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]
if len(optimal3) > 0:
    print(f"\n[LONG + Gap>=4% + Entry Gap 2-4% 조건]")
    print(f"  총 거래: {len(optimal3)}건")
    print(f"  승률: {(optimal3['pnl']>0).mean()*100:.1f}%")
    print(f"  평균 PnL: {optimal3['pnl'].mean():.2f}%")
    print(f"  총 PnL: {optimal3['pnl'].sum():.1f}%")
    
    optimal3 = optimal3.copy()
    optimal3['month'] = pd.to_datetime(optimal3['time']).dt.to_period('M')
    monthly3 = optimal3.groupby('month')['pnl'].sum()
    
    print(f"  월 평균: {monthly3.mean():.2f}%")
    print(f"  3x 레버리지: {monthly3.mean()*3:.1f}%/월")
    print(f"  수익 월: {(monthly3 > 0).sum()}/{len(monthly3)} ({(monthly3>0).mean()*100:.1f}%)")
    print(f"  최대 월 수익: {monthly3.max():.2f}%")
    print(f"  최대 월 손실: {monthly3.min():.2f}%")

# ============================================================
# 방향별 상세 비교
# ============================================================
print("\n" + "=" * 80)
print("방향별 최적 조건 비교")
print("=" * 80)

# LONG 최적
long_opt = df1[(df1['direction'] == 'LONG') & (df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]
# SHORT 최적
short_opt = df1[(df1['direction'] == 'SHORT') & (df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LONG vs SHORT 최적 조건 비교                            │
│                      (Gap>=4% + Entry Gap 2-4%)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              LONG                 SHORT                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  건수              {len(long_opt):>10}건            {len(short_opt):>10}건                     │
│  승률              {(long_opt['pnl']>0).mean()*100 if len(long_opt)>0 else 0:>10.1f}%            {(short_opt['pnl']>0).mean()*100 if len(short_opt)>0 else 0:>10.1f}%                     │
│  평균 PnL         {long_opt['pnl'].mean() if len(long_opt)>0 else 0:>10.2f}%            {short_opt['pnl'].mean() if len(short_opt)>0 else 0:>10.2f}%                     │
│  총 PnL           {long_opt['pnl'].sum() if len(long_opt)>0 else 0:>10.1f}%            {short_opt['pnl'].sum() if len(short_opt)>0 else 0:>10.1f}%                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 실패 원인 분석
# ============================================================
print("\n" + "=" * 80)
print("실패 케이스 분석")
print("=" * 80)

# 최적 조건에서의 실패 케이스
if len(optimal3) > 0:
    sl_cases = optimal3[optimal3['exit_type'] == 'SL']
    tp_cases = optimal3[optimal3['exit_type'] == 'TP']
    timeout_cases = optimal3[optimal3['exit_type'] == 'TIMEOUT']
    
    print(f"\n[LONG + Gap>=4% + Entry 2-4% 실패 분석]")
    print(f"  TP: {len(tp_cases)}건 ({len(tp_cases)/len(optimal3)*100:.1f}%)")
    print(f"  SL: {len(sl_cases)}건 ({len(sl_cases)/len(optimal3)*100:.1f}%)")
    print(f"  TIMEOUT: {len(timeout_cases)}건 ({len(timeout_cases)/len(optimal3)*100:.1f}%)")
    
    if len(sl_cases) > 0:
        print(f"\n  SL 케이스 상세:")
        print(f"    평균 손실: {sl_cases['pnl'].mean():.2f}%")
        print(f"    평균 MFE: {sl_cases['mfe'].mean():.2f}%")
        mfe_above_1 = sl_cases[sl_cases['mfe'] >= 1]
        print(f"    MFE >= 1% 후 SL: {len(mfe_above_1)}건 ({len(mfe_above_1)/len(sl_cases)*100:.1f}%)")

# ============================================================
# 최종 추천 전략
# ============================================================
print("\n" + "=" * 80)
print("최종 추천 전략")
print("=" * 80)

# 롱/숏 통합 최적 조건
combined_opt = df1[(df1['gap'] >= 4) & (df1['entry_gap'] >= 2) & (df1['entry_gap'] < 4)]
combined_opt = combined_opt.copy()
combined_opt['month'] = pd.to_datetime(combined_opt['time']).dt.to_period('M')
combined_monthly = combined_opt.groupby('month')['pnl'].sum()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         최종 추천 전략                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 전략 명: MTF 최적화 Entry Gap 전략                                        │
│                                                                            │
│  ■ 핵심 조건:                                                               │
│    - 패턴: 1H W (LONG) / M (SHORT)                                         │
│    - Gap: >= 4% (에너지 축적)                                               │
│    - Entry Gap: 2~4% (L2/H2 기준)                                          │
│    - SL: min(L1, L2) / max(H1, H2)                                         │
│    - TP: 5%                                                                 │
│                                                                            │
│  ■ 성과:                                                                    │
│    - 총 거래: {len(combined_opt)}건                                                          │
│    - 승률: {(combined_opt['pnl']>0).mean()*100:.1f}%                                                          │
│    - 평균 PnL: {combined_opt['pnl'].mean():.2f}%                                                     │
│    - 월 평균: {combined_monthly.mean():.2f}%                                                       │
│    - 3x 레버리지: {combined_monthly.mean()*3:.1f}%/월                                             │
│    - 수익 월: {(combined_monthly > 0).sum()}/{len(combined_monthly)} ({(combined_monthly>0).mean()*100:.1f}%)                                          │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ■ 진입 체크리스트:                                                          │
│    □ 1H W/M 패턴 확인 (L1-H-L2 또는 H1-L-H2)                                │
│    □ Gap >= 4% (넥라인 - 저점/고점)                                          │
│    □ 1H 다이버전스 확인 (옵션)                                               │
│    □ 진입가: L2 + 2~4% (LONG) / H2 - 2~4% (SHORT)                          │
│    □ SL = L값 / H값                                                         │
│    □ TP = 진입가 + 5% (LONG) / 진입가 - 5% (SHORT)                          │
│                                                                            │
│  ■ 실전 적용:                                                               │
│    1. W/M 패턴 형성 대기                                                     │
│    2. Gap 4% 이상 확인                                                       │
│    3. L2/H2 + 2~4% 지점에 지정가 주문                                        │
│    4. SL/TP 자동 설정                                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
combined_opt.to_csv('mtf_best_conditions_results.csv', index=False)
print("\n결과 저장: mtf_best_conditions_results.csv")

# 월별 상세 성과
print("\n" + "=" * 80)
print("월별 상세 성과 (최적 조건)")
print("=" * 80)

monthly_detail = combined_opt.groupby('month').agg({
    'pnl': ['sum', 'count'],
    'direction': lambda x: (x == 'LONG').sum()
}).round(2)
monthly_detail.columns = ['총PnL', '거래수', 'LONG수']
monthly_detail['SHORT수'] = monthly_detail['거래수'] - monthly_detail['LONG수']

print(f"\n최근 12개월:")
print(monthly_detail.tail(12).to_string())
