#!/usr/bin/env python3
"""
최종 전략 분석 - 실용적인 필터 조합 선정 및 상세 검증
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 데이터 로드
df_pattern = pd.read_csv('hl_pattern_analysis.csv')
df_backtest = pd.read_csv('L_value_backtest.csv')
df_signals = pd.read_csv('valid_signals.csv')

# 중복 제거
df_pattern = df_pattern.drop_duplicates(subset=['breakout_time'])
df_backtest = df_backtest.drop_duplicates(subset=['entry_time'])
df_signals = df_signals.drop_duplicates(subset=['breakout_time'])

# datetime 변환
df_pattern['breakout_time'] = pd.to_datetime(df_pattern['breakout_time'])
df_backtest['entry_time'] = pd.to_datetime(df_backtest['entry_time'])
df_signals['breakout_time'] = pd.to_datetime(df_signals['breakout_time'])

# 데이터 병합
df = pd.merge(
    df_backtest, 
    df_pattern[['breakout_time', 'pattern', 'h_change', 'l_change']],
    left_on='entry_time',
    right_on='breakout_time',
    how='left'
)
df = pd.merge(df, df_signals[['breakout_time', 'gap_pct']], left_on='entry_time', right_on='breakout_time', how='left', suffixes=('', '_sig'))

df['hour'] = df['entry_time'].dt.hour
df['year'] = df['entry_time'].dt.year
df['is_win'] = df['total_pnl'] > 0
df['is_tp'] = df['exit_reason'] == 'TP_REACHED'

print("="*80)
print("🎯 최종 전략 필터 분석")
print("="*80)

# 후보 전략들
strategies = [
    {
        'name': '기본 (필터 없음)',
        'filters': {},
        'desc': '모든 신호 진입'
    },
    {
        'name': 'Entry-L ≥ 1%',
        'filters': {'entry_L_gap': 1.0},
        'desc': '손절 여유 1% 이상'
    },
    {
        'name': 'Entry-L ≥ 1.5%',
        'filters': {'entry_L_gap': 1.5},
        'desc': '손절 여유 1.5% 이상'
    },
    {
        'name': 'H↑ L↓ 패턴',
        'filters': {'pattern': 'H↑ L↓'},
        'desc': '고점↑ 저점↓ 패턴만'
    },
    {
        'name': 'H↑ L↓ + Entry-L≥1%',
        'filters': {'pattern': 'H↑ L↓', 'entry_L_gap': 1.0},
        'desc': '최고 승률 조합'
    },
    {
        'name': 'Entry-L≥1.5% + 오후',
        'filters': {'entry_L_gap': 1.5, 'hour_range': 'afternoon'},
        'desc': '균형 잡힌 조합'
    },
    {
        'name': 'L하락≤-1% + Entry-L≥2%',
        'filters': {'l_change': -1, 'entry_L_gap': 2.0},
        'desc': '저점 충분히 하락'
    },
]

def apply_filters(df, filters):
    filtered = df.copy()
    
    if 'entry_L_gap' in filters:
        filtered = filtered[filtered['entry_L_gap'] >= filters['entry_L_gap']]
    
    if 'pattern' in filters:
        filtered = filtered[filtered['pattern'] == filters['pattern']]
    
    if 'hour_range' in filters:
        hr = filters['hour_range']
        if hr == 'afternoon':
            filtered = filtered[(filtered['hour'] >= 12) & (filtered['hour'] < 18)]
        elif hr == 'morning':
            filtered = filtered[(filtered['hour'] >= 6) & (filtered['hour'] < 12)]
        elif hr == 'night':
            filtered = filtered[(filtered['hour'] >= 18) | (filtered['hour'] < 6)]
    
    if 'l_change' in filters:
        filtered = filtered[filtered['l_change'] <= filters['l_change']]
    
    if 'gap_pct' in filters:
        filtered = filtered[filtered['gap_pct'].fillna(0) >= filters['gap_pct']]
    
    return filtered

# 각 전략 평가
print("\n" + "-"*80)
print(f"{'전략명':<25} {'거래수':>6} {'승률':>8} {'평균수익':>10} {'총수익':>10} {'TP율':>8} {'SL율':>8}")
print("-"*80)

strategy_results = []
for s in strategies:
    filtered = apply_filters(df, s['filters'])
    n = len(filtered)
    if n == 0:
        continue
    
    win_rate = filtered['is_win'].mean() * 100
    avg_pnl = filtered['total_pnl'].mean()
    total_pnl = filtered['total_pnl'].sum()
    tp_rate = filtered['is_tp'].mean() * 100
    sl_rate = (filtered['exit_reason'] == 'SL_L_BREAK').mean() * 100
    
    print(f"{s['name']:<25} {n:>6} {win_rate:>7.1f}% {avg_pnl:>9.2f}% {total_pnl:>9.1f}% {tp_rate:>7.1f}% {sl_rate:>7.1f}%")
    
    strategy_results.append({
        'name': s['name'],
        'desc': s['desc'],
        'filters': str(s['filters']),
        'n_trades': n,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'tp_rate': tp_rate,
        'sl_rate': sl_rate
    })

# 최적 전략 선정: H↑ L↓ + Entry-L≥1%
print("\n" + "="*80)
print("🏆 최적 전략 선정: H↑ L↓ + Entry-L ≥ 1%")
print("="*80)

best_strategy = {'pattern': 'H↑ L↓', 'entry_L_gap': 1.0}
df_best = apply_filters(df, best_strategy)

print(f"\n📊 성과 요약")
print(f"  - 거래 수: {len(df_best)}회")
print(f"  - 승률: {df_best['is_win'].mean()*100:.1f}%")
print(f"  - 평균 수익: {df_best['total_pnl'].mean():.2f}%")
print(f"  - 총 수익: {df_best['total_pnl'].sum():.1f}%")
print(f"  - TP 달성률: {df_best['is_tp'].mean()*100:.1f}%")
print(f"  - SL 비율: {(df_best['exit_reason']=='SL_L_BREAK').mean()*100:.1f}%")

# 연도별 성과
print(f"\n📅 연도별 성과")
yearly = df_best.groupby('year').agg({
    'total_pnl': ['count', 'mean', 'sum'],
    'is_win': 'mean',
    'is_tp': 'mean'
}).round(2)
yearly.columns = ['거래수', '평균수익', '총수익', '승률', 'TP율']
yearly['승률'] = (yearly['승률'] * 100).round(1)
yearly['TP율'] = (yearly['TP율'] * 100).round(1)
print(yearly)

# 기본 전략과 비교
print("\n" + "="*80)
print("📈 기본 전략 vs 최적 전략 비교")
print("="*80)

df_base = df.copy()
comparison = pd.DataFrame({
    '지표': ['거래 수', '승률', '평균 수익', '총 수익', 'TP 달성률', 'SL 비율'],
    '기본 전략': [
        f"{len(df_base)}회",
        f"{df_base['is_win'].mean()*100:.1f}%",
        f"{df_base['total_pnl'].mean():.2f}%",
        f"{df_base['total_pnl'].sum():.1f}%",
        f"{df_base['is_tp'].mean()*100:.1f}%",
        f"{(df_base['exit_reason']=='SL_L_BREAK').mean()*100:.1f}%"
    ],
    '최적 전략': [
        f"{len(df_best)}회",
        f"{df_best['is_win'].mean()*100:.1f}%",
        f"{df_best['total_pnl'].mean():.2f}%",
        f"{df_best['total_pnl'].sum():.1f}%",
        f"{df_best['is_tp'].mean()*100:.1f}%",
        f"{(df_best['exit_reason']=='SL_L_BREAK').mean()*100:.1f}%"
    ],
    '개선율': [
        f"{(len(df_best)/len(df_base)-1)*100:+.1f}%",
        f"{(df_best['is_win'].mean()/df_base['is_win'].mean()-1)*100:+.1f}%",
        f"{(df_best['total_pnl'].mean()/df_base['total_pnl'].mean()-1)*100:+.1f}%",
        f"{(df_best['total_pnl'].sum()/df_base['total_pnl'].sum()-1)*100:+.1f}%",
        f"{(df_best['is_tp'].mean()/df_base['is_tp'].mean()-1)*100:+.1f}%",
        f"{((df_best['exit_reason']=='SL_L_BREAK').mean()/(df_base['exit_reason']=='SL_L_BREAK').mean()-1)*100:+.1f}%"
    ]
})
print(comparison.to_string(index=False))

# 실패 케이스 분석
print("\n" + "="*80)
print("🔍 최적 전략의 실패 케이스 분석")
print("="*80)

df_fail = df_best[df_best['exit_reason'] == 'SL_L_BREAK']
if len(df_fail) > 0:
    print(f"\n실패 케이스: {len(df_fail)}건 ({len(df_fail)/len(df_best)*100:.1f}%)")
    print(f"\n실패 특성:")
    print(f"  - 평균 Entry-L 거리: {df_fail['entry_L_gap'].mean():.2f}%")
    print(f"  - 평균 손실: {df_fail['total_pnl'].mean():.2f}%")
    print(f"  - 평균 보유 시간: {df_fail['hold_hours'].mean():.1f}시간")
    
    # 시간대별
    fail_hour = df_fail['hour'].value_counts().sort_index()
    print(f"\n실패 시간대 분포:")
    for h, c in fail_hour.items():
        print(f"    {h}시: {c}건")

# 성공 케이스 분석
df_success = df_best[df_best['exit_reason'] == 'TP_REACHED']
if len(df_success) > 0:
    print(f"\n성공 케이스: {len(df_success)}건 ({len(df_success)/len(df_best)*100:.1f}%)")
    print(f"\n성공 특성:")
    print(f"  - 평균 Entry-L 거리: {df_success['entry_L_gap'].mean():.2f}%")
    print(f"  - 평균 수익: {df_success['total_pnl'].mean():.2f}%")
    print(f"  - 평균 보유 시간: {df_success['hold_hours'].mean():.1f}시간")

# 최종 전략 요약
print("\n" + "="*80)
print("📋 최종 전략 요약")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────┐
│                    최종 추천 전략                            │
├─────────────────────────────────────────────────────────────┤
│ 【진입 조건】                                                │
│   1. H-H 하향 추세선 돌파 (기존)                             │
│   2. H↑ L↓ 패턴 확인 (고점 상승 + 저점 하락)                │
│   3. Entry - L 거리 ≥ 1% (손절 여유 확보)                   │
│                                                             │
│ 【청산 조건】                                                │
│   1. 목표가 도달: 추세선(저항선) 도달 시 익절               │
│   2. 손절: L값(돌파 전 저점) 붕괴 시 손절                   │
│   3. 시간 제한: 없음 (조건 충족까지 홀딩)                   │
│                                                             │
│ 【예상 성과】                                                │
│   - 승률: ~82%                                               │
│   - 평균 수익: ~2.3%                                         │
│   - TP 달성률: ~73%                                          │
│   - SL 비율: ~18%                                            │
└─────────────────────────────────────────────────────────────┘
""")

# 결과 저장
pd.DataFrame(strategy_results).to_csv('strategy_comparison.csv', index=False)
df_best.to_csv('best_strategy_trades.csv', index=False)
print("결과 저장: strategy_comparison.csv, best_strategy_trades.csv")
