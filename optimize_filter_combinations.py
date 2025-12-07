#!/usr/bin/env python3
"""
페이크 신호 필터링을 위한 조건 조합 최적화
- Entry-L 거리
- 돌파 강도 (Gap)
- 저점 하락폭 (L change)
- 고점 변화 (H change)
- 패턴 (H↑L↓, H↓L↑ 등)
- 진입 시간대
"""

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

# 데이터 로드
df_pattern = pd.read_csv('hl_pattern_analysis.csv')
df_backtest = pd.read_csv('L_value_backtest.csv')

# 중복 제거
df_pattern = df_pattern.drop_duplicates(subset=['breakout_time'])
df_backtest = df_backtest.drop_duplicates(subset=['entry_time'])

print(f"패턴 데이터: {len(df_pattern)}개")
print(f"백테스트 데이터: {len(df_backtest)}개")

# 데이터 병합
df_pattern['breakout_time'] = pd.to_datetime(df_pattern['breakout_time'])
df_backtest['entry_time'] = pd.to_datetime(df_backtest['entry_time'])

df = pd.merge(
    df_backtest, 
    df_pattern[['breakout_time', 'pattern', 'h_change', 'l_change', 'h_direction', 'l_direction']],
    left_on='entry_time',
    right_on='breakout_time',
    how='left'
)

# 시간대 추출
df['hour'] = df['entry_time'].dt.hour

# 돌파 강도 (Gap) 계산 - valid_signals.csv에서 가져오기
try:
    df_signals = pd.read_csv('valid_signals.csv')
    df_signals['breakout_time'] = pd.to_datetime(df_signals['breakout_time'])
    df_signals = df_signals.drop_duplicates(subset=['breakout_time'])
    df = pd.merge(df, df_signals[['breakout_time', 'gap_pct']], left_on='entry_time', right_on='breakout_time', how='left', suffixes=('', '_sig'))
except:
    df['gap_pct'] = 0

print(f"\n병합된 데이터: {len(df)}개")
print(f"컬럼: {df.columns.tolist()}")

# 결과 저장
df['is_win'] = df['total_pnl'] > 0
df['is_tp'] = df['exit_reason'] == 'TP_REACHED'

print(f"\n=== 기본 통계 ===")
print(f"전체 승률: {df['is_win'].mean()*100:.1f}%")
print(f"평균 수익: {df['total_pnl'].mean():.2f}%")
print(f"TP 달성률: {df['is_tp'].mean()*100:.1f}%")

# 필터 조건 정의
filter_configs = {
    'entry_L_gap': [0, 0.5, 1.0, 1.5, 2.0],  # 최소 Entry-L 거리
    'gap_pct': [0, 0.5, 1.0, 1.5],  # 최소 돌파 강도
    'l_change_min': [-999, -2, -1, 0],  # L 변화 최소값 (음수 = 하락)
    'h_change_max': [999, 0, -1, -2],  # H 변화 최대값 (음수 = 하락)
    'pattern': ['all', 'H↑ L↓', 'H↓ L↑', 'good_patterns'],  # 패턴 필터
    'hour_range': ['all', 'morning', 'afternoon', 'night'],  # 시간대
}

# 시간대 정의
def filter_by_hour(df, hour_range):
    if hour_range == 'all':
        return df
    elif hour_range == 'morning':  # 6-12시
        return df[(df['hour'] >= 6) & (df['hour'] < 12)]
    elif hour_range == 'afternoon':  # 12-18시
        return df[(df['hour'] >= 12) & (df['hour'] < 18)]
    elif hour_range == 'night':  # 18-6시
        return df[(df['hour'] >= 18) | (df['hour'] < 6)]
    return df

# 패턴 필터
def filter_by_pattern(df, pattern):
    if pattern == 'all':
        return df
    elif pattern == 'good_patterns':
        return df[df['pattern'].isin(['H↑ L↓', 'H↓ L↑', 'H↑ L↑'])]
    else:
        return df[df['pattern'] == pattern]

# 조합 테스트
results = []

# 단순화된 조합 (너무 많은 조합 방지)
entry_L_gaps = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
gap_pcts = [0, 0.3, 0.5, 0.7, 1.0]
patterns = ['all', 'H↑ L↓', 'H↓ L↑', 'H↑ L↑', 'H↓ L↓']
hour_ranges = ['all', 'morning', 'afternoon', 'night']
l_change_mins = [-999, -1, 0]  # L이 얼마나 떨어졌는지
h_change_maxs = [999, 0]  # H가 얼마나 떨어졌는지

total_combinations = len(entry_L_gaps) * len(gap_pcts) * len(patterns) * len(hour_ranges) * len(l_change_mins) * len(h_change_maxs)
print(f"\n총 {total_combinations}개 조합 테스트 시작...")

count = 0
for entry_L_min in entry_L_gaps:
    for gap_min in gap_pcts:
        for pattern in patterns:
            for hour_range in hour_ranges:
                for l_change_min in l_change_mins:
                    for h_change_max in h_change_maxs:
                        count += 1
                        
                        # 필터 적용
                        filtered = df.copy()
                        
                        # Entry-L 거리 필터
                        filtered = filtered[filtered['entry_L_gap'] >= entry_L_min]
                        
                        # 돌파 강도 필터
                        if 'gap_pct' in filtered.columns:
                            filtered = filtered[filtered['gap_pct'].fillna(0) >= gap_min]
                        
                        # 패턴 필터
                        filtered = filter_by_pattern(filtered, pattern)
                        
                        # 시간대 필터
                        filtered = filter_by_hour(filtered, hour_range)
                        
                        # L 변화 필터
                        if 'l_change' in filtered.columns and l_change_min != -999:
                            filtered = filtered[filtered['l_change'].fillna(0) <= l_change_min]
                        
                        # H 변화 필터
                        if 'h_change' in filtered.columns and h_change_max != 999:
                            filtered = filtered[filtered['h_change'].fillna(0) <= h_change_max]
                        
                        # 결과 계산
                        n_trades = len(filtered)
                        if n_trades < 10:  # 최소 10개 거래
                            continue
                        
                        win_rate = filtered['is_win'].mean() * 100
                        avg_pnl = filtered['total_pnl'].mean()
                        total_pnl = filtered['total_pnl'].sum()
                        tp_rate = filtered['is_tp'].mean() * 100
                        sl_rate = (filtered['exit_reason'] == 'SL_L_BREAK').mean() * 100
                        
                        results.append({
                            'entry_L_min': entry_L_min,
                            'gap_min': gap_min,
                            'pattern': pattern,
                            'hour_range': hour_range,
                            'l_change_min': l_change_min,
                            'h_change_max': h_change_max,
                            'n_trades': n_trades,
                            'win_rate': win_rate,
                            'avg_pnl': avg_pnl,
                            'total_pnl': total_pnl,
                            'tp_rate': tp_rate,
                            'sl_rate': sl_rate,
                            'score': win_rate * 0.3 + avg_pnl * 20 + tp_rate * 0.2  # 종합 점수
                        })

print(f"\n{len(results)}개 유효 조합 발견")

# 결과 정렬 및 출력
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('score', ascending=False)

print("\n" + "="*80)
print("🏆 TOP 20 필터 조합 (종합 점수 기준)")
print("="*80)

for i, row in df_results.head(20).iterrows():
    print(f"\n#{df_results.index.get_loc(i)+1}. 점수: {row['score']:.1f}")
    print(f"   조건: Entry-L≥{row['entry_L_min']}%, Gap≥{row['gap_min']}%, 패턴={row['pattern']}, 시간={row['hour_range']}")
    print(f"         L변화≤{row['l_change_min']}%, H변화≤{row['h_change_max']}%")
    print(f"   성과: 거래수={row['n_trades']}, 승률={row['win_rate']:.1f}%, 평균수익={row['avg_pnl']:.2f}%, TP율={row['tp_rate']:.1f}%")

# 최고 승률 조합
print("\n" + "="*80)
print("🎯 최고 승률 조합 (거래수 20개 이상)")
print("="*80)
df_high_wr = df_results[df_results['n_trades'] >= 20].sort_values('win_rate', ascending=False).head(10)
for i, row in df_high_wr.iterrows():
    print(f"\n승률 {row['win_rate']:.1f}%: Entry-L≥{row['entry_L_min']}%, Gap≥{row['gap_min']}%, 패턴={row['pattern']}")
    print(f"   거래수={row['n_trades']}, 평균수익={row['avg_pnl']:.2f}%, TP율={row['tp_rate']:.1f}%")

# 최고 평균수익 조합
print("\n" + "="*80)
print("💰 최고 평균수익 조합 (거래수 20개 이상)")
print("="*80)
df_high_pnl = df_results[df_results['n_trades'] >= 20].sort_values('avg_pnl', ascending=False).head(10)
for i, row in df_high_pnl.iterrows():
    print(f"\n평균수익 {row['avg_pnl']:.2f}%: Entry-L≥{row['entry_L_min']}%, Gap≥{row['gap_min']}%, 패턴={row['pattern']}")
    print(f"   거래수={row['n_trades']}, 승률={row['win_rate']:.1f}%, TP율={row['tp_rate']:.1f}%")

# 균형 잡힌 조합 (승률 60%+, 평균수익 1%+, 거래수 30+)
print("\n" + "="*80)
print("⚖️ 균형 잡힌 추천 조합 (승률≥55%, 평균수익≥1%, 거래수≥30)")
print("="*80)
df_balanced = df_results[
    (df_results['win_rate'] >= 55) & 
    (df_results['avg_pnl'] >= 1.0) & 
    (df_results['n_trades'] >= 30)
].sort_values('score', ascending=False)

if len(df_balanced) > 0:
    for i, row in df_balanced.head(10).iterrows():
        print(f"\n점수 {row['score']:.1f}: Entry-L≥{row['entry_L_min']}%, Gap≥{row['gap_min']}%, 패턴={row['pattern']}")
        print(f"   시간={row['hour_range']}, L변화≤{row['l_change_min']}%, H변화≤{row['h_change_max']}%")
        print(f"   거래수={row['n_trades']}, 승률={row['win_rate']:.1f}%, 평균수익={row['avg_pnl']:.2f}%, TP율={row['tp_rate']:.1f}%")
else:
    print("조건을 만족하는 조합이 없습니다. 기준을 완화합니다...")
    df_balanced = df_results[
        (df_results['win_rate'] >= 50) & 
        (df_results['avg_pnl'] >= 0.8) & 
        (df_results['n_trades'] >= 20)
    ].sort_values('score', ascending=False)
    for i, row in df_balanced.head(10).iterrows():
        print(f"\n점수 {row['score']:.1f}: Entry-L≥{row['entry_L_min']}%, Gap≥{row['gap_min']}%, 패턴={row['pattern']}")
        print(f"   시간={row['hour_range']}, L변화≤{row['l_change_min']}%, H변화≤{row['h_change_max']}%")
        print(f"   거래수={row['n_trades']}, 승률={row['win_rate']:.1f}%, 평균수익={row['avg_pnl']:.2f}%, TP율={row['tp_rate']:.1f}%")

# 결과 저장
df_results.to_csv('filter_optimization_results.csv', index=False)
print(f"\n\n결과 저장: filter_optimization_results.csv")

# 최적 조합 상세 분석
print("\n" + "="*80)
print("🔬 최적 조합 상세 분석")
print("="*80)

if len(df_results) > 0:
    best = df_results.iloc[0]
    print(f"\n최적 조합:")
    print(f"  - Entry-L 거리: ≥ {best['entry_L_min']}%")
    print(f"  - 돌파 강도: ≥ {best['gap_min']}%")
    print(f"  - 패턴: {best['pattern']}")
    print(f"  - 시간대: {best['hour_range']}")
    print(f"  - L 변화: ≤ {best['l_change_min']}%")
    print(f"  - H 변화: ≤ {best['h_change_max']}%")
    print(f"\n성과:")
    print(f"  - 거래 수: {best['n_trades']}회")
    print(f"  - 승률: {best['win_rate']:.1f}%")
    print(f"  - 평균 수익: {best['avg_pnl']:.2f}%")
    print(f"  - 총 수익: {best['total_pnl']:.1f}%")
    print(f"  - TP 달성률: {best['tp_rate']:.1f}%")
    print(f"  - SL 비율: {best['sl_rate']:.1f}%")
