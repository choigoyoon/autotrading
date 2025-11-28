"""
상황별 성과 분석
- 기존 전략을 상황별로 분리 분석
- 어느 상황이 수익 좋은가?
- 어느 상황을 피해야 하는가?
- 상황별 특성 파악
"""

import pandas as pd
import numpy as np

print("="*60)
print("상황별 성과 분석")
print("="*60)

# 데이터 로드
print("\n데이터 로드 중...")
df_classified = pd.read_csv('output_mtf_situation_classified.csv')
df_classified['datetime'] = pd.to_datetime(df_classified['datetime'])

# 기존 돌파 데이터
breakouts_df = pd.read_csv('output_phase4_breakouts.csv')
trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

print(f"분류 데이터: {len(df_classified):,}개")
print(f"추세선 돌파: {len(trendline_breakouts):,}개")

# 상황 정의
SITUATIONS = {
    'A': '상승 강세',
    'B': '상승 중 눌림 (반등값 큼)',
    'C': '상승 초기',
    'D': '혼조 상승',
    'E': '하락 중 반등',
    'F': '바닥 잡기',
    'G': '하락 초기',
    'H': '혼조 하락',
}

# 각 돌파에 상황 매칭
print("\n돌파에 상황 매칭 중...")
breakout_situations = []

for idx, breakout in trendline_breakouts.iterrows():
    break_idx = breakout['break_idx']

    if break_idx >= len(df_classified):
        breakout_situations.append(None)
        continue

    situation = df_classified.iloc[break_idx]['situation']
    breakout_situations.append(situation)

trendline_breakouts = trendline_breakouts.copy()
trendline_breakouts['situation'] = breakout_situations

# 상황별 통계
print("\n"+"="*60)
print("상황별 돌파 분포")
print("="*60)

situation_counts = trendline_breakouts['situation'].value_counts().sort_index()
total_breakouts = len(trendline_breakouts[trendline_breakouts['situation'].notna()])

print(f"\n총 돌파: {total_breakouts:,}개")
for sit in sorted(SITUATIONS.keys()):
    count = situation_counts.get(sit, 0)
    pct = count / total_breakouts * 100 if total_breakouts > 0 else 0
    print(f"{sit} ({SITUATIONS[sit]}): {count:,}개 ({pct:.1f}%)")

# 상황별 수익 분석 (간단 버전)
print("\n"+"="*60)
print("상황별 수익 분석 (60봉 홀딩)")
print("="*60)

situation_performance = []

for sit in sorted(SITUATIONS.keys()):
    sit_breakouts = trendline_breakouts[trendline_breakouts['situation'] == sit]

    if len(sit_breakouts) == 0:
        continue

    returns = []

    for idx, breakout in sit_breakouts.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        end_idx = min(break_idx + 60, len(df_classified) - 1)
        if end_idx <= break_idx:
            continue

        window = df_classified.iloc[break_idx:end_idx+1]
        break_price = window.iloc[0]['close']

        if direction == 'long':
            max_high = window['high'].max()
            max_profit = (max_high - break_price) / break_price * 100

            final_price = window.iloc[-1]['close']
            final_return = (final_price - break_price) / break_price * 100
        else:
            max_low = window['low'].min()
            max_profit = (break_price - max_low) / break_price * 100

            final_price = window.iloc[-1]['close']
            final_return = (break_price - final_price) / break_price * 100

        returns.append({
            'max_profit': max_profit,
            'final_return': final_return
        })

    if len(returns) == 0:
        continue

    returns_df = pd.DataFrame(returns)

    situation_performance.append({
        'situation': sit,
        'name': SITUATIONS[sit],
        'count': len(returns),
        'avg_max': returns_df['max_profit'].mean(),
        'avg_final': returns_df['final_return'].mean(),
        'win_rate': (returns_df['final_return'] > 0).sum() / len(returns) * 100,
        'reach_1pct': (returns_df['max_profit'] >= 1.0).sum() / len(returns) * 100,
        'reach_2pct': (returns_df['max_profit'] >= 2.0).sum() / len(returns) * 100,
    })

perf_df = pd.DataFrame(situation_performance)

print("\n상황별 성과:")
print(perf_df.to_string(index=False))

# 최고/최악 상황
print("\n"+"="*60)
print("최고/최악 상황")
print("="*60)

best_final = perf_df.loc[perf_df['avg_final'].idxmax()]
worst_final = perf_df.loc[perf_df['avg_final'].idxmin()]

print(f"\n최고 평균 수익:")
print(f"  상황: {best_final['situation']} ({best_final['name']})")
print(f"  거래: {best_final['count']:.0f}회")
print(f"  평균 수익: {best_final['avg_final']:.3f}%")
print(f"  승률: {best_final['win_rate']:.1f}%")
print(f"  +2% 도달: {best_final['reach_2pct']:.1f}%")

print(f"\n최악 평균 수익:")
print(f"  상황: {worst_final['situation']} ({worst_final['name']})")
print(f"  거래: {worst_final['count']:.0f}회")
print(f"  평균 수익: {worst_final['avg_final']:.3f}%")
print(f"  승률: {worst_final['win_rate']:.1f}%")

# 롱/숏 분리 분석
print("\n"+"="*60)
print("상황별 롱/숏 분석")
print("="*60)

for sit in sorted(SITUATIONS.keys()):
    sit_breakouts = trendline_breakouts[trendline_breakouts['situation'] == sit]

    if len(sit_breakouts) == 0:
        continue

    long_count = (sit_breakouts['type'] == 'trendline_up').sum()
    short_count = (sit_breakouts['type'] == 'trendline_down').sum()

    print(f"\n{sit} ({SITUATIONS[sit]}):")
    print(f"  롱: {long_count}회 ({long_count/(long_count+short_count)*100:.1f}%)")
    print(f"  숏: {short_count}회 ({short_count/(long_count+short_count)*100:.1f}%)")

# 블러 필터 제안
print("\n"+"="*60)
print("블러 필터 제안")
print("="*60)

# 수익 마이너스 상황 찾기
negative_situations = perf_df[perf_df['avg_final'] < 0]

if len(negative_situations) > 0:
    print(f"\n수익 마이너스 상황 ({len(negative_situations)}개):")
    for idx, row in negative_situations.iterrows():
        print(f"  {row['situation']} ({row['name']}): {row['avg_final']:.3f}%")
    print(f"\n제안: 이 상황들은 진입 금지 또는 파라미터 조정 필요")
else:
    print("\n모든 상황이 플러스 수익!")

# 승률 50% 미만 상황
low_winrate = perf_df[perf_df['win_rate'] < 50]

if len(low_winrate) > 0:
    print(f"\n승률 50% 미만 상황 ({len(low_winrate)}개):")
    for idx, row in low_winrate.iterrows():
        print(f"  {row['situation']} ({row['name']}): {row['win_rate']:.1f}%")

# 방향성 필터
print("\n"+"="*60)
print("방향성 필터 제안")
print("="*60)

print(f"\n상황별 권장 방향:")
for sit in ['A', 'B', 'C', 'D']:
    print(f"  {sit} ({SITUATIONS[sit]}): 롱 우선")
for sit in ['E', 'F', 'G', 'H']:
    if sit in ['E']:
        print(f"  {sit} ({SITUATIONS[sit]}): 롱 가능 (반등)")
    else:
        print(f"  {sit} ({SITUATIONS[sit]}): 숏 우선 또는 관망")

# 저장
output_file = 'analysis_situation_performance.csv'
perf_df.to_csv(output_file, index=False)
print(f"\n저장: {output_file}")

print("\n"+"="*60)
print("분석 완료")
print("="*60)

print("\n다음 단계:")
print("  1. 블러 처리 구현 (수익 마이너스 상황 제거)")
print("  2. 상황별 파라미터 최적화")
print("  3. MTF Zone 기반 진입 필터")
print("  4. 통합 백테스트")
