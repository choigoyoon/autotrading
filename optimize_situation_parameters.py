"""
상황별 파라미터 최적화
- 각 상황(A~H)마다 최적 TP/SL 찾기
- 현재 전략은 통합 파라미터 사용
- 상황별로 다른 파라미터 사용 시 성과 향상 기대
"""

import pandas as pd
import numpy as np
from itertools import product

print("="*60)
print("상황별 파라미터 최적화")
print("="*60)

# 데이터 로드
print("\n데이터 로드 중...")
df_classified = pd.read_csv('output_mtf_situation_classified.csv')
df_classified['datetime'] = pd.to_datetime(df_classified['datetime'])

breakouts_df = pd.read_csv('output_phase4_breakouts.csv')
trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

print(f"분류 데이터: {len(df_classified):,}개")
print(f"추세선 돌파: {len(trendline_breakouts):,}개")

# 상황 매칭
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

# 파라미터 그리드
param_grid = {
    'tp_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
    'sl_pct': [0.5, 0.7, 1.0, 1.5, 2.0],
}

# 상황 정의
SITUATIONS = {
    'A': '상승 강세',
    'B': '상승 중 눌림',
    'C': '상승 초기',
    'D': '혼조 상승',
    'E': '하락 중 반등',
    'F': '바닥 잡기',
    'G': '하락 초기',
    'H': '혼조 하락',
}

def backtest_with_params(breakouts, df_classified, tp_pct, sl_pct):
    """특정 파라미터로 백테스트"""

    results = []

    for idx, breakout in breakouts.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        # 최대 200봉까지 검색 (50시간)
        max_idx = min(break_idx + 200, len(df_classified) - 1)
        if max_idx <= break_idx:
            continue

        window = df_classified.iloc[break_idx:max_idx+1]
        break_price = window.iloc[0]['close']

        # TP/SL 레벨 설정
        if direction == 'long':
            tp_level = break_price * (1 + tp_pct / 100)
            sl_level = break_price * (1 - sl_pct / 100)

            # TP/SL 도달 확인
            tp_hit = (window['high'] >= tp_level).any()
            sl_hit = (window['low'] <= sl_level).any()

            if tp_hit and sl_hit:
                # 둘 다 도달 - 먼저 도달한 것 선택
                tp_idx = window[window['high'] >= tp_level].index[0]
                sl_idx = window[window['low'] <= sl_level].index[0]

                if tp_idx < sl_idx:
                    pnl = tp_pct
                else:
                    pnl = -sl_pct
            elif tp_hit:
                pnl = tp_pct
            elif sl_hit:
                pnl = -sl_pct
            else:
                # 둘 다 안 도달 - 마지막 가격
                final_price = window.iloc[-1]['close']
                pnl = (final_price - break_price) / break_price * 100

        else:  # short
            tp_level = break_price * (1 - tp_pct / 100)
            sl_level = break_price * (1 + sl_pct / 100)

            tp_hit = (window['low'] <= tp_level).any()
            sl_hit = (window['high'] >= sl_level).any()

            if tp_hit and sl_hit:
                tp_idx = window[window['low'] <= tp_level].index[0]
                sl_idx = window[window['high'] >= sl_level].index[0]

                if tp_idx < sl_idx:
                    pnl = tp_pct
                else:
                    pnl = -sl_pct
            elif tp_hit:
                pnl = tp_pct
            elif sl_hit:
                pnl = -sl_pct
            else:
                final_price = window.iloc[-1]['close']
                pnl = (break_price - final_price) / break_price * 100

        results.append({
            'pnl': pnl,
            'tp_hit': tp_hit if direction == 'long' else tp_hit,
            'sl_hit': sl_hit if direction == 'long' else sl_hit,
        })

    if len(results) == 0:
        return None

    results_df = pd.DataFrame(results)

    return {
        'count': len(results),
        'win_rate': (results_df['pnl'] > 0).sum() / len(results) * 100,
        'avg_pnl': results_df['pnl'].mean(),
        'total_return': results_df['pnl'].sum(),
        'tp_rate': results_df['tp_hit'].sum() / len(results) * 100,
        'sl_rate': results_df['sl_hit'].sum() / len(results) * 100,
        'sharpe': results_df['pnl'].mean() / results_df['pnl'].std() if results_df['pnl'].std() > 0 else 0,
    }

print("\n" + "="*60)
print("상황별 최적 파라미터 탐색")
print("="*60)

# 각 상황별로 최적화
best_params_by_situation = {}

for sit in sorted(SITUATIONS.keys()):
    print(f"\n{'='*60}")
    print(f"상황 {sit} ({SITUATIONS[sit]}) 최적화")
    print(f"{'='*60}")

    sit_breakouts = trendline_breakouts[trendline_breakouts['situation'] == sit]

    if len(sit_breakouts) < 10:
        print(f"  거래 수 부족 ({len(sit_breakouts)}개) - 스킵")
        continue

    print(f"  거래 수: {len(sit_breakouts):,}개")

    # 그리드 서치
    best_score = -999999
    best_params = None
    all_results = []

    total_combinations = len(param_grid['tp_pct']) * len(param_grid['sl_pct'])
    current = 0

    for tp, sl in product(param_grid['tp_pct'], param_grid['sl_pct']):
        current += 1
        if current % 5 == 0:
            print(f"  진행: {current}/{total_combinations}...", end='\r')

        result = backtest_with_params(sit_breakouts, df_classified, tp, sl)

        if result is None:
            continue

        # 점수: 평균 수익 * 승률 (간단한 점수 함수)
        score = result['avg_pnl'] * result['win_rate']

        all_results.append({
            'tp': tp,
            'sl': sl,
            'score': score,
            **result
        })

        if score > best_score:
            best_score = score
            best_params = {
                'tp': tp,
                'sl': sl,
                **result
            }

    print(" " * 50, end='\r')  # 진행 표시 지우기

    if best_params is None:
        print("  최적 파라미터 없음")
        continue

    best_params_by_situation[sit] = best_params

    print(f"  최적 파라미터:")
    print(f"    TP: {best_params['tp']:.1f}%")
    print(f"    SL: {best_params['sl']:.1f}%")
    print(f"    평균 수익: {best_params['avg_pnl']:.3f}%")
    print(f"    승률: {best_params['win_rate']:.1f}%")
    print(f"    총 수익: {best_params['total_return']:.1f}%")
    print(f"    TP 도달률: {best_params['tp_rate']:.1f}%")
    print(f"    SL 도달률: {best_params['sl_rate']:.1f}%")
    print(f"    Sharpe: {best_params['sharpe']:.3f}")

    # 상위 3개 파라미터 조합
    all_results_df = pd.DataFrame(all_results)
    top3 = all_results_df.nlargest(3, 'score')

    print(f"\n  상위 3개 조합:")
    for idx, row in top3.iterrows():
        print(f"    TP:{row['tp']:.1f}% SL:{row['sl']:.1f}% → "
              f"수익:{row['avg_pnl']:.3f}% 승률:{row['win_rate']:.1f}%")

# 결과 요약
print("\n" + "="*60)
print("상황별 최적 파라미터 요약")
print("="*60)

summary_data = []

for sit in sorted(SITUATIONS.keys()):
    if sit not in best_params_by_situation:
        continue

    params = best_params_by_situation[sit]
    summary_data.append({
        'situation': sit,
        'name': SITUATIONS[sit],
        'tp': params['tp'],
        'sl': params['sl'],
        'trades': params['count'],
        'avg_pnl': params['avg_pnl'],
        'win_rate': params['win_rate'],
        'total_return': params['total_return'],
        'sharpe': params['sharpe'],
    })

summary_df = pd.DataFrame(summary_data)

print("\n상황별 최적 파라미터:")
print(summary_df.to_string(index=False))

# 통합 파라미터와 비교
print("\n" + "="*60)
print("통합 vs 상황별 비교")
print("="*60)

# 통합 파라미터 (기본값 가정: TP 2.0%, SL 1.0%)
unified_tp = 2.0
unified_sl = 1.0

print(f"\n통합 파라미터 (TP:{unified_tp}% SL:{unified_sl}%) 성과:")

unified_results = []
for sit in sorted(SITUATIONS.keys()):
    sit_breakouts = trendline_breakouts[trendline_breakouts['situation'] == sit]
    if len(sit_breakouts) < 10:
        continue

    result = backtest_with_params(sit_breakouts, df_classified, unified_tp, unified_sl)
    if result:
        unified_results.append({
            'situation': sit,
            'name': SITUATIONS[sit],
            'avg_pnl': result['avg_pnl'],
            'win_rate': result['win_rate'],
            'total_return': result['total_return'],
        })

unified_df = pd.DataFrame(unified_results)

# 상황별과 통합 비교
comparison = []
for sit in sorted(SITUATIONS.keys()):
    if sit not in best_params_by_situation:
        continue

    optimized = best_params_by_situation[sit]
    unified_row = unified_df[unified_df['situation'] == sit]

    if len(unified_row) == 0:
        continue

    unified_pnl = unified_row.iloc[0]['avg_pnl']
    optimized_pnl = optimized['avg_pnl']
    improvement = optimized_pnl - unified_pnl
    improvement_pct = improvement / abs(unified_pnl) * 100 if unified_pnl != 0 else 0

    comparison.append({
        'situation': sit,
        'name': SITUATIONS[sit],
        'unified_pnl': unified_pnl,
        'optimized_pnl': optimized_pnl,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
    })

comparison_df = pd.DataFrame(comparison)

print("\n개선 효과:")
print(comparison_df.to_string(index=False))

total_improvement = comparison_df['improvement'].sum()
print(f"\n총 개선: {total_improvement:+.3f}%포인트")

# 저장
output_file = 'optimized_situation_parameters.csv'
summary_df.to_csv(output_file, index=False)
print(f"\n저장: {output_file}")

comparison_file = 'parameter_optimization_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"저장: {comparison_file}")

print("\n" + "="*60)
print("최적화 완료")
print("="*60)

print("\n핵심 발견:")
print(f"  - 각 상황마다 최적 TP/SL 다름")
print(f"  - 상황별 최적화로 평균 {comparison_df['improvement'].mean():.3f}%p 개선")
print(f"  - TP 범위: {summary_df['tp'].min():.1f}% ~ {summary_df['tp'].max():.1f}%")
print(f"  - SL 범위: {summary_df['sl'].min():.1f}% ~ {summary_df['sl'].max():.1f}%")

print("\n다음 단계:")
print("  1. MTF Zone 추출 (L/H 꼬리 범위)")
print("  2. 방향 필터 적용 (상황별 롱/숏 제한)")
print("  3. 통합 백테스트 (최적 파라미터 적용)")
