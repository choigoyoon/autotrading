"""
전체 데이터 최적화 (고속 버전)
- 벡터화 연산으로 속도 개선
- 전체 10,104개 돌파 테스트
- 목표: 총 수익률 극대화
"""

import pandas as pd
import numpy as np
from itertools import product
import time

def fast_backtest(df, breakouts_df, params):
    """
    고속 백테스트 (벡터화)

    전략:
    1. 되돌림 범위 내 진입
    2. 고정 TP/SL
    """

    results = []

    for idx, breakout in breakouts_df.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        # 60봉 추출
        end_idx = min(break_idx + 60, len(df) - 1)
        if end_idx <= break_idx + 1:
            continue

        window = df.iloc[break_idx:end_idx+1].copy()
        break_price = window.iloc[0]['close']

        # 되돌림 계산 (벡터화)
        if direction == 'long':
            pullback = (window['close'] - break_price) / break_price * 100
        else:
            pullback = (break_price - window['close']) / break_price * 100

        # 진입 조건: 되돌림 범위 내
        entry_mask = (pullback >= params['pb_min']) & (pullback <= params['pb_max'])

        # 지지 확인: support_bars 봉 연속 상승
        support_bars = params['support_bars']
        if direction == 'long':
            # 연속 상승 체크
            rising = window['close'].diff() > 0
            rising_sum = rising.rolling(support_bars).sum()
            support_confirmed = rising_sum >= support_bars
        else:
            # 연속 하락 체크
            falling = window['close'].diff() < 0
            falling_sum = falling.rolling(support_bars).sum()
            support_confirmed = falling_sum >= support_bars

        # 진입 시점: 되돌림 범위 내 + 지지 확인
        entry_candidates = entry_mask & support_confirmed

        if not entry_candidates.any():
            continue

        # 첫 진입 시점
        entry_idx_local = entry_candidates.idxmax()
        entry_price = window.loc[entry_idx_local, 'close']

        # 진입 후 데이터
        after_entry = window.loc[entry_idx_local:]

        # 손익 계산 (벡터화)
        if direction == 'long':
            pl = (after_entry['close'] - entry_price) / entry_price * 100
            max_pl = (after_entry['high'] - entry_price) / entry_price * 100
            min_pl = (after_entry['low'] - entry_price) / entry_price * 100
        else:
            pl = (entry_price - after_entry['close']) / entry_price * 100
            max_pl = (entry_price - after_entry['low']) / entry_price * 100
            min_pl = (entry_price - after_entry['high']) / entry_price * 100

        # TP/SL 체크
        tp_hit = (max_pl >= params['tp']).any()
        sl_hit = (min_pl <= params['sl']).any()

        if tp_hit and sl_hit:
            # 둘 다 hit - 먼저 온 것
            tp_idx = (max_pl >= params['tp']).idxmax()
            sl_idx = (min_pl <= params['sl']).idxmax()

            if tp_idx <= sl_idx:
                exit_pl = params['tp']
                exit_reason = 'tp'
            else:
                exit_pl = params['sl']
                exit_reason = 'sl'
        elif tp_hit:
            exit_pl = params['tp']
            exit_reason = 'tp'
        elif sl_hit:
            exit_pl = params['sl']
            exit_reason = 'sl'
        else:
            # 타임아웃
            exit_pl = pl.iloc[-1]
            exit_reason = 'timeout'

        results.append({
            'pl': exit_pl,
            'reason': exit_reason
        })

    if len(results) == 0:
        return {
            'total_pl': 0,
            'avg_pl': 0,
            'win_rate': 0,
            'trades': 0
        }

    df_results = pd.DataFrame(results)

    return {
        'total_pl': df_results['pl'].sum(),
        'avg_pl': df_results['pl'].mean(),
        'win_rate': (df_results['pl'] > 0).sum() / len(df_results) * 100,
        'trades': len(df_results),
        'max_pl': df_results['pl'].max(),
        'min_pl': df_results['pl'].min()
    }


def optimize_parameters(df, breakouts_df):
    """파라미터 최적화 (전체 데이터)"""

    print(f"전체 돌파: {len(breakouts_df):,}개")

    # 파라미터 그리드 (축소)
    param_grid = {
        'pb_min': [-1.2, -1.0, -0.8, -0.6],
        'pb_max': [-0.6, -0.4, -0.2],
        'support_bars': [2, 3, 5],
        'tp': [1.5, 2.0, 2.5],
        'sl': [-0.6, -0.8, -1.0]
    }

    # 조합 생성
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))

    # 유효한 조합만 (pb_min <= pb_max)
    valid_combos = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        if params['pb_min'] <= params['pb_max']:
            valid_combos.append(params)

    print(f"유효 조합: {len(valid_combos):,}개\n")

    # 백테스트
    all_results = []
    start_time = time.time()

    for i, params in enumerate(valid_combos, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            if i > 1:
                per_combo = elapsed / (i - 1)
                remaining = per_combo * (len(valid_combos) - i + 1)
                print(f"진행: {i}/{len(valid_combos)} ({i/len(valid_combos)*100:.1f}%) | "
                      f"경과: {elapsed:.0f}초 | 예상 남은 시간: {remaining:.0f}초")
            else:
                print(f"진행: {i}/{len(valid_combos)} ({i/len(valid_combos)*100:.1f}%)")

        result = fast_backtest(df, breakouts_df, params)
        result['params'] = params
        all_results.append(result)

    # 정렬 (총 수익률 기준)
    all_results.sort(key=lambda x: x['total_pl'], reverse=True)

    total_time = time.time() - start_time
    print(f"\n총 소요 시간: {total_time:.0f}초 ({total_time/60:.1f}분)")

    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("전체 데이터 파라미터 최적화 (고속)")
    print("=" * 60)
    print()

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")
    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    print(f"데이터: {len(df):,}봉")
    print(f"추세선 돌파: {len(trendline_breakouts):,}개\n")

    # 최적화 실행
    results = optimize_parameters(df, trendline_breakouts)

    # 상위 결과 출력
    print("\n" + "=" * 60)
    print("상위 10개 파라미터 조합")
    print("=" * 60)

    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}위]")
        print(f"  총 수익률: {result['total_pl']:.2f}%")
        print(f"  평균 수익률: {result['avg_pl']:.3f}%")
        print(f"  승률: {result['win_rate']:.1f}%")
        print(f"  거래 횟수: {result['trades']}회")
        print(f"  파라미터:")
        for k, v in result['params'].items():
            print(f"    {k}: {v}")

    # 저장
    results_df = pd.DataFrame([{
        'rank': i+1,
        'total_pl': r['total_pl'],
        'avg_pl': r['avg_pl'],
        'win_rate': r['win_rate'],
        'trades': r['trades'],
        'max_pl': r['max_pl'],
        'min_pl': r['min_pl'],
        **r['params']
    } for i, r in enumerate(results)])

    results_df.to_csv("output/optimization_full_results.csv", index=False)

    print("\n저장: output/optimization_full_results.csv")

    # 최적 파라미터
    best = results[0]
    print("\n" + "=" * 60)
    print("🏆 최적 파라미터")
    print("=" * 60)
    print(f"\n총 수익률: {best['total_pl']:.2f}%")
    print(f"평균 수익률: {best['avg_pl']:.3f}%")
    print(f"승률: {best['win_rate']:.1f}%")
    print(f"거래: {best['trades']}회")
    print(f"\n파라미터:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
