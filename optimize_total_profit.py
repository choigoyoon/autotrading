"""
총 수익률 극대화 파라미터 최적화

전략:
1. 진입: 되돌림 완료 후 지지 확인 (L값 근접)
2. 익절: 최대 반등 포착 (trailing 또는 다음 H)
3. 목표: 총 수익률 극대화
"""

import pandas as pd
import numpy as np
from itertools import product

def backtest_strategy(df, labeled_df, breakouts_df, params):
    """
    파라미터 조합 백테스트

    params:
        pullback_entry_min: 되돌림 진입 최소 (예: -0.6)
        pullback_entry_max: 되돌림 진입 최대 (예: -1.0)
        support_bars: 지지 확인 봉 수 (예: 3)
        tp_method: 'fixed' or 'trailing' or 'next_h'
        tp_fixed: 고정 익절 % (예: 2.0)
        trailing_start: trailing 시작 % (예: 1.0)
        trailing_offset: trailing 오프셋 % (예: 0.5)
        stop_loss: 손절 % (예: -0.5)
    """

    # H/L 캐시
    h_points = labeled_df[labeled_df['label'] == 'H'].copy()
    l_points = labeled_df[labeled_df['label'] == 'L'].copy()

    results = []

    for idx, breakout in breakouts_df.iterrows():
        break_idx = breakout['break_idx']
        break_type = breakout['type']
        direction = 'long' if break_type == 'trendline_up' else 'short'

        # 진입 대기
        entry_idx = None
        entry_price = None

        # 60봉 관찰
        end_idx = min(break_idx + 60, len(df) - 1)

        for i in range(break_idx + 1, end_idx + 1):
            bar_num = i - break_idx
            current = df.iloc[i]['close']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            break_price = df.iloc[break_idx]['close']

            # 진입 전: 되돌림 확인
            if entry_idx is None:
                if direction == 'long':
                    pullback = (current - break_price) / break_price * 100
                else:
                    pullback = (break_price - current) / break_price * 100

                # 되돌림 범위 내?
                if params['pullback_entry_min'] <= pullback <= params['pullback_entry_max']:
                    # 지지 확인 (연속 상승)
                    if i >= break_idx + params['support_bars']:
                        support_confirmed = True

                        for j in range(1, params['support_bars'] + 1):
                            if direction == 'long':
                                if df.iloc[i - j + 1]['close'] <= df.iloc[i - j]['close']:
                                    support_confirmed = False
                                    break
                            else:
                                if df.iloc[i - j + 1]['close'] >= df.iloc[i - j]['close']:
                                    support_confirmed = False
                                    break

                        if support_confirmed:
                            entry_idx = i
                            entry_price = current
                            max_favorable = 0
                            continue

            # 진입 후: 포지션 관리
            if entry_idx is not None:
                if direction == 'long':
                    pl = (current - entry_price) / entry_price * 100
                    max_pl = (high - entry_price) / entry_price * 100
                    min_pl = (low - entry_price) / entry_price * 100
                else:
                    pl = (entry_price - current) / entry_price * 100
                    max_pl = (entry_price - low) / entry_price * 100
                    min_pl = (entry_price - high) / entry_price * 100

                max_favorable = max(max_favorable, max_pl)

                # 손절
                if min_pl <= params['stop_loss']:
                    exit_price = entry_price * (1 + params['stop_loss']/100 if direction == 'long' else 1 - params['stop_loss']/100)
                    exit_reason = 'stop_loss'
                    break

                # 익절 로직
                if params['tp_method'] == 'fixed':
                    # 고정 익절
                    if max_pl >= params['tp_fixed']:
                        exit_price = entry_price * (1 + params['tp_fixed']/100 if direction == 'long' else 1 - params['tp_fixed']/100)
                        exit_reason = f"tp_fixed_{params['tp_fixed']}"
                        break

                elif params['tp_method'] == 'trailing':
                    # Trailing stop
                    if max_favorable >= params['trailing_start']:
                        # trailing 활성화
                        if max_favorable - pl >= params['trailing_offset']:
                            exit_price = current
                            exit_reason = f"trailing_{max_favorable:.2f}"
                            break

                elif params['tp_method'] == 'next_h':
                    # 다음 H/L 레벨
                    if direction == 'long':
                        next_h = h_points[h_points.index > entry_idx]
                        if len(next_h) > 0:
                            next_h_price = next_h.iloc[0]['label_price']
                            if high >= next_h_price:
                                exit_price = next_h_price
                                exit_reason = 'next_h'
                                break
                    else:
                        next_l = l_points[l_points.index > entry_idx]
                        if len(next_l) > 0:
                            next_l_price = next_l.iloc[0]['label_price']
                            if low <= next_l_price:
                                exit_price = next_l_price
                                exit_reason = 'next_l'
                                break

        # 60봉 타임아웃
        if entry_idx is not None:
            if 'exit_price' not in locals() or exit_price is None:
                exit_price = df.iloc[end_idx]['close']
                exit_reason = 'timeout'

                if direction == 'long':
                    pl = (exit_price - entry_price) / entry_price * 100
                else:
                    pl = (entry_price - exit_price) / entry_price * 100
        else:
            # 진입 못함
            pl = 0
            exit_reason = 'no_entry'

        # 실현 손익
        if entry_idx is not None and 'pl' in locals():
            results.append({
                'break_idx': break_idx,
                'entry_idx': entry_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pl': pl,
                'exit_reason': exit_reason
            })

    if len(results) == 0:
        return {
            'total_pl': 0,
            'avg_pl': 0,
            'win_rate': 0,
            'trades': 0,
            'params': params
        }

    results_df = pd.DataFrame(results)

    return {
        'total_pl': results_df['pl'].sum(),
        'avg_pl': results_df['pl'].mean(),
        'win_rate': (results_df['pl'] > 0).sum() / len(results_df) * 100,
        'trades': len(results_df),
        'max_pl': results_df['pl'].max(),
        'min_pl': results_df['pl'].min(),
        'params': params,
        'results': results_df
    }


def grid_search(df, labeled_df, breakouts_df, sample_size=1000):
    """파라미터 그리드 서치"""

    # 샘플링 (빠른 테스트)
    if sample_size:
        breakouts_sample = breakouts_df.head(sample_size)
    else:
        breakouts_sample = breakouts_df

    print(f"샘플 크기: {len(breakouts_sample):,}개")

    # 파라미터 그리드
    param_grid = {
        'pullback_entry_min': [-1.2, -1.0, -0.8, -0.6],
        'pullback_entry_max': [-0.8, -0.6, -0.4, -0.2],
        'support_bars': [2, 3, 5],
        'tp_method': ['fixed', 'trailing'],
        'tp_fixed': [1.5, 2.0, 2.5],
        'trailing_start': [1.0, 1.5],
        'trailing_offset': [0.3, 0.5, 0.7],
        'stop_loss': [-0.4, -0.6, -0.8]
    }

    # 조합 생성
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))

    print(f"총 조합: {len(combinations):,}개")

    # 유효한 조합만 필터 (pullback_min <= pullback_max)
    valid_combinations = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        if params['pullback_entry_min'] <= params['pullback_entry_max']:
            valid_combinations.append(params)

    print(f"유효 조합: {len(valid_combinations):,}개\n")

    # 백테스트
    all_results = []

    for i, params in enumerate(valid_combinations, 1):
        if i % 100 == 0 or i == 1:
            print(f"진행: {i}/{len(valid_combinations)} ({i/len(valid_combinations)*100:.1f}%)")
        result = backtest_strategy(df, labeled_df, breakouts_sample, params)
        all_results.append(result)

    # 결과 정렬 (총 수익률 기준)
    all_results.sort(key=lambda x: x['total_pl'], reverse=True)

    return all_results


if __name__ == "__main__":
    print("=" * 60)
    print("총 수익률 극대화 파라미터 최적화")
    print("=" * 60)
    print()

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    labeled = df[df['label'].notna()].copy()

    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")
    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    print(f"추세선 돌파: {len(trendline_breakouts):,}개")
    print(f"H/L 포인트: {len(labeled):,}개\n")

    # 그리드 서치 (샘플)
    results = grid_search(df, labeled, trendline_breakouts, sample_size=2000)

    print("\n" + "=" * 60)
    print("상위 10개 파라미터 조합")
    print("=" * 60)

    for i, result in enumerate(results[:10], 1):
        print(f"\n[{i}위]")
        print(f"  총 수익률: {result['total_pl']:.2f}%")
        print(f"  평균 수익률: {result['avg_pl']:.3f}%")
        print(f"  승률: {result['win_rate']:.1f}%")
        print(f"  거래 횟수: {result['trades']}회")
        print(f"  최대 수익: {result['max_pl']:.2f}%")
        print(f"  최대 손실: {result['min_pl']:.2f}%")
        print(f"  파라미터:")
        for k, v in result['params'].items():
            print(f"    {k}: {v}")

    # 최적 파라미터로 전체 백테스트
    print("\n" + "=" * 60)
    print("최적 파라미터로 전체 백테스트 실행 중...")
    print("=" * 60)

    best_params = results[0]['params']
    final_result = backtest_strategy(df, labeled, trendline_breakouts, best_params)

    print(f"\n[전체 결과]")
    print(f"총 수익률: {final_result['total_pl']:.2f}%")
    print(f"평균 수익률: {final_result['avg_pl']:.3f}%")
    print(f"승률: {final_result['win_rate']:.1f}%")
    print(f"거래 횟수: {final_result['trades']}회")

    # 저장
    results_summary = pd.DataFrame([{
        'rank': i+1,
        'total_pl': r['total_pl'],
        'avg_pl': r['avg_pl'],
        'win_rate': r['win_rate'],
        'trades': r['trades'],
        **r['params']
    } for i, r in enumerate(results)])

    results_summary.to_csv("output/optimization_results.csv", index=False)
    final_result['results'].to_csv("output/best_strategy_trades.csv", index=False)

    print("\n저장 완료:")
    print("  - output/optimization_results.csv: 파라미터 조합 결과")
    print("  - output/best_strategy_trades.csv: 최적 전략 거래 내역")
