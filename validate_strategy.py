"""
전략 검증 분석
- L값 진입 근거 분석
- 조건 교집합 분석
- 필터 효과성 검증
- 수익률 검증
"""

import pandas as pd
import numpy as np

def analyze_l_basis(df, labeled_df, breakouts_df):
    """L값 진입 근거 분석"""

    print("=" * 60)
    print("1. L값 진입 근거 분석")
    print("=" * 60)

    # L 포인트 추출
    l_points = labeled_df[labeled_df['label'] == 'L'].copy()
    h_points = labeled_df[labeled_df['label'] == 'H'].copy()

    print(f"\n전체 L 포인트: {len(l_points):,}개")
    print(f"전체 H 포인트: {len(h_points):,}개")

    # L 포인트에서 60봉 후 수익률
    l_returns = []
    for idx in l_points.index:
        if idx + 60 < len(df):
            l_price = df.loc[idx, 'close']
            window = df.iloc[idx:idx+61]

            # 최대 수익
            max_high = window['high'].max()
            max_profit = (max_high - l_price) / l_price * 100

            # 최대 손실
            max_low = window['low'].min()
            max_dd = (max_low - l_price) / l_price * 100

            # 60봉 후 종가
            final_price = window.iloc[-1]['close']
            final_return = (final_price - l_price) / l_price * 100

            l_returns.append({
                'max_profit': max_profit,
                'max_dd': max_dd,
                'final_return': final_return
            })

    l_df = pd.DataFrame(l_returns)

    print(f"\nL 포인트 진입 결과 (60봉 관찰):")
    print(f"  평균 최대 수익: {l_df['max_profit'].mean():.3f}%")
    print(f"  평균 최대 손실: {l_df['max_dd'].mean():.3f}%")
    print(f"  평균 최종 수익: {l_df['final_return'].mean():.3f}%")
    print(f"  +1% 도달률: {(l_df['max_profit'] >= 1.0).sum() / len(l_df) * 100:.1f}%")
    print(f"  +2% 도달률: {(l_df['max_profit'] >= 2.0).sum() / len(l_df) * 100:.1f}%")
    print(f"  최종 수익률 >0%: {(l_df['final_return'] > 0).sum() / len(l_df) * 100:.1f}%")

    # H 포인트 비교 (숏)
    h_returns = []
    for idx in h_points.index:
        if idx + 60 < len(df):
            h_price = df.loc[idx, 'close']
            window = df.iloc[idx:idx+61]

            # 최대 수익 (숏)
            max_low = window['low'].min()
            max_profit = (h_price - max_low) / h_price * 100

            # 최대 손실 (숏)
            max_high = window['high'].max()
            max_dd = (h_price - max_high) / h_price * 100

            # 60봉 후 종가
            final_price = window.iloc[-1]['close']
            final_return = (h_price - final_price) / h_price * 100

            h_returns.append({
                'max_profit': max_profit,
                'max_dd': max_dd,
                'final_return': final_return
            })

    h_df = pd.DataFrame(h_returns)

    print(f"\nH 포인트 진입 (숏) 결과 (60봉 관찰):")
    print(f"  평균 최대 수익: {h_df['max_profit'].mean():.3f}%")
    print(f"  평균 최대 손실: {h_df['max_dd'].mean():.3f}%")
    print(f"  평균 최종 수익: {h_df['final_return'].mean():.3f}%")
    print(f"  +1% 도달률: {(h_df['max_profit'] >= 1.0).sum() / len(h_df) * 100:.1f}%")
    print(f"  +2% 도달률: {(h_df['max_profit'] >= 2.0).sum() / len(h_df) * 100:.1f}%")

    # 랜덤 포인트 비교
    np.random.seed(42)
    random_indices = np.random.choice(df.index[:-60], size=min(1000, len(l_points)), replace=False)

    random_returns = []
    for idx in random_indices:
        rand_price = df.loc[idx, 'close']
        window = df.iloc[idx:idx+61]

        max_high = window['high'].max()
        max_profit = (max_high - rand_price) / rand_price * 100

        max_low = window['low'].min()
        max_dd = (max_low - rand_price) / rand_price * 100

        final_price = window.iloc[-1]['close']
        final_return = (final_price - rand_price) / rand_price * 100

        random_returns.append({
            'max_profit': max_profit,
            'max_dd': max_dd,
            'final_return': final_return
        })

    random_df = pd.DataFrame(random_returns)

    print(f"\n랜덤 포인트 진입 결과 (60봉 관찰):")
    print(f"  평균 최대 수익: {random_df['max_profit'].mean():.3f}%")
    print(f"  평균 최대 손실: {random_df['max_dd'].mean():.3f}%")
    print(f"  평균 최종 수익: {random_df['final_return'].mean():.3f}%")
    print(f"  +2% 도달률: {(random_df['max_profit'] >= 2.0).sum() / len(random_df) * 100:.1f}%")

    print(f"\n결론:")
    print(f"  L vs 랜덤 최대 수익: {l_df['max_profit'].mean():.3f}% vs {random_df['max_profit'].mean():.3f}% (차이: {l_df['max_profit'].mean() - random_df['max_profit'].mean():.3f}%)")
    print(f"  L vs 랜덤 최종 수익: {l_df['final_return'].mean():.3f}% vs {random_df['final_return'].mean():.3f}% (차이: {l_df['final_return'].mean() - random_df['final_return'].mean():.3f}%)")

    return l_df, h_df, random_df


def analyze_trendline_filter(df, breakouts_df):
    """추세선 필터 효과 분석"""

    print("\n" + "=" * 60)
    print("2. 추세선 필터 효과 분석")
    print("=" * 60)

    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    print(f"\n전체 캔들: {len(df):,}개")
    print(f"추세선 돌파: {len(trendline_breakouts):,}개")
    print(f"필터율: {(1 - len(trendline_breakouts) / len(df)) * 100:.2f}% 제거")
    print(f"통과율: {len(trendline_breakouts) / len(df) * 100:.2f}%")

    # 추세선 돌파 수익률
    breakout_returns = []
    for idx, breakout in trendline_breakouts.iterrows():
        break_idx = breakout['break_idx']
        if break_idx + 60 < len(df):
            break_price = df.iloc[break_idx]['close']
            window = df.iloc[break_idx:break_idx+61]

            direction = 'long' if breakout['type'] == 'trendline_up' else 'short'

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

            breakout_returns.append({
                'max_profit': max_profit,
                'final_return': final_return
            })

    breakout_df = pd.DataFrame(breakout_returns)

    print(f"\n추세선 돌파 진입 결과:")
    print(f"  평균 최대 수익: {breakout_df['max_profit'].mean():.3f}%")
    print(f"  평균 최종 수익: {breakout_df['final_return'].mean():.3f}%")
    print(f"  +2% 도달률: {(breakout_df['max_profit'] >= 2.0).sum() / len(breakout_df) * 100:.1f}%")
    print(f"  최종 수익률 >0%: {(breakout_df['final_return'] > 0).sum() / len(breakout_df) * 100:.1f}%")

    return breakout_df


def analyze_pullback_intersection(df, breakouts_df):
    """되돌림 조건 교집합 분석"""

    print("\n" + "=" * 60)
    print("3. 되돌림 조건 교집합 분석")
    print("=" * 60)

    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    # 되돌림 범위별 분석
    pullback_ranges = [
        (0.0, -0.3, "0-0.3%"),
        (-0.3, -0.5, "0.3-0.5%"),
        (-0.5, -0.8, "0.5-0.8%"),
        (-0.8, -1.0, "0.8-1.0%"),
        (-1.0, -1.5, "1.0-1.5%"),
    ]

    results = []

    for pb_max, pb_min, label in pullback_ranges:
        trades = []

        for idx, breakout in trendline_breakouts.iterrows():
            break_idx = breakout['break_idx']
            break_type = breakout['type']
            direction = 'long' if break_type == 'trendline_up' else 'short'

            end_idx = min(break_idx + 60, len(df) - 1)
            if end_idx <= break_idx + 1:
                continue

            window = df.iloc[break_idx:end_idx+1].copy()
            break_price = window.iloc[0]['close']

            # 되돌림 계산
            if direction == 'long':
                pullback = (window['close'] - break_price) / break_price * 100
            else:
                pullback = (break_price - window['close']) / break_price * 100

            # 되돌림 범위 체크
            entry_mask = (pullback >= pb_min) & (pullback <= pb_max)

            if not entry_mask.any():
                continue

            # 지지 확인 (2봉)
            rising = window['close'].diff() > 0 if direction == 'long' else window['close'].diff() < 0
            rising_sum = rising.rolling(2).sum()
            support_confirmed = rising_sum >= 2

            entry_candidates = entry_mask & support_confirmed

            if not entry_candidates.any():
                continue

            entry_idx_local = entry_candidates.idxmax()
            entry_price = window.loc[entry_idx_local, 'close']

            after_entry = window.loc[entry_idx_local:]

            if direction == 'long':
                max_pl = (after_entry['high'] - entry_price) / entry_price * 100
                final_pl = (after_entry.iloc[-1]['close'] - entry_price) / entry_price * 100
            else:
                max_pl = (entry_price - after_entry['low']) / entry_price * 100
                final_pl = (entry_price - after_entry.iloc[-1]['close']) / entry_price * 100

            trades.append({
                'max_pl': max_pl.max(),
                'final_pl': final_pl
            })

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            results.append({
                'range': label,
                'count': len(trades),
                'avg_max': trades_df['max_pl'].mean(),
                'avg_final': trades_df['final_pl'].mean(),
                'win_rate': (trades_df['final_pl'] > 0).sum() / len(trades) * 100,
                'reach_1pct': (trades_df['max_pl'] >= 1.0).sum() / len(trades) * 100,
                'reach_2pct': (trades_df['max_pl'] >= 2.0).sum() / len(trades) * 100,
            })

    if len(results) > 0:
        results_df = pd.DataFrame(results)

        print(f"\n되돌림 범위별 성과:")
        print(results_df.to_string(index=False))

        # 최적 범위 찾기
        best_idx = results_df['avg_final'].idxmax()
        best = results_df.iloc[best_idx]

        print(f"\n최적 되돌림 범위: {best['range']}")
        print(f"  거래 수: {best['count']:.0f}회")
        print(f"  평균 최종 수익: {best['avg_final']:.3f}%")
        print(f"  승률: {best['win_rate']:.1f}%")
        print(f"  +2% 도달률: {best['reach_2pct']:.1f}%")

        return results_df
    else:
        print("\n경고: 조건을 만족하는 거래가 없습니다!")
        return pd.DataFrame()


def narrow_optimization(df, breakouts_df):
    """좁은 범위 재최적화 (0-0.5% 집중)"""

    print("\n" + "=" * 60)
    print("4. 좁은 범위 재최적화")
    print("=" * 60)

    from itertools import product

    trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

    # 좁은 파라미터 그리드
    param_grid = {
        'pb_min': [-0.3, -0.4, -0.5],
        'pb_max': [0.0, -0.1, -0.2],
        'support_bars': [2, 3],
        'tp': [1.5, 2.0, 2.5],
        'sl': [-0.6, -0.8]
    }

    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))
    valid_combos = [dict(zip(keys, c)) for c in combinations if c[0] <= c[1]]

    print(f"\n유효 조합: {len(valid_combos)}개")
    print("최적화 실행 중...\n")

    all_results = []

    for i, params in enumerate(valid_combos, 1):
        if i % 10 == 0:
            print(f"진행: {i}/{len(valid_combos)}")

        trades = []

        for idx, breakout in trendline_breakouts.iterrows():
            break_idx = breakout['break_idx']
            break_type = breakout['type']
            direction = 'long' if break_type == 'trendline_up' else 'short'

            end_idx = min(break_idx + 60, len(df) - 1)
            if end_idx <= break_idx + 1:
                continue

            window = df.iloc[break_idx:end_idx+1].copy()
            break_price = window.iloc[0]['close']

            # 되돌림
            if direction == 'long':
                pullback = (window['close'] - break_price) / break_price * 100
            else:
                pullback = (break_price - window['close']) / break_price * 100

            entry_mask = (pullback >= params['pb_min']) & (pullback <= params['pb_max'])

            # 지지
            support_bars = params['support_bars']
            if direction == 'long':
                rising = window['close'].diff() > 0
                rising_sum = rising.rolling(support_bars).sum()
                support_confirmed = rising_sum >= support_bars
            else:
                falling = window['close'].diff() < 0
                falling_sum = falling.rolling(support_bars).sum()
                support_confirmed = falling_sum >= support_bars

            entry_candidates = entry_mask & support_confirmed

            if not entry_candidates.any():
                continue

            entry_idx_local = entry_candidates.idxmax()
            entry_price = window.loc[entry_idx_local, 'close']

            after_entry = window.loc[entry_idx_local:]

            # 손익
            if direction == 'long':
                pl = (after_entry['close'] - entry_price) / entry_price * 100
                max_pl = (after_entry['high'] - entry_price) / entry_price * 100
                min_pl = (after_entry['low'] - entry_price) / entry_price * 100
            else:
                pl = (entry_price - after_entry['close']) / entry_price * 100
                max_pl = (entry_price - after_entry['low']) / entry_price * 100
                min_pl = (entry_price - after_entry['high']) / entry_price * 100

            # TP/SL
            tp_hit = (max_pl >= params['tp']).any()
            sl_hit = (min_pl <= params['sl']).any()

            if tp_hit and sl_hit:
                tp_idx = (max_pl >= params['tp']).idxmax()
                sl_idx = (min_pl <= params['sl']).idxmax()
                exit_pl = params['tp'] if tp_idx <= sl_idx else params['sl']
            elif tp_hit:
                exit_pl = params['tp']
            elif sl_hit:
                exit_pl = params['sl']
            else:
                exit_pl = pl.iloc[-1]

            trades.append({'pl': exit_pl})

        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            all_results.append({
                'params': params,
                'total_pl': trades_df['pl'].sum(),
                'avg_pl': trades_df['pl'].mean(),
                'win_rate': (trades_df['pl'] > 0).sum() / len(trades_df) * 100,
                'trades': len(trades_df)
            })

    # 정렬
    all_results.sort(key=lambda x: x['avg_pl'], reverse=True)

    print("\n상위 5개 결과:")
    for i, r in enumerate(all_results[:5], 1):
        print(f"\n[{i}위]")
        print(f"  평균 수익: {r['avg_pl']:.3f}%")
        print(f"  슬리피지 후: {r['avg_pl'] - 0.2:.3f}%")
        print(f"  승률: {r['win_rate']:.1f}%")
        print(f"  거래: {r['trades']}회")
        print(f"  파라미터: {r['params']}")

    return all_results


if __name__ == "__main__":
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    labeled = df[df['label'].notna()].copy()

    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")

    print(f"데이터: {len(df):,}봉")
    print(f"H/L 포인트: {len(labeled):,}개")
    print(f"돌파: {len(breakouts_df):,}개\n")

    # 1. L값 근거 분석
    l_df, h_df, random_df = analyze_l_basis(df, labeled, breakouts_df)

    # 2. 추세선 필터 효과
    breakout_df = analyze_trendline_filter(df, breakouts_df)

    # 3. 되돌림 교집합 분석
    pullback_df = analyze_pullback_intersection(df, breakouts_df)

    # 4. 좁은 범위 재최적화
    narrow_results = narrow_optimization(df, breakouts_df)

    print("\n" + "=" * 60)
    print("검증 완료")
    print("=" * 60)
