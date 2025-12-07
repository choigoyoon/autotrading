"""
Phase 5: 통계 분석
돌파 이벤트별 조건과 후속 수익률 상관관계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


def calculate_forward_returns(df, breakouts_df, periods=[10, 30, 60]):
    """
    돌파 후 N봉 수익률 계산

    Args:
        df: 원본 DataFrame
        breakouts_df: 돌파 DataFrame
        periods: 계산할 기간 리스트

    Returns:
        breakouts_df with return columns
    """
    breakouts_df = breakouts_df.copy()

    for period in periods:
        breakouts_df[f'return_{period}'] = 0.0
        breakouts_df[f'max_profit_{period}'] = 0.0
        breakouts_df[f'max_drawdown_{period}'] = 0.0

    for idx, row in breakouts_df.iterrows():
        break_idx = row['break_idx']
        break_price = row['break_price']
        breakout_type = row['type']

        # 방향 설정 (long/short)
        if breakout_type in ['trendline_up', 'hl_cross_long']:
            direction = 1  # long
        else:
            direction = -1  # short

        # 각 기간별 수익률 계산
        for period in periods:
            end_idx = min(break_idx + period, len(df) - 1)

            if end_idx > break_idx:
                # 최종 수익률
                final_price = df.iloc[end_idx]['close']
                final_return = (final_price - break_price) / break_price * 100 * direction

                # 구간 내 최대 수익/손실
                segment = df.iloc[break_idx:end_idx + 1]

                if direction == 1:  # long
                    max_price = segment['high'].max()
                    min_price = segment['low'].min()

                    max_profit = (max_price - break_price) / break_price * 100
                    max_drawdown = (min_price - break_price) / break_price * 100
                else:  # short
                    max_price = segment['high'].max()
                    min_price = segment['low'].min()

                    max_profit = (break_price - min_price) / break_price * 100
                    max_drawdown = (break_price - max_price) / break_price * 100

                breakouts_df.loc[idx, f'return_{period}'] = final_return
                breakouts_df.loc[idx, f'max_profit_{period}'] = max_profit
                breakouts_df.loc[idx, f'max_drawdown_{period}'] = max_drawdown

    return breakouts_df


def merge_breakout_conditions(df, breakouts_df, trendlines_df):
    """
    돌파 시점의 조건 정보 병합

    Args:
        df: 원본 DataFrame (다이버전스 포함)
        breakouts_df: 돌파 DataFrame
        trendlines_df: 추세선 DataFrame

    Returns:
        breakouts_df with condition columns
    """
    breakouts_df = breakouts_df.copy()

    # 추세선 정보 추가
    breakouts_df['trendline_slope'] = 0.0
    breakouts_df['trendline_duration'] = 0
    breakouts_df['trendline_touch_count'] = 0

    for idx, row in breakouts_df.iterrows():
        trendline_idx = row['trendline_idx']

        if trendline_idx >= 0 and trendline_idx < len(trendlines_df):
            tline = trendlines_df.iloc[trendline_idx]
            breakouts_df.loc[idx, 'trendline_slope'] = tline['slope']
            breakouts_df.loc[idx, 'trendline_duration'] = tline['duration']
            breakouts_df.loc[idx, 'trendline_touch_count'] = tline['touch_count']

    # 다이버전스 정보 추가
    breakouts_df['divergence_present'] = False
    breakouts_df['divergence_type'] = None
    breakouts_df['divergence_strength'] = 0

    for idx, row in breakouts_df.iterrows():
        break_idx = row['break_idx']

        # 돌파 시점의 다이버전스 정보 (최근 50 캔들 내)
        lookback_start = max(0, break_idx - 50)
        lookback_segment = df.iloc[lookback_start:break_idx + 1]

        # 다이버전스가 있는지 확인
        div_rows = lookback_segment[lookback_segment['divergence_type'].notna()]

        if len(div_rows) > 0:
            # 가장 최근 다이버전스 사용
            latest_div = div_rows.iloc[-1]
            breakouts_df.loc[idx, 'divergence_present'] = True
            breakouts_df.loc[idx, 'divergence_type'] = latest_div['divergence_type']
            breakouts_df.loc[idx, 'divergence_strength'] = latest_div['divergence_strength']

    return breakouts_df


def analyze_breakouts(df, trendlines_df, breakouts_df):
    """
    돌파 이벤트 통계 분석

    Args:
        df: 원본 DataFrame
        trendlines_df: 추세선 DataFrame
        breakouts_df: 돌파 DataFrame

    Returns:
        stats_dict
    """
    stats_dict = {}

    # 1. 후속 수익률 계산
    print("  후속 수익률 계산 중...")
    breakouts_df = calculate_forward_returns(df, breakouts_df, periods=[10, 30, 60])

    # 2. 조건 정보 병합
    print("  조건 정보 병합 중...")
    breakouts_df = merge_breakout_conditions(df, breakouts_df, trendlines_df)

    # 3. 승률 계산
    breakouts_df['win_10'] = breakouts_df['return_10'] > 0
    breakouts_df['win_30'] = breakouts_df['return_30'] > 0
    breakouts_df['win_60'] = breakouts_df['return_60'] > 0

    stats_dict['breakout_stats'] = breakouts_df

    # 4. 전체 통계
    overall_stats = {
        'total_breakouts': len(breakouts_df),
        'avg_return_10': breakouts_df['return_10'].mean(),
        'avg_return_30': breakouts_df['return_30'].mean(),
        'avg_return_60': breakouts_df['return_60'].mean(),
        'win_rate_10': breakouts_df['win_10'].mean() * 100,
        'win_rate_30': breakouts_df['win_30'].mean() * 100,
        'win_rate_60': breakouts_df['win_60'].mean() * 100,
    }
    stats_dict['overall_stats'] = overall_stats

    # 5. 유형별 성과
    print("  유형별 성과 분석 중...")
    type_performance = breakouts_df.groupby('type').agg({
        'return_10': 'mean',
        'return_30': 'mean',
        'return_60': 'mean',
        'win_10': 'mean',
        'win_30': 'mean',
        'win_60': 'mean',
    }).round(2)
    stats_dict['type_performance'] = type_performance

    # 6. 다이버전스 유무별 성과
    print("  다이버전스 영향 분석 중...")
    div_performance = breakouts_df.groupby('divergence_present').agg({
        'return_10': 'mean',
        'return_30': 'mean',
        'return_60': 'mean',
        'win_10': 'mean',
        'win_30': 'mean',
        'win_60': 'mean',
    }).round(2)
    stats_dict['divergence_performance'] = div_performance

    # 7. 지지/저항 확인 여부별 성과
    print("  지지/저항 확인 영향 분석 중...")
    support_performance = breakouts_df.groupby('support_confirmed').agg({
        'return_10': 'mean',
        'return_30': 'mean',
        'return_60': 'mean',
        'win_10': 'mean',
        'win_30': 'mean',
        'win_60': 'mean',
    }).round(2)
    stats_dict['support_performance'] = support_performance

    # 8. 상관관계 분석
    print("  상관관계 분석 중...")
    correlation_cols = [
        'trendline_slope', 'trendline_duration', 'trendline_touch_count',
        'candle_size', 'volume_ratio', 'divergence_strength',
        'return_10', 'return_30', 'return_60'
    ]

    correlation_matrix = breakouts_df[correlation_cols].corr()
    stats_dict['correlation_matrix'] = correlation_matrix

    # 9. 최고 성과 조건 찾기
    print("  최고 성과 조건 탐색 중...")

    # 30봉 수익률 상위 20% 조건
    top_20_pct = breakouts_df['return_30'].quantile(0.8)
    best_breakouts = breakouts_df[breakouts_df['return_30'] >= top_20_pct]

    best_conditions = {
        'top_20_pct_threshold': top_20_pct,
        'count': len(best_breakouts),
        'avg_return_30': best_breakouts['return_30'].mean(),
        'most_common_type': best_breakouts['type'].mode()[0] if len(best_breakouts) > 0 else None,
        'divergence_ratio': best_breakouts['divergence_present'].mean(),
        'support_confirmed_ratio': best_breakouts['support_confirmed'].mean(),
        'avg_volume_ratio': best_breakouts['volume_ratio'].mean(),
    }
    stats_dict['best_conditions'] = best_conditions

    return stats_dict


def visualize_stats(stats_dict, output_path="output_phase5_charts.png"):
    """
    통계 시각화

    Args:
        stats_dict: 통계 딕셔너리
        output_path: 차트 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Breakout Analysis Statistics', fontsize=16)

    # 1. 유형별 수익률 (30봉)
    type_perf = stats_dict['type_performance']
    type_perf['return_30'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Avg Return by Breakout Type (30 bars)')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. 유형별 승률
    (type_perf['win_30'] * 100).plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Win Rate by Breakout Type (30 bars)')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].axhline(y=50, color='r', linestyle='--', linewidth=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 다이버전스 영향
    div_perf = stats_dict['divergence_performance']
    div_perf['return_30'].plot(kind='bar', ax=axes[0, 2], color='salmon')
    axes[0, 2].set_title('Divergence Impact on Return')
    axes[0, 2].set_ylabel('Return (%)')
    axes[0, 2].set_xticklabels(['No Div', 'With Div'], rotation=0)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[0, 2].grid(axis='y', alpha=0.3)

    # 4. 지지/저항 확인 영향
    support_perf = stats_dict['support_performance']
    support_perf['return_30'].plot(kind='bar', ax=axes[1, 0], color='lightcoral')
    axes[1, 0].set_title('Support/Resistance Confirmation Impact')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].set_xticklabels(['Not Confirmed', 'Confirmed'], rotation=0)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 5. 수익률 분포 (30봉)
    breakout_stats = stats_dict['breakout_stats']
    axes[1, 1].hist(breakout_stats['return_30'], bins=50, color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Return Distribution (30 bars)')
    axes[1, 1].set_xlabel('Return (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # 6. 상관관계 히트맵 (주요 변수만)
    corr_matrix = stats_dict['correlation_matrix']
    key_vars = ['candle_size', 'volume_ratio', 'return_30']
    key_corr = corr_matrix.loc[key_vars, key_vars]

    sns.heatmap(key_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=axes[1, 2], cbar_kws={'label': 'Correlation'})
    axes[1, 2].set_title('Correlation Matrix (Key Variables)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n차트 저장: {output_path}")


if __name__ == "__main__":
    import glob

    # Phase 1-4 결과 로드
    if not all([
        glob.glob("output_phase1_labeled.csv"),
        glob.glob("output_phase2_trendlines.csv"),
        glob.glob("output_phase3_divergence.csv"),
        glob.glob("output_phase4_breakouts.csv")
    ]):
        print("Phase 1-4 실행 필요")
        exit(1)

    print("데이터 로드 중...")
    df = pd.read_csv("output_phase3_divergence.csv")  # 다이버전스 포함
    df['datetime'] = pd.to_datetime(df['datetime'])

    trendlines_df = pd.read_csv("output_phase2_trendlines.csv")
    breakouts_df = pd.read_csv("output_phase4_breakouts.csv")

    print(f"데이터: {len(df)} 캔들")
    print(f"추세선: {len(trendlines_df)}개")
    print(f"돌파: {len(breakouts_df)}개\n")

    # 통계 분석
    print("통계 분석 실행 중...")
    stats_dict = analyze_breakouts(df, trendlines_df, breakouts_df)

    # 결과 출력
    print("\n" + "=" * 80)
    print("전체 통계")
    print("=" * 80)
    for key, value in stats_dict['overall_stats'].items():
        print(f"{key}: {value:.2f}")

    print("\n" + "=" * 80)
    print("유형별 성과 (30봉)")
    print("=" * 80)
    print(stats_dict['type_performance'][['return_30', 'win_30']])

    print("\n" + "=" * 80)
    print("다이버전스 영향 (30봉)")
    print("=" * 80)
    print(stats_dict['divergence_performance'][['return_30', 'win_30']])

    print("\n" + "=" * 80)
    print("최고 성과 조건")
    print("=" * 80)
    for key, value in stats_dict['best_conditions'].items():
        print(f"{key}: {value}")

    # 시각화
    print("\n차트 생성 중...")
    visualize_stats(stats_dict)

    # 저장
    print("\n결과 저장 중...")
    stats_dict['breakout_stats'].to_csv("output_phase5_breakout_stats.csv", index=False)

    # JSON으로 요약 저장 (DataFrame 제외)
    summary = {
        'overall_stats': stats_dict['overall_stats'],
        'best_conditions': stats_dict['best_conditions'],
    }

    with open("output_phase5_stats_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("  output_phase5_breakout_stats.csv")
    print("  output_phase5_stats_summary.json")
    print("  output_phase5_charts.png")

    print("\n분석 완료!")
