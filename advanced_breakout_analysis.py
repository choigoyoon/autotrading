"""
고급 돌파 분석
1. 다이버전스 중첩도 (Divergence Stacking)
2. 다양한 승률 기준 비교
3. 단기 vs 장기 신뢰도 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_divergence_stacking(df, breakouts_df, lookback_windows=[10, 20, 50]):
    """
    돌파 시점의 다이버전스 중첩도 계산

    Args:
        df: 원본 DataFrame (divergence 포함)
        breakouts_df: 돌파 DataFrame
        lookback_windows: 확인할 과거 윈도우 크기 리스트

    Returns:
        breakouts_df with stacking columns
    """
    print("\n다이버전스 중첩도 계산 중...")

    breakouts_df = breakouts_df.copy()

    for window in lookback_windows:
        breakouts_df[f'div_count_{window}'] = 0
        breakouts_df[f'div_types_{window}'] = None

    for idx, row in breakouts_df.iterrows():
        break_idx = row['break_idx']

        for window in lookback_windows:
            lookback_start = max(0, break_idx - window)
            lookback_segment = df.iloc[lookback_start:break_idx + 1]

            # 다이버전스가 있는 캔들들
            div_rows = lookback_segment[lookback_segment['divergence_type'].notna()]

            if len(div_rows) > 0:
                # 다이버전스 개수
                breakouts_df.loc[idx, f'div_count_{window}'] = len(div_rows)

                # 다이버전스 유형들 (중복 제거)
                div_types = div_rows['divergence_type'].unique()
                breakouts_df.loc[idx, f'div_types_{window}'] = ','.join(div_types)

    return breakouts_df


def analyze_win_rates_by_criteria(breakouts_df):
    """
    다양한 기준으로 승률 계산

    Args:
        breakouts_df: 돌파 DataFrame

    Returns:
        win_rate_analysis DataFrame
    """
    print("\n다양한 승률 기준 분석 중...")

    criteria = {
        'positive': lambda x: x > 0,
        'above_0.5pct': lambda x: x > 0.5,
        'above_1pct': lambda x: x > 1.0,
        'above_1.5pct': lambda x: x > 1.5,
        'above_2pct': lambda x: x > 2.0,
    }

    results = []

    for period in [10, 30, 60]:
        return_col = f'return_{period}'

        period_results = {
            'period': f'{period}봉',
            'sample_size': len(breakouts_df)
        }

        for criterion_name, criterion_func in criteria.items():
            win_count = breakouts_df[return_col].apply(criterion_func).sum()
            win_rate = win_count / len(breakouts_df) * 100

            period_results[criterion_name] = win_rate

        # 평균 수익률
        period_results['avg_return'] = breakouts_df[return_col].mean()

        results.append(period_results)

    results_df = pd.DataFrame(results)

    print("\n승률 기준별 비교:")
    print(results_df.to_string(index=False))

    return results_df


def calculate_breakout_quality_score(breakouts_df):
    """
    돌파 품질 점수 계산 (단기 + 장기 통합)

    점수 구성:
    - 단기 성과 (10봉): 30%
    - 중기 성과 (30봉): 40%
    - 장기 성과 (60봉): 30%
    - 승률 보정
    - 리스크 조정

    Args:
        breakouts_df: 돌파 DataFrame

    Returns:
        breakouts_df with quality_score
    """
    print("\n돌파 품질 점수 계산 중...")

    breakouts_df = breakouts_df.copy()

    # 1. 각 기간별 정규화된 점수 (0-100)
    for period, weight in [(10, 0.3), (30, 0.4), (60, 0.3)]:
        return_col = f'return_{period}'

        # 수익률을 0-100 스케일로 정규화 (3%를 100점으로 가정)
        breakouts_df[f'score_{period}'] = breakouts_df[return_col].clip(-3, 3).apply(
            lambda x: (x + 3) / 6 * 100
        )

    # 2. 가중 평균 점수
    breakouts_df['quality_score'] = (
        breakouts_df['score_10'] * 0.3 +
        breakouts_df['score_30'] * 0.4 +
        breakouts_df['score_60'] * 0.3
    )

    # 3. 리스크 조정 (max drawdown 고려)
    # max_drawdown이 큰 경우 감점
    for period in [10, 30, 60]:
        dd_col = f'max_drawdown_{period}'
        if dd_col in breakouts_df.columns:
            # 최대 손실 -3% 이상이면 감점
            penalty = breakouts_df[dd_col].clip(-3, 0).apply(lambda x: abs(x) * 5)
            breakouts_df['quality_score'] -= penalty * (0.3 if period == 10 else 0.4 if period == 30 else 0.3)

    # 4. 0-100 범위로 재조정
    breakouts_df['quality_score'] = breakouts_df['quality_score'].clip(0, 100)

    # 5. 등급 부여
    def assign_grade(score):
        if score >= 80:
            return 'S (우수)'
        elif score >= 70:
            return 'A (양호)'
        elif score >= 60:
            return 'B (보통)'
        elif score >= 50:
            return 'C (평범)'
        else:
            return 'D (위험)'

    breakouts_df['quality_grade'] = breakouts_df['quality_score'].apply(assign_grade)

    print(f"\n품질 점수 통계:")
    print(f"  평균: {breakouts_df['quality_score'].mean():.1f}")
    print(f"  중앙값: {breakouts_df['quality_score'].median():.1f}")
    print(f"  표준편차: {breakouts_df['quality_score'].std():.1f}")

    print("\n등급별 분포:")
    print(breakouts_df['quality_grade'].value_counts().sort_index())

    return breakouts_df


def analyze_timeframe_reliability(breakouts_df):
    """
    단기 vs 장기 신뢰도 분석

    Args:
        breakouts_df: 돌파 DataFrame

    Returns:
        reliability_analysis dict
    """
    print("\n" + "="*80)
    print("단기 vs 장기 신뢰도 분석")
    print("="*80)

    analysis = {}

    # 1. 기간별 성과 비교
    print("\n1. 기간별 평균 성과:")
    print("-"*80)

    timeframe_stats = []
    for period in [10, 30, 60]:
        stats = {
            'period': f'{period}봉',
            'avg_return': breakouts_df[f'return_{period}'].mean(),
            'win_rate_positive': (breakouts_df[f'return_{period}'] > 0).mean() * 100,
            'win_rate_1pct': (breakouts_df[f'return_{period}'] > 1.0).mean() * 100,
            'avg_max_profit': breakouts_df[f'max_profit_{period}'].mean(),
            'avg_max_dd': breakouts_df[f'max_drawdown_{period}'].mean(),
        }
        timeframe_stats.append(stats)

    timeframe_df = pd.DataFrame(timeframe_stats)
    print(timeframe_df.to_string(index=False))

    analysis['timeframe_stats'] = timeframe_df

    # 2. 일관성 분석 (모든 기간에서 수익인 케이스)
    print("\n2. 일관성 분석:")
    print("-"*80)

    consistent_winners = breakouts_df[
        (breakouts_df['return_10'] > 0) &
        (breakouts_df['return_30'] > 0) &
        (breakouts_df['return_60'] > 0)
    ]

    consistent_losers = breakouts_df[
        (breakouts_df['return_10'] < 0) &
        (breakouts_df['return_30'] < 0) &
        (breakouts_df['return_60'] < 0)
    ]

    print(f"모든 기간 수익: {len(consistent_winners)}개 ({len(consistent_winners)/len(breakouts_df)*100:.1f}%)")
    print(f"모든 기간 손실: {len(consistent_losers)}개 ({len(consistent_losers)/len(breakouts_df)*100:.1f}%)")

    if len(consistent_winners) > 0:
        print(f"\n일관 수익 평균:")
        print(f"  10봉: {consistent_winners['return_10'].mean():.2f}%")
        print(f"  30봉: {consistent_winners['return_30'].mean():.2f}%")
        print(f"  60봉: {consistent_winners['return_60'].mean():.2f}%")

    analysis['consistent_winners'] = consistent_winners
    analysis['consistent_losers'] = consistent_losers

    # 3. 단기 vs 장기 성과 차이
    print("\n3. 수익률 증감 패턴:")
    print("-"*80)

    # 10봉 → 30봉 변화
    breakouts_df['return_change_10_to_30'] = breakouts_df['return_30'] - breakouts_df['return_10']
    # 30봉 → 60봉 변화
    breakouts_df['return_change_30_to_60'] = breakouts_df['return_60'] - breakouts_df['return_30']

    improving = breakouts_df[
        (breakouts_df['return_change_10_to_30'] > 0) &
        (breakouts_df['return_change_30_to_60'] > 0)
    ]

    deteriorating = breakouts_df[
        (breakouts_df['return_change_10_to_30'] < 0) &
        (breakouts_df['return_change_30_to_60'] < 0)
    ]

    print(f"지속 개선: {len(improving)}개 ({len(improving)/len(breakouts_df)*100:.1f}%)")
    print(f"지속 악화: {len(deteriorating)}개 ({len(deteriorating)/len(breakouts_df)*100:.1f}%)")

    analysis['improving'] = improving
    analysis['deteriorating'] = deteriorating

    # 4. 장기 신뢰도 지표
    print("\n4. 장기 신뢰도 지표 (60봉 기준):")
    print("-"*80)

    # 60봉에서 1% 이상 수익
    strong_long_term = breakouts_df[breakouts_df['return_60'] > 1.0]

    print(f"60봉 1%+ 수익: {len(strong_long_term)}개 ({len(strong_long_term)/len(breakouts_df)*100:.1f}%)")

    if len(strong_long_term) > 0:
        print(f"이 중 단기(10봉)도 수익: {(strong_long_term['return_10'] > 0).sum()}개")
        print(f"평균 60봉 수익률: {strong_long_term['return_60'].mean():.2f}%")

    analysis['strong_long_term'] = strong_long_term

    return analysis


def analyze_divergence_stacking_impact(breakouts_df):
    """
    다이버전스 중첩도별 성과 분석

    Args:
        breakouts_df: 돌파 DataFrame (with div_count columns)

    Returns:
        stacking_analysis dict
    """
    print("\n" + "="*80)
    print("다이버전스 중첩도별 성과 분석")
    print("="*80)

    analysis = {}

    for window in [10, 20, 50]:
        count_col = f'div_count_{window}'

        if count_col not in breakouts_df.columns:
            continue

        print(f"\n과거 {window}봉 다이버전스 중첩도:")
        print("-"*80)

        # 중첩도별 그룹화
        stacking_stats = breakouts_df.groupby(count_col).agg({
            'return_30': ['mean', 'count'],
            'win_30': 'mean',
            'max_profit_30': 'mean',
            'max_drawdown_30': 'mean'
        }).round(3)

        stacking_stats.columns = ['avg_return', 'count', 'win_rate', 'avg_max_profit', 'avg_max_dd']

        print(stacking_stats)

        analysis[f'window_{window}'] = stacking_stats

        # 중첩도와 수익률 상관관계
        correlation = breakouts_df[[count_col, 'return_30']].corr().iloc[0, 1]
        print(f"\n중첩도 vs 수익률 상관계수: {correlation:.3f}")

    return analysis


def generate_advanced_report(breakouts_df, win_rate_analysis, reliability_analysis,
                            stacking_analysis, output_path="ADVANCED_BREAKOUT_ANALYSIS.md"):
    """
    고급 분석 리포트 생성
    """
    from datetime import datetime

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 고급 돌파 분석 리포트\n\n")
        f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 1. 승률 기준별 비교
        f.write("## 1. 승률 기준별 비교\n\n")
        f.write("**Q: 승률 기준이 어떤 기준인가?**\n\n")
        f.write("다양한 수익률 임계값으로 승률을 계산했습니다:\n\n")

        f.write("| 기간 | 샘플 수 | 수익(>0%) | 수익(>0.5%) | 수익(>1%) | 수익(>1.5%) | 수익(>2%) | 평균 수익률 |\n")
        f.write("|------|---------|-----------|-------------|-----------|-------------|-----------|-------------|\n")

        for _, row in win_rate_analysis.iterrows():
            f.write(f"| {row['period']} | {row['sample_size']} | ")
            f.write(f"{row['positive']:.1f}% | {row['above_0.5pct']:.1f}% | ")
            f.write(f"{row['above_1pct']:.1f}% | {row['above_1.5pct']:.1f}% | ")
            f.write(f"{row['above_2pct']:.1f}% | {row['avg_return']:.2f}% |\n")
        f.write("\n")

        f.write("**분석**:\n")
        f.write("- 기존 승률(>0%)은 단순히 양수 수익 여부만 판단\n")
        f.write("- 실질적 수익(>1%)을 기준으로 하면 승률 대폭 감소\n")
        f.write("- 장기 보유(60봉)에서도 2%+ 수익은 약 20-30% 수준\n\n")

        # 2. 단기 vs 장기 분석
        f.write("## 2. 단기 vs 장기 신뢰도\n\n")
        f.write("**Q: 1.87:1, 74%는 단기 매매 기준인가?**\n\n")

        timeframe_stats = reliability_analysis['timeframe_stats']
        f.write("### 2.1 기간별 성과 비교\n\n")
        f.write("| 기간 | 평균 수익률 | 승률(>0%) | 승률(>1%) | Avg Max 수익 | Avg Max 손실 |\n")
        f.write("|------|------------|-----------|-----------|--------------|-------------|\n")

        for _, row in timeframe_stats.iterrows():
            f.write(f"| {row['period']} | {row['avg_return']:.2f}% | {row['win_rate_positive']:.1f}% | ")
            f.write(f"{row['win_rate_1pct']:.1f}% | {row['avg_max_profit']:.2f}% | {row['avg_max_dd']:.2f}% |\n")
        f.write("\n")

        f.write("**핵심 발견**:\n")
        f.write("- 10봉(2.5시간): 단기 스캘핑 수준, 낮은 수익률\n")
        f.write("- 30봉(7.5시간): 중기 트레이딩, 균형잡힌 성과\n")
        f.write("- 60봉(15시간): 장기 포지션, 수익률 증가 but 변동성도 증가\n\n")

        # 3. 일관성 분석
        consistent_winners = reliability_analysis['consistent_winners']
        consistent_losers = reliability_analysis['consistent_losers']

        f.write("### 2.2 일관성 분석\n\n")
        f.write(f"- **모든 기간 수익**: {len(consistent_winners)}개 ({len(consistent_winners)/len(breakouts_df)*100:.1f}%)\n")
        f.write(f"- **모든 기간 손실**: {len(consistent_losers)}개 ({len(consistent_losers)/len(breakouts_df)*100:.1f}%)\n\n")

        if len(consistent_winners) > 0:
            f.write("일관 수익 그룹 평균:\n")
            f.write(f"- 10봉: {consistent_winners['return_10'].mean():.2f}%\n")
            f.write(f"- 30봉: {consistent_winners['return_30'].mean():.2f}%\n")
            f.write(f"- 60봉: {consistent_winners['return_60'].mean():.2f}%\n\n")

        # 4. 다이버전스 중첩도 분석
        f.write("## 3. 다이버전스 중첩도 분석\n\n")
        f.write("**Q: 돌파 시점에 다이버전스 중첩이 몇 개나 되었나? 수익률과 상관관계는?**\n\n")

        for window in [10, 20, 50]:
            key = f'window_{window}'
            if key in stacking_analysis:
                stats = stacking_analysis[key]

                f.write(f"### 3.{window//10} 과거 {window}봉 다이버전스 중첩도\n\n")
                f.write("| 중첩 개수 | 평균 수익률 | 승률 | Avg Max 수익 | Avg Max 손실 | 샘플 수 |\n")
                f.write("|----------|------------|------|--------------|-------------|--------|\n")

                for idx, row in stats.iterrows():
                    f.write(f"| {int(idx)}개 | {row['avg_return']:.3f}% | {row['win_rate']*100:.1f}% | ")
                    f.write(f"{row['avg_max_profit']:.3f}% | {row['avg_max_dd']:.3f}% | {int(row['count'])} |\n")
                f.write("\n")

        # 5. 품질 점수 분석
        f.write("## 4. 돌파 품질 점수 시스템\n\n")
        f.write("**Q: 장기적으로 확실한 자리를 어떻게 판단할 것인가?**\n\n")

        f.write("### 4.1 품질 점수 계산 방법\n\n")
        f.write("```\n")
        f.write("품질 점수 = (10봉 성과 × 30%) + (30봉 성과 × 40%) + (60봉 성과 × 30%)\n")
        f.write("           - (최대 손실 패널티)\n")
        f.write("```\n\n")

        f.write("### 4.2 등급별 분포\n\n")
        grade_dist = breakouts_df['quality_grade'].value_counts().sort_index()
        f.write("| 등급 | 개수 | 비율 |\n")
        f.write("|------|------|------|\n")

        for grade, count in grade_dist.items():
            f.write(f"| {grade} | {count} | {count/len(breakouts_df)*100:.1f}% |\n")
        f.write("\n")

        # 6. 실전 전략
        f.write("## 5. 실전 트레이딩 전략\n\n")

        f.write("### 5.1 기간별 적합한 전략\n\n")
        f.write("**단기 (10봉, 2.5시간)**\n")
        f.write("- 평균 수익률이 낮음 (0.16%)\n")
        f.write("- 스캘핑에는 부적합\n")
        f.write("- 초기 손절 판단용으로만 활용\n\n")

        f.write("**중기 (30봉, 7.5시간)**\n")
        f.write("- 평균 수익률 0.32%, 승률 62.9%\n")
        f.write("- **가장 균형잡힌 선택**\n")
        f.write("- 대부분의 돌파 트레이딩에 적합\n\n")

        f.write("**장기 (60봉, 15시간)**\n")
        f.write("- 평균 수익률 0.40%, 승률 61.7%\n")
        f.write("- 수익률은 높지만 변동성 증가\n")
        f.write("- 고품질(S/A 등급) 돌파에만 적용\n\n")

        f.write("### 5.2 다이버전스 중첩 활용\n\n")
        f.write("- 중첩 개수가 많다고 반드시 수익률이 높지는 않음\n")
        f.write("- 다이버전스는 보조 지표로만 활용\n")
        f.write("- 추세선 품질이 더 중요\n\n")

        f.write("### 5.3 품질 등급별 전략\n\n")
        f.write("| 등급 | 보유 기간 | 포지션 크기 | 리스크 관리 |\n")
        f.write("|------|----------|------------|------------|\n")
        f.write("| S (우수) | 60봉까지 | 100% | 느슨한 손절 |\n")
        f.write("| A (양호) | 30-60봉 | 75% | 표준 손절 |\n")
        f.write("| B (보통) | 30봉 | 50% | 타이트 손절 |\n")
        f.write("| C (평범) | 10-30봉 | 25% | 매우 타이트 |\n")
        f.write("| D (위험) | 진입 금지 | 0% | - |\n\n")

        f.write("---\n\n")
        f.write("*이 리포트는 자동 생성되었습니다.*\n")

    print(f"\n고급 분석 리포트 저장: {output_path}")


def visualize_advanced_analysis(breakouts_df, win_rate_analysis, stacking_analysis,
                                output_path="advanced_analysis_charts.png"):
    """
    고급 분석 시각화
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Breakout Analysis', fontsize=16)

    # 1. 승률 기준별 비교
    criteria_names = ['positive', 'above_0.5pct', 'above_1pct', 'above_1.5pct', 'above_2pct']
    periods = win_rate_analysis['period'].tolist()

    for i, criterion in enumerate(criteria_names):
        if i == 0:
            axes[0, 0].plot(periods, win_rate_analysis[criterion], marker='o', label=criterion, linewidth=2)

    axes[0, 0].plot(periods, win_rate_analysis['positive'], marker='o', label='>0%', linewidth=2)
    axes[0, 0].plot(periods, win_rate_analysis['above_1pct'], marker='s', label='>1%', linewidth=2)
    axes[0, 0].plot(periods, win_rate_analysis['above_2pct'], marker='^', label='>2%', linewidth=2)
    axes[0, 0].set_title('Win Rate by Different Criteria')
    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].set_xlabel('Period')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. 기간별 평균 수익률
    periods_num = [10, 30, 60]
    avg_returns = [breakouts_df[f'return_{p}'].mean() for p in periods_num]

    axes[0, 1].bar([str(p) for p in periods_num], avg_returns, color=['lightblue', 'skyblue', 'steelblue'])
    axes[0, 1].set_title('Average Return by Period')
    axes[0, 1].set_ylabel('Avg Return (%)')
    axes[0, 1].set_xlabel('Period (bars)')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 품질 점수 분포
    axes[0, 2].hist(breakouts_df['quality_score'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Quality Score Distribution')
    axes[0, 2].set_xlabel('Quality Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(x=breakouts_df['quality_score'].mean(), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {breakouts_df["quality_score"].mean():.1f}')
    axes[0, 2].legend()
    axes[0, 2].grid(axis='y', alpha=0.3)

    # 4. 다이버전스 중첩도 vs 수익률 (50봉 윈도우)
    if 'div_count_50' in breakouts_df.columns:
        div_stats = breakouts_df.groupby('div_count_50')['return_30'].mean()

        axes[1, 0].bar(div_stats.index.astype(str), div_stats.values, color='coral', edgecolor='black')
        axes[1, 0].set_title('Avg Return by Divergence Count (50-bar window)')
        axes[1, 0].set_xlabel('Number of Divergences')
        axes[1, 0].set_ylabel('Avg Return (%)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
        axes[1, 0].grid(axis='y', alpha=0.3)

    # 5. 품질 등급별 수익률
    grade_returns = breakouts_df.groupby('quality_grade')['return_30'].mean().sort_index()

    axes[1, 1].bar(range(len(grade_returns)), grade_returns.values,
                   color=['green', 'lightgreen', 'yellow', 'orange', 'red'][:len(grade_returns)],
                   edgecolor='black')
    axes[1, 1].set_xticks(range(len(grade_returns)))
    axes[1, 1].set_xticklabels(grade_returns.index, rotation=45)
    axes[1, 1].set_title('Avg Return by Quality Grade')
    axes[1, 1].set_ylabel('Avg Return (%)')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # 6. 10봉 vs 60봉 수익률 산점도
    axes[1, 2].scatter(breakouts_df['return_10'], breakouts_df['return_60'],
                      alpha=0.3, s=10, color='purple')
    axes[1, 2].set_title('10-bar vs 60-bar Return Correlation')
    axes[1, 2].set_xlabel('10-bar Return (%)')
    axes[1, 2].set_ylabel('60-bar Return (%)')
    axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 2].axvline(x=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 2].grid(alpha=0.3)

    # 상관계수 표시
    corr = breakouts_df[['return_10', 'return_60']].corr().iloc[0, 1]
    axes[1, 2].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n고급 분석 차트 저장: {output_path}")


if __name__ == "__main__":
    import glob

    # 기존 분석 결과 로드
    if not all([
        glob.glob("output/divergence.csv"),
        glob.glob("output/breakout_stats.csv")
    ]):
        print("Phase 3, 5 결과 필요. 먼저 파이프라인을 실행하세요.")
        exit(1)

    print("데이터 로드 중...")
    df = pd.read_csv("output/divergence.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    breakouts_df = pd.read_csv("output/breakout_stats.csv")

    print(f"돌파 이벤트: {len(breakouts_df)}개\n")

    # 1. 다이버전스 중첩도 계산
    breakouts_df = calculate_divergence_stacking(df, breakouts_df, lookback_windows=[10, 20, 50])

    # 2. 승률 기준별 분석
    win_rate_analysis = analyze_win_rates_by_criteria(breakouts_df)

    # 3. 품질 점수 계산
    breakouts_df = calculate_breakout_quality_score(breakouts_df)

    # 4. 시간프레임 신뢰도 분석
    reliability_analysis = analyze_timeframe_reliability(breakouts_df)

    # 5. 다이버전스 중첩도 영향 분석
    stacking_analysis = analyze_divergence_stacking_impact(breakouts_df)

    # 6. 시각화
    visualize_advanced_analysis(breakouts_df, win_rate_analysis, stacking_analysis,
                               "output/advanced_analysis_charts.png")

    # 7. 리포트 생성
    generate_advanced_report(breakouts_df, win_rate_analysis, reliability_analysis,
                           stacking_analysis, "output/ADVANCED_BREAKOUT_ANALYSIS.md")

    # 8. 결과 저장
    breakouts_df.to_csv("output/breakout_stats_enhanced.csv", index=False)
    print("\n강화된 돌파 통계 저장: output/breakout_stats_enhanced.csv")

    print("\n✅ 고급 분석 완료!")
