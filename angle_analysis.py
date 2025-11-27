"""
추세선 각도별 돌파 성공률 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_trendline_angle(slope, avg_price):
    """
    추세선 기울기를 각도로 변환

    Args:
        slope: 추세선 기울기 (price change per bar)
        avg_price: 평균 가격 (정규화용)

    Returns:
        angle in degrees
    """
    # 정규화된 기울기 (캔들당 가격 변화율)
    normalized_slope = slope / avg_price

    # 라디안으로 변환 후 각도로
    angle_rad = np.arctan(normalized_slope)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def categorize_angle(angle):
    """
    각도를 카테고리로 분류

    Args:
        angle: 각도 (degrees)

    Returns:
        category string
    """
    abs_angle = abs(angle)

    if abs_angle < 5:
        return "완만 (0-5°)"
    elif abs_angle < 15:
        return "보통 (5-15°)"
    elif abs_angle < 30:
        return "가파름 (15-30°)"
    elif abs_angle < 45:
        return "매우가파름 (30-45°)"
    else:
        return "급격함 (45°+)"


def analyze_angle_impact(df, trendlines_df, breakouts_df):
    """
    추세선 각도별 돌파 성공률 분석

    Args:
        df: 원본 DataFrame
        trendlines_df: 추세선 DataFrame
        breakouts_df: 돌파 DataFrame (with return columns)

    Returns:
        angle_analysis_dict
    """
    print("\n" + "="*80)
    print("추세선 각도별 돌파 성공률 분석")
    print("="*80)

    # 1. 추세선에 각도 계산 추가
    trendlines_df = trendlines_df.copy()

    # 각 추세선의 평균 가격 계산
    trendlines_df['avg_price'] = (trendlines_df['start_price'] + trendlines_df['end_price']) / 2

    # 각도 계산
    trendlines_df['angle'] = trendlines_df.apply(
        lambda row: calculate_trendline_angle(row['slope'], row['avg_price']),
        axis=1
    )

    # 각도 카테고리
    trendlines_df['angle_category'] = trendlines_df['angle'].apply(categorize_angle)

    print(f"\n추세선 각도 통계:")
    print(f"  평균 각도: {trendlines_df['angle'].mean():.2f}°")
    print(f"  중앙값: {trendlines_df['angle'].median():.2f}°")
    print(f"  표준편차: {trendlines_df['angle'].std():.2f}°")
    print(f"  최소: {trendlines_df['angle'].min():.2f}°")
    print(f"  최대: {trendlines_df['angle'].max():.2f}°")

    # 2. 돌파 이벤트에 각도 정보 병합
    breakouts_df = breakouts_df.copy()

    # 추세선 돌파만 필터링
    trendline_breakouts = breakouts_df[breakouts_df['trendline_idx'] >= 0].copy()

    # 각도 정보 병합
    trendline_breakouts['angle'] = trendline_breakouts['trendline_idx'].apply(
        lambda idx: trendlines_df.iloc[idx]['angle'] if idx < len(trendlines_df) else 0
    )

    trendline_breakouts['angle_category'] = trendline_breakouts['angle'].apply(categorize_angle)

    # 방향별 분리 (상승 돌파 vs 하락 돌파)
    trendline_breakouts['direction'] = trendline_breakouts['type'].apply(
        lambda x: 'up' if x == 'trendline_up' else 'down'
    )

    print(f"\n추세선 돌파 수: {len(trendline_breakouts)}개")

    # 3. 각도 카테고리별 성과 분석
    print("\n" + "-"*80)
    print("각도 카테고리별 성과 (전체)")
    print("-"*80)

    angle_performance = trendline_breakouts.groupby('angle_category').agg({
        'return_30': ['mean', 'std', 'count'],
        'win_30': 'mean',
        'max_profit_30': 'mean',
        'max_drawdown_30': 'mean'
    }).round(3)

    # 컬럼명 정리
    angle_performance.columns = ['avg_return', 'std_return', 'count', 'win_rate', 'avg_max_profit', 'avg_max_dd']
    angle_performance = angle_performance.sort_values('avg_return', ascending=False)

    print(angle_performance)

    # 4. 방향 + 각도별 분석
    print("\n" + "-"*80)
    print("방향 + 각도 카테고리별 성과")
    print("-"*80)

    direction_angle_performance = trendline_breakouts.groupby(['direction', 'angle_category']).agg({
        'return_30': ['mean', 'count'],
        'win_30': 'mean'
    }).round(3)

    direction_angle_performance.columns = ['avg_return', 'count', 'win_rate']

    print("\n상승 돌파 (Trendline Up):")
    up_perf = direction_angle_performance.xs('up', level=0)
    print(up_perf.sort_values('avg_return', ascending=False))

    print("\n하락 돌파 (Trendline Down):")
    down_perf = direction_angle_performance.xs('down', level=0)
    print(down_perf.sort_values('avg_return', ascending=False))

    # 5. 각도 구간별 세부 분석 (5도 단위)
    print("\n" + "-"*80)
    print("각도 구간별 세부 성과 (5° 단위)")
    print("-"*80)

    # 각도를 5도 단위로 버킷팅
    trendline_breakouts['angle_bucket'] = (trendline_breakouts['angle'] // 5) * 5

    angle_bucket_performance = trendline_breakouts.groupby('angle_bucket').agg({
        'return_30': ['mean', 'count'],
        'win_30': 'mean'
    }).round(3)

    angle_bucket_performance.columns = ['avg_return', 'count', 'win_rate']
    angle_bucket_performance = angle_bucket_performance[angle_bucket_performance['count'] >= 10]  # 최소 10개 이상
    angle_bucket_performance = angle_bucket_performance.sort_index()

    print(angle_bucket_performance)

    # 6. 상관관계 분석
    print("\n" + "-"*80)
    print("각도와 수익률 상관관계")
    print("-"*80)

    correlation = trendline_breakouts[['angle', 'return_30', 'max_profit_30', 'max_drawdown_30']].corr()
    print(correlation['angle'].round(3))

    # 7. 최적 각도 구간 찾기
    best_angle_bucket = angle_bucket_performance['avg_return'].idxmax()
    best_performance = angle_bucket_performance.loc[best_angle_bucket]

    print("\n" + "-"*80)
    print("최적 각도 구간")
    print("-"*80)
    print(f"각도 구간: {best_angle_bucket}° ~ {best_angle_bucket + 5}°")
    print(f"평균 수익률: {best_performance['avg_return']:.3f}%")
    print(f"승률: {best_performance['win_rate']*100:.1f}%")
    print(f"샘플 수: {int(best_performance['count'])}개")

    # 결과 딕셔너리
    analysis_dict = {
        'trendlines_with_angle': trendlines_df,
        'breakouts_with_angle': trendline_breakouts,
        'angle_performance': angle_performance,
        'direction_angle_performance': direction_angle_performance,
        'angle_bucket_performance': angle_bucket_performance,
        'correlation': correlation,
        'best_angle_bucket': best_angle_bucket,
        'best_performance': best_performance
    }

    return analysis_dict


def visualize_angle_analysis(analysis_dict, output_path="angle_analysis_charts.png"):
    """
    각도 분석 시각화

    Args:
        analysis_dict: 분석 결과 딕셔너리
        output_path: 차트 저장 경로
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trendline Angle Analysis', fontsize=16)

    breakouts_df = analysis_dict['breakouts_with_angle']
    angle_perf = analysis_dict['angle_performance']
    bucket_perf = analysis_dict['angle_bucket_performance']

    # 1. 각도 분포
    axes[0, 0].hist(breakouts_df['angle'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Trendline Angle Distribution')
    axes[0, 0].set_xlabel('Angle (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. 각도 카테고리별 평균 수익률
    angle_perf['avg_return'].plot(kind='bar', ax=axes[0, 1], color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Avg Return by Angle Category')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].set_xlabel('Angle Category')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. 각도 카테고리별 승률
    (angle_perf['win_rate'] * 100).plot(kind='bar', ax=axes[0, 2], color='salmon', edgecolor='black')
    axes[0, 2].set_title('Win Rate by Angle Category')
    axes[0, 2].set_ylabel('Win Rate (%)')
    axes[0, 2].set_xlabel('Angle Category')
    axes[0, 2].axhline(y=50, color='r', linestyle='--', linewidth=0.5)
    axes[0, 2].grid(axis='y', alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. 각도 vs 수익률 산점도
    axes[1, 0].scatter(breakouts_df['angle'], breakouts_df['return_30'],
                       alpha=0.3, s=10, color='mediumpurple')
    axes[1, 0].set_title('Angle vs Return Scatter Plot')
    axes[1, 0].set_xlabel('Angle (degrees)')
    axes[1, 0].set_ylabel('Return 30-bar (%)')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 0].grid(alpha=0.3)

    # 5. 각도 버킷별 수익률 (5도 단위)
    bucket_perf['avg_return'].plot(kind='line', ax=axes[1, 1],
                                     marker='o', color='darkblue', linewidth=2)
    axes[1, 1].set_title('Avg Return by Angle Bucket (5° intervals)')
    axes[1, 1].set_xlabel('Angle Bucket (degrees)')
    axes[1, 1].set_ylabel('Avg Return (%)')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=0.5)
    axes[1, 1].grid(alpha=0.3)

    # 6. 방향별 각도 분포
    up_breakouts = breakouts_df[breakouts_df['direction'] == 'up']
    down_breakouts = breakouts_df[breakouts_df['direction'] == 'down']

    axes[1, 2].hist([up_breakouts['angle'], down_breakouts['angle']],
                    bins=30, label=['Trendline Up', 'Trendline Down'],
                    color=['lightgreen', 'lightcoral'], alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Angle Distribution by Direction')
    axes[1, 2].set_xlabel('Angle (degrees)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n각도 분석 차트 저장: {output_path}")


def generate_angle_report(analysis_dict, output_path="ANGLE_ANALYSIS_REPORT.md"):
    """
    각도 분석 리포트 생성

    Args:
        analysis_dict: 분석 결과 딕셔너리
        output_path: 리포트 저장 경로
    """
    from datetime import datetime

    angle_perf = analysis_dict['angle_performance']
    bucket_perf = analysis_dict['angle_bucket_performance']
    best_bucket = analysis_dict['best_angle_bucket']
    best_perf = analysis_dict['best_performance']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 추세선 각도별 돌파 성공률 분석 리포트\n\n")
        f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 1. Executive Summary
        f.write("## 1. Executive Summary\n\n")
        f.write("추세선의 기울기(각도)에 따른 돌파 성공률 및 수익률을 분석했습니다.\n\n")

        best_category = angle_perf.index[0]
        best_return = angle_perf.iloc[0]['avg_return']
        best_winrate = angle_perf.iloc[0]['win_rate'] * 100

        f.write(f"### 핵심 발견사항\n\n")
        f.write(f"- **최고 성과 각도 카테고리**: {best_category}\n")
        f.write(f"  - 평균 수익률: {best_return:.3f}%\n")
        f.write(f"  - 승률: {best_winrate:.1f}%\n\n")

        f.write(f"- **최적 각도 구간**: {best_bucket}° ~ {best_bucket + 5}°\n")
        f.write(f"  - 평균 수익률: {best_perf['avg_return']:.3f}%\n")
        f.write(f"  - 승률: {best_perf['win_rate']*100:.1f}%\n")
        f.write(f"  - 샘플 수: {int(best_perf['count'])}개\n\n")

        # 2. 각도 카테고리별 성과
        f.write("## 2. 각도 카테고리별 성과\n\n")
        f.write("| 각도 카테고리 | 평균 수익률 | 승률 | Max 수익 | Max 손실 | 샘플 수 |\n")
        f.write("|--------------|------------|------|----------|----------|--------|\n")

        for idx, row in angle_perf.iterrows():
            f.write(f"| {idx} | {row['avg_return']:.3f}% | {row['win_rate']*100:.1f}% | ")
            f.write(f"{row['avg_max_profit']:.3f}% | {row['avg_max_dd']:.3f}% | {int(row['count'])} |\n")
        f.write("\n")

        # 3. 방향별 최적 각도
        dir_angle_perf = analysis_dict['direction_angle_performance']

        f.write("## 3. 방향별 최적 각도\n\n")

        f.write("### 3.1 상승 돌파 (Trendline Up)\n\n")
        up_perf = dir_angle_perf.xs('up', level=0).sort_values('avg_return', ascending=False)
        f.write("| 각도 카테고리 | 평균 수익률 | 승률 | 샘플 수 |\n")
        f.write("|--------------|------------|------|--------|\n")
        for idx, row in up_perf.iterrows():
            f.write(f"| {idx} | {row['avg_return']:.3f}% | {row['win_rate']*100:.1f}% | {int(row['count'])} |\n")
        f.write("\n")

        f.write("### 3.2 하락 돌파 (Trendline Down)\n\n")
        down_perf = dir_angle_perf.xs('down', level=0).sort_values('avg_return', ascending=False)
        f.write("| 각도 카테고리 | 평균 수익률 | 승률 | 샘플 수 |\n")
        f.write("|--------------|------------|------|--------|\n")
        for idx, row in down_perf.iterrows():
            f.write(f"| {idx} | {row['avg_return']:.3f}% | {row['win_rate']*100:.1f}% | {int(row['count'])} |\n")
        f.write("\n")

        # 4. 각도 구간별 세부 성과
        f.write("## 4. 각도 구간별 세부 성과 (5° 단위)\n\n")
        f.write("최소 10개 이상 샘플이 있는 구간만 표시\n\n")
        f.write("| 각도 구간 | 평균 수익률 | 승률 | 샘플 수 |\n")
        f.write("|----------|------------|------|--------|\n")

        for angle, row in bucket_perf.iterrows():
            f.write(f"| {angle:.0f}° ~ {angle+5:.0f}° | {row['avg_return']:.3f}% | ")
            f.write(f"{row['win_rate']*100:.1f}% | {int(row['count'])} |\n")
        f.write("\n")

        # 5. 트레이딩 전략 권고
        f.write("## 5. 트레이딩 전략 권고\n\n")

        # 최고/최저 성과 카테고리
        worst_category = angle_perf.index[-1]
        worst_return = angle_perf.iloc[-1]['avg_return']

        f.write(f"### 5.1 각도 선택 기준\n\n")
        f.write(f"✅ **선호 각도**: {best_category}\n")
        f.write(f"- 평균 수익률 {best_return:.3f}%로 가장 우수\n")
        f.write(f"- 이 범위의 추세선 돌파에 우선 집중\n\n")

        f.write(f"⚠️ **회피 각도**: {worst_category}\n")
        f.write(f"- 평균 수익률 {worst_return:.3f}%로 상대적으로 낮음\n")
        f.write(f"- 이 범위의 추세선 돌파는 신중히 접근\n\n")

        f.write(f"### 5.2 각도별 리스크 관리\n\n")

        for idx, row in angle_perf.head(3).iterrows():
            f.write(f"**{idx}**\n")
            f.write(f"- 승률: {row['win_rate']*100:.1f}%\n")
            f.write(f"- 평균 최대 수익: {row['avg_max_profit']:.3f}%\n")
            f.write(f"- 평균 최대 손실: {row['avg_max_dd']:.3f}%\n")

            risk_reward = abs(row['avg_max_profit'] / row['avg_max_dd']) if row['avg_max_dd'] != 0 else 0
            f.write(f"- 리스크/보상 비율: {risk_reward:.2f}\n\n")

        # 6. 시각화
        f.write("## 6. 시각화\n\n")
        f.write("![Angle Analysis Charts](angle_analysis_charts.png)\n\n")

        f.write("---\n\n")
        f.write("*이 리포트는 자동 생성되었습니다.*\n")

    print(f"각도 분석 리포트 저장: {output_path}")


if __name__ == "__main__":
    import glob

    # Phase 2, 5 결과 로드
    if not all([
        glob.glob("output/trendlines.csv"),
        glob.glob("output/breakout_stats.csv")
    ]):
        print("Phase 2, 5 결과 필요. 먼저 파이프라인을 실행하세요.")
        exit(1)

    print("데이터 로드 중...")
    df = pd.read_csv("output/divergence.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    trendlines_df = pd.read_csv("output/trendlines.csv")
    breakouts_df = pd.read_csv("output/breakout_stats.csv")

    print(f"추세선: {len(trendlines_df)}개")
    print(f"돌파: {len(breakouts_df)}개")

    # 각도 분석 실행
    analysis_dict = analyze_angle_impact(df, trendlines_df, breakouts_df)

    # 시각화
    visualize_angle_analysis(analysis_dict, "output/angle_analysis_charts.png")

    # 리포트 생성
    generate_angle_report(analysis_dict, "output/ANGLE_ANALYSIS_REPORT.md")

    print("\n✅ 각도 분석 완료!")
