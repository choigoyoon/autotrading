"""
추세선 품질 필터링 테스트
높은 품질의 추세선만 사용하면 수익률이 개선되는지 검증
"""

import pandas as pd
import numpy as np
from phase4_breakouts import detect_breakouts
from phase5_statistics import calculate_forward_returns


def filter_trendlines_by_quality(trendlines_df, min_touches=2, min_duration=0, max_abs_slope=None):
    """
    추세선 품질 필터링

    Args:
        min_touches: 최소 터치 횟수
        min_duration: 최소 지속 기간 (봉)
        max_abs_slope: 최대 절대 기울기
    """
    filtered = trendlines_df.copy()

    # 터치 횟수 필터
    filtered = filtered[filtered['touch_count'] >= min_touches]

    # 지속 기간 필터
    filtered = filtered[filtered['duration'] >= min_duration]

    # 기울기 필터
    if max_abs_slope is not None:
        filtered = filtered[abs(filtered['slope']) <= max_abs_slope]

    return filtered


def test_quality_filters(df, trendlines_df):
    """
    다양한 품질 필터 조합 테스트
    """
    print("=" * 60)
    print("추세선 품질 필터링 테스트")
    print("=" * 60)

    results = []

    # 테스트 케이스들
    test_cases = [
        {"name": "전체 (필터 없음)", "min_touches": 2, "min_duration": 0, "max_slope": None},
        {"name": "터치 3회 이상", "min_touches": 3, "min_duration": 0, "max_slope": None},
        {"name": "터치 4회 이상", "min_touches": 4, "min_duration": 0, "max_slope": None},
        {"name": "터치 5회 이상", "min_touches": 5, "min_duration": 0, "max_slope": None},
        {"name": "지속 100봉 이상", "min_touches": 2, "min_duration": 100, "max_slope": None},
        {"name": "지속 200봉 이상", "min_touches": 2, "min_duration": 200, "max_slope": None},
        {"name": "터치 3회 + 지속 100봉", "min_touches": 3, "min_duration": 100, "max_slope": None},
        {"name": "터치 4회 + 지속 150봉", "min_touches": 4, "min_duration": 150, "max_slope": None},
        {"name": "터치 4회 + 지속 200봉", "min_touches": 4, "min_duration": 200, "max_slope": None},
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 테스트: {test['name']}")

        # 추세선 필터링
        filtered_trendlines = filter_trendlines_by_quality(
            trendlines_df,
            min_touches=test['min_touches'],
            min_duration=test['min_duration'],
            max_abs_slope=test['max_slope']
        )

        print(f"  추세선 수: {len(trendlines_df):,}개 → {len(filtered_trendlines):,}개 ({len(filtered_trendlines)/len(trendlines_df)*100:.1f}%)")

        if len(filtered_trendlines) == 0:
            print("  ⚠️  추세선이 없어서 스킵")
            continue

        # 돌파 탐지
        breakouts = detect_breakouts(df, filtered_trendlines, lookback=10)

        # 추세선 돌파만 필터
        trendline_breakouts = breakouts[breakouts['type'].isin(['trendline_up', 'trendline_down'])]

        print(f"  돌파 수: {len(trendline_breakouts):,}개")

        if len(trendline_breakouts) == 0:
            print("  ⚠️  돌파가 없어서 스킵")
            continue

        # 수익률 계산
        breakouts_with_returns = calculate_forward_returns(df, trendline_breakouts, periods=[10, 30, 60])

        # 통계
        ret_30 = breakouts_with_returns['return_30'].mean()
        win_rate_30 = (breakouts_with_returns['return_30'] > 0).sum() / len(breakouts_with_returns) * 100
        ret_60 = breakouts_with_returns['return_60'].mean()

        # 최대 수익/손실
        max_profit = breakouts_with_returns['max_profit_60'].mean()
        max_dd = breakouts_with_returns['max_drawdown_60'].mean()
        rr_ratio = abs(max_profit / max_dd) if max_dd != 0 else 0

        print(f"  30봉 수익률: {ret_30:.3f}%  승률: {win_rate_30:.1f}%")
        print(f"  60봉 수익률: {ret_60:.3f}%")
        print(f"  Risk/Reward: {rr_ratio:.2f}:1")

        results.append({
            'test_name': test['name'],
            'min_touches': test['min_touches'],
            'min_duration': test['min_duration'],
            'trendline_count': len(filtered_trendlines),
            'breakout_count': len(trendline_breakouts),
            'return_30': ret_30,
            'win_rate_30': win_rate_30,
            'return_60': ret_60,
            'max_profit': max_profit,
            'max_dd': max_dd,
            'rr_ratio': rr_ratio
        })

    return pd.DataFrame(results)


def analyze_results(results_df):
    """
    결과 분석 및 최적 필터 찾기
    """
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    print("\n30봉 수익률 기준 정렬:")
    sorted_by_return = results_df.sort_values('return_30', ascending=False)
    print(sorted_by_return[['test_name', 'return_30', 'win_rate_30', 'breakout_count']].to_string(index=False))

    print("\n승률 기준 정렬:")
    sorted_by_winrate = results_df.sort_values('win_rate_30', ascending=False)
    print(sorted_by_winrate[['test_name', 'win_rate_30', 'return_30', 'breakout_count']].head(5).to_string(index=False))

    print("\nRisk/Reward 기준 정렬:")
    sorted_by_rr = results_df.sort_values('rr_ratio', ascending=False)
    print(sorted_by_rr[['test_name', 'rr_ratio', 'return_30', 'win_rate_30']].head(5).to_string(index=False))

    # 최적 필터 찾기
    print("\n" + "=" * 60)
    print("최적 필터 추천")
    print("=" * 60)

    # 조건: 최소 1000개 샘플, 최고 수익률
    valid_results = results_df[results_df['breakout_count'] >= 1000]

    if len(valid_results) > 0:
        best_return = valid_results.loc[valid_results['return_30'].idxmax()]
        best_winrate = valid_results.loc[valid_results['win_rate_30'].idxmax()]
        best_rr = valid_results.loc[valid_results['rr_ratio'].idxmax()]

        print(f"\n최고 수익률: {best_return['test_name']}")
        print(f"  30봉 수익률: {best_return['return_30']:.3f}%")
        print(f"  승률: {best_return['win_rate_30']:.1f}%")
        print(f"  샘플 수: {int(best_return['breakout_count']):,}개")

        print(f"\n최고 승률: {best_winrate['test_name']}")
        print(f"  승률: {best_winrate['win_rate_30']:.1f}%")
        print(f"  30봉 수익률: {best_winrate['return_30']:.3f}%")
        print(f"  샘플 수: {int(best_winrate['breakout_count']):,}개")

        print(f"\n최고 Risk/Reward: {best_rr['test_name']}")
        print(f"  Risk/Reward: {best_rr['rr_ratio']:.2f}:1")
        print(f"  30봉 수익률: {best_rr['return_30']:.3f}%")
        print(f"  승률: {best_rr['win_rate_30']:.1f}%")


def generate_report(results_df):
    """
    리포트 생성
    """
    report = []
    report.append("# 추세선 품질 필터링 테스트 결과\n")
    report.append(f"생성 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("\n---\n")

    report.append("\n## 🎯 목적\n\n")
    report.append("추세선의 품질(터치 횟수, 지속 기간)을 필터링하면 돌파 수익률이 개선되는가?\n\n")

    report.append("\n## 📊 테스트 결과\n\n")
    report.append("| 필터 | 추세선 수 | 돌파 수 | 30봉 수익률 | 승률 | R/R |\n")
    report.append("|------|----------|---------|------------|------|-----|\n")

    for _, row in results_df.iterrows():
        report.append(f"| {row['test_name']} | {int(row['trendline_count']):,} | ")
        report.append(f"{int(row['breakout_count']):,} | {row['return_30']:.3f}% | ")
        report.append(f"{row['win_rate_30']:.1f}% | {row['rr_ratio']:.2f}:1 |\n")

    # 최적 필터
    valid_results = results_df[results_df['breakout_count'] >= 1000]
    if len(valid_results) > 0:
        best = valid_results.loc[valid_results['return_30'].idxmax()]
        baseline = results_df.iloc[0]  # 첫 번째 (필터 없음)

        improvement = best['return_30'] - baseline['return_30']
        improvement_pct = (improvement / abs(baseline['return_30']) * 100) if baseline['return_30'] != 0 else 0

        report.append("\n## ✅ 결론\n\n")
        report.append(f"**최적 필터**: {best['test_name']}\n\n")
        report.append(f"- 수익률: {baseline['return_30']:.3f}% → {best['return_30']:.3f}% ({improvement:+.3f}%, {improvement_pct:+.1f}%)\n")
        report.append(f"- 승률: {baseline['win_rate_30']:.1f}% → {best['win_rate_30']:.1f}% ({best['win_rate_30'] - baseline['win_rate_30']:+.1f}%p)\n")
        report.append(f"- 샘플 수: {int(baseline['breakout_count']):,}개 → {int(best['breakout_count']):,}개\n\n")

        if improvement > 0:
            report.append("**품질 필터링이 효과적!** ✅\n")
        else:
            report.append("**품질 필터링 효과 미미** ⚠️\n")

    report.append("\n---\n\n")
    report.append("*이 리포트는 자동 생성되었습니다.*\n")

    return ''.join(report)


if __name__ == "__main__":
    import os

    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_csv("output_phase1_labeled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    trendlines_df = pd.read_csv("output_phase2_trendlines.csv")

    print(f"데이터 크기: {len(df):,} 캔들")
    print(f"추세선: {len(trendlines_df):,}개\n")

    # 테스트 실행
    results_df = test_quality_filters(df, trendlines_df)

    # 결과 분석
    analyze_results(results_df)

    # 리포트 생성
    print("\n리포트 생성 중...")
    report = generate_report(results_df)

    # 저장
    os.makedirs("output", exist_ok=True)

    with open("output/QUALITY_FILTER_TEST.md", "w", encoding="utf-8") as f:
        f.write(report)

    results_df.to_csv("output/quality_filter_results.csv", index=False)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print("\n저장된 파일:")
    print("  - output/QUALITY_FILTER_TEST.md: 테스트 리포트")
    print("  - output/quality_filter_results.csv: 상세 결과")
