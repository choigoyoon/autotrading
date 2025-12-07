import pandas as pd
import numpy as np

# Load backtest results
df = pd.read_csv('backtest_complete_results.csv')

print("=" * 80)
print("📊 진입 위치 vs SL 위치 문제 분석")
print("=" * 80)
print()

# 1. 전체 결과 분포
print("📈 전체 백테스트 결과:")
print(df['exit_reason'].value_counts())
print()
print(f"총 거래: {len(df)}개")
print()

# 2. SL 손실 케이스 분석
sl_cases = df[df['exit_reason'] == 'SL'].copy()
tp1_cases = df[df['exit_reason'] == 'TP1_Breakeven'].copy()
tp2_cases = df[df['exit_reason'] == 'TP2_Full'].copy()

print("=" * 80)
print("🔴 SL 손실 케이스 분석")
print("=" * 80)
print(f"총 {len(sl_cases)}개 ({len(sl_cases)/len(df)*100:.1f}%)")
print()

# SL 손실 통계
print(f"SL 케이스 평균 power_score: {sl_cases['power_score'].mean():.2f}")
print(f"SL 케이스 중앙값 power_score: {sl_cases['power_score'].median():.1f}")
print()

# Power score별 SL 분포
print("Power Score별 SL 발생률:")
for score in sorted(df['power_score'].unique()):
    score_df = df[df['power_score'] == score]
    sl_rate = len(score_df[score_df['exit_reason'] == 'SL']) / len(score_df) * 100
    print(f"  Score {score}점: {sl_rate:.1f}% SL ({len(score_df[score_df['exit_reason'] == 'SL'])}/{len(score_df)})")
print()

# 3. TP2 승리 케이스 분석
print("=" * 80)
print("🟢 TP2 Full 승리 케이스 분석")
print("=" * 80)
print(f"총 {len(tp2_cases)}개 ({len(tp2_cases)/len(df)*100:.1f}%)")
print()

print(f"TP2 케이스 평균 power_score: {tp2_cases['power_score'].mean():.2f}")
print(f"TP2 케이스 중앙값 power_score: {tp2_cases['power_score'].median():.1f}")
print()

# Power score별 TP2 성공률
print("Power Score별 TP2 성공률:")
for score in sorted(df['power_score'].unique()):
    score_df = df[df['power_score'] == score]
    tp2_rate = len(score_df[score_df['exit_reason'] == 'TP2_Full']) / len(score_df) * 100
    print(f"  Score {score}점: {tp2_rate:.1f}% TP2 ({len(score_df[score_df['exit_reason'] == 'TP2_Full'])}/{len(score_df)})")
print()

# 4. Power Score vs 수익률 상관관계
print("=" * 80)
print("📊 Power Score vs 평균 수익률")
print("=" * 80)
print()

for score in sorted(df['power_score'].unique()):
    score_df = df[df['power_score'] == score]
    avg_pnl = score_df['pnl_pct'].mean()
    win_rate = len(score_df[score_df['pnl_pct'] > 0]) / len(score_df) * 100
    print(f"Score {score}점: 평균 {avg_pnl:+.2f}% | 승률 {win_rate:.1f}% | 거래 {len(score_df)}개")
print()

# 5. 결론 - 진입 위치 문제 판정
print("=" * 80)
print("💡 진입 위치 문제 vs SL 위치 문제 판정")
print("=" * 80)
print()

# Power score가 낮을수록 SL이 많으면 = 진입 위치 문제
# Power score가 높아도 SL이 많으면 = SL 위치 문제

low_score_sl_rate = len(df[(df['power_score'] <= 6) & (df['exit_reason'] == 'SL')]) / len(df[df['power_score'] <= 6]) * 100
high_score_sl_rate = len(df[(df['power_score'] >= 7) & (df['exit_reason'] == 'SL')]) / len(df[df['power_score'] >= 7]) * 100

print(f"낮은 Power Score (≤6점) SL 비율: {low_score_sl_rate:.1f}%")
print(f"높은 Power Score (≥7점) SL 비율: {high_score_sl_rate:.1f}%")
print()

if low_score_sl_rate > 50 and high_score_sl_rate < 30:
    print("🎯 판정: **진입 위치 문제**")
    print("   → Power score가 낮을 때 SL이 많음")
    print("   → 해결책: Power score 필터 강화 (5점 → 7점 이상)")
elif high_score_sl_rate > 30:
    print("🎯 판정: **SL 위치 문제**")
    print("   → Power score가 높아도 SL이 많음")
    print("   → 해결책: SL 여유 확대 (-0.5% → -1.0%)")
else:
    print("🎯 판정: **혼합 문제**")
    print("   → 진입 위치 & SL 위치 둘 다 개선 필요")
    print("   → 해결책: Power score 7점 이상 + SL -1.0%")
print()

# 6. 시뮬레이션: Power Score 7점 이상만 진입한다면?
print("=" * 80)
print("🔬 시뮬레이션 1: Power Score 7점 이상만 진입")
print("=" * 80)
print()

filtered_df = df[df['power_score'] >= 7]
print(f"거래 횟수: {len(df)}개 → {len(filtered_df)}개 ({len(filtered_df)/len(df)*100:.1f}%)")
print(f"총 PnL: {df['pnl_pct'].sum():.2f}% → {filtered_df['pnl_pct'].sum():.2f}%")
print(f"평균 PnL: {df['pnl_pct'].mean():.2f}% → {filtered_df['pnl_pct'].mean():.2f}%")
print(f"승률: {len(df[df['pnl_pct'] > 0])/len(df)*100:.1f}% → {len(filtered_df[filtered_df['pnl_pct'] > 0])/len(filtered_df)*100:.1f}%")
print()

# SL 비율
print(f"SL 비율: {len(df[df['exit_reason'] == 'SL'])/len(df)*100:.1f}% → {len(filtered_df[filtered_df['exit_reason'] == 'SL'])/len(filtered_df)*100:.1f}%")
print(f"TP2 비율: {len(df[df['exit_reason'] == 'TP2_Full'])/len(df)*100:.1f}% → {len(filtered_df[filtered_df['exit_reason'] == 'TP2_Full'])/len(filtered_df)*100:.1f}%")
print()

# 7. 연도별 power score 분포
print("=" * 80)
print("📅 연도별 Power Score 분포 & 수익률")
print("=" * 80)
print()

for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    print(f"{year}년: 거래 {len(year_df)}개 | 평균 score {year_df['power_score'].mean():.1f}점 | 총 PnL {year_df['pnl_pct'].sum():.2f}%")
print()

# 특히 2021년 분석 (최대 손실 연도)
print("🔍 2021년 상세 분석 (최대 손실 연도):")
year_2021 = df[df['year'] == 2021]
print(f"  거래: {len(year_2021)}개")
print(f"  평균 power_score: {year_2021['power_score'].mean():.2f}점")
print(f"  SL 비율: {len(year_2021[year_2021['exit_reason'] == 'SL'])/len(year_2021)*100:.1f}%")
print(f"  TP2 비율: {len(year_2021[year_2021['exit_reason'] == 'TP2_Full'])/len(year_2021)*100:.1f}%")
print(f"  총 PnL: {year_2021['pnl_pct'].sum():.2f}%")
print()

# 8. 최종 결론 및 권장사항
print("=" * 80)
print("🎯 최종 결론")
print("=" * 80)
print()

print("📌 문제 원인:")
if low_score_sl_rate > 50:
    print("  1. **진입 위치 문제가 주요 원인** (낮은 power score에서 SL 집중)")
    print(f"     - Power score ≤6점: SL {low_score_sl_rate:.1f}%")
if high_score_sl_rate > 20:
    print("  2. **SL 위치도 일부 문제** (높은 power score에서도 SL 발생)")
    print(f"     - Power score ≥7점: SL {high_score_sl_rate:.1f}%")
print()

print("🚀 권장 개선안:")
print()
print("  ✅ Option 1: Power Score 필터 강화")
print(f"     - 현재: 5점 이상 진입 ({len(df)}개 거래)")
print(f"     - 변경: 7점 이상 진입 ({len(filtered_df)}개 거래)")
print(f"     - 효과: 평균 PnL {df['pnl_pct'].mean():.2f}% → {filtered_df['pnl_pct'].mean():.2f}%")
print()

print("  ✅ Option 2: SL 여유 확대 (별도 백테스트 필요)")
print("     - 현재: H3 -0.5%")
print("     - 변경: H3 -1.0%")
print("     - 예상: SL 비율 감소, 승률 상승")
print()

print("  ✅ Option 3: 혼합 전략 (추천)")
print("     - Power score 7점 이상 + SL -1.0%")
print("     - 예상: 최대 성능 개선")
print()

# Save analysis
output_df = df[['entry_time', 'exit_reason', 'pnl_pct', 'power_score', 'year']].copy()
output_df = output_df.sort_values('entry_time')
output_df.to_csv('entry_sl_problem_analysis.csv', index=False)
print(f"✅ 분석 결과 저장: entry_sl_problem_analysis.csv")

