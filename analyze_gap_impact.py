"""
갭 분석: 백테스트 vs 실제 진입 가격 차이
"""

import pandas as pd
import numpy as np

print("="*60)
print("갭 분석: Close-to-Open 차이")
print("="*60)

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"총 데이터: {len(df):,}개")

# Close-to-Open 갭 계산
df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
df['gap_abs'] = df['gap'].abs()

# 통계
print(f"\n갭 통계 (15분봉):")
print(f"  평균 갭: {df['gap'].mean():.4f}%")
print(f"  절대 갭: {df['gap_abs'].mean():.4f}%")
print(f"  최대 상승 갭: {df['gap'].max():.2f}%")
print(f"  최대 하락 갭: {df['gap'].min():.2f}%")

# 갭 분포
print(f"\n갭 크기 분포:")
print(f"  0.0% 이하: {(df['gap_abs'] <= 0.0).sum():,}개 ({(df['gap_abs'] <= 0.0).sum() / len(df) * 100:.1f}%)")
print(f"  0.1% 이하: {(df['gap_abs'] <= 0.1).sum():,}개 ({(df['gap_abs'] <= 0.1).sum() / len(df) * 100:.1f}%)")
print(f"  0.2% 이하: {(df['gap_abs'] <= 0.2).sum():,}개 ({(df['gap_abs'] <= 0.2).sum() / len(df) * 100:.1f}%)")
print(f"  0.5% 이하: {(df['gap_abs'] <= 0.5).sum():,}개 ({(df['gap_abs'] <= 0.5).sum() / len(df) * 100:.1f}%)")
print(f"  1.0% 이상: {(df['gap_abs'] >= 1.0).sum():,}개 ({(df['gap_abs'] >= 1.0).sum() / len(df) * 100:.1f}%)")

# 백테스트 vs 실제 영향
breakouts = pd.read_csv('output_phase4_breakouts.csv')
print(f"\n돌파 이벤트: {len(breakouts):,}개")

# 돌파 시점의 갭 분석
gap_at_breakout = []
for idx, breakout in breakouts.iterrows():
    break_idx = breakout['break_idx']

    # 다음 봉의 갭
    if break_idx + 1 < len(df):
        next_gap = df.iloc[break_idx + 1]['gap']
        gap_at_breakout.append(next_gap)

gap_at_breakout = pd.Series(gap_at_breakout).dropna()

print(f"\n돌파 직후 갭 (Bar N → Bar N+1):")
print(f"  평균: {gap_at_breakout.mean():.4f}%")
print(f"  절대값: {gap_at_breakout.abs().mean():.4f}%")
print(f"  표준편차: {gap_at_breakout.std():.4f}%")

# 실제 진입 vs 백테스트 진입 차이
print(f"\n백테스트 vs 실제 영향:")
print(f"  백테스트 진입: Bar N close")
print(f"  실제 진입: Bar N+1 open")
print(f"  평균 차이: {gap_at_breakout.mean():.4f}%")
print(f"  TP 2% 기준: {gap_at_breakout.mean() / 2.0 * 100:.2f}% 영향")

# 승률 영향 추정
# 롱 진입 시: 양의 갭은 불리, 음의 갭은 유리
# 숏 진입 시: 반대
long_breakouts = breakouts[breakouts['type'] == 'trendline_up']
print(f"\n롱 돌파: {len(long_breakouts):,}개")

# 예상 슬리피지
slippage = gap_at_breakout.abs().mean()
print(f"\n실전 적용 시 예상 슬리피지:")
print(f"  평균: {slippage:.4f}%")
print(f"  TP 2% 대비: {slippage / 2.0 * 100:.2f}%")
print(f"  승률 90% → 예상 승률: {90 - slippage * 5:.1f}%")

print(f"\n결론:")
if gap_at_breakout.abs().mean() < 0.05:
    print(f"  ✅ 갭 영향 미미 (평균 {gap_at_breakout.abs().mean():.3f}%)")
    print(f"  ✅ 백테스트 신뢰 가능")
else:
    print(f"  ⚠️ 갭 영향 존재 (평균 {gap_at_breakout.abs().mean():.3f}%)")
    print(f"  ⚠️ 실제 성과 다를 수 있음")
