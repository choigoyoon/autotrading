"""
추세선 돌파 후 결과론적 분석
- 최대 반등 얼마
- 최대 하락 얼마
- 어느 방향이 우선되는가
"""

import pandas as pd
import numpy as np

# 기존 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv("output_phase1_labeled.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# 기존 돌파 데이터 로드
breakouts_df = pd.read_csv("output_phase4_breakouts.csv")

# 추세선 돌파만 필터
trendline_breakouts = breakouts_df[breakouts_df['type'].isin(['trendline_up', 'trendline_down'])]

print(f"추세선 돌파: {len(trendline_breakouts):,}개")

# 결과론적 분석 (60봉 내)
results = []

for idx, breakout in trendline_breakouts.iterrows():
    break_idx = breakout['break_idx']
    break_type = breakout['type']

    # 60봉 후까지 데이터
    end_idx = min(break_idx + 60, len(df) - 1)

    if end_idx <= break_idx:
        continue

    future_data = df.iloc[break_idx:end_idx + 1].copy()
    break_price = df.iloc[break_idx]['close']

    # LONG 방향 (trendline_up: 하락추세선 돌파)
    if break_type == 'trendline_up':
        # 최대 반등 (상승)
        max_high = future_data['high'].max()
        max_profit = (max_high - break_price) / break_price * 100
        max_profit_bar = future_data['high'].idxmax() - break_idx

        # 최대 하락 (손실)
        max_low = future_data['low'].min()
        max_drawdown = (max_low - break_price) / break_price * 100
        max_dd_bar = future_data['low'].idxmin() - break_idx

    # SHORT 방향 (trendline_down: 상승추세선 돌파)
    else:
        # 최대 반등 (하락, SHORT 수익)
        max_low = future_data['low'].min()
        max_profit = (break_price - max_low) / break_price * 100
        max_profit_bar = future_data['low'].idxmin() - break_idx

        # 최대 하락 (상승, SHORT 손실)
        max_high = future_data['high'].max()
        max_drawdown = (max_high - break_price) / break_price * 100
        max_dd_bar = future_data['high'].idxmax() - break_idx

    # 어느 것이 먼저 도달?
    profit_first = max_profit_bar < max_dd_bar

    # 최종 결과 (60봉 종가)
    if end_idx < len(df):
        final_price = df.iloc[end_idx]['close']
        if break_type == 'trendline_up':
            final_return = (final_price - break_price) / break_price * 100
        else:
            final_return = (break_price - final_price) / break_price * 100
    else:
        final_return = 0

    results.append({
        'break_idx': break_idx,
        'type': break_type,
        'max_profit': max_profit,
        'max_profit_bar': max_profit_bar,
        'max_drawdown': max_drawdown,
        'max_dd_bar': max_dd_bar,
        'profit_first': profit_first,
        'final_return': final_return
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("결과론적 분석 (60봉 내)")
print("=" * 60)

# 전체 통계
print("\n[전체 통계]")
print(f"평균 최대 반등: {results_df['max_profit'].mean():.3f}%")
print(f"평균 최대 하락: {results_df['max_drawdown'].mean():.3f}%")
print(f"평균 최종 수익: {results_df['final_return'].mean():.3f}%")

# Risk/Reward
avg_profit = results_df['max_profit'].mean()
avg_dd = abs(results_df['max_drawdown'].mean())
rr_ratio = avg_profit / avg_dd if avg_dd > 0 else 0

print(f"\nRisk/Reward: {rr_ratio:.2f}:1")
print(f"  (최대 반등 {avg_profit:.3f}% : 최대 하락 {avg_dd:.3f}%)")

# 어느 것이 먼저?
profit_first_pct = (results_df['profit_first'].sum() / len(results_df) * 100)
print(f"\n반등이 먼저 도달: {profit_first_pct:.1f}%")
print(f"하락이 먼저 도달: {100 - profit_first_pct:.1f}%")

# 최대 반등/하락 도달 시점
print(f"\n평균 최대 반등 도달: {results_df['max_profit_bar'].mean():.1f}봉 후")
print(f"평균 최대 하락 도달: {results_df['max_dd_bar'].mean():.1f}봉 후")

# 분포 분석
print("\n" + "=" * 60)
print("최대 반등 분포")
print("=" * 60)

profit_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
profit_labels = ['0-0.5%', '0.5-1%', '1-1.5%', '1.5-2%', '2-3%', '3-5%', '5%+']

results_df['profit_bucket'] = pd.cut(results_df['max_profit'], bins=profit_bins, labels=profit_labels)
profit_dist = results_df['profit_bucket'].value_counts().sort_index()

for bucket, count in profit_dist.items():
    pct = count / len(results_df) * 100
    print(f"{bucket:>10}: {count:>5}개 ({pct:>5.1f}%) {'█' * int(pct/2)}")

print("\n" + "=" * 60)
print("최대 하락 분포")
print("=" * 60)

dd_bins = [-100, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0]
dd_labels = ['5%+', '3-5%', '2-3%', '1.5-2%', '1-1.5%', '0.5-1%', '0-0.5%']

results_df['dd_bucket'] = pd.cut(results_df['max_drawdown'], bins=dd_bins, labels=dd_labels)
dd_dist = results_df['dd_bucket'].value_counts().sort_index()

for bucket, count in dd_dist.items():
    pct = count / len(results_df) * 100
    print(f"{bucket:>10}: {count:>5}개 ({pct:>5.1f}%) {'█' * int(pct/2)}")

# 타입별 분석
print("\n" + "=" * 60)
print("타입별 비교")
print("=" * 60)

for btype in ['trendline_up', 'trendline_down']:
    type_data = results_df[results_df['type'] == btype]

    print(f"\n[{btype}] ({len(type_data):,}개)")
    print(f"  평균 최대 반등: {type_data['max_profit'].mean():.3f}%")
    print(f"  평균 최대 하락: {type_data['max_drawdown'].mean():.3f}%")
    print(f"  평균 최종 수익: {type_data['final_return'].mean():.3f}%")

    type_rr = type_data['max_profit'].mean() / abs(type_data['max_drawdown'].mean())
    print(f"  Risk/Reward: {type_rr:.2f}:1")

    type_profit_first = (type_data['profit_first'].sum() / len(type_data) * 100)
    print(f"  반등 먼저 도달: {type_profit_first:.1f}%")

# 핵심 인사이트
print("\n" + "=" * 60)
print("💡 핵심 인사이트")
print("=" * 60)

if avg_profit > avg_dd:
    print(f"\n✅ 최대 반등({avg_profit:.3f}%) > 최대 하락({avg_dd:.3f}%)")
    print("   → 전략적으로 가능성 있음")
else:
    print(f"\n❌ 최대 반등({avg_profit:.3f}%) < 최대 하락({avg_dd:.3f}%)")
    print("   → 전략적으로 불리함")

if profit_first_pct > 60:
    print(f"\n✅ 반등이 먼저 도달 ({profit_first_pct:.1f}%)")
    print("   → 빠른 익절 가능")
elif profit_first_pct < 40:
    print(f"\n⚠️  하락이 먼저 도달 ({100-profit_first_pct:.1f}%)")
    print("   → 손절이 먼저 필요")
else:
    print(f"\n⚠️  반등/하락 비슷 (반등 {profit_first_pct:.1f}%)")
    print("   → 타이밍이 중요")

# 최적 청산 지점 추정
print("\n" + "=" * 60)
print("최적 청산 지점 추정")
print("=" * 60)

# 만약 1%에서 청산한다면?
tp_1pct = (results_df['max_profit'] >= 1.0).sum()
tp_1pct_pct = tp_1pct / len(results_df) * 100

print(f"\n목표 1% 도달 가능: {tp_1pct:,}개 ({tp_1pct_pct:.1f}%)")

if tp_1pct_pct > 50:
    print("   → 1% 목표는 현실적")
else:
    print("   → 1% 목표는 비현실적")

# 0.5%에서 청산한다면?
tp_05pct = (results_df['max_profit'] >= 0.5).sum()
tp_05pct_pct = tp_05pct / len(results_df) * 100

print(f"\n목표 0.5% 도달 가능: {tp_05pct:,}개 ({tp_05pct_pct:.1f}%)")

# 저장
results_df.to_csv("output/breakout_max_analysis.csv", index=False)
print("\n저장: output/breakout_max_analysis.csv")
