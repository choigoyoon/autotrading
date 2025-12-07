import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 MTF 패턴 심층 분석 - 왜 MTF 정렬이 오히려 나쁜가?")
print("=" * 80)

# Load MTF analysis results
mtf_df = pd.read_csv('mtf_context_analysis.csv')
mtf_df['entry_time'] = pd.to_datetime(mtf_df['entry_time'])

print(f"\n총 거래: {len(mtf_df)}개")

# Key finding: MTF aligned actually WORSE
print("\n" + "=" * 80)
print("핵심 발견: MTF 정렬 시 오히려 성과가 나쁨!")
print("=" * 80)

mtf_aligned = mtf_df[mtf_df['mtf_aligned_up'] == True]
not_aligned = mtf_df[mtf_df['mtf_aligned_up'] == False]

print(f"\n✅ MTF 상승 정렬 (4H 상승 + 1H 상승): {len(mtf_aligned)}개")
print(f"  - 평균 PnL: {mtf_aligned['pnl_pct'].mean():.2f}%")
print(f"  - TP2 성공률: {(mtf_aligned['exit_reason'] == 'TP2_Full').sum() / len(mtf_aligned) * 100:.1f}%")
print(f"  - SL 비율: {(mtf_aligned['exit_reason'] == 'SL').sum() / len(mtf_aligned) * 100:.1f}%")

print(f"\n❌ MTF 비정렬 (나머지): {len(not_aligned)}개")
print(f"  - 평균 PnL: {not_aligned['pnl_pct'].mean():.2f}%")
print(f"  - TP2 성공률: {(not_aligned['exit_reason'] == 'TP2_Full').sum() / len(not_aligned) * 100:.1f}%")
print(f"  - SL 비율: {(not_aligned['exit_reason'] == 'SL').sum() / len(not_aligned) * 100:.1f}%")

# 가설: "MTF 정렬 = 이미 많이 올랐다 = 늦은 진입"
print("\n" + "=" * 80)
print("가설 검증: MTF 정렬 시 = 이미 많이 상승한 상태?")
print("=" * 80)

print(f"\nMTF 정렬 시:")
print(f"  - 평균 1H 모멘텀: {mtf_aligned['momentum_1h'].mean():.2f}%")
print(f"  - 평균 4H 모멘텀: {mtf_aligned['momentum_4h'].mean():.2f}%")
print(f"  - 평균 1H 연속 상승: {mtf_aligned['consecutive_up_1h'].mean():.1f}개")

print(f"\nMTF 비정렬 시:")
print(f"  - 평균 1H 모멘텀: {not_aligned['momentum_1h'].mean():.2f}%")
print(f"  - 평균 4H 모멘텀: {not_aligned['momentum_4h'].mean():.2f}%")
print(f"  - 평균 1H 연속 상승: {not_aligned['consecutive_up_1h'].mean():.1f}개")

# 더 세밀한 분류: 횡보/전환 상태에서의 성과
print("\n" + "=" * 80)
print("세밀 분석: 1H 추세 상태별")
print("=" * 80)

for trend_1h in mtf_df['trend_1h'].unique():
    subset = mtf_df[mtf_df['trend_1h'] == trend_1h]
    
    print(f"\n1H {trend_1h} ({len(subset)}개):")
    print(f"  - 평균 PnL: {subset['pnl_pct'].mean():.2f}%")
    print(f"  - 총 PnL: {subset['pnl_pct'].sum():.2f}%")
    print(f"  - TP2 성공률: {(subset['exit_reason'] == 'TP2_Full').sum() / len(subset) * 100:.1f}%")
    print(f"  - SL 비율: {(subset['exit_reason'] == 'SL').sum() / len(subset) * 100:.1f}%")
    print(f"  - 평균 1H 모멘텀: {subset['momentum_1h'].mean():.2f}%")
    print(f"  - 평균 4H 모멘텀: {subset['momentum_4h'].mean():.2f}%")
    
    # 이 안에서 4H 추세별로 다시 분리
    for trend_4h in ['상승추세', '횡보/전환']:
        sub_subset = subset[subset['trend_4h'] == trend_4h]
        if len(sub_subset) == 0:
            continue
        print(f"    - 4H {trend_4h} ({len(sub_subset)}개):")
        print(f"      - 평균 PnL: {sub_subset['pnl_pct'].mean():.2f}%")
        print(f"      - TP2 성공률: {(sub_subset['exit_reason'] == 'TP2_Full').sum() / len(sub_subset) * 100:.1f}%")

# 최적 조합 찾기
print("\n" + "=" * 80)
print("최적 조합 탐색")
print("=" * 80)

combinations = [
    ('횡보/전환', '횡보/전환'),
    ('횡보/전환', '상승추세'),
    ('상승추세', '횡보/전환'),
    ('상승추세', '상승추세'),
]

best_combo = None
best_pnl = -999999

for trend_1h, trend_4h in combinations:
    subset = mtf_df[(mtf_df['trend_1h'] == trend_1h) & (mtf_df['trend_4h'] == trend_4h)]
    if len(subset) == 0:
        continue
    
    avg_pnl = subset['pnl_pct'].mean()
    total_pnl = subset['pnl_pct'].sum()
    tp2_rate = (subset['exit_reason'] == 'TP2_Full').sum() / len(subset) * 100
    sl_rate = (subset['exit_reason'] == 'SL').sum() / len(subset) * 100
    
    print(f"\n1H={trend_1h} & 4H={trend_4h} ({len(subset)}개):")
    print(f"  - 평균 PnL: {avg_pnl:.2f}%")
    print(f"  - 총 PnL: {total_pnl:.2f}%")
    print(f"  - TP2 성공률: {tp2_rate:.1f}%")
    print(f"  - SL 비율: {sl_rate:.1f}%")
    
    if avg_pnl > best_pnl:
        best_pnl = avg_pnl
        best_combo = (trend_1h, trend_4h, len(subset))

print(f"\n" + "=" * 80)
print(f"🏆 최적 조합: 1H={best_combo[0]} & 4H={best_combo[1]}")
print(f"   거래수: {best_combo[2]}개, 평균 PnL: {best_pnl:.2f}%")
print("=" * 80)

# 역설적 발견: 실제로는 15분 타이밍이 더 중요?
print("\n" + "=" * 80)
print("💡 핵심 통찰")
print("=" * 80)

print(f"""
MTF 정렬 시 오히려 성과가 나쁜 이유:

1. MTF 정렬 = 이미 4H + 1H 모두 상승 중
   → 15분봉 H3 돌파 시점은 이미 "늦은 진입"
   → 상승 동력이 소진된 후

2. 횡보/전환 상태 = 15분 돌파 시점이 "진짜 전환점"
   → 상위 타임프레임이 아직 확정 안 됨
   → 15분 H3 돌파가 실제 전환 시작점

3. 역추세 전략의 본질:
   - 4H/1H 하락 중 → 15분 상승 전환 포착 (이것이 진짜 역추세)
   - 하지만 현재는 4H/1H 하락 시 진입 자체를 안 함 (0건)
   
4. 실제 최적 조합:
   - 1H 횡보/전환 + 4H 횡보/전환
   - 여기서 15분 H3 돌파가 "새로운 움직임의 시작"
""")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)
