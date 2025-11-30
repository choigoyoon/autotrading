"""
Phase 3: 시장 구조 분석 - 포지션 관리 전략
======================================================================
목표: 진입 후 시장이 어느 단계인지 파악하고 최적 대응

시장 4단계:
1. 역추세 (하락 중) - L값 형성 단계
2. 추세돌파 - 반등 시작
3. 돌파 후 횡보 - 리테스트 or 추가 상승 준비?
4. 추가 상승 - 본격 상승

분석 항목:
- 각 단계를 어떻게 구분하나?
- 각 단계에서 최적 액션은?
- 횡보 vs 추가 상승을 미리 알 수 있나?
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("Phase 3: 시장 구조 분석 - 포지션 관리")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# L값 로드
df_l = pd.read_csv('l_labels_1h_cross.csv')
df_l['datetime'] = pd.to_datetime(df_l['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)
df_l = df_l[df_l['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"L값: {len(df_l)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 각 L값에서 진입 후 시장 구조 분석
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("L값 진입 후 시장 구조 분석")
print("=" * 70)
print()

market_structures = []

for _, l_row in df_l.iterrows():
    l_idx = int(l_row['l_idx'])
    l_price = l_row['price']

    if l_idx + 100 >= len(df):
        continue

    # 진입: L값 확정 후 (L+1)
    entry_idx = l_idx + 1
    entry_price = df.iloc[entry_idx]['open']

    # 이후 100봉 관찰
    future_50 = df.iloc[entry_idx:entry_idx+50]
    future_100 = df.iloc[entry_idx:entry_idx+100]

    # ═══════════════════════════════════════════════════════════════
    # 단계 1: 초기 반등 (첫 10봉)
    # ═══════════════════════════════════════════════════════════════

    first_10 = df.iloc[entry_idx:entry_idx+10]
    initial_high = first_10['high'].max()
    initial_gain = (initial_high - entry_price) / entry_price * 100

    # 첫 TP 도달 여부
    first_tp_hit = initial_high >= entry_price * 1.02
    first_tp_bar = None
    if first_tp_hit:
        for i, row in first_10.iterrows():
            if row['high'] >= entry_price * 1.02:
                first_tp_bar = i - entry_idx
                break

    # ═══════════════════════════════════════════════════════════════
    # 단계 2: 10-30봉 행동 (횡보 vs 추가 상승)
    # ═══════════════════════════════════════════════════════════════

    bars_10_30 = df.iloc[entry_idx+10:entry_idx+30]

    if len(bars_10_30) > 0:
        range_high = bars_10_30['high'].max()
        range_low = bars_10_30['low'].min()
        range_size = (range_high - range_low) / entry_price * 100

        # 횡보 판단: 레인지가 2% 이내
        is_consolidation = range_size < 2.0

        # 추가 상승 판단: 10봉 고점을 돌파
        additional_rise = range_high > initial_high * 1.01
    else:
        is_consolidation = None
        additional_rise = None
        range_size = None

    # ═══════════════════════════════════════════════════════════════
    # 단계 3: 최종 결과 (50봉)
    # ═══════════════════════════════════════════════════════════════

    final_high_50 = future_50['high'].max()
    final_low_50 = future_50['low'].min()

    max_gain_50 = (final_high_50 - entry_price) / entry_price * 100
    max_dd_50 = (final_low_50 - entry_price) / entry_price * 100

    final_price_50 = future_50.iloc[-1]['close']
    final_return_50 = (final_price_50 - entry_price) / entry_price * 100

    # TP 2% 도달?
    tp_hit_50 = max_gain_50 >= 2.0

    # SL -2% 도달?
    sl_hit_50 = max_dd_50 <= -2.0

    # ═══════════════════════════════════════════════════════════════
    # 패턴 분류
    # ═══════════════════════════════════════════════════════════════

    # 패턴 1: 바로 상승 (첫 10봉에 TP)
    pattern_direct = first_tp_hit and first_tp_bar and first_tp_bar <= 10

    # 패턴 2: 횡보 후 상승 (10-30봉 횡보, 이후 상승)
    pattern_consolidate = (not pattern_direct) and is_consolidation and tp_hit_50

    # 패턴 3: 추가 상승 (10봉 고점 돌파 후 계속 상승)
    pattern_continuation = additional_rise and max_gain_50 > 3.0

    # 패턴 4: 실패 (SL 도달 or 횡보만)
    pattern_fail = sl_hit_50 or (not tp_hit_50 and final_return_50 < 1.0)

    market_structures.append({
        'l_idx': l_idx,
        'entry_price': entry_price,
        'initial_gain': initial_gain,
        'first_tp_hit': first_tp_hit,
        'first_tp_bar': first_tp_bar,
        'is_consolidation': is_consolidation,
        'additional_rise': additional_rise,
        'range_size': range_size,
        'max_gain_50': max_gain_50,
        'max_dd_50': max_dd_50,
        'final_return_50': final_return_50,
        'tp_hit_50': tp_hit_50,
        'sl_hit_50': sl_hit_50,
        'pattern_direct': pattern_direct,
        'pattern_consolidate': pattern_consolidate,
        'pattern_continuation': pattern_continuation,
        'pattern_fail': pattern_fail
    })

df_market = pd.DataFrame(market_structures)

print(f"분석 완료: {len(df_market)}개 L값\n")

# ═══════════════════════════════════════════════════════════════════
# 패턴 통계
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("시장 패턴 분류")
print("=" * 70)
print()

pattern_counts = {
    '바로 상승': df_market['pattern_direct'].sum(),
    '횡보 후 상승': df_market['pattern_consolidate'].sum(),
    '추가 상승': df_market['pattern_continuation'].sum(),
    '실패': df_market['pattern_fail'].sum()
}

total = len(df_market)

for pattern, count in pattern_counts.items():
    pct = count / total * 100
    print(f"{pattern:<15}: {count:>4}개 ({pct:>5.1f}%)")

print()

# ═══════════════════════════════════════════════════════════════════
# 각 패턴 상세 분석
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("패턴별 상세 분석")
print("=" * 70)
print()

# 패턴 1: 바로 상승
direct = df_market[df_market['pattern_direct']]
if len(direct) > 0:
    print(f"【패턴 1: 바로 상승】 ({len(direct)}개)")
    print(f"  평균 TP 도달 시점: {direct['first_tp_bar'].mean():.1f}봉")
    print(f"  평균 최대 상승: {direct['max_gain_50'].mean():.2f}%")
    print(f"  평균 최종 수익: {direct['final_return_50'].mean():.2f}%")
    print(f"  → 빠른 익절 전략 유리\n")

# 패턴 2: 횡보 후 상승
consolidate = df_market[df_market['pattern_consolidate']]
if len(consolidate) > 0:
    print(f"【패턴 2: 횡보 후 상승】 ({len(consolidate)}개)")
    print(f"  평균 횡보 범위: {consolidate['range_size'].mean():.2f}%")
    print(f"  평균 최대 상승: {consolidate['max_gain_50'].mean():.2f}%")
    print(f"  평균 최종 수익: {consolidate['final_return_50'].mean():.2f}%")
    print(f"  → 횡보 중 보유 필요\n")

# 패턴 3: 추가 상승
continuation = df_market[df_market['pattern_continuation']]
if len(continuation) > 0:
    print(f"【패턴 3: 추가 상승】 ({len(continuation)}개)")
    print(f"  평균 최대 상승: {continuation['max_gain_50'].mean():.2f}%")
    print(f"  평균 최종 수익: {continuation['final_return_50'].mean():.2f}%")
    print(f"  → 장기 보유 전략 유리\n")

# 패턴 4: 실패
fail = df_market[df_market['pattern_fail']]
if len(fail) > 0:
    print(f"【패턴 4: 실패】 ({len(fail)}개)")
    print(f"  SL 도달: {fail['sl_hit_50'].sum()}개")
    print(f"  평균 최종 수익: {fail['final_return_50'].mean():.2f}%")
    print(f"  → 조기 청산 필요\n")

# ═══════════════════════════════════════════════════════════════════
# 횡보 vs 추가 상승 예측
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("횡보 vs 추가 상승 구분 가능성")
print("=" * 70)
print()

print("초기 10봉 상승률로 예측:")
print()

# 초기 상승률별 그룹화
bins = [0, 1, 2, 3, 5, 100]
labels = ['0-1%', '1-2%', '2-3%', '3-5%', '5%+']
df_market['initial_gain_bucket'] = pd.cut(df_market['initial_gain'], bins=bins, labels=labels)

for bucket in labels:
    bucket_data = df_market[df_market['initial_gain_bucket'] == bucket]

    if len(bucket_data) > 0:
        continuation_pct = bucket_data['pattern_continuation'].sum() / len(bucket_data) * 100
        consolidate_pct = bucket_data['pattern_consolidate'].sum() / len(bucket_data) * 100
        fail_pct = bucket_data['pattern_fail'].sum() / len(bucket_data) * 100

        print(f"초기 상승 {bucket}:")
        print(f"  추가 상승: {continuation_pct:>5.1f}%")
        print(f"  횡보 후 상승: {consolidate_pct:>5.1f}%")
        print(f"  실패: {fail_pct:>5.1f}%")
        print()

# ═══════════════════════════════════════════════════════════════════
# 포지션 관리 전략 제안
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("포지션 관리 전략 제안")
print("=" * 70)
print()

print("【전략 A: 기계적 TP/SL】")
print("  - 진입 후 TP 2% / SL -2%")
print("  - 장점: 단순, 감정 개입 없음")
print("  - 단점: 추가 상승 놓침")
print()

print("【전략 B: 단계별 대응】")
print("  1단계 (0-10봉):")
print("    - 초기 상승 > 3% → 부분 익절 50%")
print("    - 초기 상승 < 1% → 경계 모드")
print()
print("  2단계 (10-30봉):")
print("    - 횡보 (레인지 < 2%) → 보유")
print("    - 고점 돌파 → 추가 상승 기대, 보유")
print("    - 고점 하회 + 하락 → 청산")
print()
print("  3단계 (30-50봉):")
print("    - TP 2% 미도달 → 청산")
print("    - TP 도달 → 트레일링 스탑")
print()

print("【전략 C: 적응형】")
print("  - 초기 10봉 상승률 확인")
print("  - 3%+ → 빠른 익절")
print("  - 1-3% → 횡보 대기")
print("  - <1% → 조기 청산 고려")
print()

# ═══════════════════════════════════════════════════════════════════
# 백테스트 비교
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("전략 백테스트 비교")
print("=" * 70)
print()

# 전략 A: 기계적 TP/SL
strategy_a_returns = []
for _, row in df_market.iterrows():
    if row['tp_hit_50']:
        strategy_a_returns.append(2.0)
    elif row['sl_hit_50']:
        strategy_a_returns.append(-2.0)
    else:
        strategy_a_returns.append(row['final_return_50'])

# 전략 B: 단계별 (간단 버전)
strategy_b_returns = []
for _, row in df_market.iterrows():
    if row['pattern_direct']:
        # 빠른 상승 → 조기 익절
        strategy_b_returns.append(min(2.0, row['max_gain_50']))
    elif row['pattern_continuation']:
        # 추가 상승 → 장기 보유
        strategy_b_returns.append(min(5.0, row['max_gain_50']))
    else:
        # 기본 TP/SL
        if row['tp_hit_50']:
            strategy_b_returns.append(2.0)
        elif row['sl_hit_50']:
            strategy_b_returns.append(-2.0)
        else:
            strategy_b_returns.append(row['final_return_50'])

print(f"전략 A (기계적 TP/SL):")
print(f"  평균 수익: {np.mean(strategy_a_returns):+.2f}%")
print(f"  승률: {len([r for r in strategy_a_returns if r > 0])/len(strategy_a_returns)*100:.1f}%")
print()

print(f"전략 B (단계별 대응):")
print(f"  평균 수익: {np.mean(strategy_b_returns):+.2f}%")
print(f"  승률: {len([r for r in strategy_b_returns if r > 0])/len(strategy_b_returns)*100:.1f}%")
print()

improvement = np.mean(strategy_b_returns) - np.mean(strategy_a_returns)
print(f"개선: {improvement:+.2f}%p")

print()
print("✅ Phase 3 완료")
