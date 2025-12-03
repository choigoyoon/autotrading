"""
L값 라벨의 미래데이터 검증
======================================================================
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("L값 미래데이터 검증")
print("=" * 70)
print()

# L값 데이터 로드
df_l = pd.read_csv('l_labels_1h_cross.csv')
df_l['datetime'] = pd.to_datetime(df_l['datetime'])
df_l['dead_cross_time'] = pd.to_datetime(df_l['dead_cross_time'])
df_l['golden_cross_time'] = pd.to_datetime(df_l['golden_cross_time'])

print(f"총 L값: {len(df_l)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 검증: L값 시점과 Golden Cross 시점 비교
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("미래데이터 사용 검증")
print("=" * 70)
print()

# L값이 Golden Cross보다 앞선 경우 (정상)
df_l['l_before_golden'] = df_l['datetime'] < df_l['golden_cross_time']
before_count = df_l['l_before_golden'].sum()

print(f"L값이 Golden Cross 이전: {before_count}개 ({before_count/len(df_l)*100:.1f}%)")
print()

# 시간차 분석
df_l['hours_to_golden'] = (df_l['golden_cross_time'] - df_l['datetime']).dt.total_seconds() / 3600

print("L값 ~ Golden Cross 시간차:")
print(f"  평균: {df_l['hours_to_golden'].mean():.1f}시간")
print(f"  중앙값: {df_l['hours_to_golden'].median():.1f}시간")
print(f"  최소: {df_l['hours_to_golden'].min():.1f}시간")
print(f"  최대: {df_l['hours_to_golden'].max():.1f}시간")
print()

print("=" * 70)
print("⚠️  핵심 문제")
print("=" * 70)
print()
print("L값 라벨은 Golden Cross 이후에야 확정됨:")
print()
print("  예시:")
print("  - 2024-01-01 12:00: Dead Cross 발생")
print("  - 2024-01-02 08:00: L값 (최저점) ← 진입 시점")
print("  - 2024-01-05 14:00: Golden Cross 발생 ← 이때야 L값 확정!")
print()
print(f"  → 평균 {df_l['hours_to_golden'].mean():.0f}시간 후에야 L값 확정")
print(f"  → 실시간에서는 L값인지 알 수 없음!")
print()

# ═══════════════════════════════════════════════════════════════════
# 실시간 시뮬레이션
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("실시간 거래 시나리오")
print("=" * 70)
print()

print("현재 방식 (백테스트):")
print("  1. Dead Cross 발생")
print("  2. [미래를 봄] Golden Cross가 언제 올지 확인")
print("  3. Dead ~ Golden 사이 최저점 = L값")
print("  4. L값에서 진입")
print("  ✅ 승률: 87.77%")
print()

print("실시간 거래 (미래 정보 없음):")
print("  1. Dead Cross 발생")
print("  2. 현재 캔들이 L값인지 모름 (Golden Cross 안 옴)")
print("  3. 매 캔들마다 '혹시 여기가 L값?'")
print("  4. Golden Cross 오면 '아, 저기가 L값이었구나' (이미 늦음)")
print("  ❓ 승률: ???%")
print()

# ═══════════════════════════════════════════════════════════════════
# 해결 방안
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("💡 해결 방안")
print("=" * 70)
print()

print("방안 1: Dead Cross 이후 즉시 진입")
print("  - Dead Cross 직후 RSI<30 AND MACD<0 조건 충족 시 진입")
print("  - L값 개념 제거")
print("  - 미래 정보 불필요")
print("  ⚠️ 조기 진입으로 승률 하락 가능")
print()

print("방안 2: N-봉 지연 후 최저점 진입")
print("  - Dead Cross 후 10~20봉 관찰")
print("  - 관찰 기간 중 최저점에서 진입")
print("  - Golden Cross 기다릴 필요 없음")
print("  ⚠️ 여전히 '진짜 L값'은 아닐 수 있음")
print()

print("방안 3: MACD Hist 0-Cross 직접 사용")
print("  - 15분 MACD Hist가 음수→양수 전환 시 진입")
print("  - L값 라벨 불필요")
print("  - 즉시 확인 가능")
print("  ⚠️ 기존 백테스트와 다른 전략")
print()

print("=" * 70)
print("🎯 결론")
print("=" * 70)
print()
print("1. **현재 백테스트는 미래데이터 사용** ❌")
print("   - L값 라벨은 Golden Cross 후에야 확정")
print(f"   - 평균 {df_l['hours_to_golden'].mean():.0f}시간 미래를 봄")
print()
print("2. **87.77% 승률은 신뢰 불가** ❌")
print("   - 실시간에서는 달성 불가능")
print()
print("3. **전략 재설계 필요** ⚠️")
print("   - 미래 정보 없이 진입 가능한 방식")
print("   - 위 3가지 방안 중 선택 후 재백테스트")
print()

print("=" * 70)
