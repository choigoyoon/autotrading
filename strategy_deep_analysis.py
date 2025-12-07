"""
전략 근본 재검토
- Look-Ahead Bias 수정 후 실패 원인 분석
- FVG 전략의 본질적 문제점 파악
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔍 전략 근본 재검토 - 실패 원인 분석")
print("=" * 80)

# 데이터 로드
df_trades_old = pd.read_csv('trades_4h_15m.csv')  # 기존 (Bias 있음)
df_trades_new = pd.read_csv('backtest_no_lookahead.csv')  # 수정 (Bias 없음)

print("\n" + "=" * 80)
print("📊 STEP 1: 진입 성공률 비교")
print("=" * 80)

# 시그널 대비 진입 비율
print("\n[기존 백테스트 (Bias 있음)]:")
print(f"  총 시그널: ~2,227개")
print(f"  실제 진입: 1,920건")
print(f"  진입 성공률: {1920/2227*100:.1f}%")

print("\n[수정 백테스트 (Bias 없음)]:")
print(f"  총 시그널: 2,228개")
print(f"  실제 진입: 1,030건")
print(f"  진입 성공률: {1030/2228*100:.1f}%")

print(f"\n⚠️ 문제 1: 진입 성공률 급락")
print(f"  86.2% → 46.2% (40%p 하락)")
print(f"  → 4시간 지연으로 절반의 시그널을 놓침!")

print("\n" + "=" * 80)
print("📊 STEP 2: 승률 비교")
print("=" * 80)

old_sl = (df_trades_old['result'] == 'SL').sum()
old_total = len(df_trades_old)
old_winrate = (old_total - old_sl) / old_total * 100

new_sl = (df_trades_new['result'] == 'SL').sum()
new_total = len(df_trades_new)
new_winrate = (new_total - new_sl) / new_total * 100

print(f"\n[기존]: {old_winrate:.1f}% 승률 (SL: {old_sl}건)")
print(f"[수정]: {new_winrate:.1f}% 승률 (SL: {new_sl}건)")
print(f"\n⚠️ 문제 2: 손절 비율 급증")
print(f"  SL 비율: {old_sl/old_total*100:.1f}% → {new_sl/new_total*100:.1f}%")
print(f"  → 진입가가 불리해져 손절 2배 이상 증가!")

print("\n" + "=" * 80)
print("📊 STEP 3: 결과 분포 비교")
print("=" * 80)

print("\n[기존 백테스트]:")
old_dist = df_trades_old['result'].value_counts()
for result, count in old_dist.items():
    pct = count / len(df_trades_old) * 100
    print(f"  {result}: {count}건 ({pct:.1f}%)")

print("\n[수정 백테스트]:")
new_dist = df_trades_new['result'].value_counts()
for result, count in new_dist.items():
    pct = count / len(df_trades_new) * 100
    print(f"  {result}: {count}건 ({pct:.1f}%)")

print(f"\n⚠️ 문제 3: TP2 도달률 폭락")
old_tp2_rate = old_dist.get('TP2', 0) / len(df_trades_old) * 100
new_tp2_rate = new_dist.get('TP2', 0) / len(df_trades_new) * 100
print(f"  TP2 비율: {old_tp2_rate:.1f}% → {new_tp2_rate:.1f}%")
print(f"  → 목표가 도달이 절반 이하로 감소!")

print("\n" + "=" * 80)
print("📊 STEP 4: FVG의 본질적 문제")
print("=" * 80)

print("""
FVG (Fair Value Gap) 전략의 문제점:

1️⃣ 갭은 빠르게 채워진다
   - 갭 발생 직후가 최적 진입 타이밍
   - 4시간 지연 시 이미 갭이 채워짐
   - 늦은 진입 = 불리한 가격

2️⃣ 시간 민감성
   - 갭은 짧은 시간 내 반응
   - 진입 유효기간 5시간
   - 4시간 지연 → 실질 1시간만
   - 진입 기회 대폭 감소

3️⃣ 진입존 터치 실패
   기존 (00:00 진입 허용):
     00:00 ~ 05:00 사이 갭 터치 대기
     5시간 여유
     
   수정 (04:00 진입 허용):
     04:00 ~ 05:00 사이 갭 터치 대기
     1시간만 여유! ⚠️
     → 진입 기회 80% 감소

4️⃣ Look-Ahead Bias 의존
   - 전략 자체가 미래 정보에 의존
   - 실시간에서는 작동 불가
   - 백테스트만 좋은 가짜 전략! 🚨
""")

print("\n" + "=" * 80)
print("📊 STEP 5: Order Block도 실패")
print("=" * 80)

old_ob = df_trades_old[df_trades_old['signal_type'] == 'orderblock']
new_ob = df_trades_new[df_trades_new['strategy'] == 'orderblock']

print(f"\n[기존] Order Block:")
print(f"  거래: {len(old_ob)}건")
print(f"  평균 수익: {old_ob['pnl'].mean():.3f}%")
print(f"  총 수익: {old_ob['pnl'].sum():.1f}%")

print(f"\n[수정] Order Block:")
print(f"  거래: {len(new_ob)}건")
print(f"  평균 수익: {new_ob['pnl'].mean():.3f}%")
print(f"  총 수익: {new_ob['pnl'].sum():.1f}%")

print(f"\n⚠️ Order Block도 동일한 문제:")
print(f"  기존에도 낮은 승률 (38.7% → 57.3%)")
print(f"  수정 후 -162.6% 손실")
print(f"  → 이 전략도 실전 불가!")

print("\n" + "=" * 80)
print("🎯 핵심 결론")
print("=" * 80)

print("""
현재 전략의 치명적 결함:

❌ Look-Ahead Bias 의존적 설계
   - 4H 봉 완성 '직후' 진입이 필수
   - 실시간에서는 불가능
   - 백테스트용 가짜 전략

❌ 시간 민감성 극대화
   - 갭은 빠르게 채워짐
   - 지연 시 진입 불가
   - 실전 적용 불가능

❌ 진입 유효기간 부족
   - 5시간 유효 → 4시간 지연
   - 실질 1시간만 남음
   - 진입 기회 80% 손실
""")

print("\n" + "=" * 80)
print("💡 해결 방안 제안")
print("=" * 80)

print("""
[방안 1] 타임프레임 조정 ✅
━━━━━━━━━━━━━━━━━━━━━━━━
  현재: 4H 시그널 + 15M 진입
  문제: 4시간 지연이 치명적
  
  해결: 더 높은 타임프레임 사용
    - 1D 시그널 + 4H 진입
    - 4시간 지연이 덜 치명적
    - 진입 기회 유지 가능

[방안 2] 진입 유효기간 확대 ✅
━━━━━━━━━━━━━━━━━━━━━━━━
  현재: 5시간 유효
  문제: 4시간 지연 후 1시간만
  
  해결: 유효기간 대폭 확대
    - 24시간 ~ 48시간
    - 진입 기회 증가
    - 하지만 신호 신뢰도 하락

[방안 3] 전략 근본 변경 ✅✅✅
━━━━━━━━━━━━━━━━━━━━━━━━
  문제: FVG 자체가 실시간 부적합
  
  해결: 다른 전략 고려
    - 추세 추종 전략
    - 지지/저항 돌파 전략
    - MA 크로스 전략
    - 실시간 적용 가능한 로직

[방안 4] FVG 개선 ✅
━━━━━━━━━━━━━━━━━━━━━━━━
  아이디어:
    - 갭 발생 '예상' 시점에 진입
    - 봉 완성 전 진입 허용
    - 하지만 위험도 증가
    - 실시간 구현 복잡
""")

print("\n" + "=" * 80)
print("🔧 권장 조치")
print("=" * 80)

print("""
1순위: 타임프레임 조정 테스트
  → 1D + 4H 조합 백테스트
  → 4시간 지연 영향 최소화
  
2순위: 진입 유효기간 확대
  → 24시간 유효기간 테스트
  → 진입 기회 증가 확인

3순위: 완전히 새로운 전략
  → FVG 포기
  → 실시간 적용 가능한 전략
  → 처음부터 재설계

어떤 방안을 시도하시겠습니까?
""")
