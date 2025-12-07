import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("🔍 PNL 계산 로직 검증")
print("=" * 70)

# 현재 로직 문제점 체크
print("""
┌────────────────────────────────────────────────────────────────────┐
│  🚨 발견된 PNL 계산 문제점들                                         │
└────────────────────────────────────────────────────────────────────┘
""")

# 문제 1: SL 가격 계산
print("1️⃣ SL 가격 계산 문제")
print("-" * 50)
print("""
   현재 코드 (168번 줄):
   sl_price = entry_zone * (1 + sl)  # sl = -0.015
   
   예시: 진입가 $100,000, SL -1.5%
   sl_price = 100000 * (1 + (-0.015))
            = 100000 * 0.985
            = $98,500  ✅ 롱은 맞음
   
   숏의 경우 (168번 줄):
   sl_price = entry_zone * (1 - sl)  # sl = -0.015
            = 100000 * (1 - (-0.015))
            = 100000 * 1.015
            = $101,500  ✅ 숏도 맞음
""")

# 문제 2: 분할익절 PNL 계산
print("\n2️⃣ 분할익절 PNL 계산 문제")
print("-" * 50)
print("""
   현재 코드:
   TP1 도달 시: partial_pnl = tp1 * 100 * 0.5  (예: 2% * 0.5 = 1%)
   TP2 도달 시: pnl = partial_pnl + tp2 * 100 * 0.5  (예: 1% + 3.5% * 0.5 = 2.75%)
   
   🚨 문제: TP2에서 남은 50%는 TP2 수익이 아니라 TP1→TP2 구간 수익!
   
   올바른 계산:
   - 50%는 TP1(2%)에서 익절 → 1%
   - 50%는 TP2(3.5%)까지 보유 → 1.75%
   - 총: 2.75% ✅ (현재 코드 맞음)
   
   하지만! BE(본절) 청산 시:
   현재: pnl = partial_pnl + 0 = 1%
   → TP1에서 50% 익절(1%) + 나머지 50% 본절(0%) = 1% ✅ 맞음
   
   SL 청산 시:
   현재: pnl = partial_pnl + sl * 100 * 0.5
        = 1% + (-1.5%) * 0.5 = 1% - 0.75% = 0.25%
   
   🚨 이건 틀림! TP1 안 찍고 SL 나면 partial_pnl이 0인데...
""")

# 문제 3: TP1 안 찍고 SL 나는 경우
print("\n3️⃣ TP1 미달성 + SL 청산 시 문제")
print("-" * 50)

# 시뮬레이션으로 확인
entry = 100000
sl_pct = -0.015
tp1_pct = 0.02
tp2_pct = 0.035

print(f"""
   진입가: ${entry:,}
   TP1: ${entry * (1 + tp1_pct):,.0f} (+2%)
   TP2: ${entry * (1 + tp2_pct):,.0f} (+3.5%)
   SL:  ${entry * (1 + sl_pct):,.0f} (-1.5%)
   
   케이스 A: TP1 도달 → TP2 도달 (분할익절)
   - 50% @ TP1: +2% * 0.5 = +1%
   - 50% @ TP2: +3.5% * 0.5 = +1.75%
   - 총: +2.75% ✅
   
   케이스 B: TP1 도달 → BE 청산 (분할익절)
   - 50% @ TP1: +2% * 0.5 = +1%
   - 50% @ BE: 0% * 0.5 = 0%
   - 총: +1% ✅
   
   케이스 C: TP1 미도달 → SL 청산 (분할익절)
   현재 코드: pnl = 0 + (-1.5%) * 0.5 = -0.75%
   
   🚨 문제! 분할익절이면 전체 포지션이 SL 맞아야 함!
   올바른 계산: -1.5% (전체 손실)
   
   → partial이어도 TP1 전에 SL 나면 100% 손실!
""")

# 문제 4: 시간스탑 PNL
print("\n4️⃣ 시간스탑 PNL 계산")
print("-" * 50)
print("""
   현재 코드 (롱):
   time_pnl = (candle['close'] - entry_price) / entry_price * 100
   
   분할익절 시:
   pnl = partial_pnl + time_pnl * 0.5
   
   🚨 문제: TP1 찍고 시간스탑 나면?
   - 50%는 TP1에서 이미 익절 (partial_pnl = 1%)
   - 50%는 현재가로 청산 (time_pnl의 50%)
   
   예: TP1(2%) 찍고, 현재 +1%에서 시간스탑
   현재: pnl = 1% + 1% * 0.5 = 1.5% ✅ 맞음
   
   예: TP1 안 찍고, 현재 +1%에서 시간스탑
   현재: pnl = 0 + 1% * 0.5 = 0.5%
   
   🚨 문제! TP1 안 찍으면 분할 안 했으니까 100% 보유 중!
   올바른 계산: +1% (전체)
""")

print("\n" + "=" * 70)
print("🔴 핵심 버그 요약")
print("=" * 70)
print("""
   🚨 버그 1: 분할익절 모드에서 TP1 미도달 시
   
   - TP1 안 찍으면 아직 분할 안 함 (100% 보유)
   - 그런데 코드는 무조건 50%만 계산함
   
   현재 (틀림):
   if use_partial:
       pnl = partial_pnl + sl * 100 * 0.5  # TP1 안 찍어도 50%만 계산
   
   올바른 코드:
   if use_partial:
       if pos['tp1_hit']:  # TP1 찍었으면 50%만
           pnl = partial_pnl + sl * 100 * 0.5
       else:  # TP1 안 찍었으면 100%
           pnl = sl * 100
""")

# 실제 영향 계산
print("\n" + "=" * 70)
print("📊 버그로 인한 PNL 왜곡 추정")
print("=" * 70)

# 대략적인 영향
sl_rate = 18.7  # %
tp1_not_hit_ratio = 0.4  # TP1 안 찍고 SL 나는 비율 (추정)

wrong_sl_pnl = -0.75  # 현재 계산 (틀림)
correct_sl_pnl = -1.5  # 올바른 계산

affected_trades_pct = sl_rate * tp1_not_hit_ratio
pnl_difference_per_trade = correct_sl_pnl - wrong_sl_pnl  # -0.75%

print(f"""
   SL 비율: {sl_rate}%
   그 중 TP1 미도달 추정: {tp1_not_hit_ratio*100:.0f}%
   영향받는 거래: {affected_trades_pct:.1f}%
   
   거래당 PNL 차이: {pnl_difference_per_trade}%
   
   월 27회 거래 기준:
   - 영향받는 거래: {27 * affected_trades_pct / 100:.1f}회/월
   - 월 PNL 왜곡: {27 * affected_trades_pct / 100 * pnl_difference_per_trade:.2f}%
   
   🚨 분할익절 모드 월평균이 과대평가되었을 수 있음!
""")

