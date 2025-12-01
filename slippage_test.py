import pandas as pd
import numpy as np

print("="*80)
print("💸 슬리피지 + 수수료 적용 테스트")
print("="*80)

# 현재 전략
tp = 1.5
sl = -0.5
wins = 369
losses = 282
total = wins + losses

print(f"\n[기본 전략]")
print(f"TP: {tp}%, SL: {sl}%")
print(f"승: {wins}, 패: {losses}")
print(f"승률: {wins/total*100:.1f}%")

# 슬리피지 + 수수료 시나리오
print("\n" + "="*80)
print("📊 비용 시나리오별 결과")
print("="*80)

# 비용 = 진입 슬리피지 + 청산 슬리피지 + 수수료(왕복)
# 바이낸스 선물: 메이커 0.02%, 테이커 0.04%
# 슬리피지: 보통 0.01~0.05%

scenarios = [
    ("이상적 (비용 0%)", 0),
    ("최소 (0.03%)", 0.03),  # 메이커 왕복
    ("보통 (0.05%)", 0.05),  # 슬리피지 포함
    ("현실적 (0.08%)", 0.08),  # 테이커 + 슬리피지
    ("보수적 (0.1%)", 0.1),   # 나쁜 상황
    ("최악 (0.15%)", 0.15),   # 급등락 시
]

print(f"\n{'시나리오':<20} {'실TP':>8} {'실SL':>8} {'EV':>8} {'5년PnL':>10}")
print("-"*60)

for name, cost in scenarios:
    real_tp = tp - cost  # TP에서 비용 차감
    real_sl = sl - cost  # SL에서 비용 추가 (더 손실)
    
    ev = (wins/total) * real_tp + (losses/total) * real_sl
    total_pnl = wins * real_tp + losses * real_sl
    
    print(f"{name:<20} {real_tp:>7.2f}% {real_sl:>7.2f}% {ev:>7.3f}% {total_pnl:>9.1f}%")

print("\n" + "="*80)
print("🎯 손익분기점 계산")
print("="*80)

# EV = 0 되는 비용 찾기
# (wins/total) * (tp - cost) + (losses/total) * (sl - cost) = 0
# (wins/total) * tp + (losses/total) * sl - cost = 0
# cost = (wins/total) * tp + (losses/total) * sl

base_ev = (wins/total) * tp + (losses/total) * sl
breakeven_cost = base_ev

print(f"\n기본 EV (비용 0%): {base_ev:.3f}%")
print(f"손익분기 비용: {breakeven_cost:.3f}%")
print(f"\n→ 비용이 {breakeven_cost:.3f}% 이상이면 손실")

print("\n" + "="*80)
print("📈 레버리지별 실제 수익 (비용 0.08% 가정)")
print("="*80)

cost = 0.08
real_tp = tp - cost
real_sl = sl - cost
real_ev = (wins/total) * real_tp + (losses/total) * real_sl
real_pnl = wins * real_tp + losses * real_sl

print(f"\n실제 EV: {real_ev:.3f}%")
print(f"5년 총 PnL: {real_pnl:.1f}%")
print(f"월평균 거래: {total/60:.1f}회")
print(f"월평균 수익: {real_pnl/60:.2f}%")

print(f"\n{'레버리지':<10} {'월수익':>10} {'연수익':>10} {'5년MDD':>10}")
print("-"*45)

for lev in [1, 3, 5, 10, 20]:
    monthly = real_pnl / 60 * lev
    yearly = monthly * 12
    mdd = 3.5 * lev  # 기존 MDD에 레버리지 적용
    print(f"{lev}x{'':<8} {monthly:>9.1f}% {yearly:>9.0f}% {-mdd:>9.1f}%")

print("\n" + "="*80)
print("💡 결론")
print("="*80)
print(f"""
현실적 비용 0.08% 적용 시:
- EV: {base_ev:.3f}% → {real_ev:.3f}% (감소)
- 5년 PnL: 412% → {real_pnl:.0f}%
- 월 수익 (10x): {real_pnl/60*10:.1f}%

손익분기 비용: {breakeven_cost:.3f}%
→ 아직 여유 있음 ({breakeven_cost:.3f}% - 0.08% = {breakeven_cost-0.08:.3f}%)

하지만...
- 슬리피지 나쁘면 0.1%+ 가능
- 급등락 시 0.15%+ 가능
- 마진 좁음 (버퍼 {breakeven_cost-0.08:.3f}%밖에 없음)
""")

