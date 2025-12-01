import pandas as pd

# 현재 전략 데이터
trades = 1044
win_rate = 79.6 / 100
tp = 1.0  # %
sl = -1.5  # %
months = 69

wins = int(trades * win_rate)
losses = trades - wins

print("=" * 60)
print("📊 현재 전략 실제 수익 분석 (TP 1.0% / SL -1.5%)")
print("=" * 60)

print(f"\n[기본 정보]")
print(f"  총 거래: {trades}회 ({months}개월)")
print(f"  승: {wins}회, 패: {losses}회")
print(f"  승률: {win_rate*100:.1f}%")
print(f"  손익비: 1:{abs(sl/tp):.1f} (불리)")

# 이상적 수익 (비용 0)
ideal_pnl = wins * tp + losses * sl
ideal_monthly = ideal_pnl / months
print(f"\n[이상적 수익 - 비용 0%]")
print(f"  5년 총 PnL: {ideal_pnl:.1f}%")
print(f"  월평균: {ideal_monthly:.2f}%")
print(f"  연평균: {ideal_monthly * 12:.1f}%")

# 지정가 SL 시 비용 구조
print("\n" + "=" * 60)
print("💰 지정가 주문 시 비용 분석")
print("=" * 60)

print(f"""
[비용 구조]
  - 진입: 시장가 (Taker) = 0.04%
  - TP 청산: 지정가 (Maker) = 0.02%  
  - SL 청산: 지정가 (Maker) = 0.02%
  
  → 1회 거래 비용: 0.04% + 0.02% = 0.06%
  → 슬리피지: 지정가라 0% (또는 매우 적음)
""")

# 다양한 비용 시나리오
scenarios = [
    ("이상적 (0%)", 0),
    ("지정가만 (0.06%)", 0.06),
    ("지정가 + 슬립 0.02%", 0.08),
    ("보수적 (0.10%)", 0.10),
]

print(f"\n{'시나리오':<20} {'거래비용':>8} {'5년PnL':>10} {'월평균':>8} {'연수익':>8}")
print("-" * 60)

for name, cost in scenarios:
    # 비용 = 진입 + 청산 (왕복)
    net_tp = tp - cost
    net_sl = sl - cost
    
    total_pnl = wins * net_tp + losses * net_sl
    monthly = total_pnl / months
    yearly = monthly * 12
    
    print(f"{name:<20} {cost:>7.2f}% {total_pnl:>9.1f}% {monthly:>7.2f}% {yearly:>7.1f}%")

# 지정가 0.06% 기준 상세
print("\n" + "=" * 60)
print("📈 지정가 주문 기준 (0.06%) 상세 분석")
print("=" * 60)

cost = 0.06
net_tp = tp - cost
net_sl = sl - cost

total_pnl = wins * net_tp + losses * net_sl
monthly_avg = total_pnl / months
yearly = monthly_avg * 12
monthly_trades = trades / months

print(f"""
[실제 손익]
  실제 TP: {tp}% - {cost}% = {net_tp:.2f}%
  실제 SL: {sl}% - {cost}% = {net_sl:.2f}%
  
[5년 성과]
  총 수익: {wins}승 × {net_tp:.2f}% + {losses}패 × {net_sl:.2f}%
         = {wins * net_tp:.1f}% + ({losses * net_sl:.1f}%)
         = {total_pnl:.1f}%

[월평균 성과]
  월 거래: {monthly_trades:.1f}회
  월평균 수익: {monthly_avg:.2f}%
  연평균 수익: {yearly:.1f}%
  
[자산 성장 (원금 1000만원)]
  1년 후: {1000 * (1 + yearly/100):.0f}만원
  3년 후: {1000 * ((1 + yearly/100) ** 3):.0f}만원
  5년 후: {1000 * ((1 + yearly/100) ** 5):.0f}만원
""")

# EV 계산
ev_per_trade = win_rate * net_tp + (1 - win_rate) * net_sl
print(f"[기대값]")
print(f"  1회 거래 EV: {ev_per_trade:.3f}%")
print(f"  월 EV: {ev_per_trade * monthly_trades:.2f}%")

# 손익분기 비용
breakeven_cost = ideal_pnl / trades
print(f"\n[손익분기점]")
print(f"  손익분기 비용: {breakeven_cost:.3f}%")
print(f"  현재 비용(0.06%)과의 여유: {breakeven_cost - 0.06:.3f}%")

