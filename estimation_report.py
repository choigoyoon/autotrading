print("=" * 80)
print("롱 + 숏 전략 확장 추정 리포트")
print("=" * 80)

# 기존 롱 전략 데이터
long_trades = 371
long_annual = 65.1
long_win_rate = 26.4
long_avg_pnl = 0.60
long_cagr = 45.54
long_mdd = -16.44

print("\n📊 기존 롱 Only 전략:")
print(f"  총 거래: {long_trades}건 (연 {long_annual:.1f}건)")
print(f"  승률: {long_win_rate}%")
print(f"  평균 수익: {long_avg_pnl}%")
print(f"  연평균 수익 (CAGR): {long_cagr}%")
print(f"  MDD: {long_mdd}%")

print("\n" + "=" * 80)
print("롱 + 숏 확장 전략 추정")
print("=" * 80)

# 숏 시그널 추정
# 일반적으로 상승 추세선(L-L)이 하락 추세선(H-H)보다 적게 형성됨
# 약 50-70% 정도로 추정
short_ratio = 0.6  # 보수적 추정

estimated_short_trades = int(long_trades * short_ratio)
estimated_total_trades = long_trades + estimated_short_trades

print(f"\n📈 추정 시그널 수:")
print(f"  롱: {long_trades}건")
print(f"  숏 (추정): {estimated_short_trades}건 (롱의 {short_ratio*100:.0f}%)")
print(f"  합계: {estimated_total_trades}건")
print(f"  연평균: {estimated_total_trades/5.7:.1f}건 (롱 {long_annual:.1f} + 숏 {estimated_short_trades/5.7:.1f})")
print(f"  월평균: {estimated_total_trades/5.7/12:.1f}건")

# 숏 성과 추정 (일반적으로 롱보다 약간 낮음)
short_win_rate = long_win_rate * 0.9  # 90%
short_avg_pnl = long_avg_pnl * 0.85  # 85%

print(f"\n💰 숏 전략 추정 성과:")
print(f"  승률: {short_win_rate:.1f}% (롱의 90%)")
print(f"  평균 수익: {short_avg_pnl:.2f}% (롱의 85%)")

# 통합 성과
combined_win_rate = (long_trades * long_win_rate + estimated_short_trades * short_win_rate) / estimated_total_trades
combined_avg_pnl = (long_trades * long_avg_pnl + estimated_short_trades * short_avg_pnl) / estimated_total_trades

# 복리 계산
capital = 100
for _ in range(estimated_total_trades):
    # 롱과 숏을 비율에 맞춰 시뮬레이션
    if _ < long_trades:
        capital *= (1 + long_avg_pnl / 100)
    else:
        capital *= (1 + short_avg_pnl / 100)

combined_cagr = (capital / 100) ** (1 / 5.7) - 1

print(f"\n🎯 롱+숏 통합 성과 (추정):")
print(f"  총 거래: {estimated_total_trades}건")
print(f"  승률: {combined_win_rate:.1f}%")
print(f"  평균 수익: {combined_avg_pnl:.2f}%")
print(f"  최종 자본: {capital:.2f}")
print(f"  총 수익률: {(capital-100):.2f}%")
print(f"  연평균 (CAGR): {combined_cagr*100:.2f}%")

print("\n" + "=" * 80)
print("비교 분석")
print("=" * 80)

signal_increase = estimated_total_trades - long_trades
signal_increase_pct = signal_increase / long_trades * 100
cagr_diff = combined_cagr * 100 - long_cagr

print(f"\n시그널 증가:")
print(f"  {long_trades}건 → {estimated_total_trades}건")
print(f"  +{signal_increase}건 (+{signal_increase_pct:.1f}%)")

print(f"\n연평균 수익:")
print(f"  {long_cagr:.2f}% → {combined_cagr*100:.2f}%")
print(f"  {cagr_diff:+.2f}%p")

print(f"\n월 거래 빈도:")
print(f"  {long_annual/12:.1f}회 → {estimated_total_trades/5.7/12:.1f}회")
print(f"  +{(estimated_total_trades/5.7/12) - (long_annual/12):.1f}회")

print("\n" + "=" * 80)
print("시나리오 분석")
print("=" * 80)

scenarios = [
    ("보수적 (숏 40%)", 0.4, 0.8),
    ("기본 (숏 60%)", 0.6, 0.85),
    ("낙관적 (숏 80%)", 0.8, 0.9)
]

for name, ratio, performance in scenarios:
    short_n = int(long_trades * ratio)
    total_n = long_trades + short_n
    short_pnl = long_avg_pnl * performance
    
    cap = 100
    for i in range(total_n):
        if i < long_trades:
            cap *= (1 + long_avg_pnl / 100)
        else:
            cap *= (1 + short_pnl / 100)
    
    cagr_scenario = (cap / 100) ** (1 / 5.7) - 1
    
    print(f"\n{name}:")
    print(f"  시그널: {total_n}건 (연 {total_n/5.7:.1f}건)")
    print(f"  연평균: {cagr_scenario*100:.2f}%")
    print(f"  vs 롱 Only: {cagr_scenario*100-long_cagr:+.2f}%p")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

print(f"""
✅ **롱 + 숏 전략 확장 효과 (추정)**

1. **시그널 증가**: +{signal_increase_pct:.0f}% ({long_trades} → {estimated_total_trades}건)
   - 월평균 {long_annual/12:.1f}회 → {estimated_total_trades/5.7/12:.1f}회 거래
   - 거래 기회 증가로 수익 극대화

2. **수익성**: 연평균 {long_cagr:.1f}% → {combined_cagr*100:.1f}%
   - 롱 전략의 높은 성과 유지
   - 숏 추가로 추가 수익 확보

3. **리스크 분산**: 
   - 하락장에서도 수익 기회 (숏)
   - 양방향 전략으로 MDD 개선 가능

4. **실행 가능성**:
   - 동일한 로직 (추세선 돌파 + 마감 손절)
   - 기존 시스템 그대로 활용 가능

⚠️ **주의사항**:
- 숏 전략 성과는 실제 백테스트 필요
- 거래 빈도 증가로 심리적 부담 증가 가능
- 수수료 영향 고려 필요

🎯 **추천**:
- 먼저 롱 전략으로 안정화
- 숏 전략은 점진적으로 추가
- 실전 테스트 후 비중 조정
""")

