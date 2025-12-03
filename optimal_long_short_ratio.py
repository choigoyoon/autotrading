print("=" * 80)
print("최적 롱/숏 비중 전략")
print("=" * 80)

# 기존 데이터
long_trades = 371
long_cagr = 45.54
long_avg_pnl = 0.60
years = 5.7

print("\n💡 핵심 인사이트:")
print("  암호화폐는 장기적으로 '우상향' 자산")
print("  → 롱 비중 ↑, 숏 비중 ↓")
print("  → 숏은 보조적 역할만")

print("\n" + "=" * 80)
print("비중별 시뮬레이션")
print("=" * 80)

# 숏 시그널은 롱의 60%로 가정
estimated_short_trades = int(long_trades * 0.6)
total_signals = long_trades + estimated_short_trades

print(f"\n가용 시그널:")
print(f"  롱: {long_trades}건")
print(f"  숏: {estimated_short_trades}건")
print(f"  합계: {total_signals}건")

# 비중 시나리오
scenarios = [
    ("롱 100% / 숏 0%", 1.0, 0.0),
    ("롱 90% / 숏 10%", 0.9, 0.1),
    ("롱 80% / 숏 20%", 0.8, 0.2),
    ("롱 70% / 숏 30%", 0.7, 0.3),
    ("롱 60% / 숏 40%", 0.6, 0.4),
    ("롱 50% / 숏 50%", 0.5, 0.5),
]

print("\n" + "=" * 80)
print("시나리오별 성과 (숏 성과 = 롱의 85% 가정)")
print("=" * 80)

results = []

for name, long_weight, short_weight in scenarios:
    # 실제 거래 수
    long_trades_actual = int(long_trades * long_weight)
    short_trades_actual = int(estimated_short_trades * short_weight)
    total_trades = long_trades_actual + short_trades_actual
    
    # 성과 계산
    short_avg_pnl = long_avg_pnl * 0.85  # 숏은 롱의 85%
    
    # 복리 계산
    capital = 100
    for i in range(total_trades):
        if i < long_trades_actual:
            capital *= (1 + long_avg_pnl / 100)
        else:
            capital *= (1 + short_avg_pnl / 100)
    
    cagr = (capital / 100) ** (1 / years) - 1
    
    results.append({
        'name': name,
        'long_weight': long_weight,
        'short_weight': short_weight,
        'total_trades': total_trades,
        'annual_trades': total_trades / years,
        'final_capital': capital,
        'cagr': cagr * 100
    })
    
    print(f"\n{name}:")
    print(f"  거래: {total_trades}건 (롱 {long_trades_actual} + 숏 {short_trades_actual})")
    print(f"  연평균 거래: {total_trades/years:.1f}건")
    print(f"  최종 자본: {capital:.2f}")
    print(f"  연평균 수익: {cagr*100:.2f}%")

# 최적 비중 찾기
print("\n" + "=" * 80)
print("최적 비중 분석")
print("=" * 80)

best = max(results, key=lambda x: x['cagr'])
worst = min(results, key=lambda x: x['cagr'])

print(f"\n🏆 최고 성과: {best['name']}")
print(f"  연평균: {best['cagr']:.2f}%")
print(f"  연 거래: {best['annual_trades']:.1f}건")

print(f"\n❌ 최저 성과: {worst['name']}")
print(f"  연평균: {worst['cagr']:.2f}%")
print(f"  연 거래: {worst['annual_trades']:.1f}건")

# 우상향 논리 적용
print("\n" + "=" * 80)
print("🎯 우상향 자산 특성 고려")
print("=" * 80)

print("""
암호화폐(비트코인) 특성:
  ✅ 장기 추세: 우상향 (10년+ 관점)
  ✅ 사이클: 4년 주기 반등
  ✅ 희소성: 공급 제한 (2100만개)
  
  → 롱이 절대적으로 유리!
""")

# 추천 비중
recommended = [r for r in results if r['long_weight'] >= 0.8 and r['short_weight'] <= 0.2]

print("📊 추천 비중 (롱 80% 이상):")
for r in recommended:
    print(f"\n  {r['name']}:")
    print(f"    연평균: {r['cagr']:.2f}%")
    print(f"    연 거래: {r['annual_trades']:.1f}건")
    print(f"    장점: 롱 중심, 안정적")

print("\n" + "=" * 80)
print("실전 운용 전략")
print("=" * 80)

print("""
🎯 **추천 비중: 롱 90% / 숏 10%**

이유:
1. 우상향 자산의 본질 활용
2. 롱 전략 성과 최대화 (45.5% CAGR)
3. 숏은 보험/헤지 용도로만 사용
4. 심리적 부담 최소화

구체적 운용:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ 자본 배분                           │
│  • 롱: 전체 자본의 80-90%          │
│  • 숏: 전체 자본의 10-20%          │
│  • 현금: 5-10% (기회 대기)         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

시그널 선택:
  • 롱 371건 → 90% 활용 (334건)
  • 숏 222건 → 10% 활용 (22건)
  • 총 356건 (연 62.5건, 월 5.2건)

숏 진입 조건 (까다롭게):
  • Gap > 1.0% (강한 돌파만)
  • 명확한 저항선 확인
  • 거래량 3배 이상
  • 단기 트레이딩만 (24-48시간)

롱 진입 조건 (적극적):
  • Gap > 0.3% (여유롭게)
  • 기존 전략 그대로
  • 중장기 홀딩 가능 (72시간+)
""")

print("\n" + "=" * 80)
print("비중별 예상 성과")
print("=" * 80)

# 롱 90% / 숏 10% 시뮬레이션
long_90_trades = int(long_trades * 0.9)
short_10_trades = int(estimated_short_trades * 0.1)
total_90_10 = long_90_trades + short_10_trades

capital_90_10 = 100
for i in range(total_90_10):
    if i < long_90_trades:
        capital_90_10 *= (1 + long_avg_pnl / 100)
    else:
        capital_90_10 *= (1 + long_avg_pnl * 0.85 / 100)

cagr_90_10 = (capital_90_10 / 100) ** (1 / years) - 1

print(f"\n🎯 추천 비중 (롱 90% / 숏 10%):")
print(f"  총 거래: {total_90_10}건 (롱 {long_90_trades} + 숏 {short_10_trades})")
print(f"  연 거래: {total_90_10/years:.1f}건")
print(f"  월 거래: {total_90_10/years/12:.1f}건")
print(f"  최종 자본: {capital_90_10:.2f}")
print(f"  연평균: {cagr_90_10*100:.2f}%")
print(f"  vs 롱 Only: {cagr_90_10*100 - long_cagr:+.2f}%p")

print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

print(f"""
✅ **최적 전략: 롱 90% / 숏 10%**

1. 성과:
   • 연평균: {cagr_90_10*100:.2f}%
   • 롱 Only 대비: {cagr_90_10*100 - long_cagr:+.2f}%p
   • 거래 빈도: 월 {total_90_10/years/12:.1f}회 (적정)

2. 장점:
   • 우상향 자산 특성 최대 활용
   • 롱 전략 성과 거의 그대로 유지
   • 숏으로 추가 수익 + 헤지
   • 심리적 부담 최소

3. 운용 원칙:
   • 롱: 공격적 진입, 여유로운 홀딩
   • 숏: 보수적 진입, 빠른 청산
   • 의심스러우면 롱!
   • 숏은 확실할 때만!

4. 리스크 관리:
   • 하락장 진입 시 숏 비중 → 20% 확대
   • 상승장 진입 시 롱 비중 → 95% 확대
   • 유연한 조정이 핵심!
""")

