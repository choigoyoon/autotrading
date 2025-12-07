# 현재 전략 vs MDD 낮은 전략 + 레버리지 비교

print("=" * 65)
print("📊 MDD 기준 전략 비교: 레버리지 적용 시뮬레이션")
print("=" * 65)

# 전략 데이터 (이전 분석 결과)
strategies = {
    "현재 (TP1% SL-1.5%)": {
        "trades": 1044, "months": 69, "wr": 79.6,
        "tp": 1.0, "sl": -1.5, "mdd": -9.0, "monthly": 7.41
    },
    "저MDD A (TP1% SL-0.5%)": {
        "trades": 1075, "months": 69, "wr": 62.1,
        "tp": 1.0, "sl": -0.5, "mdd": -3.0, "monthly": 6.73
    },
    "저MDD B (TP1.5% SL-0.5%)": {
        "trades": 1065, "months": 69, "wr": 50.1,
        "tp": 1.5, "sl": -0.5, "mdd": -5.5, "monthly": 7.76
    },
}

cost = 0.06  # 지정가 비용

print(f"\n[기본 성과 비교 - 비용 {cost}% 적용, 레버리지 1x]")
print("-" * 65)
print(f"{'전략':<25} {'승률':>6} {'MDD':>6} {'월수익':>7} {'연수익':>7}")
print("-" * 65)

for name, s in strategies.items():
    net_tp = s['tp'] - cost
    net_sl = s['sl'] - cost
    wins = int(s['trades'] * s['wr'] / 100)
    losses = s['trades'] - wins
    total_pnl = wins * net_tp + losses * net_sl
    monthly = total_pnl / s['months']
    yearly = monthly * 12
    
    s['real_monthly'] = monthly
    s['real_yearly'] = yearly
    
    print(f"{name:<25} {s['wr']:>5.1f}% {s['mdd']:>5.1f}% {monthly:>6.2f}% {yearly:>6.1f}%")

# 레버리지 적용 비교
print("\n" + "=" * 65)
print("🎯 목표: MDD -30% 맞추고 레버리지 적용")
print("=" * 65)

target_mdd = -30  # 목표 MDD

print(f"\n[MDD -{abs(target_mdd)}% 기준 레버리지 적용]")
print("-" * 65)
print(f"{'전략':<25} {'기본MDD':>7} {'레버리지':>8} {'월수익':>8} {'연수익':>8} {'적용MDD':>8}")
print("-" * 65)

results = []
for name, s in strategies.items():
    leverage = target_mdd / s['mdd']  # MDD 맞추는 레버리지
    leveraged_monthly = s['real_monthly'] * leverage
    leveraged_yearly = leveraged_monthly * 12
    actual_mdd = s['mdd'] * leverage
    
    results.append({
        'name': name,
        'base_mdd': s['mdd'],
        'leverage': leverage,
        'monthly': leveraged_monthly,
        'yearly': leveraged_yearly,
        'actual_mdd': actual_mdd
    })
    
    print(f"{name:<25} {s['mdd']:>6.1f}% {leverage:>7.1f}x {leveraged_monthly:>7.2f}% {leveraged_yearly:>7.1f}% {actual_mdd:>7.1f}%")

# 최고 수익 전략 찾기
best = max(results, key=lambda x: x['monthly'])
print(f"\n✅ 최고 수익: {best['name']}")
print(f"   → {best['leverage']:.1f}x 레버리지로 월 {best['monthly']:.2f}%, 연 {best['yearly']:.1f}%")

# 다양한 MDD 목표로 비교
print("\n" + "=" * 65)
print("📈 MDD 목표별 레버리지 수익 비교")
print("=" * 65)

for target in [-10, -20, -30, -50]:
    print(f"\n[목표 MDD: {target}%]")
    print(f"{'전략':<25} {'레버':>5} {'월수익':>8} {'연수익':>8}")
    print("-" * 50)
    
    for name, s in strategies.items():
        lev = target / s['mdd']
        monthly = s['real_monthly'] * lev
        yearly = monthly * 12
        print(f"{name:<25} {lev:>4.1f}x {monthly:>7.2f}% {yearly:>7.1f}%")

# 자산 성장 비교
print("\n" + "=" * 65)
print("💰 5년 자산 성장 비교 (원금 1000만원, MDD -30% 기준)")
print("=" * 65)

print(f"\n{'전략':<25} {'레버':>5} {'1년후':>10} {'3년후':>12} {'5년후':>14}")
print("-" * 70)

for r in results:
    yearly_rate = r['yearly'] / 100
    y1 = 1000 * (1 + yearly_rate)
    y3 = 1000 * ((1 + yearly_rate) ** 3)
    y5 = 1000 * ((1 + yearly_rate) ** 5)
    
    print(f"{r['name']:<25} {r['leverage']:>4.1f}x {y1:>9.0f}만 {y3:>11.0f}만 {y5:>13.0f}만")

