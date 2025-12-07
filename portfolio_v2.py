import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

times = df_15m['datetime'].values
lows = df_15m['low'].values
highs = df_15m['high'].values
opens = df_15m['open'].values

fvgs = []
for i in range(2, len(df_4h)):
    p2, curr = df_4h.iloc[i-2], df_4h.iloc[i]
    if p2['high'] < curr['low']:
        fvgs.append({
            'dt': curr['datetime'],
            'top': curr['low'],
            'body': abs(curr['close'] - curr['open']) / curr['open'] * 100
        })

def sim(fvgs, tp, sl, min_body=0, max_body=999):
    trades = []
    for fvg in fvgs:
        if not (min_body <= fvg['body'] < max_body):
            continue
        start = np.searchsorted(times, np.datetime64(fvg['dt']))
        if start >= len(times) - 200:
            continue
        entry_idx = None
        for i in range(start+1, min(start+200, len(times))):
            if lows[i] <= fvg['top']:
                if i+1 < len(times):
                    entry_idx = i+1
                break
        if entry_idx is None:
            continue
        ep = opens[entry_idx]
        tp_p, sl_p = ep * (1 + tp/100), ep * (1 + sl/100)
        result = None
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if lows[i] <= sl_p:
                result = 'L'
                break
            if highs[i] >= tp_p:
                result = 'W'
                break
        if result:
            trades.append({
                'time': times[entry_idx],
                'result': result,
                'pnl': tp - 0.06 if result == 'W' else sl - 0.06
            })
    return trades

def stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    wr = (df['result'] == 'W').mean() * 100
    cum, peak, mdd = 0, 0, 0
    for p in df['pnl']:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    return {'n': len(df), 'wr': wr, 'mdd': mdd, 'mavg': monthly.mean(), 'months': len(monthly)}

print("=" * 65)
print("🔍 MDD 최소화 + 승률 유지 전략 탐색")
print("=" * 65)

# 전략 1: 고승률 Tier만 (몸통 2%+)
print("\n[전략 A: Tier1만 (몸통 2%+)]")
for tp in [1.0, 1.5, 2.0]:
    for sl in [-1.0, -1.5, -2.0]:
        s = stats(sim(fvgs, tp, sl, 2.0, 999))
        if s and s['wr'] >= 75:
            print(f"  TP{tp} SL{sl}: 거래 {s['n']:>3}, 승률 {s['wr']:.1f}%, MDD {s['mdd']:.1f}%, 월평균 {s['mavg']:.2f}%")

# 전략 2: Tier1 + Tier2 (몸통 1%+)
print("\n[전략 B: Tier1+2 (몸통 1%+)]")
for tp in [1.0, 1.5, 2.0]:
    for sl in [-1.0, -1.5, -2.0]:
        s = stats(sim(fvgs, tp, sl, 1.0, 999))
        if s and s['wr'] >= 75:
            print(f"  TP{tp} SL{sl}: 거래 {s['n']:>3}, 승률 {s['wr']:.1f}%, MDD {s['mdd']:.1f}%, 월평균 {s['mavg']:.2f}%")

# 전략 3: 전체 (기본)
print("\n[전략 C: 전체]")
for tp in [1.0, 1.5]:
    for sl in [-1.0, -1.5, -2.0]:
        s = stats(sim(fvgs, tp, sl, 0, 999))
        if s and s['wr'] >= 75:
            print(f"  TP{tp} SL{sl}: 거래 {s['n']:>3}, 승률 {s['wr']:.1f}%, MDD {s['mdd']:.1f}%, 월평균 {s['mavg']:.2f}%")

# 최적 조합 찾기
print("\n" + "=" * 65)
print("🏆 최종 전략 후보 비교")
print("=" * 65)

candidates = [
    ("A: 몸통2%+ TP1 SL-1.5", sim(fvgs, 1.0, -1.5, 2.0, 999)),
    ("A: 몸통2%+ TP2 SL-1.5", sim(fvgs, 2.0, -1.5, 2.0, 999)),
    ("B: 몸통1%+ TP1 SL-1.5", sim(fvgs, 1.0, -1.5, 1.0, 999)),
    ("B: 몸통1%+ TP1.5 SL-1.5", sim(fvgs, 1.5, -1.5, 1.0, 999)),
    ("C: 전체 TP1 SL-1.5", sim(fvgs, 1.0, -1.5, 0, 999)),
    ("C: 전체 TP1 SL-2.0", sim(fvgs, 1.0, -2.0, 0, 999)),
]

print(f"\n{'전략':<25} {'거래':>5} {'승률':>6} {'MDD':>6} {'월평균':>7} {'효율':>6}")
print("-" * 65)

results = []
for name, trades in candidates:
    s = stats(trades)
    if s:
        eff = s['mavg'] / abs(s['mdd']) if s['mdd'] != 0 else 0
        results.append((name, s, eff))
        print(f"{name:<25} {s['n']:>5} {s['wr']:>5.1f}% {s['mdd']:>5.1f}% {s['mavg']:>6.2f}% {eff:>6.2f}")

# MDD -30% 레버리지 적용
print("\n" + "=" * 65)
print("💰 MDD -30% 기준 레버리지 비교")
print("=" * 65)
print(f"\n{'전략':<25} {'레버':>5} {'월수익':>8} {'연수익':>8}")
print("-" * 55)

for name, s, eff in results:
    lev = -30 / s['mdd']
    monthly = s['mavg'] * lev
    yearly = monthly * 12
    print(f"{name:<25} {lev:>4.1f}x {monthly:>7.1f}% {yearly:>7.0f}%")

# 최고 효율 전략
best = max(results, key=lambda x: x[2])
print(f"\n✅ 최고 효율: {best[0]}")
print(f"   승률 {best[1]['wr']:.1f}%, MDD {best[1]['mdd']:.1f}%, 월평균 {best[1]['mavg']:.2f}%")

# 멀티 전략 (독립적 전략 동시 운용)
print("\n" + "=" * 65)
print("🎯 멀티 전략 포트폴리오 (자금 분배)")
print("=" * 65)

# 전략 A (몸통2%+)와 전략 C (전체) 동시 운용
strat_a = stats(sim(fvgs, 1.0, -1.5, 2.0, 999))  # 고승률, 저MDD
strat_c = stats(sim(fvgs, 1.0, -1.5, 0, 999))    # 고빈도

print(f"\n[개별 전략]")
print(f"  전략A (몸통2%+): 승률 {strat_a['wr']:.1f}%, MDD {strat_a['mdd']:.1f}%, 월 {strat_a['mavg']:.2f}%")
print(f"  전략C (전체): 승률 {strat_c['wr']:.1f}%, MDD {strat_c['mdd']:.1f}%, 월 {strat_c['mavg']:.2f}%")

# 자금 배분 (50:50)
print(f"\n[자금 50:50 분배 시]")
combined_mdd = (strat_a['mdd'] * 0.5) + (strat_c['mdd'] * 0.5)  # 단순 합산 (최악의 경우)
combined_monthly = (strat_a['mavg'] * 0.5) + (strat_c['mavg'] * 0.5)
print(f"  예상 MDD: {combined_mdd:.1f}% (실제는 분산효과로 더 낮을 수 있음)")
print(f"  월 수익: {combined_monthly:.2f}%")

lev = -30 / combined_mdd
print(f"\n[MDD -30% 레버리지 적용]")
print(f"  레버리지: {lev:.1f}x")
print(f"  월 수익: {combined_monthly * lev:.1f}%")
print(f"  연 수익: {combined_monthly * lev * 12:.0f}%")

