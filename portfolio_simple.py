import pandas as pd
import numpy as np

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 15m numpy 배열
times = df_15m['datetime'].values
lows = df_15m['low'].values
highs = df_15m['high'].values
opens = df_15m['open'].values

# FVG 감지
fvgs = []
for i in range(2, len(df_4h)):
    p2, curr = df_4h.iloc[i-2], df_4h.iloc[i]
    if p2['high'] < curr['low']:
        fvgs.append({
            'dt': curr['datetime'],
            'top': curr['low'],
            'body': abs(curr['close'] - curr['open']) / curr['open'] * 100
        })
print(f"FVG: {len(fvgs)}개")

# 빠른 시뮬레이션 (numpy 기반)
def sim_fast(fvgs, tp, sl, min_body=0, max_body=999):
    trades = []
    for fvg in fvgs:
        if not (min_body <= fvg['body'] < max_body):
            continue
        
        start = np.searchsorted(times, np.datetime64(fvg['dt']))
        if start >= len(times) - 200:
            continue
        
        # FVG 터치 찾기
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
                'pnl': tp - 0.06 if result == 'W' else sl - 0.06,
                'body': fvg['body']
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

print("\n" + "=" * 60)
print("🎯 Tier별 분리 전략 (몸통 크기 기준)")
print("=" * 60)

# Tier 분류
tiers = [
    ('Tier1 (몸통 2%+)', 2.0, 999, 1.0, -1.5),
    ('Tier2 (몸통 1-2%)', 1.0, 2.0, 1.0, -1.5),
    ('Tier3 (몸통 <1%)', 0, 1.0, 1.0, -1.5),
]

print(f"\n[기본 TP1% SL-1.5% 적용 시]")
print(f"{'Tier':<20} {'거래':>6} {'승률':>7} {'MDD':>7} {'월평균':>8}")
print("-" * 55)

for name, minb, maxb, tp, sl in tiers:
    t = sim_fast(fvgs, tp, sl, minb, maxb)
    s = stats(t)
    if s:
        print(f"{name:<20} {s['n']:>6} {s['wr']:>6.1f}% {s['mdd']:>6.1f}% {s['mavg']:>7.2f}%")

# 각 Tier별 최적 TP/SL 찾기
print("\n" + "=" * 60)
print("📊 Tier별 최적 TP/SL 탐색 (승률 75%+ 유지)")
print("=" * 60)

best_configs = {}
for name, minb, maxb, _, _ in tiers:
    best = None
    for tp in [0.8, 1.0, 1.5, 2.0]:
        for sl in [-1.0, -1.5, -2.0]:
            t = sim_fast(fvgs, tp, sl, minb, maxb)
            s = stats(t)
            if s and s['wr'] >= 75 and s['n'] >= 5:
                if best is None or s['mavg'] > best['mavg']:
                    best = {'tp': tp, 'sl': sl, **s}
    best_configs[name] = best
    if best:
        print(f"{name}: TP {best['tp']}% SL {best['sl']}% → 승률 {best['wr']:.1f}%, MDD {best['mdd']:.1f}%, 월평균 {best['mavg']:.2f}%")
    else:
        print(f"{name}: 75%+ 승률 조건 없음")

# 포트폴리오 구성
print("\n" + "=" * 60)
print("🏆 포트폴리오 시뮬레이션")
print("=" * 60)

portfolio = []

# Tier1: 최적 또는 기본
cfg = best_configs.get('Tier1 (몸통 2%+)')
if cfg:
    t1 = sim_fast(fvgs, cfg['tp'], cfg['sl'], 2.0, 999)
else:
    t1 = sim_fast(fvgs, 1.0, -1.5, 2.0, 999)
portfolio.extend(t1)

# Tier2
cfg = best_configs.get('Tier2 (몸통 1-2%)')
if cfg:
    t2 = sim_fast(fvgs, cfg['tp'], cfg['sl'], 1.0, 2.0)
else:
    t2 = sim_fast(fvgs, 1.0, -1.5, 1.0, 2.0)
portfolio.extend(t2)

# Tier3
cfg = best_configs.get('Tier3 (몸통 <1%)')
if cfg:
    t3 = sim_fast(fvgs, cfg['tp'], cfg['sl'], 0, 1.0)
else:
    t3 = sim_fast(fvgs, 1.0, -1.5, 0, 1.0)
portfolio.extend(t3)

# 정렬
portfolio.sort(key=lambda x: x['time'])
port_stats = stats(portfolio)

# 단일 전략
single = sim_fast(fvgs, 1.0, -1.5)
single_stats = stats(single)

print(f"\n[구성]")
c1 = best_configs.get('Tier1 (몸통 2%+)')
c2 = best_configs.get('Tier2 (몸통 1-2%)')
c3 = best_configs.get('Tier3 (몸통 <1%)')
print(f"  Tier1 (몸통2%+): TP {c1['tp'] if c1 else 1.0}% SL {c1['sl'] if c1 else -1.5}%")
print(f"  Tier2 (몸통1-2%): TP {c2['tp'] if c2 else 1.0}% SL {c2['sl'] if c2 else -1.5}%")
print(f"  Tier3 (몸통<1%): TP {c3['tp'] if c3 else 1.0}% SL {c3['sl'] if c3 else -1.5}%")

print(f"\n[단일 vs 포트폴리오 비교]")
print(f"{'지표':<12} {'단일전략':>12} {'포트폴리오':>12} {'변화':>10}")
print("-" * 50)
print(f"{'거래수':<12} {single_stats['n']:>12} {port_stats['n']:>12}")
print(f"{'승률':<12} {single_stats['wr']:>11.1f}% {port_stats['wr']:>11.1f}% {port_stats['wr']-single_stats['wr']:>+9.1f}%p")
print(f"{'MDD':<12} {single_stats['mdd']:>11.1f}% {port_stats['mdd']:>11.1f}% {port_stats['mdd']-single_stats['mdd']:>+9.1f}%")
print(f"{'월평균':<12} {single_stats['mavg']:>11.2f}% {port_stats['mavg']:>11.2f}% {port_stats['mavg']-single_stats['mavg']:>+9.2f}%")
print(f"{'연수익':<12} {single_stats['mavg']*12:>11.1f}% {port_stats['mavg']*12:>11.1f}%")

# 레버리지 적용
print("\n" + "=" * 60)
print("💰 레버리지 적용 시 (MDD -30% 기준)")
print("=" * 60)

for name, s in [('단일전략', single_stats), ('포트폴리오', port_stats)]:
    lev = -30 / s['mdd']
    monthly = s['mavg'] * lev
    yearly = monthly * 12
    print(f"\n{name}:")
    print(f"  레버리지: {lev:.1f}x")
    print(f"  월수익: {monthly:.1f}%")
    print(f"  연수익: {yearly:.0f}%")

