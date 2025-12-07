import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("🔧 승률 70%+ 전략 찾기")
print("="*80)

# 결과 요약
print("""
[이미 나온 결과 정리]

1. TP/SL 조정으로 승률 높이기:
   TP 1.0% SL -1.5%: 634건, 승률 84.2%, EV 0.606%
   TP 1.0% SL -1.0%: 652건, 승률 78.1%, EV 0.561%
   TP 1.5% SL -1.5%: 590건, 승률 74.9%, EV 0.747%

2. 4H 몸통 크기 필터:
   몸통 1.5%+: 123건, 승률 69.9%, EV 0.898%
   몸통 2.0%+: 72건, 승률 72.2%, EV 0.944%
   몸통 2.5%+: 38건, 승률 73.7%, EV 0.974%
   몸통 3.0%+: 24건, 승률 75.0%, EV 1.000%
""")

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvg_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'body_size': body_size,
                'fvg_size': fvg_size,
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)

# 거래 시뮬레이션
def simulate(fvgs, tp, sl, body_min=0, fvg_min=0):
    trades = []
    
    for fvg in fvgs:
        if fvg['body_size'] < body_min or fvg['fvg_size'] < fvg_min:
            continue
            
        fvg_time = fvg['datetime']
        fvg_top = fvg['fvg_top']
        
        mask = df_15m['datetime'] > fvg_time
        future_15m = df_15m[mask].head(200)
        
        if len(future_15m) < 50:
            continue
        
        for i, (idx, row) in enumerate(future_15m.iterrows()):
            if row['low'] <= fvg_top:
                remaining = future_15m.iloc[i+1:]
                if len(remaining) < 50:
                    break
                
                entry_price = remaining.iloc[0]['open']
                
                result = None
                for j, (_, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                    high_pct = (bar['high'] - entry_price) / entry_price * 100
                    low_pct = (bar['low'] - entry_price) / entry_price * 100
                    open_pct = (bar['open'] - entry_price) / entry_price * 100
                    
                    if high_pct >= tp and low_pct <= sl:
                        result = 'WIN' if open_pct >= 0 else 'LOSS'
                        break
                    elif high_pct >= tp:
                        result = 'WIN'
                        break
                    elif low_pct <= sl:
                        result = 'LOSS'
                        break
                
                if result:
                    trades.append(result)
                break
    
    return trades

print("="*80)
print("🎯 조합 탐색: 몸통 + TP/SL")
print("="*80)

results = []

for body_min in [0, 1.0, 1.5, 2.0, 2.5]:
    for tp in [0.8, 1.0, 1.2, 1.5]:
        for sl in [-0.5, -0.7, -1.0, -1.5]:
            trades = simulate(fvgs_4h, tp, sl, body_min=body_min)
            if len(trades) < 20:
                continue
            wins = trades.count('WIN')
            wr = wins/len(trades)*100
            ev = (wr/100)*tp - ((100-wr)/100)*abs(sl)
            
            results.append({
                'body': body_min,
                'tp': tp,
                'sl': sl,
                'n': len(trades),
                'wr': wr,
                'ev': ev,
                'monthly': len(trades)/60,
            })

# 승률 70%+ 필터
good = [r for r in results if r['wr'] >= 70]
good.sort(key=lambda x: x['ev'], reverse=True)

print(f"\n[승률 70%+ 조합] (EV 순)")
print(f"{'몸통':>6} {'TP':>6} {'SL':>6} {'거래':>6} {'승률':>8} {'EV':>8} {'월':>6}")
print("-"*55)

for r in good[:20]:
    print(f"{r['body']:>5.1f}% {r['tp']:>5.1f}% {r['sl']:>5.1f}% {r['n']:>6} {r['wr']:>7.1f}% {r['ev']:>7.3f}% {r['monthly']:>5.1f}")

# 최적 비교
print("\n" + "="*80)
print("🏆 최적 전략 비교")
print("="*80)

# 전략 1: 승률 우선 (84%)
print("\n[전략 1] 승률 우선")
t1 = simulate(fvgs_4h, 1.0, -1.5)
w1 = t1.count('WIN')
print(f"조건: TP 1.0%, SL -1.5%")
print(f"거래: {len(t1)}건 ({len(t1)/60:.1f}/월)")
print(f"승률: {w1/len(t1)*100:.1f}%")
print(f"EV: {(w1/len(t1))*1.0 - ((len(t1)-w1)/len(t1))*1.5:.3f}%")

# 전략 2: 몸통 필터 (72%)
print("\n[전략 2] 몸통 2%+ 필터")
t2 = simulate(fvgs_4h, 1.5, -0.5, body_min=2.0)
w2 = t2.count('WIN')
print(f"조건: 몸통 2%+, TP 1.5%, SL -0.5%")
print(f"거래: {len(t2)}건 ({len(t2)/60:.1f}/월)")
print(f"승률: {w2/len(t2)*100:.1f}%")
print(f"EV: {(w2/len(t2))*1.5 - ((len(t2)-w2)/len(t2))*0.5:.3f}%")

# 전략 3: 밸런스
print("\n[전략 3] 밸런스 (몸통 1.5% + SL 넓게)")
t3 = simulate(fvgs_4h, 1.0, -1.0, body_min=1.5)
w3 = t3.count('WIN')
if len(t3) > 0:
    print(f"조건: 몸통 1.5%+, TP 1.0%, SL -1.0%")
    print(f"거래: {len(t3)}건 ({len(t3)/60:.1f}/월)")
    print(f"승률: {w3/len(t3)*100:.1f}%")
    print(f"EV: {(w3/len(t3))*1.0 - ((len(t3)-w3)/len(t3))*1.0:.3f}%")

# 슬리피지 적용 비교
print("\n" + "="*80)
print("💸 슬리피지 0.08% 적용 후")
print("="*80)

cost = 0.08

for name, trades, tp, sl in [
    ("전략1 (승률우선)", t1, 1.0, -1.5),
    ("전략2 (몸통필터)", t2, 1.5, -0.5),
]:
    wins = trades.count('WIN')
    n = len(trades)
    real_tp = tp - cost
    real_sl = sl - cost
    wr = wins/n*100
    ev = (wr/100)*real_tp - ((100-wr)/100)*abs(real_sl)
    
    print(f"\n{name}:")
    print(f"  승률: {wr:.1f}%")
    print(f"  실제 TP/SL: {real_tp:.2f}% / {real_sl:.2f}%")
    print(f"  EV (비용후): {ev:.3f}%")
    print(f"  월 거래: {n/60:.1f}회")
    print(f"  월 기대수익 (10x): {ev * n/60 * 10:.1f}%")

