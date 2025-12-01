import pandas as pd
import numpy as np

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')

df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 4H FVG 감지
def detect_fvgs(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2 = df.iloc[i-2]
        curr = df.iloc[i]
        
        if prev2['high'] < curr['low']:
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'body_size': abs(curr['close'] - curr['open']) / curr['open'] * 100
            })
    return pd.DataFrame(fvgs)

fvg_df = detect_fvgs(df_4h)
print(f"감지된 4H FVG: {len(fvg_df)}개")

def simulate(fvg_df, df_15m, tp_pct, sl_pct, min_body=0):
    trades = []
    filtered = fvg_df[fvg_df['body_size'] >= min_body] if min_body > 0 else fvg_df
    
    for _, fvg in filtered.iterrows():
        fvg_time = fvg['datetime']
        fvg_top = fvg['fvg_top']
        
        # FVG 이후 15분봉
        future = df_15m[df_15m['datetime'] > fvg_time].head(200)
        if len(future) < 10:
            continue
        
        # FVG 터치 찾기
        touch_idx = None
        for i, (idx, row) in enumerate(future.iterrows()):
            if row['low'] <= fvg_top:
                touch_idx = i
                break
        
        if touch_idx is None or touch_idx + 1 >= len(future):
            continue
        
        entry_row = future.iloc[touch_idx + 1]
        entry_price = entry_row['open']
        entry_time = entry_row['datetime']
        
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        
        # 결과 판정
        after = future.iloc[touch_idx + 2:]
        result = None
        for _, row in after.iterrows():
            if row['low'] <= sl_price:
                result = 'loss'
                break
            if row['high'] >= tp_price:
                result = 'win'
                break
        
        if result:
            trades.append({
                'time': entry_time,
                'result': result,
                'pnl': tp_pct if result == 'win' else sl_pct
            })
    
    return trades

def calc_stats(trades):
    if len(trades) < 5:
        return None
    
    wins = sum(1 for t in trades if t['result'] == 'win')
    wr = wins / len(trades) * 100
    total_pnl = sum(t['pnl'] for t in trades)
    
    # MDD
    cum, peak, mdd = 0, 0, 0
    for t in trades:
        cum += t['pnl']
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    
    # 월별
    df = pd.DataFrame(trades)
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    
    return {
        'trades': len(trades),
        'wr': wr,
        'total_pnl': total_pnl,
        'mdd': mdd,
        'efficiency': total_pnl / abs(mdd) if mdd != 0 else 0,
        'monthly_avg': monthly.mean(),
        'months': len(monthly)
    }

print("\n" + "=" * 70)
print("🔍 MDD 줄이고 수익 늘리기 - 최적화 분석")
print("=" * 70)

# 현재 전략
print("\n[현재 전략] TP 1.0% / SL -1.5%")
base = calc_stats(simulate(fvg_df, df_15m, 1.0, -1.5))
if base:
    print(f"  거래: {base['trades']}회 ({base['months']}개월), 승률: {base['wr']:.1f}%")
    print(f"  총 PnL: {base['total_pnl']:.1f}%, MDD: {base['mdd']:.1f}%")
    print(f"  효율(PnL/MDD): {base['efficiency']:.1f}, 월평균: {base['monthly_avg']:.2f}%")

# 방법 1: TP/SL 조정
print("\n" + "=" * 70)
print("📊 방법 1: TP/SL 비율 조정")
print("=" * 70)
print(f"\n{'TP':>5} {'SL':>5} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7}")
print("-" * 60)

results1 = []
for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [-0.5, -1.0, -1.5, -2.0]:
        s = calc_stats(simulate(fvg_df, df_15m, tp, sl))
        if s and s['trades'] > 50:
            results1.append({'tp': tp, 'sl': sl, **s})

results1.sort(key=lambda x: x['efficiency'], reverse=True)
for r in results1[:8]:
    print(f"{r['tp']:>5.1f} {r['sl']:>5.1f} {r['trades']:>5} {r['wr']:>5.1f}% {r['total_pnl']:>6.1f}% {r['mdd']:>5.1f}% {r['efficiency']:>6.1f} {r['monthly_avg']:>6.2f}%")

# 방법 2: 몸통 필터
print("\n" + "=" * 70)
print("📊 방법 2: 4H 몸통 필터 적용")
print("=" * 70)
print(f"\n{'몸통':>6} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7}")
print("-" * 60)

for body in [0, 1.0, 1.5, 2.0, 2.5, 3.0]:
    s = calc_stats(simulate(fvg_df, df_15m, 1.0, -1.5, body))
    if s and s['trades'] > 10:
        print(f"{body:>5.1f}% {s['trades']:>5} {s['wr']:>5.1f}% {s['total_pnl']:>6.1f}% {s['mdd']:>5.1f}% {s['efficiency']:>6.1f} {s['monthly_avg']:>6.2f}%")

# 방법 3: 복합 조건
print("\n" + "=" * 70)
print("📊 방법 3: 복합 최적화 - 효율 순")
print("=" * 70)
print(f"\n{'조건':>28} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7}")
print("-" * 80)

best = []
for body in [1.0, 1.5, 2.0, 2.5]:
    for tp in [1.5, 2.0, 2.5, 3.0]:
        for sl in [-1.0, -1.5, -2.0]:
            s = calc_stats(simulate(fvg_df, df_15m, tp, sl, body))
            if s and s['trades'] > 10:
                label = f"몸통{body}%+ TP{tp}% SL{sl}%"
                best.append({'label': label, 'body': body, 'tp': tp, 'sl': sl, **s})

best.sort(key=lambda x: x['efficiency'], reverse=True)
for r in best[:10]:
    print(f"{r['label']:>28} {r['trades']:>5} {r['wr']:>5.1f}% {r['total_pnl']:>6.1f}% {r['mdd']:>5.1f}% {r['efficiency']:>6.1f} {r['monthly_avg']:>6.2f}%")

# 최종 비교
print("\n" + "=" * 70)
print("🏆 현재 vs 최적 전략 비교")
print("=" * 70)

if best and base:
    top = best[0]
    print(f"\n{'지표':<15} {'현재전략':>12} {'최적전략':>12} {'개선':>10}")
    print("-" * 55)
    print(f"{'거래 수':<15} {base['trades']:>12} {top['trades']:>12}")
    print(f"{'월 거래':<15} {base['trades']/base['months']:>11.1f}회 {top['trades']/top['months']:>11.1f}회")
    print(f"{'승률':<15} {base['wr']:>11.1f}% {top['wr']:>11.1f}% {top['wr']-base['wr']:>+9.1f}%p")
    print(f"{'총 PnL':<15} {base['total_pnl']:>11.1f}% {top['total_pnl']:>11.1f}% {top['total_pnl']-base['total_pnl']:>+9.1f}%")
    print(f"{'MDD':<15} {base['mdd']:>11.1f}% {top['mdd']:>11.1f}% {top['mdd']-base['mdd']:>+9.1f}%")
    print(f"{'효율(PnL/MDD)':<15} {base['efficiency']:>12.1f} {top['efficiency']:>12.1f} {top['efficiency']-base['efficiency']:>+10.1f}")
    print(f"{'월평균 수익':<15} {base['monthly_avg']:>11.2f}% {top['monthly_avg']:>11.2f}% {top['monthly_avg']-base['monthly_avg']:>+9.2f}%")
    
    print(f"\n🎯 최적 전략: {top['label']}")
    print(f"   조건: 4H 몸통 {top['body']}%+, TP {top['tp']}%, SL {top['sl']}%")
    print(f"   월 거래: {top['trades']/top['months']:.1f}회")

