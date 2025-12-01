import pandas as pd
import numpy as np
from datetime import datetime

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')

df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# 4H FVG 감지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2 = df.iloc[i-2]
        prev1 = df.iloc[i-1]
        curr = df.iloc[i]
        
        # 상승 FVG: 2봉전 고가 < 현재 저가 (갭 존재)
        if prev2['high'] < curr['low']:
            fvg_top = curr['low']
            fvg_bottom = prev2['high']
            fvg_size = (fvg_top - fvg_bottom) / fvg_bottom * 100
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': fvg_top,
                'fvg_bottom': fvg_bottom,
                'fvg_size': fvg_size,
                'body_size': body_size,
                'candle_close': curr['close']
            })
    return fvgs

fvgs = detect_4h_fvg(df_4h)
print(f"감지된 4H FVG: {len(fvgs)}개\n")

# 트레이드 시뮬레이션 함수
def simulate_trades(fvgs, df_15m, tp_pct, sl_pct, conditions=None):
    trades = []
    
    for fvg in fvgs:
        # 조건 필터
        if conditions:
            if 'min_body' in conditions and fvg['body_size'] < conditions['min_body']:
                continue
            if 'min_fvg' in conditions and fvg['fvg_size'] < conditions['min_fvg']:
                continue
        
        fvg_time = fvg['datetime']
        fvg_top = fvg['fvg_top']
        
        # 15분 데이터에서 FVG 이후 데이터 찾기
        future_15m = df_15m[df_15m['datetime'] > fvg_time].head(200)
        
        entry_price = None
        entry_time = None
        
        # FVG 상단 터치 찾기
        for idx, row in future_15m.iterrows():
            if row['low'] <= fvg_top:
                # 다음 봉 시가로 진입
                next_idx = future_15m.index.get_loc(idx)
                if next_idx + 1 < len(future_15m):
                    next_row = future_15m.iloc[next_idx + 1]
                    entry_price = next_row['open']
                    entry_time = next_row['datetime']
                break
        
        if entry_price is None:
            continue
        
        # TP/SL 계산
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        
        # 결과 판정
        after_entry = df_15m[df_15m['datetime'] > entry_time].head(200)
        result = None
        exit_time = None
        
        for _, row in after_entry.iterrows():
            # SL 먼저 체크 (open -> low -> high -> close 순서 가정)
            if row['low'] <= sl_price:
                result = 'loss'
                exit_time = row['datetime']
                break
            if row['high'] >= tp_price:
                result = 'win'
                exit_time = row['datetime']
                break
        
        if result:
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'result': result,
                'pnl': tp_pct if result == 'win' else sl_pct,
                'body_size': fvg['body_size'],
                'fvg_size': fvg['fvg_size']
            })
    
    return trades

# MDD 계산 함수
def calculate_mdd(trades):
    if not trades:
        return 0, 0
    
    cumulative = 0
    peak = 0
    mdd = 0
    
    for t in trades:
        cumulative += t['pnl']
        if cumulative > peak:
            peak = cumulative
        drawdown = cumulative - peak
        if drawdown < mdd:
            mdd = drawdown
    
    return mdd, cumulative

# 월별 수익 계산
def monthly_stats(trades):
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    df['month'] = df['entry_time'].dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    
    return {
        'monthly_avg': monthly.mean(),
        'monthly_std': monthly.std(),
        'positive_months': (monthly > 0).sum(),
        'total_months': len(monthly),
        'worst_month': monthly.min(),
        'best_month': monthly.max()
    }

print("=" * 70)
print("🔍 MDD 줄이고 수익 늘리기 위한 최적화 분석")
print("=" * 70)

# 1. 현재 전략 (기준점)
print("\n[현재 전략] TP 1.0% / SL -1.5%")
trades_base = simulate_trades(fvgs, df_15m, 1.0, -1.5)
mdd_base, total_pnl_base = calculate_mdd(trades_base)
wins_base = len([t for t in trades_base if t['result'] == 'win'])
wr_base = wins_base / len(trades_base) * 100 if trades_base else 0
stats_base = monthly_stats(trades_base)

print(f"  거래: {len(trades_base)}회, 승률: {wr_base:.1f}%")
print(f"  총 PnL: {total_pnl_base:.1f}%, MDD: {mdd_base:.1f}%")
print(f"  월평균: {stats_base.get('monthly_avg', 0):.2f}%, 최악의 달: {stats_base.get('worst_month', 0):.2f}%")

# 2. TP/SL 조합 테스트
print("\n" + "=" * 70)
print("📊 방법 1: TP/SL 비율 조정")
print("=" * 70)

tp_sl_combos = [
    (1.0, -1.0),
    (1.0, -1.5),
    (1.5, -1.0),
    (1.5, -1.5),
    (2.0, -1.0),
    (2.0, -1.5),
    (2.0, -2.0),
    (2.5, -1.5),
    (3.0, -1.5),
    (3.0, -2.0),
]

results = []
for tp, sl in tp_sl_combos:
    trades = simulate_trades(fvgs, df_15m, tp, sl)
    if len(trades) < 50:
        continue
    mdd, total_pnl = calculate_mdd(trades)
    wins = len([t for t in trades if t['result'] == 'win'])
    wr = wins / len(trades) * 100
    stats = monthly_stats(trades)
    
    # 효율성 지표: 수익/MDD 비율
    efficiency = total_pnl / abs(mdd) if mdd != 0 else 0
    
    results.append({
        'tp': tp, 'sl': sl,
        'trades': len(trades),
        'wr': wr,
        'total_pnl': total_pnl,
        'mdd': mdd,
        'efficiency': efficiency,
        'monthly_avg': stats.get('monthly_avg', 0),
        'worst_month': stats.get('worst_month', 0)
    })

# 효율성 순 정렬
results.sort(key=lambda x: x['efficiency'], reverse=True)
print(f"\n{'TP':>5} {'SL':>5} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7} {'최악달':>7}")
print("-" * 70)
for r in results[:10]:
    print(f"{r['tp']:>5.1f} {r['sl']:>5.1f} {r['trades']:>5} {r['wr']:>5.1f}% {r['total_pnl']:>6.1f}% {r['mdd']:>5.1f}% {r['efficiency']:>6.1f} {r['monthly_avg']:>6.2f}% {r['worst_month']:>6.2f}%")

# 3. 4H 몸통 크기 필터 추가
print("\n" + "=" * 70)
print("📊 방법 2: 4H 몸통 크기 필터 (TP 1.0% / SL -1.5%)")
print("=" * 70)

body_filters = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"\n{'몸통':>6} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7} {'월거래':>6}")
print("-" * 70)

for min_body in body_filters:
    trades = simulate_trades(fvgs, df_15m, 1.0, -1.5, {'min_body': min_body})
    if len(trades) < 20:
        continue
    mdd, total_pnl = calculate_mdd(trades)
    wins = len([t for t in trades if t['result'] == 'win'])
    wr = wins / len(trades) * 100
    stats = monthly_stats(trades)
    efficiency = total_pnl / abs(mdd) if mdd != 0 else 0
    monthly_trades = len(trades) / 69  # 69개월
    
    print(f"{min_body:>5.1f}% {len(trades):>5} {wr:>5.1f}% {total_pnl:>6.1f}% {mdd:>5.1f}% {efficiency:>6.1f} {stats.get('monthly_avg', 0):>6.2f}% {monthly_trades:>5.1f}회")

# 4. 복합 조건 테스트
print("\n" + "=" * 70)
print("📊 방법 3: 복합 조건 (몸통 + TP/SL 최적화)")
print("=" * 70)

combos = [
    {'min_body': 1.0, 'tp': 1.5, 'sl': -1.0},
    {'min_body': 1.0, 'tp': 2.0, 'sl': -1.0},
    {'min_body': 1.0, 'tp': 2.0, 'sl': -1.5},
    {'min_body': 1.5, 'tp': 1.5, 'sl': -1.0},
    {'min_body': 1.5, 'tp': 2.0, 'sl': -1.0},
    {'min_body': 1.5, 'tp': 2.0, 'sl': -1.5},
    {'min_body': 2.0, 'tp': 1.5, 'sl': -1.0},
    {'min_body': 2.0, 'tp': 2.0, 'sl': -1.0},
    {'min_body': 2.0, 'tp': 2.0, 'sl': -1.5},
    {'min_body': 2.0, 'tp': 2.5, 'sl': -1.5},
    {'min_body': 2.0, 'tp': 3.0, 'sl': -1.5},
]

print(f"\n{'조건':>20} {'거래':>5} {'승률':>6} {'총PnL':>7} {'MDD':>6} {'효율':>6} {'월평균':>7} {'월거래':>6}")
print("-" * 80)

best_results = []
for c in combos:
    trades = simulate_trades(fvgs, df_15m, c['tp'], c['sl'], {'min_body': c['min_body']})
    if len(trades) < 15:
        continue
    mdd, total_pnl = calculate_mdd(trades)
    wins = len([t for t in trades if t['result'] == 'win'])
    wr = wins / len(trades) * 100
    stats = monthly_stats(trades)
    efficiency = total_pnl / abs(mdd) if mdd != 0 else 0
    monthly_trades = len(trades) / 69
    
    label = f"몸통{c['min_body']}%+ TP{c['tp']} SL{c['sl']}"
    print(f"{label:>20} {len(trades):>5} {wr:>5.1f}% {total_pnl:>6.1f}% {mdd:>5.1f}% {efficiency:>6.1f} {stats.get('monthly_avg', 0):>6.2f}% {monthly_trades:>5.1f}회")
    
    best_results.append({
        'label': label,
        'trades': len(trades),
        'wr': wr,
        'total_pnl': total_pnl,
        'mdd': mdd,
        'efficiency': efficiency,
        'monthly_avg': stats.get('monthly_avg', 0),
        'monthly_trades': monthly_trades
    })

# 5. 최적 전략 비교
print("\n" + "=" * 70)
print("🏆 최적 전략 비교 (MDD 대비 수익 효율)")
print("=" * 70)

best_results.sort(key=lambda x: x['efficiency'], reverse=True)
print("\n[TOP 5 효율 기준]")
for i, r in enumerate(best_results[:5], 1):
    print(f"\n{i}. {r['label']}")
    print(f"   거래: {r['trades']}회 ({r['monthly_trades']:.1f}회/월), 승률: {r['wr']:.1f}%")
    print(f"   총 PnL: {r['total_pnl']:.1f}%, MDD: {r['mdd']:.1f}%, 효율: {r['efficiency']:.1f}")
    print(f"   월평균 수익: {r['monthly_avg']:.2f}%")

# 현재 전략과 최고 전략 비교
print("\n" + "=" * 70)
print("📈 현재 vs 최적 전략 비교")
print("=" * 70)

if best_results:
    best = best_results[0]
    print(f"\n{'지표':<15} {'현재 전략':>15} {'최적 전략':>15} {'개선':>10}")
    print("-" * 60)
    print(f"{'거래 수':<15} {len(trades_base):>15} {best['trades']:>15} -")
    print(f"{'월 거래':<15} {len(trades_base)/69:>14.1f}회 {best['monthly_trades']:>14.1f}회 -")
    print(f"{'승률':<15} {wr_base:>14.1f}% {best['wr']:>14.1f}% {best['wr']-wr_base:>+9.1f}%p")
    print(f"{'총 PnL':<15} {total_pnl_base:>14.1f}% {best['total_pnl']:>14.1f}% {best['total_pnl']-total_pnl_base:>+9.1f}%")
    print(f"{'MDD':<15} {mdd_base:>14.1f}% {best['mdd']:>14.1f}% {best['mdd']-mdd_base:>+9.1f}%")
    print(f"{'효율(PnL/MDD)':<15} {total_pnl_base/abs(mdd_base):>15.1f} {best['efficiency']:>15.1f} {best['efficiency']-(total_pnl_base/abs(mdd_base)):>+10.1f}")
    print(f"{'월평균 수익':<15} {stats_base.get('monthly_avg', 0):>14.2f}% {best['monthly_avg']:>14.2f}% {best['monthly_avg']-stats_base.get('monthly_avg', 0):>+9.2f}%")

