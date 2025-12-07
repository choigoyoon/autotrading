import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("🔧 현재 전략 개선 - 승률 70%+ 목표")
print("="*80)

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            # 추가 정보
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvg_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            
            # 이전 4H봉 정보
            prev_4h_bullish = prev1['close'] > prev1['open']
            prev2_4h_bullish = prev2['close'] > prev2['open']
            
            fvgs.append({
                'idx': i,
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'body_size': body_size,
                'fvg_size': fvg_size,
                'prev_4h_bullish': prev_4h_bullish,
                'prev2_4h_bullish': prev2_4h_bullish,
                'candle_high': curr['high'],
                'candle_low': curr['low'],
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)
print(f"4H FVG: {len(fvgs_4h)}개")

# 거래 시뮬레이션 함수
def simulate_trades(fvgs, tp, sl, conditions=None):
    trades = []
    
    for fvg in fvgs:
        # 조건 필터
        if conditions:
            skip = False
            for key, (op, val) in conditions.items():
                if key not in fvg:
                    continue
                if op == '>=' and fvg[key] < val:
                    skip = True
                elif op == '<=' and fvg[key] > val:
                    skip = True
                elif op == '==' and fvg[key] != val:
                    skip = True
            if skip:
                continue
        
        fvg_time = fvg['datetime']
        fvg_top = fvg['fvg_top']
        
        mask = df_15m['datetime'] > fvg_time
        future_15m = df_15m[mask].head(200)
        
        if len(future_15m) < 50:
            continue
        
        # 터치 찾기
        for i, (idx, row) in enumerate(future_15m.iterrows()):
            if row['low'] <= fvg_top:
                remaining = future_15m.iloc[i+1:]
                if len(remaining) < 50:
                    break
                
                entry_price = remaining.iloc[0]['open']
                entry_bar = remaining.iloc[0]
                
                # 진입봉 정보
                entry_bullish = entry_bar['close'] > entry_bar['open']
                
                # TP/SL 순서 체크
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
                    trades.append({
                        'result': result,
                        'body_size': fvg['body_size'],
                        'fvg_size': fvg['fvg_size'],
                        'prev_4h_bullish': fvg['prev_4h_bullish'],
                        'entry_bullish': entry_bullish,
                    })
                break
    
    return trades

# 기본 전략 결과
print("\n[기본 전략] TP 1.5%, SL -0.5%")
base_trades = simulate_trades(fvgs_4h, 1.5, -0.5)
wins = sum(1 for t in base_trades if t['result'] == 'WIN')
total = len(base_trades)
print(f"거래: {total}건, 승률: {wins/total*100:.1f}%")

# 개선 방향 테스트
print("\n" + "="*80)
print("📊 개선 조건 테스트")
print("="*80)

# 1. TP/SL 비율 조정
print("\n[1] TP/SL 비율 조정")
for tp in [1.0, 1.2, 1.5, 2.0]:
    for sl in [-0.5, -0.7, -1.0, -1.5]:
        trades = simulate_trades(fvgs_4h, tp, sl)
        if len(trades) < 50:
            continue
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        wr = wins/len(trades)*100
        ev = (wr/100)*tp - ((100-wr)/100)*abs(sl)
        if wr >= 65:  # 65% 이상만 출력
            print(f"  TP {tp}% SL {sl}%: {len(trades)}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

# 2. 4H 몸통 크기
print("\n[2] 4H 몸통 크기 필터")
for body_min in [1.5, 2.0, 2.5, 3.0]:
    cond = {'body_size': ('>=', body_min)}
    trades = simulate_trades(fvgs_4h, 1.5, -0.5, cond)
    if len(trades) < 10:
        continue
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    wr = wins/len(trades)*100
    ev = (wr/100)*1.5 - ((100-wr)/100)*0.5
    print(f"  몸통 {body_min}%+: {len(trades)}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

# 3. FVG 크기
print("\n[3] FVG 크기 필터")
for fvg_min in [0.3, 0.5, 0.7, 1.0]:
    cond = {'fvg_size': ('>=', fvg_min)}
    trades = simulate_trades(fvgs_4h, 1.5, -0.5, cond)
    if len(trades) < 10:
        continue
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    wr = wins/len(trades)*100
    ev = (wr/100)*1.5 - ((100-wr)/100)*0.5
    print(f"  FVG {fvg_min}%+: {len(trades)}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

# 4. 이전 4H봉 방향
print("\n[4] 이전 4H봉 방향")
cond = {'prev_4h_bullish': ('==', True)}
trades = simulate_trades(fvgs_4h, 1.5, -0.5, cond)
wins = sum(1 for t in trades if t['result'] == 'WIN')
wr = wins/len(trades)*100
print(f"  이전 4H 양봉: {len(trades)}건, 승률 {wr:.1f}%")

cond = {'prev_4h_bullish': ('==', False)}
trades = simulate_trades(fvgs_4h, 1.5, -0.5, cond)
wins = sum(1 for t in trades if t['result'] == 'WIN')
wr = wins/len(trades)*100
print(f"  이전 4H 음봉: {len(trades)}건, 승률 {wr:.1f}%")

# 5. 조합 테스트
print("\n" + "="*80)
print("🎯 조합 테스트 (승률 65%+ 목표)")
print("="*80)

combos = [
    ("몸통 2%+ & TP 1.5%/SL -0.5%", {'body_size': ('>=', 2)}, 1.5, -0.5),
    ("몸통 2%+ & TP 1.2%/SL -0.5%", {'body_size': ('>=', 2)}, 1.2, -0.5),
    ("몸통 2%+ & TP 1.0%/SL -0.5%", {'body_size': ('>=', 2)}, 1.0, -0.5),
    ("몸통 2%+ & TP 1.5%/SL -1.0%", {'body_size': ('>=', 2)}, 1.5, -1.0),
    ("몸통 2.5%+ & TP 1.5%/SL -0.5%", {'body_size': ('>=', 2.5)}, 1.5, -0.5),
    ("몸통 2.5%+ & TP 1.0%/SL -0.5%", {'body_size': ('>=', 2.5)}, 1.0, -0.5),
    ("FVG 0.5%+ & TP 1.5%/SL -0.5%", {'fvg_size': ('>=', 0.5)}, 1.5, -0.5),
    ("FVG 0.5%+ & TP 1.0%/SL -0.5%", {'fvg_size': ('>=', 0.5)}, 1.0, -0.5),
]

print(f"\n{'조합':<35} {'거래':>6} {'승률':>8} {'EV':>8} {'월거래':>8}")
print("-"*70)

for name, cond, tp, sl in combos:
    trades = simulate_trades(fvgs_4h, tp, sl, cond)
    if len(trades) < 10:
        continue
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    wr = wins/len(trades)*100
    ev = (wr/100)*tp - ((100-wr)/100)*abs(sl)
    monthly = len(trades)/60
    print(f"{name:<35} {len(trades):>6} {wr:>7.1f}% {ev:>7.3f}% {monthly:>7.1f}")

# 최적 조합 찾기
print("\n" + "="*80)
print("🏆 최적 조합 탐색")
print("="*80)

best_results = []

for body_min in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for tp in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for sl in [-0.3, -0.5, -0.7, -1.0]:
            cond = {'body_size': ('>=', body_min)}
            trades = simulate_trades(fvgs_4h, tp, sl, cond)
            if len(trades) < 20:
                continue
            wins = sum(1 for t in trades if t['result'] == 'WIN')
            wr = wins/len(trades)*100
            ev = (wr/100)*tp - ((100-wr)/100)*abs(sl)
            monthly = len(trades)/60
            
            if wr >= 65 and ev > 0.3:  # 승률 65%+, EV 0.3%+
                best_results.append({
                    'body': body_min,
                    'tp': tp,
                    'sl': sl,
                    'trades': len(trades),
                    'wr': wr,
                    'ev': ev,
                    'monthly': monthly,
                })

# 정렬 (EV 기준)
best_results.sort(key=lambda x: x['ev'], reverse=True)

print(f"\n{'몸통':>6} {'TP':>6} {'SL':>6} {'거래':>6} {'승률':>8} {'EV':>8} {'월거래':>8}")
print("-"*60)

for r in best_results[:15]:
    print(f"{r['body']:>5.1f}% {r['tp']:>5.1f}% {r['sl']:>5.1f}% {r['trades']:>6} {r['wr']:>7.1f}% {r['ev']:>7.3f}% {r['monthly']:>7.1f}")

