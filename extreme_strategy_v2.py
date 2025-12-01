import pandas as pd
import numpy as np

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')

df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print(f"15분봉: {len(df_15m)}개, 4시간봉: {len(df_4h)}개")

# FVG 탐지 (4H 기준)
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            gap_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvgs.append({
                'idx': i,
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'fvg_size': gap_size,
                'body_size': body_size,
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)
print(f"4H FVG 총 개수: {len(fvgs_4h)}")

# 분석
results = []

for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    fvg_size = fvg['fvg_size']
    body_size = fvg['body_size']
    
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)
    
    if len(future_15m) < 50:
        continue
    
    retest_idx = None
    entry_price = None
    
    for idx, row in future_15m.iterrows():
        if row['low'] <= fvg_top:
            retest_idx = idx
            entry_price = row['close']
            break
    
    if retest_idx is None:
        continue
    
    entry_loc = df_15m.index.get_loc(retest_idx)
    
    recent_10 = df_15m.iloc[max(0,entry_loc-10):entry_loc+1]
    recent_low = recent_10['low'].min()
    rise_from_low = (entry_price - recent_low) / recent_low * 100
    
    dist_from_fvg = (entry_price - fvg_top) / fvg_top * 100
    
    post_entry = df_15m.iloc[entry_loc+1:entry_loc+101]
    if len(post_entry) < 50:
        continue
    
    max_profit = (post_entry['high'].max() - entry_price) / entry_price * 100
    max_loss = (post_entry['low'].min() - entry_price) / entry_price * 100
    
    results.append({
        'fvg_size': fvg_size,
        'body_size': body_size,
        'rise_from_low': rise_from_low,
        'dist_from_fvg': dist_from_fvg,
        'max_profit': max_profit,
        'max_loss': max_loss,
    })

df_results = pd.DataFrame(results)
print(f"\n분석 대상: {len(df_results)}건\n")

def calc_win_rate(df, tp, sl):
    wins = sum((df['max_profit'] >= tp))
    losses = sum((df['max_profit'] < tp) & (df['max_loss'] <= sl))
    total = wins + losses
    return wins, losses, total, wins/total*100 if total > 0 else 0

print("="*80)
print("🔥 전략 A: 압도적으로 이기기")
print("="*80)

# 4H 몸통 크기가 핵심!
print("\n[4H 몸통 크기별 분석] (TP 3%, SL -2%)")
for body_min in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    cond = df_results['body_size'] >= body_min
    subset = df_results[cond]
    if len(subset) >= 5:
        wins, losses, total, wr = calc_win_rate(subset, 3.0, -2.0)
        monthly = total / 60
        pnl = wins * 3 + losses * (-2)
        avg_max = subset['max_profit'].mean()
        print(f"4H 몸통 {body_min}%+: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 평균최대익 +{avg_max:.2f}%, 5년PnL {pnl:.0f}%")

# 4H 몸통 + FVG 크기
print("\n[4H 몸통 + FVG 크기 조합] (TP 3%, SL -2%)")
combos = [
    ("4H몸통 2%+ & FVG 0.5%+", (df_results['body_size'] >= 2) & (df_results['fvg_size'] >= 0.5)),
    ("4H몸통 2%+ & FVG 1%+", (df_results['body_size'] >= 2) & (df_results['fvg_size'] >= 1)),
    ("4H몸통 2.5%+ & FVG 0.5%+", (df_results['body_size'] >= 2.5) & (df_results['fvg_size'] >= 0.5)),
    ("4H몸통 3%+ & FVG 0.5%+", (df_results['body_size'] >= 3) & (df_results['fvg_size'] >= 0.5)),
    ("4H몸통 3%+ & FVG 1%+", (df_results['body_size'] >= 3) & (df_results['fvg_size'] >= 1)),
]

for name, cond in combos:
    subset = df_results[cond]
    if len(subset) >= 5:
        wins, losses, total, wr = calc_win_rate(subset, 3.0, -2.0)
        monthly = total / 60
        pnl = wins * 3 + losses * (-2)
        avg_max = subset['max_profit'].mean()
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 평균최대익 +{avg_max:.2f}%, 5년PnL {pnl:.0f}%")

# TP 5% 테스트
print("\n[TP 5%, SL -2% 테스트]")
for name, cond in combos:
    subset = df_results[cond]
    if len(subset) >= 5:
        wins, losses, total, wr = calc_win_rate(subset, 5.0, -2.0)
        monthly = total / 60
        pnl = wins * 5 + losses * (-2)
        avg_max = subset['max_profit'].mean()
        tp5_rate = (subset['max_profit'] >= 5).mean() * 100
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, TP5도달 {tp5_rate:.1f}%, 5년PnL {pnl:.0f}%")

# 최적 TP 찾기
print("\n[최적 TP 탐색 - 4H몸통 2%+ 기준]")
best_cond = df_results['body_size'] >= 2
best_subset = df_results[best_cond]
for tp in [2, 3, 4, 5, 6, 7, 8]:
    wins, losses, total, wr = calc_win_rate(best_subset, tp, -2.0)
    pnl = wins * tp + losses * (-2)
    reach = (best_subset['max_profit'] >= tp).mean() * 100
    ev = (wr/100) * tp - ((100-wr)/100) * 2
    print(f"  TP {tp}%: 승률 {wr:.1f}%, 도달률 {reach:.1f}%, EV {ev:.2f}%, 5년PnL {pnl:.0f}%")

print("\n" + "="*80)
print("📊 전략 B: 졸라 많이 이기기")
print("="*80)

# 다양한 TP/SL 조합
print("\n[기본 조건 (FVG위 0.2%+) - TP/SL 최적화]")
base_cond = df_results['dist_from_fvg'] >= 0.2
base_subset = df_results[base_cond]
print(f"기본 데이터: {len(base_subset)}건 ({len(base_subset)/60:.1f}/월)")

for tp in [0.3, 0.5, 0.7, 1.0, 1.5]:
    for sl in [-0.3, -0.5, -0.7, -1.0]:
        wins, losses, total, wr = calc_win_rate(base_subset, tp, sl)
        if total > 0:
            pnl = wins * tp + losses * abs(sl)
            ev = (wr/100) * tp - ((100-wr)/100) * abs(sl)
            if ev > 0:
                print(f"  TP {tp}% / SL {sl}%: 승률 {wr:.1f}%, EV {ev:.3f}%, 5년PnL {pnl:.0f}%")

# 느슨한 조건들 조합
print("\n[최적 다빈도 조합]")
loose_combos = [
    ("기본 (전체)", pd.Series([True]*len(df_results))),
    ("FVG위 0.1%+", df_results['dist_from_fvg'] >= 0.1),
    ("FVG위 0.2%+", df_results['dist_from_fvg'] >= 0.2),
    ("저점 0.3%+ & FVG위 0.1%+", (df_results['rise_from_low'] >= 0.3) & (df_results['dist_from_fvg'] >= 0.1)),
    ("저점 0.5%+ & FVG위 0.1%+", (df_results['rise_from_low'] >= 0.5) & (df_results['dist_from_fvg'] >= 0.1)),
    ("4H몸통 1%+ & FVG위 0.1%+", (df_results['body_size'] >= 1) & (df_results['dist_from_fvg'] >= 0.1)),
]

for name, cond in loose_combos:
    subset = df_results[cond]
    if len(subset) >= 20:
        # TP 0.5%, SL -0.5%
        wins, losses, total, wr = calc_win_rate(subset, 0.5, -0.5)
        monthly = total / 60
        pnl = wins * 0.5 + losses * (-0.5)
        ev = (wr/100) * 0.5 - ((100-wr)/100) * 0.5
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, EV {ev:.3f}%, 5년PnL {pnl:.0f}%")

print("\n" + "="*80)
print("🏆 최종 결론")
print("="*80)

# 전략 A 최적
print("\n[전략 A - 압도적 승리]")
a_cond = df_results['body_size'] >= 2
a_subset = df_results[a_cond]
wins_a, losses_a, total_a, wr_a = calc_win_rate(a_subset, 4.0, -2.0)
pnl_a = wins_a * 4 + losses_a * (-2)
ev_a = (wr_a/100) * 4 - ((100-wr_a)/100) * 2
monthly_a = total_a / 60
print(f"조건: 4H 몸통 2%+")
print(f"TP/SL: 4% / -2%")
print(f"거래수: {total_a}건 ({monthly_a:.1f}/월)")
print(f"승률: {wr_a:.1f}%")
print(f"1회 EV: {ev_a:.2f}%")
print(f"5년 총 PnL: {pnl_a:.0f}%")
print(f"월 기대수익 (10배): {ev_a * monthly_a * 10:.1f}%")

# 전략 B 최적
print("\n[전략 B - 다빈도]")
b_cond = df_results['dist_from_fvg'] >= 0.1
b_subset = df_results[b_cond]
wins_b, losses_b, total_b, wr_b = calc_win_rate(b_subset, 0.5, -0.5)
pnl_b = wins_b * 0.5 + losses_b * (-0.5)
ev_b = (wr_b/100) * 0.5 - ((100-wr_b)/100) * 0.5
monthly_b = total_b / 60
print(f"조건: FVG 위 0.1%+")
print(f"TP/SL: 0.5% / -0.5%")
print(f"거래수: {total_b}건 ({monthly_b:.1f}/월)")
print(f"승률: {wr_b:.1f}%")
print(f"1회 EV: {ev_b:.3f}%")
print(f"5년 총 PnL: {pnl_b:.0f}%")
print(f"월 기대수익 (10배): {ev_b * monthly_b * 10:.1f}%")

# 비교
print("\n" + "="*80)
print("📈 전략 비교 (10배 레버리지 기준)")
print("="*80)
print(f"전략 A: 월 {monthly_a:.1f}회 × {ev_a:.2f}% × 10배 = 월 {ev_a * monthly_a * 10:.1f}%")
print(f"전략 B: 월 {monthly_b:.1f}회 × {ev_b:.3f}% × 10배 = 월 {ev_b * monthly_b * 10:.1f}%")

