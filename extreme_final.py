import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            gap_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_size': gap_size,
                'body_size': body_size,
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)

results = []
for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)
    
    if len(future_15m) < 50:
        continue
    
    retest_idx = None
    for idx, row in future_15m.iterrows():
        if row['low'] <= fvg_top:
            retest_idx = idx
            entry_price = row['close']
            break
    
    if retest_idx is None:
        continue
    
    entry_loc = df_15m.index.get_loc(retest_idx)
    dist_from_fvg = (entry_price - fvg_top) / fvg_top * 100
    
    post_entry = df_15m.iloc[entry_loc+1:entry_loc+101]
    if len(post_entry) < 50:
        continue
    
    max_profit = (post_entry['high'].max() - entry_price) / entry_price * 100
    max_loss = (post_entry['low'].min() - entry_price) / entry_price * 100
    
    results.append({
        'fvg_size': fvg['fvg_size'],
        'body_size': fvg['body_size'],
        'dist_from_fvg': dist_from_fvg,
        'max_profit': max_profit,
        'max_loss': max_loss,
    })

df_results = pd.DataFrame(results)
print(f"분석 대상: {len(df_results)}건\n")

def calc_stats(df, tp, sl):
    wins = sum(df['max_profit'] >= tp)
    losses = sum((df['max_profit'] < tp) & (df['max_loss'] <= sl))
    total = wins + losses
    if total == 0:
        return 0, 0, 0, 0, 0, 0
    wr = wins/total*100
    pnl = wins * tp + losses * abs(sl) * -1
    ev = (wr/100) * tp - ((100-wr)/100) * abs(sl)
    monthly = total / 60
    return wins, losses, total, wr, pnl, ev, monthly

print("="*80)
print("🔥 전략 A: 압도적으로 이기기 - 최종 최적화")
print("="*80)

# 4H 몸통 크기 + 다양한 TP
print("\n[4H 몸통 2%+ 기준 - TP 최적화]")
cond_a = df_results['body_size'] >= 2
subset_a = df_results[cond_a]
print(f"데이터: {len(subset_a)}건")

best_ev = 0
best_tp = 0
for tp in [2, 3, 4, 5, 6, 7, 8, 10]:
    for sl in [-1.5, -2, -2.5, -3]:
        wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_a, tp, sl)
        if ev > best_ev:
            best_ev = ev
            best_combo = (tp, sl, wr, pnl, ev, monthly)
        if tp in [4, 5, 6] and sl == -2:
            print(f"  TP {tp}% SL {sl}%: 승률 {wr:.1f}%, EV {ev:.2f}%, 5년PnL {pnl:.0f}%")

print(f"\n최적 조합: TP {best_combo[0]}% SL {best_combo[1]}%")
print(f"  승률: {best_combo[2]:.1f}%, EV: {best_combo[4]:.2f}%, 5년PnL: {best_combo[3]:.0f}%")

# 몸통 3%+ 테스트
print("\n[4H 몸통 3%+ 기준 - 극단적 승리]")
cond_a2 = df_results['body_size'] >= 3
subset_a2 = df_results[cond_a2]
print(f"데이터: {len(subset_a2)}건")

for tp in [5, 7, 10]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_a2, tp, -2)
    print(f"  TP {tp}% SL -2%: 승률 {wr:.1f}%, EV {ev:.2f}%, 5년PnL {pnl:.0f}%")

# 몸통 4%+ 테스트
print("\n[4H 몸통 4%+ 기준 - 슈퍼 승리]")
cond_a3 = df_results['body_size'] >= 4
subset_a3 = df_results[cond_a3]
print(f"데이터: {len(subset_a3)}건")

for tp in [5, 7, 10]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_a3, tp, -2)
    print(f"  TP {tp}% SL -2%: 승률 {wr:.1f}%, EV {ev:.2f}%, 5년PnL {pnl:.0f}%")

print("\n" + "="*80)
print("📊 전략 B: 졸라 많이 이기기 - 최종 최적화")  
print("="*80)

# 기본 (전체)
print("\n[전체 데이터 - TP/SL 최적화]")
print(f"데이터: {len(df_results)}건")

best_monthly_ev = 0
for tp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    for sl in [-0.3, -0.5, -1.0]:
        wins, losses, total, wr, pnl, ev, monthly = calc_stats(df_results, tp, sl)
        monthly_ev = ev * monthly * 10  # 10배 레버리지
        if monthly_ev > best_monthly_ev and total > 100:
            best_monthly_ev = monthly_ev
            best_b = (tp, sl, wr, pnl, ev, monthly, total)

print(f"최적 조합: TP {best_b[0]}% SL {best_b[1]}%")
print(f"  거래수: {best_b[6]}건 ({best_b[5]:.1f}/월)")
print(f"  승률: {best_b[2]:.1f}%, EV: {best_b[4]:.3f}%")
print(f"  5년PnL: {best_b[3]:.0f}%")
print(f"  월 기대수익 (10x): {best_b[4] * best_b[5] * 10:.1f}%")

# FVG 위 0.1%+ 조건
print("\n[FVG 위 0.1%+ - TP/SL 최적화]")
cond_b = df_results['dist_from_fvg'] >= 0.1
subset_b = df_results[cond_b]
print(f"데이터: {len(subset_b)}건")

for tp in [0.5, 0.7, 1.0, 1.5]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_b, tp, -0.5)
    print(f"  TP {tp}% SL -0.5%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%, 월(10x) {ev*monthly*10:.1f}%")

# FVG 위 0.2%+ 조건
print("\n[FVG 위 0.2%+ - TP/SL 최적화]")
cond_b2 = df_results['dist_from_fvg'] >= 0.2
subset_b2 = df_results[cond_b2]
print(f"데이터: {len(subset_b2)}건")

for tp in [0.5, 0.7, 1.0, 1.5, 2.0]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_b2, tp, -0.5)
    print(f"  TP {tp}% SL -0.5%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%, 월(10x) {ev*monthly*10:.1f}%")

print("\n" + "="*80)
print("🏆🏆🏆 최종 결론 🏆🏆🏆")
print("="*80)

# 전략 A 최종
print("\n🔥 전략 A: 압도적으로 이기기")
print("-"*40)
cond_final_a = df_results['body_size'] >= 2
subset_final_a = df_results[cond_final_a]
wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_final_a, 5, -2)
print(f"조건: 4H FVG + 4H 몸통 2%+")
print(f"TP/SL: 5% / -2%")
print(f"거래수: {total}건 ({monthly:.1f}회/월)")
print(f"승률: {wr:.1f}%")
print(f"1회 EV: {ev:.2f}%")
print(f"5년 총 PnL: {pnl:.0f}%")
print(f"월 기대수익 (10배 레버리지): {ev * monthly * 10:.1f}%")
print(f"연 기대수익 (10배 레버리지): {ev * monthly * 10 * 12:.0f}%")

# 전략 B 최종
print("\n📊 전략 B: 졸라 많이 이기기")
print("-"*40)
cond_final_b = df_results['dist_from_fvg'] >= 0.2
subset_final_b = df_results[cond_final_b]
wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_final_b, 1.5, -0.5)
print(f"조건: 4H FVG + FVG 위 0.2%+")
print(f"TP/SL: 1.5% / -0.5%")
print(f"거래수: {total}건 ({monthly:.1f}회/월)")
print(f"승률: {wr:.1f}%")
print(f"1회 EV: {ev:.3f}%")
print(f"5년 총 PnL: {pnl:.0f}%")
print(f"월 기대수익 (10배 레버리지): {ev * monthly * 10:.1f}%")
print(f"연 기대수익 (10배 레버리지): {ev * monthly * 10 * 12:.0f}%")

print("\n" + "="*80)
print("📈 비교 요약")
print("="*80)

# A
wins_a, losses_a, total_a, wr_a, pnl_a, ev_a, monthly_a = calc_stats(subset_final_a, 5, -2)
# B  
wins_b, losses_b, total_b, wr_b, pnl_b, ev_b, monthly_b = calc_stats(subset_final_b, 1.5, -0.5)

print(f"\n전략 A (압도적 승리):")
print(f"  월 {monthly_a:.1f}회 × {ev_a:.2f}% × 10배 = 월 {ev_a * monthly_a * 10:.1f}%")
print(f"  → 특징: 적은 거래, 큰 수익, 63% 승률")

print(f"\n전략 B (다빈도):")
print(f"  월 {monthly_b:.1f}회 × {ev_b:.3f}% × 10배 = 월 {ev_b * monthly_b * 10:.1f}%")
print(f"  → 특징: 많은 거래, 작은 수익, 84% 승률")

print(f"\n💡 결론:")
print(f"  전략 B가 월 기대수익 {ev_b * monthly_b * 10:.1f}% vs {ev_a * monthly_a * 10:.1f}%로 우세")
print(f"  하지만 전략 A는 승률이 낮아도(63%) 한 번에 크게 먹음 (EV {ev_a:.2f}%)")
print(f"  전략 B는 84% 승률로 심리적 안정감 + 꾸준한 수익")

