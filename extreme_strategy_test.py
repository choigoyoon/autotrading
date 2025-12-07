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
        # 상승 FVG: 2봉전 고가 < 현재 저가
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            gap_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvgs.append({
                'idx': i,
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'fvg_mid': (curr['low'] + prev2['high']) / 2,
                'fvg_size': gap_size,
                'body_size': body_size,
                'candle_low': curr['low'],
                'candle_high': curr['high']
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)
print(f"4H FVG 총 개수: {len(fvgs_4h)}")

# 15분봉에서 리테스트 및 결과 분석
results = []

for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    fvg_bottom = fvg['fvg_bottom']
    fvg_mid = fvg['fvg_mid']
    fvg_size = fvg['fvg_size']
    body_size = fvg['body_size']
    
    # 해당 4H봉 이후 15분봉 찾기
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)  # 50시간
    
    if len(future_15m) < 50:
        continue
    
    # 리테스트 찾기 (FVG 영역 터치)
    retest_idx = None
    entry_price = None
    
    for idx, row in future_15m.iterrows():
        if row['low'] <= fvg_top:  # FVG 상단 터치
            retest_idx = idx
            entry_price = row['close']
            entry_time = row['datetime']
            break
    
    if retest_idx is None:
        continue
    
    # 진입 시점의 조건들 계산
    entry_loc = df_15m.index.get_loc(retest_idx)
    
    # 최근 10봉 저점 대비 상승폭
    recent_10 = df_15m.iloc[max(0,entry_loc-10):entry_loc+1]
    recent_low = recent_10['low'].min()
    rise_from_low = (entry_price - recent_low) / recent_low * 100
    
    # FVG 대비 위치
    dist_from_fvg = (entry_price - fvg_top) / fvg_top * 100
    
    # 최근 20봉 고점 대비
    recent_20 = df_15m.iloc[max(0,entry_loc-20):entry_loc+1]
    recent_high = recent_20['high'].max()
    drop_from_high = (entry_price - recent_high) / recent_high * 100
    
    # 최근 3봉 중 양봉 수
    recent_3 = df_15m.iloc[max(0,entry_loc-2):entry_loc+1]
    bullish_count = sum(recent_3['close'] > recent_3['open'])
    
    # 진입 후 결과 (100봉 = 25시간)
    post_entry = df_15m.iloc[entry_loc+1:entry_loc+101]
    if len(post_entry) < 50:
        continue
    
    max_profit = (post_entry['high'].max() - entry_price) / entry_price * 100
    max_loss = (post_entry['low'].min() - entry_price) / entry_price * 100
    final_pnl = (post_entry.iloc[-1]['close'] - entry_price) / entry_price * 100
    
    results.append({
        'fvg_size': fvg_size,
        'body_size': body_size,
        'rise_from_low': rise_from_low,
        'dist_from_fvg': dist_from_fvg,
        'drop_from_high': drop_from_high,
        'bullish_count': bullish_count,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'final_pnl': final_pnl
    })

df_results = pd.DataFrame(results)
print(f"\n분석 대상: {len(df_results)}건")

# TP/SL 별 승패 계산 함수
def calc_win_rate(df, tp, sl):
    wins = 0
    losses = 0
    for _, row in df.iterrows():
        if row['max_profit'] >= tp:
            if row['max_loss'] <= sl:
                wins += 1
            else:
                wins += 1
        elif row['max_loss'] <= sl:
            losses += 1
    total = wins + losses
    return wins, losses, total, wins/total*100 if total > 0 else 0

print("\n" + "="*80)
print("🔥 전략 A: 압도적으로 이기기 (높은 TP, 높은 승률)")
print("="*80)

# 극단적 조건 조합 테스트
conditions_extreme = [
    ("저점 대비 3%+ 상승", df_results['rise_from_low'] >= 3),
    ("저점 대비 4%+ 상승", df_results['rise_from_low'] >= 4),
    ("저점 대비 5%+ 상승", df_results['rise_from_low'] >= 5),
    ("FVG 위 1%+", df_results['dist_from_fvg'] >= 1),
    ("FVG 위 1.5%+", df_results['dist_from_fvg'] >= 1.5),
    ("FVG 위 2%+", df_results['dist_from_fvg'] >= 2),
    ("4H 몸통 2%+", df_results['body_size'] >= 2),
    ("4H 몸통 3%+", df_results['body_size'] >= 3),
    ("FVG 크기 1%+", df_results['fvg_size'] >= 1),
    ("FVG 크기 1.5%+", df_results['fvg_size'] >= 1.5),
]

print("\n[개별 극단 조건] (TP 3%, SL -2%)")
for name, cond in conditions_extreme:
    subset = df_results[cond]
    if len(subset) >= 10:
        wins, losses, total, wr = calc_win_rate(subset, 3.0, -2.0)
        avg_max_profit = subset['max_profit'].mean()
        print(f"{name}: {total}건, 승률 {wr:.1f}%, 평균최대익 +{avg_max_profit:.2f}%")

# 조합 테스트 - 압도적 승리 조건
print("\n[압도적 승리 조합] (TP 3%, SL -2%)")
combos_extreme = [
    ("저점 3%+ & FVG위 1%+", 
     (df_results['rise_from_low'] >= 3) & (df_results['dist_from_fvg'] >= 1)),
    ("저점 4%+ & FVG위 1%+", 
     (df_results['rise_from_low'] >= 4) & (df_results['dist_from_fvg'] >= 1)),
    ("저점 3%+ & FVG위 1.5%+", 
     (df_results['rise_from_low'] >= 3) & (df_results['dist_from_fvg'] >= 1.5)),
    ("저점 4%+ & FVG위 1.5%+", 
     (df_results['rise_from_low'] >= 4) & (df_results['dist_from_fvg'] >= 1.5)),
    ("저점 3%+ & FVG위 1%+ & 4H몸통 2%+", 
     (df_results['rise_from_low'] >= 3) & (df_results['dist_from_fvg'] >= 1) & (df_results['body_size'] >= 2)),
    ("저점 4%+ & FVG위 1%+ & 4H몸통 2%+", 
     (df_results['rise_from_low'] >= 4) & (df_results['dist_from_fvg'] >= 1) & (df_results['body_size'] >= 2)),
    ("저점 5%+ & FVG위 1%+", 
     (df_results['rise_from_low'] >= 5) & (df_results['dist_from_fvg'] >= 1)),
]

for name, cond in combos_extreme:
    subset = df_results[cond]
    if len(subset) >= 5:
        wins, losses, total, wr = calc_win_rate(subset, 3.0, -2.0)
        avg_max_profit = subset['max_profit'].mean()
        avg_max_loss = subset['max_loss'].mean()
        monthly = total / 60
        print(f"{name}")
        print(f"  → {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 최대익 +{avg_max_profit:.2f}%, 최대손 {avg_max_loss:.2f}%")

# 더 높은 TP 테스트
print("\n[TP 5% 목표 - 압도적 승리]")
for name, cond in combos_extreme[:4]:
    subset = df_results[cond]
    if len(subset) >= 5:
        wins, losses, total, wr = calc_win_rate(subset, 5.0, -2.0)
        tp5_reach = (subset['max_profit'] >= 5).sum() / len(subset) * 100
        avg_max = subset['max_profit'].mean()
        print(f"{name}: {total}건, TP5% 도달 {tp5_reach:.1f}%, 평균최대익 +{avg_max:.2f}%")

# TP 7%, 10% 테스트
print("\n[TP 7%, 10% 목표 - 극한 승리]")
best_cond = (df_results['rise_from_low'] >= 4) & (df_results['dist_from_fvg'] >= 1)
best_subset = df_results[best_cond]
if len(best_subset) >= 5:
    for tp in [5, 7, 10]:
        wins, losses, total, wr = calc_win_rate(best_subset, tp, -2.0)
        reach = (best_subset['max_profit'] >= tp).sum() / len(best_subset) * 100
        pnl = wins * tp + losses * (-2)
        print(f"  TP {tp}%: {total}건, 도달률 {reach:.1f}%, 승률 {wr:.1f}%, 5년PnL {pnl:.0f}%")

print("\n" + "="*80)
print("📊 전략 B: 애매하게 졸라 많이 이기기 (낮은 TP, 높은 빈도)")
print("="*80)

# 느슨한 조건
conditions_loose = [
    ("저점 대비 0.5%+ 상승", df_results['rise_from_low'] >= 0.5),
    ("저점 대비 1%+ 상승", df_results['rise_from_low'] >= 1),
    ("FVG 위 0.2%+", df_results['dist_from_fvg'] >= 0.2),
    ("FVG 위 0.3%+", df_results['dist_from_fvg'] >= 0.3),
    ("4H 몸통 0.5%+", df_results['body_size'] >= 0.5),
    ("4H 몸통 1%+", df_results['body_size'] >= 1),
]

print("\n[개별 느슨 조건] (TP 1%, SL -1%)")
for name, cond in conditions_loose:
    subset = df_results[cond]
    if len(subset) >= 20:
        wins, losses, total, wr = calc_win_rate(subset, 1.0, -1.0)
        monthly = total / 60
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%")

# 느슨한 조합
print("\n[다빈도 조합] (TP 1%, SL -1%)")
combos_loose = [
    ("저점 0.5%+ & FVG위 0.2%+", 
     (df_results['rise_from_low'] >= 0.5) & (df_results['dist_from_fvg'] >= 0.2)),
    ("저점 1%+ & FVG위 0.2%+", 
     (df_results['rise_from_low'] >= 1) & (df_results['dist_from_fvg'] >= 0.2)),
    ("저점 0.5%+ & FVG위 0.3%+", 
     (df_results['rise_from_low'] >= 0.5) & (df_results['dist_from_fvg'] >= 0.3)),
    ("저점 1%+ & FVG위 0.3%+", 
     (df_results['rise_from_low'] >= 1) & (df_results['dist_from_fvg'] >= 0.3)),
    ("4H몸통 1%+ & FVG위 0.2%+", 
     (df_results['body_size'] >= 1) & (df_results['dist_from_fvg'] >= 0.2)),
    ("기본 (조건없음)", pd.Series([True]*len(df_results))),
]

for name, cond in combos_loose:
    subset = df_results[cond]
    if len(subset) >= 20:
        wins, losses, total, wr = calc_win_rate(subset, 1.0, -1.0)
        monthly = total / 60
        total_pnl = wins * 1 + losses * (-1)
        print(f"{name}")
        print(f"  → {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 5년 총 PnL: {total_pnl:.0f}%")

# TP 0.5% 테스트 (스캘핑)
print("\n[스캘핑 모드] (TP 0.5%, SL -0.5%)")
for name, cond in combos_loose:
    subset = df_results[cond]
    if len(subset) >= 20:
        wins, losses, total, wr = calc_win_rate(subset, 0.5, -0.5)
        monthly = total / 60
        total_pnl = wins * 0.5 + losses * (-0.5)
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 5년 PnL: {total_pnl:.1f}%")

# TP 0.3% 초스캘핑
print("\n[초스캘핑 모드] (TP 0.3%, SL -0.3%)")
for name, cond in combos_loose[:4]:
    subset = df_results[cond]
    if len(subset) >= 20:
        wins, losses, total, wr = calc_win_rate(subset, 0.3, -0.3)
        monthly = total / 60
        total_pnl = wins * 0.3 + losses * (-0.3)
        print(f"{name}: {total}건 ({monthly:.1f}/월), 승률 {wr:.1f}%, 5년 PnL: {total_pnl:.1f}%")

print("\n" + "="*80)
print("🏆 최종 비교")
print("="*80)

# 전략 A 최고 - 압도적 승리
best_a_cond = (df_results['rise_from_low'] >= 4) & (df_results['dist_from_fvg'] >= 1)
best_a = df_results[best_a_cond]
if len(best_a) > 0:
    wins_a, losses_a, total_a, wr_a = calc_win_rate(best_a, 5.0, -2.0)
    pnl_a = wins_a * 5 + losses_a * (-2)
    print(f"\n🔥 전략 A (압도적 승리): 저점 4%+ & FVG위 1%+")
    print(f"  - TP: 5%, SL: -2%")
    print(f"  - 거래수: {total_a}건 ({total_a/60:.1f}/월)")
    print(f"  - 승률: {wr_a:.1f}%")
    print(f"  - 5년 총 PnL: {pnl_a:.0f}%")
    print(f"  - 평균 최대익: +{best_a['max_profit'].mean():.2f}%")
    print(f"  - 1회 기대수익: {(wr_a/100)*5 - ((100-wr_a)/100)*2:.2f}%")

# 전략 B 최고 - 다빈도
best_b_cond = (df_results['rise_from_low'] >= 0.5) & (df_results['dist_from_fvg'] >= 0.2)
best_b = df_results[best_b_cond]
if len(best_b) > 0:
    wins_b, losses_b, total_b, wr_b = calc_win_rate(best_b, 0.5, -0.5)
    pnl_b = wins_b * 0.5 + losses_b * (-0.5)
    print(f"\n📊 전략 B (다빈도): 저점 0.5%+ & FVG위 0.2%+")
    print(f"  - TP: 0.5%, SL: -0.5%")
    print(f"  - 거래수: {total_b}건 ({total_b/60:.1f}/월)")
    print(f"  - 승률: {wr_b:.1f}%")
    print(f"  - 5년 총 PnL: {pnl_b:.0f}%")
    print(f"  - 1회 기대수익: {(wr_b/100)*0.5 - ((100-wr_b)/100)*0.5:.3f}%")

# 비교 요약
print("\n" + "="*80)
print("📈 기대값 비교 (레버리지 10배 가정)")
print("="*80)
if len(best_a) > 0 and len(best_b) > 0:
    ev_a = (wr_a/100)*5 - ((100-wr_a)/100)*2
    ev_b = (wr_b/100)*0.5 - ((100-wr_b)/100)*0.5
    monthly_a = total_a / 60
    monthly_b = total_b / 60
    monthly_ev_a = ev_a * monthly_a * 10  # 레버리지 10배
    monthly_ev_b = ev_b * monthly_b * 10
    print(f"전략 A: 월 {monthly_a:.1f}회 × {ev_a:.2f}% × 10배 = 월 {monthly_ev_a:.1f}%")
    print(f"전략 B: 월 {monthly_b:.1f}회 × {ev_b:.3f}% × 10배 = 월 {monthly_ev_b:.1f}%")

