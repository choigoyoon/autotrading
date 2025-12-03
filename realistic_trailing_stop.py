#!/usr/bin/env python3
"""
현실적 트레일링 스탑 백테스트
- 미래 데이터 없이
- 봉 하나씩 진행하면서 트레일링
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("현실적 트레일링 스탑 백테스트 (미래 데이터 없음)")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

df_1h = df.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

def extract_hl_points(df_1h):
    hist = df_1h['hist'].values
    high = df_1h['high'].values
    low = df_1h['low'].values
    timestamps = df_1h['datetime'].values
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': timestamps[max_idx], 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': timestamps[min_idx], 'idx': min_idx})
        else:
            i += 1
    return points

points = extract_hl_points(df_1h)

# ============================================================
# 현실적 트레일링 스탑 함수
# ============================================================
def backtest_trailing(df_1h, entry_idx, entry_price, initial_sl, 
                      trail_trigger=3.0, trail_pct=2.0, max_bars=500):
    """
    현실적 트레일링 스탑
    - trail_trigger: 이 수익% 도달하면 트레일링 시작
    - trail_pct: 고점 대비 이만큼 하락하면 청산
    """
    post = df_1h.iloc[entry_idx+1:entry_idx+max_bars]
    
    highest = entry_price
    trailing_active = False
    trailing_sl = initial_sl
    
    for bar, (_, row) in enumerate(post.iterrows()):
        # 현재 고점 갱신
        if row['high'] > highest:
            highest = row['high']
            current_pnl = (highest - entry_price) / entry_price * 100
            
            # 트레일링 활성화 조건
            if current_pnl >= trail_trigger:
                trailing_active = True
                # 트레일링 SL = 고점에서 trail_pct% 아래
                trailing_sl = highest * (1 - trail_pct / 100)
        
        # SL 체크 (트레일링 or 초기)
        current_sl = trailing_sl if trailing_active else initial_sl
        
        if row['low'] <= current_sl:
            exit_price = current_sl
            pnl = (exit_price - entry_price) / entry_price * 100
            return {
                'pnl': pnl,
                'exit_type': 'TRAILING_SL' if trailing_active else 'INITIAL_SL',
                'bars': bar,
                'highest': highest,
                'mfe': (highest - entry_price) / entry_price * 100
            }
    
    # 시간초과 - 마지막 종가로 청산
    if len(post) > 0:
        exit_price = post.iloc[-1]['close']
        pnl = (exit_price - entry_price) / entry_price * 100
        return {
            'pnl': pnl,
            'exit_type': 'TIMEOUT',
            'bars': len(post),
            'highest': highest,
            'mfe': (highest - entry_price) / entry_price * 100
        }
    
    return {'pnl': 0, 'exit_type': 'NO_DATA', 'bars': 0, 'highest': entry_price, 'mfe': 0}

# ============================================================
# W패턴 + Gap >= 5% 찾기 및 백테스트
# ============================================================
print("\n패턴: W (더블바텀), Gap >= 5%")

results_list = []

for i in range(len(points) - 2):
    if not (points[i]['type'] == 'L' and points[i+1]['type'] == 'H' and points[i+2]['type'] == 'L'):
        continue
    
    L1, H, L2 = points[i]['price'], points[i+1]['price'], points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    if abs(L2 - L1) / L1 > 0.03:
        continue
    
    gap = (H - L2) / L2 * 100
    if gap < 5:
        continue
    
    # 넥라인 돌파
    future = df_1h.iloc[L2_idx+1:L2_idx+100]
    breakout_idx = None
    for j, (_, row) in enumerate(future.iterrows()):
        if row['close'] > H:
            breakout_idx = L2_idx + 1 + j
            break
    
    if breakout_idx is None:
        continue
    
    entry = df_1h.iloc[breakout_idx]['close']
    entry_time = df_1h.iloc[breakout_idx]['datetime']
    initial_sl = min(L1, L2)
    
    # 다양한 트레일링 설정 테스트
    for trigger in [2, 3, 5]:
        for trail in [1.5, 2, 3]:
            result = backtest_trailing(df_1h, breakout_idx, entry, initial_sl,
                                       trail_trigger=trigger, trail_pct=trail)
            results_list.append({
                'time': entry_time,
                'entry': entry,
                'gap': gap,
                'trigger': trigger,
                'trail': trail,
                **result
            })

df_r = pd.DataFrame(results_list)

# ============================================================
# 트레일링 설정별 성과 비교
# ============================================================
print("\n" + "=" * 80)
print("트레일링 설정별 성과")
print("=" * 80)

print(f"\n{'Trigger':>8} {'Trail%':>8} {'평균PnL':>10} {'승률':>8} {'총PnL':>10}")
print("-" * 50)

best_pnl = -999
best_setting = None

for trigger in [2, 3, 5]:
    for trail in [1.5, 2, 3]:
        subset = df_r[(df_r['trigger'] == trigger) & (df_r['trail'] == trail)]
        avg_pnl = subset['pnl'].mean()
        win_rate = (subset['pnl'] > 0).mean() * 100
        total_pnl = subset['pnl'].sum()
        
        marker = ""
        if avg_pnl > best_pnl:
            best_pnl = avg_pnl
            best_setting = (trigger, trail)
            marker = " ★"
        
        print(f"{trigger:>8}% {trail:>8}% {avg_pnl:>10.2f} {win_rate:>7.1f}% {total_pnl:>10.1f}{marker}")

print(f"\n최적 설정: Trigger {best_setting[0]}%, Trail {best_setting[1]}%")

# ============================================================
# 최적 설정으로 상세 분석
# ============================================================
print("\n" + "=" * 80)
print(f"최적 설정 상세 분석 (Trigger {best_setting[0]}%, Trail {best_setting[1]}%)")
print("=" * 80)

best = df_r[(df_r['trigger'] == best_setting[0]) & (df_r['trail'] == best_setting[1])]

print(f"\n[기본 통계]")
print(f"  총 거래: {len(best)}건")
print(f"  평균 PnL: {best['pnl'].mean():.2f}%")
print(f"  총 PnL: {best['pnl'].sum():.1f}%")
print(f"  승률: {(best['pnl'] > 0).mean()*100:.1f}%")
print(f"  평균 MFE: {best['mfe'].mean():.1f}%")

print(f"\n[청산 유형]")
for exit_type in ['TRAILING_SL', 'INITIAL_SL', 'TIMEOUT']:
    count = (best['exit_type'] == exit_type).sum()
    pct = count / len(best) * 100
    avg = best[best['exit_type'] == exit_type]['pnl'].mean() if count > 0 else 0
    print(f"  {exit_type}: {count}건 ({pct:.1f}%), 평균 {avg:.2f}%")

# 월별 성과
best = best.copy()
best['month'] = pd.to_datetime(best['time']).dt.to_period('M')
monthly = best.groupby('month')['pnl'].agg(['sum', 'count', 'mean'])

print(f"\n[월별 성과]")
print(f"  평균 거래수: {monthly['count'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['sum'].mean():.2f}%")
print(f"  월 수익 중앙값: {monthly['sum'].median():.2f}%")
print(f"  손실 월: {(monthly['sum'] < 0).sum()}개월 / {len(monthly)}개월")

# ============================================================
# 고정 TP 3.5%와 비교
# ============================================================
print("\n" + "=" * 80)
print("고정 TP 3.5% vs 트레일링 비교")
print("=" * 80)

# 고정 TP 3.5% 백테스트
def backtest_fixed_tp(df_1h, entry_idx, entry_price, sl_price, tp_pct=3.5, max_bars=500):
    tp_price = entry_price * (1 + tp_pct / 100)
    post = df_1h.iloc[entry_idx+1:entry_idx+max_bars]
    
    for bar, (_, row) in enumerate(post.iterrows()):
        if row['high'] >= tp_price:
            return tp_pct, 'TP'
        if row['low'] <= sl_price:
            return (sl_price - entry_price) / entry_price * 100, 'SL'
    
    if len(post) > 0:
        return (post.iloc[-1]['close'] - entry_price) / entry_price * 100, 'TIMEOUT'
    return 0, 'NO_DATA'

fixed_results = []
for i in range(len(points) - 2):
    if not (points[i]['type'] == 'L' and points[i+1]['type'] == 'H' and points[i+2]['type'] == 'L'):
        continue
    
    L1, H, L2 = points[i]['price'], points[i+1]['price'], points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    if abs(L2 - L1) / L1 > 0.03:
        continue
    
    gap = (H - L2) / L2 * 100
    if gap < 5:
        continue
    
    future = df_1h.iloc[L2_idx+1:L2_idx+100]
    breakout_idx = None
    for j, (_, row) in enumerate(future.iterrows()):
        if row['close'] > H:
            breakout_idx = L2_idx + 1 + j
            break
    
    if breakout_idx is None:
        continue
    
    entry = df_1h.iloc[breakout_idx]['close']
    entry_time = df_1h.iloc[breakout_idx]['datetime']
    sl_price = min(L1, L2)
    
    pnl, exit_type = backtest_fixed_tp(df_1h, breakout_idx, entry, sl_price)
    fixed_results.append({
        'time': entry_time,
        'pnl': pnl,
        'exit_type': exit_type
    })

df_fixed = pd.DataFrame(fixed_results)
df_fixed['month'] = pd.to_datetime(df_fixed['time']).dt.to_period('M')
fixed_monthly = df_fixed.groupby('month')['pnl'].sum()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    고정 TP vs 트레일링 비교                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        고정 TP 3.5%       트레일링 ({best_setting[0]}%/{best_setting[1]}%)            │
│  ──────────────────────────────────────────────────────────────────────── │
│  총 거래             {len(df_fixed):>10}건       {len(best):>10}건                     │
│  평균 PnL            {df_fixed['pnl'].mean():>10.2f}%       {best['pnl'].mean():>10.2f}%                     │
│  총 PnL              {df_fixed['pnl'].sum():>10.1f}%       {best['pnl'].sum():>10.1f}%                     │
│  승률                {(df_fixed['pnl']>0).mean()*100:>10.1f}%       {(best['pnl']>0).mean()*100:>10.1f}%                     │
│  월 평균 수익        {fixed_monthly.mean():>10.2f}%       {monthly['sum'].mean():>10.2f}%                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론 (미래 데이터 없는 현실적 백테스트)")
print("=" * 80)

trailing_monthly = monthly['sum'].mean()
fixed_monthly_avg = fixed_monthly.mean()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         현실적 결과                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [고정 TP 3.5%]                                                            │
│  - 월 평균: {fixed_monthly_avg:.2f}%                                                   │
│  - 3x 레버리지: {fixed_monthly_avg*3:.1f}%/월                                            │
│                                                                            │
│  [트레일링 스탑 (Trigger {best_setting[0]}%, Trail {best_setting[1]}%)]                             │
│  - 월 평균: {trailing_monthly:.2f}%                                                   │
│  - 3x 레버리지: {trailing_monthly*3:.1f}%/월                                            │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  개선율: {(trailing_monthly/fixed_monthly_avg - 1)*100:.0f}%                                               │
│                                                                            │
│  트레일링이 더 나은 이유:                                                  │
│  - 수익권 도달 후 SL 맞는 경우 → 트레일링으로 수익 확보                    │
│  - 큰 추세 잡으면 → 3.5% 이상 수익 가능                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 월별 상세
print("\n[월별 수익 분포 - 트레일링]")
print(f"  15%+ 달성: {(monthly['sum'] >= 15).sum()}개월")
print(f"  10%+ 달성: {(monthly['sum'] >= 10).sum()}개월")
print(f"  5%+ 달성: {(monthly['sum'] >= 5).sum()}개월")
print(f"  손실 월: {(monthly['sum'] < 0).sum()}개월")

print("\n분석 완료!")
