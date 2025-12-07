import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("📉 MDD 분석 (TP 1.5%, SL -0.5%)")
print("="*80)

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)

# 거래 결과 시뮬레이션
trades = []

for fvg in fvgs_4h:
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
            entry_time = remaining.iloc[0]['datetime']
            
            tp_pct = 1.5
            sl_pct = -0.5
            
            result = None
            exit_time = None
            
            for j, (bar_idx, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                high_pct = (bar['high'] - entry_price) / entry_price * 100
                low_pct = (bar['low'] - entry_price) / entry_price * 100
                open_pct = (bar['open'] - entry_price) / entry_price * 100
                
                if high_pct >= tp_pct and low_pct <= sl_pct:
                    result = tp_pct if open_pct >= 0 else sl_pct
                    exit_time = bar['datetime']
                    break
                elif high_pct >= tp_pct:
                    result = tp_pct
                    exit_time = bar['datetime']
                    break
                elif low_pct <= sl_pct:
                    result = sl_pct
                    exit_time = bar['datetime']
                    break
            
            if result is not None:
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'pnl': result,
                })
            break

df_trades = pd.DataFrame(trades)
df_trades = df_trades.sort_values('entry_time').reset_index(drop=True)

print(f"\n총 거래: {len(df_trades)}건")
print(f"승: {len(df_trades[df_trades['pnl'] > 0])}건")
print(f"패: {len(df_trades[df_trades['pnl'] < 0])}건")

# 누적 수익률 계산
df_trades['cumsum'] = df_trades['pnl'].cumsum()

# MDD 계산
df_trades['peak'] = df_trades['cumsum'].cummax()
df_trades['drawdown'] = df_trades['cumsum'] - df_trades['peak']
mdd = df_trades['drawdown'].min()

print(f"\n총 수익: {df_trades['cumsum'].iloc[-1]:.1f}%")
print(f"MDD: {mdd:.1f}%")

# 연속 손실 분석
print("\n" + "="*80)
print("📊 연속 손실 분석")
print("="*80)

# 연속 손실 계산
consecutive_losses = 0
max_consecutive_losses = 0
loss_streaks = []

for pnl in df_trades['pnl']:
    if pnl < 0:
        consecutive_losses += 1
    else:
        if consecutive_losses > 0:
            loss_streaks.append(consecutive_losses)
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        consecutive_losses = 0

if consecutive_losses > 0:
    loss_streaks.append(consecutive_losses)
    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

print(f"최대 연속 손실: {max_consecutive_losses}연패")
print(f"최대 연속 손실 금액: {max_consecutive_losses * 0.5:.1f}%")

# 연속 손실 분포
from collections import Counter
streak_dist = Counter(loss_streaks)
print(f"\n연속 손실 분포:")
for streak, count in sorted(streak_dist.items()):
    print(f"  {streak}연패: {count}회")

# 레버리지별 MDD
print("\n" + "="*80)
print("📈 레버리지별 MDD")
print("="*80)

for lev in [1, 3, 5, 10, 20]:
    df_trades['cumsum_lev'] = df_trades['pnl'].cumsum() * lev
    df_trades['peak_lev'] = df_trades['cumsum_lev'].cummax()
    df_trades['dd_lev'] = df_trades['cumsum_lev'] - df_trades['peak_lev']
    mdd_lev = df_trades['dd_lev'].min()
    total_lev = df_trades['cumsum_lev'].iloc[-1]
    print(f"레버리지 {lev:2d}x: 총수익 {total_lev:6.0f}%, MDD {mdd_lev:6.1f}%")

# MDD 발생 시점
print("\n" + "="*80)
print("📅 MDD 발생 구간")
print("="*80)

mdd_idx = df_trades['drawdown'].idxmin()
mdd_row = df_trades.loc[mdd_idx]

# MDD 시작점 (peak) 찾기
peak_before_mdd = df_trades.loc[:mdd_idx]
peak_idx = peak_before_mdd['cumsum'].idxmax()
peak_row = df_trades.loc[peak_idx]

print(f"MDD 시작: {peak_row['entry_time']} (누적: {peak_row['cumsum']:.1f}%)")
print(f"MDD 최저: {mdd_row['entry_time']} (누적: {mdd_row['cumsum']:.1f}%)")
print(f"MDD 크기: {mdd:.1f}%")
print(f"MDD 구간 거래수: {mdd_idx - peak_idx}건")

# 월별 수익률
print("\n" + "="*80)
print("📆 월별 수익률")
print("="*80)

df_trades['month'] = df_trades['entry_time'].dt.to_period('M')
monthly = df_trades.groupby('month')['pnl'].agg(['sum', 'count'])
monthly.columns = ['수익률', '거래수']

print(f"\n월평균 수익률: {monthly['수익률'].mean():.2f}%")
print(f"월평균 거래수: {monthly['거래수'].mean():.1f}건")
print(f"손실 월: {len(monthly[monthly['수익률'] < 0])}개월 / {len(monthly)}개월")

# 최악의 월
worst_month = monthly['수익률'].idxmin()
print(f"최악의 월: {worst_month} ({monthly.loc[worst_month, '수익률']:.1f}%)")

