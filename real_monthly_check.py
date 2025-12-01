import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("📊 실제 월별 수익 검증 (전략1: TP 1.0%, SL -1.5%)")
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

# 거래 시뮬레이션 (시간순)
trades = []
tp, sl = 1.0, -1.5
cost = 0.08  # 슬리피지

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
            
            entry_time = remaining.iloc[0]['datetime']
            entry_price = remaining.iloc[0]['open']
            
            result = None
            pnl = 0
            
            for j, (_, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                high_pct = (bar['high'] - entry_price) / entry_price * 100
                low_pct = (bar['low'] - entry_price) / entry_price * 100
                open_pct = (bar['open'] - entry_price) / entry_price * 100
                
                if high_pct >= tp and low_pct <= sl:
                    if open_pct >= 0:
                        pnl = tp - cost
                    else:
                        pnl = sl - cost
                    break
                elif high_pct >= tp:
                    pnl = tp - cost
                    break
                elif low_pct <= sl:
                    pnl = sl - cost
                    break
            
            if pnl != 0:
                trades.append({
                    'time': entry_time,
                    'pnl': pnl,
                })
            break

df_trades = pd.DataFrame(trades)
df_trades['month'] = df_trades['time'].dt.to_period('M')

print(f"\n총 거래: {len(df_trades)}건")
print(f"총 기간: {df_trades['time'].min()} ~ {df_trades['time'].max()}")

# 월별 집계
monthly = df_trades.groupby('month').agg({
    'pnl': ['count', 'sum', lambda x: (x > 0).sum()]
}).round(2)
monthly.columns = ['거래수', '수익률', '승수']
monthly['승률'] = (monthly['승수'] / monthly['거래수'] * 100).round(1)

print("\n" + "="*80)
print("📅 월별 실제 수익 (슬리피지 0.08% 적용)")
print("="*80)

print(f"\n{'월':<10} {'거래':>6} {'승':>4} {'승률':>8} {'수익률':>10} {'10x레버':>10}")
print("-"*55)

for month, row in monthly.iterrows():
    lev_return = row['수익률'] * 10
    print(f"{str(month):<10} {int(row['거래수']):>6} {int(row['승수']):>4} {row['승률']:>7.1f}% {row['수익률']:>9.2f}% {lev_return:>9.1f}%")

# 요약 통계
print("\n" + "="*80)
print("📈 요약 통계")
print("="*80)

total_months = len(monthly)
profit_months = len(monthly[monthly['수익률'] > 0])
loss_months = len(monthly[monthly['수익률'] < 0])

print(f"\n총 월수: {total_months}개월")
print(f"수익 월: {profit_months}개월 ({profit_months/total_months*100:.1f}%)")
print(f"손실 월: {loss_months}개월 ({loss_months/total_months*100:.1f}%)")

print(f"\n월평균 거래: {monthly['거래수'].mean():.1f}건")
print(f"월평균 수익: {monthly['수익률'].mean():.2f}%")
print(f"월평균 수익 (10x): {monthly['수익률'].mean() * 10:.1f}%")

print(f"\n최고의 월: {monthly['수익률'].max():.2f}% (10x: {monthly['수익률'].max()*10:.1f}%)")
print(f"최악의 월: {monthly['수익률'].min():.2f}% (10x: {monthly['수익률'].min()*10:.1f}%)")

# 연도별
print("\n" + "="*80)
print("📆 연도별 수익")
print("="*80)

df_trades['year'] = df_trades['time'].dt.year
yearly = df_trades.groupby('year').agg({
    'pnl': ['count', 'sum', lambda x: (x > 0).sum()]
})
yearly.columns = ['거래수', '수익률', '승수']
yearly['승률'] = (yearly['승수'] / yearly['거래수'] * 100).round(1)

print(f"\n{'연도':<6} {'거래':>6} {'승률':>8} {'수익률':>10} {'10x':>10}")
print("-"*45)

for year, row in yearly.iterrows():
    print(f"{year:<6} {int(row['거래수']):>6} {row['승률']:>7.1f}% {row['수익률']:>9.1f}% {row['수익률']*10:>9.0f}%")

# MDD 계산
print("\n" + "="*80)
print("📉 MDD 분석")
print("="*80)

df_trades_sorted = df_trades.sort_values('time')
df_trades_sorted['cumsum'] = df_trades_sorted['pnl'].cumsum()
df_trades_sorted['peak'] = df_trades_sorted['cumsum'].cummax()
df_trades_sorted['dd'] = df_trades_sorted['cumsum'] - df_trades_sorted['peak']

mdd = df_trades_sorted['dd'].min()
print(f"\nMDD (1x): {mdd:.2f}%")
print(f"MDD (10x): {mdd*10:.1f}%")

# 연속 손실
consecutive = 0
max_consecutive = 0
for pnl in df_trades_sorted['pnl']:
    if pnl < 0:
        consecutive += 1
        max_consecutive = max(max_consecutive, consecutive)
    else:
        consecutive = 0

print(f"최대 연패: {max_consecutive}연패")
print(f"연패 손실 (1x): {max_consecutive * (abs(sl) + cost):.2f}%")
print(f"연패 손실 (10x): {max_consecutive * (abs(sl) + cost) * 10:.1f}%")

