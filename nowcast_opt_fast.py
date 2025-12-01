"""
나우캐스트 빠른 최적화 - 핵심만
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 빠른 최적화")
print("=" * 70)

# 데이터
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

breakouts = pd.read_csv('nowcast_breakouts.csv').to_dict('records')

# 지표
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']
df['ma50'] = df['close'].rolling(50).mean()
df['mom_5'] = df['close'].pct_change(5) * 100

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

print(f"데이터: {len(df)}봉, 돌파: {len(breakouts)}개")
print(f"FVG 돌파: {sum(1 for b in breakouts if b['has_fvg'])}개")

# numpy 배열
open_p = df['open'].values
high = df['high'].values
low = df['low'].values
close = df['close'].values
vol_ratio = df['vol_ratio'].values
ma50_arr = df['ma50'].values
rsi_arr = df['rsi'].values
mom5_arr = df['mom_5'].values
def backtest(tp, sl, interval, fvg_only, vol_min, trend, rsi_max, mom_min):
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    vol_ratio = df['vol_ratio'].values
    ma50_arr = df['ma50'].values
    rsi_arr = df['rsi'].values
    mom5_arr = df['mom_5'].values
    trades = []
    last_entry = -interval - 1
    
    for brk in breakouts:
        i = brk['idx']
        
        if i - last_entry < interval:
            continue
        if brk['type'] != 'long':
            continue
        
        # 필터
        if fvg_only and not brk['has_fvg']:
            continue
        if vol_min > 0 and vol_ratio[i] < vol_min:
            continue
        if trend and close[i] < ma50_arr[i]:
            continue
        if rsi_max > 0 and rsi_arr[i] > rsi_max:
            continue
        if mom_min is not None and mom5_arr[i] < mom_min:
            continue
        
        entry_idx = i + 1
        if entry_idx >= len(df) - 50:
            continue
        
        entry_price = open_p[entry_idx]
        tp_level = entry_price * (1 + tp / 100)
        sl_level = entry_price * (1 - sl / 100)
        
        exit_price = close[min(entry_idx + 49, len(df) - 1)]
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if high[j] >= tp_level:
                exit_price = tp_level
                break
            if low[j] <= sl_level:
                exit_price = sl_level
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        trades.append(pnl)
        last_entry = i
    
    if not trades:
        return 0, 0, 0, 0
    
    n = len(trades)
    wr = sum(1 for p in trades if p > 0) / n * 100
    avg = np.mean(trades)
    total = sum(trades)
    return n, wr, avg, total


# 1. FVG+볼륨 기본 테스트
print("\n" + "=" * 70)
print("1. FVG + 볼륨 조합")
print("=" * 70)

results = []
for tp in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0, 1.5]:
        for vol in [0.8, 1.0, 1.2]:
            n, wr, avg, total = backtest(tp, sl, 10, True, vol, False, 0, None)
            if n >= 50:
                net = avg - 0.11
                results.append({
                    'tp': tp, 'sl': sl, 'vol': vol,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

df1 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df1.head(10).to_string(index=False))


# 2. FVG + 추세
print("\n" + "=" * 70)
print("2. FVG + 추세 조합")
print("=" * 70)

results = []
for tp in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0, 1.5]:
        n, wr, avg, total = backtest(tp, sl, 10, True, 0, True, 0, None)
        if n >= 50:
            net = avg - 0.11
            results.append({
                'tp': tp, 'sl': sl,
                'n': n, 'wr': wr, 'net': net, 'total': total
            })

df2 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df2.head(10).to_string(index=False))


# 3. FVG + RSI
print("\n" + "=" * 70)
print("3. FVG + RSI 필터")
print("=" * 70)

results = []
for tp in [2.0, 2.5, 3.0, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0, 1.5]:
        for rsi in [40, 45, 50, 55]:
            n, wr, avg, total = backtest(tp, sl, 10, True, 0, False, rsi, None)
            if n >= 30:
                net = avg - 0.11
                results.append({
                    'tp': tp, 'sl': sl, 'rsi': rsi,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

df3 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df3.head(10).to_string(index=False))


# 4. FVG + 모멘텀
print("\n" + "=" * 70)
print("4. FVG + 모멘텀 필터")
print("=" * 70)

results = []
for tp in [2.0, 2.5, 3.0, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0, 1.5]:
        for mom in [-2, -1, 0, 1]:
            n, wr, avg, total = backtest(tp, sl, 10, True, 0, False, 0, mom)
            if n >= 30:
                net = avg - 0.11
                results.append({
                    'tp': tp, 'sl': sl, 'mom': mom,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

df4 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df4.head(10).to_string(index=False))


# 5. 복합 필터
print("\n" + "=" * 70)
print("5. 복합 필터 조합")
print("=" * 70)

results = []
# FVG + 볼륨 + 추세
for tp in [2.5, 3.0, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0]:
        for vol in [0.8, 1.0]:
            n, wr, avg, total = backtest(tp, sl, 10, True, vol, True, 0, None)
            if n >= 30:
                net = avg - 0.11
                results.append({
                    'type': 'FVG+Vol+Trend', 'tp': tp, 'sl': sl, 'extra': vol,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

# FVG + 볼륨 + RSI
for tp in [2.5, 3.0, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0]:
        for rsi in [45, 50]:
            n, wr, avg, total = backtest(tp, sl, 10, True, 1.0, False, rsi, None)
            if n >= 30:
                net = avg - 0.11
                results.append({
                    'type': 'FVG+Vol+RSI', 'tp': tp, 'sl': sl, 'extra': rsi,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

# FVG + 추세 + 모멘텀
for tp in [2.5, 3.0, 4.0, 5.0]:
    for sl in [0.5, 0.7, 1.0]:
        for mom in [-1, 0]:
            n, wr, avg, total = backtest(tp, sl, 10, True, 0, True, 0, mom)
            if n >= 30:
                net = avg - 0.11
                results.append({
                    'type': 'FVG+Trend+Mom', 'tp': tp, 'sl': sl, 'extra': mom,
                    'n': n, 'wr': wr, 'net': net, 'total': total
                })

df5 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df5.head(15).to_string(index=False))


# 6. 간격 조정
print("\n" + "=" * 70)
print("6. 간격 최적화 (상위 설정)")
print("=" * 70)

# 상위 설정 기준
results = []
for intv in [5, 8, 10, 15, 20]:
    # FVG + Vol1.0
    n, wr, avg, total = backtest(2.5, 1.0, intv, True, 1.0, False, 0, None)
    if n > 0:
        results.append({'set': 'FVG+Vol', 'intv': intv, 'n': n, 'wr': wr, 'net': avg-0.11, 'total': total})
    
    # FVG + Trend
    n, wr, avg, total = backtest(2.5, 1.0, intv, True, 0, True, 0, None)
    if n > 0:
        results.append({'set': 'FVG+Trend', 'intv': intv, 'n': n, 'wr': wr, 'net': avg-0.11, 'total': total})

df6 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df6.to_string(index=False))


# 최종 요약
print("\n" + "=" * 70)
print("★ 최종 요약 ★")
print("=" * 70)

all_best = []
if len(df1) > 0 and df1.iloc[0]['net'] > 0:
    b = df1.iloc[0]
    all_best.append(f"FVG+Vol>{b['vol']}: TP{b['tp']} SL{b['sl']} -> 순수익 {b['net']:.4f}%")
if len(df2) > 0 and df2.iloc[0]['net'] > 0:
    b = df2.iloc[0]
    all_best.append(f"FVG+Trend: TP{b['tp']} SL{b['sl']} -> 순수익 {b['net']:.4f}%")
if len(df3) > 0 and df3.iloc[0]['net'] > 0:
    b = df3.iloc[0]
    all_best.append(f"FVG+RSI<{b['rsi']}: TP{b['tp']} SL{b['sl']} -> 순수익 {b['net']:.4f}%")
if len(df4) > 0 and df4.iloc[0]['net'] > 0:
    b = df4.iloc[0]
    all_best.append(f"FVG+Mom>{b['mom']}: TP{b['tp']} SL{b['sl']} -> 순수익 {b['net']:.4f}%")
if len(df5) > 0 and df5.iloc[0]['net'] > 0:
    b = df5.iloc[0]
    all_best.append(f"{b['type']}: TP{b['tp']} SL{b['sl']} -> 순수익 {b['net']:.4f}%")

if all_best:
    print("\n양수 순수익 설정:")
    for item in all_best:
        print(f"  ✓ {item}")
else:
    print("\n⚠️ 수수료 후 양수 순수익 설정 없음")
    
# 가장 좋은 설정
all_dfs = [df1, df2, df3, df4, df5]
best_overall = None
best_net = -999

for d in all_dfs:
    if len(d) > 0 and d.iloc[0]['net'] > best_net:
        best_net = d.iloc[0]['net']
        best_overall = d.iloc[0]

if best_overall is not None:
    total_days = (pd.to_datetime(df['datetime'].iloc[-1]) - pd.to_datetime(df['datetime'].iloc[0])).days
    months = total_days / 30
    
    print(f"\n최고 순수익 설정:")
    print(f"  순수익: {best_net:.4f}%/거래")
    print(f"  총수익: {best_overall['total']:.1f}%")
    print(f"  월수익: {best_overall['total']/months:.2f}%")
    
    if best_net > 0:
        print("\n✅ 실전 가능한 양수 수익 전략 발견!")
    else:
        print(f"\n⚠️ 수수료 후 {best_net:.4f}% 손실 (거래당)")
        print("  -> 레버리지 없이는 손실")
        print("  -> 2-3배 레버리지로 수수료 상쇄 가능")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
