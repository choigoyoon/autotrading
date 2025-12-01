import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_1h = df_15m.groupby(df_15m['datetime'].str[:13]).agg({
    'datetime': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).reset_index(drop=True)

df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])

# ===== 다양한 전략 시그널 감지 =====

# 1. 상승 FVG (4H)
def detect_bull_fvg_4h(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['high'] < df.iloc[i]['low']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['low'], 'type': 'fvg_bull_4h'})
    return signals

# 2. 하락 FVG (4H) - 숏
def detect_bear_fvg_4h(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['high'], 'type': 'fvg_bear_4h', 'dir': 'short'})
    return signals

# 3. 상승 FVG (1H) - 더 빈번한 시그널
def detect_bull_fvg_1h(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['high'] < df.iloc[i]['low']:
            gap = (df.iloc[i]['low'] - df.iloc[i-2]['high']) / df.iloc[i-2]['high'] * 100
            if gap > 0.3:  # 최소 0.3% 갭
                signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['low'], 'type': 'fvg_bull_1h'})
    return signals

# 4. 오더블럭 (4H)
def detect_orderblock_4h(df):
    signals = []
    for i in range(3, len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['open']:  # 양봉
            if df.iloc[i-1]['close'] < df.iloc[i-1]['open']:  # 직전 음봉
                move = (df.iloc[i]['close'] - df.iloc[i-1]['low']) / df.iloc[i-1]['low'] * 100
                if move > 2:
                    signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i-1]['open'], 'type': 'ob_bull_4h'})
    return signals

# 5. 브레이커 블럭 (4H)
def detect_breaker_4h(df):
    signals = []
    for i in range(15, len(df)):
        window = df.iloc[i-15:i]
        highs_idx = argrelextrema(window['high'].values, np.greater, order=4)[0]
        if len(highs_idx) >= 1:
            high_price = window.iloc[highs_idx[-1]]['high']
            if df.iloc[i]['close'] > high_price * 1.005:  # 0.5% 돌파
                signals.append({'time': df.iloc[i]['datetime'], 'level': high_price, 'type': 'breaker_4h'})
    return signals

# 6. 골든크로스 구간 되돌림 (4H)
def detect_golden_pullback(df):
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    signals = []
    for i in range(52, len(df)):
        # 골든크로스 상태 (20>50)
        if df.iloc[i]['ema20'] > df.iloc[i]['ema50']:
            # EMA20으로 되돌림
            if df.iloc[i]['low'] <= df.iloc[i]['ema20'] * 1.005:
                signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['ema20'], 'type': 'golden_pb'})
    return signals

# 7. 지지선 리테스트 (4H)
def detect_support_retest(df):
    signals = []
    for i in range(30, len(df)):
        window = df.iloc[i-30:i-5]
        lows_idx = argrelextrema(window['low'].values, np.less, order=5)[0]
        
        if len(lows_idx) >= 2:
            support = window.iloc[lows_idx[-1]]['low']
            # 현재가가 지지선 근처
            if abs(df.iloc[i]['low'] - support) / support < 0.01:
                signals.append({'time': df.iloc[i]['datetime'], 'level': support, 'type': 'support_retest'})
    return signals

# 시뮬레이션
def simulate(signals, df_15m, tp1, tp2, sl, time_stop, direction='long', max_bars=50):
    trades = []
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    for sig in signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        # 터치 대기
        touch_idx = None
        for i in range(start+1, min(start+max_bars, len(times))):
            if direction == 'long' and lows[i] <= sig['level']:
                touch_idx = i
                break
            elif direction == 'short' and highs[i] >= sig['level']:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        
        if direction == 'long':
            tp1_p, tp2_p, sl_p = ep*(1+tp1/100), ep*(1+tp2/100), ep*(1+sl/100)
        else:
            tp1_p, tp2_p, sl_p = ep*(1-tp1/100), ep*(1-tp2/100), ep*(1-sl/100)
        
        result, pnl, tp1_hit = None, 0, False
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                if direction == 'long':
                    pnl = (closes[i] - ep) / ep * 100 - 0.06
                else:
                    pnl = (ep - closes[i]) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            if not tp1_hit:
                if direction == 'long':
                    if lows[i] <= sl_p:
                        result, pnl = 'SL', sl - 0.06
                        break
                    if highs[i] >= tp1_p:
                        tp1_hit = True
                        continue
                else:
                    if highs[i] >= sl_p:
                        result, pnl = 'SL', sl - 0.06
                        break
                    if lows[i] <= tp1_p:
                        tp1_hit = True
                        continue
            else:
                if direction == 'long':
                    if lows[i] <= ep:
                        result, pnl = 'BE', -0.06
                        break
                    if highs[i] >= tp2_p:
                        result, pnl = 'TP2', tp2 - 0.06
                        break
                else:
                    if highs[i] >= ep:
                        result, pnl = 'BE', -0.06
                        break
                    if lows[i] <= tp2_p:
                        result, pnl = 'TP2', tp2 - 0.06
                        break
        
        if result is None and tp1_hit:
            result, pnl = 'BE', -0.06
        
        if result:
            trades.append({'time': times[entry_idx], 'result': result, 'pnl': pnl, 'type': sig['type']})
    
    return trades

print("=" * 95)
print("🎯 다중 전략 시스템 V2 - 7개 전략 병렬 운용")
print("=" * 95)

# 시그널 감지
strategies_config = [
    ("FVG 상승 4H", detect_bull_fvg_4h(df_4h), 'long', {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96}),
    ("FVG 하락 4H", detect_bear_fvg_4h(df_4h), 'short', {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96}),
    ("FVG 상승 1H", detect_bull_fvg_1h(df_1h), 'long', {'tp1': 1.0, 'tp2': 2.0, 'sl': -1.0, 'time_stop': 48}),
    ("오더블럭 4H", detect_orderblock_4h(df_4h), 'long', {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96}),
    ("브레이커 4H", detect_breaker_4h(df_4h), 'long', {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96}),
    ("골든크로스 되돌림", detect_golden_pullback(df_4h), 'long', {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96}),
    ("지지선 리테스트", detect_support_retest(df_4h), 'long', {'tp1': 1.5, 'tp2': 2.5, 'sl': -1.0, 'time_stop': 72}),
]

print(f"\n[전략별 시그널 수]")
for name, signals, _, _ in strategies_config:
    print(f"  {name}: {len(signals)}개")

print("\n" + "=" * 95)
print("📊 전략별 개별 성과")
print("=" * 95)

print(f"\n{'전략':<20} {'시그널':>7} {'거래':>6} {'SL%':>7} {'승률':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 75)

all_trades = []
strategy_stats = []

for name, signals, direction, params in strategies_config:
    trades = simulate(signals, df_15m, direction=direction, **params)
    
    if len(trades) >= 10:
        df_t = pd.DataFrame(trades)
        sl = (df_t['result'] == 'SL').sum()
        months = len(pd.to_datetime(df_t['time']).dt.to_period('M').unique())
        pnl = df_t['pnl'].sum()
        
        print(f"{name:<20} {len(signals):>7} {len(trades):>6} {sl/len(trades)*100:>6.1f}% {(len(trades)-sl)/len(trades)*100:>6.1f}% {pnl:>8.1f}% {pnl/months:>7.2f}%")
        
        all_trades.extend(trades)
        strategy_stats.append({'name': name, 'trades': len(trades), 'pnl': pnl, 'mavg': pnl/months})

# 중복 체크 및 통합
print("\n" + "=" * 95)
print("🔍 중복 거래 제거 후 통합")
print("=" * 95)

df_all = pd.DataFrame(all_trades)
df_all['time'] = pd.to_datetime(df_all['time'])
df_all = df_all.sort_values('time')

# 같은 시간대 거래 중복 체크 (4시간 이내)
df_all['time_group'] = df_all['time'].dt.floor('4h')
before_dedup = len(df_all)

# 중복 제거 (같은 4시간 구간에서 같은 방향은 1개만)
df_dedup = df_all.groupby(['time_group', 'type']).first().reset_index()
after_dedup = len(df_dedup)

print(f"\n  중복 제거 전: {before_dedup}건")
print(f"  중복 제거 후: {after_dedup}건")
print(f"  제거된 중복: {before_dedup - after_dedup}건 ({(before_dedup - after_dedup)/before_dedup*100:.1f}%)")

# 최종 통계
total_sl = (df_dedup['result'] == 'SL').sum()
total_months = len(df_dedup['time'].dt.to_period('M').unique())
total_pnl = df_dedup['pnl'].sum()

print(f"\n[최종 통합 결과]")
print(f"  총 거래: {len(df_dedup)}회")
print(f"  월평균 거래: {len(df_dedup)/total_months:.1f}회")
print(f"  SL 비율: {total_sl/len(df_dedup)*100:.1f}%")
print(f"  승률: {(len(df_dedup)-total_sl)/len(df_dedup)*100:.1f}%")
print(f"  총 PnL: {total_pnl:.1f}%")
print(f"  월평균 PnL: {total_pnl/total_months:.2f}%")

# 전략별 기여도
print(f"\n[전략별 기여도]")
for stype in df_dedup['type'].unique():
    subset = df_dedup[df_dedup['type'] == stype]
    print(f"  {stype}: {len(subset)}회 ({len(subset)/len(df_dedup)*100:.1f}%), PnL {subset['pnl'].sum():.1f}%")

# 연도별
print(f"\n[연도별 성과]")
df_dedup['year'] = df_dedup['time'].dt.year
print(f"{'연도':<6} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 40)

for year in sorted(df_dedup['year'].unique()):
    yearly = df_dedup[df_dedup['year'] == year]
    sl = (yearly['result'] == 'SL').sum()
    months = len(yearly['time'].dt.to_period('M').unique())
    pnl = yearly['pnl'].sum()
    print(f"{year:<6} {len(yearly):>6} {sl/len(yearly)*100:>6.1f}% {pnl:>8.1f}% {pnl/months:>7.2f}%")

# 최종 비교
print("\n" + "=" * 95)
print("📈 최종 비교: 단일 vs 다중 전략")
print("=" * 95)

# 단일 (FVG 4H만)
single = simulate(detect_bull_fvg_4h(df_4h), df_15m, 'long', tp1=2.0, tp2=3.5, sl=-1.5, time_stop=96)
df_single = pd.DataFrame(single)
single_months = len(pd.to_datetime(df_single['time']).dt.to_period('M').unique())
single_pnl = df_single['pnl'].sum()

print(f"\n{'구분':<25} {'거래':>7} {'월거래':>7} {'승률':>7} {'월평균':>9}")
print("-" * 65)
print(f"{'단일 (FVG 4H)':<25} {len(single):>7} {len(single)/single_months:>6.1f} {(df_single['result']!='SL').mean()*100:>6.1f}% {single_pnl/single_months:>8.2f}%")
print(f"{'다중 (7전략)':<25} {len(df_dedup):>7} {len(df_dedup)/total_months:>6.1f} {(len(df_dedup)-total_sl)/len(df_dedup)*100:>6.1f}% {total_pnl/total_months:>8.2f}%")

improvement = total_pnl/total_months - single_pnl/single_months
print(f"\n  → 다중 전략으로 월평균 +{improvement:.2f}% 추가 수익!")
print(f"  → 거래 빈도 {len(df_dedup)/total_months / (len(single)/single_months):.1f}배 증가!")

