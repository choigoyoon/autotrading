import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

def calc_macd(df):
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    return exp1 - exp2

df_4h['macd'] = calc_macd(df_4h)

# ===== 전략 1: 기존 FVG 본절스탑 =====
def detect_bullish_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        p2, curr = df.iloc[i-2], df.iloc[i]
        if p2['high'] < curr['low']:  # 상승 FVG
            fvgs.append({'idx': i, 'time': curr['datetime'], 'top': curr['low'], 'type': 'bull'})
    return fvgs

# ===== 전략 2: 하락 FVG (숏) =====
def detect_bearish_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        p2, curr = df.iloc[i-2], df.iloc[i]
        if p2['low'] > curr['high']:  # 하락 FVG
            fvgs.append({'idx': i, 'time': curr['datetime'], 'bottom': curr['high'], 'type': 'bear'})
    return fvgs

# ===== 전략 3: 오더블럭 =====
def detect_orderblock(df):
    obs = []
    for i in range(3, len(df)):
        # 강한 상승 전 마지막 음봉 = 상승 오더블럭
        if df.iloc[i]['close'] > df.iloc[i]['open']:  # 현재 양봉
            if df.iloc[i-1]['close'] < df.iloc[i-1]['open']:  # 직전 음봉
                # 강한 상승인지 체크 (2% 이상)
                move = (df.iloc[i]['close'] - df.iloc[i-1]['low']) / df.iloc[i-1]['low'] * 100
                if move > 2:
                    obs.append({
                        'idx': i, 'time': df.iloc[i]['datetime'],
                        'top': df.iloc[i-1]['open'],  # 음봉의 시가
                        'bottom': df.iloc[i-1]['low'],  # 음봉의 저가
                        'type': 'bull_ob'
                    })
    return obs

# ===== 전략 4: 브레이커 블럭 =====
def detect_breaker(df):
    breakers = []
    for i in range(10, len(df)):
        # 고점 찾기
        window = df.iloc[i-10:i]
        highs_idx = argrelextrema(window['high'].values, np.greater, order=3)[0]
        
        if len(highs_idx) >= 1:
            last_high_idx = highs_idx[-1]
            high_price = window.iloc[last_high_idx]['high']
            
            # 현재가가 그 고점을 돌파했는지
            if df.iloc[i]['close'] > high_price:
                # 돌파 후 되돌림 예상 구간 = 이전 고점 영역
                breakers.append({
                    'idx': i, 'time': df.iloc[i]['datetime'],
                    'top': high_price,
                    'bottom': high_price * 0.99,  # 1% 아래까지
                    'type': 'breaker'
                })
    return breakers

# 시뮬레이션 함수
def simulate_long(signals, df_15m, tp1, tp2, sl, time_stop=96, max_touch_bars=50):
    trades = []
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    for sig in signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        entry_level = sig.get('top', sig.get('bottom', 0))
        
        touch_idx = None
        for i in range(start+1, min(start+max_touch_bars, len(times))):
            if lows[i] <= entry_level:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        tp1_p, tp2_p, sl_p = ep*(1+tp1/100), ep*(1+tp2/100), ep*(1+sl/100)
        
        result, pnl, tp1_hit = None, 0, False
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                pnl = (closes[i] - ep) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            if not tp1_hit:
                if lows[i] <= sl_p:
                    result, pnl = 'SL', sl - 0.06
                    break
                if highs[i] >= tp1_p:
                    tp1_hit = True
                    continue
            else:
                if lows[i] <= ep:
                    result, pnl = 'BE', -0.06
                    break
                if highs[i] >= tp2_p:
                    result, pnl = 'TP2', tp2 - 0.06
                    break
        
        if result is None and tp1_hit:
            result, pnl = 'BE', -0.06
        
        if result:
            trades.append({'time': times[entry_idx], 'result': result, 'pnl': pnl, 'type': sig['type']})
    
    return trades

def simulate_short(signals, df_15m, tp1, tp2, sl, time_stop=96, max_touch_bars=50):
    trades = []
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    for sig in signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        entry_level = sig.get('bottom', sig.get('top', 0))
        
        touch_idx = None
        for i in range(start+1, min(start+max_touch_bars, len(times))):
            if highs[i] >= entry_level:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        tp1_p, tp2_p, sl_p = ep*(1-tp1/100), ep*(1-tp2/100), ep*(1-sl/100)  # 숏은 반대
        
        result, pnl, tp1_hit = None, 0, False
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                pnl = (ep - closes[i]) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            if not tp1_hit:
                if highs[i] >= sl_p:  # 숏은 고점이 SL
                    result, pnl = 'SL', sl - 0.06
                    break
                if lows[i] <= tp1_p:  # 숏은 저점이 TP
                    tp1_hit = True
                    continue
            else:
                if highs[i] >= ep:  # 본절
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

def calc_stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    sl = (df['result'] == 'SL').sum()
    months = len(pd.to_datetime(df['time']).dt.to_period('M').unique())
    pnl = df['pnl'].sum()
    return {'n': len(trades), 'sl_rate': sl/len(trades)*100, 'win_rate': (len(trades)-sl)/len(trades)*100, 'pnl': pnl, 'mavg': pnl/months}

print("=" * 90)
print("🎯 다중 전략 시스템")
print("=" * 90)

# 각 전략별 시그널 감지
bull_fvg = detect_bullish_fvg(df_4h)
bear_fvg = detect_bearish_fvg(df_4h)
bull_ob = detect_orderblock(df_4h)
breakers = detect_breaker(df_4h)

print(f"\n[시그널 감지 결과]")
print(f"  상승 FVG: {len(bull_fvg)}개")
print(f"  하락 FVG: {len(bear_fvg)}개")
print(f"  상승 오더블럭: {len(bull_ob)}개")
print(f"  브레이커: {len(breakers)}개")

# 각 전략 개별 테스트
print("\n" + "=" * 90)
print("📊 전략별 개별 성과")
print("=" * 90)

strategies = [
    ("전략1: 상승FVG 롱", bull_fvg, 'long', {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96}),
    ("전략2: 하락FVG 숏", bear_fvg, 'short', {'tp1': 2.0, 'tp2': 3.5, 'sl': -1.5, 'time_stop': 96}),
    ("전략3: 오더블럭 롱", bull_ob, 'long', {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96}),
    ("전략4: 브레이커 롱", breakers, 'long', {'tp1': 1.5, 'tp2': 3.0, 'sl': -1.5, 'time_stop': 96}),
]

print(f"\n{'전략':<25} {'거래':>6} {'SL%':>7} {'승률':>7} {'총PnL':>9} {'월평균':>8}")
print("-" * 70)

all_trades = []
for name, signals, direction, params in strategies:
    if direction == 'long':
        trades = simulate_long(signals, df_15m, **params)
    else:
        trades = simulate_short(signals, df_15m, **params)
    
    s = calc_stats(trades)
    if s:
        print(f"{name:<25} {s['n']:>6} {s['sl_rate']:>6.1f}% {s['win_rate']:>6.1f}% {s['pnl']:>8.1f}% {s['mavg']:>7.2f}%")
        all_trades.extend(trades)

# 통합 성과
print("\n" + "=" * 90)
print("🏆 다중 전략 통합 성과")
print("=" * 90)

if all_trades:
    df_all = pd.DataFrame(all_trades)
    df_all['time'] = pd.to_datetime(df_all['time'])
    df_all = df_all.sort_values('time')
    
    # 전체 통계
    total_sl = (df_all['result'] == 'SL').sum()
    total_months = len(df_all['time'].dt.to_period('M').unique())
    total_pnl = df_all['pnl'].sum()
    
    print(f"\n[통합 결과]")
    print(f"  총 거래: {len(df_all)}회")
    print(f"  월평균 거래: {len(df_all)/total_months:.1f}회")
    print(f"  SL 비율: {total_sl/len(df_all)*100:.1f}%")
    print(f"  승률: {(len(df_all)-total_sl)/len(df_all)*100:.1f}%")
    print(f"  총 PnL: {total_pnl:.1f}%")
    print(f"  월평균 PnL: {total_pnl/total_months:.2f}%")
    
    # 전략별 비중
    print(f"\n[전략별 기여도]")
    for stype in df_all['type'].unique():
        subset = df_all[df_all['type'] == stype]
        print(f"  {stype}: {len(subset)}회 ({len(subset)/len(df_all)*100:.1f}%), PnL {subset['pnl'].sum():.1f}%")
    
    # 연도별 통합 성과
    print(f"\n[연도별 통합 성과]")
    df_all['year'] = df_all['time'].dt.year
    print(f"{'연도':<6} {'거래':>6} {'SL%':>7} {'총PnL':>9} {'월평균':>8}")
    print("-" * 45)
    
    for year in sorted(df_all['year'].unique()):
        yearly = df_all[df_all['year'] == year]
        sl = (yearly['result'] == 'SL').sum()
        months = len(yearly['time'].dt.to_period('M').unique())
        pnl = yearly['pnl'].sum()
        print(f"{year:<6} {len(yearly):>6} {sl/len(yearly)*100:>6.1f}% {pnl:>8.1f}% {pnl/months:>7.2f}%")

# 단일 전략 vs 다중 전략 비교
print("\n" + "=" * 90)
print("📈 단일 전략 vs 다중 전략 비교")
print("=" * 90)

# 단일 (상승 FVG만)
single_trades = simulate_long(bull_fvg, df_15m, tp1=2.0, tp2=3.5, sl=-1.5, time_stop=96)
single_stats = calc_stats(single_trades)

print(f"\n{'구분':<20} {'거래':>8} {'월거래':>8} {'승률':>8} {'월평균':>10}")
print("-" * 60)
print(f"{'단일 (상승FVG)':<20} {single_stats['n']:>8} {single_stats['n']/69:>7.1f} {single_stats['win_rate']:>7.1f}% {single_stats['mavg']:>9.2f}%")
print(f"{'다중 (4전략)':<20} {len(df_all):>8} {len(df_all)/total_months:>7.1f} {(len(df_all)-total_sl)/len(df_all)*100:>7.1f}% {total_pnl/total_months:>9.2f}%")

improvement = (total_pnl/total_months) / single_stats['mavg'] * 100 - 100
print(f"\n  → 다중 전략으로 월평균 수익 {improvement:+.1f}% 증가!")

