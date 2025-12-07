import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# MACD 계산
def calc_macd(df):
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    macd = exp1 - exp2
    return macd, macd.ewm(span=9).mean()

df_4h['macd'], df_4h['macd_sig'] = calc_macd(df_4h)

# FVG 감지
def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        p2, curr = df.iloc[i-2], df.iloc[i]
        if p2['high'] < curr['low']:
            fvgs.append({
                'idx': i,
                'time': curr['datetime'],
                'top': curr['low'],
                'body': abs(curr['close'] - curr['open']) / curr['open'] * 100
            })
    return fvgs

fvgs = detect_fvg(df_4h)

# 조건별 FVG 분류
def classify_fvg(df, fvg, lookback=20):
    """FVG 발생 전 상황 분석"""
    idx = fvg['idx']
    if idx < lookback:
        return {'diver': False, 'hh': False}
    
    window = df.iloc[idx-lookback:idx]
    
    # 저점 찾기
    lows_idx = argrelextrema(window['low'].values, np.less, order=3)[0]
    
    # 다이버전스 체크 (저점 2개 이상)
    diver = False
    if len(lows_idx) >= 2:
        l1_idx, l2_idx = lows_idx[-2], lows_idx[-1]
        l1_price = window.iloc[l1_idx]['low']
        l2_price = window.iloc[l2_idx]['low']
        l1_macd = window.iloc[l1_idx]['macd']
        l2_macd = window.iloc[l2_idx]['macd']
        
        # 가격↘ + MACD↗
        if l2_price < l1_price and l2_macd > l1_macd:
            diver = True
    
    # 고점 올림 체크
    highs_idx = argrelextrema(window['high'].values, np.greater, order=3)[0]
    hh = False
    if len(highs_idx) >= 2:
        h1 = window.iloc[highs_idx[-2]]['high']
        h2 = window.iloc[highs_idx[-1]]['high']
        if h2 > h1:
            hh = True
    
    return {'diver': diver, 'hh': hh}

# 시뮬레이션
def simulate(fvgs, df_4h, df_15m, tp, sl, require_diver=False, require_hh=False):
    trades = []
    times = df_15m['datetime'].values
    lows = df_15m['low'].values
    highs = df_15m['high'].values
    opens = df_15m['open'].values
    
    for fvg in fvgs:
        # 조건 체크
        cond = classify_fvg(df_4h, fvg)
        if require_diver and not cond['diver']:
            continue
        if require_hh and not cond['hh']:
            continue
        
        # 15분봉 진입
        start = np.searchsorted(times, np.datetime64(fvg['time']))
        if start >= len(times) - 200:
            continue
        
        entry_idx = None
        for i in range(start+1, min(start+200, len(times))):
            if lows[i] <= fvg['top']:
                if i+1 < len(times):
                    entry_idx = i+1
                break
        
        if entry_idx is None:
            continue
        
        ep = opens[entry_idx]
        tp_p, sl_p = ep * (1 + tp/100), ep * (1 + sl/100)
        
        result = None
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if lows[i] <= sl_p:
                result = 'L'
                break
            if highs[i] >= tp_p:
                result = 'W'
                break
        
        if result:
            trades.append({
                'time': times[entry_idx],
                'result': result,
                'pnl': tp - 0.06 if result == 'W' else sl - 0.06,
                'diver': cond['diver'],
                'hh': cond['hh']
            })
    
    return trades

def stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    wr = (df['result'] == 'W').mean() * 100
    pnl = df['pnl'].sum()
    
    cum, peak, mdd = 0, 0, 0
    for p in df['pnl']:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    months = len(df['month'].unique())
    
    return {'n': len(trades), 'wr': wr, 'pnl': pnl, 'mdd': mdd, 'mavg': pnl/months}

print("=" * 70)
print("🎯 조건별 FVG 전략 비교")
print("=" * 70)

# 조건 조합 테스트
configs = [
    ("기본 (FVG만)", False, False),
    ("다이버전스 O", True, False),
    ("고점올림 O", False, True),
    ("다이버 + 고점올림", True, True),
]

print(f"\n[TP 1.0% / SL -1.5% 기준]")
print(f"{'조건':<20} {'거래':>6} {'승률':>7} {'MDD':>7} {'총PnL':>8} {'월평균':>8}")
print("-" * 65)

for name, req_d, req_h in configs:
    trades = simulate(fvgs, df_4h, df_15m, 1.0, -1.5, req_d, req_h)
    s = stats(trades)
    if s:
        print(f"{name:<20} {s['n']:>6} {s['wr']:>6.1f}% {s['mdd']:>6.1f}% {s['pnl']:>7.1f}% {s['mavg']:>7.2f}%")

# 기본 전략에서 조건별 세부 분석
print("\n" + "=" * 70)
print("📊 기본 FVG 내 조건별 승률 분석")
print("=" * 70)

all_trades = simulate(fvgs, df_4h, df_15m, 1.0, -1.5, False, False)
df_all = pd.DataFrame(all_trades)

# 다이버 O vs X
diver_yes = df_all[df_all['diver'] == True]
diver_no = df_all[df_all['diver'] == False]

print(f"\n[다이버전스 여부]")
if len(diver_yes) > 0:
    wr_yes = (diver_yes['result'] == 'W').mean() * 100
    print(f"  다이버 O: {len(diver_yes)}회, 승률 {wr_yes:.1f}%")
if len(diver_no) > 0:
    wr_no = (diver_no['result'] == 'W').mean() * 100
    print(f"  다이버 X: {len(diver_no)}회, 승률 {wr_no:.1f}%")

# 고점올림 O vs X
hh_yes = df_all[df_all['hh'] == True]
hh_no = df_all[df_all['hh'] == False]

print(f"\n[고점올림 여부]")
if len(hh_yes) > 0:
    wr_yes = (hh_yes['result'] == 'W').mean() * 100
    print(f"  고점올림 O: {len(hh_yes)}회, 승률 {wr_yes:.1f}%")
if len(hh_no) > 0:
    wr_no = (hh_no['result'] == 'W').mean() * 100
    print(f"  고점올림 X: {len(hh_no)}회, 승률 {wr_no:.1f}%")

# 둘 다 O
both = df_all[(df_all['diver'] == True) & (df_all['hh'] == True)]
if len(both) > 0:
    wr_both = (both['result'] == 'W').mean() * 100
    print(f"\n[다이버 + 고점올림 둘 다]: {len(both)}회, 승률 {wr_both:.1f}%")

# 최적 조합 찾기
print("\n" + "=" * 70)
print("🏆 조건별 TP/SL 최적화")
print("=" * 70)

for name, req_d, req_h in configs:
    print(f"\n[{name}]")
    best = None
    for tp in [1.0, 1.5, 2.0]:
        for sl in [-1.0, -1.5, -2.0]:
            trades = simulate(fvgs, df_4h, df_15m, tp, sl, req_d, req_h)
            s = stats(trades)
            if s and (best is None or s['wr'] > best['wr']):
                best = {'tp': tp, 'sl': sl, **s}
    
    if best:
        print(f"  최적: TP {best['tp']}% SL {best['sl']}%")
        print(f"  거래: {best['n']}, 승률: {best['wr']:.1f}%, MDD: {best['mdd']:.1f}%, 월평균: {best['mavg']:.2f}%")

