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
                'bottom': p2['high'],
                'body': abs(curr['close'] - curr['open']) / curr['open'] * 100,
                'is_bullish': curr['close'] > curr['open']
            })
    return fvgs

fvgs = detect_fvg(df_4h)

# 조건 분류
def classify_fvg(df, fvg, lookback=20):
    idx = fvg['idx']
    if idx < lookback:
        return {'diver': False, 'hh': False}
    
    window = df.iloc[idx-lookback:idx]
    lows_idx = argrelextrema(window['low'].values, np.less, order=3)[0]
    
    diver = False
    if len(lows_idx) >= 2:
        l1_idx, l2_idx = lows_idx[-2], lows_idx[-1]
        l1_price, l2_price = window.iloc[l1_idx]['low'], window.iloc[l2_idx]['low']
        l1_macd, l2_macd = window.iloc[l1_idx]['macd'], window.iloc[l2_idx]['macd']
        if l2_price < l1_price and l2_macd > l1_macd:
            diver = True
    
    highs_idx = argrelextrema(window['high'].values, np.greater, order=3)[0]
    hh = False
    if len(highs_idx) >= 2:
        h1 = window.iloc[highs_idx[-2]]['high']
        h2 = window.iloc[highs_idx[-1]]['high']
        if h2 > h1:
            hh = True
    
    return {'diver': diver, 'hh': hh}

# 본절스탑 시뮬레이션 (기존 로직 - TP1 도달 시 전량 본절 이동)
def simulate_breakeven_original(fvgs, df_4h, df_15m, tp1, tp2, sl, filters=None):
    """
    TP1 도달 시 -> 본절로 SL 이동 -> TP2 노림
    - TP2 도달: +TP2% 수익
    - 본절 터치: 0% (수수료만 손실)
    - SL 터치: SL% 손실
    """
    if filters is None:
        filters = {}
    
    trades = []
    times = df_15m['datetime'].values
    lows = df_15m['low'].values
    highs = df_15m['high'].values
    opens = df_15m['open'].values
    closes = df_15m['close'].values
    
    for fvg in fvgs:
        cond = classify_fvg(df_4h, fvg)
        
        # 필터 적용
        if filters.get('exclude_diver_only', False):
            if cond['diver'] and not cond['hh']:
                continue
        
        if filters.get('require_hh', False):
            if not cond['hh']:
                continue
        
        if fvg['body'] < filters.get('min_body', 0):
            continue
        
        start = np.searchsorted(times, np.datetime64(fvg['time']))
        if start >= len(times) - 200:
            continue
        
        touch_idx = None
        for i in range(start+1, min(start+200, len(times))):
            if lows[i] <= fvg['top']:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        bars_to_touch = touch_idx - start
        if bars_to_touch > filters.get('max_touch_bars', 999):
            continue
        
        is_bullish_touch = closes[touch_idx] > opens[touch_idx]
        if filters.get('require_bullish_touch', False) and not is_bullish_touch:
            continue
        
        touch_depth = (fvg['top'] - lows[touch_idx]) / fvg['top'] * 100
        if touch_depth > filters.get('max_touch_depth', 999):
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
            
        ep = opens[entry_idx]
        tp1_p = ep * (1 + tp1/100)
        tp2_p = ep * (1 + tp2/100)
        sl_p = ep * (1 + sl/100)
        
        result = None
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            # SL 먼저 체크 (보수적)
            if lows[i] <= sl_p:
                result = 'SL'
                break
            # TP1 도달 -> 본절로 이동
            if highs[i] >= tp1_p:
                for j in range(i+1, min(entry_idx+200, len(times))):
                    if lows[j] <= ep:  # 본절 터치
                        result = 'BE'
                        break
                    if highs[j] >= tp2_p:
                        result = 'TP2'
                        break
                if result is None:
                    result = 'BE'
                break
        
        if result:
            if result == 'SL':
                pnl = sl - 0.06
            elif result == 'BE':
                pnl = -0.06
            else:  # TP2
                pnl = tp2 - 0.06  # 전량 TP2 익절
            
            trades.append({
                'time': times[entry_idx],
                'result': result,
                'pnl': pnl,
                'diver': cond['diver'],
                'hh': cond['hh'],
                'body': fvg['body'],
                'bars_to_touch': bars_to_touch,
                'bullish_touch': is_bullish_touch,
                'touch_depth': touch_depth
            })
    
    return trades

def calc_stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    
    sl_cnt = (df['result'] == 'SL').sum()
    be_cnt = (df['result'] == 'BE').sum()
    tp2_cnt = (df['result'] == 'TP2').sum()
    
    pnl = df['pnl'].sum()
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    months = len(df['month'].unique())
    
    return {
        'n': len(trades),
        'sl': sl_cnt,
        'be': be_cnt,
        'tp2': tp2_cnt,
        'sl_rate': sl_cnt / len(trades) * 100,
        'win_rate': (be_cnt + tp2_cnt) / len(trades) * 100,
        'pnl': pnl,
        'mavg': pnl / months
    }

print("=" * 85)
print("🎯 필터 적용 테스트 (본절스탑 전략: TP1 1.5% → 본절 → TP2 4%, SL -1.5%)")
print("   ※ TP2 도달 시 +4% 익절, 본절 시 0%, SL 시 -1.5%")
print("=" * 85)

# 테스트할 필터 조합
filter_tests = [
    ("기본 (필터 없음)", {}),
    
    # 단일 필터
    ("① 다이버만O 제외", {'exclude_diver_only': True}),
    ("② 양봉 터치 필수", {'require_bullish_touch': True}),
    ("③ 터치 깊이 < 0.5%", {'max_touch_depth': 0.5}),
    ("④ 터치까지 ≤ 20바", {'max_touch_bars': 20}),
    ("⑤ 4H 바디 ≥ 1%", {'min_body': 1.0}),
    ("⑥ 고점올림 O 필수", {'require_hh': True}),
    
    # 복합 필터
    ("⑦ ①+④ (다이버X + 빠른터치)", {'exclude_diver_only': True, 'max_touch_bars': 20}),
    ("⑧ ①+④+⑤ (+ 큰바디)", {'exclude_diver_only': True, 'max_touch_bars': 20, 'min_body': 1.0}),
    ("⑨ ②+④ (양봉+빠른터치)", {'require_bullish_touch': True, 'max_touch_bars': 20}),
    ("⑩ ①+②+④", {'exclude_diver_only': True, 'require_bullish_touch': True, 'max_touch_bars': 20}),
    ("⑪ ①+④+⑤+② 전체", {'exclude_diver_only': True, 'max_touch_bars': 20, 'min_body': 1.0, 'require_bullish_touch': True}),
]

print(f"\n{'필터':<28} {'거래':>5} {'SL':>4} {'BE':>4} {'TP2':>4} {'SL%':>6} {'승률':>6} {'총PnL':>8} {'월평균':>7}")
print("-" * 85)

results = []
for name, filters in filter_tests:
    trades = simulate_breakeven_original(fvgs, df_4h, df_15m, 1.5, 4.0, -1.5, filters)
    s = calc_stats(trades)
    if s:
        print(f"{name:<28} {s['n']:>5} {s['sl']:>4} {s['be']:>4} {s['tp2']:>4} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['pnl']:>7.1f}% {s['mavg']:>6.2f}%")
        results.append({'name': name, **s})

# 가장 효과적인 필터 분석
print("\n" + "=" * 85)
print("📊 필터 효과 분석")
print("=" * 85)

df_results = pd.DataFrame(results)
base = df_results[df_results['name'] == '기본 (필터 없음)'].iloc[0]

print(f"\n[기준: 기본 전략]")
print(f"  거래: {base['n']}회, SL: {base['sl_rate']:.1f}%, 승률: {base['win_rate']:.1f}%, 월평균: {base['mavg']:.2f}%")

print(f"\n[필터별 개선 효과]")
print(f"{'필터':<28} {'SL 감소':>8} {'승률 증가':>8} {'월평균 변화':>10} {'거래 감소':>8}")
print("-" * 70)

for _, row in df_results.iterrows():
    if row['name'] == '기본 (필터 없음)':
        continue
    sl_diff = row['sl_rate'] - base['sl_rate']
    wr_diff = row['win_rate'] - base['win_rate']
    mavg_diff = row['mavg'] - base['mavg']
    trade_diff = (row['n'] - base['n']) / base['n'] * 100
    
    print(f"{row['name']:<28} {sl_diff:>+7.1f}% {wr_diff:>+7.1f}% {mavg_diff:>+9.2f}% {trade_diff:>+7.1f}%")

# 최적 조합 추천
print("\n" + "=" * 85)
print("🏆 최적 필터 추천")
print("=" * 85)

# 월평균 기준 상위
df_sorted_mavg = df_results.sort_values('mavg', ascending=False).head(5)
print("\n[월평균 수익 TOP 5]")
for _, row in df_sorted_mavg.iterrows():
    print(f"  {row['name']}: 월평균 {row['mavg']:.2f}%, 거래 {row['n']}회, SL {row['sl_rate']:.1f}%")

# SL 낮은 순 (최소 100거래)
df_low_sl = df_results[df_results['n'] >= 100].sort_values('sl_rate').head(5)
print("\n[SL 비율 낮은 TOP 5 (100거래+)]")
for _, row in df_low_sl.iterrows():
    print(f"  {row['name']}: SL {row['sl_rate']:.1f}%, 승률 {row['win_rate']:.1f}%, 월평균 {row['mavg']:.2f}%")

# 균형 잡힌 추천 (월평균 3% 이상 + SL 30% 미만)
df_balanced = df_results[(df_results['mavg'] >= 3) | (df_results['sl_rate'] <= 25)]
print("\n[균형 추천 (월평균 3%+ 또는 SL 25%-)]")
for _, row in df_balanced.sort_values('mavg', ascending=False).iterrows():
    print(f"  {row['name']}: 월평균 {row['mavg']:.2f}%, SL {row['sl_rate']:.1f}%, 거래 {row['n']}회")

