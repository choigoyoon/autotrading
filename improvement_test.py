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

# RSI 계산
def calc_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df_4h['rsi'] = calc_rsi(df_4h)
df_15m['rsi'] = calc_rsi(df_15m)

# 볼린저 밴드
def calc_bb(df, period=20):
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return sma, sma + 2*std, sma - 2*std

df_4h['bb_mid'], df_4h['bb_upper'], df_4h['bb_lower'] = calc_bb(df_4h)

# ATR 계산
def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

df_4h['atr'] = calc_atr(df_4h)

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
                'gap_size': (curr['low'] - p2['high']) / p2['high'] * 100,
                'body': abs(curr['close'] - curr['open']) / curr['open'] * 100,
                'rsi': curr['rsi'] if pd.notna(curr['rsi']) else 50,
                'atr': curr['atr'] if pd.notna(curr['atr']) else 0,
                'bb_pos': (curr['close'] - curr['bb_lower']) / (curr['bb_upper'] - curr['bb_lower']) if pd.notna(curr['bb_lower']) else 0.5
            })
    return fvgs

fvgs = detect_fvg(df_4h)

# 분류
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

# 개선된 시뮬레이션
def simulate_improved(fvgs, df_4h, df_15m, config):
    """
    config = {
        'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5,
        'use_atr_sl': False,        # ATR 기반 SL
        'atr_mult': 1.5,
        'use_atr_tp': False,        # ATR 기반 TP
        'trailing_stop': False,     # 트레일링 스탑
        'trail_trigger': 2.0,       # 트레일링 시작 %
        'trail_dist': 1.0,          # 트레일링 거리 %
        'partial_tp': False,        # 분할 익절 (TP1에서 50%)
        'time_stop': 0,             # 시간 스탑 (바 수, 0=없음)
        'max_touch_bars': 999,
        'min_body': 0,
        'rsi_filter': None,         # (min, max) 또는 None
        'gap_size_filter': None,    # (min, max) 또는 None
    }
    """
    trades = []
    times = df_15m['datetime'].values
    lows = df_15m['low'].values
    highs = df_15m['high'].values
    opens = df_15m['open'].values
    closes = df_15m['close'].values
    
    for fvg in fvgs:
        cond = classify_fvg(df_4h, fvg)
        
        # 다이버만 O 제외
        if cond['diver'] and not cond['hh']:
            continue
        
        if fvg['body'] < config.get('min_body', 0):
            continue
        
        # RSI 필터
        rsi_filter = config.get('rsi_filter')
        if rsi_filter:
            if fvg['rsi'] < rsi_filter[0] or fvg['rsi'] > rsi_filter[1]:
                continue
        
        # 갭 크기 필터
        gap_filter = config.get('gap_size_filter')
        if gap_filter:
            if fvg['gap_size'] < gap_filter[0] or fvg['gap_size'] > gap_filter[1]:
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
        if bars_to_touch > config.get('max_touch_bars', 999):
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        
        # SL/TP 설정
        if config.get('use_atr_sl') and fvg['atr'] > 0:
            sl_pct = -fvg['atr'] / ep * 100 * config.get('atr_mult', 1.5)
        else:
            sl_pct = config['sl']
        
        if config.get('use_atr_tp') and fvg['atr'] > 0:
            tp2_pct = fvg['atr'] / ep * 100 * config.get('atr_mult', 1.5) * 2
        else:
            tp2_pct = config['tp2']
        
        tp1_pct = config['tp1']
        
        tp1_p = ep * (1 + tp1_pct/100)
        tp2_p = ep * (1 + tp2_pct/100)
        sl_p = ep * (1 + sl_pct/100)
        
        result = None
        pnl = 0
        max_price = ep
        tp1_hit = False
        
        time_stop = config.get('time_stop', 0)
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            bars_in_trade = i - entry_idx
            
            # 시간 스탑
            if time_stop > 0 and bars_in_trade >= time_stop:
                exit_price = closes[i]
                pnl = (exit_price - ep) / ep * 100 - 0.06
                result = 'TIME'
                break
            
            # 트레일링 스탑 로직
            if config.get('trailing_stop') and not tp1_hit:
                max_price = max(max_price, highs[i])
                trail_trigger_p = ep * (1 + config.get('trail_trigger', 2.0)/100)
                
                if max_price >= trail_trigger_p:
                    trail_sl = max_price * (1 - config.get('trail_dist', 1.0)/100)
                    if lows[i] <= trail_sl:
                        pnl = (trail_sl - ep) / ep * 100 - 0.06
                        result = 'TRAIL'
                        break
            
            # 기본 본절 스탑 로직
            if not tp1_hit:
                if lows[i] <= sl_p:
                    result = 'SL'
                    pnl = sl_pct - 0.06
                    break
                
                if highs[i] >= tp1_p:
                    tp1_hit = True
                    if config.get('partial_tp'):
                        pnl += tp1_pct / 2  # 50% 익절
                    continue
            
            # TP1 도달 후
            if tp1_hit:
                if lows[i] <= ep:  # 본절
                    result = 'BE'
                    if config.get('partial_tp'):
                        pnl += -0.06  # 나머지 50% 본절
                    else:
                        pnl = -0.06
                    break
                
                if highs[i] >= tp2_p:
                    result = 'TP2'
                    if config.get('partial_tp'):
                        pnl += tp2_pct / 2 - 0.06  # 나머지 50% TP2
                    else:
                        pnl = tp2_pct - 0.06
                    break
        
        if result is None and tp1_hit:
            result = 'BE'
            pnl = -0.06 if not config.get('partial_tp') else pnl - 0.06
        
        if result:
            trades.append({
                'time': times[entry_idx],
                'result': result,
                'pnl': pnl,
                'gap_size': fvg['gap_size'],
                'rsi': fvg['rsi'],
                'atr': fvg['atr']
            })
    
    return trades

def calc_stats(trades):
    if len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    
    results = df['result'].value_counts().to_dict()
    
    pnl = df['pnl'].sum()
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    months = len(df['month'].unique())
    
    # MDD 계산
    cum_pnl = df['pnl'].cumsum()
    peak = cum_pnl.expanding().max()
    mdd = (cum_pnl - peak).min()
    
    sl_cnt = results.get('SL', 0)
    win_cnt = len(trades) - sl_cnt
    
    return {
        'n': len(trades),
        'results': results,
        'sl_rate': sl_cnt / len(trades) * 100,
        'win_rate': win_cnt / len(trades) * 100,
        'pnl': pnl,
        'mdd': mdd,
        'mavg': pnl / months
    }

print("=" * 90)
print("🚀 개선 아이디어 테스트")
print("=" * 90)

# 기본 설정
base_config = {
    'tp1': 1.5, 'tp2': 4.0, 'sl': -1.5,
    'max_touch_bars': 20,  # 기본 필터 적용
}

# 테스트할 개선안들
improvements = [
    ("기준: 본절스탑 + 빠른터치", base_config),
    
    # 1. 분할 익절
    ("① 분할익절 (TP1 50% + TP2 50%)", {**base_config, 'partial_tp': True}),
    
    # 2. 트레일링 스탑
    ("② 트레일링 (2% 트리거, 1% 거리)", {**base_config, 'trailing_stop': True, 'trail_trigger': 2.0, 'trail_dist': 1.0}),
    ("③ 트레일링 (3% 트리거, 1.5% 거리)", {**base_config, 'trailing_stop': True, 'trail_trigger': 3.0, 'trail_dist': 1.5}),
    
    # 3. 시간 스탑
    ("④ 시간스탑 48바 (12시간)", {**base_config, 'time_stop': 48}),
    ("⑤ 시간스탑 96바 (24시간)", {**base_config, 'time_stop': 96}),
    
    # 4. ATR 기반 동적 SL/TP
    ("⑥ ATR SL (1.5x ATR)", {**base_config, 'use_atr_sl': True, 'atr_mult': 1.5}),
    ("⑦ ATR SL+TP", {**base_config, 'use_atr_sl': True, 'use_atr_tp': True, 'atr_mult': 1.5}),
    
    # 5. RSI 필터
    ("⑧ RSI 30-70 (중립구간)", {**base_config, 'rsi_filter': (30, 70)}),
    ("⑨ RSI < 50 (과매도 근처)", {**base_config, 'rsi_filter': (0, 50)}),
    
    # 6. 갭 크기 필터
    ("⑩ 갭 0.5%+ (큰 갭만)", {**base_config, 'gap_size_filter': (0.5, 100)}),
    ("⑪ 갭 0.3-2% (적정 갭)", {**base_config, 'gap_size_filter': (0.3, 2.0)}),
    
    # 7. TP/SL 비율 조정
    ("⑫ TP2 5% (리스크:리워드 1:3.3)", {**base_config, 'tp2': 5.0}),
    ("⑬ TP2 6% (리스크:리워드 1:4)", {**base_config, 'tp2': 6.0}),
    ("⑭ SL -1% (타이트 SL)", {**base_config, 'sl': -1.0}),
    ("⑮ SL -2% (넓은 SL)", {**base_config, 'sl': -2.0}),
    
    # 8. 복합
    ("⑯ 분할익절 + TP2 5%", {**base_config, 'partial_tp': True, 'tp2': 5.0}),
    ("⑰ 분할익절 + SL -2%", {**base_config, 'partial_tp': True, 'sl': -2.0}),
]

print(f"\n{'개선안':<35} {'거래':>5} {'SL%':>6} {'승률':>6} {'MDD':>7} {'총PnL':>8} {'월평균':>7}")
print("-" * 90)

results_list = []
for name, config in improvements:
    trades = simulate_improved(fvgs, df_4h, df_15m, config)
    s = calc_stats(trades)
    if s:
        print(f"{name:<35} {s['n']:>5} {s['sl_rate']:>5.1f}% {s['win_rate']:>5.1f}% {s['mdd']:>6.1f}% {s['pnl']:>7.1f}% {s['mavg']:>6.2f}%")
        results_list.append({'name': name, 'config': config, **s})

# 최고 성과 분석
print("\n" + "=" * 90)
print("🏆 최고 성과 분석")
print("=" * 90)

df_res = pd.DataFrame(results_list)
base_row = df_res[df_res['name'].str.contains('기준')].iloc[0]

print(f"\n[기준 대비 개선]")
print(f"{'개선안':<35} {'월평균 변화':>10} {'SL 변화':>8} {'MDD 변화':>8}")
print("-" * 70)

for _, row in df_res.iterrows():
    if '기준' in row['name']:
        continue
    mavg_diff = row['mavg'] - base_row['mavg']
    sl_diff = row['sl_rate'] - base_row['sl_rate']
    mdd_diff = row['mdd'] - base_row['mdd']
    
    # 개선된 것만 표시
    if mavg_diff > 0 or sl_diff < -1 or mdd_diff > 1:
        print(f"{row['name']:<35} {mavg_diff:>+9.2f}% {sl_diff:>+7.1f}% {mdd_diff:>+7.1f}%")

# 월평균 상위
print("\n[월평균 수익 TOP 5]")
for _, row in df_res.nlargest(5, 'mavg').iterrows():
    print(f"  {row['name']}: 월평균 {row['mavg']:.2f}%, MDD {row['mdd']:.1f}%")

# MDD 상위 (덜 나쁜 순)
print("\n[MDD 양호 TOP 5]")
for _, row in df_res.nlargest(5, 'mdd').iterrows():
    print(f"  {row['name']}: MDD {row['mdd']:.1f}%, 월평균 {row['mavg']:.2f}%")

