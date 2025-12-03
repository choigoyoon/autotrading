#!/usr/bin/env python3
"""
최적화된 MTF 롱/숏 통합 전략

이전 성공적인 롱 전략 (mtf_optimal_results.csv):
- 1055건, 51.7% 승률, 0.60% 평균 PnL, 627.8% 총 PnL
- Entry-L Gap 1.5-2%: 55.5% 승률
- Entry-L Gap 2-3%: 58.4% 승률

핵심 원리 (사용자 원칙):
1. 하락추세 시작 ~ 추세돌파까지 전체를 봐야 함
2. MTF 다이버전스 (1H + 15M)
3. Breakout → Adjustment (Retest) → 1차 상승 → 2차 상승
4. Entry-L Gap 1~3%로 손절폭 적정 유지
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("최적화된 MTF 롱/숏 통합 전략")
print("=" * 80)

# 데이터 로드
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

# 1시간봉 생성
df_1h = df_15m.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

print(f"15M: {len(df_15m):,}개")
print(f"1H: {len(df_1h):,}개")

# 기술적 지표 계산
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 1H MACD
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

# 1H RSI
df_1h['rsi'] = calc_rsi(df_1h['close'])

# 15M RSI & MACD
df_15m['rsi'] = calc_rsi(df_15m['close'])
exp1_15m = df_15m['close'].ewm(span=12, adjust=False).mean()
exp2_15m = df_15m['close'].ewm(span=26, adjust=False).mean()
df_15m['macd'] = exp1_15m - exp2_15m
df_15m['signal_line'] = df_15m['macd'].ewm(span=9, adjust=False).mean()
df_15m['hist'] = df_15m['macd'] - df_15m['signal_line']

# 15M 볼륨 평균
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# H/L 추출 함수
def extract_hl_points(df):
    hist = df['hist'].values
    high = df['high'].values
    low = df['low'].values
    timestamps = df['datetime'].values
    close = df['close'].values
    rsi = df['rsi'].values if 'rsi' in df.columns else np.zeros(len(df))
    
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({
                'type': 'H', 
                'price': high[max_idx], 
                'time': pd.Timestamp(timestamps[max_idx]), 
                'idx': max_idx,
                'rsi': rsi[max_idx] if max_idx < len(rsi) else 50
            })
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({
                'type': 'L', 
                'price': low[min_idx], 
                'time': pd.Timestamp(timestamps[min_idx]), 
                'idx': min_idx,
                'rsi': rsi[min_idx] if min_idx < len(rsi) else 50
            })
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)
print(f"1H H/L points: {len(points_1h)}")

# 15M 스윙 포인트
def get_15m_swing_points(df_15m, window=5):
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    times = df_15m['datetime'].values
    rsi = df_15m['rsi'].values
    
    swing_highs, swing_lows = [], []
    
    for i in range(window, len(df_15m) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append({'price': highs[i], 'time': pd.Timestamp(times[i]), 'idx': i, 'rsi': rsi[i]})
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append({'price': lows[i], 'time': pd.Timestamp(times[i]), 'idx': i, 'rsi': rsi[i]})
    
    return swing_highs, swing_lows

swing_highs_15m, swing_lows_15m = get_15m_swing_points(df_15m)
print(f"15M Swing Highs: {len(swing_highs_15m)}, Swing Lows: {len(swing_lows_15m)}")

# ============================================================
# 다이버전스 감지
# ============================================================
def detect_divergence(points, direction='bullish', lookback=10):
    """
    다이버전스 감지
    - bullish: Price LL, RSI HL (하락 시작부터)
    - bearish: Price HH, RSI LH (상승 시작부터)
    """
    if direction == 'bullish':
        lows = [p for p in points if p['type'] == 'L']
        if len(lows) < 2:
            return None
        
        for i in range(len(lows) - 1, max(0, len(lows) - lookback), -1):
            l2 = lows[i]
            for j in range(i-1, max(-1, i - lookback), -1):
                l1 = lows[j]
                # Price: LL, RSI: HL
                if l2['price'] <= l1['price'] * 1.01 and l2['rsi'] > l1['rsi'] + 3:
                    return {
                        'type': 'bullish',
                        'start_price': l1['price'], 'start_rsi': l1['rsi'], 'start_time': l1['time'],
                        'end_price': l2['price'], 'end_rsi': l2['rsi'], 'end_time': l2['time']
                    }
        return None
    
    else:  # bearish
        highs = [p for p in points if p['type'] == 'H']
        if len(highs) < 2:
            return None
        
        for i in range(len(highs) - 1, max(0, len(highs) - lookback), -1):
            h2 = highs[i]
            for j in range(i-1, max(-1, i - lookback), -1):
                h1 = highs[j]
                # Price: HH, RSI: LH
                if h2['price'] >= h1['price'] * 0.99 and h2['rsi'] < h1['rsi'] - 3:
                    return {
                        'type': 'bearish',
                        'start_price': h1['price'], 'start_rsi': h1['rsi'], 'start_time': h1['time'],
                        'end_price': h2['price'], 'end_rsi': h2['rsi'], 'end_time': h2['time']
                    }
        return None

# ============================================================
# LONG 진입 신호 (W 패턴)
# ============================================================
def detect_long_entry_signals(df_15m, swing_highs, swing_lows, l2_time, l2_price, h_price, l1_price):
    """
    롱 진입 신호 감지 (최적화)
    핵심: Entry-L Gap 1~3% 범위 내 진입
    """
    search_start = l2_time - pd.Timedelta(hours=6)
    search_end = l2_time + pd.Timedelta(hours=48)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 20:
        return None
    
    window_highs = [h for h in swing_highs if search_start <= h['time'] <= search_end]
    window_lows = [l for l in swing_lows if search_start <= l['time'] <= search_end]
    
    signals = {
        'trendline_break': False, 'trendline_break_time': None, 'trendline_break_price': None,
        'lh_to_hh': False, 'lh_to_hh_time': None,
        'higher_low': False, 'higher_low_time': None,
        'bullish_volume': False, 'bullish_volume_time': None,
        'entry_time': None, 'entry_price': None,
        'signal_count': 0
    }
    
    sl_price = min(l1_price, l2_price)
    entry_candidates = []
    
    # 1. 하락추세선 돌파
    if len(window_highs) >= 2:
        for i in range(len(window_highs) - 1):
            h1, h2 = window_highs[i], window_highs[i + 1]
            if h2['price'] < h1['price'] * 0.998 and h2['time'] > h1['time']:
                h2_idx = h2['idx']
                after_h2 = df_15m[(df_15m.index > h2_idx) & (df_15m['datetime'] <= search_end)]
                
                time_diff = max(1, h2['idx'] - h1['idx'])
                slope = (h2['price'] - h1['price']) / time_diff
                
                for idx, row in after_h2.iterrows():
                    bars_from_h2 = idx - h2_idx
                    trendline_price = h2['price'] + slope * bars_from_h2
                    
                    if row['close'] > trendline_price * 1.001:
                        entry_l_gap = (row['close'] - sl_price) / sl_price * 100
                        if 0.5 <= entry_l_gap <= 4:
                            signals['trendline_break'] = True
                            signals['trendline_break_time'] = row['datetime']
                            signals['trendline_break_price'] = row['close']
                            entry_candidates.append({
                                'time': row['datetime'],
                                'price': row['close'],
                                'signal': 'trendline_break',
                                'score': 3,
                                'entry_l_gap': entry_l_gap
                            })
                        break
                
                if signals['trendline_break']:
                    break
    
    # 2. LH → HH 전환
    if len(window_highs) >= 3:
        for i in range(len(window_highs) - 2):
            h1, h2, h3 = window_highs[i], window_highs[i+1], window_highs[i+2]
            if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                # h3 시점에서의 가격으로 진입
                h3_idx = h3['idx']
                if h3_idx < len(df_15m):
                    entry_price = df_15m.iloc[h3_idx]['close']
                    entry_l_gap = (entry_price - sl_price) / sl_price * 100
                    if 0.5 <= entry_l_gap <= 4:
                        signals['lh_to_hh'] = True
                        signals['lh_to_hh_time'] = h3['time']
                        entry_candidates.append({
                            'time': h3['time'],
                            'price': entry_price,
                            'signal': 'lh_to_hh',
                            'score': 2,
                            'entry_l_gap': entry_l_gap
                        })
                break
    
    # 3. HL (저점 올림)
    if len(window_lows) >= 3:
        for i in range(len(window_lows) - 2):
            l1_15m, l2_15m, l3_15m = window_lows[i], window_lows[i+1], window_lows[i+2]
            if l2_15m['price'] < l1_15m['price'] and l3_15m['price'] > l2_15m['price']:
                l3_idx = l3_15m['idx']
                if l3_idx < len(df_15m):
                    entry_price = df_15m.iloc[l3_idx]['close']
                    entry_l_gap = (entry_price - sl_price) / sl_price * 100
                    if 0.5 <= entry_l_gap <= 4:
                        signals['higher_low'] = True
                        signals['higher_low_time'] = l3_15m['time']
                        entry_candidates.append({
                            'time': l3_15m['time'],
                            'price': entry_price,
                            'signal': 'higher_low',
                            'score': 2,
                            'entry_l_gap': entry_l_gap
                        })
                break
    
    # 4. 양봉 + 거래량
    for idx, row in df_window.iterrows():
        is_bullish = row['close'] > row['open']
        body_size = abs(row['close'] - row['open']) / row['open'] * 100
        near_l2 = row['low'] <= l2_price * 1.02
        high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5
        
        if is_bullish and body_size > 0.3 and high_volume and near_l2:
            entry_l_gap = (row['close'] - sl_price) / sl_price * 100
            if 0.5 <= entry_l_gap <= 4:
                signals['bullish_volume'] = True
                signals['bullish_volume_time'] = row['datetime']
                entry_candidates.append({
                    'time': row['datetime'],
                    'price': row['close'],
                    'signal': 'bullish_volume',
                    'score': 1,
                    'entry_l_gap': entry_l_gap
                })
            break
    
    # 신호 개수
    signal_count = sum([signals['trendline_break'], signals['lh_to_hh'], 
                        signals['higher_low'], signals['bullish_volume']])
    signals['signal_count'] = signal_count
    
    # 최적 진입점 선택 (Entry-L Gap 1~3% 선호)
    if len(entry_candidates) > 0:
        # 1-3% 범위 내 우선 선택
        optimal_entries = [c for c in entry_candidates if 1 <= c['entry_l_gap'] <= 3]
        if len(optimal_entries) > 0:
            optimal_entries.sort(key=lambda x: (-x['score'], x['time']))
            best = optimal_entries[0]
        else:
            entry_candidates.sort(key=lambda x: (-x['score'], x['time']))
            best = entry_candidates[0]
        
        signals['entry_time'] = best['time']
        signals['entry_price'] = best['price']
        signals['entry_l_gap'] = best['entry_l_gap']
    
    return signals

# ============================================================
# SHORT 진입 신호 (M 패턴)
# ============================================================
def detect_short_entry_signals(df_15m, swing_highs, swing_lows, h2_time, h2_price, l_neckline_price, h1_price):
    """
    숏 진입 신호 감지 (W 패턴 로직 미러링)
    핵심: Entry-H Gap 1~3% 범위 내 진입
    """
    search_start = h2_time - pd.Timedelta(hours=6)
    search_end = h2_time + pd.Timedelta(hours=48)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 20:
        return None
    
    window_highs = [h for h in swing_highs if search_start <= h['time'] <= search_end]
    window_lows = [l for l in swing_lows if search_start <= l['time'] <= search_end]
    
    signals = {
        'trendline_break': False, 'trendline_break_time': None, 'trendline_break_price': None,
        'hl_to_ll': False, 'hl_to_ll_time': None,
        'lower_high': False, 'lower_high_time': None,
        'bearish_volume': False, 'bearish_volume_time': None,
        'entry_time': None, 'entry_price': None,
        'signal_count': 0
    }
    
    sl_price = max(h1_price, h2_price)  # SL은 고점
    entry_candidates = []
    
    # 1. 상승추세선 하향 돌파
    if len(window_lows) >= 2:
        for i in range(len(window_lows) - 1):
            l1, l2 = window_lows[i], window_lows[i + 1]
            if l2['price'] > l1['price'] * 1.002 and l2['time'] > l1['time']:  # 상승 추세선
                l2_idx = l2['idx']
                after_l2 = df_15m[(df_15m.index > l2_idx) & (df_15m['datetime'] <= search_end)]
                
                time_diff = max(1, l2['idx'] - l1['idx'])
                slope = (l2['price'] - l1['price']) / time_diff
                
                for idx, row in after_l2.iterrows():
                    bars_from_l2 = idx - l2_idx
                    trendline_price = l2['price'] + slope * bars_from_l2
                    
                    if row['close'] < trendline_price * 0.999:
                        entry_h_gap = (sl_price - row['close']) / row['close'] * 100
                        if 0.5 <= entry_h_gap <= 4:
                            signals['trendline_break'] = True
                            signals['trendline_break_time'] = row['datetime']
                            signals['trendline_break_price'] = row['close']
                            entry_candidates.append({
                                'time': row['datetime'],
                                'price': row['close'],
                                'signal': 'trendline_break',
                                'score': 3,
                                'entry_h_gap': entry_h_gap
                            })
                        break
                
                if signals['trendline_break']:
                    break
    
    # 2. HL → LL 전환 (LH→HH의 미러)
    if len(window_lows) >= 3:
        for i in range(len(window_lows) - 2):
            l1_15m, l2_15m, l3_15m = window_lows[i], window_lows[i+1], window_lows[i+2]
            if l2_15m['price'] > l1_15m['price'] and l3_15m['price'] < l2_15m['price']:
                l3_idx = l3_15m['idx']
                if l3_idx < len(df_15m):
                    entry_price = df_15m.iloc[l3_idx]['close']
                    entry_h_gap = (sl_price - entry_price) / entry_price * 100
                    if 0.5 <= entry_h_gap <= 4:
                        signals['hl_to_ll'] = True
                        signals['hl_to_ll_time'] = l3_15m['time']
                        entry_candidates.append({
                            'time': l3_15m['time'],
                            'price': entry_price,
                            'signal': 'hl_to_ll',
                            'score': 2,
                            'entry_h_gap': entry_h_gap
                        })
                break
    
    # 3. LH (고점 낮춤) - HL의 미러
    if len(window_highs) >= 3:
        for i in range(len(window_highs) - 2):
            h1_15m, h2_15m, h3_15m = window_highs[i], window_highs[i+1], window_highs[i+2]
            if h2_15m['price'] > h1_15m['price'] and h3_15m['price'] < h2_15m['price']:
                h3_idx = h3_15m['idx']
                if h3_idx < len(df_15m):
                    entry_price = df_15m.iloc[h3_idx]['close']
                    entry_h_gap = (sl_price - entry_price) / entry_price * 100
                    if 0.5 <= entry_h_gap <= 4:
                        signals['lower_high'] = True
                        signals['lower_high_time'] = h3_15m['time']
                        entry_candidates.append({
                            'time': h3_15m['time'],
                            'price': entry_price,
                            'signal': 'lower_high',
                            'score': 2,
                            'entry_h_gap': entry_h_gap
                        })
                break
    
    # 4. 음봉 + 거래량
    for idx, row in df_window.iterrows():
        is_bearish = row['close'] < row['open']
        body_size = abs(row['close'] - row['open']) / row['open'] * 100
        near_h2 = row['high'] >= h2_price * 0.98
        high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5
        
        if is_bearish and body_size > 0.3 and high_volume and near_h2:
            entry_h_gap = (sl_price - row['close']) / row['close'] * 100
            if 0.5 <= entry_h_gap <= 4:
                signals['bearish_volume'] = True
                signals['bearish_volume_time'] = row['datetime']
                entry_candidates.append({
                    'time': row['datetime'],
                    'price': row['close'],
                    'signal': 'bearish_volume',
                    'score': 1,
                    'entry_h_gap': entry_h_gap
                })
            break
    
    # 신호 개수
    signal_count = sum([signals['trendline_break'], signals['hl_to_ll'],
                        signals['lower_high'], signals['bearish_volume']])
    signals['signal_count'] = signal_count
    
    # 최적 진입점 선택 (Entry-H Gap 1~3% 선호)
    if len(entry_candidates) > 0:
        optimal_entries = [c for c in entry_candidates if 1 <= c['entry_h_gap'] <= 3]
        if len(optimal_entries) > 0:
            optimal_entries.sort(key=lambda x: (-x['score'], x['time']))
            best = optimal_entries[0]
        else:
            entry_candidates.sort(key=lambda x: (-x['score'], x['time']))
            best = entry_candidates[0]
        
        signals['entry_time'] = best['time']
        signals['entry_price'] = best['price']
        signals['entry_h_gap'] = best['entry_h_gap']
    
    return signals

# ============================================================
# 백테스트 실행
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행")
print("=" * 80)

results = []
tp_pct = 5  # Fixed TP

# LONG - W 패턴 (L1-H-L2)
print("\nLONG 패턴 스캔...")
long_count = 0
for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴: L2 <= L1 * 1.03
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    # Gap (넥라인-L2)
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 2:
        continue
    
    # 1H 다이버전스 체크
    points_before_l2 = [p for p in points_1h[:i+3]]
    div_1h = detect_divergence(points_before_l2, 'bullish', lookback=10)
    has_1h_div = div_1h is not None
    
    # 15M 신호 감지
    signals = detect_long_entry_signals(df_15m, swing_highs_15m, swing_lows_15m, 
                                         L2['time'], L2['price'], H['price'], L1['price'])
    
    if signals is None or signals['signal_count'] < 1:
        continue
    
    if signals['entry_time'] is None:
        continue
    
    entry_time = signals['entry_time']
    entry_price = signals['entry_price']
    entry_l_gap = signals.get('entry_l_gap', 0)
    
    # Entry-L Gap 필터
    if entry_l_gap < 0.5 or entry_l_gap > 4:
        continue
    
    # SL & TP
    sl_price = min(L1['price'], L2['price'])
    tp_price = entry_price * (1 + tp_pct / 100)
    
    # 15M 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    mae = 0
    
    for j, (_, row) in enumerate(post.iterrows()):
        current_pnl = (row['close'] - entry_price) / entry_price * 100
        mfe = max(mfe, (row['high'] - entry_price) / entry_price * 100)
        mae = min(mae, (row['low'] - entry_price) / entry_price * 100)
        
        if row['high'] >= tp_price:
            pnl = tp_pct
            exit_type = 'TP'
            break
        if row['low'] <= sl_price:
            pnl = (sl_price - entry_price) / entry_price * 100
            exit_type = 'SL'
            break
    
    long_count += 1
    results.append({
        'time': entry_time,
        'direction': 'LONG',
        'pattern': 'W',
        'entry': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'entry_gap': entry_l_gap,  # Entry-L Gap
        'gap': gap,
        'l_ratio': l_ratio,
        'has_1h_div': has_1h_div,
        'signal_count': signals['signal_count'],
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals['lh_to_hh'],
        'higher_low': signals['higher_low'],
        'bullish_volume': signals['bullish_volume'],
        'pnl': pnl,
        'mfe': mfe,
        'mae': mae,
        'exit_type': exit_type
    })

print(f"LONG 신호: {long_count}건")

# SHORT - M 패턴 (H1-L-H2)
print("\nSHORT 패턴 스캔...")
short_count = 0
for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'H' and 
            points_1h[i+1]['type'] == 'L' and 
            points_1h[i+2]['type'] == 'H'):
        continue
    
    H1 = points_1h[i]
    L = points_1h[i+1]
    H2 = points_1h[i+2]
    
    # M 패턴: H2 >= H1 * 0.97
    h_ratio = (H2['price'] - H1['price']) / H1['price'] * 100
    if h_ratio < -3:  # H2가 H1보다 3% 이상 낮으면 스킵
        continue
    
    # Gap (H2-넥라인)
    gap = (H2['price'] - L['price']) / L['price'] * 100
    if gap < 2:
        continue
    
    # 1H 다이버전스 체크
    points_before_h2 = [p for p in points_1h[:i+3]]
    div_1h = detect_divergence(points_before_h2, 'bearish', lookback=10)
    has_1h_div = div_1h is not None
    
    # 15M 신호 감지
    signals = detect_short_entry_signals(df_15m, swing_highs_15m, swing_lows_15m,
                                          H2['time'], H2['price'], L['price'], H1['price'])
    
    if signals is None or signals['signal_count'] < 1:
        continue
    
    if signals['entry_time'] is None:
        continue
    
    entry_time = signals['entry_time']
    entry_price = signals['entry_price']
    entry_h_gap = signals.get('entry_h_gap', 0)
    
    # Entry-H Gap 필터
    if entry_h_gap < 0.5 or entry_h_gap > 4:
        continue
    
    # SL & TP
    sl_price = max(H1['price'], H2['price'])
    tp_price = entry_price * (1 - tp_pct / 100)
    
    # 15M 백테스트
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    mfe = 0
    mae = 0
    
    for j, (_, row) in enumerate(post.iterrows()):
        current_pnl = (entry_price - row['close']) / entry_price * 100  # SHORT
        mfe = max(mfe, (entry_price - row['low']) / entry_price * 100)
        mae = min(mae, (entry_price - row['high']) / entry_price * 100)
        
        if row['low'] <= tp_price:
            pnl = tp_pct
            exit_type = 'TP'
            break
        if row['high'] >= sl_price:
            pnl = (entry_price - sl_price) / entry_price * 100  # 음수
            exit_type = 'SL'
            break
    
    short_count += 1
    results.append({
        'time': entry_time,
        'direction': 'SHORT',
        'pattern': 'M',
        'entry': entry_price,
        'sl': sl_price,
        'tp': tp_price,
        'entry_gap': entry_h_gap,  # Entry-H Gap
        'gap': gap,
        'l_ratio': h_ratio,  # h_ratio for M pattern
        'has_1h_div': has_1h_div,
        'signal_count': signals['signal_count'],
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals.get('hl_to_ll', False),  # HL→LL for SHORT
        'higher_low': signals.get('lower_high', False),  # LH for SHORT
        'bullish_volume': signals.get('bearish_volume', False),
        'pnl': pnl,
        'mfe': mfe,
        'mae': mae,
        'exit_type': exit_type
    })

print(f"SHORT 신호: {short_count}건")

# ============================================================
# 결과 분석
# ============================================================
df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

if len(df_results) == 0:
    print("신호가 없습니다.")
    exit()

# 전체 성과
print("\n" + "=" * 80)
print("전체 성과")
print("=" * 80)

total_win = (df_results['pnl'] > 0).sum()
total_loss = (df_results['pnl'] < 0).sum()
win_rate = total_win / len(df_results) * 100
avg_pnl = df_results['pnl'].mean()
total_pnl = df_results['pnl'].sum()

print(f"  총 거래: {len(df_results)}건")
print(f"  승/패: {total_win}/{total_loss}")
print(f"  승률: {win_rate:.1f}%")
print(f"  평균 PnL: {avg_pnl:.2f}%")
print(f"  총 PnL: {total_pnl:.1f}%")

# 방향별 성과
print("\n" + "=" * 80)
print("방향별 성과")
print("=" * 80)

for direction in ['LONG', 'SHORT']:
    subset = df_results[df_results['direction'] == direction]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        print(f"\n[{direction}] {len(subset)}건, 승률: {wr:.1f}%, 평균 PnL: {ap:.2f}%, 총 PnL: {tp:.1f}%")

# Entry Gap 범위별 성과
print("\n" + "=" * 80)
print("Entry Gap 범위별 성과 (핵심)")
print("=" * 80)

print(f"\n{'Entry Gap':>15} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 60)

gap_bins = [(0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4)]
for low, high in gap_bins:
    subset = df_results[(df_results['entry_gap'] >= low) & (df_results['entry_gap'] < high)]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>15} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# Gap(에너지) 범위별 성과
print("\n" + "=" * 80)
print("Gap(에너지) 범위별 성과")
print("=" * 80)

print(f"\n{'Gap':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 55)

for low, high in [(2, 4), (4, 6), (6, 8), (8, 100)]:
    subset = df_results[(df_results['gap'] >= low) & (df_results['gap'] < high)]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{f'{low}-{high}%':>10} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# 1H 다이버전스 유무별 성과
print("\n" + "=" * 80)
print("1H 다이버전스 유무별 성과")
print("=" * 80)

for has_div in [True, False]:
    subset = df_results[df_results['has_1h_div'] == has_div]
    if len(subset) > 0:
        wr = (subset['pnl'] > 0).mean() * 100
        ap = subset['pnl'].mean()
        tp = subset['pnl'].sum()
        status = "있음" if has_div else "없음"
        print(f"  1H 다이버전스 {status}: {len(subset)}건, 승률: {wr:.1f}%, 평균 PnL: {ap:.2f}%, 총 PnL: {tp:.1f}%")

# 최적 조건 탐색
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>40} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 85)

conditions = [
    ('전체', df_results),
    ('LONG만', df_results[df_results['direction'] == 'LONG']),
    ('SHORT만', df_results[df_results['direction'] == 'SHORT']),
    ('Entry Gap 1-3%', df_results[(df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('Gap>=4% + Entry Gap 1-3%', df_results[(df_results['gap'] >= 4) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('Gap>=5% + Entry Gap 1-3%', df_results[(df_results['gap'] >= 5) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('1H 다이버전스 + Entry Gap 1-3%', df_results[(df_results['has_1h_div'] == True) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('Gap>=4% + 1H 다이버전스', df_results[(df_results['gap'] >= 4) & (df_results['has_1h_div'] == True)]),
    ('추세돌파 + Entry Gap 1-3%', df_results[(df_results['trendline_break'] == True) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('LONG + Gap>=4% + Entry Gap 1-3%', df_results[(df_results['direction'] == 'LONG') & (df_results['gap'] >= 4) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
    ('SHORT + Gap>=4% + Entry Gap 1-3%', df_results[(df_results['direction'] == 'SHORT') & (df_results['gap'] >= 4) & (df_results['entry_gap'] >= 1) & (df_results['entry_gap'] < 3)]),
]

best_condition = None
best_metric = -999

for name, subset in conditions:
    if len(subset) >= 5:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        metric = win_rate * avg_pnl / 100
        print(f"{name:>40} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")
        
        if metric > best_metric and len(subset) >= 20:
            best_metric = metric
            best_condition = (name, subset)

# 월별 성과
print("\n" + "=" * 80)
print("월별 성과")
print("=" * 80)

df_results['month'] = pd.to_datetime(df_results['time']).dt.to_period('M')
monthly = df_results.groupby('month').agg({
    'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
}).round(2)
monthly.columns = ['총PnL', '거래수', '승률']

print(f"\n  총 거래: {len(df_results)}건")
print(f"  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['총PnL'].mean():.2f}%")
print(f"  3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월")
print(f"  수익 월: {(monthly['총PnL'] > 0).sum()}개월 / {len(monthly)}개월 ({(monthly['총PnL'] > 0).mean()*100:.1f}%)")
print(f"  최대 월 수익: {monthly['총PnL'].max():.2f}%")
print(f"  최대 월 손실: {monthly['총PnL'].min():.2f}%")

# 최적 조건 월별 성과
if best_condition is not None:
    best_name, best_subset = best_condition
    print(f"\n[최적 조건: {best_name}]")
    
    best_subset = best_subset.copy()
    best_subset['month'] = pd.to_datetime(best_subset['time']).dt.to_period('M')
    best_monthly = best_subset.groupby('month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
    }).round(2)
    best_monthly.columns = ['총PnL', '거래수', '승률']
    
    print(f"  총 거래: {len(best_subset)}건")
    print(f"  평균 거래수: {best_monthly['거래수'].mean():.1f}건/월")
    print(f"  월 평균 수익: {best_monthly['총PnL'].mean():.2f}%")
    print(f"  3x 레버리지: {best_monthly['총PnL'].mean()*3:.1f}%/월")
    print(f"  수익 월: {(best_monthly['총PnL'] > 0).sum()}개월 / {len(best_monthly)}개월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론: 최적화된 MTF 롱/숏 통합 전략")
print("=" * 80)

best_avg_monthly = best_monthly['총PnL'].mean() if best_condition else monthly['총PnL'].mean()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    최적화된 MTF 롱/숏 통합 전략                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 총 신호: {len(df_results)}건 (LONG: {len(df_results[df_results['direction']=='LONG'])}건, SHORT: {len(df_results[df_results['direction']=='SHORT'])}건)
│                                                                            │
│  ■ 전체 성과:                                                               │
│    - 승률: {(df_results['pnl']>0).mean()*100:.1f}%                                                          │
│    - 평균 PnL: {df_results['pnl'].mean():.2f}%                                                     │
│    - 월 평균: {monthly['총PnL'].mean():.2f}%                                                       │
│    - 3x 레버리지: {monthly['총PnL'].mean()*3:.1f}%/월                                             │
│                                                                            │
│  ■ 최적 조건: {best_name if best_condition else 'N/A':48}│
│    - 월 평균: {best_avg_monthly:.2f}%                                                       │
│    - 3x 레버리지: {best_avg_monthly*3:.1f}%/월                                             │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  ■ 핵심 원칙 (사용자):                                                       │
│    "하락추세 시작 ~ 추세돌파까지 전체를 봐야 함"                               │
│    "1H 큰 그림 + 15M 진입 타이밍 = MTF 일치해야 진짜"                          │
│    "Breakout → Adjustment → 1차 → 2차"                                     │
│                                                                            │
│  ■ 진입 체크리스트:                                                          │
│    □ 1H: W(LONG)/M(SHORT) 패턴 확인                                        │
│    □ Gap >= 4% (에너지 축적)                                                │
│    □ 1H 다이버전스 확인 (Price LL + RSI HL / Price HH + RSI LH)             │
│    □ 15M: 추세선 돌파 / HH(LL) 전환 / HL(LH) / 양(음)봉+거래량               │
│    □ Entry Gap 1~3% (적정 손절폭)                                           │
│    □ SL = min(L1,L2) 또는 max(H1,H2)                                        │
│    □ TP = 5%                                                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_optimized_long_short_results.csv', index=False)
print("\n결과 저장: mtf_optimized_long_short_results.csv")
