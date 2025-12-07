#!/usr/bin/env python3
"""
최종 MTF 전략 백테스트

사용자 원칙:
1. 1H W (더블바텀) 형성 중이면:
2. 15M에서:
   - 하락추세선 돌파
   - 또는 LH → HH 전환
   - 또는 저점 올림 (HL)
   - 양봉 + 거래량
3. → 이게 나와야 1H W 완성되면서 상승

정리:
- 1H = 큰 그림 (W 패턴)
- 15M = 진입 타이밍 (추세 깨는 신호)
- MTF 일치해야 진짜

테스트 전략:
1. 고정 TP/SL
2. 트레일링 스탑
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("최종 MTF 전략 백테스트")
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

# MACD for 1H
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

# 볼륨 MA
df_15m['vol_ma20'] = df_15m['volume'].rolling(20).mean()

# H/L 추출
def extract_hl_points(df):
    hist = df['hist'].values
    high = df['high'].values
    low = df['low'].values
    timestamps = df['datetime'].values
    points = []
    i, n = 0, len(hist)
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0: i += 1
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': pd.Timestamp(timestamps[max_idx]), 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0: i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': pd.Timestamp(timestamps[min_idx]), 'idx': min_idx})
        else:
            i += 1
    return points

points_1h = extract_hl_points(df_1h)

# 15M 스윙 포인트
def get_15m_swing_points(df_15m, window=5):
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    times = df_15m['datetime'].values
    swing_highs, swing_lows = [], []
    for i in range(window, len(df_15m) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append({'price': highs[i], 'time': pd.Timestamp(times[i]), 'idx': i})
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append({'price': lows[i], 'time': pd.Timestamp(times[i]), 'idx': i})
    return swing_highs, swing_lows

swing_highs_15m, swing_lows_15m = get_15m_swing_points(df_15m)

print(f"1H H/L points: {len(points_1h)}")
print(f"15M Swing Highs: {len(swing_highs_15m)}, Swing Lows: {len(swing_lows_15m)}")

# 15M 진입 신호 감지
def detect_15m_entry_signals(df_15m, swing_highs, swing_lows, l2_time, l2_price, h_price, l1_price):
    search_start = l2_time - pd.Timedelta(hours=6)
    search_end = l2_time + pd.Timedelta(hours=24)
    
    mask = (df_15m['datetime'] >= search_start) & (df_15m['datetime'] <= search_end)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 20:
        return None
    
    window_highs = [h for h in swing_highs if search_start <= h['time'] <= search_end]
    window_lows = [l for l in swing_lows if search_start <= l['time'] <= search_end]
    
    signals = {
        'trendline_break': False,
        'lh_to_hh': False,
        'higher_low': False,
        'bullish_volume': False,
        'entry_time': None,
        'entry_price': None,
        'signal_count': 0
    }
    
    entry_candidates = []
    sl_price = min(l1_price, l2_price)
    
    # 1. 하락추세선 돌파
    if len(window_highs) >= 2:
        for i in range(len(window_highs) - 1):
            h1, h2 = window_highs[i], window_highs[i + 1]
            if h2['price'] < h1['price'] * 0.998:
                h2_idx = h2['idx']
                after_h2 = df_15m[(df_15m.index > h2_idx) & (df_15m['datetime'] <= search_end)]
                time_diff = h2['idx'] - h1['idx']
                if time_diff > 0:
                    slope = (h2['price'] - h1['price']) / time_diff
                    for idx, row in after_h2.iterrows():
                        bars_from_h2 = idx - h2_idx
                        trendline_price = h2['price'] + slope * bars_from_h2
                        if row['close'] > trendline_price * 1.001:
                            signals['trendline_break'] = True
                            entry_candidates.append({
                                'time': row['datetime'],
                                'price': row['close'],
                                'signal': 'trendline_break',
                                'score': 3
                            })
                            break
                if signals['trendline_break']:
                    break
    
    # 2. LH → HH 전환
    if len(window_highs) >= 3:
        for i in range(len(window_highs) - 2):
            h1, h2, h3 = window_highs[i], window_highs[i+1], window_highs[i+2]
            if h2['price'] < h1['price'] and h3['price'] > h2['price']:
                signals['lh_to_hh'] = True
                entry_candidates.append({
                    'time': h3['time'],
                    'price': h3['price'],
                    'signal': 'lh_to_hh',
                    'score': 2
                })
                break
    
    # 3. HL (저점 올림)
    if len(window_lows) >= 3:
        for i in range(len(window_lows) - 2):
            l1_15m, l2_15m, l3_15m = window_lows[i], window_lows[i+1], window_lows[i+2]
            if l2_15m['price'] < l1_15m['price'] and l3_15m['price'] > l2_15m['price']:
                signals['higher_low'] = True
                if l3_15m['idx'] < len(df_15m):
                    hl_candle = df_15m.iloc[l3_15m['idx']]
                    entry_candidates.append({
                        'time': l3_15m['time'],
                        'price': hl_candle['close'],
                        'signal': 'higher_low',
                        'score': 2
                    })
                break
    
    # 4. 양봉 + 거래량
    for idx, row in df_window.iterrows():
        is_bullish = row['close'] > row['open']
        body_size = abs(row['close'] - row['open']) / row['open'] * 100
        near_l2 = row['low'] <= l2_price * 1.02
        high_volume = pd.notna(row['vol_ma20']) and row['volume'] > row['vol_ma20'] * 1.5
        if is_bullish and body_size > 0.3 and high_volume and near_l2:
            signals['bullish_volume'] = True
            entry_candidates.append({
                'time': row['datetime'],
                'price': row['close'],
                'signal': 'bullish_volume',
                'score': 1
            })
            break
    
    signals['signal_count'] = sum([signals['trendline_break'], signals['lh_to_hh'], 
                                    signals['higher_low'], signals['bullish_volume']])
    
    # 최적 진입점 (Entry-L Gap 1~3% 범위)
    if len(entry_candidates) > 0:
        valid_entries = []
        for cand in entry_candidates:
            entry_l_gap = (cand['price'] - sl_price) / sl_price * 100
            if 0.5 <= entry_l_gap <= 4:
                cand['entry_l_gap'] = entry_l_gap
                valid_entries.append(cand)
        
        if len(valid_entries) > 0:
            valid_entries.sort(key=lambda x: (-x['score'], x['time']))
            best = valid_entries[0]
            signals['entry_time'] = best['time']
            signals['entry_price'] = best['price']
    
    return signals

# ============================================================
# 백테스트 (고정 TP vs 트레일링)
# ============================================================
print("\n" + "=" * 80)
print("백테스트 실행")
print("=" * 80)

def backtest_trade(df_15m, entry_idx, entry_price, sl_price, strategy='fixed_tp', 
                   tp_pct=3.5, trail_trigger=2.0, trail_pct=1.0):
    """
    트레이드 백테스트
    
    strategy:
    - 'fixed_tp': 고정 TP
    - 'trailing': 트레일링 스탑
    """
    post = df_15m.iloc[entry_idx+1:entry_idx+500]
    
    if strategy == 'fixed_tp':
        tp_price = entry_price * (1 + tp_pct / 100)
        
        for j, (_, row) in enumerate(post.iterrows()):
            if row['high'] >= tp_price:
                return {'pnl': tp_pct, 'exit_type': 'TP', 'mfe': (row['high'] - entry_price) / entry_price * 100}
            if row['low'] <= sl_price:
                pnl = (sl_price - entry_price) / entry_price * 100
                return {'pnl': pnl, 'exit_type': 'SL', 'mfe': (row['high'] - entry_price) / entry_price * 100}
        
        return {'pnl': 0, 'exit_type': 'TIMEOUT', 'mfe': 0}
    
    elif strategy == 'trailing':
        mfe = 0
        trailing_sl = None
        trailing_activated = False
        
        for j, (_, row) in enumerate(post.iterrows()):
            current_pnl = (row['high'] - entry_price) / entry_price * 100
            mfe = max(mfe, current_pnl)
            
            # 트레일링 활성화 체크
            if not trailing_activated and mfe >= trail_trigger:
                trailing_activated = True
            
            # 트레일링 SL 업데이트
            if trailing_activated:
                new_trail_sl = entry_price * (1 + (mfe - trail_pct) / 100)
                if trailing_sl is None or new_trail_sl > trailing_sl:
                    trailing_sl = new_trail_sl
            
            # SL 체크 (트레일링 또는 고정)
            active_sl = trailing_sl if trailing_sl is not None else sl_price
            
            if row['low'] <= active_sl:
                if trailing_activated:
                    pnl = (active_sl - entry_price) / entry_price * 100
                    return {'pnl': pnl, 'exit_type': 'TRAILING_SL', 'mfe': mfe}
                else:
                    pnl = (sl_price - entry_price) / entry_price * 100
                    return {'pnl': pnl, 'exit_type': 'INITIAL_SL', 'mfe': mfe}
        
        # 타임아웃 시 현재 가격으로 청산
        if len(post) > 0:
            last_close = post.iloc[-1]['close']
            pnl = (last_close - entry_price) / entry_price * 100
            return {'pnl': pnl, 'exit_type': 'TIMEOUT', 'mfe': mfe}
        return {'pnl': 0, 'exit_type': 'TIMEOUT', 'mfe': 0}

# 신호 수집
signals_data = []

for i in range(len(points_1h) - 2):
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    l_ratio = (L2['price'] - L1['price']) / L1['price'] * 100
    if l_ratio > 3:
        continue
    
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 4:  # Gap >= 4%
        continue
    
    signals = detect_15m_entry_signals(df_15m, swing_highs_15m, swing_lows_15m, 
                                       L2['time'], L2['price'], H['price'], L1['price'])
    
    if signals is None or signals['entry_time'] is None:
        continue
    
    if signals['signal_count'] < 1:
        continue
    
    entry_time = signals['entry_time']
    entry_price = signals['entry_price']
    sl_price = min(L1['price'], L2['price'])
    entry_l_gap = (entry_price - sl_price) / sl_price * 100
    
    # Entry-L Gap 1~3% 필터
    if not (1 <= entry_l_gap <= 3):
        continue
    
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    signals_data.append({
        'time': entry_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'entry_l_gap': entry_l_gap,
        'gap': gap,
        'entry_idx': entry_15m_idx,
        'signal_count': signals['signal_count'],
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals['lh_to_hh'],
        'higher_low': signals['higher_low'],
        'bullish_volume': signals['bullish_volume']
    })

print(f"\n총 신호 (Gap>=4%, Entry-L 1-3%): {len(signals_data)}건")

# ============================================================
# 전략 비교 백테스트
# ============================================================
strategies = [
    ('고정 TP 3.5%', 'fixed_tp', {'tp_pct': 3.5}),
    ('고정 TP 5%', 'fixed_tp', {'tp_pct': 5.0}),
    ('트레일링 (2%/1%)', 'trailing', {'trail_trigger': 2.0, 'trail_pct': 1.0}),
    ('트레일링 (3%/1.5%)', 'trailing', {'trail_trigger': 3.0, 'trail_pct': 1.5}),
    ('트레일링 (2%/0.5%)', 'trailing', {'trail_trigger': 2.0, 'trail_pct': 0.5}),
]

print("\n" + "=" * 80)
print("전략별 성과 비교")
print("=" * 80)

print(f"\n{'전략':>25} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10} {'월평균':>10}")
print("-" * 85)

strategy_results = {}

for name, strat_type, params in strategies:
    results = []
    
    for sig in signals_data:
        if strat_type == 'fixed_tp':
            result = backtest_trade(df_15m, sig['entry_idx'], sig['entry_price'], 
                                   sig['sl_price'], strategy='fixed_tp', tp_pct=params['tp_pct'])
        else:
            result = backtest_trade(df_15m, sig['entry_idx'], sig['entry_price'], 
                                   sig['sl_price'], strategy='trailing', 
                                   trail_trigger=params['trail_trigger'], 
                                   trail_pct=params['trail_pct'])
        
        results.append({
            'time': sig['time'],
            'entry_price': sig['entry_price'],
            'entry_l_gap': sig['entry_l_gap'],
            'gap': sig['gap'],
            'pnl': result['pnl'],
            'exit_type': result['exit_type'],
            'mfe': result['mfe']
        })
    
    df_r = pd.DataFrame(results)
    strategy_results[name] = df_r
    
    win_rate = (df_r['pnl'] > 0).mean() * 100
    avg_pnl = df_r['pnl'].mean()
    total_pnl = df_r['pnl'].sum()
    
    # 월별 수익
    df_r['month'] = pd.to_datetime(df_r['time']).dt.to_period('M')
    monthly_pnl = df_r.groupby('month')['pnl'].sum().mean()
    
    print(f"{name:>25} {len(df_r):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f} {monthly_pnl:>10.2f}")

# ============================================================
# 최적 전략 상세 분석
# ============================================================
print("\n" + "=" * 80)
print("최적 전략 상세 분석")
print("=" * 80)

# 가장 좋은 전략 찾기
best_strat = None
best_monthly = -999
for name, df_r in strategy_results.items():
    df_r['month'] = pd.to_datetime(df_r['time']).dt.to_period('M')
    monthly_pnl = df_r.groupby('month')['pnl'].sum().mean()
    if monthly_pnl > best_monthly:
        best_monthly = monthly_pnl
        best_strat = name

print(f"\n[최적 전략: {best_strat}]")
df_best = strategy_results[best_strat]
df_best['month'] = pd.to_datetime(df_best['time']).dt.to_period('M')

monthly = df_best.groupby('month').agg({
    'pnl': ['sum', 'count', lambda x: (x > 0).mean() * 100]
}).round(2)
monthly.columns = ['월PnL', '거래수', '월승률']

print(f"\n  총 거래: {len(df_best)}건")
print(f"  전체 승률: {(df_best['pnl'] > 0).mean() * 100:.1f}%")
print(f"  평균 PnL: {df_best['pnl'].mean():.2f}%")
print(f"  총 PnL: {df_best['pnl'].sum():.1f}%")
print(f"\n  평균 거래수: {monthly['거래수'].mean():.1f}건/월")
print(f"  월 평균 수익: {monthly['월PnL'].mean():.2f}%")
print(f"  수익 월: {(monthly['월PnL'] > 0).sum()}개월 / {len(monthly)}개월")
print(f"  손실 월: {(monthly['월PnL'] < 0).sum()}개월")
print(f"  최대 월 수익: {monthly['월PnL'].max():.2f}%")
print(f"  최대 월 손실: {monthly['월PnL'].min():.2f}%")

# 청산 유형별 분포
print(f"\n[청산 유형별 분포]")
for exit_type in df_best['exit_type'].unique():
    subset = df_best[df_best['exit_type'] == exit_type]
    print(f"  {exit_type}: {len(subset)}건 ({len(subset)/len(df_best)*100:.1f}%), 평균PnL: {subset['pnl'].mean():.2f}%")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론: MTF 전략 (1H W패턴 + 15M 진입)")
print("=" * 80)

best_win = (df_best['pnl'] > 0).mean() * 100
best_avg = df_best['pnl'].mean()
best_monthly_avg = monthly['월PnL'].mean()

print(f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            최종 MTF 전략 결과                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ■ 전략 조건:                                                                    │
│    1H: W 패턴 (L1-H-L2), Gap >= 4%                                               │
│    15M: 하락추세선 돌파 / LH→HH 전환 / HL 저점올림 / 양봉+거래량                 │
│    Entry-L Gap: 1 ~ 3%                                                           │
│                                                                                  │
│  ■ 최적 청산 전략: {best_strat:30}                                 │
│                                                                                  │
│  ■ 성과:                                                                         │
│    - 총 거래: {len(df_best)}건                                                            │
│    - 승률: {best_win:.1f}%                                                                │
│    - 평균 PnL: {best_avg:.2f}%                                                          │
│    - 월 평균: {best_monthly_avg:.2f}%                                                    │
│    - 3x 레버리지: {best_monthly_avg*3:.1f}%/월                                          │
│    - 수익 월 비율: {(monthly['월PnL']>0).sum()}/{len(monthly)} ({(monthly['월PnL']>0).sum()/len(monthly)*100:.1f}%)                                      │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 사용자 원칙 정리:                                                             │
│                                                                                  │
│    "1H W 패턴 형성 중이면:                                                       │
│     15M에서:                                                                     │
│     - 하락추세선 돌파                                                            │
│     - 또는 LH → HH 전환                                                          │
│     - 또는 저점 올림 (HL)                                                        │
│     - 양봉 + 거래량                                                              │
│     → 이게 나와야 1H W 완성되면서 상승"                                          │
│                                                                                  │
│    1H = 큰 그림 (W 패턴)                                                         │
│    15M = 진입 타이밍 (추세 깨는 신호)                                            │
│    MTF 일치해야 진짜                                                             │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────── │
│                                                                                  │
│  ■ 실전 체크리스트:                                                              │
│    ✓ 1H에서 W 패턴 형성 확인 (L1-H-L2)                                           │
│    ✓ Gap >= 4% (에너지 축적 확인)                                                │
│    ✓ 15M에서 진입 신호 대기 (추세돌파/HH전환/HL/양봉거래량)                      │
│    ✓ Entry-L Gap 1~3% 확인 (적정 손절폭)                                         │
│    ✓ SL = min(L1, L2) 설정                                                       │
│    ✓ 트레일링 스탑 적용 (수익 보호)                                              │
│                                                                                  │
│  ■ 15% 월 목표 달성:                                                             │
│    - 기본 수익: {best_monthly_avg:.2f}%/월                                              │
│    - 3x 레버리지: {best_monthly_avg*3:.1f}%/월                                          │
│    - 목표 대비: {best_monthly_avg*3/15*100:.1f}% 달성                                     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_best.to_csv('mtf_final_strategy_results.csv', index=False)
print("\n결과 저장: mtf_final_strategy_results.csv")

# ============================================================
# Gap 세분화 분석
# ============================================================
print("\n" + "=" * 80)
print("Gap 세분화 성과")
print("=" * 80)

print(f"\n{'Gap':>10} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'월평균':>10}")
print("-" * 55)

for low, high in [(4, 5), (5, 6), (6, 8), (8, 12)]:
    subset = df_best[(df_best['gap'] >= low) & (df_best['gap'] < high)]
    if len(subset) >= 5:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        subset['month'] = pd.to_datetime(subset['time']).dt.to_period('M')
        monthly_avg = subset.groupby('month')['pnl'].sum().mean()
        print(f"{f'{low}-{high}%':>10} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {monthly_avg:>10.2f}")
