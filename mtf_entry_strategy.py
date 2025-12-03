#!/usr/bin/env python3
"""
MTF (Multi-Time Frame) 진입 전략

1H: W 패턴 형성 중 (큰 그림)
15M: 진입 타이밍 (추세 깨는 신호)
- 하락추세선 돌파
- LH → HH 전환  
- HL (저점 올림)
- 양봉 + 거래량
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 진입 전략: 1H 패턴 + 15M 진입 신호")
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

# MACD for 15M
exp1_15 = df_15m['close'].ewm(span=12, adjust=False).mean()
exp2_15 = df_15m['close'].ewm(span=26, adjust=False).mean()
df_15m['macd'] = exp1_15 - exp2_15
df_15m['signal'] = df_15m['macd'].ewm(span=9, adjust=False).mean()
df_15m['hist'] = df_15m['macd'] - df_15m['signal']

# 1H H/L 추출
def extract_hl_points(df, price_col_high='high', price_col_low='low'):
    hist = df['hist'].values
    high = df[price_col_high].values
    low = df[price_col_low].values
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
points_15m = extract_hl_points(df_15m)

print(f"1H H/L points: {len(points_1h)}")
print(f"15M H/L points: {len(points_15m)}")

# ============================================================
# 15M 진입 신호 감지 함수들
# ============================================================

def detect_15m_entry_signals(df_15m, points_15m, start_time, end_time, l2_price):
    """
    15M에서 진입 신호 감지
    - 하락추세선 돌파
    - LH → HH 전환
    - HL (저점 올림)
    - 양봉 + 거래량
    """
    # 해당 시간 구간의 15M 데이터
    mask = (df_15m['datetime'] >= start_time) & (df_15m['datetime'] <= end_time)
    df_window = df_15m[mask].copy()
    
    if len(df_window) < 10:
        return None
    
    # 해당 구간의 15M H/L 포인트
    window_points = [p for p in points_15m 
                     if start_time <= p['time'] <= end_time]
    
    if len(window_points) < 3:
        return None
    
    signals = {
        'trendline_break': False,
        'lh_to_hh': False,
        'higher_low': False,
        'bullish_candle_volume': False,
        'entry_time': None,
        'entry_price': None,
        'entry_idx': None
    }
    
    # 1. 하락추세선 돌파 체크
    # 최근 2개의 H 포인트로 추세선 그리기
    h_points = [p for p in window_points if p['type'] == 'H']
    if len(h_points) >= 2:
        h1, h2 = h_points[-2], h_points[-1]
        if h2['price'] < h1['price']:  # 하락추세
            # h2 이후 데이터에서 추세선 돌파 확인
            h2_idx = df_window[df_window['datetime'] == h2['time']].index
            if len(h2_idx) > 0:
                h2_idx = h2_idx[0]
                after_h2 = df_window[df_window.index > h2_idx]
                
                for idx, row in after_h2.iterrows():
                    # 단순화: h2 가격 돌파
                    if row['close'] > h2['price']:
                        signals['trendline_break'] = True
                        signals['entry_time'] = row['datetime']
                        signals['entry_price'] = row['close']
                        signals['entry_idx'] = idx
                        break
    
    # 2. LH → HH 전환 체크
    if len(h_points) >= 2:
        h1, h2 = h_points[-2], h_points[-1]
        if h2['price'] > h1['price']:  # HH 형성
            signals['lh_to_hh'] = True
            if signals['entry_time'] is None:
                # H2 형성 시점을 진입으로
                h2_row = df_window[df_window['datetime'] == h2['time']]
                if len(h2_row) > 0:
                    signals['entry_time'] = h2['time']
                    signals['entry_price'] = h2['price']
    
    # 3. HL (저점 올림) 체크
    l_points = [p for p in window_points if p['type'] == 'L']
    if len(l_points) >= 2:
        l1, l2 = l_points[-2], l_points[-1]
        if l2['price'] > l1['price']:  # HL 형성
            signals['higher_low'] = True
    
    # 4. 양봉 + 거래량 체크
    # L2 근처에서 양봉 + 평균 이상 거래량
    avg_vol = df_window['volume'].mean()
    recent = df_window.tail(5)
    
    for idx, row in recent.iterrows():
        is_bullish = row['close'] > row['open']
        high_volume = row['volume'] > avg_vol * 1.5
        near_l2 = abs(row['low'] - l2_price) / l2_price < 0.01  # 1% 이내
        
        if is_bullish and high_volume:
            signals['bullish_candle_volume'] = True
            if signals['entry_time'] is None:
                signals['entry_time'] = row['datetime']
                signals['entry_price'] = row['close']
    
    # 신호 개수
    signal_count = sum([
        signals['trendline_break'],
        signals['lh_to_hh'],
        signals['higher_low'],
        signals['bullish_candle_volume']
    ])
    signals['signal_count'] = signal_count
    
    return signals

# ============================================================
# 1H W 패턴 + 15M 진입 신호 백테스트
# ============================================================
print("\n" + "=" * 80)
print("MTF 백테스트: 1H W패턴 + 15M 진입신호")
print("=" * 80)

results = []

for i in range(len(points_1h) - 2):
    # 1H W 패턴 (L1 - H - L2)
    if not (points_1h[i]['type'] == 'L' and 
            points_1h[i+1]['type'] == 'H' and 
            points_1h[i+2]['type'] == 'L'):
        continue
    
    L1 = points_1h[i]
    H = points_1h[i+1]
    L2 = points_1h[i+2]
    
    # W 패턴 조건: L1 ≈ L2
    if abs(L2['price'] - L1['price']) / L1['price'] > 0.03:
        continue
    
    # Gap 조건
    gap = (H['price'] - L2['price']) / L2['price'] * 100
    if gap < 5:
        continue
    
    # 15M 진입 신호 감지 (L2 형성 시점 ~ L2 이후 24시간)
    start_time = L2['time'] - pd.Timedelta(hours=6)
    end_time = L2['time'] + pd.Timedelta(hours=24)
    
    signals = detect_15m_entry_signals(df_15m, points_15m, start_time, end_time, L2['price'])
    
    if signals is None:
        continue
    
    # 신호 개수에 따른 분류
    signal_count = signals['signal_count']
    
    # 진입 (넥라인 돌파 대기 or 15M 신호로 진입)
    if signals['entry_time'] is not None:
        entry_time = signals['entry_time']
        entry_price = signals['entry_price']
    else:
        # 신호 없으면 넥라인 돌파 대기
        l2_idx = L2['idx']
        future_1h = df_1h.iloc[l2_idx+1:l2_idx+50]
        breakout_idx = None
        for j, (_, row) in enumerate(future_1h.iterrows()):
            if row['close'] > H['price']:
                breakout_idx = l2_idx + 1 + j
                break
        if breakout_idx is None:
            continue
        entry_time = df_1h.iloc[breakout_idx]['datetime']
        entry_price = df_1h.iloc[breakout_idx]['close']
    
    # SL = L2
    sl_price = min(L1['price'], L2['price'])
    
    # 백테스트 (15M 기준)
    entry_15m_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_15m_idx) == 0:
        continue
    entry_15m_idx = entry_15m_idx[0]
    
    # TP 3.5% or SL
    tp_price = entry_price * 1.035
    post = df_15m.iloc[entry_15m_idx+1:entry_15m_idx+500]
    
    pnl = 0
    exit_type = 'TIMEOUT'
    
    for _, row in post.iterrows():
        if row['high'] >= tp_price:
            pnl = 3.5
            exit_type = 'TP'
            break
        if row['low'] <= sl_price:
            pnl = (sl_price - entry_price) / entry_price * 100
            exit_type = 'SL'
            break
    
    results.append({
        'time': entry_time,
        'entry': entry_price,
        'sl': sl_price,
        'gap': gap,
        'signal_count': signal_count,
        'trendline_break': signals['trendline_break'],
        'lh_to_hh': signals['lh_to_hh'],
        'higher_low': signals['higher_low'],
        'bullish_volume': signals['bullish_candle_volume'],
        'pnl': pnl,
        'exit_type': exit_type
    })

df_results = pd.DataFrame(results)
print(f"\n총 신호: {len(df_results)}건")

# ============================================================
# 신호 개수별 성과
# ============================================================
print("\n" + "=" * 80)
print("15M 신호 개수별 성과")
print("=" * 80)

print(f"\n{'신호개수':>8} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10}")
print("-" * 50)

for sig_count in [0, 1, 2, 3, 4]:
    subset = df_results[df_results['signal_count'] == sig_count]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        total_pnl = subset['pnl'].sum()
        print(f"{sig_count:>8} {len(subset):>8} {win_rate:>10.1f} {avg_pnl:>10.2f} {total_pnl:>10.1f}")

# ============================================================
# 개별 신호별 성과
# ============================================================
print("\n" + "=" * 80)
print("개별 15M 신호별 성과")
print("=" * 80)

for signal_name in ['trendline_break', 'lh_to_hh', 'higher_low', 'bullish_volume']:
    subset = df_results[df_results[signal_name] == True]
    if len(subset) > 0:
        win_rate = (subset['pnl'] > 0).mean() * 100
        avg_pnl = subset['pnl'].mean()
        print(f"\n[{signal_name}]")
        print(f"  건수: {len(subset)}, 승률: {win_rate:.1f}%, 평균PnL: {avg_pnl:.2f}%")

# ============================================================
# MTF 일치 (2개 이상 신호) vs 불일치
# ============================================================
print("\n" + "=" * 80)
print("MTF 일치 vs 불일치")
print("=" * 80)

mtf_match = df_results[df_results['signal_count'] >= 2]
mtf_nomatch = df_results[df_results['signal_count'] < 2]

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    MTF 일치 vs 불일치 비교                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        MTF 불일치 (0-1)      MTF 일치 (2+)                 │
│  ──────────────────────────────────────────────────────────────────────── │
│  건수                  {len(mtf_nomatch):>10}건       {len(mtf_match):>10}건                     │
│  승률                  {(mtf_nomatch['pnl']>0).mean()*100 if len(mtf_nomatch)>0 else 0:>10.1f}%       {(mtf_match['pnl']>0).mean()*100 if len(mtf_match)>0 else 0:>10.1f}%                     │
│  평균 PnL             {mtf_nomatch['pnl'].mean() if len(mtf_nomatch)>0 else 0:>10.2f}%       {mtf_match['pnl'].mean() if len(mtf_match)>0 else 0:>10.2f}%                     │
│  총 PnL               {mtf_nomatch['pnl'].sum() if len(mtf_nomatch)>0 else 0:>10.1f}%       {mtf_match['pnl'].sum() if len(mtf_match)>0 else 0:>10.1f}%                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 월별 성과
if len(mtf_match) > 0:
    mtf_match = mtf_match.copy()
    mtf_match['month'] = pd.to_datetime(mtf_match['time']).dt.to_period('M')
    monthly = mtf_match.groupby('month')['pnl'].agg(['sum', 'count'])
    
    print(f"\n[MTF 일치 월별 성과]")
    print(f"  평균 거래수: {monthly['count'].mean():.1f}건/월")
    print(f"  월 평균 수익: {monthly['sum'].mean():.2f}%")
    print(f"  손실 월: {(monthly['sum'] < 0).sum()}개월 / {len(monthly)}개월")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

if len(mtf_match) > 0:
    mtf_monthly = monthly['sum'].mean()
    mtf_winrate = (mtf_match['pnl'] > 0).mean() * 100
else:
    mtf_monthly = 0
    mtf_winrate = 0

nomatch_monthly = 0
if len(mtf_nomatch) > 0:
    mtf_nomatch_c = mtf_nomatch.copy()
    mtf_nomatch_c['month'] = pd.to_datetime(mtf_nomatch_c['time']).dt.to_period('M')
    nomatch_monthly = mtf_nomatch_c.groupby('month')['pnl'].sum().mean()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         MTF 전략 결론                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1H W패턴 단독:                                                            │
│  - 월 평균: ~2%                                                            │
│                                                                            │
│  1H W패턴 + 15M 신호 2개 이상 (MTF 일치):                                  │
│  - 승률: {mtf_winrate:.1f}%                                                         │
│  - 월 평균: {mtf_monthly:.2f}%                                                      │
│  - 3x 레버리지: {mtf_monthly*3:.1f}%/월                                             │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  핵심:                                                                     │
│  "1H 큰 그림 + 15M 진입 타이밍 = MTF 일치해야 진짜"                        │
│                                                                            │
│  15M 진입 신호:                                                            │
│  ✓ 하락추세선 돌파                                                         │
│  ✓ LH → HH 전환                                                            │
│  ✓ HL (저점 올림)                                                          │
│  ✓ 양봉 + 거래량                                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 결과 저장
df_results.to_csv('mtf_entry_results.csv', index=False)
print("\n결과 저장: mtf_entry_results.csv")
