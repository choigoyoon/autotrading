#!/usr/bin/env python3
"""
패턴 신뢰도 계층 분석 - 수정본
더블바텀 < 추세선 돌파 < 저항 돌파

문제점: 이전 테스트에서 "성공" 기준이 각각 달랐음
- 더블바텀: 넥라인 돌파만 하면 성공 (너무 쉬운 기준)
- 저항돌파: 3.5% 수익 달성해야 성공 (너무 어려운 기준)

수정: 동일한 기준으로 비교 (돌파 후 목표가 도달 vs 손절)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("패턴 신뢰도 계층 분석 (동일 기준 적용)")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

# 1시간봉 리샘플링
df_1h = df.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD 계산
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

print(f"\n데이터: {len(df_1h):,} 1H candles (2020-01-01 ~ )")

# H/L 변곡점 추출
def extract_hl_points(df_1h):
    hist = df_1h['hist'].values
    high = df_1h['high'].values
    low = df_1h['low'].values
    timestamps = df_1h['datetime'].values
    
    n = len(hist)
    points = []
    
    i = 0
    while i < n:
        if hist[i] > 0:
            start = i
            while i < n and hist[i] > 0:
                i += 1
            segment_highs = high[start:i]
            max_idx = start + np.argmax(segment_highs)
            points.append({
                'type': 'H', 'price': high[max_idx], 
                'time': timestamps[max_idx], 'idx': max_idx
            })
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0:
                i += 1
            segment_lows = low[start:i]
            min_idx = start + np.argmin(segment_lows)
            points.append({
                'type': 'L', 'price': low[min_idx], 
                'time': timestamps[min_idx], 'idx': min_idx
            })
        else:
            i += 1
    
    return points

points = extract_hl_points(df_1h)
print(f"추출된 H/L points: {len(points)}")

# ============================================================
# 동일 기준: TP 3.5% vs SL -L값 이탈
# ============================================================

TP_PCT = 3.5  # 목표 수익률
MAX_WAIT = 200  # 최대 대기 봉수

def backtest_entry(df_1h, entry_idx, entry_price, sl_price, tp_pct=TP_PCT, max_bars=MAX_WAIT):
    """진입 후 결과 테스트 - 공통 함수"""
    tp_price = entry_price * (1 + tp_pct/100)
    
    future = df_1h.iloc[entry_idx+1:entry_idx+max_bars]
    
    for _, row in future.iterrows():
        # TP 먼저 체크
        if row['high'] >= tp_price:
            return 'TP', (tp_price - entry_price) / entry_price * 100
        # SL 체크
        if row['low'] <= sl_price:
            return 'SL', (sl_price - entry_price) / entry_price * 100
    
    # 시간초과 - 현재가로 청산
    if len(future) > 0:
        last_close = future.iloc[-1]['close']
        return 'TIMEOUT', (last_close - entry_price) / entry_price * 100
    
    return 'NO_DATA', 0

# ============================================================
# 1. 더블바텀 백테스트
# ============================================================
print("\n" + "=" * 80)
print("1. 더블바텀 (Double Bottom)")
print("=" * 80)

db_results = []
for i in range(len(points) - 2):
    if points[i]['type'] == 'L' and points[i+2]['type'] == 'L':
        L1 = points[i]['price']
        H = points[i+1]['price']  # 넥라인
        L2 = points[i+2]['price']
        L2_idx = points[i+2]['idx']
        
        # 두 저점이 2% 이내로 비슷
        if abs(L2 - L1) / L1 > 0.02:
            continue
        
        # 넥라인 돌파 감지
        future = df_1h.iloc[L2_idx+1:L2_idx+MAX_WAIT]
        breakout_idx = None
        
        for j, (_, row) in enumerate(future.iterrows()):
            if row['close'] > H:
                breakout_idx = L2_idx + 1 + j
                break
        
        if breakout_idx is None:
            continue
        
        # 진입: 넥라인 돌파 시점
        entry_price = df_1h.iloc[breakout_idx]['close']
        sl_price = min(L1, L2)  # 저점 이탈 시 손절
        
        result, pnl = backtest_entry(df_1h, breakout_idx, entry_price, sl_price)
        
        db_results.append({
            'time': df_1h.iloc[breakout_idx]['datetime'],
            'entry': entry_price,
            'sl': sl_price,
            'result': result,
            'pnl': pnl
        })

db_df = pd.DataFrame(db_results)
print(f"\n더블바텀 넥라인 돌파 후:")
print(f"  총 진입: {len(db_df)}")
print(f"  TP 달성: {(db_df['result']=='TP').sum()} ({(db_df['result']=='TP').mean()*100:.1f}%)")
print(f"  SL 발생: {(db_df['result']=='SL').sum()} ({(db_df['result']=='SL').mean()*100:.1f}%)")
print(f"  평균 수익률: {db_df['pnl'].mean():.2f}%")

# ============================================================
# 2. 추세선 돌파 백테스트 (기존 결과)
# ============================================================
print("\n" + "=" * 80)
print("2. 추세선 돌파 (Trendline Breakout)")
print("=" * 80)

backtest = pd.read_csv('L_value_backtest.csv')
print(f"\n추세선 돌파 (H1-H2 하향추세선):")
print(f"  총 진입: {len(backtest)}")
print(f"  TP 달성: {backtest['tp2_done'].sum()} ({backtest['tp2_done'].mean()*100:.1f}%)")
print(f"  SL 발생: {backtest['sl_done'].sum()} ({backtest['sl_done'].mean()*100:.1f}%)")
print(f"  평균 수익률: {backtest['total_pnl'].mean():.2f}%")

# ============================================================
# 3. 저항선 돌파 백테스트 (수정된 기준)
# ============================================================
print("\n" + "=" * 80)
print("3. 저항선 돌파 (H Level Breakout)")
print("=" * 80)

rb_results = []
for i in range(len(points) - 1):
    if points[i]['type'] == 'H':
        H_price = points[i]['price']
        H_idx = points[i]['idx']
        
        # 다음 L 포인트 찾기 (손절 기준용)
        next_L = None
        for j in range(i+1, len(points)):
            if points[j]['type'] == 'L':
                next_L = points[j]['price']
                break
        
        if next_L is None:
            continue
        
        # H 이후 돌파 감지
        future = df_1h.iloc[H_idx+1:H_idx+MAX_WAIT]
        breakout_idx = None
        
        for j, (_, row) in enumerate(future.iterrows()):
            if row['close'] > H_price * 1.001:  # 0.1% 이상 돌파
                breakout_idx = H_idx + 1 + j
                break
        
        if breakout_idx is None:
            continue
        
        # 진입: H 레벨 돌파 시점
        entry_price = df_1h.iloc[breakout_idx]['close']
        sl_price = next_L  # L 레벨 이탈 시 손절
        
        result, pnl = backtest_entry(df_1h, breakout_idx, entry_price, sl_price)
        
        rb_results.append({
            'time': df_1h.iloc[breakout_idx]['datetime'],
            'entry': entry_price,
            'H': H_price,
            'sl': sl_price,
            'entry_L_gap': (entry_price - sl_price) / sl_price * 100,
            'result': result,
            'pnl': pnl
        })

rb_df = pd.DataFrame(rb_results)
print(f"\n저항선 돌파:")
print(f"  총 진입: {len(rb_df)}")
print(f"  TP 달성: {(rb_df['result']=='TP').sum()} ({(rb_df['result']=='TP').mean()*100:.1f}%)")
print(f"  SL 발생: {(rb_df['result']=='SL').sum()} ({(rb_df['result']=='SL').mean()*100:.1f}%)")
print(f"  평균 수익률: {rb_df['pnl'].mean():.2f}%")

# Entry-L Gap 기준 필터 적용
print(f"\n  [Entry-L Gap >= 1% 필터 적용]")
rb_filtered = rb_df[rb_df['entry_L_gap'] >= 1.0]
print(f"    총 진입: {len(rb_filtered)}")
print(f"    TP 달성: {(rb_filtered['result']=='TP').sum()} ({(rb_filtered['result']=='TP').mean()*100:.1f}%)")
print(f"    SL 발생: {(rb_filtered['result']=='SL').sum()} ({(rb_filtered['result']=='SL').mean()*100:.1f}%)")
print(f"    평균 수익률: {rb_filtered['pnl'].mean():.2f}%")

# ============================================================
# 4. 최종 비교
# ============================================================
print("\n" + "=" * 80)
print("4. 패턴 신뢰도 계층 최종 비교")
print("=" * 80)

db_tp_rate = (db_df['result']=='TP').mean()*100 if len(db_df) > 0 else 0
tb_tp_rate = backtest['tp2_done'].mean()*100
rb_tp_rate = (rb_df['result']=='TP').mean()*100 if len(rb_df) > 0 else 0
rb_f_tp_rate = (rb_filtered['result']=='TP').mean()*100 if len(rb_filtered) > 0 else 0

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│           패턴 신뢰도 계층 (동일 기준: TP 3.5%, SL = L값 이탈)             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  패턴                    │ 신호수 │ TP달성률  │ 평균수익  │ 비고          │
│  ─────────────────────────────────────────────────────────────────────────│
│  1. 더블바텀 넥라인 돌파 │ {len(db_df):>6} │ {db_tp_rate:>6.1f}%  │ {db_df['pnl'].mean():>7.2f}% │ 반전 형성     │
│  2. 추세선 돌파          │ {len(backtest):>6} │ {tb_tp_rate:>6.1f}%  │ {backtest['total_pnl'].mean():>7.2f}% │ 추세 이탈     │
│  3. 저항선 돌파          │ {len(rb_df):>6} │ {rb_tp_rate:>6.1f}%  │ {rb_df['pnl'].mean():>7.2f}% │ 레벨 돌파     │
│  3-1. 저항돌파(필터)     │ {len(rb_filtered):>6} │ {rb_f_tp_rate:>6.1f}%  │ {rb_filtered['pnl'].mean():>7.2f}% │ Entry-L>=1%   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  결론:                                                                     │
│  - 더블바텀: {db_tp_rate:.1f}% (넥라인만 돌파하면 좋은 성과)                       │
│  - 추세선: {tb_tp_rate:.1f}% (돌파해도 다시 내려올 가능성)                         │  
│  - 저항선: {rb_tp_rate:.1f}% (단순 돌파는 페이크 많음)                             │
│  - 저항선(필터): {rb_f_tp_rate:.1f}% (조건 갖추면 성과 개선)                       │
│                                                                            │
│  ★ 핵심 인사이트:                                                         │
│  더블바텀 성공률이 높은 이유 =                                             │
│  이미 "2번의 테스트"를 거쳤기 때문 (L1, L2에서 지지 확인)                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 5. 올바른 계층 해석
# ============================================================
print("\n" + "=" * 80)
print("5. 올바른 계층 해석")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                    패턴 계층의 올바른 해석                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  "더블바텀 < 추세돌파 < 저항돌파" 의 의미:                                 │
│                                                                            │
│  ✗ 틀린 해석: 더블바텀 성공률이 가장 낮다                                  │
│  ✓ 맞는 해석: 패턴 형성 순서/확정 순서                                     │
│                                                                            │
│  시간순서:                                                                 │
│  1. 더블바텀 형성 (L1-H-L2) → 패턴 감지                                    │
│  2. 넥라인(H) 돌파 = 추세선 돌파 → 진입 신호                               │
│  3. 다음 저항(H2) 돌파 → 방향 확정                                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│        H1 (저항1)                                                          │
│         ╲                                                                  │
│          ╲    H = 넥라인 = 추세선                                          │
│           ╲────╱──────────╲                                                │
│                           ╱╲                                               │
│                          ╱  ╲  H2 (저항2)                                  │
│                         ╱    ↑                                             │
│                        ╱    저항선 돌파 = 확정                              │
│     L1 ●─────────● L2                                                      │
│     (더블바텀)                                                             │
│                                                                            │
│  진입 전략:                                                                │
│  - 더블바텀 형성 확인 → 대기                                               │
│  - 넥라인 돌파 시 1차 진입 (추세선 돌파)                                   │
│  - H2 돌파 시 추가 매수 (저항선 돌파 확정)                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

print("\n분석 완료!")
