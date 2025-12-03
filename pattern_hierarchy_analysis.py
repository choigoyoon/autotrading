#!/usr/bin/env python3
"""
패턴 신뢰도 계층 분석
더블바텀 < 추세선 돌파 < 저항 돌파

각 패턴의 성공률을 비교하여 계층 구조 검증
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("패턴 신뢰도 계층 분석: 더블바텀 < 추세돌파 < 저항돌파")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

# 1시간봉 리샘플링
df_1h = df.set_index('datetime').resample('1H').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD 계산
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

print(f"\n데이터: {len(df_1h):,} 1H candles (2020-01-01 ~ )")

# ============================================================
# H/L 변곡점 추출
# ============================================================
def extract_hl_points(df_1h):
    hist = df_1h['hist'].values
    high = df_1h['high'].values
    low = df_1h['low'].values
    close = df_1h['close'].values
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
# 1. 더블바텀 성공률
# ============================================================
print("\n" + "=" * 80)
print("1. 더블바텀 (Double Bottom) 성공률")
print("=" * 80)

def test_double_bottom(points, df_1h, tolerance=0.02, max_wait_bars=200):
    """더블바텀 패턴 성공률 테스트"""
    results = []
    
    for i in range(len(points) - 2):
        if points[i]['type'] == 'L' and points[i+2]['type'] == 'L':
            L1 = points[i]['price']
            H = points[i+1]['price']  # neckline
            L2 = points[i+2]['price']
            
            # 두 저점이 비슷한지 (tolerance 이내)
            diff_pct = abs(L2 - L1) / L1
            if diff_pct > tolerance:
                continue
            
            # L2 이후 데이터에서 성공 여부 확인
            l2_idx = points[i+2]['idx']
            future_data = df_1h.iloc[l2_idx+1:l2_idx+max_wait_bars]
            
            if len(future_data) < 10:
                continue
            
            success = False
            fail_reason = 'NO_BREAKOUT'
            
            for _, row in future_data.iterrows():
                # 넥라인(H) 돌파 = 성공
                if row['close'] > H:
                    success = True
                    fail_reason = None
                    break
                # L2보다 아래로 떨어짐 = 실패
                if row['low'] < min(L1, L2) * 0.99:
                    fail_reason = 'SL_HIT'
                    break
            
            results.append({
                'time': points[i+2]['time'],
                'L1': L1, 'H': H, 'L2': L2,
                'success': success,
                'fail_reason': fail_reason
            })
    
    return results

db_results = test_double_bottom(points, df_1h)
db_success = sum(1 for r in db_results if r['success'])
db_total = len(db_results)
db_rate = db_success / db_total * 100 if db_total > 0 else 0

print(f"\n더블바텀 패턴:")
print(f"  - 총 패턴 수: {db_total}")
print(f"  - 성공: {db_success} ({db_rate:.1f}%)")
print(f"  - 실패 원인:")
fail_reasons = {}
for r in db_results:
    if not r['success']:
        reason = r['fail_reason']
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
for reason, count in fail_reasons.items():
    print(f"    - {reason}: {count}")

# ============================================================
# 2. 추세선 돌파 성공률 (기존 백테스트 결과)
# ============================================================
print("\n" + "=" * 80)
print("2. 추세선 돌파 (Trendline Breakout) 성공률")
print("=" * 80)

# 기존 백테스트 데이터 로드
backtest = pd.read_csv('L_value_backtest.csv')
tb_total = len(backtest)
tb_success = (backtest['total_pnl'] > 0).sum()
tb_rate = tb_success / tb_total * 100

print(f"\n추세선 돌파 (H1-H2 하향추세선):")
print(f"  - 총 신호 수: {tb_total}")
print(f"  - 성공 (수익): {tb_success} ({tb_rate:.1f}%)")
print(f"  - TP 달성: {backtest['tp2_done'].sum()} ({backtest['tp2_done'].mean()*100:.1f}%)")
print(f"  - SL 발생: {backtest['sl_done'].sum()} ({backtest['sl_done'].mean()*100:.1f}%)")

# ============================================================
# 3. 저항 돌파 성공률
# ============================================================
print("\n" + "=" * 80)
print("3. 저항 돌파 (Resistance Breakout) 성공률")
print("=" * 80)

def test_resistance_breakout(points, df_1h, max_wait_bars=200):
    """저항선(H 레벨) 돌파 성공률 테스트"""
    results = []
    
    for i in range(len(points) - 1):
        if points[i]['type'] == 'H':
            H_price = points[i]['price']
            H_idx = points[i]['idx']
            
            # H 이후 데이터
            future_data = df_1h.iloc[H_idx+1:H_idx+max_wait_bars]
            
            if len(future_data) < 10:
                continue
            
            # 돌파 감지
            breakout_detected = False
            breakout_idx = None
            
            for j, (_, row) in enumerate(future_data.iterrows()):
                if row['close'] > H_price:
                    breakout_detected = True
                    breakout_idx = j
                    break
            
            if not breakout_detected:
                continue
            
            # 돌파 이후 성공 여부 확인
            post_breakout = future_data.iloc[breakout_idx+1:breakout_idx+100]
            
            if len(post_breakout) < 5:
                continue
            
            entry_price = future_data.iloc[breakout_idx]['close']
            tp_price = entry_price * 1.035  # 3.5% TP
            sl_price = H_price * 0.99  # H 레벨 아래로 이탈 = SL
            
            success = False
            fail_reason = 'NO_TP'
            
            for _, row in post_breakout.iterrows():
                if row['high'] >= tp_price:
                    success = True
                    fail_reason = None
                    break
                if row['low'] < sl_price:
                    fail_reason = 'SL_HIT'
                    break
            
            results.append({
                'time': future_data.iloc[breakout_idx]['datetime'],
                'H_price': H_price,
                'entry': entry_price,
                'success': success,
                'fail_reason': fail_reason
            })
    
    return results

rb_results = test_resistance_breakout(points, df_1h)
rb_success = sum(1 for r in rb_results if r['success'])
rb_total = len(rb_results)
rb_rate = rb_success / rb_total * 100 if rb_total > 0 else 0

print(f"\n저항선 돌파 (H level breakout):")
print(f"  - 총 돌파 수: {rb_total}")
print(f"  - 성공 (TP 달성): {rb_success} ({rb_rate:.1f}%)")
print(f"  - 실패 원인:")
fail_reasons = {}
for r in rb_results:
    if not r['success']:
        reason = r['fail_reason']
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
for reason, count in fail_reasons.items():
    print(f"    - {reason}: {count}")

# ============================================================
# 4. 계층 비교
# ============================================================
print("\n" + "=" * 80)
print("4. 패턴 신뢰도 계층 비교")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                      패턴 신뢰도 계층 구조                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  패턴              │ 신호수 │ 성공률  │ 의미                              │
│  ─────────────────────────────────────────────────────────────────────────│
│  더블바텀          │ {db_total:>5}  │ {db_rate:>5.1f}%  │ 반전 패턴 형성                   │
│  추세선 돌파       │ {tb_total:>5}  │ {tb_rate:>5.1f}%  │ 하락 추세 이탈                   │
│  저항선 돌파       │ {rb_total:>5}  │ {rb_rate:>5.1f}%  │ 명확한 레벨 돌파                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  예상 계층: 더블바텀 < 추세선 돌파 < 저항선 돌파                           │
│                                                                            │
│  검증 결과: {db_rate:.1f}% {'<' if db_rate < tb_rate else '>='} {tb_rate:.1f}% {'<' if tb_rate < rb_rate else '>='} {rb_rate:.1f}%                                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 5. 복합 패턴 (더블바텀 + 추세선 돌파)
# ============================================================
print("\n" + "=" * 80)
print("5. 복합 패턴 분석")
print("=" * 80)

print("""
복합 패턴 = 더블바텀 + 추세선 돌파 + 저항선 돌파

┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│        H1                                                            │
│         ╲                                                            │
│          ╲    H2 (= 넥라인)                                          │
│           ╲────╲                                                     │
│                 ╲    ← 추세선 돌파 (Entry 1)                         │
│                  ╲                                                   │
│                   ╲── H3 (저항선)                                    │
│                        ↑                                             │
│                   저항선 돌파 (Entry 2, 추매)                        │
│                                                                      │
│     L1 ●─────────● L2 (더블바텀)                                     │
│                                                                      │
│  진입 전략:                                                          │
│  1차 진입: 추세선 돌파 시 (소량)                                     │
│  2차 진입: 저항선 돌파 시 (추매)                                     │
│  손절: L2 이탈 시                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 6. 왜 이 계층이 맞는가?
# ============================================================
print("\n" + "=" * 80)
print("6. 계층 구조의 논리")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                      왜 더블바텀 < 추세돌파 < 저항돌파 인가?               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. 더블바텀 (낮은 신뢰도)                                                 │
│     - 단순히 "비슷한 저점 2개" 형성                                        │
│     - 아직 방향 전환 미확정                                                │
│     - 가격이 다시 내려갈 가능성 높음                                       │
│     - 매수세 유입 여부 불확실                                              │
│                                                                            │
│  2. 추세선 돌파 (중간 신뢰도)                                              │
│     - 하락 추세에서 이탈 시도                                              │
│     - 매도세 약화 신호                                                     │
│     - 하지만 아직 "저항 돌파"는 아님                                       │
│     - 페이크아웃 가능성 존재                                               │
│                                                                            │
│  3. 저항선 돌파 (높은 신뢰도)                                              │
│     - 명확한 가격 레벨 돌파                                                │
│     - 매도 물량 소화 완료                                                  │
│     - 방향 전환 확정                                                       │
│     - 새로운 매수세 유입 확인                                              │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  따라서:                                                                   │
│  - 더블바텀만으로 진입 ❌ (너무 이른 진입)                                 │
│  - 추세선 돌파로 1차 진입 ✓ (초기 포지션)                                  │
│  - 저항선 돌파로 2차 진입 ✓ (확정 후 추매)                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

print("\n분석 완료!")
