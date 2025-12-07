#!/usr/bin/env python3
"""
패턴 확정 단계 분석
더블바텀 → 추세돌파 → 저항돌파 (확정 순서)

핵심: 각 단계를 거칠수록 확률이 올라가야 함
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("패턴 확정 단계 분석: 더블바텀 → 추세돌파 → 저항돌파")
print("=" * 80)

# 데이터 로드
df_raw = pd.read_csv('analysis_15m.csv')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df = df_raw[df_raw['datetime'] >= '2020-01-01'].copy()

# 1시간봉 리샘플링
df_1h = df.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# MACD
exp1 = df_1h['close'].ewm(span=12, adjust=False).mean()
exp2 = df_1h['close'].ewm(span=26, adjust=False).mean()
df_1h['macd'] = exp1 - exp2
df_1h['signal'] = df_1h['macd'].ewm(span=9, adjust=False).mean()
df_1h['hist'] = df_1h['macd'] - df_1h['signal']

# H/L 추출
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
            max_idx = start + np.argmax(high[start:i])
            points.append({'type': 'H', 'price': high[max_idx], 'time': timestamps[max_idx], 'idx': max_idx})
        elif hist[i] < 0:
            start = i
            while i < n and hist[i] < 0:
                i += 1
            min_idx = start + np.argmin(low[start:i])
            points.append({'type': 'L', 'price': low[min_idx], 'time': timestamps[min_idx], 'idx': min_idx})
        else:
            i += 1
    return points

points = extract_hl_points(df_1h)
print(f"\n데이터: {len(df_1h):,} 1H candles")
print(f"H/L points: {len(points)}")

# ============================================================
# 단계별 진입 테스트
# ============================================================

def backtest_stage(df_1h, entry_idx, entry_price, sl_price, tp_pct=3.5, max_bars=200):
    """공통 백테스트"""
    tp_price = entry_price * (1 + tp_pct/100)
    future = df_1h.iloc[entry_idx+1:entry_idx+max_bars]
    
    for _, row in future.iterrows():
        if row['high'] >= tp_price:
            return 'TP', tp_pct
        if row['low'] <= sl_price:
            return 'SL', (sl_price - entry_price) / entry_price * 100
    
    if len(future) > 0:
        return 'TIMEOUT', (future.iloc[-1]['close'] - entry_price) / entry_price * 100
    return 'NO_DATA', 0

# ============================================================
# 분석: 3단계 확정 과정
# ============================================================
print("\n" + "=" * 80)
print("단계별 확정 과정 분석")
print("=" * 80)

stage_results = []

for i in range(len(points) - 4):
    # 더블바텀 패턴 찾기: L1 - H(넥라인) - L2
    if not (points[i]['type'] == 'L' and 
            points[i+1]['type'] == 'H' and 
            points[i+2]['type'] == 'L'):
        continue
    
    L1 = points[i]['price']
    L1_idx = points[i]['idx']
    neckline = points[i+1]['price']
    neckline_idx = points[i+1]['idx']
    L2 = points[i+2]['price']
    L2_idx = points[i+2]['idx']
    
    # 더블바텀 조건: L1 ≈ L2 (2% 이내)
    if abs(L2 - L1) / L1 > 0.02:
        continue
    
    # 이후 H 포인트들 찾기 (저항선들)
    future_H_points = []
    for j in range(i+3, min(i+8, len(points))):
        if points[j]['type'] == 'H':
            future_H_points.append(points[j])
    
    if len(future_H_points) < 1:
        continue
    
    H2 = future_H_points[0]  # 다음 저항선
    
    # ----------------------------------------------------------------
    # Stage 1: 더블바텀 형성 시점에 진입 (넥라인 돌파 전)
    # ----------------------------------------------------------------
    # L2 형성 직후 진입 (가장 빠른 진입)
    stage1_entry_idx = L2_idx + 1
    if stage1_entry_idx >= len(df_1h):
        continue
        
    stage1_entry = df_1h.iloc[stage1_entry_idx]['close']
    stage1_sl = min(L1, L2) * 0.99
    stage1_result, stage1_pnl = backtest_stage(df_1h, stage1_entry_idx, stage1_entry, stage1_sl)
    
    # ----------------------------------------------------------------
    # Stage 2: 넥라인(추세선) 돌파 시 진입
    # ----------------------------------------------------------------
    stage2_entry_idx = None
    for j in range(L2_idx + 1, min(L2_idx + 100, len(df_1h))):
        if df_1h.iloc[j]['close'] > neckline:
            stage2_entry_idx = j
            break
    
    stage2_result, stage2_pnl = 'NO_SIGNAL', 0
    if stage2_entry_idx:
        stage2_entry = df_1h.iloc[stage2_entry_idx]['close']
        stage2_sl = min(L1, L2)
        stage2_result, stage2_pnl = backtest_stage(df_1h, stage2_entry_idx, stage2_entry, stage2_sl)
    
    # ----------------------------------------------------------------
    # Stage 3: 저항선(H2) 돌파 시 진입
    # ----------------------------------------------------------------
    stage3_entry_idx = None
    H2_price = H2['price']
    H2_idx = H2['idx']
    
    for j in range(H2_idx + 1, min(H2_idx + 100, len(df_1h))):
        if df_1h.iloc[j]['close'] > H2_price:
            stage3_entry_idx = j
            break
    
    stage3_result, stage3_pnl = 'NO_SIGNAL', 0
    if stage3_entry_idx:
        stage3_entry = df_1h.iloc[stage3_entry_idx]['close']
        stage3_sl = min(L1, L2)  # 여전히 L값이 손절 기준
        stage3_result, stage3_pnl = backtest_stage(df_1h, stage3_entry_idx, stage3_entry, stage3_sl)
    
    stage_results.append({
        'time': points[i+2]['time'],
        'L1': L1, 'neckline': neckline, 'L2': L2, 'H2': H2_price,
        'stage1_result': stage1_result, 'stage1_pnl': stage1_pnl,
        'stage2_result': stage2_result, 'stage2_pnl': stage2_pnl,
        'stage3_result': stage3_result, 'stage3_pnl': stage3_pnl,
    })

df_stages = pd.DataFrame(stage_results)
print(f"\n분석된 더블바텀 패턴: {len(df_stages)}")

# ============================================================
# 결과 분석
# ============================================================
print("\n" + "=" * 80)
print("단계별 성과 비교")
print("=" * 80)

# Stage 1: 더블바텀 형성 시
s1_valid = df_stages[df_stages['stage1_result'] != 'NO_DATA']
s1_tp = (s1_valid['stage1_result'] == 'TP').sum()
s1_sl = (s1_valid['stage1_result'] == 'SL').sum()

# Stage 2: 넥라인 돌파 시
s2_valid = df_stages[df_stages['stage2_result'].isin(['TP', 'SL', 'TIMEOUT'])]
s2_tp = (s2_valid['stage2_result'] == 'TP').sum()
s2_sl = (s2_valid['stage2_result'] == 'SL').sum()

# Stage 3: 저항선 돌파 시
s3_valid = df_stages[df_stages['stage3_result'].isin(['TP', 'SL', 'TIMEOUT'])]
s3_tp = (s3_valid['stage3_result'] == 'TP').sum()
s3_sl = (s3_valid['stage3_result'] == 'SL').sum()

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    단계별 진입 성과 비교                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Stage │ 진입 시점              │ 신호수 │ TP율   │ SL율   │ 평균PnL    │
│  ──────────────────────────────────────────────────────────────────────── │
│    1   │ 더블바텀 형성 직후      │ {len(s1_valid):>6} │ {s1_tp/len(s1_valid)*100 if len(s1_valid)>0 else 0:>5.1f}% │ {s1_sl/len(s1_valid)*100 if len(s1_valid)>0 else 0:>5.1f}% │ {s1_valid['stage1_pnl'].mean():>7.2f}%  │
│    2   │ 넥라인(추세선) 돌파 시  │ {len(s2_valid):>6} │ {s2_tp/len(s2_valid)*100 if len(s2_valid)>0 else 0:>5.1f}% │ {s2_sl/len(s2_valid)*100 if len(s2_valid)>0 else 0:>5.1f}% │ {s2_valid['stage2_pnl'].mean():>7.2f}%  │
│    3   │ 저항선(H2) 돌파 시      │ {len(s3_valid):>6} │ {s3_tp/len(s3_valid)*100 if len(s3_valid)>0 else 0:>5.1f}% │ {s3_sl/len(s3_valid)*100 if len(s3_valid)>0 else 0:>5.1f}% │ {s3_valid['stage3_pnl'].mean():>7.2f}%  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 핵심 인사이트
# ============================================================
print("\n" + "=" * 80)
print("핵심 인사이트")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│              "더블바텀 < 추세돌파 < 저항돌파" 의 올바른 해석               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  이 계층은 "성공률 순서"가 아니라 "확정 단계"를 의미함                     │
│                                                                            │
│  차트 구조:                                                                │
│                                                                            │
│         H1 ─────────────────────────────                                   │
│           ╲                                                                │
│            ╲   neckline (H)                                                │
│             ╲───────╲───────── H2 (저항선)                                 │
│                      ╲       ↗                                             │
│                       ╲     ╱                                              │
│                        ╲   ╱                                               │
│         L1 ●───────────● L2                                                │
│                                                                            │
│  확정 단계:                                                                │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  Stage 1. 더블바텀 형성                                                    │
│     - L1과 L2가 비슷한 레벨에서 형성                                       │
│     - 의미: "이 가격대에서 지지됨" → 반전 가능성                           │
│     - 확정도: 낮음 (아직 방향 미확정)                                      │
│                                                                            │
│  Stage 2. 넥라인(추세선) 돌파                                              │
│     - neckline 위로 가격이 상승                                            │
│     - 의미: "하락 추세 이탈" → 상승 시도 중                                │
│     - 확정도: 중간 (상승 시도했지만 저항선 미돌파)                         │
│                                                                            │
│  Stage 3. 저항선(H2) 돌파                                                  │
│     - 새로운 저항선(H2) 돌파                                               │
│     - 의미: "상승 방향 확정" → 추세 전환 완료                              │
│     - 확정도: 높음                                                         │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  결론:                                                                     │
│  - 더블바텀만 보고 진입 = 너무 이른 진입 (페이크 많음)                     │
│  - 추세선 돌파 시 진입 = 적절한 진입 (확인 후 진입)                        │
│  - 저항선 돌파 시 진입 = 안전한 진입 (확정 후 추매)                        │
│                                                                            │
│  추천 전략:                                                                │
│  - Stage 2 (추세선 돌파)에서 1차 진입                                      │
│  - Stage 3 (저항선 돌파)에서 추가 매수                                     │
│  - 손절: L2 이탈 시                                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 단계 진행 분석
print("\n" + "=" * 80)
print("단계 진행 분석")
print("=" * 80)

# Stage 1 → Stage 2로 진행한 케이스
s1_to_s2 = df_stages[(df_stages['stage1_result'].isin(['TP', 'SL', 'TIMEOUT'])) & 
                      (df_stages['stage2_result'].isin(['TP', 'SL', 'TIMEOUT']))]

# Stage 2 → Stage 3로 진행한 케이스
s2_to_s3 = df_stages[(df_stages['stage2_result'].isin(['TP', 'SL', 'TIMEOUT'])) & 
                      (df_stages['stage3_result'].isin(['TP', 'SL', 'TIMEOUT']))]

# Stage 2에서 TP인데 Stage 3도 있는 케이스
s2_tp_then_s3 = df_stages[(df_stages['stage2_result'] == 'TP') & 
                           (df_stages['stage3_result'].isin(['TP', 'SL', 'TIMEOUT']))]

print(f"""
단계 진행 통계:
- Stage 1 → Stage 2 진행: {len(s1_to_s2)} / {len(s1_valid)} ({len(s1_to_s2)/len(s1_valid)*100 if len(s1_valid)>0 else 0:.1f}%)
- Stage 2 → Stage 3 진행: {len(s2_to_s3)} / {len(s2_valid)} ({len(s2_to_s3)/len(s2_valid)*100 if len(s2_valid)>0 else 0:.1f}%)
- Stage 2 TP 후 Stage 3 존재: {len(s2_tp_then_s3)}
  - Stage 3도 TP: {(s2_tp_then_s3['stage3_result']=='TP').sum()}
  - Stage 3에서 SL: {(s2_tp_then_s3['stage3_result']=='SL').sum()}
""")

print("\n분석 완료!")
