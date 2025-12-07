#!/usr/bin/env python3
"""
BB 수축 중 위치 → 돌파 방향 예측 검증 (미래 데이터 없이)

핵심: 수축 중일 때 현재 위치만 보고 돌파 방향 예측 가능한가?
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 수축 중 위치 → 돌파 방향 예측 검증 (실시간)")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"15M 데이터: {len(df):,}개")

# BB 계산
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

df = df.iloc[100:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값 (20%): {width_threshold:.2f}%")

# ============================================================
# 실시간 시뮬레이션
# ============================================================
print("\n실시간 예측 시뮬레이션...")

results = []

# 수축 상태 추적
in_squeeze = False
squeeze_start_idx = None
squeeze_positions = []  # 수축 중 bb_position 기록

for i in range(1, len(df) - 200):
    curr_width = df.iloc[i]['bb_width']
    prev_width = df.iloc[i-1]['bb_width']
    
    if pd.isna(curr_width) or pd.isna(prev_width):
        continue
    
    # 수축 진입
    if not in_squeeze and curr_width <= width_threshold:
        in_squeeze = True
        squeeze_start_idx = i
        squeeze_positions = [df.iloc[i]['bb_position']]
    
    # 수축 중
    elif in_squeeze and curr_width <= width_threshold:
        squeeze_positions.append(df.iloc[i]['bb_position'])
    
    # 수축 → 확장 (돌파!)
    elif in_squeeze and curr_width > width_threshold:
        in_squeeze = False
        
        if len(squeeze_positions) < 4:  # 최소 1시간
            squeeze_positions = []
            continue
        
        # === 예측 시점: 돌파 직전 (i-1) ===
        # 수축 중 평균 위치로 방향 예측
        avg_pos = np.mean(squeeze_positions)
        
        if avg_pos > 60:
            predicted_dir = 'UP'
        elif avg_pos < 40:
            predicted_dir = 'DOWN'
        else:
            predicted_dir = 'UNKNOWN'
        
        # === 실제 돌파 방향 (i 시점) ===
        brk = df.iloc[i]
        if brk['close'] > brk['bb_upper']:
            actual_dir = 'UP'
        elif brk['close'] < brk['bb_lower']:
            actual_dir = 'DOWN'
        else:
            actual_dir = 'UP' if brk['close'] > df.iloc[i-1]['close'] else 'DOWN'
        
        # === 예측 정확도 ===
        correct = (predicted_dir == actual_dir)
        
        # === 실제 결과 (20시간 후) ===
        future = df.iloc[i+1:i+81]
        if len(future) < 80:
            squeeze_positions = []
            continue
        
        entry = brk['close']
        if actual_dir == 'UP':
            pnl = (future.iloc[79]['close'] - entry) / entry * 100
            mfe = (future['high'].max() - entry) / entry * 100
        else:
            pnl = (entry - future.iloc[79]['close']) / entry * 100
            mfe = (entry - future['low'].min()) / entry * 100
        
        results.append({
            'time': brk['datetime'],
            'avg_pos': avg_pos,
            'predicted': predicted_dir,
            'actual': actual_dir,
            'correct': correct,
            'duration': len(squeeze_positions),
            'min_width': df.iloc[squeeze_start_idx:i]['bb_width'].min(),
            'rsi': brk['rsi'],
            'pnl': pnl,
            'mfe': mfe
        })
        
        squeeze_positions = []

df_r = pd.DataFrame(results)
print(f"분석 완료: {len(df_r)}건")

# ============================================================
# 예측 정확도 분석
# ============================================================
print("\n" + "=" * 80)
print("예측 정확도 분석")
print("=" * 80)

# 전체 (UNKNOWN 제외)
known = df_r[df_r['predicted'] != 'UNKNOWN']
print(f"\n전체 (예측 가능): {len(known)}건")
print(f"예측 정확도: {known['correct'].mean()*100:.1f}%")

# 위치별
print("\n" + "-" * 60)
print(f"{'예측':>10} {'건수':>8} {'정확도':>10} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 60)

for pred in ['UP', 'DOWN', 'UNKNOWN']:
    s = df_r[df_r['predicted'] == pred]
    if len(s) >= 5:
        acc = s['correct'].mean() * 100 if pred != 'UNKNOWN' else 0
        wr = (s['pnl'] > 0).mean() * 100
        print(f"{pred:>10} {len(s):>8} {acc:>10.1f}% {s['mfe'].mean():>10.2f}% {s['pnl'].mean():>10.2f}% {wr:>10.1f}%")

# ============================================================
# 예측 정확 vs 오류 케이스 분석
# ============================================================
print("\n" + "=" * 80)
print("예측 정확 vs 오류 케이스")
print("=" * 80)

correct_cases = known[known['correct'] == True]
wrong_cases = known[known['correct'] == False]

print(f"\n예측 정확: {len(correct_cases)}건 ({len(correct_cases)/len(known)*100:.1f}%)")
print(f"  MFE: {correct_cases['mfe'].mean():.2f}%")
print(f"  PnL: {correct_cases['pnl'].mean():.2f}%")
print(f"  승률: {(correct_cases['pnl']>0).mean()*100:.1f}%")

print(f"\n예측 오류: {len(wrong_cases)}건 ({len(wrong_cases)/len(known)*100:.1f}%)")
print(f"  MFE: {wrong_cases['mfe'].mean():.2f}%")
print(f"  PnL: {wrong_cases['pnl'].mean():.2f}%")
print(f"  승률: {(wrong_cases['pnl']>0).mean()*100:.1f}%")

# ============================================================
# 위치 극단값일 때 정확도
# ============================================================
print("\n" + "=" * 80)
print("위치 극단값별 예측 정확도")
print("=" * 80)

print(f"\n{'위치':>15} {'건수':>8} {'정확도':>10} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 70)

for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
    s = df_r[(df_r['avg_pos'] >= lo) & (df_r['avg_pos'] < hi)]
    if len(s) >= 10:
        # 극단값일수록 예측이 더 정확해야 함
        s_known = s[s['predicted'] != 'UNKNOWN']
        acc = s_known['correct'].mean() * 100 if len(s_known) > 0 else 0
        wr = (s['pnl'] > 0).mean() * 100
        print(f"{f'{lo}-{hi}%':>15} {len(s):>8} {acc:>10.1f}% {s['mfe'].mean():>10.2f}% {s['pnl'].mean():>10.2f}% {wr:>10.1f}%")

# ============================================================
# 예측 정확 + 방향대로 진입 시 성과
# ============================================================
print("\n" + "=" * 80)
print("예측대로 진입 시 성과")
print("=" * 80)

# UP 예측 → 롱 진입
up_pred = df_r[df_r['predicted'] == 'UP']
# DOWN 예측 → 숏 진입  
down_pred = df_r[df_r['predicted'] == 'DOWN']

print(f"\n[UP 예측 → 롱 진입] {len(up_pred)}건")
if len(up_pred) > 0:
    # 롱 관점 PnL 재계산
    up_results = []
    for _, row in up_pred.iterrows():
        idx = df[df['datetime'] == row['time']].index[0]
        entry = df.iloc[idx]['close']
        future = df.iloc[idx+1:idx+81]
        if len(future) >= 80:
            pnl_long = (future.iloc[79]['close'] - entry) / entry * 100
            mfe_long = (future['high'].max() - entry) / entry * 100
            up_results.append({'pnl': pnl_long, 'mfe': mfe_long})
    
    if up_results:
        up_df = pd.DataFrame(up_results)
        print(f"  MFE: {up_df['mfe'].mean():.2f}%")
        print(f"  PnL: {up_df['pnl'].mean():.2f}%")
        print(f"  승률: {(up_df['pnl']>0).mean()*100:.1f}%")

print(f"\n[DOWN 예측 → 숏 진입] {len(down_pred)}건")
if len(down_pred) > 0:
    # 숏 관점 PnL 재계산
    down_results = []
    for _, row in down_pred.iterrows():
        idx = df[df['datetime'] == row['time']].index[0]
        entry = df.iloc[idx]['close']
        future = df.iloc[idx+1:idx+81]
        if len(future) >= 80:
            pnl_short = (entry - future.iloc[79]['close']) / entry * 100
            mfe_short = (entry - future['low'].min()) / entry * 100
            down_results.append({'pnl': pnl_short, 'mfe': mfe_short})
    
    if down_results:
        down_df = pd.DataFrame(down_results)
        print(f"  MFE: {down_df['mfe'].mean():.2f}%")
        print(f"  PnL: {down_df['pnl'].mean():.2f}%")
        print(f"  승률: {(down_df['pnl']>0).mean()*100:.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

known = df_r[df_r['predicted'] != 'UNKNOWN']
acc = known['correct'].mean() * 100

print(f"""
■ 예측 방법: 
  - 수축 중 평균 BB 위치 > 60% → UP 예측
  - 수축 중 평균 BB 위치 < 40% → DOWN 예측

■ 예측 정확도: {acc:.1f}%
  - UP 예측: {len(df_r[df_r['predicted']=='UP'])}건, 정확도 {df_r[df_r['predicted']=='UP']['correct'].mean()*100:.1f}%
  - DOWN 예측: {len(df_r[df_r['predicted']=='DOWN'])}건, 정확도 {df_r[df_r['predicted']=='DOWN']['correct'].mean()*100:.1f}%

■ 실제 수익:
  - 예측 정확 시: PnL {correct_cases['pnl'].mean():.2f}%, 승률 {(correct_cases['pnl']>0).mean()*100:.1f}%
  - 예측 오류 시: PnL {wrong_cases['pnl'].mean():.2f}%, 승률 {(wrong_cases['pnl']>0).mean()*100:.1f}%
""")

df_r.to_csv('bb_realtime_test_results.csv', index=False)
print("저장: bb_realtime_test_results.csv")
