#!/usr/bin/env python3
"""
볼린저밴드 수축 후 추세 전환 분석 (15분봉) v2
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 수축 후 추세 전환 분석 (15분봉)")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"15M 데이터: {len(df):,}개")

# 볼린저밴드 (30)
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

# 밴드 내 위치 (0~100%)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

# 거래량
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

# NaN 제거 (앞부분만)
df = df.iloc[100:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# ============================================================
# 밴드폭 분포
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 분포 (15M, BB30)")
print("=" * 80)

bb_width = df['bb_width'].dropna()
print(f"\n  평균: {bb_width.mean():.2f}%")
print(f"  중앙값: {bb_width.median():.2f}%")
print(f"  5%: {bb_width.quantile(0.05):.2f}%")
print(f"  10%: {bb_width.quantile(0.10):.2f}%")
print(f"  20%: {bb_width.quantile(0.20):.2f}%")

# 수축 임계값
width_threshold = bb_width.quantile(0.20)
print(f"\n수축 임계값 (20%): {width_threshold:.2f}%")

# ============================================================
# 수축→확장 전환점 찾기
# ============================================================
print("\n수축→확장 전환점 분석 중...")

results = []

for i in range(50, len(df) - 200):
    curr = df.iloc[i]['bb_width']
    prev = df.iloc[i-1]['bb_width']
    
    if pd.isna(curr) or pd.isna(prev):
        continue
    
    # 수축→확장 전환
    if prev <= width_threshold and curr > width_threshold:
        
        # 수축 시작점
        sq_start = i - 1
        while sq_start > 0:
            w = df.iloc[sq_start]['bb_width']
            if pd.isna(w) or w > width_threshold:
                break
            sq_start -= 1
        
        sq_duration = i - sq_start
        if sq_duration < 4:
            continue
        
        # 수축 구간
        sq_data = df.iloc[sq_start:i]
        
        # 돌파 캔들
        brk = df.iloc[i]
        
        # 수축 전 추세 (20봉)
        pre_start = max(0, sq_start - 20)
        pre_data = df.iloc[pre_start:sq_start]
        if len(pre_data) >= 5:
            pre_chg = (pre_data.iloc[-1]['close'] - pre_data.iloc[0]['close']) / pre_data.iloc[0]['close'] * 100
            if pre_chg > 1:
                pre_trend = 'UP'
            elif pre_chg < -1:
                pre_trend = 'DOWN'
            else:
                pre_trend = 'SIDE'
        else:
            pre_trend = 'SIDE'
        
        # 수축 중 위치
        avg_pos = sq_data['bb_position'].mean()
        if pd.isna(avg_pos):
            continue
        if avg_pos > 60:
            sq_pos = 'UPPER'
        elif avg_pos < 40:
            sq_pos = 'LOWER'
        else:
            sq_pos = 'MIDDLE'
        
        # 돌파 방향
        if brk['close'] > brk['bb_upper']:
            brk_dir = 'UP'
        elif brk['close'] < brk['bb_lower']:
            brk_dir = 'DOWN'
        else:
            brk_dir = 'UP' if brk['close'] > df.iloc[i-1]['close'] else 'DOWN'
        
        # 미래 데이터
        future = df.iloc[i+1:i+201]
        if len(future) < 80:
            continue
        
        entry = brk['close']
        
        if brk_dir == 'UP':
            mfe = (future['high'].max() - entry) / entry * 100
            mae = (future['low'].min() - entry) / entry * 100
            pnl_20h = (future.iloc[79]['close'] - entry) / entry * 100
        else:
            mfe = (entry - future['low'].min()) / entry * 100
            mae = (entry - future['high'].max()) / entry * 100
            pnl_20h = (entry - future.iloc[79]['close']) / entry * 100
        
        results.append({
            'time': brk['datetime'],
            'pre_trend': pre_trend,
            'sq_pos': sq_pos,
            'avg_pos': avg_pos,
            'brk_dir': brk_dir,
            'duration': sq_duration,
            'min_width': sq_data['bb_width'].min(),
            'rsi': brk['rsi'],
            'vol_ratio': brk['vol_ratio'],
            'mfe': mfe,
            'mae': mae,
            'pnl': pnl_20h
        })

df_r = pd.DataFrame(results)
print(f"분석 완료: {len(df_r)}건")

if len(df_r) == 0:
    print("결과 없음")
    exit()

# ============================================================
# 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 중 위치별")
print("=" * 80)

print(f"\n{'위치':>8} {'건수':>6} {'MFE':>8} {'PnL':>8} {'승률':>8} {'UP%':>8}")
print("-" * 55)

for pos in ['UPPER', 'MIDDLE', 'LOWER']:
    s = df_r[df_r['sq_pos'] == pos]
    if len(s) >= 5:
        wr = (s['pnl'] > 0).mean() * 100
        up = (s['brk_dir'] == 'UP').mean() * 100
        print(f"{pos:>8} {len(s):>6} {s['mfe'].mean():>8.2f}% {s['pnl'].mean():>8.2f}% {wr:>8.1f}% {up:>8.0f}%")

print("\n" + "=" * 80)
print("돌파 방향별")
print("=" * 80)

for d in ['UP', 'DOWN']:
    s = df_r[df_r['brk_dir'] == d]
    if len(s) > 0:
        wr = (s['pnl'] > 0).mean() * 100
        print(f"\n[{d}] {len(s)}건, MFE: {s['mfe'].mean():.2f}%, PnL: {s['pnl'].mean():.2f}%, 승률: {wr:.1f}%")

print("\n" + "=" * 80)
print("RSI 구간별")
print("=" * 80)

print(f"\n{'RSI':>8} {'건수':>6} {'MFE':>8} {'PnL':>8} {'승률':>8}")
print("-" * 45)

for lo, hi in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    s = df_r[(df_r['rsi'] >= lo) & (df_r['rsi'] < hi)]
    if len(s) >= 10:
        wr = (s['pnl'] > 0).mean() * 100
        print(f"{f'{lo}-{hi}':>8} {len(s):>6} {s['mfe'].mean():>8.2f}% {s['pnl'].mean():>8.2f}% {wr:>8.1f}%")

print("\n" + "=" * 80)
print("밴드폭별")
print("=" * 80)

print(f"\n{'밴드폭':>10} {'건수':>6} {'MFE':>8} {'PnL':>8} {'승률':>8}")
print("-" * 50)

for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3)]:
    s = df_r[(df_r['min_width'] >= lo) & (df_r['min_width'] < hi)]
    if len(s) >= 10:
        wr = (s['pnl'] > 0).mean() * 100
        print(f"{f'{lo}-{hi}%':>10} {len(s):>6} {s['mfe'].mean():>8.2f}% {s['pnl'].mean():>8.2f}% {wr:>8.1f}%")

print("\n" + "=" * 80)
print("최적 조건")
print("=" * 80)

print(f"\n{'조건':>35} {'건수':>6} {'MFE':>8} {'PnL':>8} {'승률':>8}")
print("-" * 75)

conds = [
    ('전체', df_r),
    ('상단+상승', df_r[(df_r['sq_pos'] == 'UPPER') & (df_r['brk_dir'] == 'UP')]),
    ('하단+하락', df_r[(df_r['sq_pos'] == 'LOWER') & (df_r['brk_dir'] == 'DOWN')]),
    ('상단+하락(역)', df_r[(df_r['sq_pos'] == 'UPPER') & (df_r['brk_dir'] == 'DOWN')]),
    ('하단+상승(역)', df_r[(df_r['sq_pos'] == 'LOWER') & (df_r['brk_dir'] == 'UP')]),
    ('RSI<30+상승', df_r[(df_r['rsi'] < 30) & (df_r['brk_dir'] == 'UP')]),
    ('RSI>70+하락', df_r[(df_r['rsi'] > 70) & (df_r['brk_dir'] == 'DOWN')]),
    ('거래량>2x', df_r[df_r['vol_ratio'] > 2]),
    ('밴드폭<1%', df_r[df_r['min_width'] < 1]),
    ('상단+RSI>50+상승', df_r[(df_r['sq_pos'] == 'UPPER') & (df_r['rsi'] > 50) & (df_r['brk_dir'] == 'UP')]),
    ('하단+RSI<50+하락', df_r[(df_r['sq_pos'] == 'LOWER') & (df_r['rsi'] < 50) & (df_r['brk_dir'] == 'DOWN')]),
]

for name, s in conds:
    if len(s) >= 5:
        wr = (s['pnl'] > 0).mean() * 100
        print(f"{name:>35} {len(s):>6} {s['mfe'].mean():>8.2f}% {s['pnl'].mean():>8.2f}% {wr:>8.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

upper = df_r[df_r['sq_pos'] == 'UPPER']
lower = df_r[df_r['sq_pos'] == 'LOWER']

print(f"""
■ 수축 중 위치 → 돌파 방향 예측:
  - 상단(UPPER): 상승돌파 {(upper['brk_dir']=='UP').mean()*100:.0f}%
  - 하단(LOWER): 하락돌파 {(lower['brk_dir']=='DOWN').mean()*100:.0f}%

■ 전체:
  - {len(df_r)}건
  - 평균 MFE: {df_r['mfe'].mean():.2f}%
  - 평균 PnL: {df_r['pnl'].mean():.2f}%
  - 승률: {(df_r['pnl']>0).mean()*100:.1f}%
""")

df_r.to_csv('bb_15m_results.csv', index=False)
print("저장: bb_15m_results.csv")
