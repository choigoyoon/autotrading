#!/usr/bin/env python3
"""
볼린저밴드 수축 후 추세 전환 분석 (15분봉)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 수축 후 추세 전환 분석 (15분봉)")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
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

# 추세 (20봉 변화)
df['trend_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100

df = df.dropna().reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# ============================================================
# 밴드폭 분포
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 분포 (15M, BB30)")
print("=" * 80)

bb_width = df['bb_width']
print(f"\n  최소: {bb_width.min():.3f}%")
print(f"  평균: {bb_width.mean():.2f}%")
print(f"  중앙값: {bb_width.median():.2f}%")
print(f"  5%: {bb_width.quantile(0.05):.2f}%")
print(f"  10%: {bb_width.quantile(0.10):.2f}%")
print(f"  20%: {bb_width.quantile(0.20):.2f}%")

# ============================================================
# 밴드폭 기준으로 수축/확장 찾기
# ============================================================
print("\n수축→확장 전환점 분석 중...")

# 밴드폭 임계값 (절대값 기준)
width_threshold = bb_width.quantile(0.20)  # 하위 20%
print(f"수축 임계값: {width_threshold:.2f}%")

results = []

for i in range(50, len(df) - 200):
    current_width = df.iloc[i]['bb_width']
    prev_width = df.iloc[i-1]['bb_width']
    
    # 수축 → 확장 (밴드폭이 임계값 이하에서 위로)
    if prev_width <= width_threshold and current_width > width_threshold:
        
        # 수축 구간 찾기
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width'] <= width_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < 4:  # 최소 1시간 (4봉)
            continue
        
        # 수축 전 20봉
        pre_start = max(0, squeeze_start - 20)
        pre_squeeze = df.iloc[pre_start:squeeze_start]
        
        # 수축 구간
        during_squeeze = df.iloc[squeeze_start:i]
        
        # 돌파 캔들
        breakout = df.iloc[i]
        
        # 수축 전 추세
        if len(pre_squeeze) >= 5:
            pre_change = (pre_squeeze.iloc[-1]['close'] - pre_squeeze.iloc[0]['close']) / pre_squeeze.iloc[0]['close'] * 100
            if pre_change > 1:
                pre_trend = 'UP'
            elif pre_change < -1:
                pre_trend = 'DOWN'
            else:
                pre_trend = 'SIDE'
        else:
            pre_trend = 'SIDE'
        
        # 수축 중 위치
        avg_position = during_squeeze['bb_position'].mean()
        if avg_position > 60:
            sq_position = 'UPPER'
        elif avg_position < 40:
            sq_position = 'LOWER'
        else:
            sq_position = 'MIDDLE'
        
        # 돌파 방향
        if breakout['close'] > breakout['bb_upper']:
            break_dir = 'UP'
        elif breakout['close'] < breakout['bb_lower']:
            break_dir = 'DOWN'
        else:
            break_dir = 'UP' if breakout['close'] > df.iloc[i-1]['close'] else 'DOWN'
        
        # 결과 측정 (80봉=20시간, 200봉=50시간)
        future = df.iloc[i+1:i+201]
        if len(future) < 80:
            continue
        
        entry = breakout['close']
        
        if break_dir == 'UP':
            mfe = (future['high'].max() - entry) / entry * 100
            mae = (future['low'].min() - entry) / entry * 100
            pnl_20h = (future.iloc[79]['close'] - entry) / entry * 100
        else:
            mfe = (entry - future['low'].min()) / entry * 100
            mae = (entry - future['high'].max()) / entry * 100
            pnl_20h = (entry - future.iloc[79]['close']) / entry * 100
        
        results.append({
            'time': breakout['datetime'],
            'pre_trend': pre_trend,
            'sq_position': sq_position,
            'avg_position': avg_position,
            'break_dir': break_dir,
            'duration': squeeze_duration,
            'min_width': during_squeeze['bb_width'].min(),
            'rsi': breakout['rsi'],
            'vol_ratio': breakout['vol_ratio'],
            'mfe': mfe,
            'mae': mae,
            'pnl_20h': pnl_20h
        })

df_results = pd.DataFrame(results)
print(f"분석 완료: {len(df_results)}건")

if len(df_results) == 0:
    print("결과 없음")
    exit()

# ============================================================
# 수축 중 위치별
# ============================================================
print("\n" + "=" * 80)
print("수축 중 가격 위치별")
print("=" * 80)

print(f"\n{'위치':>10} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10} {'UP돌파':>10}")
print("-" * 65)

for pos in ['UPPER', 'MIDDLE', 'LOWER']:
    subset = df_results[df_results['sq_position'] == pos]
    if len(subset) >= 5:
        wr = (subset['pnl_20h'] > 0).mean() * 100
        up_pct = (subset['break_dir'] == 'UP').mean() * 100
        print(f"{pos:>10} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {wr:>10.1f}% {up_pct:>10.0f}%")

# ============================================================
# 돌파 방향별
# ============================================================
print("\n" + "=" * 80)
print("돌파 방향별")
print("=" * 80)

for dir in ['UP', 'DOWN']:
    subset = df_results[df_results['break_dir'] == dir]
    if len(subset) > 0:
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"\n[{dir}] {len(subset)}건, MFE: {subset['mfe'].mean():.2f}%, PnL: {subset['pnl_20h'].mean():.2f}%, 승률: {wr:.1f}%")

# ============================================================
# RSI 구간별
# ============================================================
print("\n" + "=" * 80)
print("RSI 구간별")
print("=" * 80)

print(f"\n{'RSI':>10} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 55)

for low, high in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    subset = df_results[(df_results['rsi'] >= low) & (df_results['rsi'] < high)]
    if len(subset) >= 10:
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}':>10} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {wr:>10.1f}%")

# ============================================================
# 수축 강도별
# ============================================================
print("\n" + "=" * 80)
print("수축 강도별 (밴드폭)")
print("=" * 80)

print(f"\n{'밴드폭':>12} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 55)

for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0)]:
    subset = df_results[(df_results['min_width'] >= low) & (df_results['min_width'] < high)]
    if len(subset) >= 10:
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}%':>12} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {wr:>10.1f}%")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>40} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 85)

conditions = [
    ('전체', df_results),
    ('상단 + 상승돌파', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['break_dir'] == 'UP')]),
    ('하단 + 하락돌파', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['break_dir'] == 'DOWN')]),
    ('상단 + 하락돌파 (역추세)', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['break_dir'] == 'DOWN')]),
    ('하단 + 상승돌파 (역추세)', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['break_dir'] == 'UP')]),
    ('RSI<30 + 상승', df_results[(df_results['rsi'] < 30) & (df_results['break_dir'] == 'UP')]),
    ('RSI>70 + 하락', df_results[(df_results['rsi'] > 70) & (df_results['break_dir'] == 'DOWN')]),
    ('거래량>2x + 상승', df_results[(df_results['vol_ratio'] > 2) & (df_results['break_dir'] == 'UP')]),
    ('거래량>2x + 하락', df_results[(df_results['vol_ratio'] > 2) & (df_results['break_dir'] == 'DOWN')]),
    ('밴드폭<0.5% + 상승', df_results[(df_results['min_width'] < 0.5) & (df_results['break_dir'] == 'UP')]),
    ('밴드폭<0.5% + 하락', df_results[(df_results['min_width'] < 0.5) & (df_results['break_dir'] == 'DOWN')]),
    ('상단+RSI>60+상승', df_results[(df_results['sq_position'] == 'UPPER') & (df_results['rsi'] > 60) & (df_results['break_dir'] == 'UP')]),
    ('하단+RSI<40+하락', df_results[(df_results['sq_position'] == 'LOWER') & (df_results['rsi'] < 40) & (df_results['break_dir'] == 'DOWN')]),
]

for name, subset in conditions:
    if len(subset) >= 5:
        wr = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>40} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {wr:>10.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론 (15분봉)")
print("=" * 80)

upper = df_results[df_results['sq_position'] == 'UPPER']
lower = df_results[df_results['sq_position'] == 'LOWER']

print(f"""
■ 수축 중 위치 → 돌파 방향:
  - 상단(UPPER): 상승돌파 {(upper['break_dir']=='UP').mean()*100:.0f}%
  - 하단(LOWER): 하락돌파 {(lower['break_dir']=='DOWN').mean()*100:.0f}%

■ 전체 결과:
  - 총 {len(df_results)}건
  - 평균 MFE: {df_results['mfe'].mean():.2f}%
  - 평균 PnL: {df_results['pnl_20h'].mean():.2f}%
  - 승률: {(df_results['pnl_20h']>0).mean()*100:.1f}%
""")

df_results.to_csv('bb_15m_results.csv', index=False)
print("저장: bb_15m_results.csv")
