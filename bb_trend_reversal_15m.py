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

# 밴드폭 백분위
df['bb_width_min_100'] = df['bb_width'].rolling(100).min()
df['bb_width_max_100'] = df['bb_width'].rolling(100).max()
df['bb_width_pct'] = (df['bb_width'] - df['bb_width_min_100']) / (df['bb_width_max_100'] - df['bb_width_min_100']) * 100

# 추세 판단용
df['ma20'] = df['close'].rolling(20).mean()
df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / df['ma20'].shift(10) * 100

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

# 거래량
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

df = df.dropna().reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# ============================================================
# 밴드폭 분포
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 분포 (15M, BB30)")
print("=" * 80)

bb_width = df['bb_width']
print(f"\n  평균: {bb_width.mean():.2f}%")
print(f"  중앙값: {bb_width.median():.2f}%")
print(f"  5%: {bb_width.quantile(0.05):.2f}%")
print(f"  10%: {bb_width.quantile(0.1):.2f}%")
print(f"  20%: {bb_width.quantile(0.2):.2f}%")

# ============================================================
# 수축 구간 분석
# ============================================================
print("\n수축 구간 분석 중...")

squeeze_threshold = 20
results = []

for i in range(100, len(df) - 200):
    current_pct = df.iloc[i]['bb_width_pct']
    prev_pct = df.iloc[i-1]['bb_width_pct']
    
    if prev_pct <= squeeze_threshold and current_pct > squeeze_threshold:
        
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width_pct'] <= squeeze_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < 3:
            continue
        
        pre_squeeze = df.iloc[max(0, squeeze_start-20):squeeze_start]
        during_squeeze = df.iloc[squeeze_start:i]
        breakout = df.iloc[i]
        
        # 수축 전 추세
        if len(pre_squeeze) >= 10:
            pre_trend_pct = (pre_squeeze.iloc[-1]['close'] - pre_squeeze.iloc[0]['close']) / pre_squeeze.iloc[0]['close'] * 100
            if pre_trend_pct > 1:
                pre_trend = 'UP'
            elif pre_trend_pct < -1:
                pre_trend = 'DOWN'
            else:
                pre_trend = 'SIDE'
        else:
            pre_trend = 'SIDE'
            pre_trend_pct = 0
        
        # 수축 중 위치
        avg_bb_position = during_squeeze['bb_position'].mean()
        if avg_bb_position > 60:
            squeeze_position = 'UPPER'
        elif avg_bb_position < 40:
            squeeze_position = 'LOWER'
        else:
            squeeze_position = 'MIDDLE'
        
        # 돌파 방향
        if breakout['close'] > breakout['bb_upper']:
            breakout_dir = 'UP'
        elif breakout['close'] < breakout['bb_lower']:
            breakout_dir = 'DOWN'
        else:
            breakout_dir = 'UP' if breakout['close'] > df.iloc[i-1]['close'] else 'DOWN'
        
        # 추세 전환 여부
        if pre_trend == 'UP' and breakout_dir == 'DOWN':
            trend_change = 'REVERSAL_DOWN'
        elif pre_trend == 'DOWN' and breakout_dir == 'UP':
            trend_change = 'REVERSAL_UP'
        elif pre_trend == breakout_dir:
            trend_change = 'CONTINUATION'
        else:
            trend_change = 'NEUTRAL'
        
        rsi_at_breakout = breakout['rsi']
        vol_at_breakout = breakout['vol_ratio']
        min_width = during_squeeze['bb_width'].min()
        
        # 결과 측정 (80봉=20H, 200봉=50H)
        future = df.iloc[i+1:i+201]
        if len(future) < 80:
            continue
        
        entry_price = breakout['close']
        
        if breakout_dir == 'UP':
            mfe = (future['high'].max() - entry_price) / entry_price * 100
            mae = (future['low'].min() - entry_price) / entry_price * 100
            pnl_20h = (future.iloc[79]['close'] - entry_price) / entry_price * 100
            pnl_50h = (future.iloc[-1]['close'] - entry_price) / entry_price * 100
        else:
            mfe = (entry_price - future['low'].min()) / entry_price * 100
            mae = (entry_price - future['high'].max()) / entry_price * 100
            pnl_20h = (entry_price - future.iloc[79]['close']) / entry_price * 100
            pnl_50h = (entry_price - future.iloc[-1]['close']) / entry_price * 100
        
        results.append({
            'time': breakout['datetime'],
            'pre_trend': pre_trend,
            'squeeze_position': squeeze_position,
            'avg_bb_position': avg_bb_position,
            'breakout_dir': breakout_dir,
            'trend_change': trend_change,
            'squeeze_duration': squeeze_duration,
            'min_width': min_width,
            'rsi': rsi_at_breakout,
            'vol_ratio': vol_at_breakout,
            'mfe': mfe,
            'mae': mae,
            'pnl_20h': pnl_20h,
            'pnl_50h': pnl_50h
        })

df_results = pd.DataFrame(results)
print(f"분석 완료: {len(df_results)}건")

# ============================================================
# 수축 중 위치별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 중 가격 위치별 분석")
print("=" * 80)

print(f"\n{'위치':>10} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10} {'UP돌파':>10}")
print("-" * 65)

for position in ['UPPER', 'MIDDLE', 'LOWER']:
    subset = df_results[df_results['squeeze_position'] == position]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        up_pct = (subset['breakout_dir'] == 'UP').mean() * 100
        print(f"{position:>10} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}% {up_pct:>10.0f}%")

# ============================================================
# 추세 전환 vs 지속
# ============================================================
print("\n" + "=" * 80)
print("추세 전환 vs 지속")
print("=" * 80)

print(f"\n{'유형':>20} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 65)

for change_type in ['REVERSAL_UP', 'REVERSAL_DOWN', 'CONTINUATION', 'NEUTRAL']:
    subset = df_results[df_results['trend_change'] == change_type]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{change_type:>20} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# RSI 구간별
# ============================================================
print("\n" + "=" * 80)
print("RSI 구간별")
print("=" * 80)

print(f"\n{'RSI':>10} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 55)

for low, high in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    subset = df_results[(df_results['rsi'] >= low) & (df_results['rsi'] < high)]
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}':>10} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 밴드폭 절대값
# ============================================================
print("\n" + "=" * 80)
print("밴드폭 절대값별")
print("=" * 80)

print(f"\n{'밴드폭':>10} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 55)

for low, high in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5)]:
    subset = df_results[(df_results['min_width'] >= low) & (df_results['min_width'] < high)]
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}%':>10} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>45} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 90)

conditions = [
    ('전체', df_results),
    ('상단수축 + 상승돌파', df_results[(df_results['squeeze_position'] == 'UPPER') & (df_results['breakout_dir'] == 'UP')]),
    ('하단수축 + 하락돌파', df_results[(df_results['squeeze_position'] == 'LOWER') & (df_results['breakout_dir'] == 'DOWN')]),
    ('상단수축 + 하락돌파 (전환)', df_results[(df_results['squeeze_position'] == 'UPPER') & (df_results['breakout_dir'] == 'DOWN')]),
    ('하단수축 + 상승돌파 (전환)', df_results[(df_results['squeeze_position'] == 'LOWER') & (df_results['breakout_dir'] == 'UP')]),
    ('추세지속 (CONTINUATION)', df_results[df_results['trend_change'] == 'CONTINUATION']),
    ('밴드폭<1% + 상승돌파', df_results[(df_results['min_width'] < 1) & (df_results['breakout_dir'] == 'UP')]),
    ('밴드폭<1% + 하락돌파', df_results[(df_results['min_width'] < 1) & (df_results['breakout_dir'] == 'DOWN')]),
    ('RSI<30 + 상승돌파', df_results[(df_results['rsi'] < 30) & (df_results['breakout_dir'] == 'UP')]),
    ('RSI>70 + 하락돌파', df_results[(df_results['rsi'] > 70) & (df_results['breakout_dir'] == 'DOWN')]),
    ('거래량>2x + 상승돌파', df_results[(df_results['vol_ratio'] > 2) & (df_results['breakout_dir'] == 'UP')]),
    ('거래량>2x + 하락돌파', df_results[(df_results['vol_ratio'] > 2) & (df_results['breakout_dir'] == 'DOWN')]),
    ('상단 + RSI>60 + 상승', df_results[(df_results['squeeze_position'] == 'UPPER') & (df_results['rsi'] > 60) & (df_results['breakout_dir'] == 'UP')]),
    ('하단 + RSI<40 + 하락', df_results[(df_results['squeeze_position'] == 'LOWER') & (df_results['rsi'] < 40) & (df_results['breakout_dir'] == 'DOWN')]),
]

for name, subset in conditions:
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>45} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론 (15분봉)")
print("=" * 80)

upper_up = df_results[(df_results['squeeze_position'] == 'UPPER') & (df_results['breakout_dir'] == 'UP')]
lower_down = df_results[(df_results['squeeze_position'] == 'LOWER') & (df_results['breakout_dir'] == 'DOWN')]

print(f"""
■ 수축 중 위치 → 돌파 방향:
  - 상단(UPPER): 상승돌파 {(df_results[df_results['squeeze_position']=='UPPER']['breakout_dir']=='UP').mean()*100:.0f}%
  - 하단(LOWER): 하락돌파 {(df_results[df_results['squeeze_position']=='LOWER']['breakout_dir']=='DOWN').mean()*100:.0f}%

■ 최적 전략:
  - 상단수축 + 상승돌파: {len(upper_up)}건, 승률 {(upper_up['pnl_20h']>0).mean()*100:.1f}%, PnL {upper_up['pnl_20h'].mean():.2f}%
  - 하단수축 + 하락돌파: {len(lower_down)}건, 승률 {(lower_down['pnl_20h']>0).mean()*100:.1f}%, PnL {lower_down['pnl_20h'].mean():.2f}%
""")

df_results.to_csv('bb_trend_reversal_15m_results.csv', index=False)
print("결과 저장: bb_trend_reversal_15m_results.csv")
