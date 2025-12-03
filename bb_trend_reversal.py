#!/usr/bin/env python3
"""
볼린저밴드 수축 후 추세 전환 분석

핵심 질문: 어떤 상황에서 추세가 바뀌는가?

분석 항목:
1. 수축 전 추세 (상승/하락/횡보)
2. 수축 중 가격 위치 (밴드 상단/중간/하단)
3. 돌파 방향 vs 기존 추세
4. 추세 전환 vs 추세 지속
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 수축 후 추세 전환 분석")
print("=" * 80)

# 데이터 로드
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

df = df_15m.set_index('datetime').resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

print(f"1H 데이터: {len(df):,}개")

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

# 추세 판단용 이동평균
df['ma20'] = df['close'].rolling(20).mean()
df['ma50'] = df['close'].rolling(50).mean()
df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(10)) / df['ma20'].shift(10) * 100  # 10봉 기울기

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

# 거래량 이동평균
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

df = df.dropna().reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# ============================================================
# 수축 구간 분석
# ============================================================
print("\n수축 구간 분석 중...")

squeeze_threshold = 20  # 백분위 20% 이하
results = []

for i in range(100, len(df) - 50):
    current_pct = df.iloc[i]['bb_width_pct']
    prev_pct = df.iloc[i-1]['bb_width_pct']
    
    # 수축 → 확장 전환점
    if prev_pct <= squeeze_threshold and current_pct > squeeze_threshold:
        
        # 수축 시작점 찾기
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width_pct'] <= squeeze_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < 3:
            continue
        
        # 수축 전 구간 (수축 시작 전 20봉)
        pre_squeeze = df.iloc[max(0, squeeze_start-20):squeeze_start]
        
        # 수축 중 구간
        during_squeeze = df.iloc[squeeze_start:i]
        
        # 돌파 캔들
        breakout = df.iloc[i]
        
        # === 수축 전 추세 ===
        if len(pre_squeeze) >= 10:
            pre_trend_pct = (pre_squeeze.iloc[-1]['close'] - pre_squeeze.iloc[0]['close']) / pre_squeeze.iloc[0]['close'] * 100
            if pre_trend_pct > 2:
                pre_trend = 'UP'
            elif pre_trend_pct < -2:
                pre_trend = 'DOWN'
            else:
                pre_trend = 'SIDE'
        else:
            pre_trend = 'UNKNOWN'
            pre_trend_pct = 0
        
        # === 수축 중 가격 위치 ===
        avg_bb_position = during_squeeze['bb_position'].mean()
        if avg_bb_position > 60:
            squeeze_position = 'UPPER'  # 밴드 상단
        elif avg_bb_position < 40:
            squeeze_position = 'LOWER'  # 밴드 하단
        else:
            squeeze_position = 'MIDDLE'  # 중간
        
        # === 돌파 방향 ===
        if breakout['close'] > breakout['bb_upper']:
            breakout_dir = 'UP'
        elif breakout['close'] < breakout['bb_lower']:
            breakout_dir = 'DOWN'
        else:
            if breakout['close'] > df.iloc[i-1]['close']:
                breakout_dir = 'UP'
            else:
                breakout_dir = 'DOWN'
        
        # === 추세 전환 여부 ===
        if pre_trend == 'UP' and breakout_dir == 'DOWN':
            trend_change = 'REVERSAL_DOWN'
        elif pre_trend == 'DOWN' and breakout_dir == 'UP':
            trend_change = 'REVERSAL_UP'
        elif pre_trend == breakout_dir:
            trend_change = 'CONTINUATION'
        else:
            trend_change = 'NEUTRAL'
        
        # === 추가 지표 ===
        rsi_at_breakout = breakout['rsi']
        vol_at_breakout = breakout['vol_ratio']
        ma_slope = breakout['ma20_slope']
        min_width = during_squeeze['bb_width'].min()
        
        # === 결과 측정 (20H, 50H) ===
        future = df.iloc[i+1:i+51]
        if len(future) < 20:
            continue
        
        entry_price = breakout['close']
        
        if breakout_dir == 'UP':
            mfe = (future['high'].max() - entry_price) / entry_price * 100
            mae = (future['low'].min() - entry_price) / entry_price * 100
            pnl_20h = (future.iloc[19]['close'] - entry_price) / entry_price * 100
            pnl_50h = (future.iloc[-1]['close'] - entry_price) / entry_price * 100
        else:
            mfe = (entry_price - future['low'].min()) / entry_price * 100
            mae = (entry_price - future['high'].max()) / entry_price * 100
            pnl_20h = (entry_price - future.iloc[19]['close']) / entry_price * 100
            pnl_50h = (entry_price - future.iloc[-1]['close']) / entry_price * 100
        
        results.append({
            'time': breakout['datetime'],
            'pre_trend': pre_trend,
            'pre_trend_pct': pre_trend_pct,
            'squeeze_position': squeeze_position,
            'avg_bb_position': avg_bb_position,
            'breakout_dir': breakout_dir,
            'trend_change': trend_change,
            'squeeze_duration': squeeze_duration,
            'min_width': min_width,
            'rsi': rsi_at_breakout,
            'vol_ratio': vol_at_breakout,
            'ma_slope': ma_slope,
            'mfe': mfe,
            'mae': mae,
            'pnl_20h': pnl_20h,
            'pnl_50h': pnl_50h
        })

df_results = pd.DataFrame(results)
print(f"분석 완료: {len(df_results)}건")

# ============================================================
# 추세 전환 vs 지속 분석
# ============================================================
print("\n" + "=" * 80)
print("추세 전환 vs 지속 분석")
print("=" * 80)

print(f"\n{'유형':>20} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 65)

for change_type in ['REVERSAL_UP', 'REVERSAL_DOWN', 'CONTINUATION', 'NEUTRAL']:
    subset = df_results[df_results['trend_change'] == change_type]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{change_type:>20} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 수축 전 추세별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 전 추세별 분석")
print("=" * 80)

print(f"\n{'수축 전 추세':>15} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 60)

for pre_trend in ['UP', 'DOWN', 'SIDE']:
    subset = df_results[df_results['pre_trend'] == pre_trend]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{pre_trend:>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 수축 중 위치별 분석
# ============================================================
print("\n" + "=" * 80)
print("수축 중 가격 위치별 분석")
print("=" * 80)

print(f"\n{'위치':>15} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 60)

for position in ['UPPER', 'MIDDLE', 'LOWER']:
    subset = df_results[df_results['squeeze_position'] == position]
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        # 각 위치에서 돌파 방향 분포
        up_pct = (subset['breakout_dir'] == 'UP').mean() * 100
        print(f"{position:>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}% (UP:{up_pct:.0f}%)")

# ============================================================
# RSI 구간별 분석
# ============================================================
print("\n" + "=" * 80)
print("돌파 시점 RSI별 분석")
print("=" * 80)

print(f"\n{'RSI':>15} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 60)

for low, high in [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]:
    subset = df_results[(df_results['rsi'] >= low) & (df_results['rsi'] < high)]
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}':>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 거래량 분석
# ============================================================
print("\n" + "=" * 80)
print("돌파 시점 거래량별 분석")
print("=" * 80)

print(f"\n{'거래량 비율':>15} {'건수':>8} {'MFE':>10} {'PnL_20H':>10} {'승률':>10}")
print("-" * 60)

for low, high in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 10)]:
    subset = df_results[(df_results['vol_ratio'] >= low) & (df_results['vol_ratio'] < high)]
    if len(subset) >= 10:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{f'{low}-{high}x':>15} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 조합 분석 - 추세 전환 조건
# ============================================================
print("\n" + "=" * 80)
print("추세 전환 최적 조건 탐색")
print("=" * 80)

print(f"\n{'조건':>50} {'건수':>8} {'MFE':>10} {'PnL':>10} {'승률':>10}")
print("-" * 95)

conditions = [
    ('전체', df_results),
    
    # 하락 후 상승 전환
    ('하락추세 후 상승돌파 (REVERSAL_UP)', df_results[df_results['trend_change'] == 'REVERSAL_UP']),
    ('하락추세 + 하단수축 + 상승돌파', df_results[(df_results['pre_trend'] == 'DOWN') & (df_results['squeeze_position'] == 'LOWER') & (df_results['breakout_dir'] == 'UP')]),
    ('하락추세 + RSI<40 + 상승돌파', df_results[(df_results['pre_trend'] == 'DOWN') & (df_results['rsi'] < 40) & (df_results['breakout_dir'] == 'UP')]),
    ('하락추세 + 하단 + RSI<40 + 상승돌파', df_results[(df_results['pre_trend'] == 'DOWN') & (df_results['squeeze_position'] == 'LOWER') & (df_results['rsi'] < 40) & (df_results['breakout_dir'] == 'UP')]),
    
    # 상승 후 하락 전환
    ('상승추세 후 하락돌파 (REVERSAL_DOWN)', df_results[df_results['trend_change'] == 'REVERSAL_DOWN']),
    ('상승추세 + 상단수축 + 하락돌파', df_results[(df_results['pre_trend'] == 'UP') & (df_results['squeeze_position'] == 'UPPER') & (df_results['breakout_dir'] == 'DOWN')]),
    ('상승추세 + RSI>60 + 하락돌파', df_results[(df_results['pre_trend'] == 'UP') & (df_results['rsi'] > 60) & (df_results['breakout_dir'] == 'DOWN')]),
    
    # 추세 지속
    ('추세 지속 (CONTINUATION)', df_results[df_results['trend_change'] == 'CONTINUATION']),
    ('상승추세 + 상승돌파', df_results[(df_results['pre_trend'] == 'UP') & (df_results['breakout_dir'] == 'UP')]),
    ('하락추세 + 하락돌파', df_results[(df_results['pre_trend'] == 'DOWN') & (df_results['breakout_dir'] == 'DOWN')]),
    
    # 거래량 조합
    ('상승돌파 + 거래량>1.5x', df_results[(df_results['breakout_dir'] == 'UP') & (df_results['vol_ratio'] > 1.5)]),
    ('하락돌파 + 거래량>1.5x', df_results[(df_results['breakout_dir'] == 'DOWN') & (df_results['vol_ratio'] > 1.5)]),
    
    # 수축 강도
    ('밴드폭<3% + 상승돌파', df_results[(df_results['min_width'] < 3) & (df_results['breakout_dir'] == 'UP')]),
    ('밴드폭<3% + 하락돌파', df_results[(df_results['min_width'] < 3) & (df_results['breakout_dir'] == 'DOWN')]),
]

for name, subset in conditions:
    if len(subset) >= 5:
        win_rate = (subset['pnl_20h'] > 0).mean() * 100
        print(f"{name:>50} {len(subset):>8} {subset['mfe'].mean():>10.2f}% {subset['pnl_20h'].mean():>10.2f}% {win_rate:>10.1f}%")

# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

# 최고 조건 찾기
best_name = None
best_pnl = -999
for name, subset in conditions:
    if len(subset) >= 20:
        avg_pnl = subset['pnl_20h'].mean()
        if avg_pnl > best_pnl:
            best_pnl = avg_pnl
            best_name = name

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                    추세 전환 분석 결과                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ■ 총 분석: {len(df_results)}건                                                          │
│                                                                            │
│  ■ 추세 전환 vs 지속:                                                        │
│    - REVERSAL_UP (하락→상승): {len(df_results[df_results['trend_change']=='REVERSAL_UP'])}건                                       │
│    - REVERSAL_DOWN (상승→하락): {len(df_results[df_results['trend_change']=='REVERSAL_DOWN'])}건                                     │
│    - CONTINUATION (추세지속): {len(df_results[df_results['trend_change']=='CONTINUATION'])}건                                       │
│                                                                            │
│  ■ 최적 조건: {best_name if best_name else 'N/A'}
│    - 평균 PnL: {best_pnl:.2f}%                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

# 저장
df_results.to_csv('bb_trend_reversal_results.csv', index=False)
print("결과 저장: bb_trend_reversal_results.csv")
