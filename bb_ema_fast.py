#!/usr/bin/env python3
"""
BB 중간선 익절 + EMA 필터 (최적화 버전)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 중간선 익절 + EMA 필터")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"\n15M 데이터: {len(df):,}개")

# BB 계산
period = 30
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

# EMA 계산
for p in [20, 50, 100, 200]:
    df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# NaN 제거
df = df.iloc[200:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값: {width_threshold:.3f}%")

# numpy 배열로 변환 (속도 향상)
close = df['close'].values
high = df['high'].values
low = df['low'].values
bb_mid = df['bb_mid'].values
bb_upper = df['bb_upper'].values
bb_lower = df['bb_lower'].values
bb_width = df['bb_width'].values
ema_20 = df['ema_20'].values
ema_50 = df['ema_50'].values
ema_100 = df['ema_100'].values
ema_200 = df['ema_200'].values
datetimes = df['datetime'].values

# ============================================================
# 돌파 신호 미리 찾기
# ============================================================
print("\n돌파 신호 탐색 중...")

signals = []
min_squeeze = 4

for i in range(50, len(df) - 100):
    if bb_width[i-1] <= width_threshold and bb_width[i] > width_threshold:
        # 수축 구간 찾기
        squeeze_start = i - 1
        while squeeze_start > 0 and bb_width[squeeze_start] <= width_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < min_squeeze:
            continue
        
        # 돌파 방향
        if close[i] > bb_upper[i-1]:
            direction = 'LONG'
        elif close[i] < bb_lower[i-1]:
            direction = 'SHORT'
        else:
            continue
        
        signals.append({
            'idx': i,
            'direction': direction,
            'squeeze_duration': squeeze_duration,
            'close': close[i],
            'ema_20': ema_20[i],
            'ema_50': ema_50[i],
            'ema_100': ema_100[i],
            'ema_200': ema_200[i]
        })

print(f"돌파 신호: {len(signals)}건")

# ============================================================
# 시뮬레이션 함수 (최적화)
# ============================================================
def simulate_trade_fast(idx, direction, max_holding=80):
    entry_price = close[idx]
    band_width = bb_upper[idx] - bb_lower[idx]
    
    if direction == 'LONG':
        stop_loss = entry_price - band_width * 0.5
    else:
        stop_loss = entry_price + band_width * 0.5
    
    went_outside = False
    max_favorable = 0
    
    for j in range(idx + 1, min(idx + max_holding + 1, len(close))):
        if direction == 'LONG':
            favorable = (high[j] - entry_price) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            
            if high[j] > bb_upper[j]:
                went_outside = True
            
            if low[j] <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                return {'pnl': pnl, 'exit_reason': 'STOP_LOSS', 'bars': j - idx, 'mfe': max_favorable}
            
            if went_outside and close[j] <= bb_mid[j]:
                pnl = (close[j] - entry_price) / entry_price * 100
                return {'pnl': pnl, 'exit_reason': 'MID_LINE_TP', 'bars': j - idx, 'mfe': max_favorable}
        else:
            favorable = (entry_price - low[j]) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            
            if low[j] < bb_lower[j]:
                went_outside = True
            
            if high[j] >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price * 100
                return {'pnl': pnl, 'exit_reason': 'STOP_LOSS', 'bars': j - idx, 'mfe': max_favorable}
            
            if went_outside and close[j] >= bb_mid[j]:
                pnl = (entry_price - close[j]) / entry_price * 100
                return {'pnl': pnl, 'exit_reason': 'MID_LINE_TP', 'bars': j - idx, 'mfe': max_favorable}
    
    # 시간 만료
    final_idx = min(idx + max_holding, len(close) - 1)
    if direction == 'LONG':
        pnl = (close[final_idx] - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - close[final_idx]) / entry_price * 100
    return {'pnl': pnl, 'exit_reason': 'TIME_OUT', 'bars': max_holding, 'mfe': max_favorable}

# ============================================================
# 모든 신호 시뮬레이션
# ============================================================
print("\n시뮬레이션 실행 중...")

for sig in signals:
    result = simulate_trade_fast(sig['idx'], sig['direction'])
    sig.update(result)

df_signals = pd.DataFrame(signals)
print(f"시뮬레이션 완료: {len(df_signals)}건")

# ============================================================
# 기준선 (EMA 필터 없음)
# ============================================================
print("\n" + "=" * 80)
print("★ 기준선: EMA 필터 없음 ★")
print("=" * 80)

baseline = df_signals.copy()
print(f"\n전체 건수: {len(baseline)}")
print(f"평균 PnL: {baseline['pnl'].mean():.2f}%")
print(f"승률: {(baseline['pnl'] > 0).mean() * 100:.1f}%")
print(f"손절률: {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%")
print(f"익절률: {(baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:.1f}%")

# 익절만
tp_only = baseline[baseline['exit_reason'] == 'MID_LINE_TP']
print(f"\n익절 케이스: {len(tp_only)}건")
print(f"  평균 PnL: {tp_only['pnl'].mean():.2f}%")
print(f"  승률: {(tp_only['pnl'] > 0).mean() * 100:.1f}%")

# ============================================================
# EMA 필터 테스트
# ============================================================
print("\n" + "=" * 80)
print("★ EMA 트렌드 필터 효과 ★")
print("=" * 80)

print(f"\n{'EMA':>10} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'손절률':>10} {'익절률':>10}")
print("-" * 70)

print(f"{'없음':>10} {len(baseline):>8} {baseline['pnl'].mean():>10.2f}% {(baseline['pnl'] > 0).mean() * 100:>10.1f}% {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:>10.1f}% {(baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")

for ema_p in [20, 50, 100, 200]:
    ema_col = f'ema_{ema_p}'
    # 트렌드 필터: LONG은 EMA 위, SHORT는 EMA 아래
    filtered = df_signals[
        ((df_signals['direction'] == 'LONG') & (df_signals['close'] > df_signals[ema_col])) |
        ((df_signals['direction'] == 'SHORT') & (df_signals['close'] < df_signals[ema_col]))
    ]
    
    if len(filtered) > 0:
        sl_rate = (filtered['exit_reason'] == 'STOP_LOSS').mean() * 100
        tp_rate = (filtered['exit_reason'] == 'MID_LINE_TP').mean() * 100
        print(f"{f'EMA{ema_p}':>10} {len(filtered):>8} {filtered['pnl'].mean():>10.2f}% {(filtered['pnl'] > 0).mean() * 100:>10.1f}% {sl_rate:>10.1f}% {tp_rate:>10.1f}%")

# ============================================================
# 방향별 상세 분석 (EMA200)
# ============================================================
print("\n" + "=" * 80)
print("★ EMA200 방향별 상세 분석 ★")
print("=" * 80)

print(f"\n{'조건':>45} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'익절PnL':>10} {'익절승률':>10}")
print("-" * 100)

# LONG + EMA200 위 (트렌드)
long_above = df_signals[(df_signals['direction'] == 'LONG') & (df_signals['close'] > df_signals['ema_200'])]
if len(long_above) > 0:
    tp = long_above[long_above['exit_reason'] == 'MID_LINE_TP']
    print(f"{'LONG + 가격>EMA200 (상승트렌드)':>45} {len(long_above):>8} {long_above['pnl'].mean():>10.2f}% {(long_above['pnl'] > 0).mean() * 100:>10.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}% {(tp['pnl'] > 0).mean() * 100 if len(tp) > 0 else 0:>10.1f}%")

# LONG + EMA200 아래 (역추세)
long_below = df_signals[(df_signals['direction'] == 'LONG') & (df_signals['close'] < df_signals['ema_200'])]
if len(long_below) > 0:
    tp = long_below[long_below['exit_reason'] == 'MID_LINE_TP']
    print(f"{'LONG + 가격<EMA200 (역추세반등)':>45} {len(long_below):>8} {long_below['pnl'].mean():>10.2f}% {(long_below['pnl'] > 0).mean() * 100:>10.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}% {(tp['pnl'] > 0).mean() * 100 if len(tp) > 0 else 0:>10.1f}%")

# SHORT + EMA200 아래 (트렌드)
short_below = df_signals[(df_signals['direction'] == 'SHORT') & (df_signals['close'] < df_signals['ema_200'])]
if len(short_below) > 0:
    tp = short_below[short_below['exit_reason'] == 'MID_LINE_TP']
    print(f"{'SHORT + 가격<EMA200 (하락트렌드)':>45} {len(short_below):>8} {short_below['pnl'].mean():>10.2f}% {(short_below['pnl'] > 0).mean() * 100:>10.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}% {(tp['pnl'] > 0).mean() * 100 if len(tp) > 0 else 0:>10.1f}%")

# SHORT + EMA200 위 (역추세)
short_above = df_signals[(df_signals['direction'] == 'SHORT') & (df_signals['close'] > df_signals['ema_200'])]
if len(short_above) > 0:
    tp = short_above[short_above['exit_reason'] == 'MID_LINE_TP']
    print(f"{'SHORT + 가격>EMA200 (역추세하락)':>45} {len(short_above):>8} {short_above['pnl'].mean():>10.2f}% {(short_above['pnl'] > 0).mean() * 100:>10.1f}% {tp['pnl'].mean() if len(tp) > 0 else 0:>10.2f}% {(tp['pnl'] > 0).mean() * 100 if len(tp) > 0 else 0:>10.1f}%")

# ============================================================
# 복합 필터: EMA + 수축 기간
# ============================================================
print("\n" + "=" * 80)
print("★ EMA200 트렌드 + 수축 기간 조합 ★")
print("=" * 80)

# EMA200 트렌드 필터된 데이터
ema200_trend = df_signals[
    ((df_signals['direction'] == 'LONG') & (df_signals['close'] > df_signals['ema_200'])) |
    ((df_signals['direction'] == 'SHORT') & (df_signals['close'] < df_signals['ema_200']))
]

print(f"\n{'조건':>45} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'손절률':>10} {'익절률':>10}")
print("-" * 100)

for dur_low, dur_high in [(4, 10), (10, 20), (20, 50), (50, 500)]:
    subset = ema200_trend[(ema200_trend['squeeze_duration'] >= dur_low) & (ema200_trend['squeeze_duration'] < dur_high)]
    if len(subset) >= 10:
        sl_rate = (subset['exit_reason'] == 'STOP_LOSS').mean() * 100
        tp_rate = (subset['exit_reason'] == 'MID_LINE_TP').mean() * 100
        print(f"{f'EMA200트렌드 + 수축{dur_low}-{dur_high}봉':>45} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {sl_rate:>10.1f}% {tp_rate:>10.1f}%")

# ============================================================
# 최적 조건 탐색
# ============================================================
print("\n" + "=" * 80)
print("★ 최적 조건 TOP 10 ★")
print("=" * 80)

results = []

for ema_p in [20, 50, 100, 200]:
    ema_col = f'ema_{ema_p}'
    
    for mode in ['trend', 'counter']:
        if mode == 'trend':
            filtered = df_signals[
                ((df_signals['direction'] == 'LONG') & (df_signals['close'] > df_signals[ema_col])) |
                ((df_signals['direction'] == 'SHORT') & (df_signals['close'] < df_signals[ema_col]))
            ]
        else:
            filtered = df_signals[
                ((df_signals['direction'] == 'LONG') & (df_signals['close'] < df_signals[ema_col])) |
                ((df_signals['direction'] == 'SHORT') & (df_signals['close'] > df_signals[ema_col]))
            ]
        
        for dur_low, dur_high in [(4, 10), (10, 20), (20, 50), (50, 500)]:
            subset = filtered[(filtered['squeeze_duration'] >= dur_low) & (filtered['squeeze_duration'] < dur_high)]
            
            if len(subset) >= 20:
                tp_only = subset[subset['exit_reason'] == 'MID_LINE_TP']
                if len(tp_only) >= 10:
                    results.append({
                        'EMA': f'EMA{ema_p}',
                        'Mode': mode,
                        'Squeeze': f'{dur_low}-{dur_high}',
                        'Count': len(subset),
                        'TP_Count': len(tp_only),
                        'PnL': subset['pnl'].mean(),
                        'WinRate': (subset['pnl'] > 0).mean() * 100,
                        'SL_Rate': (subset['exit_reason'] == 'STOP_LOSS').mean() * 100,
                        'TP_Rate': (subset['exit_reason'] == 'MID_LINE_TP').mean() * 100,
                        'TP_PnL': tp_only['pnl'].mean(),
                        'TP_WinRate': (tp_only['pnl'] > 0).mean() * 100
                    })

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    # 승률 기준 정렬
    df_results = df_results.sort_values('WinRate', ascending=False)
    
    print(f"\n{'EMA':>8} {'Mode':>8} {'수축':>10} {'건수':>6} {'PnL':>8} {'승률':>8} {'손절률':>8} {'익절률':>8}")
    print("-" * 80)
    
    for _, row in df_results.head(10).iterrows():
        print(f"{row['EMA']:>8} {row['Mode']:>8} {row['Squeeze']:>10} {row['Count']:>6} {row['PnL']:>8.2f}% {row['WinRate']:>8.1f}% {row['SL_Rate']:>8.1f}% {row['TP_Rate']:>8.1f}%")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 결론 ★★★")
print("=" * 80)

# 최적 조건
if len(df_results) > 0:
    best = df_results.iloc[0]
    
    print(f"""
■ 기준선 (필터 없음):
  - 건수: {len(baseline)}건
  - 평균 PnL: {baseline['pnl'].mean():.2f}%
  - 승률: {(baseline['pnl'] > 0).mean() * 100:.1f}%
  - 손절률: {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%

■ 최적 조건 ({best['EMA']} {best['Mode']} + 수축 {best['Squeeze']}봉):
  - 건수: {best['Count']}건
  - 평균 PnL: {best['PnL']:.2f}%
  - 승률: {best['WinRate']:.1f}%
  - 손절률: {best['SL_Rate']:.1f}%
  - 익절률: {best['TP_Rate']:.1f}%

■ EMA 필터 효과:
  - 승률: {(baseline['pnl'] > 0).mean() * 100:.1f}% → {best['WinRate']:.1f}% ({'+' if best['WinRate'] > (baseline['pnl'] > 0).mean() * 100 else ''}{best['WinRate'] - (baseline['pnl'] > 0).mean() * 100:.1f}%p)
  - 손절률: {(baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}% → {best['SL_Rate']:.1f}% ({'+' if best['SL_Rate'] > (baseline['exit_reason'] == 'STOP_LOSS').mean() * 100 else ''}{best['SL_Rate'] - (baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:.1f}%p)

■ 권장 전략:
  - EMA200 트렌드 필터 사용 (가격>EMA면 LONG만, 가격<EMA면 SHORT만)
  - 수축 기간 20봉 이상 선호
  - 중간선 도달 시 익절
""")

# 저장
df_signals.to_csv('bb_ema_all_signals.csv', index=False)
df_results.to_csv('bb_ema_optimization.csv', index=False)
print(f"\n저장: bb_ema_all_signals.csv, bb_ema_optimization.csv")
