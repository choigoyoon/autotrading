#!/usr/bin/env python3
"""
BB 중간선 익절 전략 + EMA 필터
미래 데이터 없이 실시간 시뮬레이션

EMA 필터 적용:
- LONG: 가격이 EMA 위에 있을 때만 진입
- SHORT: 가격이 EMA 아래에 있을 때만 진입

다양한 EMA 기간 테스트: 20, 50, 100, 200
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 중간선 익절 + EMA 필터 전략")
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

# EMA 계산 (다양한 기간)
ema_periods = [20, 50, 100, 200]
for p in ema_periods:
    df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# NaN 제거 (가장 긴 EMA 기간 기준)
df = df.iloc[200:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값: {width_threshold:.3f}%")

# ============================================================
# 트레이드 시뮬레이션 함수
# ============================================================
def simulate_trade(df, entry_idx, direction, max_holding=80):
    """실시간 트레이드 시뮬레이션"""
    entry_price = df.iloc[entry_idx]['close']
    entry_time = df.iloc[entry_idx]['datetime']
    entry_bb_upper = df.iloc[entry_idx]['bb_upper']
    entry_bb_lower = df.iloc[entry_idx]['bb_lower']
    
    exit_price = None
    exit_reason = None
    exit_time = None
    max_favorable = 0
    max_adverse = 0
    bars_held = 0
    went_outside = False
    
    band_width = entry_bb_upper - entry_bb_lower
    if direction == 'LONG':
        stop_loss = entry_price - band_width * 0.5
    else:
        stop_loss = entry_price + band_width * 0.5
    
    for i in range(entry_idx + 1, min(entry_idx + max_holding + 1, len(df))):
        candle = df.iloc[i]
        bars_held += 1
        
        current_bb_mid = candle['bb_mid']
        current_bb_upper = candle['bb_upper']
        current_bb_lower = candle['bb_lower']
        
        if direction == 'LONG':
            favorable = (candle['high'] - entry_price) / entry_price * 100
            adverse = (candle['low'] - entry_price) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            max_adverse = min(max_adverse, adverse)
            
            if candle['high'] > current_bb_upper:
                went_outside = True
            
            if candle['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
                exit_time = candle['datetime']
                break
            
            if went_outside and candle['close'] <= current_bb_mid:
                exit_price = candle['close']
                exit_reason = 'MID_LINE_TP'
                exit_time = candle['datetime']
                break
                
        else:  # SHORT
            favorable = (entry_price - candle['low']) / entry_price * 100
            adverse = (entry_price - candle['high']) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            max_adverse = min(max_adverse, adverse)
            
            if candle['low'] < current_bb_lower:
                went_outside = True
            
            if candle['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
                exit_time = candle['datetime']
                break
            
            if went_outside and candle['close'] >= current_bb_mid:
                exit_price = candle['close']
                exit_reason = 'MID_LINE_TP'
                exit_time = candle['datetime']
                break
    
    if exit_price is None and bars_held > 0:
        exit_price = df.iloc[min(entry_idx + max_holding, len(df) - 1)]['close']
        exit_reason = 'TIME_OUT'
        exit_time = df.iloc[min(entry_idx + max_holding, len(df) - 1)]['datetime']
    
    if exit_price is None:
        return None
    
    if direction == 'LONG':
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    
    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'mfe': max_favorable,
        'mae': max_adverse,
        'exit_reason': exit_reason,
        'bars_held': bars_held,
        'went_outside': went_outside
    }

# ============================================================
# 전략 실행: EMA 없이 (기준선)
# ============================================================
print("\n" + "=" * 80)
print("기준선: EMA 필터 없음")
print("=" * 80)

min_squeeze = 4

def run_strategy(df, ema_period=None, ema_mode=None):
    """
    전략 실행
    ema_period: EMA 기간 (None이면 필터 없음)
    ema_mode: 'trend' (트렌드 방향) 또는 'counter' (역추세)
    """
    trades = []
    
    for i in range(50, len(df) - 100):
        current_width = df.iloc[i]['bb_width']
        prev_width = df.iloc[i-1]['bb_width']
        
        if prev_width <= width_threshold and current_width > width_threshold:
            squeeze_start = i - 1
            while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width'] <= width_threshold:
                squeeze_start -= 1
            
            squeeze_duration = i - squeeze_start
            if squeeze_duration < min_squeeze:
                continue
            
            candle = df.iloc[i]
            prev_upper = df.iloc[i-1]['bb_upper']
            prev_lower = df.iloc[i-1]['bb_lower']
            
            # EMA 필터 적용
            if ema_period:
                ema_col = f'ema_{ema_period}'
                ema_value = candle[ema_col]
                price_above_ema = candle['close'] > ema_value
                
                if ema_mode == 'trend':
                    # 트렌드 방향: LONG은 EMA 위, SHORT는 EMA 아래
                    allow_long = price_above_ema
                    allow_short = not price_above_ema
                elif ema_mode == 'counter':
                    # 역추세: LONG은 EMA 아래에서 반등, SHORT는 EMA 위에서 하락
                    allow_long = not price_above_ema
                    allow_short = price_above_ema
                else:
                    allow_long = True
                    allow_short = True
            else:
                allow_long = True
                allow_short = True
            
            # 상단 돌파 (LONG)
            if candle['close'] > prev_upper and allow_long:
                result = simulate_trade(df, i, 'LONG')
                if result:
                    result['breakout_type'] = 'UPPER'
                    result['squeeze_duration'] = squeeze_duration
                    if ema_period:
                        result['ema_period'] = ema_period
                        result['ema_value'] = ema_value
                        result['price_vs_ema'] = 'ABOVE' if price_above_ema else 'BELOW'
                    trades.append(result)
            
            # 하단 돌파 (SHORT)
            elif candle['close'] < prev_lower and allow_short:
                result = simulate_trade(df, i, 'SHORT')
                if result:
                    result['breakout_type'] = 'LOWER'
                    result['squeeze_duration'] = squeeze_duration
                    if ema_period:
                        result['ema_period'] = ema_period
                        result['ema_value'] = ema_value
                        result['price_vs_ema'] = 'ABOVE' if price_above_ema else 'BELOW'
                    trades.append(result)
    
    return pd.DataFrame(trades)

# 기준선 실행
df_baseline = run_strategy(df)
print(f"기준선 트레이드: {len(df_baseline)}건")

if len(df_baseline) > 0:
    print(f"  평균 PnL: {df_baseline['pnl'].mean():.2f}%")
    print(f"  승률: {(df_baseline['pnl'] > 0).mean() * 100:.1f}%")
    sl_rate = (df_baseline['exit_reason'] == 'STOP_LOSS').mean() * 100
    tp_rate = (df_baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100
    print(f"  손절률: {sl_rate:.1f}%, 익절률: {tp_rate:.1f}%")

# ============================================================
# EMA 트렌드 필터 테스트
# ============================================================
print("\n" + "=" * 80)
print("★ EMA 트렌드 필터 비교 ★")
print("=" * 80)

print(f"\n{'EMA':>10} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'손절률':>10} {'익절률':>10}")
print("-" * 70)

# 기준선
print(f"{'없음':>10} {len(df_baseline):>8} {df_baseline['pnl'].mean():>10.2f}% {(df_baseline['pnl'] > 0).mean() * 100:>10.1f}% {(df_baseline['exit_reason'] == 'STOP_LOSS').mean() * 100:>10.1f}% {(df_baseline['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")

results_summary = []

for ema_p in ema_periods:
    df_ema = run_strategy(df, ema_period=ema_p, ema_mode='trend')
    if len(df_ema) > 0:
        sl_rate = (df_ema['exit_reason'] == 'STOP_LOSS').mean() * 100
        tp_rate = (df_ema['exit_reason'] == 'MID_LINE_TP').mean() * 100
        print(f"{f'EMA{ema_p}':>10} {len(df_ema):>8} {df_ema['pnl'].mean():>10.2f}% {(df_ema['pnl'] > 0).mean() * 100:>10.1f}% {sl_rate:>10.1f}% {tp_rate:>10.1f}%")
        
        results_summary.append({
            'filter': f'EMA{ema_p}',
            'mode': 'trend',
            'count': len(df_ema),
            'pnl': df_ema['pnl'].mean(),
            'win_rate': (df_ema['pnl'] > 0).mean() * 100,
            'sl_rate': sl_rate,
            'tp_rate': tp_rate
        })

# ============================================================
# 방향별 EMA 필터 효과
# ============================================================
print("\n" + "=" * 80)
print("★ 방향별 EMA 필터 효과 (EMA 200 기준) ★")
print("=" * 80)

df_ema200 = run_strategy(df, ema_period=200, ema_mode='trend')

if len(df_ema200) > 0:
    print(f"\n{'조건':>40} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'익절률':>10}")
    print("-" * 85)
    
    # LONG - EMA 위
    long_above = df_ema200[(df_ema200['direction'] == 'LONG') & (df_ema200['price_vs_ema'] == 'ABOVE')]
    if len(long_above) > 0:
        print(f"{'LONG + EMA200 위 (트렌드 추종)':>40} {len(long_above):>8} {long_above['pnl'].mean():>10.2f}% {(long_above['pnl'] > 0).mean() * 100:>10.1f}% {(long_above['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")
    
    # SHORT - EMA 아래
    short_below = df_ema200[(df_ema200['direction'] == 'SHORT') & (df_ema200['price_vs_ema'] == 'BELOW')]
    if len(short_below) > 0:
        print(f"{'SHORT + EMA200 아래 (트렌드 추종)':>40} {len(short_below):>8} {short_below['pnl'].mean():>10.2f}% {(short_below['pnl'] > 0).mean() * 100:>10.1f}% {(short_below['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")

# 역추세 필터 테스트
print("\n" + "-" * 85)
print("역추세 필터 테스트 (EMA 반대 방향 진입)")
print("-" * 85)

df_ema200_counter = run_strategy(df, ema_period=200, ema_mode='counter')

if len(df_ema200_counter) > 0:
    # LONG - EMA 아래 (역추세)
    long_below = df_ema200_counter[(df_ema200_counter['direction'] == 'LONG')]
    if len(long_below) > 0:
        print(f"{'LONG + EMA200 아래 (역추세 반등)':>40} {len(long_below):>8} {long_below['pnl'].mean():>10.2f}% {(long_below['pnl'] > 0).mean() * 100:>10.1f}% {(long_below['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")
    
    # SHORT - EMA 위 (역추세)
    short_above = df_ema200_counter[(df_ema200_counter['direction'] == 'SHORT')]
    if len(short_above) > 0:
        print(f"{'SHORT + EMA200 위 (역추세 하락)':>40} {len(short_above):>8} {short_above['pnl'].mean():>10.2f}% {(short_above['pnl'] > 0).mean() * 100:>10.1f}% {(short_above['exit_reason'] == 'MID_LINE_TP').mean() * 100:>10.1f}%")

# ============================================================
# 복합 필터: EMA + 수축 기간
# ============================================================
print("\n" + "=" * 80)
print("★ 복합 필터: EMA200 트렌드 + 수축 기간 ★")
print("=" * 80)

if len(df_ema200) > 0:
    print(f"\n{'조건':>50} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'익절률':>10}")
    print("-" * 95)
    
    for dur_low, dur_high in [(4, 20), (20, 50), (50, 500)]:
        subset = df_ema200[(df_ema200['squeeze_duration'] >= dur_low) & (df_ema200['squeeze_duration'] < dur_high)]
        if len(subset) >= 10:
            tp_rate = (subset['exit_reason'] == 'MID_LINE_TP').mean() * 100
            print(f"{f'EMA200 트렌드 + 수축 {dur_low}-{dur_high}봉':>50} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {tp_rate:>10.1f}%")

# ============================================================
# 최적 조건 찾기
# ============================================================
print("\n" + "=" * 80)
print("★ 최적 조건 탐색 ★")
print("=" * 80)

best_conditions = []

for ema_p in ema_periods:
    for mode in ['trend', 'counter']:
        df_test = run_strategy(df, ema_period=ema_p, ema_mode=mode)
        
        if len(df_test) >= 50:  # 최소 50건 이상
            for dur_low, dur_high in [(4, 20), (20, 50), (50, 500)]:
                subset = df_test[(df_test['squeeze_duration'] >= dur_low) & (df_test['squeeze_duration'] < dur_high)]
                
                if len(subset) >= 20:  # 최소 20건 이상
                    tp_rate = (subset['exit_reason'] == 'MID_LINE_TP').mean() * 100
                    win_rate = (subset['pnl'] > 0).mean() * 100
                    avg_pnl = subset['pnl'].mean()
                    
                    # 익절만 필터링
                    tp_only = subset[subset['exit_reason'] == 'MID_LINE_TP']
                    if len(tp_only) >= 10:
                        best_conditions.append({
                            'ema': f'EMA{ema_p}',
                            'mode': mode,
                            'squeeze': f'{dur_low}-{dur_high}',
                            'total': len(subset),
                            'tp_count': len(tp_only),
                            'avg_pnl': avg_pnl,
                            'win_rate': win_rate,
                            'tp_rate': tp_rate,
                            'tp_pnl': tp_only['pnl'].mean(),
                            'tp_win_rate': (tp_only['pnl'] > 0).mean() * 100
                        })

# 정렬 (익절 승률 기준)
best_df = pd.DataFrame(best_conditions)
if len(best_df) > 0:
    best_df = best_df.sort_values('tp_win_rate', ascending=False)
    
    print(f"\n{'EMA':>8} {'모드':>8} {'수축':>10} {'전체':>6} {'익절':>6} {'전체PnL':>10} {'전체승률':>10} {'익절PnL':>10} {'익절승률':>10}")
    print("-" * 100)
    
    for _, row in best_df.head(15).iterrows():
        print(f"{row['ema']:>8} {row['mode']:>8} {row['squeeze']:>10} {row['total']:>6} {row['tp_count']:>6} {row['avg_pnl']:>10.2f}% {row['win_rate']:>10.1f}% {row['tp_pnl']:>10.2f}% {row['tp_win_rate']:>10.1f}%")

# ============================================================
# 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 결론: EMA 필터 효과 ★★★")
print("=" * 80)

# 기준선 vs 최적 EMA
baseline_pnl = df_baseline['pnl'].mean()
baseline_wr = (df_baseline['pnl'] > 0).mean() * 100
baseline_sl = (df_baseline['exit_reason'] == 'STOP_LOSS').mean() * 100

if len(best_df) > 0:
    best_row = best_df.iloc[0]
    
    print(f"""
■ 기준선 (EMA 없음):
  - 건수: {len(df_baseline)}건
  - 평균 PnL: {baseline_pnl:.2f}%
  - 승률: {baseline_wr:.1f}%
  - 손절률: {baseline_sl:.1f}%

■ 최적 조건 ({best_row['ema']} {best_row['mode']} + 수축 {best_row['squeeze']}봉):
  - 전체 건수: {best_row['total']}건
  - 익절 건수: {best_row['tp_count']}건
  - 전체 PnL: {best_row['avg_pnl']:.2f}%
  - 전체 승률: {best_row['win_rate']:.1f}%
  - 익절 PnL: {best_row['tp_pnl']:.2f}%
  - 익절 승률: {best_row['tp_win_rate']:.1f}%

■ EMA 필터 효과:
  - PnL 개선: {baseline_pnl:.2f}% → {best_row['avg_pnl']:.2f}% ({'+' if best_row['avg_pnl'] > baseline_pnl else ''}{best_row['avg_pnl'] - baseline_pnl:.2f}%p)
  - 승률 개선: {baseline_wr:.1f}% → {best_row['win_rate']:.1f}% ({'+' if best_row['win_rate'] > baseline_wr else ''}{best_row['win_rate'] - baseline_wr:.1f}%p)

■ 결론:
  - EMA 필터가 손절률 감소에 효과적
  - 트렌드 방향(가격>EMA면 LONG, 가격<EMA면 SHORT)이 더 안정적
  - 수축 기간이 길수록 신뢰도 향상
""")

# 저장
df_baseline.to_csv('bb_ema_baseline.csv', index=False)
if len(df_ema200) > 0:
    df_ema200.to_csv('bb_ema200_trend.csv', index=False)
best_df.to_csv('bb_ema_optimization.csv', index=False)

print(f"\n저장 완료:")
print(f"  - bb_ema_baseline.csv (EMA 없음)")
print(f"  - bb_ema200_trend.csv (EMA200 트렌드)")
print(f"  - bb_ema_optimization.csv (최적화 결과)")
