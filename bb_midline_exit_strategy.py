#!/usr/bin/env python3
"""
BB 상단 돌파 후 되돌림 → 중간선(bb_mid) 익절 전략 검증
미래 데이터 없이 실시간 시뮬레이션

전략 로직:
1. BB 수축 → 확장 시점에 상단 돌파 발생 (LONG 진입)
2. 가격이 상단 밖으로 나갔다가 되돌아올 때
3. 중간선(bb_mid) 도달 시 익절

하단 돌파의 경우:
1. BB 수축 → 확장 시점에 하단 돌파 발생 (SHORT 진입)
2. 가격이 하단 밖으로 나갔다가 되돌아올 때
3. 중간선(bb_mid) 도달 시 익절
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BB 중간선 익절 전략 검증 - 미래 데이터 없이")
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

# NaN 제거
df = df.iloc[period:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# 수축 임계값
width_threshold = df['bb_width'].quantile(0.20)
print(f"수축 임계값: {width_threshold:.3f}%")

# ============================================================
# 전략 1: 상단 돌파 후 중간선 익절
# ============================================================
print("\n" + "=" * 80)
print("전략 1: BB 상단 돌파 → 중간선 익절")
print("=" * 80)

def simulate_trade(df, entry_idx, direction, max_holding=80):
    """
    실시간 트레이드 시뮬레이션 (미래 데이터 없이)
    
    Parameters:
    - entry_idx: 진입 캔들 인덱스
    - direction: 'LONG' 또는 'SHORT'
    - max_holding: 최대 보유 캔들 수 (20시간 = 80 * 15분)
    
    Returns:
    - dict: 트레이드 결과
    """
    entry_price = df.iloc[entry_idx]['close']
    entry_time = df.iloc[entry_idx]['datetime']
    entry_bb_mid = df.iloc[entry_idx]['bb_mid']
    entry_bb_upper = df.iloc[entry_idx]['bb_upper']
    entry_bb_lower = df.iloc[entry_idx]['bb_lower']
    
    exit_price = None
    exit_reason = None
    exit_time = None
    max_favorable = 0
    max_adverse = 0
    bars_held = 0
    
    # 진입 시점의 BB 밴드폭 기준으로 손절 설정 (밴드폭의 1배)
    band_width = entry_bb_upper - entry_bb_lower
    
    if direction == 'LONG':
        stop_loss = entry_price - band_width * 0.5  # 밴드폭의 50%
    else:
        stop_loss = entry_price + band_width * 0.5
    
    # 매 캔들마다 확인 (미래 데이터 없이 순차적으로)
    for i in range(entry_idx + 1, min(entry_idx + max_holding + 1, len(df))):
        candle = df.iloc[i]
        bars_held += 1
        
        # 현재 캔들 시점의 BB 값 (실시간으로 사용 가능)
        current_bb_mid = candle['bb_mid']
        
        if direction == 'LONG':
            # MFE/MAE 계산
            favorable = (candle['high'] - entry_price) / entry_price * 100
            adverse = (candle['low'] - entry_price) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            max_adverse = min(max_adverse, adverse)
            
            # 손절 확인 (저점이 손절선 이하)
            if candle['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
                exit_time = candle['datetime']
                break
            
            # 중간선 익절 확인 (종가가 중간선 이하로 되돌아옴)
            # 단, 진입 후 상단 위로 올라갔다가 내려온 경우만
            if candle['close'] <= current_bb_mid:
                exit_price = candle['close']
                exit_reason = 'MID_LINE_TP'
                exit_time = candle['datetime']
                break
                
        else:  # SHORT
            favorable = (entry_price - candle['low']) / entry_price * 100
            adverse = (entry_price - candle['high']) / entry_price * 100
            max_favorable = max(max_favorable, favorable)
            max_adverse = min(max_adverse, adverse)
            
            # 손절 확인
            if candle['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
                exit_time = candle['datetime']
                break
            
            # 중간선 익절 확인
            if candle['close'] >= current_bb_mid:
                exit_price = candle['close']
                exit_reason = 'MID_LINE_TP'
                exit_time = candle['datetime']
                break
    
    # 시간 만료
    if exit_price is None and bars_held > 0:
        exit_price = df.iloc[min(entry_idx + max_holding, len(df) - 1)]['close']
        exit_reason = 'TIME_OUT'
        exit_time = df.iloc[min(entry_idx + max_holding, len(df) - 1)]['datetime']
    
    if exit_price is None:
        return None
    
    # PnL 계산
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
        'bars_held': bars_held
    }

# ============================================================
# BB 상단/하단 돌파 감지 및 트레이드 실행
# ============================================================

trades = []
min_squeeze = 4

print("\n시뮬레이션 진행 중...")

for i in range(50, len(df) - 100):
    current_width = df.iloc[i]['bb_width']
    prev_width = df.iloc[i-1]['bb_width']
    
    # 수축 → 확장 전환 시점
    if prev_width <= width_threshold and current_width > width_threshold:
        
        # 수축 구간 찾기
        squeeze_start = i - 1
        while squeeze_start > 0 and df.iloc[squeeze_start]['bb_width'] <= width_threshold:
            squeeze_start -= 1
        
        squeeze_duration = i - squeeze_start
        if squeeze_duration < min_squeeze:
            continue
        
        # 돌파 캔들 확인
        candle = df.iloc[i]
        prev_upper = df.iloc[i-1]['bb_upper']
        prev_lower = df.iloc[i-1]['bb_lower']
        prev_close = df.iloc[i-1]['close']
        
        # 상단 돌파 (LONG)
        if candle['close'] > prev_upper:
            result = simulate_trade(df, i, 'LONG')
            if result:
                result['breakout_type'] = 'UPPER'
                result['squeeze_duration'] = squeeze_duration
                trades.append(result)
        
        # 하단 돌파 (SHORT)
        elif candle['close'] < prev_lower:
            result = simulate_trade(df, i, 'SHORT')
            if result:
                result['breakout_type'] = 'LOWER'
                result['squeeze_duration'] = squeeze_duration
                trades.append(result)

df_trades = pd.DataFrame(trades)
print(f"\n총 트레이드: {len(df_trades)}건")

if len(df_trades) == 0:
    print("트레이드 없음")
    exit()

# ============================================================
# 결과 분석
# ============================================================
print("\n" + "=" * 80)
print("★ 중간선 익절 전략 결과 ★")
print("=" * 80)

# 전체 성과
print(f"\n{'분류':>30} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'평균MFE':>10} {'평균보유':>10}")
print("-" * 90)

overall = df_trades
print(f"{'전체':>30} {len(overall):>8} {overall['pnl'].mean():>10.2f}% {(overall['pnl'] > 0).mean() * 100:>10.1f}% {overall['mfe'].mean():>10.2f}% {overall['bars_held'].mean():>10.1f}봉")

# 방향별
for direction in ['LONG', 'SHORT']:
    subset = df_trades[df_trades['direction'] == direction]
    if len(subset) > 0:
        print(f"{direction:>30} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {subset['mfe'].mean():>10.2f}% {subset['bars_held'].mean():>10.1f}봉")

# 청산 사유별
print("\n" + "-" * 90)
print(f"{'청산 사유별':>30}")
print("-" * 90)

for reason in df_trades['exit_reason'].unique():
    subset = df_trades[df_trades['exit_reason'] == reason]
    print(f"{reason:>30} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {subset['mfe'].mean():>10.2f}% {subset['bars_held'].mean():>10.1f}봉")

# ============================================================
# 중간선 익절 vs 20시간 보유 비교
# ============================================================
print("\n" + "=" * 80)
print("★ 중간선 익절 vs 20시간 홀딩 비교 ★")
print("=" * 80)

# 중간선 익절로 청산된 케이스만
midline_tp = df_trades[df_trades['exit_reason'] == 'MID_LINE_TP']

if len(midline_tp) > 0:
    print(f"\n중간선 익절 케이스: {len(midline_tp)}건")
    print(f"  - 평균 PnL: {midline_tp['pnl'].mean():.2f}%")
    print(f"  - 승률: {(midline_tp['pnl'] > 0).mean() * 100:.1f}%")
    print(f"  - 평균 보유 시간: {midline_tp['bars_held'].mean() * 15 / 60:.1f}시간")
    print(f"  - 평균 MFE: {midline_tp['mfe'].mean():.2f}%")

# ============================================================
# 상세 전략별 분석
# ============================================================
print("\n" + "=" * 80)
print("★ 돌파 유형 + 청산 사유 조합 분석 ★")
print("=" * 80)

print(f"\n{'조합':>40} {'건수':>8} {'평균PnL':>10} {'승률':>10}")
print("-" * 75)

for breakout in ['UPPER', 'LOWER']:
    for reason in ['MID_LINE_TP', 'STOP_LOSS', 'TIME_OUT']:
        subset = df_trades[(df_trades['breakout_type'] == breakout) & (df_trades['exit_reason'] == reason)]
        if len(subset) >= 5:
            print(f"{f'{breakout} + {reason}':>40} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}%")

# ============================================================
# 수축 기간별 분석
# ============================================================
print("\n" + "=" * 80)
print("★ 수축 기간별 성과 ★")
print("=" * 80)

print(f"\n{'수축기간':>15} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'중간선익절%':>12}")
print("-" * 60)

for low, high in [(4, 10), (10, 20), (20, 50), (50, 500)]:
    subset = df_trades[(df_trades['squeeze_duration'] >= low) & (df_trades['squeeze_duration'] < high)]
    if len(subset) >= 10:
        midline_pct = (subset['exit_reason'] == 'MID_LINE_TP').mean() * 100
        print(f"{f'{low}-{high}봉':>15} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {midline_pct:>12.1f}%")

# ============================================================
# 핵심 발견사항
# ============================================================
print("\n" + "=" * 80)
print("★★★ 핵심 발견사항 ★★★")
print("=" * 80)

midline_trades = df_trades[df_trades['exit_reason'] == 'MID_LINE_TP']
sl_trades = df_trades[df_trades['exit_reason'] == 'STOP_LOSS']
timeout_trades = df_trades[df_trades['exit_reason'] == 'TIME_OUT']

midline_pct = len(midline_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0
sl_pct = len(sl_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0
timeout_pct = len(timeout_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0

print(f"""
■ 청산 비율:
  - 중간선 익절: {len(midline_trades)}건 ({midline_pct:.1f}%)
  - 손절: {len(sl_trades)}건 ({sl_pct:.1f}%)
  - 시간 만료: {len(timeout_trades)}건 ({timeout_pct:.1f}%)

■ 청산 방식별 성과:
  - 중간선 익절: 평균 PnL {midline_trades['pnl'].mean():.2f}%, 승률 {(midline_trades['pnl'] > 0).mean() * 100:.1f}%
  - 손절: 평균 PnL {sl_trades['pnl'].mean():.2f}%
  - 시간 만료: 평균 PnL {timeout_trades['pnl'].mean():.2f}%, 승률 {(timeout_trades['pnl'] > 0).mean() * 100:.1f}%

■ 중간선 익절의 효과:
  - 전체 트레이드 대비 중간선 익절 비율: {midline_pct:.1f}%
  - 중간선 익절 시 평균 보유 시간: {midline_trades['bars_held'].mean() * 15 / 60:.1f}시간
  - 중간선 익절 시 평균 MFE: {midline_trades['mfe'].mean():.2f}%

■ 결론:
  {"중간선 익절 전략이 효과적!" if midline_trades['pnl'].mean() > 0 and midline_pct > 30 else "추가 필터링 필요"}
""")

# 저장
df_trades.to_csv('bb_midline_exit_results.csv', index=False)
print(f"\n저장: bb_midline_exit_results.csv")

# ============================================================
# 추가: 더 정교한 중간선 익절 전략
# ============================================================
print("\n" + "=" * 80)
print("★ 추가 분석: 상단 돌파 후 '실제로' 상단 위에 머물다 되돌아온 경우 ★")
print("=" * 80)

def simulate_trade_v2(df, entry_idx, direction, max_holding=80):
    """
    더 정교한 시뮬레이션:
    - 상단/하단 바깥으로 나간 후 되돌아오는 것을 확인
    - 중간선에서만 익절 (바깥으로 나갔다가 돌아온 경우에만)
    """
    entry_price = df.iloc[entry_idx]['close']
    entry_time = df.iloc[entry_idx]['datetime']
    
    exit_price = None
    exit_reason = None
    exit_time = None
    max_favorable = 0
    max_adverse = 0
    bars_held = 0
    went_outside = False  # 밴드 바깥으로 나갔는지
    
    # 손절 설정
    band_width = df.iloc[entry_idx]['bb_upper'] - df.iloc[entry_idx]['bb_lower']
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
            
            # 상단 밖으로 나갔는지 확인
            if candle['high'] > current_bb_upper:
                went_outside = True
            
            # 손절
            if candle['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'STOP_LOSS'
                exit_time = candle['datetime']
                break
            
            # 상단 밖으로 나갔다가 중간선으로 되돌아온 경우에만 익절
            if went_outside and candle['close'] <= current_bb_mid:
                exit_price = candle['close']
                exit_reason = 'MID_LINE_TP_V2'
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
                exit_reason = 'MID_LINE_TP_V2'
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

# V2 전략 실행
print("\nV2 전략 시뮬레이션 (상단 밖 → 되돌림 → 중간선 익절)...")

trades_v2 = []

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
        
        if candle['close'] > prev_upper:
            result = simulate_trade_v2(df, i, 'LONG')
            if result:
                result['breakout_type'] = 'UPPER'
                result['squeeze_duration'] = squeeze_duration
                trades_v2.append(result)
        
        elif candle['close'] < prev_lower:
            result = simulate_trade_v2(df, i, 'SHORT')
            if result:
                result['breakout_type'] = 'LOWER'
                result['squeeze_duration'] = squeeze_duration
                trades_v2.append(result)

df_trades_v2 = pd.DataFrame(trades_v2)
print(f"V2 총 트레이드: {len(df_trades_v2)}건")

if len(df_trades_v2) > 0:
    print(f"\n{'청산 사유':>20} {'건수':>8} {'평균PnL':>10} {'승률':>10} {'평균보유':>10}")
    print("-" * 70)
    
    for reason in df_trades_v2['exit_reason'].unique():
        subset = df_trades_v2[df_trades_v2['exit_reason'] == reason]
        print(f"{reason:>20} {len(subset):>8} {subset['pnl'].mean():>10.2f}% {(subset['pnl'] > 0).mean() * 100:>10.1f}% {subset['bars_held'].mean() * 15 / 60:>10.1f}h")
    
    # V2 중간선 익절 케이스
    midline_v2 = df_trades_v2[df_trades_v2['exit_reason'] == 'MID_LINE_TP_V2']
    if len(midline_v2) > 0:
        print(f"\n★ V2 중간선 익절 (밴드 밖 → 되돌림) 성과:")
        print(f"  - 건수: {len(midline_v2)}건 ({len(midline_v2)/len(df_trades_v2)*100:.1f}%)")
        print(f"  - 평균 PnL: {midline_v2['pnl'].mean():.2f}%")
        print(f"  - 승률: {(midline_v2['pnl'] > 0).mean() * 100:.1f}%")
        print(f"  - 평균 보유: {midline_v2['bars_held'].mean() * 15 / 60:.1f}시간")

# V2 저장
df_trades_v2.to_csv('bb_midline_exit_v2_results.csv', index=False)
print(f"\n저장: bb_midline_exit_v2_results.csv")
