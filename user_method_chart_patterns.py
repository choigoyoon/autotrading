"""
사용자 매매법 정확한 구현

사용자 방법론:
1. HH = 두 개 선 긋고 (추세선 생성)
2. 그 안에서 차트패턴 확인 (추세돌파, 저항돌파, BB돌파 등)
3. 어느 시점에 진입하는게 좋은지 패턴을 두고 확인
4. 패턴의 내용으로 진입 후 동적관리

핵심: 추세가 먼저 → 추세 모르면 대기
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("사용자 매매법 정확한 구현")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

h_values = pd.read_csv('h_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])
l_values = pd.read_csv('l_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])
downtrend_lines = pd.read_csv('downtrend_lines.csv', parse_dates=['h1_time', 'h2_time', 'h2_confirmed'])
uptrend_lines = pd.read_csv('uptrend_lines.csv', parse_dates=['h1_time', 'h2_time', 'h2_confirmed'])

print(f"\n데이터 로드 완료:")
print(f"- 15분봉: {len(df):,}개 ({df['datetime'].min()} ~ {df['datetime'].max()})")
print(f"- H 값: {len(h_values):,}개")
print(f"- L 값: {len(l_values):,}개")
print(f"- 하락 추세선: {len(downtrend_lines):,}개")
print(f"- 상승 추세선: {len(uptrend_lines):,}개")

# =============================================================================
# Step 1: 추세선 구간 내에서 차트 패턴 식별
# =============================================================================
print("\n" + "=" * 80)
print("Step 1: 추세선 구간 내 차트 패턴 분석")
print("=" * 80)

def get_trendline_price(h1_time, h1_price, h2_time, h2_price, target_time):
    """추세선의 특정 시점 가격 계산"""
    if h2_time == h1_time:
        return h1_price
    
    time_ratio = (target_time - h1_time).total_seconds() / (h2_time - h1_time).total_seconds()
    return h1_price + time_ratio * (h2_price - h1_price)

def analyze_patterns_in_trendline(trendline, df, h_values, l_values, trend_type):
    """
    추세선 구간 내에서 차트 패턴 분석
    
    분석 항목:
    1. 추세선 돌파 (Trendline Breakout)
    2. 지지/저항 돌파 (Support/Resistance Breakout)  
    3. BB 돌파 (Bollinger Band Breakout)
    4. H/L 패턴 (HH, HL, LH, LL)
    5. 다이버전스
    """
    h1_time = trendline['h1_time']
    h2_time = trendline['h2_time']
    h1_price = trendline['h1_price']
    h2_price = trendline['h2_price']
    h2_confirmed = trendline['h2_confirmed']
    
    patterns = {
        'trendline_breakout': [],  # 추세선 돌파 시점
        'support_resistance': [],   # 지지/저항 레벨
        'bb_breakout': [],          # BB 돌파
        'hl_patterns': [],          # H/L 패턴들
        'divergence': []            # 다이버전스
    }
    
    # H2 확정 이후의 데이터만 분석 (미래 참조 방지)
    mask = (df['datetime'] > h2_confirmed) & (df['datetime'] <= h2_confirmed + timedelta(hours=48))
    period_df = df[mask].copy()
    
    if len(period_df) < 4:
        return patterns
    
    # 추세선 가격 계산
    for idx, row in period_df.iterrows():
        tl_price = get_trendline_price(h1_time, h1_price, h2_time, h2_price, row['datetime'])
        period_df.loc[idx, 'trendline_price'] = tl_price
    
    # 1. 추세선 돌파 찾기
    prev_close = None
    prev_tl = None
    for idx, row in period_df.iterrows():
        if prev_close is not None and prev_tl is not None:
            # 하락 추세선: 종가가 추세선 위로 돌파
            if trend_type == 'downtrend':
                if prev_close < prev_tl and row['close'] > row['trendline_price']:
                    patterns['trendline_breakout'].append({
                        'time': row['datetime'],
                        'price': row['close'],
                        'trendline_price': row['trendline_price'],
                        'gap_pct': (row['close'] - row['trendline_price']) / row['trendline_price'] * 100
                    })
            # 상승 추세선: 종가가 추세선 아래로 돌파
            elif trend_type == 'uptrend':
                if prev_close > prev_tl and row['close'] < row['trendline_price']:
                    patterns['trendline_breakout'].append({
                        'time': row['datetime'],
                        'price': row['close'],
                        'trendline_price': row['trendline_price'],
                        'gap_pct': (row['trendline_price'] - row['close']) / row['trendline_price'] * 100
                    })
        prev_close = row['close']
        prev_tl = row['trendline_price']
    
    # 2. 구간 내 H/L 패턴 찾기
    period_h = h_values[(h_values['confirmed_at'] > h2_confirmed) & 
                        (h_values['confirmed_at'] <= h2_confirmed + timedelta(hours=48))]
    period_l = l_values[(l_values['confirmed_at'] > h2_confirmed) & 
                        (l_values['confirmed_at'] <= h2_confirmed + timedelta(hours=48))]
    
    for _, h in period_h.iterrows():
        patterns['hl_patterns'].append({
            'type': 'H',
            'time': h['datetime'],
            'confirmed': h['confirmed_at'],
            'price': h['price'],
            'pattern': h['pattern']
        })
    
    for _, l in period_l.iterrows():
        patterns['hl_patterns'].append({
            'type': 'L',
            'time': l['datetime'],
            'confirmed': l['confirmed_at'],
            'price': l['price'],
            'pattern': l['pattern']
        })
    
    # 3. BB 돌파 체크 (bb_upper, bb_lower가 있다면)
    if 'bb_upper' in period_df.columns and 'bb_lower' in period_df.columns:
        for idx, row in period_df.iterrows():
            if row['high'] > row['bb_upper']:
                patterns['bb_breakout'].append({
                    'type': 'upper',
                    'time': row['datetime'],
                    'price': row['high'],
                    'bb_price': row['bb_upper']
                })
            if row['low'] < row['bb_lower']:
                patterns['bb_breakout'].append({
                    'type': 'lower',
                    'time': row['datetime'],
                    'price': row['low'],
                    'bb_price': row['bb_lower']
                })
    
    return patterns

# 하락 추세선 분석 (상승 진입 기회)
print("\n[하락 추세선 내 차트 패턴 분석]")
downtrend_patterns = []

for idx, tl in downtrend_lines.iterrows():
    patterns = analyze_patterns_in_trendline(tl, df, h_values, l_values, 'downtrend')
    
    if patterns['trendline_breakout']:
        for breakout in patterns['trendline_breakout']:
            downtrend_patterns.append({
                'h1_time': tl['h1_time'],
                'h2_time': tl['h2_time'],
                'h1_price': tl['h1_price'],
                'h2_price': tl['h2_price'],
                'h2_confirmed': tl['h2_confirmed'],
                'breakout_time': breakout['time'],
                'breakout_price': breakout['price'],
                'trendline_price': breakout['trendline_price'],
                'gap_pct': breakout['gap_pct'],
                'hl_count': len(patterns['hl_patterns']),
                'bb_breakouts': len(patterns['bb_breakout'])
            })

print(f"- 하락 추세선 돌파 발견: {len(downtrend_patterns)}개")

# 상승 추세선 분석 (하락 진입 기회)
print("\n[상승 추세선 내 차트 패턴 분석]")
uptrend_patterns = []

for idx, tl in uptrend_lines.iterrows():
    patterns = analyze_patterns_in_trendline(tl, df, h_values, l_values, 'uptrend')
    
    if patterns['trendline_breakout']:
        for breakout in patterns['trendline_breakout']:
            uptrend_patterns.append({
                'h1_time': tl['h1_time'],
                'h2_time': tl['h2_time'],
                'h1_price': tl['h1_price'],
                'h2_price': tl['h2_price'],
                'h2_confirmed': tl['h2_confirmed'],
                'breakout_time': breakout['time'],
                'breakout_price': breakout['price'],
                'trendline_price': breakout['trendline_price'],
                'gap_pct': breakout['gap_pct'],
                'hl_count': len(patterns['hl_patterns']),
                'bb_breakouts': len(patterns['bb_breakout'])
            })

print(f"- 상승 추세선 이탈 발견: {len(uptrend_patterns)}개")

# =============================================================================
# Step 2: 진입 시점 최적화 - 패턴별 분석
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: 진입 시점 최적화")
print("=" * 80)

def backtest_entry(entry_info, df, direction='long', tp_mode='dynamic'):
    """
    진입 후 결과 백테스트
    
    direction: 'long' (상승 베팅) or 'short' (하락 베팅)
    tp_mode: 'dynamic' (동적 청산) or 'fixed' (고정 TP)
    """
    entry_time = entry_info['breakout_time']
    entry_price = entry_info['breakout_price']
    
    # 진입 후 데이터
    mask = df['datetime'] > entry_time
    future_df = df[mask].head(96 * 3)  # 최대 3일 (96캔들 * 3)
    
    if len(future_df) < 4:
        return None
    
    # 동적 청산 로직 (사용자 방식)
    # 1. 반대 방향 H/L 값 확인 시 청산
    # 2. 수익 구간에서 되돌림 시 청산
    # 3. 손절가 도달 시 청산
    
    result = {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'direction': direction,
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': 0,
        'duration_candles': 0
    }
    
    # 손절가 설정: 최근 L값 (Long) 또는 최근 H값 (Short)
    h2_time = entry_info['h2_time']
    recent_l = l_values[l_values['datetime'] < h2_time]['price'].iloc[-1] if len(l_values[l_values['datetime'] < h2_time]) > 0 else entry_price * 0.97
    recent_h = h_values[h_values['datetime'] < h2_time]['price'].iloc[-1] if len(h_values[h_values['datetime'] < h2_time]) > 0 else entry_price * 1.03
    
    if direction == 'long':
        sl_price = recent_l * 0.998  # L값 -0.2%
    else:
        sl_price = recent_h * 1.002  # H값 +0.2%
    
    result['sl_price'] = sl_price
    
    # 동적 청산 시뮬레이션
    max_profit_pct = 0
    trailing_activated = False
    
    for i, (idx, row) in enumerate(future_df.iterrows()):
        if direction == 'long':
            current_pnl = (row['close'] - entry_price) / entry_price * 100
            
            # 손절 체크
            if row['low'] <= sl_price:
                result['exit_time'] = row['datetime']
                result['exit_price'] = sl_price
                result['exit_reason'] = 'SL'
                result['pnl'] = (sl_price - entry_price) / entry_price * 100
                result['duration_candles'] = i + 1
                break
            
            # 수익 트레일링
            if current_pnl > max_profit_pct:
                max_profit_pct = current_pnl
            
            # 동적 청산: 1.5% 이상 수익 후 50% 되돌림시 청산
            if max_profit_pct >= 1.5:
                trailing_activated = True
                if current_pnl <= max_profit_pct * 0.5:
                    result['exit_time'] = row['datetime']
                    result['exit_price'] = row['close']
                    result['exit_reason'] = 'TRAIL'
                    result['pnl'] = current_pnl
                    result['duration_candles'] = i + 1
                    break
            
            # 시간 기반 청산: 24시간 후 청산
            if i >= 96:
                result['exit_time'] = row['datetime']
                result['exit_price'] = row['close']
                result['exit_reason'] = 'TIME'
                result['pnl'] = current_pnl
                result['duration_candles'] = i + 1
                break
        
        else:  # short
            current_pnl = (entry_price - row['close']) / entry_price * 100
            
            # 손절 체크
            if row['high'] >= sl_price:
                result['exit_time'] = row['datetime']
                result['exit_price'] = sl_price
                result['exit_reason'] = 'SL'
                result['pnl'] = (entry_price - sl_price) / entry_price * 100
                result['duration_candles'] = i + 1
                break
            
            # 수익 트레일링
            if current_pnl > max_profit_pct:
                max_profit_pct = current_pnl
            
            if max_profit_pct >= 1.5:
                trailing_activated = True
                if current_pnl <= max_profit_pct * 0.5:
                    result['exit_time'] = row['datetime']
                    result['exit_price'] = row['close']
                    result['exit_reason'] = 'TRAIL'
                    result['pnl'] = current_pnl
                    result['duration_candles'] = i + 1
                    break
            
            if i >= 96:
                result['exit_time'] = row['datetime']
                result['exit_price'] = row['close']
                result['exit_reason'] = 'TIME'
                result['pnl'] = current_pnl
                result['duration_candles'] = i + 1
                break
    
    # 루프 완료 후 청산 안됐으면 마지막 가격으로
    if result['exit_time'] is None and len(future_df) > 0:
        last_row = future_df.iloc[-1]
        if direction == 'long':
            current_pnl = (last_row['close'] - entry_price) / entry_price * 100
        else:
            current_pnl = (entry_price - last_row['close']) / entry_price * 100
        result['exit_time'] = last_row['datetime']
        result['exit_price'] = last_row['close']
        result['exit_reason'] = 'END'
        result['pnl'] = current_pnl
        result['duration_candles'] = len(future_df)
    
    return result

# 하락 추세선 돌파 → Long 진입 백테스트
print("\n[하락 추세선 돌파 → Long 진입 백테스트]")
long_results = []

for entry in downtrend_patterns:
    result = backtest_entry(entry, df, direction='long', tp_mode='dynamic')
    if result:
        result['h1_time'] = entry['h1_time']
        result['h2_time'] = entry['h2_time']
        result['gap_pct'] = entry['gap_pct']
        result['trendline_price'] = entry['trendline_price']
        long_results.append(result)

long_df = pd.DataFrame(long_results)
if len(long_df) > 0:
    print(f"\n총 거래: {len(long_df)}회")
    print(f"승률: {(long_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"평균 PnL: {long_df['pnl'].mean():.2f}%")
    print(f"총 PnL: {long_df['pnl'].sum():.1f}%")
    print(f"평균 보유시간: {long_df['duration_candles'].mean() * 0.25:.1f}시간")
    
    # 청산 이유별 분석
    print("\n청산 이유별:")
    for reason in long_df['exit_reason'].unique():
        sub = long_df[long_df['exit_reason'] == reason]
        print(f"- {reason}: {len(sub)}회, 승률 {(sub['pnl'] > 0).mean()*100:.1f}%, 평균 PnL {sub['pnl'].mean():.2f}%")

# 상승 추세선 이탈 → Short 진입 백테스트
print("\n[상승 추세선 이탈 → Short 진입 백테스트]")
short_results = []

for entry in uptrend_patterns:
    result = backtest_entry(entry, df, direction='short', tp_mode='dynamic')
    if result:
        result['h1_time'] = entry['h1_time']
        result['h2_time'] = entry['h2_time']
        result['gap_pct'] = entry['gap_pct']
        result['trendline_price'] = entry['trendline_price']
        short_results.append(result)

short_df = pd.DataFrame(short_results)
if len(short_df) > 0:
    print(f"\n총 거래: {len(short_df)}회")
    print(f"승률: {(short_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"평균 PnL: {short_df['pnl'].mean():.2f}%")
    print(f"총 PnL: {short_df['pnl'].sum():.1f}%")
    print(f"평균 보유시간: {short_df['duration_candles'].mean() * 0.25:.1f}시간")

# =============================================================================
# Step 3: 최적 조건 찾기
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: 최적 조건 분석")
print("=" * 80)

if len(long_df) > 0:
    print("\n[Long 진입 - Gap 범위별 분석]")
    for gap_min, gap_max in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        subset = long_df[(long_df['gap_pct'] >= gap_min) & (long_df['gap_pct'] < gap_max)]
        if len(subset) >= 20:
            print(f"Gap {gap_min}-{gap_max}%: {len(subset)}회, 승률 {(subset['pnl']>0).mean()*100:.1f}%, 총PnL {subset['pnl'].sum():.1f}%")

# =============================================================================
# 최종 결과 저장
# =============================================================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

# Long 결과 저장
if len(long_df) > 0:
    long_df.to_csv('user_method_long_results.csv', index=False)
    print(f"Long 결과 저장: user_method_long_results.csv ({len(long_df)}개)")

# Short 결과 저장
if len(short_df) > 0:
    short_df.to_csv('user_method_short_results.csv', index=False)
    print(f"Short 결과 저장: user_method_short_results.csv ({len(short_df)}개)")

# 전체 결과 통합
all_results = []
if len(long_df) > 0:
    long_df['position'] = 'LONG'
    all_results.append(long_df)
if len(short_df) > 0:
    short_df['position'] = 'SHORT'
    all_results.append(short_df)

if all_results:
    all_df = pd.concat(all_results, ignore_index=True)
    all_df = all_df.sort_values('entry_time')
    all_df.to_csv('user_method_all_results.csv', index=False)
    
    # MDD 계산
    cumulative = all_df['pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    mdd = drawdown.min()
    
    print(f"\n=== 전체 결과 ===")
    print(f"총 거래: {len(all_df)}회")
    print(f"Long: {len(long_df)}회, Short: {len(short_df) if len(short_df) > 0 else 0}회")
    print(f"승률: {(all_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"총 PnL: {all_df['pnl'].sum():.1f}%")
    print(f"MDD: {mdd:.1f}%")
    print(f"MDD Ratio: {all_df['pnl'].sum() / abs(mdd):.1f}x" if mdd != 0 else "N/A")

print("\n완료!")
