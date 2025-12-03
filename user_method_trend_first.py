"""
사용자 매매법 정확한 구현: 추세가 먼저!

사용자 방법론:
1. 추세 확인이 먼저 (LH+LL = 하락추세, HH+HL = 상승추세)
2. 추세 모르면 대기
3. HH (또는 LH-LH) = 두 개 선 긋고 (추세선 생성)
4. 그 안에서 차트패턴 확인 (추세돌파, 저항돌파, BB돌파 등)
5. 패턴의 내용으로 진입 후 동적관리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 80)
print("사용자 매매법: 추세가 먼저!")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

h_values = pd.read_csv('h_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])
l_values = pd.read_csv('l_values_macd_based.csv', parse_dates=['datetime', 'confirmed_at'])

print(f"\n데이터 로드 완료:")
print(f"- 15분봉: {len(df):,}개")
print(f"- H 값: {len(h_values):,}개")
print(f"- L 값: {len(l_values):,}개")

# =============================================================================
# Step 1: 추세 확인 로직
# =============================================================================
print("\n" + "=" * 80)
print("Step 1: 추세 확인 (LH+LL=하락, HH+HL=상승)")
print("=" * 80)

def identify_trend_at_time(target_time, h_values, l_values, lookback=3):
    """
    특정 시점에서 추세 확인
    
    하락추세: 최근 H가 LH이고, 최근 L이 LL
    상승추세: 최근 H가 HH이고, 최근 L이 HL
    불확실: 위 조건을 만족하지 않음
    
    Returns: ('downtrend', 'uptrend', 'unclear')
    """
    # 해당 시점 이전에 확정된 H/L 값만 사용
    recent_h = h_values[h_values['confirmed_at'] < target_time].tail(lookback)
    recent_l = l_values[l_values['confirmed_at'] < target_time].tail(lookback)
    
    if len(recent_h) < 2 or len(recent_l) < 2:
        return 'unclear', None, None
    
    # 최근 2개의 H 패턴 확인
    h_patterns = recent_h['pattern'].tolist()
    l_patterns = recent_l['pattern'].tolist()
    
    # 마지막 H, L 값
    last_h = recent_h.iloc[-1]
    last_l = recent_l.iloc[-1]
    
    # 하락추세: LH + LL
    is_downtrend = False
    if len(h_patterns) >= 2 and 'LH' in h_patterns[-2:]:
        if len(l_patterns) >= 2 and 'LL' in l_patterns[-2:]:
            is_downtrend = True
    
    # 상승추세: HH + HL
    is_uptrend = False
    if len(h_patterns) >= 2 and 'HH' in h_patterns[-2:]:
        if len(l_patterns) >= 2 and 'HL' in l_patterns[-2:]:
            is_uptrend = True
    
    if is_downtrend and not is_uptrend:
        return 'downtrend', last_h, last_l
    elif is_uptrend and not is_downtrend:
        return 'uptrend', last_h, last_l
    else:
        return 'unclear', last_h, last_l

# 추세 전환 시점 찾기
trend_changes = []
prev_trend = None

for idx, row in h_values.iterrows():
    confirmed_at = row['confirmed_at']
    if pd.isna(confirmed_at):
        continue
    
    trend, last_h, last_l = identify_trend_at_time(confirmed_at, h_values, l_values)
    
    if trend != prev_trend and trend != 'unclear':
        trend_changes.append({
            'time': confirmed_at,
            'trend': trend,
            'h_price': last_h['price'] if last_h is not None else None,
            'l_price': last_l['price'] if last_l is not None else None
        })
    prev_trend = trend

print(f"추세 전환 횟수: {len(trend_changes)}회")

# 추세별 통계
trend_df = pd.DataFrame(trend_changes)
print(f"- 하락추세 시작: {(trend_df['trend'] == 'downtrend').sum()}회")
print(f"- 상승추세 시작: {(trend_df['trend'] == 'uptrend').sum()}회")

# =============================================================================
# Step 2: 추세 내에서 진입 시그널 찾기
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: 추세 내 진입 시그널")
print("=" * 80)

def get_trendline_price(h1_time, h1_price, h2_time, h2_price, target_time):
    """추세선의 특정 시점 가격 계산"""
    if h2_time == h1_time:
        return h1_price
    time_diff_total = (h2_time - h1_time).total_seconds()
    time_diff_target = (target_time - h1_time).total_seconds()
    return h1_price + (time_diff_target / time_diff_total) * (h2_price - h1_price)

def find_entry_signals_in_trend(trend_info, df, h_values, l_values, next_trend_time=None):
    """
    추세 내에서 진입 시그널 찾기
    
    하락추세에서:
    - LH-LH 추세선 긋기
    - 추세선 돌파 시 Long 진입 (역추세 반등)
    
    상승추세에서:
    - HL-HL 추세선 긋기  
    - 추세선 이탈 시 Short 진입 (역추세 반등)
    """
    signals = []
    trend_start = trend_info['time']
    trend_type = trend_info['trend']
    
    # 다음 추세 시작 또는 48시간 후까지
    if next_trend_time is None:
        trend_end = trend_start + timedelta(hours=48)
    else:
        trend_end = next_trend_time
    
    if trend_type == 'downtrend':
        # 하락추세: LH 패턴을 가진 H 값들로 추세선 긋기
        period_h = h_values[(h_values['confirmed_at'] >= trend_start) & 
                           (h_values['confirmed_at'] < trend_end) &
                           (h_values['pattern'] == 'LH')]
        
        if len(period_h) < 2:
            # LH가 2개 미만이면 이전 LH 포함
            prev_lh = h_values[(h_values['confirmed_at'] < trend_start) & 
                              (h_values['pattern'] == 'LH')].tail(2)
            period_h = pd.concat([prev_lh, period_h]).drop_duplicates()
        
        if len(period_h) >= 2:
            # 마지막 2개의 LH로 추세선 생성
            h1 = period_h.iloc[-2]
            h2 = period_h.iloc[-1]
            
            h1_time = h1['datetime']
            h2_time = h2['datetime']
            h1_price = h1['price']
            h2_price = h2['price']
            h2_confirmed = h2['confirmed_at']
            
            # H2 확정 이후 추세선 돌파 찾기
            mask = (df['datetime'] > h2_confirmed) & (df['datetime'] < trend_end)
            check_df = df[mask].copy()
            
            if len(check_df) > 1:
                prev_close = None
                prev_tl = None
                
                for _, row in check_df.iterrows():
                    tl_price = get_trendline_price(h1_time, h1_price, h2_time, h2_price, row['datetime'])
                    
                    if prev_close is not None and prev_tl is not None:
                        # 추세선 돌파: 이전 종가가 추세선 아래, 현재 종가가 위
                        if prev_close < prev_tl and row['close'] > tl_price:
                            signals.append({
                                'trend': 'downtrend',
                                'direction': 'long',
                                'h1_time': h1_time,
                                'h2_time': h2_time,
                                'h1_price': h1_price,
                                'h2_price': h2_price,
                                'h2_confirmed': h2_confirmed,
                                'signal_time': row['datetime'],
                                'signal_price': row['close'],
                                'trendline_price': tl_price,
                                'gap_pct': (row['close'] - tl_price) / tl_price * 100
                            })
                            break  # 첫 번째 돌파만
                    
                    prev_close = row['close']
                    prev_tl = tl_price
    
    elif trend_type == 'uptrend':
        # 상승추세: HL 패턴을 가진 L 값들로 추세선 긋기
        period_l = l_values[(l_values['confirmed_at'] >= trend_start) & 
                           (l_values['confirmed_at'] < trend_end) &
                           (l_values['pattern'] == 'HL')]
        
        if len(period_l) < 2:
            prev_hl = l_values[(l_values['confirmed_at'] < trend_start) & 
                              (l_values['pattern'] == 'HL')].tail(2)
            period_l = pd.concat([prev_hl, period_l]).drop_duplicates()
        
        if len(period_l) >= 2:
            l1 = period_l.iloc[-2]
            l2 = period_l.iloc[-1]
            
            l1_time = l1['datetime']
            l2_time = l2['datetime']
            l1_price = l1['price']
            l2_price = l2['price']
            l2_confirmed = l2['confirmed_at']
            
            mask = (df['datetime'] > l2_confirmed) & (df['datetime'] < trend_end)
            check_df = df[mask].copy()
            
            if len(check_df) > 1:
                prev_close = None
                prev_tl = None
                
                for _, row in check_df.iterrows():
                    tl_price = get_trendline_price(l1_time, l1_price, l2_time, l2_price, row['datetime'])
                    
                    if prev_close is not None and prev_tl is not None:
                        # 추세선 이탈: 이전 종가가 추세선 위, 현재 종가가 아래
                        if prev_close > prev_tl and row['close'] < tl_price:
                            signals.append({
                                'trend': 'uptrend',
                                'direction': 'short',
                                'h1_time': l1_time,
                                'h2_time': l2_time,
                                'h1_price': l1_price,
                                'h2_price': l2_price,
                                'h2_confirmed': l2_confirmed,
                                'signal_time': row['datetime'],
                                'signal_price': row['close'],
                                'trendline_price': tl_price,
                                'gap_pct': (tl_price - row['close']) / tl_price * 100
                            })
                            break
                    
                    prev_close = row['close']
                    prev_tl = tl_price
    
    return signals

# 모든 추세에서 시그널 찾기
all_signals = []

for i, trend_info in enumerate(trend_changes):
    next_trend_time = trend_changes[i+1]['time'] if i+1 < len(trend_changes) else None
    signals = find_entry_signals_in_trend(trend_info, df, h_values, l_values, next_trend_time)
    all_signals.extend(signals)

print(f"총 시그널: {len(all_signals)}개")
print(f"- Long (하락추세 돌파): {sum(1 for s in all_signals if s['direction'] == 'long')}개")
print(f"- Short (상승추세 이탈): {sum(1 for s in all_signals if s['direction'] == 'short')}개")

# =============================================================================
# Step 3: 백테스트 (동적 관리)
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: 백테스트 (동적 관리)")
print("=" * 80)

def backtest_with_dynamic_exit(signal, df, h_values, l_values):
    """
    동적 관리 백테스트
    
    청산 조건:
    1. 손절: 최근 L/H 값 기준
    2. 동적 익절: 수익 구간에서 되돌림 시
    3. 시간 청산: 24시간 후
    """
    entry_time = signal['signal_time']
    entry_price = signal['signal_price']
    direction = signal['direction']
    
    mask = df['datetime'] > entry_time
    future_df = df[mask].head(96)  # 24시간
    
    if len(future_df) < 2:
        return None
    
    # 손절가 설정
    if direction == 'long':
        recent_l = l_values[l_values['confirmed_at'] < entry_time].tail(1)
        if len(recent_l) > 0:
            sl_price = recent_l.iloc[0]['price'] * 0.997
        else:
            sl_price = entry_price * 0.97
    else:  # short
        recent_h = h_values[h_values['confirmed_at'] < entry_time].tail(1)
        if len(recent_h) > 0:
            sl_price = recent_h.iloc[0]['price'] * 1.003
        else:
            sl_price = entry_price * 1.03
    
    result = {
        'trend': signal['trend'],
        'direction': direction,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'gap_pct': signal['gap_pct'],
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': 0,
        'duration_candles': 0,
        'max_profit_pct': 0
    }
    
    max_profit_pct = 0
    
    for i, (_, row) in enumerate(future_df.iterrows()):
        if direction == 'long':
            current_pnl = (row['close'] - entry_price) / entry_price * 100
            high_pnl = (row['high'] - entry_price) / entry_price * 100
            
            if high_pnl > max_profit_pct:
                max_profit_pct = high_pnl
            
            # 손절
            if row['low'] <= sl_price:
                result['exit_time'] = row['datetime']
                result['exit_price'] = sl_price
                result['exit_reason'] = 'SL'
                result['pnl'] = (sl_price - entry_price) / entry_price * 100
                result['duration_candles'] = i + 1
                result['max_profit_pct'] = max_profit_pct
                return result
            
            # 동적 익절: 2% 이상 후 50% 되돌림
            if max_profit_pct >= 2.0:
                trail_exit = entry_price * (1 + max_profit_pct / 100 * 0.5)
                if row['low'] <= trail_exit:
                    result['exit_time'] = row['datetime']
                    result['exit_price'] = trail_exit
                    result['exit_reason'] = 'TRAIL'
                    result['pnl'] = max_profit_pct * 0.5
                    result['duration_candles'] = i + 1
                    result['max_profit_pct'] = max_profit_pct
                    return result
        
        else:  # short
            current_pnl = (entry_price - row['close']) / entry_price * 100
            low_pnl = (entry_price - row['low']) / entry_price * 100
            
            if low_pnl > max_profit_pct:
                max_profit_pct = low_pnl
            
            # 손절
            if row['high'] >= sl_price:
                result['exit_time'] = row['datetime']
                result['exit_price'] = sl_price
                result['exit_reason'] = 'SL'
                result['pnl'] = (entry_price - sl_price) / entry_price * 100
                result['duration_candles'] = i + 1
                result['max_profit_pct'] = max_profit_pct
                return result
            
            # 동적 익절
            if max_profit_pct >= 2.0:
                trail_exit = entry_price * (1 - max_profit_pct / 100 * 0.5)
                if row['high'] >= trail_exit:
                    result['exit_time'] = row['datetime']
                    result['exit_price'] = trail_exit
                    result['exit_reason'] = 'TRAIL'
                    result['pnl'] = max_profit_pct * 0.5
                    result['duration_candles'] = i + 1
                    result['max_profit_pct'] = max_profit_pct
                    return result
    
    # 시간 청산
    last_row = future_df.iloc[-1]
    if direction == 'long':
        result['pnl'] = (last_row['close'] - entry_price) / entry_price * 100
    else:
        result['pnl'] = (entry_price - last_row['close']) / entry_price * 100
    
    result['exit_time'] = last_row['datetime']
    result['exit_price'] = last_row['close']
    result['exit_reason'] = 'TIME'
    result['duration_candles'] = len(future_df)
    result['max_profit_pct'] = max_profit_pct
    
    return result

# 백테스트 실행
results = []
for signal in all_signals:
    result = backtest_with_dynamic_exit(signal, df, h_values, l_values)
    if result:
        results.append(result)

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(f"\n=== 전체 결과 ===")
    print(f"총 거래: {len(results_df)}회")
    print(f"Long: {(results_df['direction'] == 'long').sum()}회")
    print(f"Short: {(results_df['direction'] == 'short').sum()}회")
    print(f"승률: {(results_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"평균 PnL: {results_df['pnl'].mean():.2f}%")
    print(f"총 PnL: {results_df['pnl'].sum():.1f}%")
    
    # MDD 계산
    cumulative = results_df.sort_values('entry_time')['pnl'].cumsum()
    peak = cumulative.cummax()
    mdd = (cumulative - peak).min()
    
    print(f"MDD: {mdd:.1f}%")
    print(f"MDD Ratio: {results_df['pnl'].sum() / abs(mdd):.1f}x" if mdd < 0 else "N/A")
    
    # 방향별 분석
    print("\n[방향별 분석]")
    for direction in ['long', 'short']:
        sub = results_df[results_df['direction'] == direction]
        if len(sub) > 0:
            cum = sub.sort_values('entry_time')['pnl'].cumsum()
            sub_mdd = (cum - cum.cummax()).min()
            print(f"{direction.upper()}: {len(sub)}회, 승률 {(sub['pnl']>0).mean()*100:.1f}%, "
                  f"총PnL {sub['pnl'].sum():.1f}%, MDD {sub_mdd:.1f}%")
    
    # 청산 이유별 분석
    print("\n[청산 이유별 분석]")
    for reason in results_df['exit_reason'].unique():
        sub = results_df[results_df['exit_reason'] == reason]
        print(f"{reason}: {len(sub)}회, 승률 {(sub['pnl']>0).mean()*100:.1f}%, 평균 PnL {sub['pnl'].mean():.2f}%")
    
    # Gap 범위별 분석 (Long만)
    print("\n[Long Gap 범위별 분석]")
    long_df = results_df[results_df['direction'] == 'long']
    for gap_min, gap_max in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 3.0)]:
        sub = long_df[(long_df['gap_pct'] >= gap_min) & (long_df['gap_pct'] < gap_max)]
        if len(sub) >= 5:
            cum = sub.sort_values('entry_time')['pnl'].cumsum()
            sub_mdd = (cum - cum.cummax()).min()
            mdd_ratio = sub['pnl'].sum() / abs(sub_mdd) if sub_mdd < 0 else 0
            print(f"Gap {gap_min}-{gap_max}%: {len(sub)}회, 승률 {(sub['pnl']>0).mean()*100:.1f}%, "
                  f"총PnL {sub['pnl'].sum():.1f}%, MDD {sub_mdd:.1f}%, MDDRatio {mdd_ratio:.1f}x")

# =============================================================================
# 결과 저장
# =============================================================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

if len(results_df) > 0:
    results_df.to_csv('user_method_trend_first_results.csv', index=False)
    print(f"결과 저장: user_method_trend_first_results.csv ({len(results_df)}개)")
    
    # 최종 요약
    years = (results_df['entry_time'].max() - results_df['entry_time'].min()).days / 365
    annual_pnl = results_df['pnl'].sum() / years if years > 0 else 0
    
    cumulative = results_df.sort_values('entry_time')['pnl'].cumsum()
    peak = cumulative.cummax()
    mdd = (cumulative - peak).min()
    
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│            사용자 매매법 (추세가 먼저) 백테스트 결과                  │
├──────────────────────────────────────────────────────────────────┤
│  핵심 원칙: 추세 확인 → 추세선 긋기 → 패턴 확인 → 진입              │
│  기간: {years:.1f}년                                               │
├──────────────────────────────────────────────────────────────────┤
│  총 거래: {len(results_df):,}회 (연간 {len(results_df)/years:.0f}회)                 │
│  Long: {(results_df['direction']=='long').sum()}회, Short: {(results_df['direction']=='short').sum()}회        │
│  승률: {(results_df['pnl'] > 0).mean() * 100:.1f}%                                      │
│  평균 PnL: {results_df['pnl'].mean():.2f}%                                  │
│  총 PnL: {results_df['pnl'].sum():.1f}%                                     │
│  연평균 PnL: {annual_pnl:.1f}%                                    │
│  MDD: {mdd:.1f}%                                               │
│  MDD Ratio: {results_df['pnl'].sum() / abs(mdd):.1f}x                          │
└──────────────────────────────────────────────────────────────────┘
""")

print("\n완료!")
