"""
사용자 매매법 정확한 구현 (수정 버전)

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
print("사용자 매매법 정확한 구현 (수정 버전)")
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
# Helper Functions
# =============================================================================

def get_trendline_price(h1_time, h1_price, h2_time, h2_price, target_time):
    """추세선의 특정 시점 가격 계산"""
    if h2_time == h1_time:
        return h1_price
    
    time_ratio = (target_time - h1_time).total_seconds() / (h2_time - h1_time).total_seconds()
    return h1_price + time_ratio * (h2_price - h1_price)

# =============================================================================
# Step 1: 하락 추세선 돌파 시그널 찾기 (Long 진입용)
# =============================================================================
print("\n" + "=" * 80)
print("Step 1: 하락 추세선 돌파 시그널 찾기 (Long)")
print("=" * 80)

long_signals = []

for idx, tl in downtrend_lines.iterrows():
    h1_time = tl['h1_time']
    h2_time = tl['h2_time']
    h1_price = tl['h1_price']
    h2_price = tl['h2_price']
    h2_confirmed = tl['h2_confirmed']
    
    # H2 확정 이후의 데이터만 분석 (미래 참조 방지)
    mask = (df['datetime'] > h2_confirmed) & (df['datetime'] <= h2_confirmed + timedelta(hours=48))
    period_df = df[mask].copy()
    
    if len(period_df) < 2:
        continue
    
    # 추세선 가격 계산
    period_df['trendline_price'] = period_df['datetime'].apply(
        lambda x: get_trendline_price(h1_time, h1_price, h2_time, h2_price, x)
    )
    
    # 추세선 돌파 찾기 (종가 기준)
    prev_close = None
    prev_tl = None
    for i, (idx2, row) in enumerate(period_df.iterrows()):
        if prev_close is not None and prev_tl is not None:
            # 이전 종가가 추세선 아래, 현재 종가가 추세선 위
            if prev_close < prev_tl and row['close'] > row['trendline_price']:
                # 첫 번째 돌파만 기록 (중복 방지)
                long_signals.append({
                    'h1_time': h1_time,
                    'h2_time': h2_time,
                    'h1_price': h1_price,
                    'h2_price': h2_price,
                    'h2_confirmed': h2_confirmed,
                    'signal_time': row['datetime'],
                    'signal_price': row['close'],
                    'trendline_price': row['trendline_price'],
                    'gap_pct': (row['close'] - row['trendline_price']) / row['trendline_price'] * 100
                })
                break  # 첫 번째 돌파만
        prev_close = row['close']
        prev_tl = row['trendline_price']

print(f"Long 시그널 발견: {len(long_signals)}개")

# =============================================================================
# Step 2: Long 백테스트 (동적 관리)
# =============================================================================
print("\n" + "=" * 80)
print("Step 2: Long 백테스트 (동적 관리)")
print("=" * 80)

def backtest_long(signal, df, l_values):
    """
    Long 진입 백테스트 (동적 관리)
    
    청산 조건:
    1. 손절: 최근 L값 아래 (-0.2%)
    2. 동적 익절: 1.5% 이상 수익 후 50% 되돌림
    3. 시간 청산: 24시간 후
    """
    entry_time = signal['signal_time']
    entry_price = signal['signal_price']
    h2_time = signal['h2_time']
    
    # 진입 후 데이터
    mask = df['datetime'] > entry_time
    future_df = df[mask].head(96)  # 최대 24시간 (96캔들)
    
    if len(future_df) < 2:
        return None
    
    # 손절가: 최근 L값의 -0.3%
    recent_l = l_values[l_values['datetime'] < entry_time]
    if len(recent_l) > 0:
        sl_price = recent_l.iloc[-1]['price'] * 0.997
    else:
        sl_price = entry_price * 0.97
    
    result = {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'h1_time': signal['h1_time'],
        'h2_time': signal['h2_time'],
        'gap_pct': signal['gap_pct'],
        'trendline_price': signal['trendline_price'],
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': 0,
        'duration_candles': 0,
        'max_profit_pct': 0
    }
    
    max_profit_pct = 0
    
    for i, (idx, row) in enumerate(future_df.iterrows()):
        current_pnl = (row['close'] - entry_price) / entry_price * 100
        high_pnl = (row['high'] - entry_price) / entry_price * 100
        low_pnl = (row['low'] - entry_price) / entry_price * 100
        
        # 최고 수익 갱신
        if high_pnl > max_profit_pct:
            max_profit_pct = high_pnl
        
        # 1. 손절 체크
        if row['low'] <= sl_price:
            result['exit_time'] = row['datetime']
            result['exit_price'] = sl_price
            result['exit_reason'] = 'SL'
            result['pnl'] = (sl_price - entry_price) / entry_price * 100
            result['duration_candles'] = i + 1
            result['max_profit_pct'] = max_profit_pct
            return result
        
        # 2. 동적 익절: 1.5% 이상 수익 후 50% 되돌림
        if max_profit_pct >= 1.5:
            # 50% 되돌림 체크
            trailing_exit = entry_price * (1 + max_profit_pct / 100 * 0.5)
            if row['low'] <= trailing_exit:
                result['exit_time'] = row['datetime']
                result['exit_price'] = trailing_exit
                result['exit_reason'] = 'TRAIL'
                result['pnl'] = max_profit_pct * 0.5
                result['duration_candles'] = i + 1
                result['max_profit_pct'] = max_profit_pct
                return result
    
    # 3. 시간 청산 (24시간)
    last_row = future_df.iloc[-1]
    result['exit_time'] = last_row['datetime']
    result['exit_price'] = last_row['close']
    result['exit_reason'] = 'TIME'
    result['pnl'] = (last_row['close'] - entry_price) / entry_price * 100
    result['duration_candles'] = len(future_df)
    result['max_profit_pct'] = max_profit_pct
    
    return result

# Long 백테스트 실행
long_results = []
for signal in long_signals:
    result = backtest_long(signal, df, l_values)
    if result:
        long_results.append(result)

long_df = pd.DataFrame(long_results)

if len(long_df) > 0:
    print(f"\n=== Long 결과 ===")
    print(f"총 거래: {len(long_df)}회")
    print(f"승률: {(long_df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"평균 PnL: {long_df['pnl'].mean():.2f}%")
    print(f"총 PnL: {long_df['pnl'].sum():.1f}%")
    print(f"평균 보유시간: {long_df['duration_candles'].mean() * 0.25:.1f}시간")
    
    print("\n청산 이유별:")
    for reason in long_df['exit_reason'].unique():
        sub = long_df[long_df['exit_reason'] == reason]
        print(f"  {reason}: {len(sub)}회, 승률 {(sub['pnl']>0).mean()*100:.1f}%, 평균 PnL {sub['pnl'].mean():.2f}%")
    
    # MDD 계산
    cumulative = long_df.sort_values('entry_time')['pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    mdd = drawdown.min()
    
    print(f"\nMDD: {mdd:.1f}%")
    print(f"MDD Ratio: {long_df['pnl'].sum() / abs(mdd):.1f}x" if mdd < 0 else "MDD Ratio: N/A (no drawdown)")

# =============================================================================
# Step 3: 조건별 분석
# =============================================================================
print("\n" + "=" * 80)
print("Step 3: 조건별 분석")
print("=" * 80)

if len(long_df) > 0:
    print("\n[Gap 범위별 분석]")
    for gap_min, gap_max in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        subset = long_df[(long_df['gap_pct'] >= gap_min) & (long_df['gap_pct'] < gap_max)]
        if len(subset) >= 10:
            cum = subset.sort_values('entry_time')['pnl'].cumsum()
            mdd = (cum - cum.cummax()).min()
            print(f"Gap {gap_min}-{gap_max}%: {len(subset)}회, 승률 {(subset['pnl']>0).mean()*100:.1f}%, "
                  f"총PnL {subset['pnl'].sum():.1f}%, MDD {mdd:.1f}%")

# =============================================================================
# Step 4: 최적 조합 찾기
# =============================================================================
print("\n" + "=" * 80)
print("Step 4: 최적 조합 탐색")
print("=" * 80)

if len(long_df) > 0:
    # H2 확정 후 대기 시간 계산
    long_df['wait_hours'] = (pd.to_datetime(long_df['entry_time']) - 
                            pd.to_datetime(long_df['h2_time'])).dt.total_seconds() / 3600
    
    print("\n[H2 후 대기시간별 분석]")
    for wait_min, wait_max in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 48)]:
        subset = long_df[(long_df['wait_hours'] >= wait_min) & (long_df['wait_hours'] < wait_max)]
        if len(subset) >= 10:
            cum = subset.sort_values('entry_time')['pnl'].cumsum()
            mdd = (cum - cum.cummax()).min()
            mdd_ratio = subset['pnl'].sum() / abs(mdd) if mdd < 0 else 0
            print(f"H2+{wait_min}-{wait_max}h: {len(subset)}회, 승률 {(subset['pnl']>0).mean()*100:.1f}%, "
                  f"총PnL {subset['pnl'].sum():.1f}%, MDD {mdd:.1f}%, MDDRatio {mdd_ratio:.1f}x")
    
    # 최적 조합 찾기
    print("\n[최적 조합 탐색: Gap + 대기시간]")
    best_combos = []
    
    for gap_min, gap_max in [(0, 0.2), (0.1, 0.3), (0.2, 0.5), (0.3, 0.7), (0.5, 1.0)]:
        for wait_min, wait_max in [(0, 5), (2, 10), (5, 15), (10, 25), (0, 48)]:
            subset = long_df[(long_df['gap_pct'] >= gap_min) & (long_df['gap_pct'] < gap_max) &
                            (long_df['wait_hours'] >= wait_min) & (long_df['wait_hours'] < wait_max)]
            
            if len(subset) >= 30:
                cum = subset.sort_values('entry_time')['pnl'].cumsum()
                mdd = (cum - cum.cummax()).min()
                mdd_ratio = subset['pnl'].sum() / abs(mdd) if mdd < 0 else 0
                
                best_combos.append({
                    'gap_range': f"{gap_min}-{gap_max}%",
                    'wait_range': f"{wait_min}-{wait_max}h",
                    'trades': len(subset),
                    'win_rate': (subset['pnl'] > 0).mean() * 100,
                    'total_pnl': subset['pnl'].sum(),
                    'mdd': mdd,
                    'mdd_ratio': mdd_ratio,
                    'avg_pnl': subset['pnl'].mean()
                })
    
    if best_combos:
        combo_df = pd.DataFrame(best_combos)
        combo_df = combo_df.sort_values('mdd_ratio', ascending=False)
        
        print("\n상위 10개 조합 (MDD Ratio 기준):")
        for i, row in combo_df.head(10).iterrows():
            print(f"  Gap {row['gap_range']}, Wait {row['wait_range']}: "
                  f"{row['trades']}회, 승률 {row['win_rate']:.1f}%, "
                  f"PnL {row['total_pnl']:.1f}%, MDD {row['mdd']:.1f}%, "
                  f"MDDRatio {row['mdd_ratio']:.1f}x")
        
        # 최적 조합 저장
        combo_df.to_csv('user_method_optimal_combos.csv', index=False)
        print(f"\n조합 분석 결과 저장: user_method_optimal_combos.csv")

# =============================================================================
# 결과 저장
# =============================================================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

if len(long_df) > 0:
    long_df.to_csv('user_method_long_corrected.csv', index=False)
    print(f"Long 결과 저장: user_method_long_corrected.csv ({len(long_df)}개)")

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

if len(long_df) > 0:
    years = (long_df['entry_time'].max() - long_df['entry_time'].min()).days / 365
    annual_pnl = long_df['pnl'].sum() / years if years > 0 else 0
    
    cumulative = long_df.sort_values('entry_time')['pnl'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    mdd = drawdown.min()
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    사용자 매매법 백테스트 결과                     │
├─────────────────────────────────────────────────────────────────┤
│  전략: 하락 추세선 (LH-LH) 돌파 후 Long 진입                       │
│  기간: {years:.1f}년                                              │
├─────────────────────────────────────────────────────────────────┤
│  총 거래: {len(long_df):,}회 (연간 {len(long_df)/years:.0f}회)                │
│  승률: {(long_df['pnl'] > 0).mean() * 100:.1f}%                                     │
│  평균 PnL: {long_df['pnl'].mean():.2f}%                                 │
│  총 PnL: {long_df['pnl'].sum():.1f}%                                    │
│  연평균 PnL: {annual_pnl:.1f}%                                   │
│  MDD: {mdd:.1f}%                                              │
│  MDD Ratio: {long_df['pnl'].sum() / abs(mdd):.1f}x                         │
│  평균 보유시간: {long_df['duration_candles'].mean() * 0.25:.1f}시간                   │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n완료!")
