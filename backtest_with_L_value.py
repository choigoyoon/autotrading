"""
올바른 로직 백테스트
- SL: 돌파 전 L값 (저점) 이탈 시 손절
- TP: 저항선 도달 시 분할 익절
- 시간 제한: 없음
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')
l_values = pd.read_csv('all_L_values.csv')

# 시간 변환
signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
signals['h1_time'] = pd.to_datetime(signals['h1_time'])
signals['h2_time'] = pd.to_datetime(signals['h2_time'])
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
l_values['datetime'] = pd.to_datetime(l_values['datetime'])

df_15m = df_15m.sort_values('datetime').reset_index(drop=True)
l_values = l_values.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("돌파 전 L값 기준 백테스트")
print("="*80)

# 각 시그널에 돌파 전 L값 찾기
def find_L_before_breakout(breakout_time, l_values_df):
    """돌파 시점 직전의 L값 찾기"""
    before = l_values_df[l_values_df['datetime'] < breakout_time]
    if len(before) == 0:
        return None, None
    last_L = before.iloc[-1]
    return last_L['datetime'], last_L['L_value']

print("\n돌파 전 L값 매칭 중...")

l_times = []
l_prices = []

for idx, signal in signals.iterrows():
    l_time, l_price = find_L_before_breakout(signal['breakout_time'], l_values)
    l_times.append(l_time)
    l_prices.append(l_price)

signals['L_time'] = l_times
signals['L_price'] = l_prices

# L값 없는 시그널 제외
valid_signals = signals.dropna(subset=['L_price']).copy()
print(f"L값 매칭된 시그널: {len(valid_signals)}개 / 전체 {len(signals)}개")

# Entry - L 거리 분석
valid_signals['entry_L_gap'] = (valid_signals['breakout_price'] - valid_signals['L_price']) / valid_signals['breakout_price'] * 100

print(f"\n📊 Entry - L 거리 분석")
print(f"  평균: {valid_signals['entry_L_gap'].mean():.2f}%")
print(f"  중앙값: {valid_signals['entry_L_gap'].median():.2f}%")
print(f"  최소: {valid_signals['entry_L_gap'].min():.2f}%")
print(f"  최대: {valid_signals['entry_L_gap'].max():.2f}%")

# L이 Entry보다 높은 문제 케이스
L_above = (valid_signals['L_price'] > valid_signals['breakout_price']).sum()
print(f"\n⚠️ L이 Entry보다 높은 경우: {L_above}건 ({L_above/len(valid_signals)*100:.1f}%)")

print("\n" + "="*80)
print("백테스트 실행")
print("="*80)
print("  - 손절: L값 이탈 시")
print("  - 익절: 분할 (Entry+2%, Entry+5%)")
print("  - 시간제한: 없음")

# 트렌드라인 함수
def get_trendline_price(h1_time, h1_price, h2_time, h2_price, target_time):
    time_diff_h = (h2_time - h1_time).total_seconds() / 3600
    if time_diff_h == 0:
        return h2_price
    slope_per_hour = (h2_price - h1_price) / time_diff_h
    target_diff_h = (target_time - h1_time).total_seconds() / 3600
    return h1_price + slope_per_hour * target_diff_h

# 백테스트
results = []

for idx, signal in valid_signals.iterrows():
    entry_time = signal['breakout_time']
    entry_price = signal['breakout_price']
    L_price = signal['L_price']
    h1_time = signal['h1_time']
    h1_price = signal['h1_price']
    h2_time = signal['h2_time']
    h2_price = signal['h2_price']
    
    # L이 Entry보다 높으면 스킵
    if L_price >= entry_price:
        continue
    
    # 미래 데이터 (최대 60일)
    future_data = df_15m[df_15m['datetime'] > entry_time].head(60 * 24 * 4)
    
    if len(future_data) == 0:
        continue
    
    # TP 목표: 단순하게 Entry 기준 %
    tp1_target = entry_price * 1.02  # +2%
    tp2_target = entry_price * 1.05  # +5%
    
    # SL: L값
    sl_price = L_price
    
    position = 1.0
    total_pnl = 0
    tp1_done = False
    tp2_done = False
    sl_done = False
    exit_reason = None
    exit_time = None
    exit_price = None
    hold_hours = 0
    
    for _, candle in future_data.iterrows():
        candle_time = candle['datetime']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        hold_hours = (candle_time - entry_time).total_seconds() / 3600
        
        # 1. SL 체크: L값 이탈
        if low <= sl_price and position > 0:
            sl_pnl = (sl_price - entry_price) / entry_price * 100 * position
            total_pnl += sl_pnl
            sl_done = True
            exit_reason = 'SL_L_BREAK'
            exit_time = candle_time
            exit_price = sl_price
            position = 0
            break
        
        # 2. TP1 (+2%)
        if not tp1_done and high >= tp1_target and position > 0:
            tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * 0.5
            total_pnl += tp1_pnl
            position -= 0.5
            tp1_done = True
        
        # 3. TP2 (+5%)
        if not tp2_done and high >= tp2_target and position > 0:
            tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position
            total_pnl += tp2_pnl
            exit_reason = 'TP_REACHED'
            exit_time = candle_time
            exit_price = tp2_target
            position = 0
            tp2_done = True
            break
    
    # 포지션 남아있으면
    if position > 0 and len(future_data) > 0:
        last_candle = future_data.iloc[-1]
        unrealized = (last_candle['close'] - entry_price) / entry_price * 100 * position
        total_pnl += unrealized
        exit_reason = 'STILL_HOLDING'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
    
    results.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'L_price': L_price,
        'entry_L_gap': (entry_price - L_price) / entry_price * 100,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'hold_hours': hold_hours,
        'tp1_done': tp1_done,
        'tp2_done': tp2_done,
        'sl_done': sl_done,
        'total_pnl': total_pnl
    })

result_df = pd.DataFrame(results)

# 결과
print("\n" + "="*80)
print("📊 결과")
print("="*80)

print(f"\n총 거래: {len(result_df)}건")
print(f"승률: {(result_df['total_pnl'] > 0).mean() * 100:.1f}%")
print(f"평균 수익: {result_df['total_pnl'].mean():+.2f}%")
print(f"총 수익: {result_df['total_pnl'].sum():+.1f}%")

print(f"\nTP1 달성: {result_df['tp1_done'].mean() * 100:.1f}%")
print(f"TP2 달성: {result_df['tp2_done'].mean() * 100:.1f}%")
print(f"L값 손절: {result_df['sl_done'].mean() * 100:.1f}%")

# 출구별
print("\n📊 출구별 분석")
exit_stats = result_df.groupby('exit_reason').agg({
    'total_pnl': ['count', 'mean', 'sum'],
    'hold_hours': 'mean'
}).round(2)
exit_stats.columns = ['건수', '평균수익', '총수익', '평균보유']
print(exit_stats)

# Entry-L 거리별 성과
print("\n📊 Entry-L 거리별 성과")
result_df['gap_bin'] = pd.cut(result_df['entry_L_gap'], 
                               bins=[0, 1, 2, 3, 5, 100],
                               labels=['0-1%', '1-2%', '2-3%', '3-5%', '5%+'])
gap_stats = result_df.groupby('gap_bin').agg({
    'total_pnl': ['count', 'mean'],
    'sl_done': 'mean'
}).round(2)
gap_stats.columns = ['건수', '평균수익', '손절률']
print(gap_stats)

result_df.to_csv('L_value_backtest.csv', index=False)
print(f"\n저장: L_value_backtest.csv")

