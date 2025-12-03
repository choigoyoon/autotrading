"""
올바른 로직으로 백테스트
- SL: HL 이탈 시 손절
- TP: 트렌드라인(저항선) 도달 시 분할 익절
- 시간 제한: 없음
- 눌림목: 추매 기회 (나중에 추가)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')

signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
signals['h1_time'] = pd.to_datetime(signals['h1_time'])
signals['h2_time'] = pd.to_datetime(signals['h2_time'])
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("올바른 로직 백테스트")
print("="*80)
print(f"\n시그널: {len(signals)}개")

# 트렌드라인 연장 함수 (시간이 지나면 저항선도 변함)
def get_trendline_price(h1_time, h1_price, h2_time, h2_price, target_time):
    """특정 시간의 트렌드라인(저항선) 가격 계산"""
    time_diff_h = (h2_time - h1_time).total_seconds() / 3600
    if time_diff_h == 0:
        return h2_price
    
    slope_per_hour = (h2_price - h1_price) / time_diff_h
    target_diff_h = (target_time - h1_time).total_seconds() / 3600
    
    return h1_price + slope_per_hour * target_diff_h

print("\n전략 로직:")
print("  - 손절: HL(저점) 이탈 시")
print("  - 익절: 트렌드라인(저항선) 도달 시 분할")
print("  - 시간제한: 없음 (조건 충족까지 홀딩)")

# 백테스트 실행
results = []

for idx, signal in signals.iterrows():
    entry_time = signal['breakout_time']
    entry_price = signal['breakout_price']
    h1_time = signal['h1_time']
    h1_price = signal['h1_price']
    h2_time = signal['h2_time']
    h2_price = signal['h2_price']
    hl_price = signal['hl_price']  # 이게 손절 기준!
    
    # 미래 데이터 (최대 30일까지 보자)
    future_data = df_15m[df_15m['datetime'] > entry_time].head(30 * 24 * 4)
    
    if len(future_data) == 0:
        continue
    
    # 초기 설정
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
        
        # 현재 시점의 트렌드라인(저항선) 가격 계산
        trendline_now = get_trendline_price(h1_time, h1_price, h2_time, h2_price, candle_time)
        
        # TP 목표: 트렌드라인까지의 거리
        # 진입가에서 트렌드라인까지 50% 지점 = TP1
        # 트렌드라인 도달 = TP2
        entry_to_trendline = trendline_now - entry_price
        
        # 트렌드라인이 진입가 아래면 (이미 돌파한 상태)
        # → 트렌드라인 위로 더 상승하는 것을 목표로
        if entry_to_trendline < 0:
            # 돌파 후이므로, 진입가 위로 일정% 상승을 목표
            tp1_target = entry_price * 1.02  # 2% 위
            tp2_target = entry_price * 1.05  # 5% 위
        else:
            tp1_target = entry_price + entry_to_trendline * 0.5
            tp2_target = trendline_now
        
        # 1. 손절 체크: HL 이탈
        if low <= hl_price and position > 0:
            sl_pnl = (hl_price - entry_price) / entry_price * 100 * position
            total_pnl += sl_pnl
            sl_done = True
            exit_reason = 'SL_HL_BREAK'
            exit_time = candle_time
            exit_price = hl_price
            position = 0
            break
        
        # 2. TP1 체크 (50% 지점 또는 2%)
        if not tp1_done and high >= tp1_target and position > 0:
            tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * 0.5  # 50% 매도
            total_pnl += tp1_pnl
            position -= 0.5
            tp1_done = True
        
        # 3. TP2 체크 (트렌드라인 도달 또는 5%)
        if not tp2_done and high >= tp2_target and position > 0:
            tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position
            total_pnl += tp2_pnl
            exit_reason = 'TP_TRENDLINE'
            exit_time = candle_time
            exit_price = tp2_target
            position = 0
            tp2_done = True
            break
    
    # 아직 포지션 남아있으면 (30일 후에도 조건 미충족)
    if position > 0 and len(future_data) > 0:
        last_candle = future_data.iloc[-1]
        # 마지막 가격으로 미실현 손익 계산
        unrealized_pnl = (last_candle['close'] - entry_price) / entry_price * 100 * position
        total_pnl += unrealized_pnl
        exit_reason = 'STILL_HOLDING'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
    
    results.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'hl_price': hl_price,
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

# 결과 출력
print("\n" + "="*80)
print("📊 전체 성과")
print("="*80)

print(f"\n총 거래: {len(result_df)}건")
print(f"승률: {(result_df['total_pnl'] > 0).mean() * 100:.1f}%")
print(f"평균 수익: {result_df['total_pnl'].mean():+.2f}%")
print(f"총 수익: {result_df['total_pnl'].sum():+.1f}%")

print(f"\nTP1 달성: {result_df['tp1_done'].mean() * 100:.1f}%")
print(f"TP2 달성: {result_df['tp2_done'].mean() * 100:.1f}%")
print(f"HL 손절: {result_df['sl_done'].mean() * 100:.1f}%")

# 출구별 분석
print("\n" + "="*80)
print("📊 출구별 분석")
print("="*80)

exit_stats = result_df.groupby('exit_reason').agg({
    'total_pnl': ['count', 'mean', 'sum'],
    'hold_hours': 'mean'
}).round(2)
exit_stats.columns = ['건수', '평균수익', '총수익', '평균보유시간']
exit_stats['비율'] = (exit_stats['건수'] / len(result_df) * 100).round(1)

print(exit_stats)

# 보유 시간 분석
print("\n" + "="*80)
print("📊 보유 시간 분석")
print("="*80)

print(f"평균 보유: {result_df['hold_hours'].mean():.1f}시간 ({result_df['hold_hours'].mean()/24:.1f}일)")
print(f"최소 보유: {result_df['hold_hours'].min():.1f}시간")
print(f"최대 보유: {result_df['hold_hours'].max():.1f}시간 ({result_df['hold_hours'].max()/24:.1f}일)")

# 연도별 성과
print("\n" + "="*80)
print("📊 연도별 성과")
print("="*80)

result_df['year'] = pd.to_datetime(result_df['entry_time']).dt.year
yearly = result_df.groupby('year').agg({
    'total_pnl': ['count', 'sum', 'mean'],
    'tp1_done': 'sum',
    'sl_done': 'sum'
}).round(2)
yearly.columns = ['거래수', '총수익', '평균수익', 'TP달성', 'SL발생']
yearly['승률'] = (result_df.groupby('year')['total_pnl'].apply(lambda x: (x > 0).sum()) / yearly['거래수'] * 100).round(1)

print(yearly)

# 저장
result_df.to_csv('correct_logic_backtest.csv', index=False)
print(f"\n저장: correct_logic_backtest.csv")

