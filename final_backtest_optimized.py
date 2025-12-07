"""
최적화된 전략으로 최종 백테스트
고정 TP 10% / SL 3% / Trailing SL / 72h / Split 30%
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')

signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("최적화된 전략 최종 백테스트")
print("="*80)

# 최적 파라미터
TP_PCT = 10  # 10%
SL_PCT = 3   # 3%
USE_TRAILING = True
MAX_HOLD_HOURS = 72
SPLIT_RATIO = 0.3  # 30%

print(f"\n전략 파라미터:")
print(f"  TP: +{TP_PCT}% (분할: {int(TP_PCT/2)}%, {TP_PCT}%)")
print(f"  SL: -{SL_PCT}%")
print(f"  Trailing SL: {USE_TRAILING}")
print(f"  최대 보유: {MAX_HOLD_HOURS}시간")
print(f"  1차 익절 비율: {SPLIT_RATIO*100}%")

# 백테스트 실행
results = []

for idx, signal in signals.iterrows():
    entry_time = signal['breakout_time']
    entry_price = signal['breakout_price']
    hl_price = signal['hl_price']
    
    # TP/SL 목표가
    tp1_target = entry_price * (1 + TP_PCT / 100 * 0.5)  # 5%
    tp2_target = entry_price * (1 + TP_PCT / 100)  # 10%
    sl_price = entry_price * (1 - SL_PCT / 100)  # -3%
    
    # 미래 데이터
    future_data = df_15m[df_15m['datetime'] > entry_time].head(MAX_HOLD_HOURS * 4)
    
    if len(future_data) == 0:
        continue
    
    position = 1.0
    total_pnl = 0
    tp1_done = False
    tp2_done = False
    sl_done = False
    current_sl = sl_price
    exit_reason = None
    exit_time = None
    exit_price = None
    
    for _, candle in future_data.iterrows():
        candle_time = candle['datetime']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        hold_hours = (candle_time - entry_time).total_seconds() / 3600
        
        # 시간 초과
        if hold_hours > MAX_HOLD_HOURS:
            if position > 0:
                pnl = (close - entry_price) / entry_price * 100 * position
                total_pnl += pnl
                exit_reason = 'TIME'
                exit_time = candle_time
                exit_price = close
                position = 0
            break
        
        # 손절
        if low <= current_sl and position > 0:
            sl_pnl = (current_sl - entry_price) / entry_price * 100 * position
            total_pnl += sl_pnl
            sl_done = True
            exit_reason = 'SL' if not tp1_done else 'SL_AFTER_TP1'
            exit_time = candle_time
            exit_price = current_sl
            position = 0
            break
        
        # TP1
        if not tp1_done and high >= tp1_target and position > 0:
            tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * SPLIT_RATIO
            total_pnl += tp1_pnl
            position -= SPLIT_RATIO
            tp1_done = True
            
            if USE_TRAILING:
                current_sl = entry_price
        
        # TP2
        if not tp2_done and high >= tp2_target and position > 0:
            tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position
            total_pnl += tp2_pnl
            exit_reason = 'TP2'
            exit_time = candle_time
            exit_price = tp2_target
            position = 0
            tp2_done = True
            break
    
    # 잔여 포지션
    if position > 0 and len(future_data) > 0:
        last_candle = future_data.iloc[-1]
        pnl = (last_candle['close'] - entry_price) / entry_price * 100 * position
        total_pnl += pnl
        exit_reason = exit_reason or 'HOLD'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
    
    results.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'hl_price': hl_price,
        'tp1_target': tp1_target,
        'tp2_target': tp2_target,
        'sl_price': sl_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'tp1_done': tp1_done,
        'tp2_done': tp2_done,
        'sl_done': sl_done,
        'total_pnl': total_pnl
    })

result_df = pd.DataFrame(results)

# 전체 성과
print("\n" + "="*80)
print("전체 성과")
print("="*80)

print(f"\n총 거래 수: {len(result_df)}")
print(f"승률 (PnL > 0): {(result_df['total_pnl'] > 0).mean() * 100:.1f}%")
print(f"평균 수익률: {result_df['total_pnl'].mean():+.2f}%")
print(f"총 수익률: {result_df['total_pnl'].sum():+.1f}%")

print(f"\nTP1 달성률: {result_df['tp1_done'].mean() * 100:.1f}%")
print(f"TP2 달성률: {result_df['tp2_done'].mean() * 100:.1f}%")
print(f"손절률: {result_df['sl_done'].mean() * 100:.1f}%")

# 출구별 분석
print("\n" + "="*80)
print("출구별 분석")
print("="*80)

exit_stats = result_df.groupby('exit_reason').agg({
    'total_pnl': ['count', 'mean', 'sum']
}).round(2)
exit_stats.columns = ['count', 'avg_pnl', 'total_pnl']
exit_stats['pct'] = exit_stats['count'] / len(result_df) * 100

print(exit_stats)

# 연도별 성과
print("\n" + "="*80)
print("연도별 성과")
print("="*80)

result_df['year'] = pd.to_datetime(result_df['entry_time']).dt.year

yearly = result_df.groupby('year').agg({
    'total_pnl': ['count', 'sum', 'mean'],
    'tp1_done': 'sum',
    'tp2_done': 'sum',
    'sl_done': 'sum'
}).round(2)

yearly.columns = ['trades', 'total_pnl', 'avg_pnl', 'tp1_count', 'tp2_count', 'sl_count']
yearly['win_rate'] = (result_df.groupby('year')['total_pnl'].apply(lambda x: (x > 0).sum()) / yearly['trades'] * 100).round(1)

print(yearly)

# CAGR 계산
total_years = (result_df['entry_time'].max() - result_df['entry_time'].min()).days / 365.25
cumulative_return = result_df['total_pnl'].sum() / 100
cagr = ((1 + cumulative_return) ** (1 / total_years) - 1) * 100

print(f"\n기간: {result_df['entry_time'].min().strftime('%Y-%m-%d')} ~ {result_df['entry_time'].max().strftime('%Y-%m-%d')}")
print(f"총 기간: {total_years:.1f}년")
print(f"누적 수익률: {cumulative_return * 100:+.1f}%")
print(f"CAGR: {cagr:.1f}%")

# 수익분포
print("\n" + "="*80)
print("수익 분포")
print("="*80)

def categorize_pnl(pnl):
    if pnl < -5:
        return '1. 큰 손실 (<-5%)'
    elif pnl < -2:
        return '2. 중간 손실 (-5~-2%)'
    elif pnl < 0:
        return '3. 소 손실 (-2~0%)'
    elif pnl < 2:
        return '4. 소 이익 (0~2%)'
    elif pnl < 5:
        return '5. 중간 이익 (2~5%)'
    else:
        return '6. 큰 이익 (>5%)'

result_df['pnl_category'] = result_df['total_pnl'].apply(categorize_pnl)
dist = result_df['pnl_category'].value_counts().sort_index()

for cat, count in dist.items():
    subset = result_df[result_df['pnl_category'] == cat]
    avg = subset['total_pnl'].mean()
    print(f"{cat}: {count}건 ({count/len(result_df)*100:.1f}%) | 평균: {avg:+.2f}%")

# 결과 저장
result_df.to_csv('optimized_backtest_results.csv', index=False)
print(f"\n결과 저장: optimized_backtest_results.csv ({len(result_df)}건)")

# 기존 대비 비교
print("\n" + "="*80)
print("📊 기존 전략 대비 비교")
print("="*80)

print("""
┌─────────────────┬─────────────┬─────────────┐
│      항목       │   기존전략   │  최적화전략  │
├─────────────────┼─────────────┼─────────────┤
│ TP/SL           │  5%/-2%     │  10%/-3%    │
│ Trailing SL     │  없음       │  있음       │
│ 분할 익절       │  없음       │  30%/70%    │
├─────────────────┼─────────────┼─────────────┤""")
print(f"│ TP 달성률       │  17.3%      │  {result_df['tp1_done'].mean()*100:.1f}%      │")
print(f"│ 승률            │  26.4%      │  {(result_df['total_pnl'] > 0).mean()*100:.1f}%      │")
print(f"│ 평균 수익       │  +0.60%     │  {result_df['total_pnl'].mean():+.2f}%     │")
print(f"│ CAGR            │  ~50%       │  {cagr:.1f}%      │")
print("└─────────────────┴─────────────┴─────────────┘")

print(f"\n개선율:")
print(f"  - TP1 달성률: {(result_df['tp1_done'].mean()*100 / 17.3 - 1) * 100:+.0f}%")
print(f"  - 승률: {((result_df['total_pnl'] > 0).mean()*100 / 26.4 - 1) * 100:+.0f}%")
print(f"  - 평균 수익: {(result_df['total_pnl'].mean() / 0.60 - 1) * 100:+.0f}%")

