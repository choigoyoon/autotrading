#!/usr/bin/env python3
"""
MTF 전략 + 트레일링 스탑

문제점:
- MFE >= 1% 후 SL 비율: 59.1%
- 수익권 진입 후 손절이 너무 많음

해결책:
1. MFE 2% 도달 시 SL을 손익분기(BE)로 이동
2. MFE 3% 도달 시 SL을 +1%로 이동
3. TP는 5% 유지
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 전략 + 트레일링 스탑")
print("=" * 80)

# 이전 결과 로드
df = pd.read_csv('mtf_optimized_long_short_results.csv')
print(f"총 신호: {len(df)}건")

# 15M 데이터 로드
df_15m = pd.read_csv('analysis_15m.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m[df_15m['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

# 최적 조건 필터
df_optimal = df[(df['gap'] >= 4) & (df['entry_gap'] >= 2) & (df['entry_gap'] < 4)].copy()
print(f"최적 조건 (Gap>=4%, Entry Gap 2-4%): {len(df_optimal)}건")

# ============================================================
# 트레일링 스탑 백테스트
# ============================================================
print("\n" + "=" * 80)
print("트레일링 스탑 백테스트")
print("=" * 80)

def backtest_with_trailing(row, df_15m, trailing_type='fixed'):
    """
    트레일링 스탑 적용 백테스트
    
    trailing_type:
    - 'fixed': 고정 TP 5%
    - 'trailing_be': MFE 2% 도달 시 BE로 이동
    - 'trailing_1%': MFE 3% 도달 시 +1%로 이동
    - 'trailing_2%': MFE 4% 도달 시 +2%로 이동
    """
    entry_time = pd.to_datetime(row['time'])
    entry_price = row['entry']
    sl_price = row['sl']
    direction = row['direction']
    
    # TP 가격 계산
    if direction == 'LONG':
        tp_price = entry_price * 1.05
    else:
        tp_price = entry_price * 0.95
    
    # 15M 데이터 찾기
    entry_idx = df_15m[df_15m['datetime'] >= entry_time].index
    if len(entry_idx) == 0:
        return None
    entry_idx = entry_idx[0]
    
    post = df_15m.iloc[entry_idx+1:entry_idx+500]
    
    if len(post) == 0:
        return None
    
    current_sl = sl_price
    mfe = 0
    be_triggered = False
    trail_triggered = False
    
    for idx, candle in post.iterrows():
        if direction == 'LONG':
            current_pnl = (candle['high'] - entry_price) / entry_price * 100
            mfe = max(mfe, current_pnl)
            
            # 트레일링 스탑 적용
            if trailing_type in ['trailing_be', 'trailing_1%', 'trailing_2%']:
                if mfe >= 2 and not be_triggered:
                    current_sl = entry_price  # BE로 이동
                    be_triggered = True
                
                if trailing_type == 'trailing_1%' and mfe >= 3 and not trail_triggered:
                    current_sl = entry_price * 1.01  # +1%로 이동
                    trail_triggered = True
                
                if trailing_type == 'trailing_2%' and mfe >= 4 and not trail_triggered:
                    current_sl = entry_price * 1.02  # +2%로 이동
                    trail_triggered = True
            
            # TP 체크
            if candle['high'] >= tp_price:
                return {
                    'pnl': 5.0,
                    'exit_type': 'TP',
                    'mfe': mfe,
                    'be_triggered': be_triggered,
                    'trail_triggered': trail_triggered
                }
            
            # SL 체크
            if candle['low'] <= current_sl:
                pnl = (current_sl - entry_price) / entry_price * 100
                return {
                    'pnl': pnl,
                    'exit_type': 'SL' if current_sl == sl_price else 'TRAIL_SL',
                    'mfe': mfe,
                    'be_triggered': be_triggered,
                    'trail_triggered': trail_triggered
                }
        
        else:  # SHORT
            current_pnl = (entry_price - candle['low']) / entry_price * 100
            mfe = max(mfe, current_pnl)
            
            # 트레일링 스탑 적용
            if trailing_type in ['trailing_be', 'trailing_1%', 'trailing_2%']:
                if mfe >= 2 and not be_triggered:
                    current_sl = entry_price  # BE로 이동
                    be_triggered = True
                
                if trailing_type == 'trailing_1%' and mfe >= 3 and not trail_triggered:
                    current_sl = entry_price * 0.99  # -1%로 이동 (SHORT)
                    trail_triggered = True
                
                if trailing_type == 'trailing_2%' and mfe >= 4 and not trail_triggered:
                    current_sl = entry_price * 0.98  # -2%로 이동 (SHORT)
                    trail_triggered = True
            
            # TP 체크
            if candle['low'] <= tp_price:
                return {
                    'pnl': 5.0,
                    'exit_type': 'TP',
                    'mfe': mfe,
                    'be_triggered': be_triggered,
                    'trail_triggered': trail_triggered
                }
            
            # SL 체크
            if candle['high'] >= current_sl:
                pnl = (entry_price - current_sl) / entry_price * 100
                return {
                    'pnl': pnl,
                    'exit_type': 'SL' if current_sl == sl_price else 'TRAIL_SL',
                    'mfe': mfe,
                    'be_triggered': be_triggered,
                    'trail_triggered': trail_triggered
                }
    
    # TIMEOUT
    return {
        'pnl': 0,
        'exit_type': 'TIMEOUT',
        'mfe': mfe,
        'be_triggered': be_triggered,
        'trail_triggered': trail_triggered
    }

# 각 트레일링 타입별 백테스트
trailing_types = ['fixed', 'trailing_be', 'trailing_1%', 'trailing_2%']
results_comparison = {}

for ttype in trailing_types:
    print(f"\n[{ttype}] 백테스트 중...")
    
    results = []
    for idx, row in df_optimal.iterrows():
        result = backtest_with_trailing(row, df_15m, ttype)
        if result:
            result['time'] = row['time']
            result['direction'] = row['direction']
            result['gap'] = row['gap']
            result['entry_gap'] = row['entry_gap']
            results.append(result)
    
    df_result = pd.DataFrame(results)
    results_comparison[ttype] = df_result
    
    if len(df_result) > 0:
        wr = (df_result['pnl'] > 0).mean() * 100
        ap = df_result['pnl'].mean()
        tp = df_result['pnl'].sum()
        
        df_result['month'] = pd.to_datetime(df_result['time']).dt.to_period('M')
        monthly = df_result.groupby('month')['pnl'].sum()
        
        print(f"  건수: {len(df_result)}")
        print(f"  승률: {wr:.1f}%")
        print(f"  평균 PnL: {ap:.2f}%")
        print(f"  총 PnL: {tp:.1f}%")
        print(f"  월 평균: {monthly.mean():.2f}%")
        print(f"  3x 레버리지: {monthly.mean()*3:.1f}%/월")

# ============================================================
# 결과 비교
# ============================================================
print("\n" + "=" * 80)
print("트레일링 스탑 전략 비교")
print("=" * 80)

print(f"\n{'전략':>20} {'건수':>8} {'승률%':>10} {'평균PnL':>10} {'총PnL':>10} {'월평균':>10} {'3x월':>10}")
print("-" * 90)

for ttype, df_result in results_comparison.items():
    if len(df_result) > 0:
        wr = (df_result['pnl'] > 0).mean() * 100
        ap = df_result['pnl'].mean()
        tp = df_result['pnl'].sum()
        
        df_result['month'] = pd.to_datetime(df_result['time']).dt.to_period('M')
        monthly = df_result.groupby('month')['pnl'].sum()
        ma = monthly.mean()
        
        print(f"{ttype:>20} {len(df_result):>8} {wr:>10.1f} {ap:>10.2f} {tp:>10.1f} {ma:>10.2f} {ma*3:>10.1f}")

# ============================================================
# 트레일링 BE 상세 분석
# ============================================================
print("\n" + "=" * 80)
print("트레일링 BE 상세 분석")
print("=" * 80)

df_trail = results_comparison.get('trailing_be')
if df_trail is not None and len(df_trail) > 0:
    print(f"\n[Exit Type 분포]")
    print(df_trail['exit_type'].value_counts())
    
    print(f"\n[BE 트리거 비율]")
    print(f"  BE 트리거: {df_trail['be_triggered'].sum()} / {len(df_trail)} ({df_trail['be_triggered'].mean()*100:.1f}%)")
    
    # BE 트리거 후 결과
    be_triggered = df_trail[df_trail['be_triggered'] == True]
    if len(be_triggered) > 0:
        print(f"\n[BE 트리거 후 결과]")
        print(f"  건수: {len(be_triggered)}")
        print(f"  승률: {(be_triggered['pnl']>0).mean()*100:.1f}%")
        print(f"  평균 PnL: {be_triggered['pnl'].mean():.2f}%")
        print(be_triggered['exit_type'].value_counts())

# ============================================================
# LONG vs SHORT 비교
# ============================================================
print("\n" + "=" * 80)
print("방향별 트레일링 스탑 비교")
print("=" * 80)

df_trail = results_comparison.get('trailing_be')
if df_trail is not None and len(df_trail) > 0:
    for direction in ['LONG', 'SHORT']:
        subset = df_trail[df_trail['direction'] == direction]
        if len(subset) > 0:
            wr = (subset['pnl'] > 0).mean() * 100
            ap = subset['pnl'].mean()
            tp = subset['pnl'].sum()
            print(f"\n[{direction}]")
            print(f"  건수: {len(subset)}")
            print(f"  승률: {wr:.1f}%")
            print(f"  평균 PnL: {ap:.2f}%")
            print(f"  총 PnL: {tp:.1f}%")

# ============================================================
# 최종 추천
# ============================================================
print("\n" + "=" * 80)
print("최종 추천")
print("=" * 80)

# 가장 좋은 전략 찾기
best_type = None
best_monthly = 0

for ttype, df_result in results_comparison.items():
    if len(df_result) > 0:
        df_result['month'] = pd.to_datetime(df_result['time']).dt.to_period('M')
        monthly = df_result.groupby('month')['pnl'].sum().mean()
        if monthly > best_monthly:
            best_monthly = monthly
            best_type = ttype

print(f"\n최적 전략: {best_type}")
print(f"월 평균 수익: {best_monthly:.2f}%")
print(f"3x 레버리지: {best_monthly*3:.1f}%/월")

# 최적 결과 저장
if best_type:
    best_df = results_comparison[best_type]
    best_df.to_csv('mtf_trailing_stop_results.csv', index=False)
    print(f"\n결과 저장: mtf_trailing_stop_results.csv")
