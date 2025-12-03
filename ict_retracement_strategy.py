"""
진짜 ICT 매매 방식: 되돌림(Retracement) 기반 진입
1. BB 확장 + Large Candle로 FVG/OB 생성
2. 가격이 FVG/OB로 되돌아올 때까지 대기 (3~24시간)
3. FVG/OB 구간에서 반등 확인 후 진입
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data with ICT indicators
df = pd.read_csv('analysis_1h_with_ict.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Load squeeze data for momentum
squeeze_df = pd.read_csv('squeeze_with_ict_indicators.csv', parse_dates=['datetime'])

# Merge momentum data
df = df.merge(squeeze_df[['datetime', 'momentum_50', 'momentum_100', 'momentum_200']], 
              on='datetime', how='left')

print(f"데이터 로드 완료: {len(df)} 캔들")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# ============================================================================
# FVG/OB 되돌림 매매 시뮬레이션
# ============================================================================

def find_retracement_entry(df, start_idx, fvg_bottom, fvg_top, direction='bullish', max_wait_hours=24):
    """
    FVG/OB 구간으로 되돌아오는 지점 찾기
    """
    for wait in range(1, max_wait_hours + 1):
        idx = start_idx + wait
        if idx >= len(df):
            return None, None, None
        
        current_low = df.loc[idx, 'low']
        current_high = df.loc[idx, 'high']
        
        if direction == 'bullish':
            # Bullish: 가격이 FVG 구간으로 하락했는지 확인
            if current_low <= fvg_top and current_low >= fvg_bottom:
                # 반등 확인: 양봉이거나 긴 아래꼬리
                is_bullish_candle = df.loc[idx, 'close'] > df.loc[idx, 'open']
                lower_wick = df.loc[idx, 'open'] - df.loc[idx, 'low'] if not is_bullish_candle else df.loc[idx, 'close'] - df.loc[idx, 'low']
                body_size = abs(df.loc[idx, 'close'] - df.loc[idx, 'open'])
                
                # 반등 조건: 양봉 또는 아래꼬리가 몸통보다 김
                if is_bullish_candle or lower_wick > body_size * 0.5:
                    entry_price = (fvg_bottom + fvg_top) / 2  # FVG 중간 가격
                    return idx, entry_price, wait
            
            # 너무 많이 하락하면 포기
            if current_low < fvg_bottom * 0.97:
                return None, None, None
    
    return None, None, None

def calculate_forward_returns(df, entry_idx, entry_price, hold_periods=[168, 336]):
    """진입 이후 수익률 계산"""
    returns = {}
    
    for period in hold_periods:
        exit_idx = entry_idx + period
        if exit_idx >= len(df):
            returns[f'pnl_{period}h'] = None
        else:
            exit_price = df.loc[exit_idx, 'close']
            pnl = (exit_price - entry_price) / entry_price * 100
            returns[f'pnl_{period}h'] = pnl
    
    return returns

# ============================================================================
# 전략 시뮬레이션
# ============================================================================

print("\n" + "="*80)
print("ICT 되돌림 전략 시뮬레이션")
print("="*80)

strategies = [
    {
        'name': 'M200≥8 + Large(≥2%) + FVG 되돌림',
        'momentum_threshold': 8,
        'large_candle_threshold': 2.0,
        'max_wait': 24
    },
    {
        'name': 'M200≥8 + Large(≥3%) + FVG 되돌림',
        'momentum_threshold': 8,
        'large_candle_threshold': 3.0,
        'max_wait': 24
    },
    {
        'name': 'M200≥10 + Large(≥2%) + FVG 되돌림',
        'momentum_threshold': 10,
        'large_candle_threshold': 2.0,
        'max_wait': 24
    },
    {
        'name': 'M200≥8 + Large(≥2%) + FVG 되돌림 (10H)',
        'momentum_threshold': 8,
        'large_candle_threshold': 2.0,
        'max_wait': 10
    },
    {
        'name': 'M200≥5 + Large(≥2%) + FVG 되돌림',
        'momentum_threshold': 5,
        'large_candle_threshold': 2.0,
        'max_wait': 24
    },
    {
        'name': 'M100≥5 + Large(≥2%) + FVG 되돌림',
        'momentum_threshold': 0,  # M100 사용
        'momentum_col': 'momentum_100',
        'momentum_min': 5,
        'large_candle_threshold': 2.0,
        'max_wait': 24
    },
]

all_trades = []

for strategy in strategies:
    print(f"\n{'='*80}")
    print(f"전략: {strategy['name']}")
    print(f"{'='*80}")
    
    trades = []
    
    # FVG 생성 시점 찾기
    for i in range(100, len(df) - 400):
        # FVG bullish 확인
        if not df.loc[i, 'fvg_bullish']:
            continue
        
        # Momentum check
        momentum_col = strategy.get('momentum_col', 'momentum_200')
        momentum = df.loc[i, momentum_col] if pd.notna(df.loc[i, momentum_col]) else 0
        momentum_min = strategy.get('momentum_min', strategy['momentum_threshold'])
        
        if momentum < momentum_min:
            continue
        
        # Large Candle check (최근 3개 캔들)
        is_large = False
        for j in range(i, max(0, i-3), -1):
            if (df.loc[j, 'candle_size_pct'] >= strategy['large_candle_threshold'] and 
                df.loc[j, 'large_candle_direction'] == 1):
                is_large = True
                break
        
        if not is_large:
            continue
        
        # FVG 구간 계산
        if i < 2:
            continue
        
        fvg_bottom = df.loc[i-2, 'high']
        fvg_top = df.loc[i, 'low']
        
        if fvg_bottom >= fvg_top:
            continue
        
        fvg_size = (fvg_top - fvg_bottom) / fvg_bottom * 100
        
        # 되돌림 찾기
        entry_idx, entry_price, wait_hours = find_retracement_entry(
            df, i, fvg_bottom, fvg_top, 'bullish', strategy['max_wait']
        )
        
        if entry_idx is None:
            continue
        
        # 수익률 계산
        returns = calculate_forward_returns(df, entry_idx, entry_price)
        
        if returns['pnl_336h'] is None:
            continue
        
        trades.append({
            'strategy': strategy['name'],
            'fvg_datetime': df.loc[i, 'datetime'],
            'entry_datetime': df.loc[entry_idx, 'datetime'],
            'wait_hours': wait_hours,
            'entry_price': entry_price,
            'fvg_size': fvg_size,
            'momentum': momentum,
            'pnl_168h': returns['pnl_168h'],
            'pnl_336h': returns['pnl_336h'],
        })
    
    if len(trades) == 0:
        print(f"✖️  거래 없음")
        continue
    
    trades_df = pd.DataFrame(trades)
    all_trades.extend(trades)
    
    # 성과 분석
    total = len(trades_df)
    win_336 = (trades_df['pnl_336h'] > 0).sum()
    win_rate_336 = win_336 / total * 100
    avg_pnl_336 = trades_df['pnl_336h'].mean()
    total_pnl_336 = trades_df['pnl_336h'].sum()
    avg_wait = trades_df['wait_hours'].mean()
    
    win_168 = (trades_df['pnl_168h'] > 0).sum()
    win_rate_168 = win_168 / total * 100
    avg_pnl_168 = trades_df['pnl_168h'].mean()
    
    print(f"✅ 거래 횟수: {total}회 (연 {total/5:.1f}회)")
    print(f"   평균 대기 시간: {avg_wait:.1f}시간")
    print(f"   336h 승률: {win_rate_336:.1f}% | 평균: {avg_pnl_336:+.2f}% | 누적: {total_pnl_336:+.0f}%")
    print(f"   168h 승률: {win_rate_168:.1f}% | 평균: {avg_pnl_168:+.2f}%")
    print(f"   Score: {win_rate_336 * avg_pnl_336:.1f}")

# 결과 저장
if len(all_trades) > 0:
    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv('ict_retracement_trades.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 기존 즉시진입 vs ICT 되돌림 전략 비교")
    print("="*80)
    
    # 기존 결과 로드
    prev_results = pd.read_csv('ict_integrated_strategy_results.csv')
    
    print("\n🔵 기존 전략 (즉시 진입) TOP 3")
    print("-" * 80)
    for idx, row in prev_results.head(3).iterrows():
        print(f"{row['strategy']}")
        print(f"  {row['annual']:.1f}회/년 | 승률 {row['win_rate_336h']:.1f}% | 평균 {row['avg_pnl_336h']:+.2f}% | Score {row['score']:.1f}")
        print()
    
    # ICT 되돌림 요약
    print("🟢 ICT 되돌림 전략 (FVG 되돌림 대기)")
    print("-" * 80)
    summary = []
    for strat_name in set([t['strategy'] for t in all_trades]):
        strat_trades = [t for t in all_trades if t['strategy'] == strat_name]
        strat_df = pd.DataFrame(strat_trades)
        total = len(strat_df)
        win_336 = (strat_df['pnl_336h'] > 0).sum()
        win_rate_336 = win_336 / total * 100
        avg_pnl_336 = strat_df['pnl_336h'].mean()
        score = win_rate_336 * avg_pnl_336
        avg_wait = strat_df['wait_hours'].mean()
        
        summary.append({
            'strategy': strat_name,
            'annual': total / 5,
            'win_rate': win_rate_336,
            'avg_pnl': avg_pnl_336,
            'score': score,
            'avg_wait': avg_wait
        })
    
    summary_df = pd.DataFrame(summary).sort_values('score', ascending=False)
    for idx, row in summary_df.iterrows():
        print(f"{row['strategy']}")
        print(f"  {row['annual']:.1f}회/년 | 승률 {row['win_rate']:.1f}% | 평균 {row['avg_pnl']:+.2f}% | Score {row['score']:.1f}")
        print(f"  ⏰ 평균 대기: {row['avg_wait']:.1f}시간")
        print()
    
    print("="*80)
    print("✅ 결과 저장: ict_retracement_trades.csv")
    print("="*80)

