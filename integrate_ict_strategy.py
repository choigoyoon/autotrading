"""
ICT (Inner Circle Trader) 개념을 BB 수축-확장 전략에 통합
- Fair Value Gaps (FVG) - 이미 계산됨
- Large Candles (Institutional Moves)
- Market Structure (BOS/ChoCh)
- Order Blocks (OB)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load existing data with FVG already calculated
df = pd.read_csv('analysis_1h.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"데이터 로드 완료: {len(df)} 캔들")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# ============================================================================
# ICT 인디케이터 계산
# ============================================================================

def identify_large_candles(df, threshold_pct=2.0):
    """Large Candles (기관 개입) 식별"""
    df['candle_size_pct'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['is_large_candle'] = df['candle_size_pct'] >= threshold_pct
    df['large_candle_direction'] = 0
    df.loc[(df['is_large_candle']) & (df['close'] > df['open']), 'large_candle_direction'] = 1
    df.loc[(df['is_large_candle']) & (df['close'] < df['open']), 'large_candle_direction'] = -1
    
    return df

def identify_market_structure(df, lookback=20):
    """Market Structure (BOS/ChoCh) 식별"""
    # Swing highs/lows already exist in the data
    # Add HH/LL/BOS/ChoCh
    df['hh'] = False
    df['ll'] = False
    df['bos'] = False
    df['choch'] = False
    
    last_high = None
    last_low = None
    trend = 0  # 1: uptrend, -1: downtrend, 0: neutral
    
    for i in range(lookback, len(df)):
        if df.loc[i, 'swing_high']:
            if last_high is not None:
                if df.loc[i, 'high'] > last_high:
                    df.loc[i, 'hh'] = True
                    if trend == 1:
                        df.loc[i, 'bos'] = True  # Break of Structure (상승 지속)
                    elif trend == -1:
                        df.loc[i, 'choch'] = True  # Change of Character (추세 전환)
                        trend = 1
                    else:
                        trend = 1
            last_high = df.loc[i, 'high']
        
        if df.loc[i, 'swing_low']:
            if last_low is not None:
                if df.loc[i, 'low'] < last_low:
                    df.loc[i, 'll'] = True
                    if trend == -1:
                        df.loc[i, 'bos'] = True  # Break of Structure (하락 지속)
                    elif trend == 1:
                        df.loc[i, 'choch'] = True  # Change of Character (추세 전환)
                        trend = -1
                    else:
                        trend = -1
            last_low = df.loc[i, 'low']
    
    return df

print("\n" + "="*80)
print("ICT 인디케이터 계산 중...")
print("="*80)

# 1. FVG already calculated
print(f"✅ Bullish FVG: {df['fvg_bullish'].sum()}개 (기존 데이터)")
print(f"✅ Bearish FVG: {df['fvg_bearish'].sum()}개 (기존 데이터)")

# 2. Large Candles
df = identify_large_candles(df, threshold_pct=2.0)
print(f"✅ Large Candles (≥2%): {df['is_large_candle'].sum()}개")
print(f"   - Bullish: {(df['large_candle_direction'] == 1).sum()}개")
print(f"   - Bearish: {(df['large_candle_direction'] == -1).sum()}개")

# 3. Market Structure
df = identify_market_structure(df, lookback=20)
print(f"✅ Higher Highs: {df['hh'].sum()}개")
print(f"✅ Lower Lows: {df['ll'].sum()}개")
print(f"✅ BOS (Break of Structure): {df['bos'].sum()}개")
print(f"✅ ChoCh (Change of Character): {df['choch'].sum()}개")

# ============================================================================
# BB 수축-확장 데이터와 결합
# ============================================================================

print("\n" + "="*80)
print("BB 수축-확장 데이터와 ICT 지표 결합 중...")
print("="*80)

# Load BB squeeze data
squeeze_df = pd.read_csv('expanded_squeeze_analysis.csv', parse_dates=['datetime'])

print(f"BB 수축-확장 이벤트: {len(squeeze_df)}개")

# Merge ICT indicators
squeeze_df = squeeze_df.merge(
    df[['datetime', 'is_large_candle', 'large_candle_direction', 'candle_size_pct',
        'fvg_bullish', 'fvg_bearish', 'fvg_size', 'hh', 'll', 'bos', 'choch']],
    on='datetime',
    how='left'
)

print(f"✅ 데이터 결합 완료: {len(squeeze_df)}개 이벤트")

# ============================================================================
# ICT 통합 전략 백테스트
# ============================================================================

print("\n" + "="*80)
print("ICT 통합 전략 백테스트")
print("="*80)

strategies = [
    # 기존 최고 전략
    {
        'name': '1. BB: M200≥8 + above_ema50',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['above_ema200']
        )
    },
    
    # ICT 추가 전략
    {
        'name': '2. BB+ICT: M200≥8 + FVG + Large Candle',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['fvg_bullish'] and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1
        )
    },
    {
        'name': '3. BB+ICT: M200≥8 + FVG + BOS',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['fvg_bullish'] and
            row['bos']
        )
    },
    {
        'name': '4. BB+ICT: M200≥8 + Large Candle + ChoCh',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1 and
            row['choch']
        )
    },
    {
        'name': '5. BB+ICT: M100≥0 + EMA정배열 + FVG + Large',
        'conditions': lambda row: (
            row['momentum_100'] >= 0 and
            row['ema_bull'] and
            row['fvg_bullish'] and
            row['is_large_candle']
        )
    },
    {
        'name': '6. BB+ICT: M200≥5 + FVG + HH',
        'conditions': lambda row: (
            row['momentum_200'] >= 5 and
            row['fvg_bullish'] and
            row['HH']
        )
    },
    {
        'name': '7. BB+ICT: M200≥8 + Large(≥3%) + BOS',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['candle_size_pct'] >= 3.0 and
            row['large_candle_direction'] == 1 and
            row['bos']
        )
    },
    {
        'name': '8. BB+ICT: M200≥10 + FVG + Large + HH',
        'conditions': lambda row: (
            row['momentum_200'] >= 10 and
            row['fvg_bullish'] and
            row['is_large_candle'] and
            row['HH']
        )
    },
]

results = []

for strategy in strategies:
    # Filter trades
    trades = squeeze_df[squeeze_df.apply(strategy['conditions'], axis=1)].copy()
    
    if len(trades) == 0:
        results.append({
            'strategy': strategy['name'],
            'count': 0,
            'annual': 0,
            'win_rate_336h': 0,
            'avg_pnl_336h': 0,
            'total_pnl_336h': 0,
            'win_rate_168h': 0,
            'avg_pnl_168h': 0,
            'score': 0
        })
        continue
    
    # 336h performance
    win_336 = (trades['long_336h'] > 0).sum()
    total = len(trades)
    win_rate_336 = win_336 / total * 100 if total > 0 else 0
    avg_pnl_336 = trades['long_336h'].mean()
    total_pnl_336 = trades['long_336h'].sum()
    
    # 168h performance
    win_168 = (trades['long_168h'] > 0).sum()
    win_rate_168 = win_168 / total * 100 if total > 0 else 0
    avg_pnl_168 = trades['long_168h'].mean()
    
    results.append({
        'strategy': strategy['name'],
        'count': total,
        'annual': total / 5,
        'win_rate_336h': win_rate_336,
        'avg_pnl_336h': avg_pnl_336,
        'total_pnl_336h': total_pnl_336,
        'win_rate_168h': win_rate_168,
        'avg_pnl_168h': avg_pnl_168,
        'score': win_rate_336 * avg_pnl_336
    })

results_df = pd.DataFrame(results).sort_values('score', ascending=False)

print("\n" + "="*80)
print("🏆 ICT 통합 전략 성과 (Score 순위)")
print("="*80)
for idx, row in results_df.iterrows():
    if row['count'] == 0:
        print(f"\n❌ {row['strategy']}")
        print(f"   거래 없음")
        continue
    
    print(f"\n📊 {row['strategy']}")
    print(f"   거래 횟수: {row['count']}회 (연 {row['annual']:.1f}회)")
    print(f"   336h 승률: {row['win_rate_336h']:.1f}% | 평균: {row['avg_pnl_336h']:+.2f}% | 누적: {row['total_pnl_336h']:+.0f}%")
    print(f"   168h 승률: {row['win_rate_168h']:.1f}% | 평균: {row['avg_pnl_168h']:+.2f}%")
    print(f"   Score: {row['score']:.1f}")

# Additional analysis: 고빈도 전략
print("\n" + "="*80)
print("📈 고빈도 전략 (연 30회 이상)")
print("="*80)

high_freq = results_df[results_df['annual'] >= 30].copy()
if len(high_freq) > 0:
    for idx, row in high_freq.iterrows():
        print(f"\n✅ {row['strategy']}")
        print(f"   {row['annual']:.1f}회/년 | 승률 {row['win_rate_336h']:.1f}% | 평균 {row['avg_pnl_336h']:+.2f}%")
else:
    print("고빈도 조건을 만족하는 전략이 없습니다.")

# Save results
results_df.to_csv('ict_integrated_strategy_results.csv', index=False)
squeeze_df.to_csv('squeeze_with_ict_indicators.csv', index=False)

# Save ICT indicators to main dataframe
df.to_csv('analysis_1h_with_ict.csv', index=False)

print("\n" + "="*80)
print("✅ 결과 저장 완료:")
print("   - ict_integrated_strategy_results.csv")
print("   - squeeze_with_ict_indicators.csv")
print("   - analysis_1h_with_ict.csv")
print("="*80)

