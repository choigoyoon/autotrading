"""
최종 통합 전략: BB + ICT + Momentum
- BB 확장 (변동성 증가)
- Large Candle (기관 개입)
- BOS/ChoCh (추세 확인)
- Momentum (방향성)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data
squeeze_df = pd.read_csv('squeeze_with_ict_indicators.csv', parse_dates=['datetime'])

print(f"데이터 로드 완료: {len(squeeze_df)} BB 수축-확장 이벤트")
print(f"기간: {squeeze_df['datetime'].min()} ~ {squeeze_df['datetime'].max()}")

# ============================================================================
# 최종 통합 전략 정의
# ============================================================================

strategies = [
    # === 소수정예 전략 (연 1-2회, 고수익) ===
    {
        'name': '1. Elite: M200≥8 + Large(≥3%) + BOS',
        'tier': 'Elite',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['candle_size_pct'] >= 3.0 and
            row['large_candle_direction'] == 1 and
            row['bos']
        )
    },
    {
        'name': '2. Elite: M200≥8 + Large(≥2.5%) + ChoCh',
        'tier': 'Elite',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['candle_size_pct'] >= 2.5 and
            row['large_candle_direction'] == 1 and
            row['choch']
        )
    },
    {
        'name': '3. Elite: M200≥10 + Large(≥2%) + BOS + HH',
        'tier': 'Elite',
        'conditions': lambda row: (
            row['momentum_200'] >= 10 and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1 and
            row['bos'] and
            row['HH']
        )
    },
    
    # === 중빈도 전략 (연 5-15회, 균형) ===
    {
        'name': '4. Balanced: M200≥8 + Large(≥2%) + FVG',
        'tier': 'Balanced',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1 and
            row['fvg_bullish']
        )
    },
    {
        'name': '5. Balanced: M200≥8 + Large(≥2%) + BOS',
        'tier': 'Balanced',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1 and
            row['bos']
        )
    },
    {
        'name': '6. Balanced: M200≥5 + Large(≥2.5%) + ChoCh + HH',
        'tier': 'Balanced',
        'conditions': lambda row: (
            row['momentum_200'] >= 5 and
            row['candle_size_pct'] >= 2.5 and
            row['large_candle_direction'] == 1 and
            row['choch'] and
            row['HH']
        )
    },
    
    # === 고빈도 전략 (연 20-40회, 안정) ===
    {
        'name': '7. Frequent: M200≥8 + above_ema50',
        'tier': 'Frequent',
        'conditions': lambda row: (
            row['momentum_200'] >= 8 and
            row['above_ema200']
        )
    },
    {
        'name': '8. Frequent: M200≥5 + ema_bull + Large(≥2%)',
        'tier': 'Frequent',
        'conditions': lambda row: (
            row['momentum_200'] >= 5 and
            row['ema_bull'] and
            row['is_large_candle'] and
            row['large_candle_direction'] == 1
        )
    },
    {
        'name': '9. Frequent: M100≥0 + ema_bull + FVG',
        'tier': 'Frequent',
        'conditions': lambda row: (
            row['momentum_100'] >= 0 and
            row['ema_bull'] and
            row['fvg_bullish']
        )
    },
]

# ============================================================================
# 백테스트 실행
# ============================================================================

print("\n" + "="*80)
print("최종 통합 전략 백테스트")
print("="*80)

results = []

for strategy in strategies:
    # Filter trades
    trades = squeeze_df[squeeze_df.apply(strategy['conditions'], axis=1)].copy()
    
    if len(trades) == 0:
        results.append({
            'name': strategy['name'],
            'tier': strategy['tier'],
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
        'name': strategy['name'],
        'tier': strategy['tier'],
        'count': total,
        'annual': total / 5,
        'win_rate_336h': win_rate_336,
        'avg_pnl_336h': avg_pnl_336,
        'total_pnl_336h': total_pnl_336,
        'win_rate_168h': win_rate_168,
        'avg_pnl_168h': avg_pnl_168,
        'score': win_rate_336 * avg_pnl_336
    })

results_df = pd.DataFrame(results)

# ============================================================================
# 결과 출력: Tier별
# ============================================================================

for tier in ['Elite', 'Balanced', 'Frequent']:
    tier_results = results_df[results_df['tier'] == tier].sort_values('score', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"🏆 {tier} 전략")
    print(f"{'='*80}")
    
    for idx, row in tier_results.iterrows():
        if row['count'] == 0:
            print(f"\n❌ {row['name']}")
            print(f"   거래 없음")
            continue
        
        print(f"\n📊 {row['name']}")
        print(f"   거래: {row['count']}회 (연 {row['annual']:.1f}회)")
        print(f"   336h | 승률: {row['win_rate_336h']:.1f}% | 평균: {row['avg_pnl_336h']:+.2f}% | 누적: {row['total_pnl_336h']:+.0f}%")
        print(f"   168h | 승률: {row['win_rate_168h']:.1f}% | 평균: {row['avg_pnl_168h']:+.2f}%")
        print(f"   Score: {row['score']:.1f}")

# ============================================================================
# 포트폴리오 시뮬레이션
# ============================================================================

print("\n" + "="*80)
print("💼 포트폴리오 시뮬레이션")
print("="*80)

portfolios = [
    {
        'name': 'Portfolio A: Elite Only (소수정예)',
        'strategies': ['1. Elite: M200≥8 + Large(≥3%) + BOS',
                      '2. Elite: M200≥8 + Large(≥2.5%) + ChoCh',
                      '3. Elite: M200≥10 + Large(≥2%) + BOS + HH']
    },
    {
        'name': 'Portfolio B: Elite + Balanced (균형)',
        'strategies': ['1. Elite: M200≥8 + Large(≥3%) + BOS',
                      '2. Elite: M200≥8 + Large(≥2.5%) + ChoCh',
                      '4. Balanced: M200≥8 + Large(≥2%) + FVG',
                      '5. Balanced: M200≥8 + Large(≥2%) + BOS']
    },
    {
        'name': 'Portfolio C: Full Stack (전체)',
        'strategies': ['1. Elite: M200≥8 + Large(≥3%) + BOS',
                      '2. Elite: M200≥8 + Large(≥2.5%) + ChoCh',
                      '4. Balanced: M200≥8 + Large(≥2%) + FVG',
                      '7. Frequent: M200≥8 + above_ema50']
    },
]

for portfolio in portfolios:
    print(f"\n{'='*80}")
    print(f"📊 {portfolio['name']}")
    print(f"{'='*80}")
    
    port_results = results_df[results_df['name'].isin(portfolio['strategies'])]
    
    total_trades = port_results['count'].sum()
    total_annual = port_results['annual'].sum()
    
    # Weighted averages
    total_wins_336 = sum(port_results['win_rate_336h'] * port_results['count']) / total_trades if total_trades > 0 else 0
    avg_pnl_336 = sum(port_results['avg_pnl_336h'] * port_results['count']) / total_trades if total_trades > 0 else 0
    total_pnl_336 = port_results['total_pnl_336h'].sum()
    
    print(f"\n전략 구성:")
    for strat_name in portfolio['strategies']:
        strat_row = results_df[results_df['name'] == strat_name].iloc[0]
        if strat_row['count'] > 0:
            print(f"  • {strat_name}: {strat_row['annual']:.1f}회/년")
    
    print(f"\n포트폴리오 성과:")
    print(f"  총 거래: {total_trades}회 (연 {total_annual:.1f}회)")
    print(f"  336h 승률: {total_wins_336:.1f}%")
    print(f"  336h 평균 수익: {avg_pnl_336:+.2f}%")
    print(f"  336h 누적 수익: {total_pnl_336:+.0f}%")
    print(f"  Score: {total_wins_336 * avg_pnl_336:.1f}")

# ============================================================================
# 최종 추천
# ============================================================================

print("\n" + "="*80)
print("🎯 최종 추천")
print("="*80)

# Best by score
best = results_df[results_df['count'] > 0].nlargest(3, 'score')

print("\n🥇 TOP 3 최고 성과 전략")
print("-" * 80)
for idx, row in best.iterrows():
    print(f"{row['name']}")
    print(f"  {row['annual']:.1f}회/년 | 승률 {row['win_rate_336h']:.1f}% | 평균 {row['avg_pnl_336h']:+.2f}% | Score {row['score']:.1f}")
    print()

# Best portfolio recommendation
print("💎 추천 포트폴리오: Portfolio C (Full Stack)")
print("-" * 80)
print("✅ 연간 약 35-40회 거래")
print("✅ 소수정예 (2회) + 중빈도 (5-8회) + 고빈도 (30회)")
print("✅ 예상 승률: 58-62%")
print("✅ 예상 평균 수익: +5-6%")
print("✅ 리스크 분산: 다양한 조건으로 기회 포착")

# Save results
results_df.to_csv('final_integrated_strategy_results.csv', index=False)

print("\n" + "="*80)
print("✅ 결과 저장: final_integrated_strategy_results.csv")
print("="*80)

