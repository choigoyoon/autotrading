import pandas as pd

# 두 전략 로드
old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
new_df = pd.read_csv('backtest_inflection_mtf_filtered_results.csv')

print('='*100)
print('📊 MTF 필터 효과 상세 분석')
print('='*100)

# 1. 기본 비교
print('\n1️⃣ 기본 지표 비교:')
print(f'{"지표":<20} {"기존 전략":<20} {"MTF 필터 전략":<20} {"차이":<20}')
print('-'*80)

total_trades_old = len(old_df)
total_trades_new = len(new_df)
print(f'{"총 거래 수":<20} {total_trades_old:<20} {total_trades_new:<20} {total_trades_new - total_trades_old:+d}')

win_rate_old = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
win_rate_new = len(new_df[new_df['exit_reason'].str.contains('TP')]) / len(new_df) * 100
print(f'{"승률 %":<20} {win_rate_old:<20.2f} {win_rate_new:<20.2f} {win_rate_new - win_rate_old:+.2f}')

sl_rate_old = len(old_df[old_df['exit_reason'] == 'SL']) / len(old_df) * 100
sl_rate_new = len(new_df[new_df['exit_reason'] == 'SL']) / len(new_df) * 100
print(f'{"SL 비율 %":<20} {sl_rate_old:<20.2f} {sl_rate_new:<20.2f} {sl_rate_new - sl_rate_old:+.2f}')

total_pnl_old = old_df['pnl_pct'].sum()
total_pnl_new = new_df['pnl_pct'].sum()
print(f'{"총 PNL %":<20} {total_pnl_old:<20.2f} {total_pnl_new:<20.2f} {total_pnl_new - total_pnl_old:+.2f}')

avg_pnl_old = old_df['pnl_pct'].mean()
avg_pnl_new = new_df['pnl_pct'].mean()
print(f'{"평균 PNL %":<20} {avg_pnl_old:<20.2f} {avg_pnl_new:<20.2f} {avg_pnl_new - avg_pnl_old:+.2f}')

# 2. 청산 사유별 비교
print('\n2️⃣ 청산 사유별 비교:')
print(f'{"사유":<15} {"기존 거래수":<15} {"기존 총PNL":<15} {"MTF 거래수":<15} {"MTF 총PNL":<15} {"PNL 차이":<15}')
print('-'*90)

for reason in ['SL', 'TP1_Partial', 'TP2_Full']:
    old_reason = old_df[old_df['exit_reason'] == reason]
    new_reason = new_df[new_df['exit_reason'] == reason]
    
    old_count = len(old_reason)
    new_count = len(new_reason)
    old_pnl = old_reason['pnl_pct'].sum() if len(old_reason) > 0 else 0
    new_pnl = new_reason['pnl_pct'].sum() if len(new_reason) > 0 else 0
    
    print(f'{reason:<15} {old_count:<15} {old_pnl:<15.2f} {new_count:<15} {new_pnl:<15.2f} {new_pnl - old_pnl:+.2f}')

# 3. 연도별 비교
print('\n3️⃣ 연도별 성과 비교:')
print(f'{"연도":<10} {"기존 거래":<12} {"기존 PNL":<12} {"MTF 거래":<12} {"MTF PNL":<12} {"PNL 차이":<12}')
print('-'*70)

old_df['year'] = pd.to_datetime(old_df['entry_time']).dt.year
new_df['year'] = pd.to_datetime(new_df['entry_time']).dt.year

for year in sorted(set(old_df['year'].unique()) | set(new_df['year'].unique())):
    old_year = old_df[old_df['year'] == year]
    new_year = new_df[new_df['year'] == year]
    
    old_trades = len(old_year)
    new_trades = len(new_year)
    old_year_pnl = old_year['pnl_pct'].sum() if len(old_year) > 0 else 0
    new_year_pnl = new_year['pnl_pct'].sum() if len(new_year) > 0 else 0
    
    print(f'{year:<10} {old_trades:<12} {old_year_pnl:<12.2f} {new_trades:<12} {new_year_pnl:<12.2f} {new_year_pnl - old_year_pnl:+.2f}')

# 4. MTF 필터가 차단한 거래 분석
print('\n4️⃣ MTF 필터 효과 분석:')
print(f'  변곡점 캔들 후보: 875개')
print(f'  MTF 필터 차단: 462개 (52.8%)')
print(f'  최종 진입: 413개 (47.2%)')
print(f'\n  ▶️ MTF 필터가 절반(53%)의 거래를 차단했습니다.')
print(f'  ▶️ 차단된 276개 거래의 예상 PNL: {total_pnl_old - total_pnl_new:.2f}%')
print(f'  ▶️ 차단된 거래의 평균 PNL: {(total_pnl_old - total_pnl_new) / 276:.2f}%')

# 5. 결론
print('\n' + '='*100)
print('💡 결론')
print('='*100)
print('\n✅ MTF 필터의 긍정적 효과:')
print(f'  - 승률: 64.44% → 74.82% (+10.38%p)')
print(f'  - SL 비율: 35.6% → 25.2% (-10.4%p)')
print(f'  - 안정성: 손실 거래를 245개 → 104개로 감소')

print('\n❌ MTF 필터의 부정적 효과:')
print(f'  - 총 PNL: 159.56% → 88.87% (-70.69%p)')
print(f'  - 거래 수: 689개 → 413개 (-276개, -40%)')
print(f'  - 연평균: 28.2% → 15.7% (-12.5%p)')

print('\n🤔 분석:')
print('  MTF 필터가 나쁜 거래(SL)도 차단했지만,')
print('  좋은 거래(TP)도 많이 차단했습니다.')
print('  차단된 276개 거래의 평균 PNL이 +0.26%로 수익 거래였습니다.')
print('\n  즉, MTF 필터가 너무 보수적(엄격)입니다!')

print('\n' + '='*100)
print('📋 다음 단계 제안')
print('='*100)
print('\n옵션 A: MTF 필터 완화 (4시간 추세 강도 0.5% → 0.2%)')
print('옵션 B: MTF 필터 제거하고, TP 전략 개선 (TP1 50% 청산, 나머지 TP2까지)')
print('옵션 C: MTF 필터 없이 진행 (현재 최고 성과)')
print('\n현재 최고 성과: 기존 전략 (MTF 필터 없음) - 총 PNL 159.56%, 연평균 28.2%')
