import pandas as pd
import numpy as np

"""
P0 + P1 통합 전략 - 빠른 버전
기존 거래 데이터에 필터만 적용하여 테스트
"""

# 기존 거래 데이터 로드
print("="*100)
print("📈 P0 + P1 통합 전략 백테스트 (빠른 버전)")
print("="*100)

trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print(f"\n기존 거래 데이터: {len(trades_df)}건")

# 기술적 지표 계산
print("기술적 지표 계산 중...")

candles_df['ema_20'] = candles_df['close'].ewm(span=20, adjust=False).mean()
candles_df['ema_50'] = candles_df['close'].ewm(span=50, adjust=False).mean()
candles_df['ema_200'] = candles_df['close'].ewm(span=200, adjust=False).mean()

# RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

candles_df['rsi'] = calculate_rsi(candles_df['close'], 14)
candles_df['change_20'] = ((candles_df['close'] - candles_df['close'].shift(20)) / candles_df['close'].shift(20)) * 100

# MACD
candles_df['macd_fast'] = candles_df['close'].ewm(span=12, adjust=False).mean()
candles_df['macd_slow'] = candles_df['close'].ewm(span=26, adjust=False).mean()
candles_df['macd_line'] = candles_df['macd_fast'] - candles_df['macd_slow']
candles_df['macd_signal'] = candles_df['macd_line'].ewm(span=9, adjust=False).mean()
candles_df['macd_hist'] = candles_df['macd_line'] - candles_df['macd_signal']

print("지표 계산 완료!")

# 각 거래에 지표 추가
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# 매칭을 위한 timestamp 인덱스
candles_df.set_index('datetime', inplace=True)

# 거래별 지표 추출
indicators = []
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    
    # 가장 가까운 캔들 찾기
    try:
        # 정확히 일치하는 시간 찾기
        idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        candle = candles_df.iloc[idx]
        
        indicators.append({
            'rsi': candle['rsi'],
            'ema_20': candle['ema_20'],
            'ema_50': candle['ema_50'],
            'ema_200': candle['ema_200'],
            'close': candle['close'],
            'change_20': candle['change_20'],
            'macd_hist': candle['macd_hist'],
            'above_ema20': candle['close'] > candle['ema_20'],
            'above_ema50': candle['close'] > candle['ema_50'],
            'above_ema200': candle['close'] > candle['ema_200'],
            'ema_aligned': candle['ema_20'] > candle['ema_50'] > candle['ema_200'] if pd.notna(candle['ema_200']) else False
        })
    except:
        indicators.append({
            'rsi': 50, 'ema_20': 0, 'ema_50': 0, 'ema_200': 0,
            'close': 0, 'change_20': 0, 'macd_hist': 0,
            'above_ema20': False, 'above_ema50': False, 'above_ema200': False,
            'ema_aligned': False
        })

indicators_df = pd.DataFrame(indicators)
trades_df = pd.concat([trades_df.reset_index(drop=True), indicators_df], axis=1)

# 필터 조합 테스트
print("\n필터 조합 테스트 중...")

filters = {
    '기본 (필터 없음)': lambda df: df,
    'RSI > 50': lambda df: df[df['rsi'] > 50],
    'RSI > 55': lambda df: df[df['rsi'] > 55],
    'RSI > 60': lambda df: df[df['rsi'] > 60],
    'EMA20 위': lambda df: df[df['above_ema20'] == True],
    'EMA50 위': lambda df: df[df['above_ema50'] == True],
    'EMA200 위': lambda df: df[df['above_ema200'] == True],
    'EMA 정배열': lambda df: df[df['ema_aligned'] == True],
    'RSI>55 + EMA20위': lambda df: df[(df['rsi'] > 55) & (df['above_ema20'] == True)],
    'RSI>55 + EMA50위': lambda df: df[(df['rsi'] > 55) & (df['above_ema50'] == True)],
    'RSI>55 + 모멘텀>0.3%': lambda df: df[(df['rsi'] > 55) & (df['change_20'] > 0.3)],
    'EMA20위 + 모멘텀>0.3%': lambda df: df[(df['above_ema20'] == True) & (df['change_20'] > 0.3)],
    'RSI>55 + EMA정배열': lambda df: df[(df['rsi'] > 55) & (df['ema_aligned'] == True)],
    'RSI>50 + EMA50위 + 모멘텀': lambda df: df[(df['rsi'] > 50) & (df['above_ema50'] == True) & (df['change_20'] > 0)],
    'MACD양수': lambda df: df[df['macd_hist'] > 0],
    'RSI>55 + MACD양수': lambda df: df[(df['rsi'] > 55) & (df['macd_hist'] > 0)],
}

results = []
for name, filter_fn in filters.items():
    filtered = filter_fn(trades_df)
    
    if len(filtered) == 0:
        continue
    
    tp_trades = filtered[filtered['exit_reason'].str.contains('TP')]
    sl_trades = filtered[filtered['exit_reason'].str.contains('SL')]
    
    win_rate = len(tp_trades) / len(filtered) * 100
    total_pnl = filtered['pnl_pct'].sum()
    avg_pnl = filtered['pnl_pct'].mean()
    
    # MDD 계산
    cumulative = filtered['pnl_pct'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    mdd = drawdown.min()
    
    results.append({
        'Filter': name,
        'Trades': len(filtered),
        'Blocked': len(trades_df) - len(filtered),
        'Win Rate': win_rate,
        'SL Ratio': len(sl_trades) / len(filtered) * 100,
        'Total PNL': total_pnl,
        'Avg PNL': avg_pnl,
        'MDD': mdd
    })

# 결과 정렬 (승률 순)
results = sorted(results, key=lambda x: x['Win Rate'], reverse=True)

print("\n" + "="*130)
print("📊 P0 + P1 통합 전략 필터 비교 결과")
print("="*130)

print(f"\n{'필터':<30} {'거래':>7} {'차단':>7} {'승률':>8} {'SL비율':>8} {'총PNL':>10} {'평균PNL':>10} {'MDD':>10}")
print("-"*110)

for r in results:
    flag = "⭐" if r['Win Rate'] > 65 and r['Avg PNL'] > 0.24 else ""
    print(f"{r['Filter']:<30} {r['Trades']:>7} {r['Blocked']:>7} {r['Win Rate']:>7.1f}% {r['SL Ratio']:>7.1f}% {r['Total PNL']:>9.1f}% {r['Avg PNL']:>9.3f}% {r['MDD']:>9.1f}% {flag}")

# 분석
print("\n" + "="*130)
print("📈 분석 요약")
print("="*130)

base = next((r for r in results if '기본' in r['Filter']), None)
best_win = max(results, key=lambda x: x['Win Rate'])
best_pnl = max(results, key=lambda x: x['Total PNL'])
best_avg = max(results, key=lambda x: x['Avg PNL'] if x['Trades'] >= 100 else 0)

print(f"\n기본 전략:")
print(f"  거래: {base['Trades']}건, 승률: {base['Win Rate']:.1f}%, 총PNL: {base['Total PNL']:.1f}%, MDD: {base['MDD']:.1f}%")

print(f"\n최고 승률 필터: {best_win['Filter']}")
print(f"  거래: {best_win['Trades']}건 (-{base['Trades']-best_win['Trades']}), 승률: {best_win['Win Rate']:.1f}% (+{best_win['Win Rate']-base['Win Rate']:.1f}%p)")
print(f"  총PNL: {best_win['Total PNL']:.1f}% ({best_win['Total PNL']-base['Total PNL']:+.1f}%p)")

print(f"\n최고 총PNL 필터: {best_pnl['Filter']}")
print(f"  거래: {best_pnl['Trades']}건, 총PNL: {best_pnl['Total PNL']:.1f}%")

print(f"\n최고 평균PNL 필터 (≥100거래): {best_avg['Filter']}")
print(f"  거래: {best_avg['Trades']}건, 평균PNL: {best_avg['Avg PNL']:.3f}%")

# 트레이드오프 분석
print("\n" + "="*130)
print("⚖️ 트레이드오프 분석")
print("="*130)

for r in results:
    if r['Filter'] == '기본 (필터 없음)':
        continue
    
    win_gain = r['Win Rate'] - base['Win Rate']
    pnl_change = r['Total PNL'] - base['Total PNL']
    trade_loss = base['Trades'] - r['Trades']
    
    if win_gain > 5 and r['Trades'] >= 200:
        print(f"\n✅ {r['Filter']}:")
        print(f"   승률 +{win_gain:.1f}%p, 총PNL {pnl_change:+.1f}%p, 거래 -{trade_loss}건")
        
        if pnl_change >= 0:
            print(f"   → 추천! 승률 향상 + PNL 유지/증가")
        else:
            print(f"   → 승률은 좋지만 총PNL 감소")

# 결론
print("\n" + "="*130)
print("🎯 결론: P0 + P1 통합 가능 여부")
print("="*130)

# 승률 65% 이상이면서 총PNL 손실이 30% 미만인 필터 찾기
good_filters = [r for r in results if r['Win Rate'] >= 65 and (r['Total PNL'] >= base['Total PNL'] * 0.7)]

if good_filters:
    print("\n✅ 예, 둘 다 사용 가능합니다!")
    print("\n추천 필터 조합:")
    for gf in good_filters[:3]:
        print(f"  • {gf['Filter']}: 승률 {gf['Win Rate']:.1f}%, 총PNL {gf['Total PNL']:.1f}%, MDD {gf['MDD']:.1f}%")
else:
    print("\n⚠️ P1 필터 적용 시 주의 필요:")
    print("  - 승률은 상승하지만 총PNL이 크게 감소할 수 있음")
    print("  - 필터 강도 조절 권장")

# 최종 추천
print("\n" + "="*130)
print("💡 최종 추천")
print("="*130)

# 균형잡힌 필터 찾기 (승률 향상 + PNL 손실 최소화)
balanced = sorted([r for r in results if r['Win Rate'] > base['Win Rate']], 
                  key=lambda x: (x['Win Rate'] - base['Win Rate']) + (x['Total PNL'] - base['Total PNL']) / 10,
                  reverse=True)

if balanced:
    rec = balanced[0]
    print(f"\n균형 잡힌 필터: {rec['Filter']}")
    print(f"  • 거래: {rec['Trades']}건 (기존 {base['Trades']}건)")
    print(f"  • 승률: {rec['Win Rate']:.1f}% (기존 {base['Win Rate']:.1f}%, +{rec['Win Rate']-base['Win Rate']:.1f}%p)")
    print(f"  • 총PNL: {rec['Total PNL']:.1f}% (기존 {base['Total PNL']:.1f}%, {rec['Total PNL']-base['Total PNL']:+.1f}%p)")
    print(f"  • MDD: {rec['MDD']:.1f}% (기존 {base['MDD']:.1f}%)")

# 저장
comparison_df = pd.DataFrame(results)
comparison_df.to_csv('backtest_p0_p1_comparison.csv', index=False)
print(f"\n✅ 결과 저장: backtest_p0_p1_comparison.csv")
