import pandas as pd
import numpy as np

"""
빠른 필터 테스트 - 기존 결과 기반으로 분석
"""

# 결과 로드
trades = pd.read_csv('backtest_ema_sl_results.csv')
candles = pd.read_csv('btc_15m_ohlcv.csv')
candles['datetime'] = pd.to_datetime(candles['datetime'])

# 지표 계산
candles['ema_200'] = candles['close'].ewm(span=200, adjust=False).mean()
candles['ema_50'] = candles['close'].ewm(span=50, adjust=False).mean()
candles['ema_20'] = candles['close'].ewm(span=20, adjust=False).mean()

delta = candles['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
candles['rsi'] = 100 - (100 / (1 + gain / loss))
candles['change_20'] = candles['close'].pct_change(20) * 100

print("="*80)
print("📊 필터 조합별 성과 분석 (기존 거래 기반)")
print("="*80)

trades['entry_time'] = pd.to_datetime(trades['entry_time'])

# 각 거래에 지표 추가
def add_indicators(row):
    idx = candles[candles['datetime'] <= row['entry_time']].index
    if len(idx) == 0:
        return pd.Series({})
    i = idx[-1]
    c = candles.loc[i]
    return pd.Series({
        'above_ema20': c['close'] > c['ema_20'],
        'above_ema50': c['close'] > c['ema_50'],
        'above_ema200': c['close'] > c['ema_200'],
        'rsi': c['rsi'],
        'change_20': c['change_20']
    })

print("\n지표 추가 중...")
indicators = trades.apply(add_indicators, axis=1)
trades = pd.concat([trades, indicators], axis=1)

# 필터 조합 테스트
filters = [
    ('기본 (필터 없음)', lambda df: df),
    ('EMA 20 위', lambda df: df[df['above_ema20'] == True]),
    ('EMA 50 위', lambda df: df[df['above_ema50'] == True]),
    ('EMA 20+50 위', lambda df: df[(df['above_ema20'] == True) & (df['above_ema50'] == True)]),
    ('RSI 50+', lambda df: df[df['rsi'] >= 50]),
    ('RSI 55+', lambda df: df[df['rsi'] >= 55]),
    ('RSI 60+', lambda df: df[df['rsi'] >= 60]),
    ('상승 0.3%+', lambda df: df[df['change_20'] >= 0.3]),
    ('상승 0.5%+', lambda df: df[df['change_20'] >= 0.5]),
    ('EMA20 + RSI55', lambda df: df[(df['above_ema20'] == True) & (df['rsi'] >= 55)]),
    ('EMA50 + RSI55', lambda df: df[(df['above_ema50'] == True) & (df['rsi'] >= 55)]),
    ('EMA20 + RSI50 + 상승0.3%', lambda df: df[(df['above_ema20'] == True) & (df['rsi'] >= 50) & (df['change_20'] >= 0.3)]),
    ('EMA20 + RSI55 + 상승0.3%', lambda df: df[(df['above_ema20'] == True) & (df['rsi'] >= 55) & (df['change_20'] >= 0.3)]),
]

print("\n" + "="*100)
print(f"{'필터':<30} {'거래수':<8} {'승률':<10} {'총PNL':<12} {'평균PNL':<10} {'개선':<10}")
print("-"*80)

base_winrate = None
base_pnl = None

results = []
for name, filter_fn in filters:
    filtered = filter_fn(trades)
    
    if len(filtered) > 0:
        wins = len(filtered[filtered['exit_reason'].str.contains('TP')])
        total = len(filtered)
        win_rate = wins / total * 100
        total_pnl = filtered['pnl_pct'].sum()
        avg_pnl = filtered['pnl_pct'].mean()
        
        if base_winrate is None:
            base_winrate = win_rate
            base_pnl = total_pnl
            improve = "-"
        else:
            improve = f"+{win_rate - base_winrate:.1f}%p"
        
        results.append({
            'name': name,
            'trades': total,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        })
        
        marker = "⭐" if win_rate > base_winrate + 5 else ""
        print(f"{name:<30} {total:<8} {win_rate:<10.1f}% {total_pnl:<12.2f}% {avg_pnl:<10.3f}% {improve:<10} {marker}")

# 최고 성과
print("\n" + "="*80)
print("🏆 최고 성과 필터")
print("="*80)

best_winrate = max(results, key=lambda x: x['win_rate'])
best_efficiency = max(results, key=lambda x: x['avg_pnl'] if x['trades'] >= 100 else -999)

print(f"\n✅ 최고 승률: {best_winrate['name']}")
print(f"   승률 {best_winrate['win_rate']:.1f}%, 거래 {best_winrate['trades']}건, 총PNL {best_winrate['total_pnl']:.2f}%")

print(f"\n✅ 최고 효율 (100건+): {best_efficiency['name']}")
print(f"   평균PNL {best_efficiency['avg_pnl']:.3f}%, 거래 {best_efficiency['trades']}건, 승률 {best_efficiency['win_rate']:.1f}%")

# 최적 필터 시뮬레이션
print("\n" + "="*80)
print("📈 최적 필터 예상 성과")
print("="*80)

# 최고 승률 필터로 예상
best_filter = trades[(trades['above_ema20'] == True) & (trades['rsi'] >= 55)]
if len(best_filter) > 0:
    wins = len(best_filter[best_filter['exit_reason'].str.contains('TP')])
    losses = len(best_filter[best_filter['exit_reason'].str.contains('SL')])
    win_rate = wins / len(best_filter) * 100
    
    # MDD 계산
    cumsum = best_filter['pnl_pct'].cumsum()
    drawdown = cumsum - cumsum.cummax()
    mdd = drawdown.min()
    
    print(f"\n필터: EMA20 + RSI55")
    print(f"  거래 수: {len(best_filter)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  총 PNL: {best_filter['pnl_pct'].sum():.2f}%")
    print(f"  평균 PNL: {best_filter['pnl_pct'].mean():.3f}%")
    print(f"  MDD: {mdd:.2f}%")
    
    # 연도별
    best_filter['year'] = pd.to_datetime(best_filter['entry_time']).dt.year
    print("\n  연도별:")
    for year in sorted(best_filter['year'].unique()):
        yt = best_filter[best_filter['year'] == year]
        yr_win = len(yt[yt['exit_reason'].str.contains('TP')]) / len(yt) * 100
        print(f"    {year}: {len(yt)}건, PNL {yt['pnl_pct'].sum():+.2f}%, 승률 {yr_win:.1f}%")
