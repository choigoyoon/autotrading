import pandas as pd
import numpy as np

"""
위험 진입 자리 필터링 후 매매횟수 및 성과 비교
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("="*100)
print("📊 진입 자리 필터링 - 매매횟수 비교")
print("="*100)

# 기술적 지표 계산
candles_df['ema_20'] = candles_df['close'].ewm(span=20, adjust=False).mean()
candles_df['ema_50'] = candles_df['close'].ewm(span=50, adjust=False).mean()
candles_df['ema_200'] = candles_df['close'].ewm(span=200, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

candles_df['rsi'] = calculate_rsi(candles_df['close'], 14)
candles_df['change_4h'] = ((candles_df['close'] - candles_df['close'].shift(16)) / candles_df['close'].shift(16)) * 100
candles_df['ema20_slope'] = (candles_df['ema_20'] - candles_df['ema_20'].shift(4)) / candles_df['ema_20'].shift(4) * 100

candles_df.set_index('datetime', inplace=True)

# 각 거래에 지표 추가
indicators = []
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    try:
        idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        candle = candles_df.iloc[idx]
        
        indicators.append({
            'rsi': candle['rsi'],
            'change_4h': candle['change_4h'],
            'ema20_slope': candle['ema20_slope'],
            'above_ema20': candle['close'] > candle['ema_20'],
            'above_ema200': candle['close'] > candle['ema_200'],
            'dist_ema200': ((candle['close'] - candle['ema_200']) / candle['ema_200']) * 100 if candle['ema_200'] > 0 else 0
        })
    except:
        indicators.append({
            'rsi': 50, 'change_4h': 0, 'ema20_slope': 0,
            'above_ema20': True, 'above_ema200': True, 'dist_ema200': 0
        })

indicators_df = pd.DataFrame(indicators)
df = pd.concat([trades_df.reset_index(drop=True), indicators_df], axis=1)

# MDD 계산 함수
def calc_metrics(data):
    if len(data) == 0:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'mdd': 0}
    
    tp = len(data[data['exit_reason'].str.contains('TP')])
    win_rate = tp / len(data) * 100
    total_pnl = data['pnl_pct'].sum()
    avg_pnl = data['pnl_pct'].mean()
    
    # MDD
    cum = data['pnl_pct'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    mdd = dd.min()
    
    return {
        'trades': len(data),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'mdd': mdd
    }

# 기본 성과
base = calc_metrics(df)

print(f"\n📈 기본 전략 (필터 없음)")
print(f"   거래: {base['trades']}건, 승률: {base['win_rate']:.1f}%")
print(f"   총PNL: {base['total_pnl']:.1f}%, MDD: {base['mdd']:.1f}%")

# 다양한 필터 조합 테스트
print("\n" + "="*100)
print("📊 진입 자리 필터별 매매횟수 비교")
print("="*100)

filters = {
    # 위험 자리 회피 필터 (발견된 문제 패턴 기반)
    '4시간 -2%~-1% 하락 회피': lambda d: d[~((d['change_4h'] >= -2) & (d['change_4h'] < -1))],
    '4시간 -1% 이상 하락 회피': lambda d: d[d['change_4h'] >= -1],
    'EMA20 기울기 하락 회피 (<-0.2%)': lambda d: d[d['ema20_slope'] >= -0.2],
    'EMA200 근처 회피 (거리 0~2%)': lambda d: d[~((d['dist_ema200'] >= 0) & (d['dist_ema200'] < 2))],
    
    # 안전 조건 추가
    'RSI > 50': lambda d: d[d['rsi'] > 50],
    'RSI > 55': lambda d: d[d['rsi'] > 55],
    '4시간 상승 중 (>0%)': lambda d: d[d['change_4h'] > 0],
    'EMA20 위': lambda d: d[d['above_ema20']],
    
    # 복합 필터
    '4시간 하락 회피 + RSI>50': lambda d: d[(d['change_4h'] >= -1) & (d['rsi'] > 50)],
    '4시간 하락 회피 + EMA20위': lambda d: d[(d['change_4h'] >= -1) & (d['above_ema20'])],
    '약한 하락 회피 + RSI>50': lambda d: d[~((d['change_4h'] >= -2) & (d['change_4h'] < -1)) & (d['rsi'] > 50)],
    
    # 가장 위험한 조합만 회피
    '손실률 57% 구간만 회피': lambda d: d[~((d['change_4h'] >= -2) & (d['change_4h'] < -1))],
}

results = []
for name, filter_fn in filters.items():
    filtered = filter_fn(df.copy()).reset_index(drop=True)
    metrics = calc_metrics(filtered)
    
    results.append({
        'filter': name,
        'trades': metrics['trades'],
        'blocked': base['trades'] - metrics['trades'],
        'block_pct': (base['trades'] - metrics['trades']) / base['trades'] * 100,
        'win_rate': metrics['win_rate'],
        'total_pnl': metrics['total_pnl'],
        'avg_pnl': metrics['avg_pnl'],
        'mdd': metrics['mdd'],
        'mdd_change': metrics['mdd'] - base['mdd']
    })

results_df = pd.DataFrame(results).sort_values('mdd', ascending=False)

print(f"\n{'필터':<35} {'거래':>6} {'차단':>6} {'차단%':>7} {'승률':>7} {'총PNL':>8} {'MDD':>8} {'MDD변화':>8}")
print("-"*105)

for _, r in results_df.iterrows():
    mdd_flag = "✅" if r['mdd_change'] > 2 else "⚠️" if r['mdd_change'] > 0 else "❌"
    print(f"{r['filter']:<35} {r['trades']:>6} {r['blocked']:>6} {r['block_pct']:>6.1f}% {r['win_rate']:>6.1f}% {r['total_pnl']:>7.1f}% {r['mdd']:>7.1f}% {r['mdd_change']:>+7.1f}% {mdd_flag}")

# 최적 필터 찾기
print("\n" + "="*100)
print("🏆 최적 필터 분석")
print("="*100)

# MDD 개선된 필터 중 거래수 가장 많은 것
mdd_improved = [r for r in results if r['mdd_change'] > 0]
if mdd_improved:
    best_mdd = max(mdd_improved, key=lambda x: x['trades'])
    print(f"\n✅ MDD 개선 + 거래수 최대: {best_mdd['filter']}")
    print(f"   거래: {best_mdd['trades']}건 (-{best_mdd['blocked']}건, -{best_mdd['block_pct']:.1f}%)")
    print(f"   승률: {base['win_rate']:.1f}% → {best_mdd['win_rate']:.1f}% ({best_mdd['win_rate']-base['win_rate']:+.1f}%p)")
    print(f"   MDD: {base['mdd']:.1f}% → {best_mdd['mdd']:.1f}% ({best_mdd['mdd_change']:+.1f}%p 개선)")

# 승률 가장 높은 것
best_winrate = max(results, key=lambda x: x['win_rate'])
print(f"\n⭐ 최고 승률: {best_winrate['filter']}")
print(f"   거래: {best_winrate['trades']}건 (-{best_winrate['blocked']}건)")
print(f"   승률: {best_winrate['win_rate']:.1f}%")

# 총 PNL 가장 높은 것 (필터 적용 후)
best_pnl = max(results, key=lambda x: x['total_pnl'])
print(f"\n💰 최고 총PNL: {best_pnl['filter']}")
print(f"   거래: {best_pnl['trades']}건, 총PNL: {best_pnl['total_pnl']:.1f}%")

# 요약 테이블
print("\n" + "="*100)
print("📋 요약: 필터별 거래 횟수 변화")
print("="*100)

print(f"\n{'조건':<40} {'기존 거래':>10} {'필터 후':>10} {'감소':>10}")
print("-"*75)
print(f"{'기본 (필터 없음)':<40} {base['trades']:>10} {base['trades']:>10} {0:>10}")

for r in sorted(results, key=lambda x: x['trades'], reverse=True)[:8]:
    print(f"{r['filter']:<40} {base['trades']:>10} {r['trades']:>10} {r['blocked']:>10} ({r['block_pct']:.1f}%)")

print("\n" + "="*100)
print("🎯 결론")
print("="*100)

# 가장 효율적인 필터 (차단 적고 MDD 개선)
efficient = sorted([r for r in results if r['mdd_change'] > 0], 
                   key=lambda x: x['mdd_change'] / (x['block_pct'] + 0.1), reverse=True)

if efficient:
    best = efficient[0]
    print(f"\n💡 추천 필터: {best['filter']}")
    print(f"\n   📊 매매횟수 변화:")
    print(f"      기존: {base['trades']}건")
    print(f"      필터 후: {best['trades']}건")
    print(f"      차이: -{best['blocked']}건 (-{best['block_pct']:.1f}%)")
    print(f"\n   📈 성과 변화:")
    print(f"      승률: {base['win_rate']:.1f}% → {best['win_rate']:.1f}% ({best['win_rate']-base['win_rate']:+.1f}%p)")
    print(f"      총PNL: {base['total_pnl']:.1f}% → {best['total_pnl']:.1f}% ({best['total_pnl']-base['total_pnl']:+.1f}%p)")
    print(f"      MDD: {base['mdd']:.1f}% → {best['mdd']:.1f}% ({best['mdd_change']:+.1f}%p 개선)")
else:
    print("\n⚠️ MDD 개선 필터 없음 - 다른 접근 필요")
