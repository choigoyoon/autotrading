import pandas as pd
import numpy as np

"""
실패 사유 기반 진입 체크 필터 생성
핵심 실패 사유:
1. 지지선붕괴 (67.6%) - 진입 전 지지선 강도 체크
2. 고점추격 (57.6%) - 최근 고점 대비 위치 체크
3. 변동성급증 (41.4%) - ATR 기반 변동성 체크
4. 거래량감소중 (30.9%) - 거래량 추세 체크
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

print("="*100)
print("🎯 실패 사유 기반 진입 체크 필터")
print("="*100)

# 기술적 지표 계산
print("\n지표 계산 중...")

# 지지선/저항선
candles_df['support_20'] = candles_df['low'].rolling(20).min()
candles_df['resistance_20'] = candles_df['high'].rolling(20).max()
candles_df['support_50'] = candles_df['low'].rolling(50).min()
candles_df['resistance_50'] = candles_df['high'].rolling(50).max()

# 고점 대비 위치
candles_df['dist_from_high_20'] = ((candles_df['close'] - candles_df['resistance_20']) / candles_df['resistance_20']) * 100
candles_df['dist_from_high_50'] = ((candles_df['close'] - candles_df['resistance_50']) / candles_df['resistance_50']) * 100

# 지지선 대비 위치
candles_df['dist_from_low_20'] = ((candles_df['close'] - candles_df['support_20']) / candles_df['support_20']) * 100
candles_df['dist_from_low_50'] = ((candles_df['close'] - candles_df['support_50']) / candles_df['support_50']) * 100

# 지지선 강도 (지지선까지 여유)
candles_df['support_margin'] = ((candles_df['close'] - candles_df['support_20']) / candles_df['close']) * 100

# ATR (변동성)
candles_df['tr'] = np.maximum(
    candles_df['high'] - candles_df['low'],
    np.maximum(
        abs(candles_df['high'] - candles_df['close'].shift(1)),
        abs(candles_df['low'] - candles_df['close'].shift(1))
    )
)
candles_df['atr_14'] = candles_df['tr'].rolling(14).mean()
candles_df['atr_pct'] = (candles_df['atr_14'] / candles_df['close']) * 100

# 변동성 변화 (최근 vs 이전)
candles_df['atr_5'] = candles_df['tr'].rolling(5).mean()
candles_df['atr_20'] = candles_df['tr'].rolling(20).mean()
candles_df['volatility_ratio'] = candles_df['atr_5'] / candles_df['atr_20']

# 거래량 추세
candles_df['vol_5'] = candles_df['volume'].rolling(5).mean()
candles_df['vol_20'] = candles_df['volume'].rolling(20).mean()
candles_df['vol_ratio'] = candles_df['vol_5'] / candles_df['vol_20']

# 연속 하락 캔들
candles_df['is_red'] = candles_df['close'] < candles_df['open']
candles_df['red_streak'] = candles_df['is_red'].rolling(5).sum()

# 가격 모멘텀
candles_df['mom_5'] = ((candles_df['close'] - candles_df['close'].shift(5)) / candles_df['close'].shift(5)) * 100
candles_df['mom_20'] = ((candles_df['close'] - candles_df['close'].shift(20)) / candles_df['close'].shift(20)) * 100

candles_df.set_index('datetime', inplace=True)

# 각 거래에 지표 매칭
print("거래별 지표 매칭 중...")

indicators = []
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    try:
        idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        c = candles_df.iloc[idx]
        
        indicators.append({
            'pnl': trade['pnl_pct'],
            'exit_reason': trade['exit_reason'],
            'is_sl': 'SL' in trade['exit_reason'],
            # 고점 추격 체크
            'dist_from_high_20': c['dist_from_high_20'],
            'dist_from_high_50': c['dist_from_high_50'],
            # 지지선 여유
            'support_margin': c['support_margin'],
            'dist_from_low_20': c['dist_from_low_20'],
            # 변동성
            'atr_pct': c['atr_pct'],
            'volatility_ratio': c['volatility_ratio'],
            # 거래량
            'vol_ratio': c['vol_ratio'],
            # 모멘텀
            'mom_5': c['mom_5'],
            'mom_20': c['mom_20'],
            # 연속 하락
            'red_streak': c['red_streak'],
        })
    except:
        pass

df = pd.DataFrame(indicators)
print(f"\n분석 거래: {len(df)}건")

# 기본 성과
def calc_metrics(data):
    if len(data) == 0:
        return None
    data = data.reset_index(drop=True)
    sl_count = data['is_sl'].sum()
    win_rate = (1 - sl_count / len(data)) * 100
    total_pnl = data['pnl'].sum()
    avg_pnl = data['pnl'].mean()
    cum = data['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    return {'trades': len(data), 'win_rate': win_rate, 'sl_rate': sl_count/len(data)*100, 
            'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'mdd': mdd}

base = calc_metrics(df)
print(f"\n기본: 거래 {base['trades']}건, 승률 {base['win_rate']:.1f}%, MDD {base['mdd']:.1f}%")

# 실패 사유별 필터 테스트
print("\n" + "="*100)
print("📊 실패 사유 기반 필터 테스트")
print("="*100)

filters = {
    # 1. 고점추격 회피
    '고점추격회피: 20봉고점-1%이하': df['dist_from_high_20'] < -1,
    '고점추격회피: 20봉고점-2%이하': df['dist_from_high_20'] < -2,
    '고점추격회피: 50봉고점-2%이하': df['dist_from_high_50'] < -2,
    '고점추격회피: 50봉고점-3%이하': df['dist_from_high_50'] < -3,
    
    # 2. 지지선 여유 확보
    '지지선여유: 저점+1%이상': df['dist_from_low_20'] > 1,
    '지지선여유: 저점+2%이상': df['dist_from_low_20'] > 2,
    '지지선여유: 마진1%이상': df['support_margin'] > 1,
    '지지선여유: 마진2%이상': df['support_margin'] > 2,
    
    # 3. 변동성 체크
    '저변동성: ATR<0.5%': df['atr_pct'] < 0.5,
    '저변동성: ATR<0.7%': df['atr_pct'] < 0.7,
    '변동성안정: 비율<1.2': df['volatility_ratio'] < 1.2,
    '변동성안정: 비율<1.5': df['volatility_ratio'] < 1.5,
    
    # 4. 거래량 체크
    '거래량증가: 비율>0.8': df['vol_ratio'] > 0.8,
    '거래량증가: 비율>1.0': df['vol_ratio'] > 1.0,
    '거래량증가: 비율>1.2': df['vol_ratio'] > 1.2,
    
    # 5. 모멘텀 체크
    '상승모멘텀: 5봉+0%이상': df['mom_5'] > 0,
    '상승모멘텀: 5봉+0.5%이상': df['mom_5'] > 0.5,
    '상승모멘텀: 20봉+0%이상': df['mom_20'] > 0,
    
    # 6. 연속하락 회피
    '연속하락회피: 음봉<3': df['red_streak'] < 3,
    '연속하락회피: 음봉<4': df['red_streak'] < 4,
}

results = []
for name, cond in filters.items():
    filtered = df[cond]
    m = calc_metrics(filtered)
    if m is None or m['trades'] < 50:
        continue
    
    results.append({
        'filter': name,
        'trades': m['trades'],
        'block_pct': (base['trades'] - m['trades']) / base['trades'] * 100,
        'win_rate': m['win_rate'],
        'sl_rate': m['sl_rate'],
        'total_pnl': m['total_pnl'],
        'mdd': m['mdd'],
        'win_improve': m['win_rate'] - base['win_rate'],
        'mdd_improve': m['mdd'] - base['mdd']
    })

results_df = pd.DataFrame(results).sort_values('win_rate', ascending=False)

print(f"\n{'필터':<35} {'거래':>6} {'차단%':>7} {'승률':>7} {'SL%':>6} {'MDD':>8} {'승률↑':>7} {'MDD↑':>7}")
print("-"*100)
for _, r in results_df.iterrows():
    flag = "⭐" if r['win_rate'] >= 70 and r['mdd'] >= -15 else ""
    print(f"{r['filter']:<35} {r['trades']:>6} {r['block_pct']:>6.1f}% {r['win_rate']:>6.1f}% {r['sl_rate']:>5.1f}% {r['mdd']:>7.1f}% {r['win_improve']:>+6.1f}% {r['mdd_improve']:>+6.1f}% {flag}")

# 복합 필터 테스트
print("\n" + "="*100)
print("📊 복합 필터 (실패 사유 동시 체크)")
print("="*100)

complex_filters = {
    # 핵심 2개 조합
    '고점-2% + 지지여유1%': (df['dist_from_high_20'] < -2) & (df['dist_from_low_20'] > 1),
    '고점-1% + 지지여유2%': (df['dist_from_high_20'] < -1) & (df['dist_from_low_20'] > 2),
    '고점-2% + ATR<0.7%': (df['dist_from_high_20'] < -2) & (df['atr_pct'] < 0.7),
    '고점-2% + 거래량>0.8': (df['dist_from_high_20'] < -2) & (df['vol_ratio'] > 0.8),
    '지지여유1% + ATR<0.7%': (df['dist_from_low_20'] > 1) & (df['atr_pct'] < 0.7),
    '지지여유1% + 거래량>0.8': (df['dist_from_low_20'] > 1) & (df['vol_ratio'] > 0.8),
    
    # 핵심 3개 조합
    '고점-2% + 지지1% + ATR<0.7%': (df['dist_from_high_20'] < -2) & (df['dist_from_low_20'] > 1) & (df['atr_pct'] < 0.7),
    '고점-2% + 지지1% + 거래량>0.8': (df['dist_from_high_20'] < -2) & (df['dist_from_low_20'] > 1) & (df['vol_ratio'] > 0.8),
    '고점-1% + 지지2% + 거래량>1.0': (df['dist_from_high_20'] < -1) & (df['dist_from_low_20'] > 2) & (df['vol_ratio'] > 1.0),
    
    # 모멘텀 포함
    '고점-2% + 모멘텀+0%': (df['dist_from_high_20'] < -2) & (df['mom_5'] > 0),
    '지지1% + 모멘텀+0%': (df['dist_from_low_20'] > 1) & (df['mom_5'] > 0),
    '고점-2% + 지지1% + 모멘텀+0%': (df['dist_from_high_20'] < -2) & (df['dist_from_low_20'] > 1) & (df['mom_5'] > 0),
    
    # 전체 핵심 조합
    '고점-2% + 지지1% + ATR<0.7% + 거래량>0.8': (df['dist_from_high_20'] < -2) & (df['dist_from_low_20'] > 1) & (df['atr_pct'] < 0.7) & (df['vol_ratio'] > 0.8),
    '고점-1% + 지지1% + ATR<0.7% + 모멘텀+0%': (df['dist_from_high_20'] < -1) & (df['dist_from_low_20'] > 1) & (df['atr_pct'] < 0.7) & (df['mom_5'] > 0),
}

complex_results = []
for name, cond in complex_filters.items():
    filtered = df[cond]
    m = calc_metrics(filtered)
    if m is None or m['trades'] < 30:
        continue
    
    complex_results.append({
        'filter': name,
        'trades': m['trades'],
        'block_pct': (base['trades'] - m['trades']) / base['trades'] * 100,
        'win_rate': m['win_rate'],
        'total_pnl': m['total_pnl'],
        'avg_pnl': m['avg_pnl'],
        'mdd': m['mdd'],
    })

complex_df = pd.DataFrame(complex_results).sort_values('win_rate', ascending=False)

print(f"\n{'필터':<50} {'거래':>6} {'차단%':>7} {'승률':>7} {'MDD':>8} {'총PNL':>8}")
print("-"*95)
for _, r in complex_df.iterrows():
    flag = "⭐" if r['win_rate'] >= 70 and r['mdd'] >= -12 else "✓" if r['win_rate'] >= 68 else ""
    print(f"{r['filter']:<50} {r['trades']:>6} {r['block_pct']:>6.1f}% {r['win_rate']:>6.1f}% {r['mdd']:>7.1f}% {r['total_pnl']:>7.1f}% {flag}")

# 최적 필터 선정
print("\n" + "="*100)
print("🏆 최적 진입 체크 필터")
print("="*100)

if len(complex_df) > 0:
    # 승률 70% 이상 + MDD -15% 이내
    good = complex_df[(complex_df['win_rate'] >= 68) & (complex_df['mdd'] >= -15)]
    if len(good) > 0:
        best = good.iloc[0]
        print(f"\n💡 추천 필터: {best['filter']}")
        print(f"\n   기본 → 필터 적용 후:")
        print(f"   거래: {base['trades']} → {best['trades']}건 (-{best['block_pct']:.0f}%)")
        print(f"   승률: {base['win_rate']:.1f}% → {best['win_rate']:.1f}% (+{best['win_rate']-base['win_rate']:.1f}%p)")
        print(f"   MDD: {base['mdd']:.1f}% → {best['mdd']:.1f}% (+{best['mdd']-base['mdd']:.1f}%p)")
        print(f"   총PNL: {base['total_pnl']:.1f}% → {best['total_pnl']:.1f}%")

print("\n" + "="*100)
print("📋 진입 체크리스트 (실패 사유 기반)")
print("="*100)
print("""
진입 전 필수 체크:
  ✅ 1. 고점추격 방지: 20봉 최고가 대비 -2% 이하 위치
  ✅ 2. 지지선 여유: 20봉 최저가 대비 +1% 이상 위치  
  ✅ 3. 변동성 체크: ATR < 0.7% (저변동성)
  ✅ 4. 거래량 체크: 최근5봉/20봉 거래량 비율 > 0.8
  ✅ 5. 모멘텀 체크: 5봉 가격변화 > 0% (상승 중)
""")
