import pandas as pd
import numpy as np

"""
P0 + P1 통합 전략 백테스트

P0 (Inflection Point 전략):
- HL 발생 → 변곡점 캔들 확인 → 진입

P1 (RSI/MACD/Volume 최적화):
- RSI > 55 (상승 모멘텀)
- EMA 정배열 (EMA20 > EMA50 > EMA200)
- 상승 모멘텀 (20봉 가격 변화 > 0.5%)
- Volume 확인
"""

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("📈 P0 + P1 통합 전략 백테스트")
print("="*100)
print("P0: HL + 변곡점 캔들 진입")
print("P1: RSI/EMA/Momentum 필터")
print("="*100)

# 기술적 지표 계산
print("\n기술적 지표 계산 중...")

# EMA
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

# MACD
candles_df['macd_fast'] = candles_df['close'].ewm(span=12, adjust=False).mean()
candles_df['macd_slow'] = candles_df['close'].ewm(span=26, adjust=False).mean()
candles_df['macd_line'] = candles_df['macd_fast'] - candles_df['macd_slow']
candles_df['macd_signal'] = candles_df['macd_line'].ewm(span=9, adjust=False).mean()
candles_df['macd_hist'] = candles_df['macd_line'] - candles_df['macd_signal']

# 가격 변화 (20봉)
candles_df['change_20'] = ((candles_df['close'] - candles_df['close'].shift(20)) / candles_df['close'].shift(20)) * 100

# 필터 조합 설정
FILTER_CONFIGS = {
    '기본 (필터 없음)': {
        'enabled': False
    },
    'P1-Light (RSI>55)': {
        'enabled': True,
        'rsi_min': 55,
        'ema_alignment': False,
        'momentum': False
    },
    'P1-Medium (RSI>55 + EMA20위)': {
        'enabled': True,
        'rsi_min': 55,
        'above_ema20': True,
        'ema_alignment': False,
        'momentum': False
    },
    'P1-Full (RSI>55 + EMA정배열)': {
        'enabled': True,
        'rsi_min': 55,
        'above_ema20': True,
        'ema_alignment': True,
        'momentum': False
    },
    'P1-Full + Momentum': {
        'enabled': True,
        'rsi_min': 55,
        'above_ema20': True,
        'ema_alignment': True,
        'momentum': True,
        'momentum_min': 0.3
    },
    'P1-Balanced (RSI>50 + EMA50위)': {
        'enabled': True,
        'rsi_min': 50,
        'above_ema50': True,
        'ema_alignment': False,
        'momentum': False
    }
}

def check_p1_filter(i, config):
    """P1 필터 조건 체크"""
    if not config.get('enabled', True):
        return True
    
    candle = candles_df.iloc[i]
    
    # RSI 조건
    if config.get('rsi_min'):
        if pd.isna(candle['rsi']) or candle['rsi'] < config['rsi_min']:
            return False
    
    # EMA20 위 조건
    if config.get('above_ema20'):
        if pd.isna(candle['ema_20']) or candle['close'] < candle['ema_20']:
            return False
    
    # EMA50 위 조건
    if config.get('above_ema50'):
        if pd.isna(candle['ema_50']) or candle['close'] < candle['ema_50']:
            return False
    
    # EMA 정배열 조건 (20 > 50 > 200)
    if config.get('ema_alignment'):
        if pd.isna(candle['ema_20']) or pd.isna(candle['ema_50']) or pd.isna(candle['ema_200']):
            return False
        if not (candle['ema_20'] > candle['ema_50'] > candle['ema_200']):
            return False
    
    # 모멘텀 조건
    if config.get('momentum'):
        min_change = config.get('momentum_min', 0.3)
        if pd.isna(candle['change_20']) or candle['change_20'] < min_change:
            return False
    
    return True


def run_backtest(filter_name, filter_config):
    """필터 적용 백테스트 실행"""
    trades = []
    position = None
    
    recent_lows = []
    searching_inflection = False
    hl_event = None
    
    # 필터 통계
    filter_blocked = 0
    filter_passed = 0
    
    for i in range(200, len(candles_df)):  # EMA 200 계산 대기
        candle = candles_df.iloc[i]
        prev_candles = candles_df.iloc[max(0, i-20):i]
        ema_value = candles_df.iloc[i]['ema_200']
        
        # === 저점 감지 ===
        if len(prev_candles) >= 10:
            recent_low = prev_candles['low'].tail(10).min()
            
            if candle['low'] <= recent_low * 1.002:
                recent_lows.append({
                    'price': candle['low'],
                    'time': candle['datetime'],
                    'index': i
                })
                if len(recent_lows) > 5:
                    recent_lows.pop(0)
        
        # === HL 감지 ===
        if len(recent_lows) >= 2 and not searching_inflection:
            current_low = recent_lows[-1]['price']
            previous_low = recent_lows[-2]['price']
            
            if current_low > previous_low:
                if candle['close'] > current_low * 1.003:
                    hl_strength = ((current_low - previous_low) / previous_low) * 100
                    
                    if hl_strength >= 0.5:
                        hl_event = {
                            'hl_time': recent_lows[-1]['time'],
                            'hl_price': current_low,
                            'hl_strength': hl_strength,
                            'hl_index': recent_lows[-1]['index']
                        }
                        searching_inflection = True
        
        # === 청산 체크 ===
        if position is not None:
            # 1. 기존 SL
            if candle['low'] <= position['sl_price']:
                exit_price = position['sl_price']
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': candle['datetime'],
                    'exit_price': exit_price,
                    'exit_reason': 'SL_HL',
                    'pnl_pct': pnl_pct,
                    'hl_strength': position['hl_strength'],
                    'rsi_at_entry': position.get('rsi_at_entry', 0),
                    'above_ema20': position.get('above_ema20', False)
                })
                position = None
                searching_inflection = False
                continue
            
            # 2. EMA 이탈 SL
            if position.get('was_above_ema', False) and candle['close'] < ema_value:
                exit_price = candle['close']
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': candle['datetime'],
                    'exit_price': exit_price,
                    'exit_reason': 'SL_EMA',
                    'pnl_pct': pnl_pct,
                    'hl_strength': position['hl_strength'],
                    'rsi_at_entry': position.get('rsi_at_entry', 0),
                    'above_ema20': position.get('above_ema20', False)
                })
                position = None
                searching_inflection = False
                continue
            
            # 3. TP2
            if candle['high'] >= position['tp2_price']:
                exit_price = position['tp2_price']
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': candle['datetime'],
                    'exit_price': exit_price,
                    'exit_reason': 'TP2',
                    'pnl_pct': pnl_pct,
                    'hl_strength': position['hl_strength'],
                    'rsi_at_entry': position.get('rsi_at_entry', 0),
                    'above_ema20': position.get('above_ema20', False)
                })
                position = None
                searching_inflection = False
                continue
            
            # 4. TP1
            if candle['high'] >= position['tp1_price']:
                exit_price = position['tp1_price']
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': candle['datetime'],
                    'exit_price': exit_price,
                    'exit_reason': 'TP1',
                    'pnl_pct': pnl_pct,
                    'hl_strength': position['hl_strength'],
                    'rsi_at_entry': position.get('rsi_at_entry', 0),
                    'above_ema20': position.get('above_ema20', False)
                })
                position = None
                searching_inflection = False
                continue
        
        # === 변곡점 캔들 진입 ===
        if position is None and searching_inflection and hl_event is not None:
            if i - hl_event['hl_index'] > 20:
                searching_inflection = False
                hl_event = None
                continue
            
            # 변곡점 조건
            if candle['close'] <= candle['open']:
                continue
            
            body_size = candle['close'] - candle['open']
            body_pct = (body_size / candle['open']) * 100
            if body_pct < 0.3:
                continue
            
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                continue
            body_to_range = (body_size / total_range) * 100
            if body_to_range < 60:
                continue
            
            if i >= 5:
                prev_vol = candles_df.iloc[i-5:i]['volume'].mean()
                vol_ratio = candle['volume'] / prev_vol if prev_vol > 0 else 1
                if vol_ratio < 1.0:
                    continue
            
            # === P1 필터 적용 ===
            if not check_p1_filter(i, filter_config):
                filter_blocked += 1
                continue
            
            filter_passed += 1
            
            # 진입!
            entry_price = candle['close']
            hl_strength = hl_event['hl_strength']
            
            if hl_strength >= 5:
                tp1_pct, tp2_pct = 2.0, 4.0
            elif hl_strength >= 2:
                tp1_pct, tp2_pct = 1.5, 3.0
            elif hl_strength >= 1:
                tp1_pct, tp2_pct = 1.0, 2.0
            else:
                tp1_pct, tp2_pct = 0.7, 1.5
            
            position = {
                'entry_time': candle['datetime'],
                'entry_price': entry_price,
                'tp1_price': entry_price * (1 + tp1_pct / 100),
                'tp2_price': entry_price * (1 + tp2_pct / 100),
                'sl_price': hl_event['hl_price'] * 0.99,
                'hl_strength': hl_strength,
                'was_above_ema': candle['close'] > ema_value,
                'rsi_at_entry': candle['rsi'],
                'above_ema20': candle['close'] > candle['ema_20'] if pd.notna(candle['ema_20']) else False
            }
            searching_inflection = False
    
    return trades, filter_passed, filter_blocked


# 모든 필터 조합 테스트
print("\n각 필터 조합 백테스트 실행 중...")
results = []

for filter_name, filter_config in FILTER_CONFIGS.items():
    trades, passed, blocked = run_backtest(filter_name, filter_config)
    
    if len(trades) == 0:
        continue
    
    trades_df = pd.DataFrame(trades)
    
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'].str.contains('SL')]
    
    win_rate = len(tp_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    # MDD 계산
    cumulative = trades_df['pnl_pct'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    mdd = drawdown.min()
    
    results.append({
        'Filter': filter_name,
        'Trades': len(trades_df),
        'Passed': passed,
        'Blocked': blocked,
        'Win Rate': win_rate,
        'SL Ratio': len(sl_trades) / len(trades_df) * 100,
        'Total PNL': total_pnl,
        'Avg PNL': avg_pnl,
        'MDD': mdd,
        'trades_df': trades_df
    })

# 결과 출력
print("\n" + "="*120)
print("📊 P0 + P1 통합 전략 비교 결과")
print("="*120)

print(f"\n{'필터':<35} {'거래수':>8} {'승률':>8} {'SL비율':>8} {'총PNL':>10} {'평균PNL':>10} {'MDD':>10}")
print("-"*100)

for r in results:
    print(f"{r['Filter']:<35} {r['Trades']:>8} {r['Win Rate']:>7.1f}% {r['SL Ratio']:>7.1f}% {r['Total PNL']:>9.1f}% {r['Avg PNL']:>9.3f}% {r['MDD']:>9.1f}%")

# 최적 필터 찾기
best_by_win_rate = max(results, key=lambda x: x['Win Rate'])
best_by_pnl = max(results, key=lambda x: x['Total PNL'])
best_by_avg_pnl = max(results, key=lambda x: x['Avg PNL'])

print("\n" + "="*120)
print("🏆 최적 필터 추천")
print("="*120)
print(f"  최고 승률: {best_by_win_rate['Filter']} ({best_by_win_rate['Win Rate']:.1f}%)")
print(f"  최고 총 PNL: {best_by_pnl['Filter']} ({best_by_pnl['Total PNL']:.1f}%)")
print(f"  최고 평균 PNL: {best_by_avg_pnl['Filter']} ({best_by_avg_pnl['Avg PNL']:.3f}%)")

# 기본 vs 최적 필터 상세 비교
print("\n" + "="*120)
print("📈 기본 전략 vs 최적 필터 전략 상세 비교")
print("="*120)

base = next((r for r in results if '기본' in r['Filter']), None)
optimal = best_by_win_rate

if base and optimal:
    print(f"\n{'지표':<20} {'기본 전략':>15} {'최적 필터':>15} {'차이':>15}")
    print("-"*70)
    print(f"{'거래 수':<20} {base['Trades']:>15} {optimal['Trades']:>15} {optimal['Trades']-base['Trades']:>+15}")
    print(f"{'승률':<20} {base['Win Rate']:>14.1f}% {optimal['Win Rate']:>14.1f}% {optimal['Win Rate']-base['Win Rate']:>+14.1f}%p")
    print(f"{'SL 비율':<20} {base['SL Ratio']:>14.1f}% {optimal['SL Ratio']:>14.1f}% {optimal['SL Ratio']-base['SL Ratio']:>+14.1f}%p")
    print(f"{'총 PNL':<20} {base['Total PNL']:>14.1f}% {optimal['Total PNL']:>14.1f}% {optimal['Total PNL']-base['Total PNL']:>+14.1f}%p")
    print(f"{'평균 PNL':<20} {base['Avg PNL']:>13.3f}% {optimal['Avg PNL']:>13.3f}% {(optimal['Avg PNL']-base['Avg PNL'])*100:>+13.1f}bp")
    print(f"{'MDD':<20} {base['MDD']:>14.1f}% {optimal['MDD']:>14.1f}% {optimal['MDD']-base['MDD']:>+14.1f}%p")

# 연도별 비교 (기본 vs 최적)
print("\n" + "="*120)
print("📅 연도별 성과 비교 (기본 vs 최적 필터)")
print("="*120)

if base and optimal:
    base_df = base['trades_df'].copy()
    optimal_df = optimal['trades_df'].copy()
    
    base_df['year'] = pd.to_datetime(base_df['entry_time']).dt.year
    optimal_df['year'] = pd.to_datetime(optimal_df['entry_time']).dt.year
    
    years = sorted(set(base_df['year'].unique()) | set(optimal_df['year'].unique()))
    
    print(f"\n{'연도':<8} {'기본 거래':>10} {'기본 PNL':>10} {'최적 거래':>10} {'최적 PNL':>10} {'PNL 차이':>10}")
    print("-"*70)
    
    for year in years:
        base_year = base_df[base_df['year'] == year]
        opt_year = optimal_df[optimal_df['year'] == year]
        
        base_trades = len(base_year)
        base_pnl = base_year['pnl_pct'].sum() if len(base_year) > 0 else 0
        opt_trades = len(opt_year)
        opt_pnl = opt_year['pnl_pct'].sum() if len(opt_year) > 0 else 0
        
        print(f"{year:<8} {base_trades:>10} {base_pnl:>9.1f}% {opt_trades:>10} {opt_pnl:>9.1f}% {opt_pnl-base_pnl:>+9.1f}%p")

# 결과 저장
print("\n" + "="*120)
print("💾 결과 저장")
print("="*120)

# 최적 필터 결과 저장
if optimal:
    optimal['trades_df'].to_csv('backtest_p0_p1_optimal_results.csv', index=False)
    print(f"  ✅ 최적 필터 결과: backtest_p0_p1_optimal_results.csv")

# 비교 결과 저장
comparison_df = pd.DataFrame([{
    'Filter': r['Filter'],
    'Trades': r['Trades'],
    'Win_Rate': r['Win Rate'],
    'SL_Ratio': r['SL Ratio'],
    'Total_PNL': r['Total PNL'],
    'Avg_PNL': r['Avg PNL'],
    'MDD': r['MDD']
} for r in results])
comparison_df.to_csv('backtest_p0_p1_comparison.csv', index=False)
print(f"  ✅ 필터 비교: backtest_p0_p1_comparison.csv")

print("\n" + "="*120)
print("🎯 결론")
print("="*120)

if base and optimal and optimal['Filter'] != base['Filter']:
    print(f"\n  P1 필터 ({optimal['Filter']}) 적용 시:")
    print(f"  - 승률: {base['Win Rate']:.1f}% → {optimal['Win Rate']:.1f}% (+{optimal['Win Rate']-base['Win Rate']:.1f}%p)")
    print(f"  - SL 비율: {base['SL Ratio']:.1f}% → {optimal['SL Ratio']:.1f}% ({optimal['SL Ratio']-base['SL Ratio']:+.1f}%p)")
    
    trade_reduction = (base['Trades'] - optimal['Trades']) / base['Trades'] * 100
    pnl_change = optimal['Total PNL'] - base['Total PNL']
    
    print(f"\n  ⚠️ 트레이드오프:")
    print(f"  - 거래 수 {trade_reduction:.0f}% 감소 ({base['Trades']} → {optimal['Trades']})")
    print(f"  - 총 PNL {pnl_change:+.1f}%p 변화")
    
    if pnl_change < 0:
        print(f"\n  💡 추천: 필터가 좋은 거래도 필터링하므로, 필터 강도 조절 필요")
    else:
        print(f"\n  ✅ 필터 적용이 수익성도 개선!")
else:
    print(f"\n  기본 전략이 가장 효과적입니다.")
