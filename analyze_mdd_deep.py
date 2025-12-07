import pandas as pd
import numpy as np

"""
MDD 심층 분석
- 연속 손실 발생 원인
- 시장 상황별 손실 패턴
- 진입 자리 문제점 정밀 분석
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("="*100)
print("📉 MDD 심층 분석 - 진입 자리 문제점")
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

# 다양한 시간대 가격 변화
candles_df['change_1h'] = ((candles_df['close'] - candles_df['close'].shift(4)) / candles_df['close'].shift(4)) * 100
candles_df['change_4h'] = ((candles_df['close'] - candles_df['close'].shift(16)) / candles_df['close'].shift(16)) * 100
candles_df['change_12h'] = ((candles_df['close'] - candles_df['close'].shift(48)) / candles_df['close'].shift(48)) * 100
candles_df['change_24h'] = ((candles_df['close'] - candles_df['close'].shift(96)) / candles_df['close'].shift(96)) * 100

# 변동성
candles_df['volatility_4h'] = candles_df['high'].rolling(16).max() / candles_df['low'].rolling(16).min() - 1
candles_df['volatility_24h'] = candles_df['high'].rolling(96).max() / candles_df['low'].rolling(96).min() - 1

# ATR
candles_df['tr'] = np.maximum(
    candles_df['high'] - candles_df['low'],
    np.maximum(
        abs(candles_df['high'] - candles_df['close'].shift(1)),
        abs(candles_df['low'] - candles_df['close'].shift(1))
    )
)
candles_df['atr_14'] = candles_df['tr'].rolling(14).mean()
candles_df['atr_pct'] = candles_df['atr_14'] / candles_df['close'] * 100

# EMA 기울기
candles_df['ema20_slope'] = (candles_df['ema_20'] - candles_df['ema_20'].shift(4)) / candles_df['ema_20'].shift(4) * 100
candles_df['ema50_slope'] = (candles_df['ema_50'] - candles_df['ema_50'].shift(4)) / candles_df['ema_50'].shift(4) * 100

# 고점/저점 대비 위치
candles_df['high_24h'] = candles_df['high'].rolling(96).max()
candles_df['low_24h'] = candles_df['low'].rolling(96).min()
candles_df['position_in_range'] = (candles_df['close'] - candles_df['low_24h']) / (candles_df['high_24h'] - candles_df['low_24h'] + 0.001) * 100

candles_df.set_index('datetime', inplace=True)

# 각 거래에 지표 추가
def get_trade_conditions(trade_df, candles):
    conditions = []
    for _, trade in trade_df.iterrows():
        entry_time = trade['entry_time']
        try:
            idx = candles.index.get_indexer([entry_time], method='nearest')[0]
            candle = candles.iloc[idx]
            
            conditions.append({
                'pnl': trade['pnl_pct'],
                'exit_reason': trade['exit_reason'],
                'entry_time': entry_time,
                'rsi': candle['rsi'],
                'change_1h': candle['change_1h'],
                'change_4h': candle['change_4h'],
                'change_12h': candle['change_12h'],
                'change_24h': candle['change_24h'],
                'volatility_4h': candle['volatility_4h'] * 100,
                'volatility_24h': candle['volatility_24h'] * 100,
                'atr_pct': candle['atr_pct'],
                'ema20_slope': candle['ema20_slope'],
                'ema50_slope': candle['ema50_slope'],
                'position_in_range': candle['position_in_range'],
                'above_ema20': candle['close'] > candle['ema_20'],
                'above_ema50': candle['close'] > candle['ema_50'],
                'above_ema200': candle['close'] > candle['ema_200'],
                'ema_aligned': candle['ema_20'] > candle['ema_50'] > candle['ema_200'] if pd.notna(candle['ema_200']) else False,
                'dist_from_ema20': ((candle['close'] - candle['ema_20']) / candle['ema_20']) * 100 if pd.notna(candle['ema_20']) else 0,
                'dist_from_ema200': ((candle['close'] - candle['ema_200']) / candle['ema_200']) * 100 if pd.notna(candle['ema_200']) else 0,
            })
        except:
            pass
    return pd.DataFrame(conditions)

print("\n거래별 진입 조건 분석 중...")
all_conditions = get_trade_conditions(trades_df, candles_df)
all_conditions['is_loss'] = all_conditions['pnl'] < 0
all_conditions['is_big_loss'] = all_conditions['pnl'] < -1  # 큰 손실 (-1% 이상)

loss_conditions = all_conditions[all_conditions['is_loss']]
win_conditions = all_conditions[~all_conditions['is_loss']]
big_loss_conditions = all_conditions[all_conditions['is_big_loss']]

print(f"\n전체 거래: {len(all_conditions)}건")
print(f"손실 거래: {len(loss_conditions)}건 ({len(loss_conditions)/len(all_conditions)*100:.1f}%)")
print(f"큰 손실 (-1% 이상): {len(big_loss_conditions)}건 ({len(big_loss_conditions)/len(all_conditions)*100:.1f}%)")

# 1. 다양한 조건별 손실률 분석
print("\n" + "="*100)
print("📊 조건별 손실률 상세 분석")
print("="*100)

def analyze_condition_bins(df, column, bins, labels=None):
    """조건 구간별 손실률 분석"""
    df = df.copy()
    df['bin'] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    
    results = []
    for bin_label in df['bin'].unique():
        if pd.isna(bin_label):
            continue
        bin_data = df[df['bin'] == bin_label]
        total = len(bin_data)
        if total < 10:
            continue
        loss_count = bin_data['is_loss'].sum()
        loss_rate = loss_count / total * 100
        avg_pnl = bin_data['pnl'].mean()
        
        results.append({
            'bin': str(bin_label),
            'total': total,
            'loss_count': loss_count,
            'loss_rate': loss_rate,
            'avg_pnl': avg_pnl
        })
    
    return pd.DataFrame(results).sort_values('loss_rate', ascending=False)

# 4시간 가격 변화
print("\n🔹 4시간 가격 변화별 손실률:")
bins_4h = [-100, -3, -2, -1, 0, 1, 2, 3, 100]
result_4h = analyze_condition_bins(all_conditions, 'change_4h', bins_4h)
for _, r in result_4h.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# 24시간 가격 변화
print("\n🔹 24시간 가격 변화별 손실률:")
bins_24h = [-100, -5, -3, -1, 1, 3, 5, 100]
result_24h = analyze_condition_bins(all_conditions, 'change_24h', bins_24h)
for _, r in result_24h.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# RSI 구간
print("\n🔹 RSI별 손실률:")
bins_rsi = [0, 30, 40, 50, 60, 70, 80, 100]
result_rsi = analyze_condition_bins(all_conditions, 'rsi', bins_rsi)
for _, r in result_rsi.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# 24시간 레인지 내 위치
print("\n🔹 24시간 레인지 내 위치별 손실률:")
bins_pos = [0, 20, 40, 60, 80, 100]
result_pos = analyze_condition_bins(all_conditions, 'position_in_range', bins_pos)
for _, r in result_pos.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# ATR
print("\n🔹 ATR(변동성)별 손실률:")
bins_atr = [0, 0.3, 0.5, 0.7, 1.0, 10]
result_atr = analyze_condition_bins(all_conditions, 'atr_pct', bins_atr)
for _, r in result_atr.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# EMA20 기울기
print("\n🔹 EMA20 기울기별 손실률:")
bins_slope = [-100, -0.5, -0.2, 0, 0.2, 0.5, 100]
result_slope = analyze_condition_bins(all_conditions, 'ema20_slope', bins_slope)
for _, r in result_slope.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# EMA200 거리
print("\n🔹 EMA200 거리별 손실률:")
bins_dist = [-100, -5, -2, 0, 2, 5, 100]
result_dist = analyze_condition_bins(all_conditions, 'dist_from_ema200', bins_dist)
for _, r in result_dist.iterrows():
    flag = "🔴" if r['loss_rate'] > 50 else "✅" if r['loss_rate'] < 35 else "⚠️"
    print(f"  {r['bin']:<15}: {r['total']:>4}건, 손실률 {r['loss_rate']:>5.1f}%, 평균PNL {r['avg_pnl']:>+6.2f}% {flag}")

# 2. 위험 진입 조합 찾기
print("\n" + "="*100)
print("🚨 위험 진입 패턴 조합")
print("="*100)

# 여러 조건 조합으로 위험 패턴 찾기
dangerous_patterns = []

# 패턴 1: 4시간 하락 + RSI 낮음
pattern1 = all_conditions[(all_conditions['change_4h'] < -1) & (all_conditions['rsi'] < 50)]
if len(pattern1) >= 10:
    loss_rate = pattern1['is_loss'].sum() / len(pattern1) * 100
    dangerous_patterns.append({
        'pattern': '4시간 -1% 하락 + RSI < 50',
        'count': len(pattern1),
        'loss_rate': loss_rate,
        'avg_pnl': pattern1['pnl'].mean()
    })

# 패턴 2: EMA 역배열 + 24시간 하락
pattern2 = all_conditions[(~all_conditions['ema_aligned']) & (all_conditions['change_24h'] < -2)]
if len(pattern2) >= 10:
    loss_rate = pattern2['is_loss'].sum() / len(pattern2) * 100
    dangerous_patterns.append({
        'pattern': 'EMA 역배열 + 24시간 -2% 하락',
        'count': len(pattern2),
        'loss_rate': loss_rate,
        'avg_pnl': pattern2['pnl'].mean()
    })

# 패턴 3: 레인지 하단 + EMA20 아래
pattern3 = all_conditions[(all_conditions['position_in_range'] < 30) & (~all_conditions['above_ema20'])]
if len(pattern3) >= 10:
    loss_rate = pattern3['is_loss'].sum() / len(pattern3) * 100
    dangerous_patterns.append({
        'pattern': '24시간 레인지 하단 + EMA20 아래',
        'count': len(pattern3),
        'loss_rate': loss_rate,
        'avg_pnl': pattern3['pnl'].mean()
    })

# 패턴 4: 고변동성 + RSI 중립
pattern4 = all_conditions[(all_conditions['atr_pct'] > 0.7) & (all_conditions['rsi'] > 40) & (all_conditions['rsi'] < 60)]
if len(pattern4) >= 10:
    loss_rate = pattern4['is_loss'].sum() / len(pattern4) * 100
    dangerous_patterns.append({
        'pattern': '고변동성(ATR>0.7%) + RSI 중립(40-60)',
        'count': len(pattern4),
        'loss_rate': loss_rate,
        'avg_pnl': pattern4['pnl'].mean()
    })

# 패턴 5: EMA20 하락 기울기 + 가격 EMA20 아래
pattern5 = all_conditions[(all_conditions['ema20_slope'] < -0.2) & (~all_conditions['above_ema20'])]
if len(pattern5) >= 10:
    loss_rate = pattern5['is_loss'].sum() / len(pattern5) * 100
    dangerous_patterns.append({
        'pattern': 'EMA20 하락(-0.2% 기울기) + 가격 EMA20 아래',
        'count': len(pattern5),
        'loss_rate': loss_rate,
        'avg_pnl': pattern5['pnl'].mean()
    })

# 패턴 6: EMA200 아래 + 하락 모멘텀
pattern6 = all_conditions[(~all_conditions['above_ema200']) & (all_conditions['change_4h'] < 0)]
if len(pattern6) >= 10:
    loss_rate = pattern6['is_loss'].sum() / len(pattern6) * 100
    dangerous_patterns.append({
        'pattern': 'EMA200 아래 + 4시간 하락',
        'count': len(pattern6),
        'loss_rate': loss_rate,
        'avg_pnl': pattern6['pnl'].mean()
    })

# 정렬 및 출력
dangerous_df = pd.DataFrame(dangerous_patterns).sort_values('loss_rate', ascending=False)
print(f"\n{'패턴':<45} {'거래수':>8} {'손실률':>10} {'평균PNL':>10}")
print("-"*80)
for _, p in dangerous_df.iterrows():
    flag = "🔴 위험" if p['loss_rate'] > 50 else "⚠️ 주의" if p['loss_rate'] > 40 else ""
    print(f"{p['pattern']:<45} {p['count']:>8} {p['loss_rate']:>9.1f}% {p['avg_pnl']:>+9.2f}% {flag}")

# 3. 안전 진입 패턴 찾기
print("\n" + "="*100)
print("✅ 안전 진입 패턴 조합")
print("="*100)

safe_patterns = []

# 안전 패턴 1: RSI 상승 + 4시간 상승
safe1 = all_conditions[(all_conditions['rsi'] > 55) & (all_conditions['change_4h'] > 0)]
if len(safe1) >= 10:
    loss_rate = safe1['is_loss'].sum() / len(safe1) * 100
    safe_patterns.append({
        'pattern': 'RSI > 55 + 4시간 상승',
        'count': len(safe1),
        'loss_rate': loss_rate,
        'avg_pnl': safe1['pnl'].mean()
    })

# 안전 패턴 2: EMA 정배열 + 레인지 중간
safe2 = all_conditions[(all_conditions['ema_aligned']) & (all_conditions['position_in_range'] > 40) & (all_conditions['position_in_range'] < 70)]
if len(safe2) >= 10:
    loss_rate = safe2['is_loss'].sum() / len(safe2) * 100
    safe_patterns.append({
        'pattern': 'EMA 정배열 + 레인지 중간(40-70%)',
        'count': len(safe2),
        'loss_rate': loss_rate,
        'avg_pnl': safe2['pnl'].mean()
    })

# 안전 패턴 3: 24시간 상승 + EMA200 위
safe3 = all_conditions[(all_conditions['change_24h'] > 2) & (all_conditions['above_ema200'])]
if len(safe3) >= 10:
    loss_rate = safe3['is_loss'].sum() / len(safe3) * 100
    safe_patterns.append({
        'pattern': '24시간 +2% 상승 + EMA200 위',
        'count': len(safe3),
        'loss_rate': loss_rate,
        'avg_pnl': safe3['pnl'].mean()
    })

# 안전 패턴 4: 저변동성 + EMA20 위
safe4 = all_conditions[(all_conditions['atr_pct'] < 0.5) & (all_conditions['above_ema20'])]
if len(safe4) >= 10:
    loss_rate = safe4['is_loss'].sum() / len(safe4) * 100
    safe_patterns.append({
        'pattern': '저변동성(ATR<0.5%) + EMA20 위',
        'count': len(safe4),
        'loss_rate': loss_rate,
        'avg_pnl': safe4['pnl'].mean()
    })

# 안전 패턴 5: RSI 높음 + EMA20 상승 기울기
safe5 = all_conditions[(all_conditions['rsi'] > 60) & (all_conditions['ema20_slope'] > 0.1)]
if len(safe5) >= 10:
    loss_rate = safe5['is_loss'].sum() / len(safe5) * 100
    safe_patterns.append({
        'pattern': 'RSI > 60 + EMA20 상승(+0.1% 기울기)',
        'count': len(safe5),
        'loss_rate': loss_rate,
        'avg_pnl': safe5['pnl'].mean()
    })

safe_df = pd.DataFrame(safe_patterns).sort_values('loss_rate')
print(f"\n{'패턴':<45} {'거래수':>8} {'손실률':>10} {'평균PNL':>10}")
print("-"*80)
for _, p in safe_df.iterrows():
    flag = "✅ 안전" if p['loss_rate'] < 35 else ""
    print(f"{p['pattern']:<45} {p['count']:>8} {p['loss_rate']:>9.1f}% {p['avg_pnl']:>+9.2f}% {flag}")

# 4. 결론
print("\n" + "="*100)
print("🎯 MDD 원인 및 개선안")
print("="*100)

# 가장 위험한 패턴
worst_pattern = dangerous_df.iloc[0] if len(dangerous_df) > 0 else None
best_pattern = safe_df.iloc[0] if len(safe_df) > 0 else None

if worst_pattern is not None:
    print(f"\n🚨 가장 위험한 진입 패턴:")
    print(f"   {worst_pattern['pattern']}")
    print(f"   손실률 {worst_pattern['loss_rate']:.1f}%, 평균 PNL {worst_pattern['avg_pnl']:+.2f}%")
    print(f"   → 이 패턴에서 {int(worst_pattern['count'])}건 진입, 회피 시 MDD 개선 가능")

if best_pattern is not None:
    print(f"\n✅ 가장 안전한 진입 패턴:")
    print(f"   {best_pattern['pattern']}")
    print(f"   손실률 {best_pattern['loss_rate']:.1f}%, 평균 PNL {best_pattern['avg_pnl']:+.2f}%")

print("\n💡 MDD 개선을 위한 진입 필터 추천:")
print("  1. 4시간 -1% 이상 하락 중 진입 금지")
print("  2. RSI 50 이하 + EMA20 아래 진입 금지")
print("  3. 24시간 레인지 하단 30% 이하 진입 주의")
print("  4. EMA20 기울기 하락 시 진입 주의")

# 개선 효과 시뮬레이션
print("\n" + "="*100)
print("📈 필터 적용 시 예상 효과 시뮬레이션")
print("="*100)

# 위험 패턴 제외
filtered = all_conditions[
    (all_conditions['change_4h'] >= -1) |  # 4시간 -1% 이상 하락 제외 (OR: 적어도 하나 조건 통과)
    ((all_conditions['rsi'] >= 50) | (all_conditions['above_ema20']))  # RSI 50 이상 또는 EMA20 위
]

# 더 엄격한 필터
strict_filtered = all_conditions[
    (all_conditions['change_4h'] >= -0.5) &  # 4시간 -0.5% 이상 하락 제외
    (all_conditions['rsi'] >= 50) &  # RSI 50 이상
    (all_conditions['above_ema20'])  # EMA20 위
]

print(f"\n{'지표':<25} {'현재':>12} {'완화 필터':>12} {'엄격 필터':>12}")
print("-"*65)

# 현재
current_trades = len(all_conditions)
current_loss_rate = all_conditions['is_loss'].sum() / current_trades * 100
current_pnl = all_conditions['pnl'].sum()

# MDD 계산 함수
def calc_mdd(df):
    cum = df['pnl'].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    return dd.min()

current_mdd = calc_mdd(all_conditions)

# 완화 필터
filtered_trades = len(filtered)
filtered_loss_rate = filtered['is_loss'].sum() / filtered_trades * 100 if filtered_trades > 0 else 0
filtered_pnl = filtered['pnl'].sum()
filtered_mdd = calc_mdd(filtered.reset_index(drop=True)) if len(filtered) > 0 else 0

# 엄격 필터
strict_trades = len(strict_filtered)
strict_loss_rate = strict_filtered['is_loss'].sum() / strict_trades * 100 if strict_trades > 0 else 0
strict_pnl = strict_filtered['pnl'].sum()
strict_mdd = calc_mdd(strict_filtered.reset_index(drop=True)) if len(strict_filtered) > 0 else 0

print(f"{'거래 수':<25} {current_trades:>12} {filtered_trades:>12} {strict_trades:>12}")
print(f"{'손실률':<25} {current_loss_rate:>11.1f}% {filtered_loss_rate:>11.1f}% {strict_loss_rate:>11.1f}%")
print(f"{'총 PNL':<25} {current_pnl:>11.1f}% {filtered_pnl:>11.1f}% {strict_pnl:>11.1f}%")
print(f"{'MDD':<25} {current_mdd:>11.1f}% {filtered_mdd:>11.1f}% {strict_mdd:>11.1f}%")
print(f"{'평균 PNL':<25} {current_pnl/current_trades:>11.3f}% {filtered_pnl/filtered_trades:>11.3f}% {strict_pnl/strict_trades:>11.3f}%")

# 저장
all_conditions.to_csv('trade_entry_conditions_analysis.csv', index=False)
print(f"\n✅ 분석 결과 저장: trade_entry_conditions_analysis.csv")
