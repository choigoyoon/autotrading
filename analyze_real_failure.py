import pandas as pd
import numpy as np

"""
진짜 실패 사유 분석
- 손실 vs 승리 거래의 "진입 시점" 차이점 정밀 분석
- 단순 지표가 아닌, 실제 패턴 차이
"""

trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("="*100)
print("🔍 진짜 실패 사유 분석 - 승리 vs 손실 차이점")
print("="*100)

# 지표 계산
candles_df['ema_20'] = candles_df['close'].ewm(span=20, adjust=False).mean()
candles_df['ema_50'] = candles_df['close'].ewm(span=50, adjust=False).mean()
candles_df['ema_200'] = candles_df['close'].ewm(span=200, adjust=False).mean()

# 여러 시간대 모멘텀
for period in [5, 10, 20, 40, 80]:
    candles_df[f'mom_{period}'] = ((candles_df['close'] - candles_df['close'].shift(period)) / candles_df['close'].shift(period)) * 100

# 변동성
candles_df['range_5'] = (candles_df['high'].rolling(5).max() - candles_df['low'].rolling(5).min()) / candles_df['close'] * 100
candles_df['range_20'] = (candles_df['high'].rolling(20).max() - candles_df['low'].rolling(20).min()) / candles_df['close'] * 100

# RSI
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

candles_df['rsi'] = calc_rsi(candles_df['close'], 14)

# 볼륨
candles_df['vol_ratio'] = candles_df['volume'] / candles_df['volume'].rolling(20).mean()

# EMA 위치
candles_df['above_ema20'] = candles_df['close'] > candles_df['ema_20']
candles_df['above_ema50'] = candles_df['close'] > candles_df['ema_50']
candles_df['above_ema200'] = candles_df['close'] > candles_df['ema_200']
candles_df['ema_aligned'] = (candles_df['ema_20'] > candles_df['ema_50']) & (candles_df['ema_50'] > candles_df['ema_200'])

# EMA 기울기
candles_df['ema20_slope'] = (candles_df['ema_20'] - candles_df['ema_20'].shift(5)) / candles_df['ema_20'].shift(5) * 100
candles_df['ema50_slope'] = (candles_df['ema_50'] - candles_df['ema_50'].shift(5)) / candles_df['ema_50'].shift(5) * 100

# 캔들 패턴
candles_df['body_pct'] = abs(candles_df['close'] - candles_df['open']) / candles_df['open'] * 100
candles_df['upper_wick'] = (candles_df['high'] - candles_df[['close', 'open']].max(axis=1)) / candles_df['open'] * 100
candles_df['lower_wick'] = (candles_df[['close', 'open']].min(axis=1) - candles_df['low']) / candles_df['open'] * 100

candles_df.set_index('datetime', inplace=True)

# 거래별 지표 매칭
indicators = []
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    try:
        idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        c = candles_df.iloc[idx]
        
        # 진입 후 결과
        is_win = trade['pnl_pct'] > 0
        is_sl = 'SL' in trade['exit_reason']
        
        indicators.append({
            'pnl': trade['pnl_pct'],
            'is_win': is_win,
            'is_sl': is_sl,
            'hl_strength': trade['hl_strength'],
            # 모멘텀
            'mom_5': c['mom_5'],
            'mom_10': c['mom_10'],
            'mom_20': c['mom_20'],
            'mom_40': c['mom_40'],
            'mom_80': c['mom_80'],
            # EMA
            'above_ema20': c['above_ema20'],
            'above_ema50': c['above_ema50'],
            'above_ema200': c['above_ema200'],
            'ema_aligned': c['ema_aligned'],
            'ema20_slope': c['ema20_slope'],
            'ema50_slope': c['ema50_slope'],
            # RSI
            'rsi': c['rsi'],
            # 변동성
            'range_5': c['range_5'],
            'range_20': c['range_20'],
            # 볼륨
            'vol_ratio': c['vol_ratio'],
            # 캔들
            'body_pct': c['body_pct'],
            'upper_wick': c['upper_wick'],
            'lower_wick': c['lower_wick'],
        })
    except:
        pass

df = pd.DataFrame(indicators)
win_df = df[df['is_win']]
loss_df = df[~df['is_win']]
sl_df = df[df['is_sl']]

print(f"\n총 거래: {len(df)}건")
print(f"승리: {len(win_df)}건 ({len(win_df)/len(df)*100:.1f}%)")
print(f"손실: {len(loss_df)}건 ({len(loss_df)/len(df)*100:.1f}%)")
print(f"SL: {len(sl_df)}건 ({len(sl_df)/len(df)*100:.1f}%)")

# 승리 vs 손실 비교
print("\n" + "="*100)
print("📊 승리 vs 손실 거래 지표 비교")
print("="*100)

compare_cols = ['mom_5', 'mom_10', 'mom_20', 'mom_40', 'mom_80', 
                'ema20_slope', 'ema50_slope', 'rsi', 
                'range_5', 'range_20', 'vol_ratio', 
                'body_pct', 'hl_strength']

print(f"\n{'지표':<20} {'승리 평균':>12} {'손실 평균':>12} {'차이':>12} {'유의미':>8}")
print("-"*70)

significant = []
for col in compare_cols:
    win_mean = win_df[col].mean()
    loss_mean = loss_df[col].mean()
    diff = win_mean - loss_mean
    
    # 차이의 유의미성 (표준편차 대비)
    std = df[col].std()
    sig = abs(diff) / std if std > 0 else 0
    sig_flag = "⭐⭐" if sig > 0.3 else "⭐" if sig > 0.15 else ""
    
    if sig > 0.1:
        significant.append({'col': col, 'win': win_mean, 'loss': loss_mean, 'diff': diff, 'sig': sig})
    
    print(f"{col:<20} {win_mean:>12.3f} {loss_mean:>12.3f} {diff:>+12.3f} {sig_flag}")

# 불리언 지표 비교
print("\n" + "="*100)
print("📊 조건별 승률 비교")
print("="*100)

bool_cols = ['above_ema20', 'above_ema50', 'above_ema200', 'ema_aligned']

print(f"\n{'조건':<20} {'True 승률':>12} {'False 승률':>12} {'차이':>12}")
print("-"*60)

for col in bool_cols:
    true_win = df[df[col] == True]['is_win'].mean() * 100
    false_win = df[df[col] == False]['is_win'].mean() * 100
    diff = true_win - false_win
    flag = "⭐" if abs(diff) > 5 else ""
    print(f"{col:<20} {true_win:>11.1f}% {false_win:>11.1f}% {diff:>+11.1f}%p {flag}")

# 구간별 승률 분석
print("\n" + "="*100)
print("📊 핵심 지표 구간별 승률")
print("="*100)

def analyze_bins(col, bins, labels=None):
    df_temp = df.copy()
    df_temp['bin'] = pd.cut(df_temp[col], bins=bins, labels=labels)
    
    results = []
    for b in df_temp['bin'].dropna().unique():
        bin_data = df_temp[df_temp['bin'] == b]
        if len(bin_data) >= 20:
            win_rate = bin_data['is_win'].mean() * 100
            results.append({'bin': str(b), 'count': len(bin_data), 'win_rate': win_rate})
    
    return pd.DataFrame(results).sort_values('win_rate', ascending=False)

# 5봉 모멘텀
print("\n🔹 5봉 모멘텀별 승률:")
mom5_bins = analyze_bins('mom_5', [-100, -2, -1, -0.5, 0, 0.5, 1, 2, 100])
for _, r in mom5_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# 20봉 모멘텀
print("\n🔹 20봉 모멘텀별 승률:")
mom20_bins = analyze_bins('mom_20', [-100, -3, -1, 0, 1, 3, 100])
for _, r in mom20_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# RSI
print("\n🔹 RSI별 승률:")
rsi_bins = analyze_bins('rsi', [0, 30, 40, 50, 55, 60, 70, 100])
for _, r in rsi_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# EMA20 기울기
print("\n🔹 EMA20 기울기별 승률:")
slope_bins = analyze_bins('ema20_slope', [-100, -0.3, -0.1, 0, 0.1, 0.3, 100])
for _, r in slope_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# HL 강도
print("\n🔹 HL 강도별 승률:")
hl_bins = analyze_bins('hl_strength', [0, 1, 2, 3, 5, 100])
for _, r in hl_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# 변동성
print("\n🔹 변동성(range_5)별 승률:")
range_bins = analyze_bins('range_5', [0, 1, 2, 3, 5, 100])
for _, r in range_bins.iterrows():
    flag = "⭐" if r['win_rate'] > 70 else "🔴" if r['win_rate'] < 55 else ""
    print(f"  {r['bin']:<15}: {r['count']:>4}건, 승률 {r['win_rate']:>5.1f}% {flag}")

# 핵심 발견
print("\n" + "="*100)
print("💡 핵심 발견: 승리 거래의 특징")
print("="*100)

print("\n승리 거래 vs 손실 거래 차이:")
for s in sorted(significant, key=lambda x: -abs(x['sig']))[:7]:
    direction = "높음" if s['diff'] > 0 else "낮음"
    print(f"  • {s['col']}: 승리 거래가 더 {direction} ({s['win']:.2f} vs {s['loss']:.2f})")

# 최적 필터 도출
print("\n" + "="*100)
print("🎯 최적 진입 필터 테스트")
print("="*100)

def calc_metrics(data):
    if len(data) < 10:
        return None
    win_rate = data['is_win'].mean() * 100
    total_pnl = data['pnl'].sum()
    cum = data['pnl'].cumsum()
    mdd = (cum - cum.cummax()).min()
    return {'trades': len(data), 'win_rate': win_rate, 'total_pnl': total_pnl, 'mdd': mdd}

base = calc_metrics(df)
print(f"\n기본: {base['trades']}건, 승률 {base['win_rate']:.1f}%, MDD {base['mdd']:.1f}%")

# 발견된 패턴 기반 필터
test_filters = {
    # 모멘텀 기반
    '5봉 모멘텀 > -0.5%': df['mom_5'] > -0.5,
    '5봉 모멘텀 > 0%': df['mom_5'] > 0,
    '5봉 모멘텀 > 0.5%': df['mom_5'] > 0.5,
    '20봉 모멘텀 > -1%': df['mom_20'] > -1,
    '20봉 모멘텀 > 0%': df['mom_20'] > 0,
    
    # RSI 기반
    'RSI > 50': df['rsi'] > 50,
    'RSI > 55': df['rsi'] > 55,
    'RSI 40-70': (df['rsi'] > 40) & (df['rsi'] < 70),
    
    # EMA 기반
    'EMA20 기울기 > 0': df['ema20_slope'] > 0,
    'EMA20 기울기 > 0.1': df['ema20_slope'] > 0.1,
    'EMA20 위': df['above_ema20'],
    
    # 복합
    'mom5>0 + RSI>55': (df['mom_5'] > 0) & (df['rsi'] > 55),
    'mom5>0 + EMA20위': (df['mom_5'] > 0) & (df['above_ema20']),
    'mom5>-0.5 + RSI>50': (df['mom_5'] > -0.5) & (df['rsi'] > 50),
    'mom20>0 + RSI>55': (df['mom_20'] > 0) & (df['rsi'] > 55),
    'EMA기울기>0 + RSI>55': (df['ema20_slope'] > 0) & (df['rsi'] > 55),
}

results = []
for name, cond in test_filters.items():
    m = calc_metrics(df[cond].reset_index(drop=True))
    if m:
        results.append({
            'filter': name,
            'trades': m['trades'],
            'block_pct': (base['trades'] - m['trades']) / base['trades'] * 100,
            'win_rate': m['win_rate'],
            'mdd': m['mdd'],
            'total_pnl': m['total_pnl']
        })

results_df = pd.DataFrame(results).sort_values('win_rate', ascending=False)

print(f"\n{'필터':<30} {'거래':>6} {'차단%':>7} {'승률':>7} {'MDD':>8} {'총PNL':>8}")
print("-"*75)
for _, r in results_df.iterrows():
    flag = "⭐" if r['win_rate'] >= 68 and r['mdd'] >= -15 else ""
    print(f"{r['filter']:<30} {r['trades']:>6} {r['block_pct']:>6.1f}% {r['win_rate']:>6.1f}% {r['mdd']:>7.1f}% {r['total_pnl']:>7.1f}% {flag}")
