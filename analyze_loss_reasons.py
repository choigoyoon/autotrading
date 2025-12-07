import pandas as pd
import numpy as np

"""
손실 거래의 "하락 사유" 정밀 분석
- 단순 하락도가 아닌, 하락의 원인/상황 파악
- 추세 전환? 조정? 급락? 횡보 후 이탈?
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("="*100)
print("🔍 손실 거래 - 하락 사유 정밀 분석")
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
candles_df['change_24h'] = ((candles_df['close'] - candles_df['close'].shift(96)) / candles_df['close'].shift(96)) * 100

# 고점/저점 대비
candles_df['high_24h'] = candles_df['high'].rolling(96).max()
candles_df['low_24h'] = candles_df['low'].rolling(96).min()
candles_df['high_4h'] = candles_df['high'].rolling(16).max()
candles_df['low_4h'] = candles_df['low'].rolling(16).min()

# 고점 대비 하락폭
candles_df['drop_from_high_24h'] = ((candles_df['close'] - candles_df['high_24h']) / candles_df['high_24h']) * 100
candles_df['drop_from_high_4h'] = ((candles_df['close'] - candles_df['high_4h']) / candles_df['high_4h']) * 100

# EMA 배열 상태
candles_df['ema_aligned'] = (candles_df['ema_20'] > candles_df['ema_50']) & (candles_df['ema_50'] > candles_df['ema_200'])
candles_df['ema_reverse'] = (candles_df['ema_20'] < candles_df['ema_50']) & (candles_df['ema_50'] < candles_df['ema_200'])

# EMA 기울기
candles_df['ema20_slope'] = (candles_df['ema_20'] - candles_df['ema_20'].shift(4)) / candles_df['ema_20'].shift(4) * 100
candles_df['ema50_slope'] = (candles_df['ema_50'] - candles_df['ema_50'].shift(4)) / candles_df['ema_50'].shift(4) * 100

# 변동성 (ATR)
candles_df['tr'] = np.maximum(
    candles_df['high'] - candles_df['low'],
    np.maximum(
        abs(candles_df['high'] - candles_df['close'].shift(1)),
        abs(candles_df['low'] - candles_df['close'].shift(1))
    )
)
candles_df['atr'] = candles_df['tr'].rolling(14).mean()
candles_df['atr_pct'] = candles_df['atr'] / candles_df['close'] * 100

# 급락 감지 (직전 캔들 대비)
candles_df['candle_drop'] = ((candles_df['close'] - candles_df['open']) / candles_df['open']) * 100
candles_df['is_big_red'] = candles_df['candle_drop'] < -1  # 1% 이상 음봉

# 연속 하락 캔들 수
candles_df['is_red'] = candles_df['close'] < candles_df['open']
candles_df['red_streak'] = candles_df['is_red'].rolling(10).sum()  # 최근 10개 중 음봉 수

candles_df.set_index('datetime', inplace=True)

# 각 거래에 상세 정보 추가
def classify_market_situation(candle):
    """시장 상황 분류"""
    situations = []
    
    # 1. 추세 상태
    if candle['ema_aligned']:
        situations.append('상승추세(EMA정배열)')
    elif candle['ema_reverse']:
        situations.append('하락추세(EMA역배열)')
    else:
        situations.append('횡보/혼조')
    
    # 2. 단기 모멘텀
    if candle['change_4h'] > 2:
        situations.append('4H강한상승')
    elif candle['change_4h'] > 0:
        situations.append('4H상승')
    elif candle['change_4h'] > -1:
        situations.append('4H약보합')
    elif candle['change_4h'] > -2:
        situations.append('4H약한하락')
    else:
        situations.append('4H급락')
    
    # 3. 고점 대비 위치
    if candle['drop_from_high_24h'] > -1:
        situations.append('24H고점근처')
    elif candle['drop_from_high_24h'] > -3:
        situations.append('24H고점-3%이내')
    elif candle['drop_from_high_24h'] > -5:
        situations.append('24H고점-5%이내')
    else:
        situations.append('24H고점-5%이상하락')
    
    # 4. RSI 상태
    if candle['rsi'] > 70:
        situations.append('RSI과매수')
    elif candle['rsi'] > 55:
        situations.append('RSI강세')
    elif candle['rsi'] > 45:
        situations.append('RSI중립')
    elif candle['rsi'] > 30:
        situations.append('RSI약세')
    else:
        situations.append('RSI과매도')
    
    # 5. 연속 하락
    if candle['red_streak'] >= 7:
        situations.append('연속음봉(7+)')
    elif candle['red_streak'] >= 5:
        situations.append('연속음봉(5+)')
    
    # 6. 변동성
    if candle['atr_pct'] > 1:
        situations.append('고변동성')
    elif candle['atr_pct'] < 0.3:
        situations.append('저변동성')
    
    return situations

# 모든 거래 분석
results = []
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    try:
        idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        candle = candles_df.iloc[idx]
        
        situations = classify_market_situation(candle)
        
        results.append({
            'entry_time': entry_time,
            'pnl': trade['pnl_pct'],
            'exit_reason': trade['exit_reason'],
            'is_loss': trade['pnl_pct'] < 0,
            'is_sl': 'SL' in trade['exit_reason'],
            'change_4h': candle['change_4h'],
            'change_24h': candle['change_24h'],
            'drop_from_high_24h': candle['drop_from_high_24h'],
            'drop_from_high_4h': candle['drop_from_high_4h'],
            'rsi': candle['rsi'],
            'ema_aligned': candle['ema_aligned'],
            'ema_reverse': candle['ema_reverse'],
            'ema20_slope': candle['ema20_slope'],
            'red_streak': candle['red_streak'],
            'atr_pct': candle['atr_pct'],
            'situations': '|'.join(situations),
            'trend': situations[0],
            'momentum_4h': situations[1],
            'position_24h': situations[2],
            'rsi_state': situations[3]
        })
    except Exception as e:
        pass

df = pd.DataFrame(results)
loss_df = df[df['is_loss']]
win_df = df[~df['is_loss']]

print(f"\n전체 거래: {len(df)}건")
print(f"손실 거래: {len(loss_df)}건 ({len(loss_df)/len(df)*100:.1f}%)")
print(f"승리 거래: {len(win_df)}건 ({len(win_df)/len(df)*100:.1f}%)")

# 1. 추세별 손실률
print("\n" + "="*100)
print("📊 1. 추세(EMA배열) 상태별 손실률")
print("="*100)

for trend in df['trend'].unique():
    t_df = df[df['trend'] == trend]
    loss_rate = t_df['is_loss'].sum() / len(t_df) * 100
    avg_pnl = t_df['pnl'].mean()
    flag = "🔴" if loss_rate > 45 else "✅" if loss_rate < 35 else "⚠️"
    print(f"  {trend:<25}: {len(t_df):>4}건, 손실률 {loss_rate:>5.1f}%, 평균PNL {avg_pnl:>+6.2f}% {flag}")

# 2. 4시간 모멘텀별 손실률
print("\n" + "="*100)
print("📊 2. 4시간 모멘텀별 손실률")
print("="*100)

for mom in ['4H급락', '4H약한하락', '4H약보합', '4H상승', '4H강한상승']:
    m_df = df[df['momentum_4h'] == mom]
    if len(m_df) < 5:
        continue
    loss_rate = m_df['is_loss'].sum() / len(m_df) * 100
    avg_pnl = m_df['pnl'].mean()
    flag = "🔴" if loss_rate > 45 else "✅" if loss_rate < 35 else "⚠️"
    print(f"  {mom:<25}: {len(m_df):>4}건, 손실률 {loss_rate:>5.1f}%, 평균PNL {avg_pnl:>+6.2f}% {flag}")

# 3. 24시간 고점 대비 위치별
print("\n" + "="*100)
print("📊 3. 24시간 고점 대비 위치별 손실률")
print("="*100)

for pos in ['24H고점근처', '24H고점-3%이내', '24H고점-5%이내', '24H고점-5%이상하락']:
    p_df = df[df['position_24h'] == pos]
    if len(p_df) < 5:
        continue
    loss_rate = p_df['is_loss'].sum() / len(p_df) * 100
    avg_pnl = p_df['pnl'].mean()
    flag = "🔴" if loss_rate > 45 else "✅" if loss_rate < 35 else "⚠️"
    print(f"  {pos:<25}: {len(p_df):>4}건, 손실률 {loss_rate:>5.1f}%, 평균PNL {avg_pnl:>+6.2f}% {flag}")

# 4. RSI 상태별
print("\n" + "="*100)
print("📊 4. RSI 상태별 손실률")
print("="*100)

for rsi_state in ['RSI과매도', 'RSI약세', 'RSI중립', 'RSI강세', 'RSI과매수']:
    r_df = df[df['rsi_state'] == rsi_state]
    if len(r_df) < 5:
        continue
    loss_rate = r_df['is_loss'].sum() / len(r_df) * 100
    avg_pnl = r_df['pnl'].mean()
    flag = "🔴" if loss_rate > 45 else "✅" if loss_rate < 35 else "⚠️"
    print(f"  {rsi_state:<25}: {len(r_df):>4}건, 손실률 {loss_rate:>5.1f}%, 평균PNL {avg_pnl:>+6.2f}% {flag}")

# 5. 복합 상황 분석 - 위험 조합 찾기
print("\n" + "="*100)
print("🚨 5. 위험 상황 조합 (손실률 높은 패턴)")
print("="*100)

combinations = {}
for _, row in df.iterrows():
    key = f"{row['trend']} + {row['momentum_4h']}"
    if key not in combinations:
        combinations[key] = {'total': 0, 'loss': 0, 'pnl_sum': 0}
    combinations[key]['total'] += 1
    combinations[key]['loss'] += 1 if row['is_loss'] else 0
    combinations[key]['pnl_sum'] += row['pnl']

comb_list = []
for key, val in combinations.items():
    if val['total'] >= 15:
        comb_list.append({
            'combination': key,
            'total': val['total'],
            'loss_rate': val['loss'] / val['total'] * 100,
            'avg_pnl': val['pnl_sum'] / val['total']
        })

comb_df = pd.DataFrame(comb_list).sort_values('loss_rate', ascending=False)

print(f"\n{'상황 조합':<45} {'거래':>6} {'손실률':>8} {'평균PNL':>10}")
print("-"*75)
for _, c in comb_df.iterrows():
    flag = "🔴 위험" if c['loss_rate'] > 50 else "✅ 안전" if c['loss_rate'] < 30 else ""
    print(f"{c['combination']:<45} {c['total']:>6} {c['loss_rate']:>7.1f}% {c['avg_pnl']:>+9.2f}% {flag}")

# 6. 하락 사유 심층 분석
print("\n" + "="*100)
print("🔍 6. 손실 거래의 하락 사유 심층 분석")
print("="*100)

# 손실 거래만 분석
print("\n손실 거래의 진입 당시 상황:")

# 손실 케이스별 분류
sl_trades = df[df['is_sl']]
print(f"\n총 손절(SL) 거래: {len(sl_trades)}건")

# 손절 사유별 분류
sl_reasons = {
    '추세 전환 중 진입': sl_trades[(sl_trades['ema20_slope'] < 0) & (~sl_trades['ema_aligned'])],
    '급락장 진입': sl_trades[sl_trades['change_4h'] < -2],
    '고점 추격 진입': sl_trades[(sl_trades['drop_from_high_24h'] > -1) & (sl_trades['rsi'] > 60)],
    '약한 하락 중 진입': sl_trades[(sl_trades['change_4h'] >= -2) & (sl_trades['change_4h'] < -1)],
    '연속 하락 중 진입': sl_trades[sl_trades['red_streak'] >= 5],
    '하락추세 역배열 진입': sl_trades[sl_trades['ema_reverse']],
    '상승추세 조정 진입': sl_trades[(sl_trades['ema_aligned']) & (sl_trades['change_4h'] < 0)],
}

print(f"\n{'손절 사유':<30} {'건수':>8} {'비율':>8}")
print("-"*50)
for reason, data in sl_reasons.items():
    pct = len(data) / len(sl_trades) * 100 if len(sl_trades) > 0 else 0
    print(f"{reason:<30} {len(data):>8} {pct:>7.1f}%")

# 7. 핵심 발견 및 권장 필터
print("\n" + "="*100)
print("💡 7. 핵심 발견 및 권장 진입 조건")
print("="*100)

# 가장 위험한 조합 찾기
if len(comb_df) > 0:
    worst = comb_df.iloc[0]
    best = comb_df.iloc[-1]
    
    print(f"\n🔴 가장 위험한 상황:")
    print(f"   {worst['combination']}")
    print(f"   손실률 {worst['loss_rate']:.1f}%, 평균PNL {worst['avg_pnl']:+.2f}%")
    
    print(f"\n✅ 가장 안전한 상황:")
    print(f"   {best['combination']}")
    print(f"   손실률 {best['loss_rate']:.1f}%, 평균PNL {best['avg_pnl']:+.2f}%")

# 필터 제안
print("\n📋 진입 자리 체크 조건 (하락 사유 기반):")
print("  1. 추세 전환 감지: EMA20 기울기 하락 + EMA 비정배열 시 진입 금지")
print("  2. 급락장 회피: 4시간 -2% 이상 하락 시 진입 금지 (단, -5% 이상 폭락 후는 OK)")
print("  3. 고점 추격 금지: 24시간 고점 -1% 이내 + RSI 60 이상 시 진입 금지")
print("  4. 연속 하락 회피: 최근 10봉 중 음봉 5개 이상 시 주의")
print("  5. 하락추세 역배열 진입 주의: EMA 역배열 시 진입 금지 또는 축소")

# 저장
df.to_csv('trade_loss_reasons_analysis.csv', index=False)
print(f"\n✅ 분석 결과 저장: trade_loss_reasons_analysis.csv")
