import pandas as pd
import numpy as np

"""
최적화된 필터 전략

발견된 승리 패턴:
1. EMA 20 위 (88.8% vs 74.1%) → +14.7%p
2. EMA 50 위 (80.3% vs 68.7%) → +11.6%p  
3. RSI 55+ (59.57 vs 53.91) → +5.65
4. 최근 20캔들 상승중 (0.97% vs 0.20%) → +0.77%p

필터 조합 테스트!
"""

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("📈 최적화된 필터 전략 백테스트")
print("="*100)

# 지표 계산
candles_df['ema_200'] = candles_df['close'].ewm(span=200, adjust=False).mean()
candles_df['ema_50'] = candles_df['close'].ewm(span=50, adjust=False).mean()
candles_df['ema_20'] = candles_df['close'].ewm(span=20, adjust=False).mean()

# RSI
delta = candles_df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
candles_df['rsi'] = 100 - (100 / (1 + gain / loss))

# 최근 변화율
candles_df['change_20'] = candles_df['close'].pct_change(20) * 100

print("✅ 지표 계산 완료")

# 백테스트 함수
def run_backtest(filter_config):
    trades = []
    position = None
    recent_lows = []
    searching_inflection = False
    hl_event = None
    
    for i in range(300, len(candles_df)):
        candle = candles_df.iloc[i]
        prev_candles = candles_df.iloc[max(0, i-20):i]
        
        # 저점 감지
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
        
        # HL 감지
        if len(recent_lows) >= 2 and not searching_inflection:
            current_low = recent_lows[-1]['price']
            previous_low = recent_lows[-2]['price']
            
            if current_low > previous_low:
                if candle['close'] > current_low * 1.003:
                    hl_strength = ((current_low - previous_low) / previous_low) * 100
                    if hl_strength >= 0.5:
                        hl_event = {
                            'hl_price': current_low,
                            'hl_strength': hl_strength,
                            'hl_index': recent_lows[-1]['index']
                        }
                        searching_inflection = True
        
        # 청산
        if position is not None:
            ema_200 = candles_df.iloc[i]['ema_200']
            
            # SL (HL -1%)
            if candle['low'] <= position['sl_price']:
                pnl = ((position['sl_price'] - position['entry_price']) / position['entry_price']) * 100
                trades.append({'pnl': pnl, 'reason': 'SL_HL'})
                position = None
                searching_inflection = False
                continue
            
            # SL (EMA 이탈)
            if position.get('above_ema') and candle['close'] < ema_200:
                pnl = ((candle['close'] - position['entry_price']) / position['entry_price']) * 100
                trades.append({'pnl': pnl, 'reason': 'SL_EMA'})
                position = None
                searching_inflection = False
                continue
            
            # TP
            if candle['high'] >= position['tp_price']:
                pnl = ((position['tp_price'] - position['entry_price']) / position['entry_price']) * 100
                trades.append({'pnl': pnl, 'reason': 'TP'})
                position = None
                searching_inflection = False
                continue
        
        # 진입
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
            if (body_size / total_range) * 100 < 60:
                continue
            
            # === 필터 적용 ===
            c = candles_df.iloc[i]
            
            # EMA 20 필터
            if filter_config.get('ema20') and c['close'] <= c['ema_20']:
                continue
            
            # EMA 50 필터
            if filter_config.get('ema50') and c['close'] <= c['ema_50']:
                continue
            
            # RSI 필터
            if filter_config.get('rsi_min') and c['rsi'] < filter_config['rsi_min']:
                continue
            
            # 최근 상승 필터
            if filter_config.get('change_min') and c['change_20'] < filter_config['change_min']:
                continue
            
            # 진입!
            entry_price = candle['close']
            hl_strength = hl_event['hl_strength']
            
            if hl_strength >= 2:
                tp_pct = 1.5
            elif hl_strength >= 1:
                tp_pct = 1.0
            else:
                tp_pct = 0.7
            
            position = {
                'entry_price': entry_price,
                'tp_price': entry_price * (1 + tp_pct / 100),
                'sl_price': hl_event['hl_price'] * 0.99,
                'above_ema': c['close'] > c['ema_200']
            }
            searching_inflection = False
    
    return trades

# 다양한 필터 조합 테스트
configs = [
    {'name': '기본 (필터 없음)', 'config': {}},
    {'name': 'EMA 20 위', 'config': {'ema20': True}},
    {'name': 'EMA 50 위', 'config': {'ema50': True}},
    {'name': 'EMA 20+50 위', 'config': {'ema20': True, 'ema50': True}},
    {'name': 'RSI 50+', 'config': {'rsi_min': 50}},
    {'name': 'RSI 55+', 'config': {'rsi_min': 55}},
    {'name': '최근 상승 0.5%+', 'config': {'change_min': 0.5}},
    {'name': 'EMA20 + RSI55', 'config': {'ema20': True, 'rsi_min': 55}},
    {'name': 'EMA50 + RSI55', 'config': {'ema50': True, 'rsi_min': 55}},
    {'name': '종합: EMA20+50 + RSI55', 'config': {'ema20': True, 'ema50': True, 'rsi_min': 55}},
    {'name': '종합: EMA20 + RSI55 + 상승0.5%', 'config': {'ema20': True, 'rsi_min': 55, 'change_min': 0.5}},
]

print("\n" + "="*100)
print("📊 필터 조합별 성과 비교")
print("="*100)
print(f"{'필터':<35} {'거래수':<10} {'승률':<10} {'총PNL':<12} {'평균PNL':<10}")
print("-"*80)

results = []
for cfg in configs:
    trades = run_backtest(cfg['config'])
    
    if len(trades) > 0:
        df = pd.DataFrame(trades)
        wins = len(df[df['reason'] == 'TP'])
        total = len(df)
        win_rate = wins / total * 100
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        results.append({
            'name': cfg['name'],
            'trades': total,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        })
        
        print(f"{cfg['name']:<35} {total:<10} {win_rate:<10.1f}% {total_pnl:<12.2f}% {avg_pnl:<10.2f}%")
    else:
        print(f"{cfg['name']:<35} {'0':<10} {'-':<10} {'-':<12} {'-':<10}")

# 최고 성과 찾기
print("\n" + "="*100)
print("🏆 최고 성과")
print("="*100)

if results:
    best_winrate = max(results, key=lambda x: x['win_rate'])
    best_pnl = max(results, key=lambda x: x['total_pnl'])
    best_avg = max(results, key=lambda x: x['avg_pnl'])
    
    print(f"\n최고 승률: {best_winrate['name']}")
    print(f"  승률 {best_winrate['win_rate']:.1f}%, {best_winrate['trades']}건, 총PNL {best_winrate['total_pnl']:.2f}%")
    
    print(f"\n최고 총PNL: {best_pnl['name']}")
    print(f"  총PNL {best_pnl['total_pnl']:.2f}%, {best_pnl['trades']}건, 승률 {best_pnl['win_rate']:.1f}%")
    
    print(f"\n최고 평균PNL: {best_avg['name']}")
    print(f"  평균PNL {best_avg['avg_pnl']:.2f}%, {best_avg['trades']}건, 승률 {best_avg['win_rate']:.1f}%")
